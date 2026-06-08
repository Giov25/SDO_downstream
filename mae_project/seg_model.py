"""
Segmentazione binaria su encoder MAE pretrained.

Architettura:
    - Encoder: MAE ViT-Base (patch_embed + 12 blocks + norm), caricato da checkpoint.
    - Decoder: SegUNetDecoder a 4 livelli che fonde feature estratte dagli strati 3,6,9,12.
    - Il decoder MAE originale (decoder_embed, decoder_blocks, ...) è ereditato per
      compatibilità con il caricamento dei pesi, ma NON viene usato nel forward.
"""
from functools import partial
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from mae.models_mae_2 import MaskedAutoencoderViT


# ------------------------------------------------------------------ #
# Decoder UNet multi-scala                                           #
# ------------------------------------------------------------------ #

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, p: int = 1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, k, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class SegUNetDecoder(nn.Module):
    """
    Decoder UNet che fonde feature multi-depth dell'encoder ViT.

    Input : 4 feature [N, num_patches, embed_dim] da strati encoder (shallow → deep).
    Output: [N, 1, img_h, img_w] logit.

    Per patch_size=16: 4 × 2× upsampling = 16× (es. 64×64 → 1024×1024).
    Tutti gli strati encoder hanno la stessa risoluzione spaziale (h×w),
    il decoder le porta progressivamente alla risoluzione originale dell'immagine.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        patch_size: int = 16,
        channels: Tuple[int, int, int, int] = (256, 128, 64, 32),
    ):
        super().__init__()
        self.patch_size = patch_size
        c4, c3, c2, c1 = channels

        # Proiezione di ciascun livello encoder → canali decoder
        self.proj4 = ConvBNReLU(embed_dim, c4)
        self.proj3 = ConvBNReLU(embed_dim, c3)
        self.proj2 = ConvBNReLU(embed_dim, c2)
        self.proj1 = ConvBNReLU(embed_dim, c1)

        # Fusione: upsample + skip connection da strato encoder meno profondo
        self.fuse3 = ConvBNReLU(c4 + c3, c3)
        self.fuse2 = ConvBNReLU(c3 + c2, c2)
        self.fuse1 = ConvBNReLU(c2 + c1, c1)

        # Testa di segmentazione
        self.head = nn.Sequential(
            ConvBNReLU(c1, c1),
            nn.Conv2d(c1, 1, 1),
        )

    @staticmethod
    def _reshape_feat(feat: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """[N, h*w, D] → [N, D, h, w]"""
        N, _, D = feat.shape
        return feat.reshape(N, h, w, D).permute(0, 3, 1, 2).contiguous()

    def forward(
        self,
        feats: List[torch.Tensor],
        img_h: int,
        img_w: int,
    ) -> torch.Tensor:
        """
        feats: [f1 (shallow), f2, f3, f4 (deep)], ciascuno [N, L, D].
        img_h, img_w: dimensioni dell'immagine originale.
        """
        h = img_h // self.patch_size  # es. 64 per 1024px/patch16
        w = img_w // self.patch_size

        f1, f2, f3, f4 = [self._reshape_feat(f, h, w) for f in feats]

        p4 = self.proj4(f4)  # [N, c4, h, w]
        p3 = self.proj3(f3)  # [N, c3, h, w]
        p2 = self.proj2(f2)  # [N, c2, h, w]
        p1 = self.proj1(f1)  # [N, c1, h, w]

        def up(t: torch.Tensor, factor: float) -> torch.Tensor:
            return F.interpolate(t, scale_factor=factor, mode='bilinear', align_corners=False)

        # h×w → 2h×2w
        x = self.fuse3(torch.cat([up(p4, 2), up(p3, 2)], dim=1))
        # 2h×2w → 4h×4w
        x = self.fuse2(torch.cat([up(x, 2), up(p2, 4)], dim=1))
        # 4h×4w → 8h×8w
        x = self.fuse1(torch.cat([up(x, 2), up(p1, 8)], dim=1))
        # 8h×8w → img_h×img_w (×2 = ×16 totali per patch_size=16)
        x = F.interpolate(x, size=(img_h, img_w), mode='bilinear', align_corners=False)
        return self.head(x)  # [N, 1, img_h, img_w]


# ------------------------------------------------------------------ #
# Modello principale                                                  #
# ------------------------------------------------------------------ #

class MAEForBinarySegmentation(MaskedAutoencoderViT):
    """
    MAE encoder + SegUNetDecoder per segmentazione binaria.

    - Feature multi-scala estratte dagli strati [3, 6, 9, 12] (1-indexed su depth=12).
    - Decoder UNet addestrato da zero.
    - Il decoder MAE originale è ereditato (compatibilità checkpoint) ma NON usato.
    """

    _TAP_LAYERS: Tuple[int, ...] = (3, 6, 9, 12)

    def __init__(
        self,
        *args,
        # Loss pixel-wise
        pixel_loss: str = 'focal',          # 'bce' | 'focal'
        pixel_weight: float = 1.0,
        focal_gamma_pixel: float = 2.0,
        focal_alpha_pixel: float = 0.25,
        pos_weight: Optional[float] = None,
        # Loss region
        region_loss: str = 'focal_tversky', # 'dice' | 'tversky' | 'focal_tversky' | 'dice_tversky'
        region_weight: float = 1.0,
        tversky_alpha: float = 0.3,         # peso FP
        tversky_beta: float = 0.7,          # peso FN (β>α → favorisce recall)
        dice_tversky_weight: float = 0.5,   # weight per combinare Dice (1-w) e Tversky (w)
        focal_gamma_region: float = 4.0 / 3.0,
        # Decoder e misc
        seg_decoder_channels: Tuple[int, int, int, int] = (256, 128, 64, 32),
        use_gradient_checkpointing: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        assert pixel_loss in ('bce', 'focal')
        assert region_loss in ('dice', 'tversky', 'focal_tversky', 'dice_tversky')
        self.pixel_loss         = pixel_loss
        self.pixel_weight       = pixel_weight
        self.focal_gamma_pixel  = focal_gamma_pixel
        self.focal_alpha_pixel  = focal_alpha_pixel
        self.region_loss        = region_loss
        self.region_weight      = region_weight
        self.tversky_alpha      = tversky_alpha
        self.tversky_beta       = tversky_beta
        self.dice_tversky_weight = dice_tversky_weight
        self.focal_gamma_region = focal_gamma_region
        if pos_weight is not None:
            self.register_buffer('pos_weight', torch.tensor([float(pos_weight)]))
        else:
            self.pos_weight = None
        self.use_gradient_checkpointing = use_gradient_checkpointing

        self.seg_decoder = SegUNetDecoder(
            embed_dim=self.embed_dim,
            patch_size=self.patch_size,
            channels=seg_decoder_channels,
        )

    # ---------------------------------------------------------------- #
    # Encoder forward senza mascheramento, con tap multi-scala         #
    # ---------------------------------------------------------------- #

    def _forward_encoder_multiscale(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Passa l'immagine intera (no masking) attraverso l'encoder e raccoglie
        le feature intermedie agli strati _TAP_LAYERS.

        Returns: lista di 4 tensori [N, num_patches, embed_dim].
        """
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]

        # Spectral embedding: tutti i canali visibili (nessun mascheramento canale)
        all_ids = torch.arange(self.in_chans, dtype=torch.long, device=x.device)
        x = x + self.channel_embed(all_ids).sum(0)[None, None, :]

        cls = (self.cls_token + self.pos_embed[:, :1, :]).expand(x.shape[0], -1, -1)
        x = torch.cat((cls, x), dim=1)

        features: List[torch.Tensor] = []
        for i, blk in enumerate(self.blocks):
            use_ckpt = (
                self.use_gradient_checkpointing
                and self.training
                and any(p.requires_grad for p in blk.parameters())
            )
            x = checkpoint(blk, x, use_reentrant=False) if use_ckpt else blk(x)

            if (i + 1) in self._TAP_LAYERS:
                # Applica norm e rimuovi cls token
                features.append(self.norm(x)[:, 1:, :])

        return features  # 4 × [N, num_patches, embed_dim]

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        """imgs: [N, C, H, W] → logits: [N, 1, H, W]"""
        H, W = imgs.shape[2], imgs.shape[3]

        enc_frozen = not any(p.requires_grad for p in self.patch_embed.parameters())
        if enc_frozen:
            with torch.no_grad():
                feats = self._forward_encoder_multiscale(imgs)
        else:
            feats = self._forward_encoder_multiscale(imgs)

        return self.seg_decoder(feats, H, W)

    # ---------------------------------------------------------------- #
    # Loss                                                             #
    # ---------------------------------------------------------------- #

    def _pixel_loss(self, logits, targets, valid_mask):
        if self.pixel_loss == 'bce':
            loss = F.binary_cross_entropy_with_logits(
                logits, targets,
                pos_weight=self.pos_weight,
                reduction='none',
            )
        else:  # focal
            bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
            p_t = torch.exp(-bce)
            alpha_t = self.focal_alpha_pixel * targets + (1 - self.focal_alpha_pixel) * (1 - targets)
            loss = alpha_t * (1 - p_t).pow(self.focal_gamma_pixel) * bce

        return self._masked_mean(loss, valid_mask)

    @staticmethod
    def _masked_mean(loss_per_pixel, valid_mask):
        if valid_mask is None:
            return loss_per_pixel.mean()
        return (loss_per_pixel * valid_mask).sum() / valid_mask.sum().clamp(min=1.0)

    def _region_loss(self, probs, targets, valid_mask, smooth: float = 1.0):
        dims = (0, 2, 3)
        if valid_mask is not None:
            probs   = probs   * valid_mask
            targets = targets * valid_mask
            neg_p   = (1 - probs)   * valid_mask
            neg_t   = (1 - targets) * valid_mask
        else:
            neg_p, neg_t = 1 - probs, 1 - targets

        if self.region_loss == 'dice':
            inter = (probs * targets).sum(dim=dims)
            denom = probs.sum(dim=dims) + targets.sum(dim=dims)
            return (1 - (2 * inter + smooth) / (denom + smooth)).mean()

        tp = (probs * targets).sum(dim=dims)
        fp = (probs * neg_t).sum(dim=dims)
        fn = (neg_p * targets).sum(dim=dims)
        
        if self.region_loss == 'tversky':
            tv = (tp + smooth) / (tp + self.tversky_alpha * fp + self.tversky_beta * fn + smooth)
            loss = (1 - tv)
        elif self.region_loss == 'focal_tversky':
            tv = (tp + smooth) / (tp + self.tversky_alpha * fp + self.tversky_beta * fn + smooth)
            loss = (1 - tv).clamp(min=1e-8).pow(self.focal_gamma_region)
        elif self.region_loss == 'dice_tversky':
            # Combina Dice e Tversky loss con weight
            dice = (probs * targets).sum(dim=dims)
            denom_dice = probs.sum(dim=dims) + targets.sum(dim=dims)
            dice_loss = 1 - (2 * dice + smooth) / (denom_dice + smooth)
            
            tv = (tp + smooth) / (tp + self.tversky_alpha * fp + self.tversky_beta * fn + smooth)
            tversky_loss = 1 - tv
            
            loss = (1 - self.dice_tversky_weight) * dice_loss + self.dice_tversky_weight * tversky_loss
        else:
            raise ValueError(f"Unknown region_loss: {self.region_loss}")
        
        return loss.mean()

    def compute_loss(self, logits, targets, valid_mask=None):
        pix = self._pixel_loss(logits, targets, valid_mask)
        reg = self._region_loss(torch.sigmoid(logits), targets, valid_mask)
        return self.pixel_weight * pix + self.region_weight * reg

    # ---------------------------------------------------------------- #
    # Freeze encoder                                                   #
    # ---------------------------------------------------------------- #

    def freeze_encoder(self):
        for m in (self.patch_embed, self.blocks, self.norm, self.channel_embed):
            for p in m.parameters():
                p.requires_grad = False
            m.eval()
        for p in (self.cls_token, self.pos_embed):
            p.requires_grad = False
        for attr in ('mask_token', 'channel_mask_values'):
            if hasattr(self, attr):
                getattr(self, attr).requires_grad = False

    def keep_encoder_eval(self):
        """Chiamare dopo model.train() per tenere l'encoder in eval()."""
        for m in (self.patch_embed, self.blocks, self.norm, self.channel_embed):
            m.eval()


# ------------------------------------------------------------------ #
# Factory                                                            #
# ------------------------------------------------------------------ #

def build_seg_model(
    pretrained_ckpt: Optional[str],
    img_size: int = 1024,
    patch_size: int = 16,
    in_chans: int = 9,
    embed_dim: int = 768,
    depth: int = 12,
    num_heads: int = 12,
    decoder_embed_dim: int = 512,
    decoder_depth: int = 8,
    decoder_num_heads: int = 16,
    freeze_encoder: bool = True,
    seg_decoder_channels: Tuple[int, int, int, int] = (256, 128, 64, 32),
    # Loss
    pixel_loss: str = 'focal',
    pixel_weight: float = 1.0,
    focal_gamma_pixel: float = 2.0,
    focal_alpha_pixel: float = 0.25,
    pos_weight: Optional[float] = None,
    region_loss: str = 'focal_tversky',
    region_weight: float = 1.0,
    tversky_alpha: float = 0.3,
    tversky_beta: float = 0.7,
    dice_tversky_weight: float = 0.5,
    focal_gamma_region: float = 4.0 / 3.0,
    use_gradient_checkpointing: bool = True,
    verbose: bool = True,
) -> MAEForBinarySegmentation:

    model = MAEForBinarySegmentation(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        out_chans=1,            # per compatibilità decoder_pred (non usato)
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        decoder_embed_dim=decoder_embed_dim,
        decoder_depth=decoder_depth,
        decoder_num_heads=decoder_num_heads,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        grid_size=2,
        mask_mode='spatial',
        use_channel_attention=False,
        mask_ratio=0.0,
        pixel_loss=pixel_loss,
        pixel_weight=pixel_weight,
        focal_gamma_pixel=focal_gamma_pixel,
        focal_alpha_pixel=focal_alpha_pixel,
        pos_weight=pos_weight,
        region_loss=region_loss,
        region_weight=region_weight,
        tversky_alpha=tversky_alpha,
        tversky_beta=tversky_beta,
        dice_tversky_weight=dice_tversky_weight,
        focal_gamma_region=focal_gamma_region,
        use_gradient_checkpointing=use_gradient_checkpointing,
        seg_decoder_channels=seg_decoder_channels,
    )

    if pretrained_ckpt is not None:
        ckpt = torch.load(pretrained_ckpt, map_location='cpu', weights_only=False)
        state_dict = ckpt.get('model_state_dict', ckpt.get('model', ckpt))

        # Rimuovi prefisso '_orig_mod.' prodotto da torch.compile al salvataggio
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

        # Escludiamo decoder_pred (shape diversa) e seg_decoder (addestrato da scratch);
        # escludiamo anche le chiavi con shape incompatibile (es. patch_embed con p diverso).
        skip_prefixes = ('decoder_pred', 'seg_decoder')
        model_sd = model.state_dict()
        filtered, skipped_shape = {}, []
        for k, v in state_dict.items():
            if any(k.startswith(p) for p in skip_prefixes):
                continue
            if k in model_sd and model_sd[k].shape != v.shape:
                skipped_shape.append(f"{k}: ckpt{tuple(v.shape)} vs model{tuple(model_sd[k].shape)}")
                continue
            filtered[k] = v

        missing, unexpected = model.load_state_dict(filtered, strict=False)
        if verbose:
            print(f"[ckpt] MAE pretrain caricato da: {pretrained_ckpt}")
            if skipped_shape:
                print(f"[ckpt] Shape mismatch skippati ({len(skipped_shape)}): {skipped_shape[:3]}")
            n_miss = len(missing)
            n_unex = len(unexpected)
            print(f"[ckpt] Missing  ({n_miss}): {missing[:5]}{'...' if n_miss > 5 else ''}")
            print(f"[ckpt] Unexpected ({n_unex}): {unexpected[:5]}{'...' if n_unex > 5 else ''}")

    if freeze_encoder:
        model.freeze_encoder()
        if verbose:
            n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
            n_total = sum(p.numel() for p in model.parameters())
            print(
                f"[freeze] Encoder congelato. "
                f"Trainable: {n_train/1e6:.2f}M / {n_total/1e6:.2f}M "
                f"({100*n_train/n_total:.1f}%)"
            )

    return model
