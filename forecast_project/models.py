from functools import partial

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from timm.models.vision_transformer import Block, PatchEmbed
from timm.models.vision_transformer import Block, PatchEmbed
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Checkpoint loading helper
# ---------------------------------------------------------------------------

def _strip_compiled(state_dict):
    """Strip '_orig_mod.' prefix added by torch.compile()."""
    return {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}


def load_mae_encoder_weights(checkpoint_path, device='cpu'):
    """Return cleaned state-dict from a MAE checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get('model_state_dict', ckpt)
    return _strip_compiled(state)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class MAE_TemporalForecaster(nn.Module):
    """
    Temporal forecasting downstream task built on a pre-trained MAE encoder.

    Architecture
    ------------
    1. MAE encoder (frozen by default): patch-embed + positional + spectral
       embeddings → 12 ViT blocks → LayerNorm → [B, N+1, D]
    2. Delta-t conditioning: Embedding(num_horizons, D) added to every token
    3. Temporal adaptation: `num_temporal_blocks` trainable ViT blocks + norm
    4. Decoder: linear projection → decoder ViT blocks → LayerNorm →
       linear head → unpatchify → [B, in_chans, H, W]

    Parameters
    ----------
    mae_checkpoint : str
        Path to pretrained MAE .pth file (trained with torch.compile OK).
    img_size, patch_size, in_chans, embed_dim, depth, num_heads
        Must match the pre-trained model (1024, 16, 9, 768, 12, 12).
    delta_t_values : list[int]
        Discrete set of forecast horizons in hours, e.g. [12,24,36,48,168].
    freeze_encoder : bool
        If True (default), encoder weights are frozen.
    num_temporal_blocks : int
        Number of trainable transformer blocks after the frozen encoder (4–6).
    decoder_embed_dim, decoder_depth, decoder_num_heads
        Decoder architecture.
    use_gradient_checkpointing : bool
        Apply torch.utils.checkpoint on encoder blocks to save VRAM.
    """

    def __init__(
        self,
        mae_checkpoint,
        img_size=1024,
        patch_size=16,
        in_chans=9,
        embed_dim=768,
        depth=12,
        num_heads=12,
        decoder_embed_dim=512,
        decoder_depth=6,
        decoder_num_heads=16,
        num_temporal_blocks=4,
        delta_t_values=None,
        freeze_encoder=True,
        use_gradient_checkpointing=True,
        device='cpu',
    ):
        super().__init__()

        if delta_t_values is None:
            delta_t_values = [12, 24, 36, 48, 168]

        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2
        self.delta_t_values = delta_t_values
        self.use_gc = use_gradient_checkpointing
        self.freeze_encoder = freeze_encoder

        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        # ---- Encoder (pretrained) ----------------------------------------
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.encoder_blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio=4., qkv_bias=True, norm_layer=norm_layer)
            for _ in range(depth)
        ])
        self.encoder_norm = norm_layer(embed_dim)
        self.channel_embed = nn.Embedding(in_chans, embed_dim)

        self._load_pretrained(mae_checkpoint, device)

        if freeze_encoder:
            self._freeze_encoder()

        # ---- Delta-t conditioning ----------------------------------------
        self.delta_t_embed = nn.Embedding(len(delta_t_values), embed_dim)

        # ---- Temporal adaptation blocks (trainable) ----------------------
        self.temporal_blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio=4., qkv_bias=True, norm_layer=norm_layer)
            for _ in range(num_temporal_blocks)
        ])
        self.temporal_norm = norm_layer(embed_dim)

        # ---- Decoder -----------------------------------------------------
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, decoder_embed_dim)
        )
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio=4.,
                  qkv_bias=True, norm_layer=norm_layer)
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)

        self._init_new_weights()

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _load_pretrained(self, ckpt_path, device):
        state = load_mae_encoder_weights(ckpt_path, device)
        enc_keys = {'patch_embed', 'cls_token', 'pos_embed',
                    'encoder_blocks', 'encoder_norm', 'channel_embed'}
        # Map MAE names → this module's names
        remap = {
            'blocks.': 'encoder_blocks.',
            'norm.': 'encoder_norm.',
        }
        own_state = {}
        for k, v in state.items():
            new_k = k
            for old, new in remap.items():
                if new_k.startswith(old):
                    new_k = new + new_k[len(old):]
                    break
            # Keep only encoder-related keys
            root = new_k.split('.')[0]
            if root in enc_keys or new_k in ('cls_token', 'pos_embed'):
                own_state[new_k] = v

        missing, unexpected = self.load_state_dict(own_state, strict=False)
        enc_missing = [k for k in missing if k.split('.')[0] in enc_keys]
        if enc_missing:
            raise RuntimeError(f'MAE encoder weights missing: {enc_missing[:5]}')

    def _freeze_encoder(self):
        for param in (
            list(self.patch_embed.parameters())
            + list(self.encoder_blocks.parameters())
            + list(self.encoder_norm.parameters())
            + list(self.channel_embed.parameters())
            + [self.cls_token, self.pos_embed]
        ):
            param.requires_grad = False

    def _init_new_weights(self):
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.delta_t_embed.weight, std=0.02)
        for m in list(self.temporal_blocks.modules()) + list(self.decoder_blocks.modules()):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        nn.init.xavier_uniform_(self.decoder_embed.weight)
        nn.init.xavier_uniform_(self.decoder_pred.weight)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def encode(self, x):
        """Full (no-masking) encoding of a 9-channel image.

        x : [B, 9, H, W]
        returns : [B, N+1, D]   (N patch tokens + 1 CLS token)
        """
        x = self.patch_embed(x)                           # [B, N, D]
        x = x + self.pos_embed[:, 1:, :]                  # spatial pos

        # Spectral embedding: all 9 channels visible
        all_ids = torch.arange(self.in_chans, dtype=torch.long, device=x.device)
        spectral = self.channel_embed(all_ids).sum(0)     # [D]
        x = x + spectral[None, None, :]

        cls = (self.cls_token + self.pos_embed[:, :1, :]).expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)                    # [B, N+1, D]

        for blk in self.encoder_blocks:
            if self.use_gc and self.training:
                x = checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        x = self.encoder_norm(x)
        return x

    def forward(self, x, delta_t_idx):
        """
        x           : [B, 9, H, W]  input image at time t
        delta_t_idx : [B] long      index into self.delta_t_values

        returns     : [B, 9, H, W]  predicted image at t + Δt
        """
        # 1. Encode — skip autograd graph when encoder is frozen
        if self.freeze_encoder:
            with torch.no_grad():
                feats = self.encode(x)
            feats = feats.detach()
        else:
            feats = self.encode(x)                        # [B, N+1, D]

        # 2. Condition on forecast horizon
        dt_emb = self.delta_t_embed(delta_t_idx)         # [B, D]
        feats = feats + dt_emb.unsqueeze(1)              # broadcast to all tokens

        # 3. Temporal adaptation
        for blk in self.temporal_blocks:
            feats = blk(feats)
        feats = self.temporal_norm(feats)                 # [B, N+1, D]

        # 4. Decode
        dec = self.decoder_embed(feats)                   # [B, N+1, dec_D]
        dec = dec + self.decoder_pos_embed
        for blk in self.decoder_blocks:
            dec = blk(dec)
        dec = self.decoder_pred(dec)                      # [B, N+1, p²·C]
        dec = dec[:, 1:, :]

        # 5. Unpatchify + constrain to [0, 1] (matches dataset normalization)
        return torch.sigmoid(self.unpatchify(dec))        # [B, 9, H, W]

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------
    def total_variation_loss(img):
        # img ha shape [B, C, H, W]
        tv_h = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]).mean()
        tv_w = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]).mean()
        return tv_h + tv_w
    def compute_loss(self, pred, target, norm_pix=False, tv_weight=0):
            """Per-patch normalised MSE (same convention as MAE pre-training)."""
            target_p = self.patchify(target)  # [B, N, p²·C]
            pred_p = self.patchify(pred)
            if norm_pix:
                mean = target_p.mean(dim=-1, keepdim=True)
                var = target_p.var(dim=-1, keepdim=True)
                target_p = (target_p - mean) / (var + 1e-6).sqrt()
            mse = ((pred_p - target_p) ** 2).mean()
            tv_h = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :]).mean()
            tv_w = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1]).mean()
            tv_loss = tv_h + tv_w
            return mse + tv_weight * tv_loss
            #return ((pred_p - target_p) ** 2).mean()


    # ------------------------------------------------------------------
    # Patch utilities
    # ------------------------------------------------------------------

    def patchify(self, imgs):
        """[B, C, H, W] → [B, N, p²·C]"""
        p = self.patch_size
        h = w = imgs.shape[2] // p
        c = imgs.shape[1]
        x = imgs.reshape(imgs.shape[0], c, h, p, w, p)
        x = torch.einsum('nchpwq->nhwpqc', x)
        return x.reshape(imgs.shape[0], h * w, p * p * c)

    def unpatchify(self, x):
        """[B, N, p²·C] → [B, C, H, W]"""
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        c = self.in_chans
        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = torch.einsum('nhwpqc->nchpwq', x)
        return x.reshape(x.shape[0], c, h * p, w * p)


# ---------------------------------------------------------------------------
# MAE_FullForecaster — riusa encoder + decoder pre-addestrati del MAE
# ---------------------------------------------------------------------------

class MAE_FullForecaster(nn.Module):
    """
    Forecaster che riusa il MAE completo (encoder + decoder pre-addestrati).
    Aggiunge solo un embedding Δt (nn.Embedding) come unico nuovo parametro.

    Forward:
      1. Encode senza masking  →  [B, N+1, D]
      2. Somma l'embedding Δt a ogni token
      3. Decoder MAE (pre-addestrato, fine-tunato)  →  future image [B, 9, H, W]

    L'architettura del decoder (8 blocchi, dec_dim=512) corrisponde al MAE
    pre-addestrato e non è modificabile senza ricaricare pesi diversi.
    """

    def __init__(
        self,
        mae_checkpoint,
        img_size=1024,
        patch_size=16,
        in_chans=9,
        embed_dim=768,
        depth=12,
        num_heads=12,
        decoder_embed_dim=512,
        decoder_depth=8,       # deve corrispondere al MAE pre-addestrato
        decoder_num_heads=16,
        delta_t_values=None,
        freeze_encoder=True,
        use_gradient_checkpointing=True,
        device='cpu',
    ):
        super().__init__()

        self.img_size   = img_size
        self.patch_size = patch_size
        self.in_chans   = in_chans
        self.embed_dim  = embed_dim
        self.num_patches = (img_size // patch_size) ** 2
        self.delta_t_values = delta_t_values or [12]
        self.use_gc = use_gradient_checkpointing
        self.freeze_encoder = freeze_encoder

        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        # ── Encoder ───────────────────────────────────────────────────────
        self.patch_embed   = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.cls_token     = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed     = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.blocks        = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio=4., qkv_bias=True, norm_layer=norm_layer)
            for _ in range(depth)
        ])
        self.norm          = norm_layer(embed_dim)
        self.channel_embed = nn.Embedding(in_chans, embed_dim)

        # ── Decoder MAE (pesi pre-addestrati) ─────────────────────────────
        self.decoder_embed     = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, decoder_embed_dim)
        )
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio=4.,
                  qkv_bias=True, norm_layer=norm_layer)
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)
        # mask_token: presente nel checkpoint MAE, non usato in inferenza forecast
        self.mask_token   = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        # ── Unico nuovo parametro: embedding Δt ───────────────────────────
        self.delta_t_embed = nn.Embedding(len(self.delta_t_values), embed_dim)

        self._load_pretrained(mae_checkpoint, device)

        if freeze_encoder:
            self._freeze_encoder()

        nn.init.trunc_normal_(self.delta_t_embed.weight, std=0.02)

    # ------------------------------------------------------------------
    def _load_pretrained(self, ckpt_path, device):
        state = load_mae_encoder_weights(ckpt_path, device)
        missing, unexpected = self.load_state_dict(state, strict=False)
        new_params = {'delta_t_embed.weight'}
        real_missing = [k for k in missing if k not in new_params
                        and 'channel_mask_values' not in k]
        if real_missing:
            print(f'[WARN] MAE_FullForecaster — missing encoder/decoder keys: {real_missing[:8]}')
        n_loaded = len(state) - len(unexpected)
        print(f'MAE full weights loaded: {n_loaded} tensors  '
              f'| unexpected (skipped): {len(unexpected)}  '
              f'| new (random init): {[k for k in missing]}')

    def _freeze_encoder(self):
        for p in (
            list(self.patch_embed.parameters())
            + list(self.blocks.parameters())
            + list(self.norm.parameters())
            + list(self.channel_embed.parameters())
            + [self.cls_token, self.pos_embed]
        ):
            p.requires_grad = False

    # ------------------------------------------------------------------
    def encode(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)                         # [B, N, D]
        x = x + self.pos_embed[:, 1:]

        all_ids  = torch.arange(self.in_chans, dtype=torch.long, device=x.device)
        spectral = self.channel_embed(all_ids).sum(0)   # [D]
        x = x + spectral[None, None, :]

        cls = (self.cls_token + self.pos_embed[:, :1]).expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)                # [B, N+1, D]

        for blk in self.blocks:
            if self.use_gc and self.training:
                x = checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        return self.norm(x)                              # [B, N+1, D]

    def forward(self, x, delta_t_idx):
        if self.freeze_encoder:
            with torch.no_grad():
                feats = self.encode(x)
            feats = feats.detach()
        else:
            feats = self.encode(x)

        # Conditioning Δt: sommato a tutti i token (cls + patch)
        dt_emb = self.delta_t_embed(delta_t_idx)        # [B, D]
        feats  = feats + dt_emb.unsqueeze(1)

        # Decode — nessun mask_token: tutti i patch sono visibili
        dec = self.decoder_embed(feats)                  # [B, N+1, dec_D]
        dec = dec + self.decoder_pos_embed
        for blk in self.decoder_blocks:
            dec = blk(dec)
        dec = self.decoder_norm(dec)
        dec = self.decoder_pred(dec[:, 1:])              # rimuove CLS → [B, N, p²·C]

        return self.unpatchify(dec)                      # [B, 9, H, W]

    # ------------------------------------------------------------------
    def patchify(self, imgs):
        p = self.patch_size
        h = w = imgs.shape[2] // p
        c = imgs.shape[1]
        x = imgs.reshape(imgs.shape[0], c, h, p, w, p)
        x = torch.einsum('nchpwq->nhwpqc', x)
        return x.reshape(imgs.shape[0], h * w, p * p * c)

    def unpatchify(self, x):
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        c = self.in_chans
        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = torch.einsum('nhwpqc->nchpwq', x)
        return x.reshape(x.shape[0], c, h * p, w * p)

    def compute_loss(self, pred, target, norm_pix=False):
        target_p = self.patchify(target)
        pred_p   = self.patchify(pred)
        if norm_pix:
            mean     = target_p.mean(dim=-1, keepdim=True)
            var      = target_p.var(dim=-1, keepdim=True)
            target_p = (target_p - mean) / (var + 1e-6).sqrt()
        return ((pred_p - target_p) ** 2).mean()
