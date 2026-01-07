# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, Block
import random
from mae.util.pos_embed import get_2d_sincos_pos_embed


class CrossChannelAttentionBlock(nn.Module):
    """
    Blocco di attenzione cross-channel per catturare correlazioni tra canali.
    Utile per il mascheramento dei canali: i canali visibili possono comunicare
    tra loro per migliorare la ricostruzione dei canali mascherati.
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, channel_mask=None):
        """
        Args:
            x: [B, N, D] - input features (N = num_patches)
            channel_mask: [B, in_chans] - binary mask (1 = canale visibile, 0 = mascherato)
        
        Returns:
            x: [B, N, D] - output features con attenzione cross-channel applicata
        """
        B, N, D = x.shape
        
        # Layer norm
        x_norm = self.norm(x)
        
        # QKV projection
        qkv = self.qkv(x_norm).reshape(B, N, 3, self.num_heads, D // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # [B, num_heads, N, head_dim]
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # Opzionalmente, applica maschera per impedire attenzione ai canali mascherati
        # (questo è più complesso e richiede reshaping per canali)
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Apply attention
        x_attn = (attn @ v).transpose(1, 2).reshape(B, N, D)
        x_attn = self.proj(x_attn)
        x_attn = self.proj_drop(x_attn)
        
        # Residual connection
        x = x + x_attn
        
        return x


class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=672, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16, n_img_mask = 1,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, grid_size=3,
                 mask_mode='spatial', use_channel_attention=False, num_channel_attn_blocks=2):
        super().__init__()

        # Store grid size for masking logic
        self.grid_size = grid_size # e.g., 3 for a 3x3 grid
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.n_img_mask = n_img_mask
        self.embed_dim = embed_dim
        self.mask_mode = mask_mode  # 'spatial' or 'channel'
        self.use_channel_attention = use_channel_attention

        # Calculate patches per side for sub-images
        assert img_size % grid_size == 0, "Image size must be divisible by grid size"
        sub_img_size = img_size // grid_size
        assert sub_img_size % patch_size == 0, "Sub-image size must be divisible by patch size"
        self.sub_patches_per_side = sub_img_size // patch_size # e.g., (192/3) / 16 = 64 / 16 = 4

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.num_patches = num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=True)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=True)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        
        # --------------------------------------------------------------------------
        # Cross-channel attention blocks (opzionale, per mascheramento canali)
        if use_channel_attention:
            self.channel_attn_blocks = nn.ModuleList([
                CrossChannelAttentionBlock(
                    decoder_embed_dim, 
                    num_heads=decoder_num_heads // 2,  # Usa meno heads per efficienza
                    qkv_bias=True
                )
                for _ in range(num_channel_attn_blocks)
            ])
        else:
            self.channel_attn_blocks = None
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # Inizializzazione allenabile invece di sin-cos fisso
        torch.nn.init.trunc_normal_(self.pos_embed, std=.02)
        torch.nn.init.trunc_normal_(self.decoder_pos_embed, std=.02)

        # Resto dell'inizializzazione (patch_embed, cls_token, ecc.)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, C, H, W)
        x: (N, L, patch_size**2 * C)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
        h = w = imgs.shape[2] // p
        c = imgs.shape[1]
        x = imgs.reshape(shape=(imgs.shape[0], c, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * c))
        return x


    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 * C)
        imgs: (N, C, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        c = self.in_chans # Use stored in_chans
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p)) # Use w*p for width too

        return imgs

    @staticmethod
    def get_sub_image_indices(k, L_total, sub_patches_per_side, grid_size):
        """
        Calculates the linear patch indices for a specific sub-image within a larger grid.

        Args:
            k (int): The index of the sub-image (0 to grid_size**2 - 1).
            L_total (int): Total number of patches (e.g., 144).
            sub_patches_per_side (int): Number of patches along one side of a sub-image (e.g., 4).
            grid_size (int): The grid dimension (e.g., 3 for 3x3).


        Returns:
            torch.Tensor: A tensor containing the linear indices for the k-th sub-image.
        """
        L_sqrt = int(L_total**0.5) # Total patches per side (e.g., 12)
        # grid_sqrt = L_sqrt // sub_patches_per_side # Should be same as grid_size
        assert k >= 0 and k < grid_size**2, "Sub-image index k out of bounds"

        sub_row = k // grid_size
        sub_col = k % grid_size

        indices = []
        start_row = sub_row * sub_patches_per_side
        start_col = sub_col * sub_patches_per_side

        for r in range(start_row, start_row + sub_patches_per_side):
            for c in range(start_col, start_col + sub_patches_per_side):
                idx = r * L_sqrt + c
                indices.append(idx)
        return torch.tensor(indices, dtype=torch.long)

    def mask_specific_subgrid(self, x, subgrid_index_list):
        """
        Maschera una specifica sotto-immagine nella griglia
        
        Args:
            x: [N, L, D] - input sequence
            subgrid_index: int - indice della sotto-immagine da mascherare (0-8 per griglia 3x3)
        """
        N, L, D = x.shape
        
        all_target_patch_indices = []

        for subgrid_index in subgrid_index_list:
            # Ottieni gli indici delle patch per la sotto-immagine target
            target_patch_indices = self.get_sub_image_indices(
                subgrid_index, L, self.sub_patches_per_side, self.grid_size
            ).to(x.device)
            all_target_patch_indices.append(target_patch_indices)

        all_target_patch_indices = torch.cat(all_target_patch_indices, dim=0)

        all_target_patch_indices = torch.unique(all_target_patch_indices)

        mask = torch.zeros([N, L], device=x.device)
        mask[:, all_target_patch_indices] = 1


        # Calcola gli indici delle patch da mantenere per il primo campione
        ids_keep_first = (mask[0] == 0).nonzero(as_tuple=False).squeeze(1)
        # Replica per ogni elemento del batch
        ids_keep = ids_keep_first.unsqueeze(0).repeat(N, 1)

        # Maschera l'input
        x_masked = torch.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D)
        )
        
        # Crea ids_restore per il decoder 
        ids_restore = torch.arange(L, device=x.device).unsqueeze(0).repeat(N, 1)

        return x_masked, mask, ids_restore

    def mask_specific_channels(self, imgs, channel_indices):
        """
        Maschera specifici canali dell'immagine impostando i loro valori a zero prima dell'embedding
        
        Args:
            imgs: [N, C, H, W] - input images
            channel_indices: list - indici dei canali da mascherare
        
        Returns:
            imgs_masked: [N, C, H, W] - immagini con canali mascherati
            mask: [N, L] - maschera binaria (1 = mascherato, 0 = visibile)
        """
        N, C, H, W = imgs.shape
        imgs_masked = imgs.clone()
        
        # Maschera i canali selezionati impostando i loro valori a zero
        for channel_idx in channel_indices:
            if channel_idx < C:
                imgs_masked[:, channel_idx, :, :] = 0
        
        # Calcola la maschera a livello di patch
        # Poiché stiamo mascherando interi canali, tutte le patch sono parzialmente mascherate
        # ma possiamo creare una maschera che indica quali patch contengono canali mascherati
        num_patches = (H // self.patch_size) * (W // self.patch_size)
        
        # Opzione 1: Tutte le patch sono considerate mascherate (perché contengono canali mascherati)
        # mask = torch.ones([N, num_patches], device=imgs.device)
        
        # Opzione 2: Maschera solo se TUTTI i canali sono mascherati (più conservativa)
        if len(channel_indices) == C:
            mask = torch.ones([N, num_patches], device=imgs.device)
        else:
            # Solo una frazione dei canali è mascherata, quindi calcoliamo una maschera proporzionale
            mask = torch.ones([N, num_patches], device=imgs.device) * (len(channel_indices) / C)
        
        return imgs_masked, mask



    def forward_encoder(self, x, n_img_mask=None):
        n_img_mask = n_img_mask if n_img_mask is not None else self.n_img_mask
        
        # Inizializza channel_indices per entrambe le modalità
        channel_indices = None
        
        if self.mask_mode == 'channel':
            # Modalità mascheramento canali
            if n_img_mask is None:
                n_img_mask = random.randint(1, self.in_chans - 1)
            
            # Seleziona casualmente i canali da mascherare
            channel_indices = random.sample(range(self.in_chans), n_img_mask)
            
            # Maschera i canali prima dell'embedding
            x_masked, mask = self.mask_specific_channels(x, channel_indices)
            
            # Embed patches dalle immagini mascherate
            x = self.patch_embed(x_masked)
            
            # add pos embed w/o cls token
            x = x + self.pos_embed[:, 1:, :]
            
            # Per il mascheramento dei canali, non rimuoviamo patch dall'encoder
            # quindi ids_restore è semplicemente l'identità
            ids_restore = torch.arange(x.shape[1], device=x.device).unsqueeze(0).repeat(x.shape[0], 1)
            
        else:  # mask_mode == 'spatial'
            # Modalità mascheramento spaziale (originale)
            # embed patches
            x = self.patch_embed(x)

            # add pos embed w/o cls token
            x = x + self.pos_embed[:, 1:, :]
            
            if n_img_mask is None:
                n_img_mask = random.randint(1, 8)
            
            # Scegli casualmente le sotto-immagini da mascherare
            channel_indices = random.sample(range(0, 8), n_img_mask)
            
            # Maschera le sotto-immagini specifiche
            x, mask, ids_restore = self.mask_specific_subgrid(x, channel_indices)
        
        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore, channel_indices

    def forward_decoder(self, x, ids_restore, channel_indices=None):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle

        # If cls_token was used in encoder it should be handled here too
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        
        # Applica cross-channel attention se abilitato e in modalità channel masking
        if self.use_channel_attention and self.channel_attn_blocks is not None and channel_indices is not None:
            # Applica i blocchi di attenzione cross-channel
            for ch_attn_blk in self.channel_attn_blocks:
                x = ch_attn_blk(x, channel_mask=channel_indices)
        
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, C, H, W]
        pred: [N, L, p*p*C]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, n_img_mask=None):
        latent, mask, ids_restore, channel_indices = self.forward_encoder(imgs, n_img_mask)
        pred = self.forward_decoder(latent, ids_restore, channel_indices)  # [N, L, p*p*C]
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask

def mae_model_for_pretraining(**kwargs):            #random masking
    """ MAE model for pretraining with random masking
    """
    model = MaskedAutoencoderViT(
        img_size=672, patch_size=14, embed_dim=768, depth=12, num_heads=12, n_img_mask=None, # Ensure img_size is correct if not default
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16, in_chans=3,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), grid_size=3, mask_mode='spatial') 
    return model

def mae_model_channel_masking_9ch_with_temporal_attn(**kwargs):
    """ MAE model with 9 channels, channel masking AND cross-channel attention
    Questo modello usa attenzione temporale per catturare correlazioni tra i canali NON mascherati
    """
    model = MaskedAutoencoderViT(
        img_size=1024, patch_size=16, embed_dim=768, depth=12, num_heads=12, n_img_mask=None,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16, in_chans=9,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), norm_pix_loss=False, grid_size=2, 
        mask_mode='channel', use_channel_attention=True, num_channel_attn_blocks=3)
    return model

