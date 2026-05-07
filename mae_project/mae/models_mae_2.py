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
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

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
        B, N, D = x.shape
        x_norm = self.norm(x)
        qkv = self.qkv(x_norm).reshape(B, N, 3, self.num_heads, D // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        dropout_p = self.attn_drop.p if self.training else 0.0
        x_attn = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
        x_attn = x_attn.transpose(1, 2).reshape(B, N, D)
        x_attn = self.proj(x_attn)
        x_attn = self.proj_drop(x_attn)
        return x + x_attn


class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=672, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16, n_img_mask = 1,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, grid_size=3,
                 mask_mode='spatial', use_channel_attention=False, num_channel_attn_blocks=2, out_chans=None,
                 mask_ratio=0.75):
        super().__init__()

        # Store grid size for masking logic
        self.grid_size = grid_size # e.g., 3 for a 3x3 grid
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.out_chans = out_chans if out_chans is not None else in_chans  # Default: same as input
        self.n_img_mask = n_img_mask
        self.embed_dim = embed_dim
        self.mask_ratio = mask_ratio
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
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * self.out_chans, bias=True) # decoder to patch
        
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

        # Learned per-channel mask value (replaces zeros — avoids domain gap with downstream)
        self.channel_mask_values = nn.Parameter(torch.zeros(in_chans))
        # Spectral embedding: tells the encoder which wavelengths are visible
        self.channel_embed = nn.Embedding(in_chans, embed_dim)

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
        nn.init.normal_(self.channel_embed.weight, std=0.02)

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


    def unpatchify(self, x, use_out_chans=True):
        """
        x: (N, L, patch_size**2 * C)
        imgs: (N, C, H, W)
        
        Args:
            use_out_chans: se True, usa self.out_chans (per predizioni decoder)
                          se False, usa self.in_chans (per patchify inverso)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        c = self.out_chans if use_out_chans else self.in_chans
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
        
        # Correct ids_restore: [kept_indices | masked_indices] → argsort = inverse permutation
        ids_masked = all_target_patch_indices  # sorted (torch.unique returns sorted)
        ids_shuffle = torch.cat([ids_keep_first, ids_masked], dim=0)   # [L]
        ids_restore = torch.argsort(ids_shuffle).unsqueeze(0).repeat(N, 1)  # [N, L]

        return x_masked, mask, ids_restore

    def random_masking(self, x, mask_ratio):
        """Spatial masking casuale: rimuove mask_ratio patch dall'encoder."""
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_kept = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))

        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_kept, mask, ids_restore

    def mask_specific_channels(self, imgs, channel_indices):
        """Sostituisce i canali mascherati con un valore appreso (evita domain gap con il downstream)."""
        imgs_masked = imgs.clone()
        C = imgs.shape[1]
        for idx in channel_indices:
            if idx < C:
                imgs_masked[:, idx, :, :] = self.channel_mask_values[idx]
        return imgs_masked



    def forward_encoder(self, x, n_img_mask=None):
        n_img_mask = n_img_mask if n_img_mask is not None else self.n_img_mask
        
        # Inizializza channel_indices per entrambe le modalità
        channel_indices = None
        
        if self.mask_mode == 'channel':
            if n_img_mask is None:
                n_img_mask = random.randint(1, self.in_chans - 1)
            channel_indices = random.sample(range(self.in_chans), n_img_mask)

            # Sostituisce i canali mascherati con il valore appreso, poi embedd
            x = self.patch_embed(self.mask_specific_channels(x, channel_indices))
            x = x + self.pos_embed[:, 1:, :]

            # Spectral embedding: somma degli embedding dei canali visibili
            visible_ids = [i for i in range(self.in_chans) if i not in channel_indices]
            spectral_emb = self.channel_embed(
                torch.tensor(visible_ids, dtype=torch.long, device=x.device)
            ).sum(0)  # (embed_dim,)
            x = x + spectral_emb[None, None, :]

            # Spatial masking per efficienza encoder: riduce la sequenza di mask_ratio
            x, mask, ids_restore = self.random_masking(x, self.mask_ratio)
            
        else:  # mask_mode == 'spatial'
            # Modalità mascheramento spaziale (originale)
            # embed patches
            x = self.patch_embed(x)

            # add pos embed w/o cls token
            x = x + self.pos_embed[:, 1:, :]

            # Spectral embedding: tutti i canali visibili in modalità spaziale
            all_ids = torch.arange(self.in_chans, dtype=torch.long, device=x.device)
            spectral_emb = self.channel_embed(all_ids).sum(0)
            x = x + spectral_emb[None, None, :]

            if n_img_mask is None:
                n_img_mask = random.randint(1, self.grid_size ** 2)

            subgrid_indices = random.sample(range(self.grid_size ** 2), n_img_mask)
            x, mask, ids_restore = self.mask_specific_subgrid(x, subgrid_indices)
            channel_indices = None  # spatial mode: no channel masking
        
        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks (gradient checkpointing per risparmiare VRAM con sequenze lunghe)
        for blk in self.blocks:
            x = checkpoint(blk, x, use_reentrant=False)
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
            x = checkpoint(blk, x, use_reentrant=False)
        
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

    def forward_loss(self, imgs, pred, channel_indices, spatial_mask):
        """
        imgs: [N, C, H, W]
        pred: [N, L, p²*C]
        channel_indices: canali mascherati (None in spatial masking mode)
        spatial_mask: [N, L], 1 = patch mascherata
        """
        target = self.patchify(imgs)  # [N, L, p²*C]
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2  # [N, L, p²*C]

        if self.mask_mode == 'channel' and channel_indices is not None:
            # Loss solo sui canali mascherati, su tutte le posizioni spaziali.
            # I canali sono azzerate ovunque → task: ricostruire il canale mancante
            # su ogni posizione usando i canali visibili.
            p = self.patch_embed.patch_size[0]
            C = self.in_chans
            ch_weight = torch.zeros(C, device=imgs.device)
            for idx in channel_indices:
                ch_weight[idx] = 1.0
            ch_weight = ch_weight.repeat(p * p)  # [p²*C]
            loss = (loss * ch_weight).sum(dim=-1) / ch_weight.sum()  # [N, L]
            return loss.mean()
        else:
            # Spatial masking mode: loss solo sulle patch mascherate, tutti i canali
            loss = loss.mean(dim=-1)  # [N, L]
            return (loss * spatial_mask).sum() / spatial_mask.sum()

    def forward(self, imgs, n_img_mask=None):
        latent, spatial_mask, ids_restore, channel_indices = self.forward_encoder(imgs, n_img_mask)
        pred = self.forward_decoder(latent, ids_restore, channel_indices)  # [N, L, p²*C]
        loss = self.forward_loss(imgs, pred, channel_indices, spatial_mask)
        return loss, pred, spatial_mask

def mae_model_for_pretraining(**kwargs):            #random masking
    """ MAE model for pretraining with random masking
    """
    model = MaskedAutoencoderViT(
        img_size=672, patch_size=14, embed_dim=768, depth=12, num_heads=12, n_img_mask=None, # Ensure img_size is correct if not default
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16, in_chans=3,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), grid_size=3, mask_mode='spatial') 
    return model

def mae_model_channel_masking_9ch_with_temporal_attn(**kwargs):
    img_size = kwargs.get('img_size', 1024)
    patch_size = kwargs.get('patch_size', 16)
    in_chans = kwargs.get('in_chans', 9)
    mask_ratio = kwargs.get('mask_ratio', 0.75)

    model = MaskedAutoencoderViT(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        n_img_mask=None,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        in_chans=in_chans,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        norm_pix_loss=False,
        grid_size=2,
        mask_mode='channel',
        use_channel_attention=True,
        num_channel_attn_blocks=3,
        mask_ratio=mask_ratio,
    )
    return model

def mae_model_fixed_channel_masking(**kwargs):
    """MAE model con numero fisso di canali mascherati.

    Args:
        n_channels_to_mask (int): numero di canali da mascherare ad ogni forward pass.
                                 Default: 3 (maschera 3 canali su 9).
        img_size (int): dimensione dell'immagine. Default: 1024.
        patch_size (int): dimensione della patch. Default: 16.
        in_chans (int): numero totale di canali. Default: 9.
    """
    img_size = kwargs.get('img_size', 1024)
    patch_size = kwargs.get('patch_size', 16)
    in_chans = kwargs.get('in_chans', 9)
    n_channels_masked = kwargs.get('n_channels_to_mask', 3)  # Corretto il nome del parametro

    assert 1 <= n_channels_masked < in_chans, (
        f"n_channels_masked ({n_channels_masked}) deve essere tra 1 e in_chans-1 ({in_chans - 1})"
    )

    model = MaskedAutoencoderViT(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        n_img_mask=n_channels_masked,  # valore fisso: non verrà randomizzato
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        in_chans=in_chans,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        norm_pix_loss=False,
        grid_size=2,
        mask_mode='channel',
        use_channel_attention=True,
        num_channel_attn_blocks=3
    )
    return model