import torch
import torch.nn as nn
import torch.nn.functional as F
import torch

import torch
import torch.nn.functional as F

class CommonFeatureDecoder(nn.Module):
    def __init__(self, patch_dim=768, target_size=224, threshold=0.7, num_classes=1, grid_size=None):
        super().__init__()
        self.patch_dim = patch_dim
        self.target_size = target_size
        self.num_classes = num_classes
        self.threshold = threshold
        self.grid_size = grid_size
        #self.image10_features = nn.Parameter(torch.randn(1, 256, patch_dim))
        self.conv3d_1 = nn.Conv3d(768, 768, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0))  # 9->5
        self.conv3d_2 = nn.Conv3d(768, 768, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0))  # 5->3
        self.conv3d_3 = nn.Conv3d(768, 768, kernel_size=(3, 1, 1), padding=0)  # 3->1
        # self.final_conv = nn.Sequential(
        #     nn.Conv2d(1, 16, 3, 1, 1),
        #     nn.ReLU(),
        #     nn.Conv2d(16, num_classes, 1, 1, 0)
        # )
        
    def forward(self, grid_features):
        """
        Estrae features comuni e genera maschera binaria
        grid_features: tensor di shape [B, num_patches, embed_dim]
        """
        # Calcola dinamicamente grid_size se non fornito
        B, num_patches, embed_dim = grid_features.shape
        if self.grid_size is None:
            grid_h = grid_w = int(num_patches ** 0.5)
        else:
            grid_h = grid_w = self.grid_size
        
        feature_grid = grid_features.reshape(B, grid_h, grid_w, embed_dim)

        # Dividi la griglia in 3x3 sub-grids dinamicamente
        sub_h = grid_h // 3
        sub_w = grid_w // 3
        
        image_features_list = []
        image_features_list = []
        for i in range(3):
            for j in range(3):
                sub_grid = feature_grid[:, i*sub_h:(i+1)*sub_h, j*sub_w:(j+1)*sub_w, :]
                #image_features_list.append(sub_grid.reshape(grid_features.shape[0], sub_h*sub_w, grid_features.shape[2]))
                image_features_list.append(sub_grid)
        stacked_features = torch.stack(image_features_list, dim=1) 
        x = stacked_features.permute(0, 4, 1, 2, 3)
        x = self.conv3d_1(x)
        x = self.conv3d_2(x)
        x = self.conv3d_3(x)
        output = x.squeeze(2)
        
        # feature_common_vector = torch.mean(stacked_features, dim=1) 
        # feature = feature_common_vector.permute(0,3,1,2)
        
        
        return output


class DecoderBlock(nn.Module):
    """
    Un blocco decoder più potente con una connessione residua.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # Blocco convoluzionale principale
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv_block(x)
        return x



class SingleImageSegmentationDecoder(nn.Module):
    def __init__(self, patch_dim=768, target_size=224):
        super().__init__()
        self.target_size = target_size
        
        self.channel_reduction = nn.Sequential(
            nn.Conv2d(768, 512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        # Decoder con skip connections
        self.decoder = nn.ModuleList([
            # 7x7 -> 14x14
            nn.Sequential(
                nn.ConvTranspose2d(512, 256, 4, 2, 1),
                nn.BatchNorm2d(256),
                nn.ReLU()
            ),
            # 14x14 -> 28x28
            nn.Sequential(
                nn.ConvTranspose2d(256, 128, 4, 2, 1),
                nn.BatchNorm2d(128),
                nn.ReLU()
            ),
            # 28x28 -> 56x56
            nn.Sequential(
                nn.ConvTranspose2d(128, 64, 4, 2, 1),
                nn.BatchNorm2d(64),
                nn.ReLU()
            ),
            # 56x56 -> 112x112
            nn.Sequential(
                nn.ConvTranspose2d(64, 32, 4, 2, 1),
                nn.BatchNorm2d(32),
                nn.ReLU()
            ),
            # 112x112 -> 224x224
            nn.Sequential(
                nn.ConvTranspose2d(32, 16, 4, 2, 1),
                nn.BatchNorm2d(16),
                nn.ReLU()
            )
        ])
        
        # Final layer con inizializzazione corretta
        self.final_conv = nn.Conv2d(16, 1, 3, 1, 1)
        
        # Inizializzazione critica
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # Bias finale leggermente positivo
        nn.init.constant_(self.final_conv.bias, 0.1)
    
    def forward(self, grid_features):
        # Aggregazione più sofisticata
        x = self.channel_reduction(grid_features)  # [B, 512, 16, 16]
        # Progressive upsampling
        for decoder_layer in self.decoder:
            x = decoder_layer(x)
        
        # Final output
        logits = self.final_conv(x)
        
        return logits


import torch.nn as nn
from torch.nn.modules.upsampling import Upsample
from torch.nn.functional import interpolate

class Upsample(nn.Module):
    def __init__(self, scale_factor, mode, align_corners=False):
        super(Upsample, self).__init__()
        self.interp = interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners=align_corners

    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode)
        return x
    
    
class _SepConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(_SepConv2d, self).__init__()
        self.conv_s = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False, groups=in_planes)
        self.bn_s = nn.BatchNorm2d(out_planes)
        self.relu_s = nn.ReLU()

        self.conv_t = nn.Conv2d(out_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_t = nn.BatchNorm2d(out_planes)
        self.relu_t = nn.ReLU()

    def forward(self, x):
        x = self.conv_s(x)
        x = self.bn_s(x)
        x = self.relu_s(x)

        x = self.conv_t(x)
        x = self.bn_t(x)
        x = self.relu_t(x)
        return x 

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-3, momentum=0.001, affine=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
class ChannelAttention(nn.Module):
    """Channel Attention Module per dare più peso ai canali importanti"""
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = self.sigmoid(avg_out + max_out)
        return x * out


class SpatialAttention(nn.Module):
    """Spatial Attention Module per focalizzarsi sulle regioni spaziali importanti"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(x_cat))
        return x * attention


class DeepDecoderBlock(nn.Module):
    """
    Blocco decoder profondo con:
    - Multiple convoluzioni
    - Connessioni residuali
    - Attention mechanisms
    - Dropout per regolarizzazione
    """
    def __init__(self, in_channels, out_channels, num_conv=3, use_attention=True, dropout=0.1):
        super(DeepDecoderBlock, self).__init__()
        self.use_attention = use_attention
        
        # Prima convoluzione per cambiare il numero di canali
        self.conv_reduce = BasicConv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        
        # Multiple convoluzioni sequenziali
        conv_layers = []
        for i in range(num_conv):
            conv_layers.append(_SepConv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
            if dropout > 0:
                conv_layers.append(nn.Dropout2d(dropout))
        
        self.conv_block = nn.Sequential(*conv_layers)
        
        # Attention modules
        if use_attention:
            self.channel_attention = ChannelAttention(out_channels)
            self.spatial_attention = SpatialAttention()
        
        # Connessione residuale
        self.residual_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        # Riduzione canali
        x = self.conv_reduce(x)
        
        # Salva per connessione residuale
        identity = x
        
        # Convoluzioni multiple
        x = self.conv_block(x)
        
        # Attention
        if self.use_attention:
            x = self.channel_attention(x)
            x = self.spatial_attention(x)
        
        # Connessione residuale
        x = x + self.residual_conv(identity)
        
        return x


class DeepDecoder(nn.Module):
    """
    Decoder profondo e migliorato per segmentazione
    Più layer, attention, residual connections
    """
    def __init__(self, num_classes=2, patch_dim=768, in_channel=512, 
                 out_channel=[256, 128, 64, 32], out_sigmoid=False, dropout=0.1):
        super(DeepDecoder, self).__init__()
        self.num_classes = num_classes
        self.out_sigmoid = out_sigmoid
        
        # Riduzione canali da 768 a 512 con più profondità
        self.channel_reduction = nn.Sequential(
            nn.Conv2d(768, 640, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(640),
            nn.ReLU(inplace=True),
            nn.Conv2d(640, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        )
        
        # Decoder blocks più profondi con attention
        # 16x16 -> 32x32
        self.decoder_block1 = DeepDecoderBlock(in_channel, out_channel[0], num_conv=4, use_attention=True, dropout=dropout)
        self.upsample1 = Upsample(scale_factor=2, mode='bilinear')
        
        # 32x32 -> 64x64
        self.decoder_block2 = DeepDecoderBlock(out_channel[0], out_channel[1], num_conv=4, use_attention=True, dropout=dropout)
        self.upsample2 = Upsample(scale_factor=2, mode='bilinear')
        
        # 64x64 -> 128x128
        self.decoder_block3 = DeepDecoderBlock(out_channel[1], out_channel[2], num_conv=3, use_attention=True, dropout=dropout)
        self.upsample3 = Upsample(scale_factor=2, mode='bilinear')
        
        # 128x128 -> 224x224
        self.decoder_block4 = DeepDecoderBlock(out_channel[2], out_channel[3], num_conv=3, use_attention=True, dropout=dropout)
        self.upsample4 = Upsample(scale_factor=1.75, mode='bilinear')
        
        # Blocco finale di raffinamento
        self.final_refinement = nn.Sequential(
            BasicConv2d(out_channel[3], out_channel[3], kernel_size=3, stride=1, padding=1),
            _SepConv2d(out_channel[3], out_channel[3], kernel_size=3, stride=1, padding=1),
            nn.Conv2d(out_channel[3], out_channel[3]//2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel[3]//2),
            nn.ReLU(inplace=True)
        )
        
        # Layer finale per ottenere il numero di classi desiderato
        self.last_conv = nn.Conv2d(out_channel[3]//2, num_classes, kernel_size=1, stride=1, bias=True)
        
        if self.out_sigmoid:
            self.sigmoid = nn.Sigmoid()
        
        self._init_weights()
     
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # Bias finale leggermente positivo
        nn.init.constant_(self.last_conv.bias, 0.1)
    
    def forward(self, grid_features):
        # Input: [B, 768, H, W] dove H,W possono essere 12x12, 16x16, 21x21, o 32x32
        # Output: [B, num_classes, 512, 512]
        # Riduzione canali: [B, 768, H, W] -> [B, 512, H, W]
        B, C, H, W = grid_features.shape
        x = self.channel_reduction(grid_features)
        

        if H == 16 and W == 16:
            # Input 16x16: percorso 16->32->64->128->256->512
            x = self.decoder_block1(x)  # [B, 256, 16, 16]
            x = self.upsample1(x)       # [B, 256, 32, 32]
            
            x = self.decoder_block2(x)  # [B, 128, 32, 32]
            x = self.upsample2(x)       # [B, 128, 64, 64]
            
            x = self.decoder_block3(x)  # [B, 64, 64, 64]
            x = self.upsample3(x)       # [B, 64, 128, 128]
            
            x = self.decoder_block4(x)  # [B, 32, 128, 128]
            x = self.upsample4(x)       # [B, 32, 256, 256]
            
            # Ulteriore upsampling 256->512
            x = nn.functional.interpolate(x, size=(512, 512), mode='bilinear', align_corners=True)
            
        elif H == 21 and W == 21:
            # Input 21x21: upsample per raggiungere 512
            # 21 -> 64 -> 128 -> 256 -> 512
            x = self.decoder_block1(x)  # [B, 256, 21, 21]
            x = nn.functional.interpolate(x, size=(64, 64), mode='bilinear', align_corners=True)
            
            x = self.decoder_block2(x)  # [B, 128, 64, 64]
            x = nn.functional.interpolate(x, size=(128, 128), mode='bilinear', align_corners=True)
            
            x = self.decoder_block3(x)  # [B, 64, 128, 128]
            x = nn.functional.interpolate(x, size=(256, 256), mode='bilinear', align_corners=True)
            
            x = self.decoder_block4(x)  # [B, 32, 256, 256]
            x = nn.functional.interpolate(x, size=(512, 512), mode='bilinear', align_corners=True)
            
        elif H == 32 and W == 32:
            # Input 32x32: percorso 32->64->128->256->512
            x = self.decoder_block1(x)  # [B, 256, 32, 32]
            
            x = self.decoder_block2(x)  # [B, 128, 32, 32]
            x = self.upsample2(x)       # [B, 128, 64, 64]
            
            x = self.decoder_block3(x)  # [B, 64, 64, 64]
            x = self.upsample3(x)       # [B, 64, 128, 128]
            
            x = self.decoder_block4(x)  # [B, 32, 128, 128]
            x = nn.functional.interpolate(x, size=(512, 512), mode='bilinear', align_corners=True)
            
        else:
            raise ValueError(f"Unsupported input spatial size: {H}x{W}. Supported sizes: 12x12, 16x16, 21x21, or 32x32")
        
        # Raffinamento finale
        x = self.final_refinement(x)  # [B, 16, 512, 512]
        
        # Output finale
        x = self.last_conv(x)       # [B, num_classes, 512, 512]
        
        if self.out_sigmoid:
            x = self.sigmoid(x)
        
        return x


class Decoder5(nn.Module):
    def __init__(self, num_classes=2, patch_dim=768, in_channel=512, out_channel=[256, 128, 64, 32], out_sigmoid=False):
        super(Decoder5, self).__init__()
        self.num_classes = num_classes
        self.out_sigmoid = out_sigmoid
        
        # Riduzione canali da 768 a 512
        self.channel_reduction = nn.Sequential(
            nn.Conv2d(768, 512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        # Decoder layers per upsampling da 16x16 a 224x224
        # 16x16 -> 32x32 -> 64x64 -> 128x128 -> 224x224 (circa 4 step di upsampling)
        self.deconvlayer5_5 = self._make_deconv(in_channel, out_channel[0], num_conv=3)
        self.upsample5_5 = Upsample(scale_factor=2, mode='bilinear')  # 16->32
        
        self.deconvlayer5_4 = self._make_deconv(out_channel[0], out_channel[1], num_conv=3)
        self.upsample5_4 = Upsample(scale_factor=2, mode='bilinear')  # 32->64
        
        self.deconvlayer5_3 = self._make_deconv(out_channel[1], out_channel[2], num_conv=2)
        self.upsample5_3 = Upsample(scale_factor=2, mode='bilinear')  # 64->128
        
        self.deconvlayer5_2 = self._make_deconv(out_channel[2], out_channel[3], num_conv=2)
        self.upsample5_2 = Upsample(scale_factor=1.75, mode='bilinear')  # 128->224
        
        # Layer finale per ottenere il numero di classi desiderato
        self.last_conv5 = nn.Conv2d(out_channel[3], num_classes, kernel_size=1, stride=1, bias=True)
        
        if self.out_sigmoid:
            self.sigmoid = nn.Sigmoid()
        
        self._init_weights()
     
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        nn.init.constant_(self.last_conv5.bias, 0.1)
    
    def _make_deconv(self, in_channel, out_channel, num_conv=1, kernel_size=3, stride=1, padding=1):
        layers = []
        layers.append(BasicConv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding))
        for i in range(1, num_conv):
            layers.append(_SepConv2d(out_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding))
        
        return nn.Sequential(*layers)
    
    def forward(self, grid_features):
        # Input: [B, 768, 16, 16]
        # Riduzione canali: [B, 768, 16, 16] -> [B, 512, 16, 16]
        x = self.channel_reduction(grid_features)
        
        # Upsampling progressivo
        x = self.deconvlayer5_5(x)  # [B, 256, 16, 16]
        x = self.upsample5_5(x)     # [B, 256, 32, 32]
        
        x = self.deconvlayer5_4(x)  # [B, 128, 32, 32]
        x = self.upsample5_4(x)     # [B, 128, 64, 64]
        
        x = self.deconvlayer5_3(x)  # [B, 64, 64, 64]
        x = self.upsample5_3(x)     # [B, 64, 128, 128]
        
        x = self.deconvlayer5_2(x)  # [B, 32, 128, 128]
        x = self.upsample5_2(x)     # [B, 32, 224, 224]
        
        # Output finale
        x = self.last_conv5(x)      # [B, 2, 224, 224]
        
        if self.out_sigmoid:
            x = self.sigmoid(x)
        
        return x
    
    
class MAESegmentationModel(nn.Module):
    def __init__(self, mae_model, num_classes=1, freeze_encoder=True, decoder_type='deep', dropout=0.1):
        """
        Args:
            mae_model: Modello MAE pre-addestrato
            num_classes: Numero di classi per la segmentazione
            freeze_encoder: Se True, congela i pesi dell'encoder
            decoder_type: Tipo di decoder da usare ('basic' o 'deep')
            dropout: Dropout rate per il decoder profondo
        """
        super().__init__()
        
        # Encoder components
        self.patch_embed = mae_model.patch_embed
        self.cls_token = mae_model.cls_token
        self.pos_embed = mae_model.pos_embed
        self.blocks = mae_model.blocks
        self.norm = mae_model.norm
        self.num_classes = num_classes
        
        # Parameters
        self.embed_dim = mae_model.embed_dim
        self.patch_size = mae_model.patch_size
        self.img_size = mae_model.img_size
        self.grid_size = mae_model.grid_size
        self.single_img_size = self.img_size // 3
        
        # Calcola le dimensioni corrette
        self.patches_per_side = self.img_size // self.patch_size
        
        # Decoder più profondo
        self.extract_feature = CommonFeatureDecoder(patch_dim=self.embed_dim, target_size=self.img_size, threshold=0.5, num_classes=self.num_classes)
        #self.seg_decoder = SingleImageSegmentationDecoder(patch_dim=self.embed_dim, target_size=self.img_size)
        
        # Decoder - scelta tra basic e deep
        self.extract_feature = CommonFeatureDecoder(patch_dim=self.embed_dim, target_size=self.img_size, 
                                                    threshold=0.5, num_classes=self.num_classes, 
                                                    grid_size=self.patches_per_side)
        
        if decoder_type == 'deep':
            print(f"Using DeepDecoder with dropout={dropout}")
            self.seg_decoder = DeepDecoder(in_channel=512, out_channel=[256, 128, 64, 32], 
                                          out_sigmoid=False, num_classes=self.num_classes, dropout=dropout)
        else:
            print("Using Decoder5 (basic)")
            self.seg_decoder = Decoder5(in_channel=512, out_channel=[256, 128, 64, 32], 
                                       out_sigmoid=False, num_classes=self.num_classes)
        
        # Freeze encoder

        
        if freeze_encoder:
            self._freeze_encoder()
        
    def _freeze_encoder(self):
        """Freeze encoder parameters"""
        for param in self.patch_embed.parameters():
            param.requires_grad = False
        
        self.cls_token.requires_grad = False
        self.pos_embed.requires_grad = False
        
        for block in self.blocks:
            for param in block.parameters():
                param.requires_grad = False
        
        for param in self.norm.parameters():
            param.requires_grad = False
        
        print("Encoder frozen - only decoder will be trained")
        
    def forward(self, x):
        # Encoder
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed
        
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)                
        
        x = x[:, 1:, :]  # [N, num_patches, embed_dim]
        
        common_feature = self.extract_feature(x)
        output = self.seg_decoder(common_feature)
        return output
 
 
 


####MODIFICHE APPORTATE 5/01/2026

import torch
import torch.nn as nn

class PixelShuffleUpsample(nn.Module):
    """Upsample usando PixelShuffle per ridurre gli artefatti 'checkerboard'"""
    def __init__(self, in_channels, out_channels, upscale_factor=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * (upscale_factor ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.bn(x)
        return self.relu(x)

class ImprovedDeepDecoder(nn.Module):
    def __init__(self, num_classes=2, in_channels=768, dropout=0.1):
        super().__init__()
        
        # 1. Riduzione iniziale
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        # 2. Sequenza di Upsampling Progressivo (assumendo input 16x16)
        # 16 -> 32
        self.up1 = PixelShuffleUpsample(512, 256) 
        # 32 -> 64
        self.up2 = PixelShuffleUpsample(256, 128)
        # 64 -> 128
        self.up3 = PixelShuffleUpsample(128, 64)
        # 128 -> 256
        self.up4 = PixelShuffleUpsample(64, 32)
        # 256 -> 512
        self.up5 = PixelShuffleUpsample(32, 16)

        self.final_head = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, num_classes, kernel_size=1)
        )

    def forward(self, x):
        # Se x è una sequenza di token [B, L, D], reshape in [B, D, H, W] prima
        x = self.reduce(x)
        
        x = self.up1(x) # 32x32
        x = self.up2(x) # 64x64
        x = self.up3(x) # 128x128
        x = self.up4(x) # 256x256
        x = self.up5(x) # 512x512
        
        return self.final_head(x)
    
class MAEFeatureExtractor(nn.Module):
    def __init__(self, mae_model):
        super().__init__()
        self.mae = mae_model
        # Definiamo i layer da cui vogliamo estrarre le feature
        # Se il MAE ha 12 blocchi, prendiamo i quarti
        self.intermediate_layers = [2, 5, 8, 11] 

    def forward(self, x):
        x = self.mae.patch_embed(x)
        # Aggiunge pos_embed escludendo il CLS se presente
        if x.shape[1] == self.mae.pos_embed.shape[1] - 1:
            x = x + self.mae.pos_embed[:, 1:, :]
        else:
            x = x + self.mae.pos_embed
        
        features = []
        for i, blk in enumerate(self.mae.blocks):
            x = blk(x)
            if i in self.intermediate_layers:
                features.append(x)
        
        B, L, C = features[0].shape
        # Se L è dispari (es 1025), c'è il CLS token in posizione 0
        has_cls = (L % 2 != 0)
        H_p = W_p = int((L-1)**0.5) if has_cls else int(L**0.5)
        
        output_features = []
        for f in features:
            if has_cls:
                f = f[:, 1:, :] # Rimuove il CLS token
            f_2d = f.transpose(1, 2).reshape(B, C, H_p, W_p)
            output_features.append(f_2d)
                
        return output_features
    
class UNetViTDecoder(nn.Module):
    def __init__(self, num_classes=2, embed_dim=768):
        super().__init__()
        
        # Blocchi di raffinamento per portare i token del MAE a dimensioni gestibili
        self.refine4 = nn.Conv2d(embed_dim, 512, kernel_size=3, padding=1)
        self.refine3 = nn.Conv2d(embed_dim, 256, kernel_size=3, padding=1)
        self.refine2 = nn.Conv2d(embed_dim, 128, kernel_size=3, padding=1)
        self.refine1 = nn.Conv2d(embed_dim, 64, kernel_size=3, padding=1)

        # Upsampling progressivo (3 step per patch 8: 64->128, 128->256, 256->512)
        self.up1 = PixelShuffleUpsample(512, 256)       # 64 -> 128
        self.up2 = PixelShuffleUpsample(256 + 256, 128) # 128 -> 256 (in: up1 + refine3)
        self.up3 = PixelShuffleUpsample(128 + 128, 64)  # 256 -> 512 (in: up2 + refine2)
        
        # Testa finale profonda
        self.final_head = nn.Sequential(
            nn.Conv2d(64 + 64, 64, kernel_size=3, padding=1), # 64 (up3) + 64 (refine1)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=1)
        )

    def forward(self, features):
        # features sono tutti 64x64 (Patch 8 con input 512)
        f1, f2, f3, f4 = features 

        # 64 -> 128
        x = self.refine4(f4)      
        x = self.up1(x)           

        # 128 (Concat con Layer 9)
        s3 = self.refine3(f3)     
        s3 = nn.functional.interpolate(s3, size=x.shape[2:], mode='bilinear')
        x = torch.cat([x, s3], dim=1)
        x = self.up2(x)           

        # 256 (Concat con Layer 6)
        s2 = self.refine2(f2)     
        s2 = nn.functional.interpolate(s2, size=x.shape[2:], mode='bilinear')
        x = torch.cat([x, s2], dim=1)
        x = self.up3(x)           

        # 512 (Concat finale con Layer 3)
        s1 = self.refine1(f1)     
        s1 = nn.functional.interpolate(s1, size=x.shape[2:], mode='bilinear')
        x = torch.cat([x, s1], dim=1)
        
        # Output finale raffinato
        return self.final_head(x)
    
    
class MAE_UNet_Segmentation(nn.Module):
    def __init__(self, mae_model, num_classes=2):
        super().__init__()
        self.encoder = MAEFeatureExtractor(mae_model)
        # Assumendo embed_dim=768 per il MAE Base
        self.decoder = UNetViTDecoder(num_classes=num_classes, embed_dim=768)

    def forward(self, x):
        # Estrae la lista di 4 feature map (layer 3, 6, 9, 12)
        features = self.encoder(x)
        # Il decoder fa skip-connection e upsampling fino a 512x512
        mask = self.decoder(features)
        return mask