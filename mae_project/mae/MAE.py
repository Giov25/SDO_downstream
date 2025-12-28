import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn.functional as F
from skimage.measure import block_reduce

from . import reconstructions
from .mae3d import MaskedAutoencoderViT3D

ALL_WAVELENGTHS = [
    "131A",
    "1600A",
    "1700A",
    "171A",
    "193A",
    "211A",
    "304A",
    "335A",
    "94A",
]
import lightning.pytorch as pl
import torch


class BaseModule(pl.LightningModule):
    def __init__(
        self,
        optimiser: str = "adam",
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        hyperparam_ignore=[],
        # pass to pl.LightningModule
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters(ignore=hyperparam_ignore)

        # optimiser values
        self.optimiser = optimiser
        self.lr = lr
        self.weight_decay = weight_decay

    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError

    def configure_optimizers(self):
        match (self.optimiser):
            case "adam":
                optimiser = torch.optim.Adam(
                    self.parameters(),
                    lr=self.lr,
                    weight_decay=self.weight_decay,
                )
            case "sgd":
                optimiser = torch.optim.SGD(
                    self.parameters(),
                    lr=self.lr,
                    weight_decay=self.weight_decay,
                )
            case "adamw":
                optimiser = torch.optim.AdamW(
                    self.parameters(),
                    lr=self.lr,
                    weight_decay=self.weight_decay,
                )
            case _:
                raise NameError(f"Unknown optimizer {optimiser}")
        return optimiser

class MAE(BaseModule):
    def __init__(
        self,
        # MAE specific
        img_size=224,
        patch_size=16,
        num_frames=3,
        tubelet_size=1,
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_layer="LayerNorm",
        norm_pix_loss=False,
        masking_ratio=0.75,
        limb_mask=None,
        # pass to BaseModule
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # self.validation_step_outputs = {'x': [], 'x_hat': []}
        self.validation_metrics = []
        self.masking_ratio = masking_ratio

        # block reduce limb_mask
        limb_mask_ids = None
        if limb_mask is not None:
            new_matrix = block_reduce(
                limb_mask.numpy(), block_size=(16, 16), func=np.max
            )
            limb_mask_ids = torch.tensor(
                np.argwhere(new_matrix.reshape(1024) == 0).reshape(-1)
            )

        self.autoencoder = MaskedAutoencoderViT3D(
            img_size,
            patch_size,
            num_frames,
            tubelet_size,
            in_chans,
            embed_dim,
            depth,
            num_heads,
            decoder_embed_dim,
            decoder_depth,
            decoder_num_heads,
            mlp_ratio,
            norm_layer,
            norm_pix_loss,
            limb_mask_ids,
        )
        # self.autoencoder = PrithviEncoder(self.mae)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x = batch
        loss, x_hat, mask = self.autoencoder(x, mask_ratio=self.masking_ratio)
        x_hat = self.autoencoder.unpatchify(x_hat)
        loss = F.mse_loss(x_hat, x)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        loss, x_hat, mask = self.autoencoder(x, mask_ratio=self.masking_ratio)
        x_hat = self.autoencoder.unpatchify(x_hat)
        loss = F.mse_loss(x_hat, x)
        for i in range(x.shape[0]):
            for frame in range(x.shape[2]):
                self.validation_metrics.append(
                    reconstructions.get_metrics(
                        x[i, :, frame, :, :], x_hat[i, :, frame, :, :], ALL_WAVELENGTHS
                    )
                )
        return loss
        #self.log("val_loss", loss) ho tolto perchè ho aggiunto il return

    def forward(self, x):
        loss, x_hat, mask = self.autoencoder(x, mask_ratio=self.masking_ratio)
        x_hat = self.autoencoder.unpatchify(x_hat)
        return loss, x_hat, mask

    def forward_encoder(self, x, mask_ratio):
        return self.autoencoder.forward_encoder(x, mask_ratio=mask_ratio)

    def on_validation_epoch_end(self):

        merged_metrics = reconstructions.merge_metrics(self.validation_metrics)
        batch_metrics = reconstructions.mean_metrics(merged_metrics)

        if isinstance(self.logger, pl.loggers.wandb.WandbLogger):
            import wandb
            from pandas import DataFrame

            # this only occurs on rank zero only
            df = DataFrame(batch_metrics)
            df["mean"] = df.mean(numeric_only=True, axis=1)
            df["metric"] = df.index
            cols = df.columns.tolist()
            self.logger.log_table(
                key="val_reconstruction",
                dataframe=df[cols[-1:] + cols[:-1]],
                step=self.validation_step,
            )
            for k, v in batch_metrics.items():
                # sync_dist as this tries to include all
                for i, j in v.items():
                    self.log(f"val_{k}_{i}", j)

            # model_artifact = wandb.Artifact("model", type="model")
            # model_artifact.add_reference(f"gs://sdofm-checkpoints/{wandb.run.id}-{wandb.run.name}/model-step{wandb.run.step}.ckpt")
        else:
            for k in batch_metrics.keys():
                batch_metrics[k]["channel"] = k
            for k, v in batch_metrics.items():
                # sync_dist as this tries to include all
                self.log_dict(v, sync_dist=True)  # This doesn't work?

        # reset
        # self.validation_step_outputs['x'].clear()
        # self.validation_step_outputs['x_hat'].clear()
        self.validation_metrics.clear()


def new_mae_trial_small_patches(**kwargs):
    model = MAE(
        img_size=512, 
        num_frames=1, 
        patch_size=16,  # ← Patch più piccole
        tubelet_size=1, 
        embed_dim=128,  # ← Mantieni dimensione embedding bassa
        depth=12,       # ← Riduci depth per compensare più patches
        num_heads=8,    # ← Riduci num_heads 
        decoder_embed_dim=256,  # ← Riduci decoder dim
        decoder_depth=4,        # ← Riduci decoder depth
        decoder_num_heads=8,    # ← Riduci decoder heads
        in_chans=9, 
        mlp_ratio=2,    # ← Riduci mlp_ratio
        norm_layer='LayerNorm'
    )
    return model

def new_mae_trial_256_patches(**kwargs):
    # 512/32 = 16, quindi 16² = 256 patches
    model = MAE(
        img_size=512,    
        num_frames=1, 
        patch_size=32,   
        tubelet_size=1, 
        embed_dim=96,    # ← Riduci ancora per 256 patches
        depth=8,         # ← Depth più basso
        num_heads=6,     
        decoder_embed_dim=192, 
        decoder_depth=3,       
        decoder_num_heads=6,   
        in_chans=9, 
        mlp_ratio=2,    
        norm_layer='LayerNorm'
    )
    return model

