# Codes adapted from MONAI

# Citation:
# Cardoso, M. Jorge, et al.
# "Monai: An open-source framework for deep learning in healthcare."
# arXiv preprint arXiv:2211.02701 (2022).
# Open-Source code: https://github.com/Project-MONAI/MONAI/tree/dev

import torch
import torch.nn as nn

from monai.networks.nets.swin_unetr import SwinTransformer as SwinViT
from monai.utils import ensure_tuple_rep


class SSLHead2D(nn.Module):
    """
    args:
    - spatial_dims = spatial dim of input data (int: 3 or 2)
    - in_channels = no. of input channels (default: 1)
    - feature_size = embedding size (default: 48)
    - dropout_path_rate = drop path rate (default: 0.0)
    - use_checkpoint = use gradient checkpointing to save memory (action="store_true")

    """
    def __init__(self, args, upsample="vae", dim=64):
        super(SSLHead2D, self).__init__()
        patch_size = ensure_tuple_rep(2, args["spatial_dims"])
        window_size = ensure_tuple_rep(7, args["spatial_dims"])
        self.swinViT = SwinViT(
            in_chans=args["in_channels"], # 1 for grayscale
            embed_dim=args["feature_size"], # 48 default (first block embedding size(?))
            window_size=window_size,
            patch_size=patch_size,
            depths=[2, 2, 2, 2], # len(depth) determine how many layers in each stage
            num_heads=[4, 8, 16, 32],
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=args["dropout_path_rate"],
            norm_layer=torch.nn.LayerNorm,
            use_checkpoint=args["use_checkpoint"],
            spatial_dims=args["spatial_dims"],
        )
        self.rotation_pre = nn.Identity()
        self.rotation_head = nn.Linear(dim, 4) # 4 for 4 rotations 
        self.contrastive_pre = nn.Identity()
        self.contrastive_head = nn.Linear(dim, 256) # sure?
        
        if upsample == "large_kernel_deconv":
            self.conv = nn.ConvTranspose3d(dim, args["in_channels"], kernel_size=(32, 32, 32), stride=(32, 32, 32))
        elif upsample == "deconv":
            self.conv = nn.Sequential(
                nn.ConvTranspose3d(dim, dim // 2, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
                nn.ConvTranspose3d(dim // 2, dim // 4, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
                nn.ConvTranspose3d(dim // 4, dim // 8, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
                nn.ConvTranspose3d(dim // 8, dim // 16, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
                nn.ConvTranspose3d(dim // 16, args["in_channels"], kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            )
        elif upsample == "vae":
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 2, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm2d(dim // 2),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(dim // 2, dim // 4, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm2d(dim // 4),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(dim // 4, dim // 8, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm2d(dim // 8),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(dim // 8, dim // 16, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm2d(dim // 16),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(dim // 16, dim // 16, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm2d(dim // 16),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(dim // 16, args["in_channels"], kernel_size=1, stride=1),
            )

    def forward(self, x):
        x_out = self.swinViT(x.contiguous())[4] 
        _, c, h, w = x_out.shape
        
        x_reshape = x_out.flatten(start_dim=2, end_dim=3)
        
        # reconstruction head
        x_rec = x_reshape.view(-1, c, h, w)
        x_rec = self.conv(x_rec) # pass to vae for reconstructing
        
        # rotation head
        x_reshape = x_reshape.transpose(1, 2) # sure?
        x_rot = self.rotation_pre(x_reshape[:, 0])
        x_rot = self.rotation_head(x_rot)
        
        # contrastive head
        x_con = self.contrastive_pre(x_reshape[:, 1])
        x_con = self.contrastive_head(x_con)
        
        return x_rec, x_rot, x_con


class SSLHead2DSim(nn.Module):
    """
    args:
    - spatial_dims = spatial dim of input data (int: 3 or 2)
    - in_channels = no. of input channels (default: 1)
    - feature_size = embedding size (default: 48)
    - dropout_path_rate = drop path rate (default: 0.0)
    - use_checkpoint = use gradient checkpointing to save memory (action="store_true")

    """
    def __init__(self, args, upsample="vae", dim=64):
        super(SSLHead2DSim, self).__init__()
        patch_size = ensure_tuple_rep(2, args["spatial_dims"])
        window_size = ensure_tuple_rep(7, args["spatial_dims"])
        self.swinViT = SwinViT(
            in_chans=args["in_channels"], # 1 for grayscale
            embed_dim=args["feature_size"], # 48 default (first block embedding size(?))
            window_size=window_size,
            patch_size=patch_size,
            depths=[2, 2, 2, 2], # len(depth) determine how many layers in each stage
            num_heads=[4, 8, 16, 32],
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=args["dropout_path_rate"],
            norm_layer=torch.nn.LayerNorm,
            use_checkpoint=args["use_checkpoint"],
            spatial_dims=args["spatial_dims"],
        )

        if upsample == "large_kernel_deconv":
            self.conv = nn.ConvTranspose3d(dim, args["in_channels"], kernel_size=(32, 32, 32), stride=(32, 32, 32))
        elif upsample == "deconv":
            self.conv = nn.Sequential(
                nn.ConvTranspose3d(dim, dim // 2, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
                nn.ConvTranspose3d(dim // 2, dim // 4, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
                nn.ConvTranspose3d(dim // 4, dim // 8, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
                nn.ConvTranspose3d(dim // 8, dim // 16, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
                nn.ConvTranspose3d(dim // 16, args["in_channels"], kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            )
        elif upsample == "vae":
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 2, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm2d(dim // 2),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(dim // 2, dim // 4, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm2d(dim // 4),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(dim // 4, dim // 8, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm2d(dim // 8),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(dim // 8, dim // 16, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm2d(dim // 16),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(dim // 16, dim // 16, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm2d(dim // 16),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(dim // 16, args["in_channels"], kernel_size=1, stride=1),
            )

    def forward(self, x):
        x_out = self.swinViT(x.contiguous())[4] # sure?
        _, c, h, w = x_out.shape

        x_rec = x_out.flatten(start_dim=2, end_dim=3) # sure?
        x_rec = x_rec.view(-1, c, h, w)
        x_rec = self.conv(x_rec) # pass to vae for reconstructing
        return x_rec
