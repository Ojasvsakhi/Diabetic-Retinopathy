import torch
import torch.nn as nn
from torchvision import models
from transformers import ViTModel, ViTConfig


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block that handles both 4D (B,C,H,W) and 2D (B,C) inputs."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channels = channels
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: either [B,C,H,W] or [B,C]
        if x.dim() == 4:
            b, c, _, _ = x.size()
            y = self.squeeze(x).view(b, c)
            y = self.excitation(y).view(b, c, 1, 1)
            return x * y.expand_as(x)
        elif x.dim() == 2:
            y = self.excitation(x)
            return x * y
        else:
            return x


class AttentionFusion(nn.Module):
    """Simple feature fusion: project and concatenate CNN and ViT features."""
    def __init__(self, dim1, dim2, hidden=512):
        super().__init__()
        self.proj1 = nn.Linear(dim1, hidden)
        self.proj2 = nn.Linear(dim2, hidden)
        self.out_proj = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(hidden * 2, dim1 + dim2)
        )

    def forward(self, cnn_feat, vit_feat):
        # cnn_feat: [B, dim1]  vit_feat: [B, dim2]
        p1 = self.proj1(cnn_feat)
        p2 = self.proj2(vit_feat)
        fused = torch.cat([p1, p2], dim=1)
        out = self.out_proj(fused)
        return out


class CNNViTHybrid(nn.Module):
    """Hybrid model combining a ResNet CNN backbone and a ViT global branch.

    Usage:
        model = CNNViTHybrid(num_classes=5, pretrained_cnn=True, pretrained_vit=True)
        logits = model(images)  # images: [B,3,H,W]
    """
    def __init__(self, num_classes=5, pretrained_cnn=True, pretrained_vit=True):
        super().__init__()
        # CNN Branch (ResNet50)
        backbone = models.resnet50(pretrained=pretrained_cnn)
        backbone.fc = nn.Identity()
        self.cnn = backbone
        cnn_dim = 2048

        # Vision Transformer Branch
        if pretrained_vit:
            try:
                self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224')
                vit_dim = self.vit.config.hidden_size
            except Exception:
                # fallback to randomly initialized ViT
                cfg = ViTConfig()
                self.vit = ViTModel(cfg)
                vit_dim = cfg.hidden_size
        else:
            cfg = ViTConfig()
            self.vit = ViTModel(cfg)
            vit_dim = cfg.hidden_size

        # SE Block
        self.se_block = SEBlock(cnn_dim, reduction=16)

        # Fusion and classifier
        self.attention_fusion = AttentionFusion(cnn_dim, vit_dim)
        fused_dim = cnn_dim + vit_dim

        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # x: [B,3,H,W]
        # CNN features
        cnn_feat = self.cnn(x)  # [B, 2048]
        # If CNN produced 4D (some backbones), flatten
        if cnn_feat.dim() == 4:
            cnn_feat = cnn_feat.view(cnn_feat.size(0), -1)
        cnn_feat = self.se_block(cnn_feat)

        # ViT features
        # ViTModel expects pixel_values shaped [B,3,H,W]
        vit_out = self.vit(pixel_values=x)
        # CLS token
        vit_feat = vit_out.last_hidden_state[:, 0]

        fused = self.attention_fusion(cnn_feat, vit_feat)
        logits = self.classifier(fused)
        return logits
