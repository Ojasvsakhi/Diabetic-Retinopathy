import torch
import torch.nn as nn
import torchvision.models as models
try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("Warning: timm not available. Vision Transformer models will not work. Install with: pip install timm")


class MultiTaskModel(nn.Module):
    """Multi-task model supporting both ResNet and Vision Transformer backbones."""
    def __init__(self, backbone_name='resnet50', num_classes=5, aux_output=1, pretrained=True):
        super().__init__()
        self.backbone_name = backbone_name
        
        if backbone_name == 'resnet50':
            # Fix deprecated pretrained parameter
            try:
                from torchvision.models import ResNet50_Weights
                weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
                backbone = models.resnet50(weights=weights)
            except ImportError:
                # Fallback for older torchvision versions
                backbone = models.resnet50(pretrained=pretrained)
            in_features = backbone.fc.in_features
            backbone.fc = nn.Identity()
            self.backbone = backbone
            
        elif backbone_name.startswith('vit'):
            if not TIMM_AVAILABLE:
                raise ImportError("timm is required for Vision Transformer. Install with: pip install timm")
            
            # Support different ViT variants
            vit_model_name = backbone_name if backbone_name in ['vit_base_patch16_224', 'vit_small_patch16_224', 'vit_tiny_patch16_224'] else 'vit_base_patch16_224'
            
            # Create ViT model without classification head, get all patch tokens
            backbone = timm.create_model(
                vit_model_name,
                pretrained=pretrained,
                num_classes=0,  # Remove classification head
                global_pool=''  # Return all tokens, not pooled
            )
            
            # Get feature dimension by doing a forward pass with dummy input
            # ViT outputs shape: (batch, num_patches + 1, embed_dim) where +1 is CLS token
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, 224, 224)
                try:
                    dummy_output = backbone(dummy_input)
                    # Handle different output formats
                    if isinstance(dummy_output, tuple):
                        dummy_output = dummy_output[0]
                    if dummy_output.dim() == 3:
                        # Shape: (batch, num_patches+1, embed_dim)
                        in_features = dummy_output.shape[-1]
                    elif dummy_output.dim() == 2:
                        # Already pooled (shouldn't happen with global_pool='')
                        in_features = dummy_output.shape[-1]
                    else:
                        raise ValueError(f"Unexpected ViT output shape: {dummy_output.shape}")
                except Exception as e:
                    # Fallback: use typical ViT embedding dimensions
                    if 'base' in vit_model_name:
                        in_features = 768
                    elif 'small' in vit_model_name:
                        in_features = 384
                    elif 'tiny' in vit_model_name:
                        in_features = 192
                    else:
                        in_features = 768  # Default to base
                    print(f"Warning: Could not infer ViT feature dim, using {in_features}. Error: {e}")
            
            self.backbone = backbone
            
        else:
            raise ValueError(f'Unsupported backbone: {backbone_name}. Supported: resnet50, vit_base_patch16_224, vit_small_patch16_224, vit_tiny_patch16_224')

        # main classification head (DR grade)
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

        # auxiliary binary head (e.g., referable DR)
        self.aux_head = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, aux_output)
        )

    def forward(self, x):
        feat = self.backbone(x)
        
        # Handle ViT output (which is a tensor of shape [batch, num_patches+1, embed_dim])
        if self.backbone_name.startswith('vit'):
            if isinstance(feat, tuple):
                feat = feat[0]
            # For ViT, the first token is the CLS token, or we can average pool all tokens
            if feat.dim() == 3:
                # Use CLS token (first token) - typically better than average pooling
                feat = feat[:, 0, :]  # [batch, embed_dim]
        
        out_main = self.classifier(feat)
        out_aux = self.aux_head(feat)
        return out_main, out_aux


# Backward compatibility alias
MultiTaskResNet = MultiTaskModel
