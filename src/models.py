import torch
import torch.nn as nn
import torchvision.models as models
try:
    from torchvision.models import ResNet50_Weights
    RESNET50_WEIGHTS_AVAILABLE = True
except ImportError:
    RESNET50_WEIGHTS_AVAILABLE = False
try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("Warning: timm not available. Vision Transformer models will not work. Install with: pip install timm")


class MultiTaskModel(nn.Module):
    """Multi-task model supporting ResNet, Vision Transformer, and hybrid CNN+LSTM backbones."""
    def __init__(self, backbone_name='resnet50', num_classes=5, aux_output=1, pretrained=True):
        super().__init__()
        self.backbone_name = backbone_name
        self.use_lstm = False
        
        if backbone_name == 'resnet50':
            # Use weights API when available to avoid deprecated pretrained= parameter
            if RESNET50_WEIGHTS_AVAILABLE:
                weights = ResNet50_Weights.DEFAULT if pretrained else None
                backbone = models.resnet50(weights=weights)
            else:
                backbone = models.resnet50(pretrained=pretrained)
            in_features = backbone.fc.in_features
            backbone.fc = nn.Identity()
            self.backbone = backbone
            
        elif backbone_name in ('hybrid_cnn_lstm', 'resnet50_lstm', 'cnn_lstm'):
            # Hybrid CNN (ResNet50) + LSTM sequence model
            if RESNET50_WEIGHTS_AVAILABLE:
                weights = ResNet50_Weights.DEFAULT if pretrained else None
                full_backbone = models.resnet50(weights=weights)
            else:
                full_backbone = models.resnet50(pretrained=pretrained)
            in_features_backbone = full_backbone.fc.in_features  # 2048 for ResNet50
            # Remove avgpool and fc to preserve spatial grid (output: B, 2048, 7, 7)
            self.backbone = nn.Sequential(*list(full_backbone.children())[:-2])
            
            self.use_lstm = True
            self.lstm_hidden = 512
            self.lstm_layers = 1
            self.lstm_bidirectional = True
            lstm_dropout = 0.0 if self.lstm_layers == 1 else 0.3
            self.lstm = nn.LSTM(
                input_size=in_features_backbone,
                hidden_size=self.lstm_hidden,
                num_layers=self.lstm_layers,
                batch_first=True,
                bidirectional=self.lstm_bidirectional,
                dropout=lstm_dropout
            )
            in_features = self.lstm_hidden * (2 if self.lstm_bidirectional else 1)
            
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
            raise ValueError(
                f"Unsupported backbone: {backbone_name}. Supported: resnet50, vit_base_patch16_224, "
                "vit_small_patch16_224, vit_tiny_patch16_224, hybrid_cnn_lstm, resnet50_lstm, cnn_lstm"
            )

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
        if self.use_lstm:
            if feat.dim() == 4:
                b, c, h, w = feat.size()
                seq = feat.view(b, c, h * w).permute(0, 2, 1)  # [batch, seq_len, features]
            elif feat.dim() == 3:
                # Already sequence-like (batch, seq_len, features)
                seq = feat
            else:
                seq = feat.unsqueeze(1)
            
            lstm_out, _ = self.lstm(seq)
            # Use the last time-step (equivalent to sequence summary); LSTM is bidirectional so last contains both directions
            feat = lstm_out[:, -1, :]
        
        elif self.backbone_name.startswith('vit'):
            if isinstance(feat, tuple):
                feat = feat[0]
            # For ViT, try both CLS token and average pooling, then combine
            if feat.dim() == 3:
                # CLS token (first token) - contains global information
                cls_token = feat[:, 0, :]  # [batch, embed_dim]
                # Average pool all tokens (including CLS) - contains spatial information
                avg_pooled = feat.mean(dim=1)  # [batch, embed_dim]
                # Combine both for richer representation
                feat = (cls_token + avg_pooled) / 2.0  # Average of CLS and spatial features
        
        out_main = self.classifier(feat)
        out_aux = self.aux_head(feat)
        return out_main, out_aux


# Backward compatibility alias
MultiTaskResNet = MultiTaskModel
