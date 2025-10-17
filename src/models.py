import torch
import torch.nn as nn
import torchvision.models as models


class MultiTaskResNet(nn.Module):
    def __init__(self, backbone_name='resnet50', num_classes=5, aux_output=1, pretrained=True):
        super().__init__()
        if backbone_name == 'resnet50':
            backbone = models.resnet50(pretrained=pretrained)
            in_features = backbone.fc.in_features
            backbone.fc = nn.Identity()
        else:
            raise ValueError('Unsupported backbone')

        self.backbone = backbone
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
        out_main = self.classifier(feat)
        out_aux = self.aux_head(feat)
        return out_main, out_aux
