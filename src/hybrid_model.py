import torch
import torch.nn as nn
from torchvision import models


class CNNFeatureExtractor(nn.Module):
    """CNN backbone that returns a 1D feature vector per image."""
    def __init__(self, backbone_name='resnet50', pretrained=True):
        super().__init__()
        if backbone_name == 'resnet50':
            backbone = models.resnet50(pretrained=pretrained)
            # remove final fc
            modules = list(backbone.children())[:-1]
            self.cnn = nn.Sequential(*modules)
            self.out_features = backbone.fc.in_features
        else:
            raise ValueError('Unsupported backbone')

    def forward(self, x):
        # x: [B, 3, H, W]
        f = self.cnn(x)
        f = f.view(f.size(0), -1)
        return f


class HybridCNNLSTM(nn.Module):
    """Hybrid CNN-LSTM model that accepts single images or sequences.

    If input is a single image tensor [B, 3, H, W], we extract features and
    treat them as a sequence of length 1 for the LSTM. For a true sequence of
    images, pass tensors shaped [B, S, 3, H, W] and set `is_sequence=True` in
    the forward call.
    """
    def __init__(self, backbone_name='resnet50', lstm_hidden_size=512, num_layers=1, num_classes=5, pretrained=True):
        super().__init__()
        self.cnn = CNNFeatureExtractor(backbone_name=backbone_name, pretrained=pretrained)
        self.lstm_input_size = self.cnn.out_features
        self.lstm = nn.LSTM(input_size=self.lstm_input_size, hidden_size=lstm_hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=0.0)
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x, is_sequence=False):
        """Forward accepts either:
        - x: [B, 3, H, W] (single image per sample)
        - x: [B, S, 3, H, W] (sequence of S images per sample)
        Set is_sequence=True when passing sequences.
        """
        if is_sequence:
            # collapse batch and sequence dims to run through CNN efficiently
            B, S, C, H, W = x.shape
            x = x.view(B * S, C, H, W)
            feats = self.cnn(x)
            feats = feats.view(B, S, -1)  # [B, S, F]
        else:
            # single image -> extract features and add seq dim = 1
            feats = self.cnn(x)  # [B, F]
            feats = feats.unsqueeze(1)  # [B, 1, F]

        lstm_out, (h_n, c_n) = self.lstm(feats)
        # use last hidden state
        out = self.fc(h_n[-1])
        return out
