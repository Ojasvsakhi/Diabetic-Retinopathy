from dataclasses import dataclass

@dataclass
class Config:
    img_size: int = 224
    batch_size: int = 16
    epochs: int = 5
    lr: float = 1e-4
    num_workers: int = 4
    model_name: str = 'resnet50'  # Options: 'resnet50', 'vit_base_patch16_224', 'vit_small_patch16_224', 'vit_tiny_patch16_224', 'hybrid_cnn_lstm'
    num_classes: int = 5  # DR grades 0-4
    seed: int = 42

cfg = Config()
