from dataclasses import dataclass

@dataclass
class Config:
    img_size: int = 224
    batch_size: int = 16
    epochs: int = 5
    lr: float = 1e-4
    num_workers: int = 4
    model_name: str = 'resnet50'
    num_classes: int = 5  # DR grades 0-4
    seed: int = 42

cfg = Config()
