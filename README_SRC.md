README: src/ package — Multitasking DR project

This document explains the `src/` package in this repository, the training contract, data shapes, how to run the training (locally and on Colab), and practical next steps and experiments.

Purpose
-------
The project is a multitask image model for Diabetic Retinopathy (DR) that trains a ResNet backbone with two heads:
- Main head: 5-way classification (DR grades 0..4)
- Auxiliary head: binary classification (e.g., referable DR = label >= 2)

The implementation lives in `src/`. The Colab notebook in the repo uses these same modules so behavior is identical when run in Colab.

Files in `src/`
----------------
- `config.py`
  - Contains a dataclass `Config` with defaults (img_size, batch_size, epochs, lr, model_name, num_classes, seed).
  - Use `from src.config import cfg` to read defaults.

- `data_loader.py`
  - `FundusDataset(Dataset)`: reads a CSV and returns (image_tensor, label_tensor).
    - Accepts CSV columns `id` or `id_code` and `label` or `Label`.
    - Robust image path handling: supports ids with/without extension and `.jpg`/`.jpeg`/`.png`.
    - Raises informative errors for missing columns or missing image files.
    - Applies torchvision transforms (resize, flip, to-tensor, normalize).
  - `get_loaders(data_dir, batch_size, img_size, num_workers)`: convenience loader for a single dataset.
  - `get_train_val_loaders(..., val_split=0.2)`: stratified split (sklearn), creates temp CSVs for train/val, returns (train_loader, val_loader).

- `models.py`
  - `MultiTaskResNet(backbone_name='resnet50', num_classes=5, aux_output=1, pretrained=True)`:
    - Loads a pretrained ResNet backbone (currently resnet50).
    - Strips final fc (replaces with Identity) and adds two heads:
      - `classifier`: Linear -> ReLU -> Dropout -> Linear(num_classes)
      - `aux_head`: Linear -> ReLU -> Dropout -> Linear(aux_output)
    - `forward(x)` returns `(out_main_logits, out_aux_logits)`.

- `train.py`
  - Entrypoint `train(args)` that:
    - Sets random seeds
    - Builds train/val loaders via `get_train_val_loaders`
    - Instantiates `MultiTaskResNet` and moves to `device` (cuda if available)
    - Uses loss functions:
      - Main: `nn.CrossEntropyLoss()`
      - Aux: `nn.BCEWithLogitsLoss()`
    - Uses `optim.Adam` optimizer
    - Training loop:
      - For each batch: compute `loss_main`, `aux_labels = (labels >= 2).float().unsqueeze(1)`, `loss_aux`, `loss = loss_main + 0.5 * loss_aux`.
      - After each epoch run validation and compute accuracy and macro-F1 (sklearn.metrics).
      - Saves best checkpoint in `checkpoints/best_epoch_{epoch}.pt` (dict with `model_state`, `epoch`, `val_f1`).
  - CLI (`if __name__ == '__main__'`): standard argparse for `--data-dir`, `--epochs`, `--batch-size`, `--img-size`, `--lr`, `--num-workers`.

Data shapes & preprocessing
---------------------------
- CSV: expects rows with `id` (or `id_code`) and `label` (0..4). Labels are integers.
- Image input: PIL image -> transforms -> torch.Tensor shape `[3, img_size, img_size]`.
- Normalization uses ImageNet mean/std: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225].
- Model outputs:
  - Main logits: shape `[batch, num_classes]` (use `torch.argmax` for predicted class)
  - Aux logits: shape `[batch, 1]` (apply `sigmoid` to convert to probability if needed)

Training contract (inputs → outputs)
------------------------------------
- Inputs: `data_dir` containing `dr_labels.csv` and `DR_images/`.
- Command-line args (or notebook Namespace) supply training hyperparameters.
- Outputs:
  - Console logs and tqdm progress bars
  - Saved checkpoint files in `checkpoints/` (best model by val macro-F1)
  - Printed validation metrics: accuracy and macro-F1 per epoch

Metrics
-------
- Accuracy: fraction of correct main-class predictions (simple global metric).
- Macro-F1: average F1 across classes (1..K), treats each class equally — preferred when classes are imbalanced.
- Auxiliary metric: aux head uses BCEWithLogitsLoss; if you need to evaluate it convert logits via sigmoid and compute roc_auc or F1.

Common edge cases and failures
-----------------------------
- CSV column mismatch → `FundusDataset` will raise ValueError.
- Missing images → `FundusDataset` will raise FileNotFoundError listing candidate paths it tried.
- Small datasets: training ResNet-50 on a few hundred images can lead to overfitting or weak generalization.
- Class imbalance: accuracy may be misleading; prefer macro-F1 and per-class confusion matrices.
- Torchvision deprecation warning: the code uses `pretrained=True`. This still works but emits a warning — recommended to update `models.py` to the `weights=` API.

Quick run examples
------------------
- Colab (notebook `colab_train_fixed.ipynb` handles cloning, installing, Drive mounting):
  - Set runtime to GPU, run the top cell, then run the training cell. Notebook auto-sets `DATA_DIR` to your Drive `DR dataset` folder by default.

- Local (PowerShell example):
```powershell
# create venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# install deps (adjust torch wheel for CUDA if needed)
pip install -r requirements_colab.txt
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu117

# run training
python -m src.train --data-dir data --epochs 20 --batch-size 16
```

Recommended quick experiments (order matters)
--------------------------------------------
1. Increase epochs (e.g., 20–50). Many pretrained models fine-tune over more epochs.
2. Compute class weights from `dr_labels.csv` and pass to `nn.CrossEntropyLoss(weight=weights_tensor)`. This helps imbalance.
3. Add a learning-rate scheduler, for example ReduceLROnPlateau or CosineAnnealingLR.
4. Improve augmentations: RandomResizedCrop, ColorJitter, small rotations.
5. Freeze backbone for first few epochs (train only heads) then unfreeze with a smaller LR.
6. Try smaller or lighter architectures if overfitting (EfficientNet, MobileNet, or smaller ResNet).
7. Use `torch.cuda.amp` for mixed-precision training if running on GPUs with limited memory.

Small code snippets (where to change)
------------------------------------
- Use class weights in `train.py` (conceptual):
```python
# compute weights (example)
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
labels = train_df['label'].values
class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
weights = torch.tensor(class_weights, dtype=torch.float).to(device)
criterion_main = nn.CrossEntropyLoss(weight=weights)
```

- Replace deprecated pretrained API in `models.py`:
```python
from torchvision.models import resnet50, ResNet50_Weights
backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
```

- Add a scheduler snippet in `train.py`:
```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
# after validation: scheduler.step(val_f1)
```

Next steps I can implement for you (pick any)
---------------------------------------------
- Update `models.py` to the new `weights=` API to silence warnings.
- Add class-weight computation and use in `CrossEntropyLoss`.
- Add a learning-rate scheduler and integrate stepping with validation macro-F1.
- Add TensorBoard logging for loss and per-class metrics.
- Add an `infer.py` script for running model inference on a folder of images.

If you want, tell me which three changes to implement first and I will make them, run quick local syntax checks, and create a commit. If you'd rather I just produce patches/snippets you can paste into the code, tell me which ones and I'll provide them.

*** End of README_SRC.md