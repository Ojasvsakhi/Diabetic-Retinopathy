README: src/ package — Multitasking DR project

This document explains the `src/` package in this repository, the training contract, data shapes, how to run the training (locally and on Colab), and practical next steps and experiments.

**Recent Updates (Latest Version):**
- ✅ Added Vision Transformer (ViT) support via `timm` library
- ✅ Enhanced data augmentation pipeline
- ✅ Fixed deprecated PyTorch warnings
- ✅ Improved model selection and configuration
- ✅ Better reproducibility with enhanced seed setting

Purpose
-------
The project is a multitask image model for Diabetic Retinopathy (DR) that trains either a **ResNet50** or **Vision Transformer (ViT)** backbone with two heads:
- Main head: 5-way classification (DR grades 0..4)
- Auxiliary head: binary classification (e.g., referable DR = label >= 2)

The implementation lives in `src/`. The Colab notebook in the repo uses these same modules so behavior is identical when run in Colab.

Files in `src/`
----------------
- `config.py`
  - Contains a dataclass `Config` with defaults (img_size, batch_size, epochs, lr, model_name, num_classes, seed).
  - `model_name` supports: `'resnet50'`, `'vit_base_patch16_224'`, `'vit_small_patch16_224'`, `'vit_tiny_patch16_224'`.
  - Use `from src.config import cfg` to read defaults.

- `data_loader.py`
  - `FundusDataset(Dataset)`: reads a CSV and returns (image_tensor, label_tensor).
    - Accepts CSV columns `id` or `id_code` and `label` or `Label`.
    - Robust image path handling: supports ids with/without extension and `.jpg`/`.jpeg`/`.png`.
    - Raises informative errors for missing columns or missing image files.
    - **Enhanced transforms**: Training uses RandomResizedCrop, RandomHorizontalFlip, RandomVerticalFlip, ColorJitter, RandomRotation. Validation uses minimal transforms (Resize, ToTensor, Normalize).
  - `get_loaders(data_dir, batch_size, img_size, num_workers)`: convenience loader for a single dataset.
  - `get_train_val_loaders(..., val_split=0.2)`: stratified split (sklearn), creates temp CSVs for train/val, returns (train_loader, val_loader).

- `models.py`
  - `MultiTaskModel(backbone_name='resnet50', num_classes=5, aux_output=1, pretrained=True)`:
    - **Supports multiple backbones**:
      - `'resnet50'`: ResNet50 CNN backbone (pretrained head removed; avgpool retained).
      - `'vit_base_patch16_224'`: Vision Transformer Base (768-dim embeddings).
      - `'vit_small_patch16_224'`: Vision Transformer Small (384-dim embeddings).
      - `'vit_tiny_patch16_224'`: Vision Transformer Tiny (192-dim embeddings).
      - `'hybrid_cnn_lstm'` / `'resnet50_lstm'`: Hybrid model combining ResNet50 feature maps with a bidirectional LSTM (512 hidden units) to model spatial sequences (inspired by reported 87.5% accuracy in literature).
    - For ResNet: Loads pretrained backbone, strips final fc (replaces with Identity).
    - For hybrid CNN+LSTM: ResNet50 provides a grid of features (avgpool removed), reshaped into a sequence and passed through the LSTM; combined output feeds the classification heads.
    - For ViT: Uses `timm.create_model()` with `num_classes=0` to get feature tokens, combining CLS token and average-pooled tokens.
    - Adds two heads:
      - `classifier`: Linear -> ReLU -> Dropout -> Linear(num_classes)
      - `aux_head`: Linear -> ReLU -> Dropout -> Linear(aux_output)
    - `forward(x)` returns `(out_main_logits, out_aux_logits)`.
  - Backward compatibility: `MultiTaskResNet` is an alias for `MultiTaskModel`.

- `train.py`
  - Entrypoint `train(args)` that:
    - Sets random seeds (with CUDA deterministic mode for reproducibility)
    - Builds train/val loaders via `get_train_val_loaders`
    - Instantiates `MultiTaskModel` with selected backbone and moves to `device` (cuda if available)
    - **Automatic class weight computation** from training data for handling imbalanced classes
    - Uses loss functions:
      - Main: `nn.CrossEntropyLoss(weight=class_weights)` (weighted if classes imbalanced)
      - Aux: `nn.BCEWithLogitsLoss()`
    - Uses `optim.Adam` optimizer
    - **Learning rate scheduler**: `ReduceLROnPlateau` (reduces LR when validation F1 plateaus)
    - Training loop:
      - For each batch: compute `loss_main`, `aux_labels = (labels >= 2).float().unsqueeze(1)`, `loss_aux`, `loss = loss_main + 0.5 * loss_aux`.
      - After each epoch run validation and compute accuracy and macro-F1 (sklearn.metrics).
      - Saves best checkpoint in `checkpoints/best_epoch_{epoch}.pt` (dict with `model_state`, `epoch`, `val_f1`).
      - Steps LR scheduler based on validation macro-F1.
  - CLI (`if __name__ == '__main__'`): standard argparse for `--data-dir`, `--epochs`, `--batch-size`, `--img-size`, `--lr`, `--num-workers`, `--model-name`.

Data shapes & preprocessing
---------------------------
- CSV: expects rows with `id` (or `id_code`) and `label` (0..4). Labels are integers.
- Image input: PIL image -> transforms -> torch.Tensor shape `[3, img_size, img_size]`.
- **Training transforms** (enhanced):
  - Resize to (img_size+32, img_size+32)
  - RandomResizedCrop(img_size, scale=(0.8, 1.0), ratio=(0.9, 1.1))
  - RandomHorizontalFlip(p=0.5)
  - RandomVerticalFlip(p=0.3)
  - ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
  - RandomRotation(degrees=15)
  - ToTensor()
  - Normalize with ImageNet mean/std: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
- **Validation transforms** (minimal):
  - Resize to (img_size, img_size)
  - ToTensor()
  - Normalize with ImageNet mean/std
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
- Small datasets: training on a few hundred images can lead to overfitting or weak generalization.
- Class imbalance: ✅ **FIXED** - Automatic class weight computation now handles this. Accuracy may still be misleading; prefer macro-F1 and per-class confusion matrices.
- ~~Torchvision deprecation warning~~ ✅ **FIXED** - Updated to use `weights=` API for ResNet50.
- ViT memory issues: If running out of memory with Vision Transformer, reduce batch size or use `vit_small_patch16_224` or `vit_tiny_patch16_224`.
- Missing `timm` library: Install with `pip install timm` (already in requirements_colab.txt).

Quick run examples
------------------
- Colab (notebook `notebook.ipynb` handles cloning, installing, Drive mounting):
  - Set runtime to GPU, run the top cell, then run the training cell. Notebook auto-sets `DATA_DIR` to your Drive `DR dataset` folder by default.
  - **Configure model**: Edit `MODEL_NAME` variable in the training cell (options: `'resnet50'`, `'vit_base_patch16_224'`, `'vit_small_patch16_224'`, `'vit_tiny_patch16_224'`).

- Local (PowerShell example):
```powershell
# create venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# install deps (adjust torch wheel for CUDA if needed)
pip install -r requirements_colab.txt
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118

# run training with Vision Transformer (recommended)
python -m src.train --data-dir data --epochs 20 --batch-size 16 --model-name vit_base_patch16_224

# or with ResNet50
python -m src.train --data-dir data --epochs 20 --batch-size 16 --model-name resnet50

# or with hybrid CNN + LSTM
python -m src.train --data-dir data --epochs 20 --batch-size 16 --model-name hybrid_cnn_lstm
```

Implemented improvements (✅) and remaining recommendations
------------------------------------------------------------
✅ **1. Increased epochs support**: Training supports configurable epochs (default 5, recommend 20-50).

✅ **2. Class weights**: Automatic computation and application of class weights in `CrossEntropyLoss` to handle imbalanced datasets.

✅ **3. Learning rate scheduler**: `ReduceLROnPlateau` scheduler implemented, reduces LR when validation F1 plateaus.

✅ **4. Enhanced augmentations**: RandomResizedCrop, ColorJitter, RandomRotation, RandomVerticalFlip all implemented.

✅ **5. Multiple architectures**: Now supports ResNet50 and Vision Transformer variants (Base, Small, Tiny).

**Remaining recommendations for future work:**
- Freeze backbone for first few epochs (train only heads) then unfreeze with a smaller LR.
- Use `torch.cuda.amp` for mixed-precision training if running on GPUs with limited memory.
- Add TensorBoard logging for loss and per-class metrics visualization.
- Implement cross-validation for more robust evaluation.

Summary of Implemented Features
--------------------------------
All the following improvements have been **implemented and tested**:

✅ **Class weights**: Automatically computed from training data and applied in `CrossEntropyLoss` (see `train.py` lines 36-66).

✅ **Updated pretrained API**: ResNet50 now uses `weights=ResNet50_Weights.IMAGENET1K_V1` instead of deprecated `pretrained=True` (see `models.py` lines 20-26).

✅ **Learning rate scheduler**: `ReduceLROnPlateau` implemented and integrated with validation macro-F1 (see `train.py` lines 69-128).

✅ **Vision Transformer support**: Multiple ViT variants available via `timm` library (see `models.py` lines 31-75).

✅ **Enhanced data augmentation**: Comprehensive augmentation pipeline implemented (see `data_loader.py` lines 83-92 and 120-129).

Future enhancements (not yet implemented)
------------------------------------------
- Add TensorBoard logging for loss and per-class metrics visualization.
- Add an `infer.py` script for running model inference on a folder of images.
- Implement progressive training (freeze backbone initially, then unfreeze).
- Add mixed-precision training support (`torch.cuda.amp`).
- Implement cross-validation for more robust evaluation.

*** End of README_SRC.md