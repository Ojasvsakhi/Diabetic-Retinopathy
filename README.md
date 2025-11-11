# Multitasking Learning Multimodal Network for Diabetic Retinopathy

Lightweight PyTorch project that trains a multitask model for diabetic retinopathy grading (main multi-class head) and an auxiliary binary head (e.g. referable DR). The repository supports both **ResNet50** and **Vision Transformer (ViT)** architectures for improved accuracy. The repository includes a Colab notebook to run training on a GPU and scripts to load data and run training locally.

**Recent Improvements:**
- ✅ Added Vision Transformer (ViT) support with multiple variants (Base, Small, Tiny)
- ✅ Enhanced data augmentation (RandomResizedCrop, ColorJitter, Rotation, Vertical Flip)
- ✅ Fixed deprecated PyTorch warnings
- ✅ Improved model selection and configuration
- ✅ Better error handling and reproducibility

For more detailed information about the project implementation, architecture details, and technical specifications, please refer to [README_SRC.md](README_SRC.md).

## Dataset

The project uses the Diabetic Retinopathy dataset which consists of high-resolution retina images taken under varying imaging conditions. 

Dataset access: [Google Drive Link](https://drive.google.com/drive/folders/1CvS_HmNtJ3U3mREmA9kalnS5w2rXTyk9)

The dataset contains:
- Retinal fundus photographs labeled with DR grade (0-4)
- Images are in high resolution
- Labels are provided in CSV format with image IDs and corresponding DR grades

## Features & Improvements

### Model Architectures Supported
- **ResNet50**: Baseline CNN architecture (original implementation)
- **Vision Transformer Base** (`vit_base_patch16_224`): Recommended for best accuracy
- **Vision Transformer Small** (`vit_small_patch16_224`): Balanced performance and speed
- **Vision Transformer Tiny** (`vit_tiny_patch16_224`): Fastest option with lower memory usage

### Enhanced Data Augmentation
- RandomResizedCrop for better generalization
- RandomHorizontalFlip and RandomVerticalFlip
- ColorJitter (brightness, contrast, saturation, hue variations)
- RandomRotation for rotation invariance
- All augmentations applied during training only (validation uses minimal transforms)

### Training Features
- Automatic class weight computation for handling imbalanced datasets
- Learning rate scheduler (ReduceLROnPlateau) for adaptive learning
- Model checkpointing (saves best model based on validation macro-F1)
- Reproducible training with seed setting

## Project Limitations

1. **Single Dataset Dependency**: 
   - The current implementation only supports one type of dataset format (CSV with specific columns 'id'/'id_code' and 'label'/'Label')
   - Limited to single-source data, no multi-source data fusion capability

2. **Model Architecture**:
   - ~~Uses only ResNet50 as backbone~~ ✅ **FIXED**: Now supports ResNet50 and multiple ViT variants
   - Fixed auxiliary task (binary classification for referable DR)
   - No support for ensemble models

3. **Training Limitations**:
   - ~~Basic data augmentation (only horizontal flips)~~ ✅ **FIXED**: Enhanced augmentation pipeline implemented
   - ~~No support for advanced augmentation techniques~~ ✅ **FIXED**: Multiple augmentation techniques added
   - ~~Fixed learning rate without scheduling~~ ✅ **FIXED**: ReduceLROnPlateau scheduler implemented
   - No cross-validation implementation

4. **Evaluation Metrics**:
   - Limited to accuracy and macro F1-score
   - No support for medical-specific metrics like sensitivity/specificity
   - No confusion matrix visualization

5. **Clinical Integration**:
   - No interface for real-time predictions
   - Lacks deployment-ready features
   - No DICOM format support

## Repo structure
- `src/` — core Python package
	- `config.py` — configuration dataclass with model selection options
	- `data_loader.py` — `FundusDataset` and loader helpers with enhanced augmentation
	- `models.py` — `MultiTaskModel` supporting ResNet50 and ViT architectures
	- `train.py` — training loop and validation (entrypoint) with class weights and LR scheduling
- `notebook.ipynb` — Colab notebook to run training on Google Colab (GPU) with model selection
- `requirements_colab.txt` — Python dependencies including `timm` for Vision Transformers

## Dataset layout (required)
Place your dataset folder so `data_dir` points to a folder with:
- `dr_labels.csv` — CSV with columns `id` (or `id_code`) and `label` (or `Label`)
- `DR_images/` — folder containing image files referenced by the CSV

Example:
```
data/
	dr_labels.csv
	DR_images/
		0001.jpg
		0002.jpg
		...
```

If using Google Drive, the Colab notebook expects the dataset at `/content/drive/MyDrive/DR dataset` by default. You can change `DATA_DIR` inside the notebook if your folder is named differently.

## Quickstart — Google Colab (recommended)
1. Push the repository to GitHub (if not already).
2. Open Colab: File → Open notebook → GitHub → paste `https://github.com/Ojasvsakhi/Diabetic-Retinopathy` and open `notebook.ipynb`.
3. Runtime → Change runtime type → Hardware accelerator: GPU.
4. Run the top cell to auto-clone and mount Drive. Confirm the notebook finds `DATA_DIR` (the notebook prints the path it will use).
5. Run the install cell to install PyTorch (CUDA wheel) and other dependencies (including `timm` for ViT).
6. **Configure your model** in the training cell:
   - Set `MODEL_NAME = 'vit_base_patch16_224'` for Vision Transformer (recommended)
   - Or use `MODEL_NAME = 'resnet50'` for ResNet50 baseline
   - Adjust `BATCH_SIZE` if needed (ViT models use more memory)
7. Run the final cell to start training. Best checkpoints will be saved under `checkpoints/` (the notebook attempts to persist them to Drive).

## Quickstart — Local (Windows PowerShell)
1. (Optional) Create and activate a virtual environment:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
2. Install dependencies and PyTorch (CPU or appropriate CUDA wheel):
```powershell
pip install -r requirements_colab.txt
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118
```
3. Prepare your dataset under `data/` or pass a different path.
4. Run training with model selection:
```powershell
# Using Vision Transformer (recommended for better accuracy)
python -m src.train --data-dir data --epochs 20 --batch-size 16 --model-name vit_base_patch16_224

# Using ResNet50 (baseline)
python -m src.train --data-dir data --epochs 20 --batch-size 16 --model-name resnet50

# Using smaller ViT variant (if memory constrained)
python -m src.train --data-dir data --epochs 20 --batch-size 16 --model-name vit_small_patch16_224
```

**Note**: Vision Transformer models typically achieve better accuracy than ResNet50 but require more GPU memory. Reduce batch size if you encounter out-of-memory errors.

