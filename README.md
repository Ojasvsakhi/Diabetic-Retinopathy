# Multitasking Learning Multimodal Network for Diabetic Retinopathy

Lightweight PyTorch project that trains a multitask ResNet model for diabetic retinopathy grading (main multi-class head) and an auxiliary binary head (e.g. referable DR). The repository includes a Colab notebook to run training on a GPU and scripts to load data and run training locally.

## Repo structure
- `src/` — core Python package
	- `config.py` — configuration dataclass
	- `data_loader.py` — `FundusDataset` and loader helpers
	- `models.py` — `MultiTaskResNet` model definition
	- `train.py` — training loop and validation (entrypoint)
- `colab_train_fixed.ipynb` — Colab notebook to run training on Google Colab (GPU)
- `requirements_colab.txt` — non-torch Python dependencies

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
2. Open Colab: File → Open notebook → GitHub → paste `https://github.com/Ojasvsakhi/Diabetic-Retinopathy` and open `colab_train_fixed.ipynb`.
3. Runtime → Change runtime type → Hardware accelerator: GPU.
4. Run the top cell to auto-clone and mount Drive. Confirm the notebook finds `DATA_DIR` (the notebook prints the path it will use).
5. Run the install cell to install PyTorch (CUDA wheel) and other dependencies.
6. Run the final cell to start training. Best checkpoints will be saved under `checkpoints/` (the notebook attempts to persist them to Drive).

## Quickstart — Local (Windows PowerShell)
1. (Optional) Create and activate a virtual environment:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
2. Install dependencies and PyTorch (CPU or appropriate CUDA wheel):
```powershell
pip install -r requirements_colab.txt
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu117
```
3. Prepare your dataset under `data/` or pass a different path.
4. Run training:
```powershell
python -m src.train --data-dir data --epochs 5 --batch-size 16
```

## Checkpoints & persistence
- `src/train.py` saves best checkpoints into the `checkpoints/` folder by default.
- When using Colab, the notebook attempts to create a Drive-backed checkpoints folder so trained models persist between sessions.

## Troubleshooting
- JSON parse error opening the notebook in Colab: open `colab_train_fixed.ipynb` (not the original) or re-upload the fixed file.
- `getcwd`/clone errors: the notebook forces the working directory to `/content` before cloning; run the top cell first.
- Dataset not found: check the printed Drive root listing and set `DATA_DIR` in the notebook to the correct path.

## Next improvements (suggested)
- Add `infer.py` to load a checkpoint and run inference on a folder of images.
- Add TensorBoard or Weights & Biases logging.
- Add a dataset sanity-check cell to the notebook that prints CSV samples and shows a few images before training.

## License
Add a `LICENSE` file at the repo root with your preferred license.

If you want I can commit this `README.md` to your GitHub repo for you. Tell me if you want any edits or additions.
