Colab usage for Multitasking-learning-multimodal-network-for-Diabetic-Retinopathy

1) Open this repository in Google Colab or upload this folder to your Google Drive.

If your project is on GitHub (recommended): https://github.com/Ojasvsakhi/Diabetic-Retinopathy

You can open the notebook directly from GitHub in Colab or let the notebook auto-clone the repo into the Colab session.

2) In Colab: set Runtime -> Change runtime type -> GPU

3) Install requirements and mount drive (if needed):

```python
!pip install -r requirements_colab.txt
from google.colab import drive
drive.mount('/content/drive')
```

4) Ensure your dataset folder contains `dr_labels.csv` and a `DR_images/` folder. You can place the data in `/content/data` (upload) or on Drive (e.g. `/content/drive/MyDrive/datasets/DR`).

5) Run `colab_train.ipynb` and set the `data_dir` argument in the notebook to point to your data location.

Notes:
- The notebook uses the project's existing `src/train.py` entrypoint, so training behavior remains the same.
- If your repository is hosted on GitHub, clone it in a cell before running the notebook: `!git clone <url>` and `cd` into the repo.
