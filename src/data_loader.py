import os
from typing import Optional

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class FundusDataset(Dataset):
    """PyTorch Dataset for fundus images.

    This loader is robust to the CSV formats commonly used in DR datasets. It will
    accept either (id or id_code) and (label or Label) column names. It also
    handles ids that already include the file extension (e.g. "1.jpg") or ids
    without extension (e.g. "1").
    """

    def __init__(self, csv_path, images_dir, transform: Optional[transforms.Compose] = None):
        self.df = pd.read_csv(csv_path)
        # normalize column names to lowercase
        self.df.columns = [c.strip() for c in self.df.columns]
        # possible name mapping
        if 'id_code' in self.df.columns:
            self.df.rename(columns={'id_code': 'id'}, inplace=True)
        if 'Label' in self.df.columns:
            self.df.rename(columns={'Label': 'label'}, inplace=True)

        if 'id' not in self.df.columns or 'label' not in self.df.columns:
            raise ValueError("CSV must contain 'id' (or 'id_code') and 'label' (or 'Label') columns")

        self.images_dir = images_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = str(row['id']).strip()
        label = int(row['label'])

        # if id already contains an extension, try it directly
        candidates = []
        base, ext = os.path.splitext(img_id)
        if ext.lower() in ('.jpg', '.jpeg', '.png'):
            candidates.append(os.path.join(self.images_dir, img_id))
        else:
            # try with common extensions
            for e in ('.jpg', '.jpeg', '.png'):
                candidates.append(os.path.join(self.images_dir, img_id + e))

        # also try if the CSV uses just a numeric id but the images are named like '1.jpg'
        # (already covered above), and finally try using img_id as-is
        candidates.append(os.path.join(self.images_dir, img_id))

        img_path = None
        for p in candidates:
            if os.path.exists(p):
                img_path = p
                break

        if img_path is None:
            raise FileNotFoundError(f'Image for id {img_id} not found in {self.images_dir}; tried: {candidates[:5]}')

        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.long)


def get_loaders(data_dir, batch_size=16, img_size=224, num_workers=4):
    """Create a single DataLoader for the whole CSV. Use `data_dir='data'` for
    your project layout where the CSV is `data/dr_labels.csv` and images live in
    `data/DR_images`.
    """
    csv_path = os.path.join(data_dir, 'dr_labels.csv')
    # default images folder in your workspace is 'DR_images'
    images_dir = os.path.join(data_dir, 'DR_images')

    train_transform = transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),  # Slightly larger for crop
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),  # Fundus images can be flipped
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    ds = FundusDataset(csv_path, images_dir, transform=train_transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return loader


def get_train_val_loaders(csv_path=None, images_dir=None, data_dir=None, batch_size=16, img_size=224, num_workers=4, val_split=0.2, random_state=42):
    """Return (train_loader, val_loader).

    You can either provide csv_path and images_dir explicitly or pass data_dir (defaults to data/dr_labels.csv and data/DR_images).
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split

    if data_dir is not None:
        csv_path = csv_path or os.path.join(data_dir, 'dr_labels.csv')
        images_dir = images_dir or os.path.join(data_dir, 'DR_images')

    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    if 'id_code' in df.columns:
        df.rename(columns={'id_code': 'id'}, inplace=True)
    if 'Label' in df.columns:
        df.rename(columns={'Label': 'label'}, inplace=True)

    if 'id' not in df.columns or 'label' not in df.columns:
        raise ValueError("CSV must contain 'id' (or 'id_code') and 'label' (or 'Label') columns")

    train_df, val_df = train_test_split(df, test_size=val_split, stratify=df['label'], random_state=random_state)

    # define transforms for train/val
    train_transform = transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),  # Slightly larger for crop
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),  # Fundus images can be flipped
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Note: DataLoader expects Dataset objects, so we need to pass the actual DataFrame differently.
    # To keep the FundusDataset API simple, we'll implement small wrappers that accept a DataFrame directly if csv_path is a string of CSV content.
    # The FundusDataset currently expects a file path; to avoid major refactor we will save temporary CSVs.

    import tempfile
    tmp_train = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv')
    tmp_val = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv')
    train_df.to_csv(tmp_train.name, index=False)
    val_df.to_csv(tmp_val.name, index=False)
    tmp_train.close()
    tmp_val.close()

    train_ds = FundusDataset(tmp_train.name, images_dir, transform=train_transform)
    val_ds = FundusDataset(tmp_val.name, images_dir, transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader
