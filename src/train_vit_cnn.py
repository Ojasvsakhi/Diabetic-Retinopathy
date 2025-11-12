import argparse
import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

from .config import cfg
from .data_loader import get_train_val_loaders
from sklearn.metrics import accuracy_score, f1_score
from .vit_cnn import CNNViTHybrid
import numpy as np
from sklearn.utils.class_weight import compute_class_weight


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(args):
    set_seed(cfg.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, val_loader = get_train_val_loaders(data_dir=args.data_dir, batch_size=args.batch_size, img_size=args.img_size, num_workers=args.num_workers)

    model = CNNViTHybrid(num_classes=cfg.num_classes, pretrained_cnn=args.pretrained_cnn, pretrained_vit=args.pretrained_vit).to(device)

    # compute class weights from training dataset
    try:
        train_labels = []
        if hasattr(train_loader.dataset, 'df'):
            train_labels = train_loader.dataset.df['label'].dropna().astype(int).values
        else:
            for _, lbls in train_loader:
                train_labels.extend(lbls.numpy().tolist())

        classes = np.unique(train_labels)
        if len(classes) > 1:
            class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=train_labels)
            weights_arr = np.ones(cfg.num_classes, dtype=np.float32)
            for c, w in zip(classes, class_weights):
                weights_arr[int(c)] = float(w)
            weights_tensor = torch.tensor(weights_arr, dtype=torch.float).to(device)
            criterion = nn.CrossEntropyLoss(weight=weights_tensor)
            print('Using class weights:', weights_arr)
        else:
            print('Only one class found in training labels; using unweighted loss')
            criterion = nn.CrossEntropyLoss()
    except Exception as e:
        print('Could not compute class weights:', e)
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    use_amp = torch.cuda.is_available()
    scaler = GradScaler() if use_amp else None

    best_val_f1 = 0.0
    for epoch in range(args.epochs):
        model.train()
        loop = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs} [train]')
        total_loss = 0.0
        for imgs, labels in loop:
            imgs = imgs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            if use_amp:
                with autocast():
                    out = model(imgs)
                    logits = out[0] if isinstance(out, (tuple, list)) else out
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                out = model(imgs)
                logits = out[0] if isinstance(out, (tuple, list)) else out
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=total_loss / (loop.n + 1))

        # validation
        model.eval()
        all_preds = []
        all_targets = []
        val_loss = 0.0
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc='Validation'):
                imgs = imgs.to(device)
                labels = labels.to(device)
                if use_amp:
                    with autocast():
                        out = model(imgs)
                        logits = out[0] if isinstance(out, (tuple, list)) else out
                        loss = criterion(logits, labels)
                else:
                    out = model(imgs)
                    logits = out[0] if isinstance(out, (tuple, list)) else out
                    loss = criterion(logits, labels)
                val_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy().tolist())
                all_targets.extend(labels.cpu().numpy().tolist())

        val_acc = accuracy_score(all_targets, all_preds)
        val_f1 = f1_score(all_targets, all_preds, average='macro')
        print(f'Epoch {epoch+1} validation -- acc: {val_acc:.4f} macro-F1: {val_f1:.4f} val_loss: {val_loss/len(val_loader):.4f}')

        # checkpoint
        os.makedirs('checkpoints', exist_ok=True)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            ckpt = os.path.join('checkpoints', f'best_vitcnn_epoch_{epoch+1}.pt')
            torch.save({'model_state': model.state_dict(), 'epoch': epoch+1, 'val_f1': val_f1}, ckpt)

        # scheduler step on validation loss
        try:
            scheduler.step(val_loss / len(val_loader))
        except Exception:
            pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--model-name', type=str, default='vit_cnn', help='model identifier (e.g. vit_cnn)')
    parser.add_argument('--pretrained-cnn', action='store_true')
    parser.add_argument('--pretrained-vit', action='store_true')
    args = parser.parse_args()
    # pass args into loader creation
    # Note: get_train_val_loaders ignores pretrained flags; it uses img_size, batch_size
    train(args)
