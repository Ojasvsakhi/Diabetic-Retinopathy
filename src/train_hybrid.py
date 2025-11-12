import argparse
import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from .config import cfg
from .data_loader import get_train_val_loaders
from sklearn.metrics import accuracy_score, f1_score
from .hybrid_model import HybridCNNLSTM
from .models import MultiTaskResNet


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(args):
    set_seed(cfg.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, val_loader = get_train_val_loaders(data_dir=args.data_dir, batch_size=args.batch_size, img_size=args.img_size, num_workers=args.num_workers)

    model_name = getattr(args, 'model_name', None) or cfg.model_name
    model_name = str(model_name).lower()
    if model_name in ('hybrid_cnn_lstm', 'hybrid', 'cnn_lstm'):
        model = HybridCNNLSTM(backbone_name='resnet50', num_classes=cfg.num_classes).to(device)
    else:
        # default: instantiate the multitask ResNet from src.models
        model = MultiTaskResNet(backbone_name=cfg.model_name, num_classes=cfg.num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        loop = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs} [train]')
        total_loss = 0.0
        for imgs, labels in loop:
            imgs = imgs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            # Support both hybrid model (returns logits) and multitask ResNet (returns tuple)
            if isinstance(model, HybridCNNLSTM):
                logits = model(imgs, is_sequence=False)
            else:
                out_main, _ = model(imgs)
                logits = out_main
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            loop.set_postfix(loss=total_loss / (loop.n + 1))

        # validation
        model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc='Validation'):
                imgs = imgs.to(device)
                labels = labels.to(device)
                if isinstance(model, HybridCNNLSTM):
                    logits = model(imgs, is_sequence=False)
                else:
                    logits, _ = model(imgs)
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy().tolist())
                all_targets.extend(labels.cpu().numpy().tolist())

        val_acc = accuracy_score(all_targets, all_preds)
        val_f1 = f1_score(all_targets, all_preds, average='macro')
        print(f'Epoch {epoch+1} validation -- acc: {val_acc:.4f} macro-F1: {val_f1:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--epochs', type=int, default=cfg.epochs)
    parser.add_argument('--batch-size', type=int, default=cfg.batch_size)
    parser.add_argument('--img-size', type=int, default=cfg.img_size)
    parser.add_argument('--lr', type=float, default=cfg.lr)
    parser.add_argument('--num-workers', type=int, default=cfg.num_workers)
    parser.add_argument('--model-name', type=str, default=cfg.model_name, help='model to instantiate: resnet50 (multitask) or hybrid_cnn_lstm')
    args = parser.parse_args()
    train(args)
