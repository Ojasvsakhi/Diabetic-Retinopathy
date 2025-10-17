import argparse
import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from .config import cfg
from .data_loader import get_loaders, get_train_val_loaders
from sklearn.metrics import accuracy_score, f1_score
from .models import MultiTaskResNet


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(args):
    set_seed(cfg.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, val_loader = get_train_val_loaders(data_dir=args.data_dir, batch_size=args.batch_size, img_size=args.img_size, num_workers=args.num_workers)
    model = MultiTaskResNet(backbone_name=cfg.model_name, num_classes=cfg.num_classes).to(device)

    criterion_main = nn.CrossEntropyLoss()
    criterion_aux = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_f1 = 0.0
    model.train()
    for epoch in range(args.epochs):
        loop = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs} [train]')
        total_loss = 0.0
        for imgs, labels in loop:
            imgs = imgs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            out_main, out_aux = model(imgs)
            loss_main = criterion_main(out_main, labels)
            aux_labels = (labels >= 2).float().unsqueeze(1).to(device)
            loss_aux = criterion_aux(out_aux, aux_labels)
            loss = loss_main + 0.5 * loss_aux
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
                out_main, _ = model(imgs)
                preds = torch.argmax(out_main, dim=1)
                all_preds.extend(preds.cpu().numpy().tolist())
                all_targets.extend(labels.cpu().numpy().tolist())

        val_acc = accuracy_score(all_targets, all_preds)
        val_f1 = f1_score(all_targets, all_preds, average='macro')
        print(f'Epoch {epoch+1} validation -- acc: {val_acc:.4f} macro-F1: {val_f1:.4f}')

        # checkpoint best
        os.makedirs('checkpoints', exist_ok=True)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            ckpt_path = os.path.join('checkpoints', f'best_epoch_{epoch+1}.pt')
            torch.save({'model_state': model.state_dict(), 'epoch': epoch+1, 'val_f1': val_f1}, ckpt_path)

        model.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--epochs', type=int, default=cfg.epochs)
    parser.add_argument('--batch-size', type=int, default=cfg.batch_size)
    parser.add_argument('--img-size', type=int, default=cfg.img_size)
    parser.add_argument('--lr', type=float, default=cfg.lr)
    parser.add_argument('--num-workers', type=int, default=cfg.num_workers)
    args = parser.parse_args()
    train(args)
