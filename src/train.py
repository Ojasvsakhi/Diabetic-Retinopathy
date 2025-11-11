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
from .models import MultiTaskModel, MultiTaskResNet
import numpy as np
from sklearn.utils.class_weight import compute_class_weight


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Don't set deterministic=True as it significantly slows training
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False


def train(args):
    set_seed(cfg.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, val_loader = get_train_val_loaders(data_dir=args.data_dir, batch_size=args.batch_size, img_size=args.img_size, num_workers=args.num_workers)
    
    # Get model name from args if provided, otherwise use config
    model_name = getattr(args, 'model_name', None) or cfg.model_name
    
    print(f'Using model: {model_name}')
    model = MultiTaskModel(backbone_name=model_name, num_classes=cfg.num_classes).to(device)
    
    # ViT models need different training setup - lower LR and layer-wise LR decay
    is_vit = model_name.startswith('vit')
    if is_vit:
        # Auto-adjust learning rate for ViT if using default
        if args.lr == cfg.lr:
            args.lr = args.lr * 0.1  # ViT typically needs 10x lower LR (1e-5 instead of 1e-4)
            print(f'Auto-adjusted learning rate for ViT: {args.lr:.6f}')
    
    # compute class weights from the training dataset to help with imbalance
    try:
        # train_loader.dataset is our FundusDataset; extract labels from its dataframe if available
        train_labels = []
        if hasattr(train_loader.dataset, 'df'):
            train_labels = train_loader.dataset.df['label'].dropna().astype(int).values
        else:
            # fallback: iterate once over the loader to collect labels (small overhead)
            for _, lbls in train_loader:
                train_labels.extend(lbls.numpy().tolist())

        if len(train_labels) == 0:
            raise ValueError('No labels found in training split')

        classes = np.unique(train_labels)
        if len(classes) == 1:
            # single-class in training split -> cannot compute balanced weights
            print(f'Only one class present in training split: {classes}. Skipping class-weight computation.')
            criterion_main = nn.CrossEntropyLoss()
        else:
            class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=train_labels)
            # compute_class_weight returns weights in the order of `classes`, so we need a full vector of length num_classes
            weights_arr = np.ones(cfg.num_classes, dtype=np.float32)
            for c, w in zip(classes, class_weights):
                weights_arr[int(c)] = float(w)
            weights_tensor = torch.tensor(weights_arr, dtype=torch.float).to(device)
            # Sample message for debugging
            criterion_main = nn.CrossEntropyLoss(weight=weights_tensor)
            print(f'Using class weights: {weights_arr}')
    except Exception as e:
        print(f'Could not compute class weights automatically: {e}. Falling back to unweighted CrossEntropyLoss')
        criterion_main = nn.CrossEntropyLoss()
    criterion_aux = nn.BCEWithLogitsLoss()
    
    # Use different optimizers for ViT vs ResNet
    if is_vit:
        # ViT: Use AdamW with layer-wise learning rate decay
        # Separate parameters: backbone vs classification heads
        backbone_params = []
        head_params = []
        for name, param in model.named_parameters():
            if 'backbone' in name:
                backbone_params.append(param)
            else:
                head_params.append(param)
        
        # Use lower LR for pretrained backbone, higher for new heads
        optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': args.lr * 0.1, 'weight_decay': 0.01},  # Very low LR for pretrained
            {'params': head_params, 'lr': args.lr, 'weight_decay': 0.01}  # Normal LR for new heads
        ])
        print(f'Using AdamW with layer-wise LR: backbone={args.lr * 0.1:.6f}, heads={args.lr:.6f}')
    else:
        # ResNet: Use standard Adam optimizer
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # learning rate scheduler that reduces LR when validation metric plateaus
    # Create scheduler in a version-safe way: some PyTorch versions accept `verbose`,
    # some do not. Use inspect to decide which kwargs to pass.
    # Create scheduler simply without `verbose` to avoid compatibility issues
    # Learning rate scheduler - less aggressive to match original behavior
    try:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    except Exception:
        scheduler = None

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
            
            # Gradient clipping for ViT (helps with training stability)
            if is_vit:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
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

        # step scheduler with validation metric (macro-F1)
        if scheduler is not None:
            old_lrs = [group['lr'] for group in optimizer.param_groups]
            try:
                scheduler.step(val_f1)
            except Exception:
                # If scheduler is present but fails for any reason, continue training
                pass
            new_lrs = [group['lr'] for group in optimizer.param_groups]
            if new_lrs != old_lrs:
                print(f'Learning rates reduced: {old_lrs} -> {new_lrs}')

        model.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--epochs', type=int, default=cfg.epochs)
    parser.add_argument('--batch-size', type=int, default=cfg.batch_size)
    parser.add_argument('--img-size', type=int, default=cfg.img_size)
    parser.add_argument('--lr', type=float, default=cfg.lr)
    parser.add_argument('--num-workers', type=int, default=cfg.num_workers)
    parser.add_argument('--model-name', type=str, default=cfg.model_name, 
                        help='Model backbone: resnet50, vit_base_patch16_224, vit_small_patch16_224, vit_tiny_patch16_224')
    args = parser.parse_args()
    train(args)
