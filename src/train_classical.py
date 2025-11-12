import argparse
import os
import joblib
import numpy as np
import torch
from tqdm import tqdm

from .data_loader import get_train_val_loaders
from torchvision import models
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_sample_weight


def extract_features(model, loader, device):
    model.eval()
    feats_list = []
    labels_list = []
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc='Extracting features'):
            imgs = imgs.to(device)
            out = model(imgs)
            # flatten if needed
            if out.dim() == 4:
                out = out.view(out.size(0), -1)
            feats_list.append(out.cpu().numpy())
            labels_list.append(labels.numpy())

    X = np.concatenate(feats_list, axis=0)
    y = np.concatenate(labels_list, axis=0)
    return X, y


def train_classical(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    # loaders
    train_loader, val_loader = get_train_val_loaders(data_dir=args.data_dir,
                                                     batch_size=args.batch_size,
                                                     img_size=args.img_size,
                                                     num_workers=args.num_workers)

    # feature extractor backbone (ResNet50 without fc)
    backbone = models.resnet50(pretrained=args.pretrained_backbone)
    backbone.fc = torch.nn.Identity()
    backbone = backbone.to(device)

    # extract features
    X_train, y_train = extract_features(backbone, train_loader, device)
    X_val, y_val = extract_features(backbone, val_loader, device)

    print('Train features shape:', X_train.shape, 'Val features shape:', X_val.shape)

    # scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # compute sample weights (balanced)
    sample_weight = compute_sample_weight(class_weight='balanced', y=y_train)

    model_name = args.model_name.lower()
    clf = None

    if model_name in ('random_forest', 'rf'):
        clf = RandomForestClassifier(n_estimators=args.n_estimators, n_jobs=-1, random_state=args.seed)
        fit_kwargs = {'sample_weight': sample_weight}
    elif model_name in ('adaboost', 'ada'):
        clf = AdaBoostClassifier(n_estimators=args.n_estimators, random_state=args.seed)
        fit_kwargs = {'sample_weight': sample_weight}
    elif model_name in ('svm', 'svc'):
        clf = SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=args.seed)
        fit_kwargs = {'sample_weight': sample_weight}
    elif model_name in ('xgboost', 'xgb'):
        try:
            import xgboost as xgb
        except Exception as e:
            raise RuntimeError('xgboost is not installed: ' + str(e))
        clf = xgb.XGBClassifier(n_estimators=args.n_estimators, use_label_encoder=False, eval_metric='mlogloss', verbosity=1)
        fit_kwargs = {'sample_weight': sample_weight}
    else:
        raise ValueError('Unknown classical model: ' + args.model_name)

    print('Training classifier:', args.model_name)
    clf.fit(X_train, y_train, **fit_kwargs)

    # evaluate
    preds = clf.predict(X_val)
    acc = accuracy_score(y_val, preds)
    f1 = f1_score(y_val, preds, average='macro')
    print(f'Validation -- acc: {acc:.4f} macro-F1: {f1:.4f}')

    # save model + scaler
    os.makedirs('classical_models', exist_ok=True)
    out_path = os.path.join('classical_models', f'{args.model_name}.joblib')
    joblib.dump({'model': clf, 'scaler': scaler}, out_path)
    print('Saved model to', out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--model-name', type=str, default='random_forest', help='random_forest|xgboost|adaboost|svm')
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--n-estimators', type=int, default=100)
    parser.add_argument('--pretrained-backbone', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    train_classical(args)
