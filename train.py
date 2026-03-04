"""
train.py — Train PatchTST on UCI Household Electric Power Consumption Dataset

Usage:
    python train.py --data_path data/household_power_consumption.txt \
                    --horizon 24 --epochs 50 --batch_size 64

Dataset download:
    https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption
"""
import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))
from forecaster.model import PatchTST


# ─── Dataset ──────────────────────────────────────────────────────────────────
class UCIEnergyDataset(Dataset):
    def __init__(self, data: np.ndarray, input_len: int = 168, pred_len: int = 24):
        self.data = torch.FloatTensor(data)
        self.input_len = input_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.data) - self.input_len - self.pred_len

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.input_len]         # (input_len, C)
        y = self.data[idx + self.input_len:               # target = global active power only
                      idx + self.input_len + self.pred_len, 0]
        return x, y


# ─── Data loading ─────────────────────────────────────────────────────────────
def load_uci_data(path: str) -> np.ndarray:
    print(f"Loading data from {path}...")
    df = pd.read_csv(path, sep=';', na_values='?', low_memory=False,
                     parse_dates=[['Date', 'Time']], dayfirst=True,
                     index_col='Date_Time')

    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.resample('h').mean()       # hourly aggregation
    df = df.fillna(method='ffill')     # forward-fill gaps

    print(f"  Shape after hourly resampling: {df.shape}")

    # Add time features
    df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)

    # Normalize
    values = df.values
    mean = values.mean(axis=0)
    std = values.std(axis=0) + 1e-8
    normalized = (values - mean) / std

    return normalized, mean, std


# ─── Training loop ────────────────────────────────────────────────────────────
def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    data, mean, std = load_uci_data(args.data_path)
    n = len(data)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    train_ds = UCIEnergyDataset(data[:train_end], args.input_len, args.horizon)
    val_ds   = UCIEnergyDataset(data[train_end:val_end], args.input_len, args.horizon)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=2)

    print(f"Train: {len(train_ds)} samples | Val: {len(val_ds)} samples")

    # Model
    in_channels = data.shape[1]
    model = PatchTST(
        input_len=args.input_len,
        pred_len=args.horizon,
        in_channels=in_channels,
        patch_len=16,
        stride=8,
        d_model=128,
        n_heads=8,
        n_layers=3,
        d_ff=256,
        dropout=0.1,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"PatchTST parameters: {n_params:,}")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.MSELoss()

    best_val_mae = float('inf')
    os.makedirs('backend/forecaster/weights', exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        train_loss = 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        # Validate
        model.eval()
        val_mae = 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                val_mae += torch.mean(torch.abs(pred - y)).item()

        train_loss /= len(train_dl)
        val_mae /= len(val_dl)
        scheduler.step()

        print(f"Epoch {epoch:3d}/{args.epochs} | Loss: {train_loss:.4f} | Val MAE: {val_mae:.4f}")

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save({
                'model_state': model.state_dict(),
                'mean': mean,
                'std': std,
                'hyperparams': {
                    'input_len': args.input_len,
                    'pred_len': args.horizon,
                    'in_channels': in_channels,
                }
            }, f'backend/forecaster/weights/patchtst_h{args.horizon}.pt')
            print(f"  ✓ Saved best model (val_mae={val_mae:.4f})")

    print(f"\nTraining complete. Best Val MAE: {best_val_mae:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='data/household_power_consumption.txt')
    parser.add_argument('--horizon', type=int, default=24)
    parser.add_argument('--input_len', type=int, default=168)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()
    train(args)
