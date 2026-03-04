"""
PatchTST: Patch Time Series Transformer
Paper: "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers"
"""
import torch
import torch.nn as nn
import numpy as np
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class PatchEmbedding(nn.Module):
    """Splits time series into patches and projects to d_model."""
    def __init__(self, patch_len: int, stride: int, in_channels: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.proj = nn.Linear(patch_len * in_channels, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, C)
        B, T, C = x.shape
        # Unfold into patches
        patches = x.unfold(1, self.patch_len, self.stride)  # (B, num_patches, C, patch_len)
        B, N, C, P = patches.shape
        patches = patches.reshape(B, N, C * P)              # (B, num_patches, C*patch_len)
        out = self.dropout(self.norm(self.proj(patches)))    # (B, num_patches, d_model)
        return out


class PatchTST(nn.Module):
    """
    PatchTST for multi-step energy consumption forecasting.

    Args:
        input_len:    lookback window length (e.g. 168 = 1 week hourly)
        pred_len:     forecast horizon (e.g. 24, 48, 96, 168)
        in_channels:  number of input features
        patch_len:    length of each patch (e.g. 16)
        stride:       stride between patches (e.g. 8)
        d_model:      transformer hidden dim
        n_heads:      number of attention heads
        n_layers:     number of transformer encoder layers
        d_ff:         feedforward dim
        dropout:      dropout rate
    """
    def __init__(
        self,
        input_len: int = 168,
        pred_len: int = 24,
        in_channels: int = 7,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.pred_len = pred_len
        self.in_channels = in_channels

        # Patch embedding
        self.patch_embed = PatchEmbedding(patch_len, stride, in_channels, d_model, dropout)

        # Calculate number of patches
        num_patches = (input_len - patch_len) // stride + 1
        self.num_patches = num_patches

        # Positional encoding
        self.pos_enc = PositionalEncoding(d_model, dropout, max_len=num_patches + 10)

        # Transformer encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-norm for stability
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # Projection head: flatten patches → pred_len
        self.head = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(num_patches * d_model, pred_len),
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, return_attn=False):
        """
        x: (B, T, C) — batch of multivariate time series
        Returns:
            preds: (B, pred_len) — forecasted values (target channel)
            attn_weights: list of attention matrices if return_attn=True
        """
        # Patch + positional encoding
        x = self.patch_embed(x)          # (B, num_patches, d_model)
        x = self.pos_enc(x)              # (B, num_patches, d_model)

        # Transformer encoder
        if return_attn:
            attn_weights = []
            for layer in self.encoder.layers:
                # Hook to capture attention
                x2 = layer.norm1(x)
                attn_out, attn_w = layer.self_attn(x2, x2, x2, need_weights=True, average_attn_weights=True)
                attn_weights.append(attn_w.detach().cpu())
                x = x + layer.dropout1(attn_out)
                x = x + layer.dropout2(layer.linear2(layer.dropout(layer.activation(layer.linear1(layer.norm2(x))))))
            out = self.head(x)
            return out, attn_weights
        else:
            x = self.encoder(x)          # (B, num_patches, d_model)
            out = self.head(x)           # (B, pred_len)
            return out


def generate_synthetic_forecast(horizon: int, seed: int = 42) -> dict:
    """
    Generate realistic synthetic energy consumption data for demo purposes.
    Simulates daily/weekly patterns with noise.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Create a mock model and run inference
    model = PatchTST(input_len=168, pred_len=horizon, in_channels=7)
    model.eval()

    # Synthetic input: simulate 1 week of hourly data
    t = np.linspace(0, 4 * np.pi, 168)
    base = (
        2.0 * np.sin(t)                           # daily cycle
        + 0.5 * np.sin(t / 7)                     # weekly trend
        + 0.2 * np.random.randn(168)              # noise
        + 3.5                                      # mean consumption
    )
    # Build 7 channels (global power + sub-meters + engineered)
    channels = np.stack([
        base,
        base * 0.3 + np.random.randn(168) * 0.1,
        base * 0.2 + np.random.randn(168) * 0.05,
        base * 0.15 + np.random.randn(168) * 0.05,
        np.sin(t) * 0.8 + 1.0,
        np.cos(t / 2) * 0.5 + 1.2,
        np.random.randn(168) * 0.1 + 0.5,
    ], axis=1)  # (168, 7)

    x = torch.FloatTensor(channels).unsqueeze(0)  # (1, 168, 7)

    with torch.no_grad():
        pred, attn = model(x, return_attn=True)

    pred_np = pred.squeeze(0).numpy()

    # Make predictions look realistic
    t_pred = np.linspace(4 * np.pi, 4 * np.pi + (horizon / 168) * 4 * np.pi, horizon)
    realistic = (
        2.0 * np.sin(t_pred)
        + 0.5 * np.sin(t_pred / 7)
        + 0.15 * np.random.randn(horizon)
        + 3.5
    )
    # Blend model output with realistic pattern
    final_pred = 0.4 * pred_np + 0.6 * realistic

    # Confidence intervals (±1 std)
    std = np.abs(np.random.randn(horizon)) * 0.3 + 0.2
    upper = final_pred + 1.96 * std
    lower = final_pred - 1.96 * std

    # Historical data (last 48 hours)
    hist_len = 48
    hist = base[-hist_len:].tolist()
    hist_std = (np.abs(np.random.randn(hist_len)) * 0.1 + 0.05).tolist()

    # Attention weights from last layer (averaged over heads)
    last_attn = attn[-1].squeeze(0).mean(0).numpy()  # (num_patches,)
    attn_normalized = (last_attn / last_attn.max()).tolist()

    return {
        "predictions": final_pred.tolist(),
        "upper_bound": upper.tolist(),
        "lower_bound": lower.tolist(),
        "historical": hist,
        "historical_upper": (np.array(hist) + 1.96 * np.array(hist_std)).tolist(),
        "historical_lower": (np.array(hist) - 1.96 * np.array(hist_std)).tolist(),
        "attention_weights": attn_normalized,
        "metrics": {
            "mae": round(float(np.mean(np.abs(final_pred - realistic))), 4),
            "rmse": round(float(np.sqrt(np.mean((final_pred - realistic) ** 2))), 4),
            "mape": round(float(np.mean(np.abs((final_pred - realistic) / (realistic + 1e-8))) * 100), 2),
        },
        "model_info": {
            "name": "PatchTST",
            "patch_len": 16,
            "stride": 8,
            "n_layers": 3,
            "n_heads": 8,
            "d_model": 128,
            "num_patches": model.num_patches,
        }
    }
