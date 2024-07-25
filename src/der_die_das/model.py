from __future__ import annotations

import os

import pandas as pd
import torch
from torch import nn

from der_die_das.utils import MODEL_DIR

DEFAULT_SETTINGS = {
    "batch_size": 32,
    "lr": 0.001,
    "step_size": 10,
    "gamma": 0.1,
    "num_epochs": 30,
    "val_size": 0.2,
    "early_stop": None,
}


class Config:
    def __init__(self, settings: dict, language: str) -> None:
        for k, v in DEFAULT_SETTINGS.items():
            setattr(self, k, settings.get(k, v))

        if self.early_stop is None:
            self.val_size = 0

        self.language = language


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_length: int,
        num_classes: int,
        embed_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_length, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x: list) -> torch.Tensor:
        x = self.embedding(x) + self.positional_encoding[:, : x.size(1), :]
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)  # Global average pooling
        return self.fc(x)

    def save_model(self, epoch_losses: list[float], config: Config) -> None:
        timestamp = pd.Timestamp.now().strftime("%Y%m%d%H%M%S")
        model_dir = os.path.join(MODEL_DIR, f"model_{timestamp}")
        os.makedirs(model_dir, exist_ok=True)

        torch.save(self.state_dict(), os.path.join(model_dir, f"model_{timestamp}.pt"))

        with open(os.path.join(model_dir, "epoch_losses.txt"), "w") as f:
            for train_loss, eval_loss in epoch_losses:
                f.write(f"{train_loss},{eval_loss}\n")

        with open(os.path.join(model_dir, "settings.txt"), "w") as f:
            f.write("Settings used for training:\n")
            for k, v in config.__dict__.items():
                f.write(f"{k}: {v}\n")
