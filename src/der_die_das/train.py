from __future__ import annotations

import os

import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from der_die_das.utils import MODEL_DIR, GermanNouns


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

    def save_model(self) -> None:
        timestamp = pd.Timestamp.now().strftime("%Y%m%d%H%M%S")
        torch.save(self.state_dict(), os.path.join(MODEL_DIR, f"model_{timestamp}.pt"))


def train() -> None:
    dataset = GermanNouns()
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = TransformerClassifier(
        vocab_size=dataset.vocab_size, max_length=dataset.max_length, num_classes=len(set(dataset.labels))
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    for epoch in range(50):
        running_loss = 0.0

        for batch_words, batch_labels in dataloader:
            optimizer.zero_grad()
            outputs = model(torch.tensor(batch_words))
            loss = criterion(outputs, torch.tensor(batch_labels))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        scheduler.step()

        print(f"Epoch {epoch + 1}, Running loss: {running_loss / len(dataloader)}, Loss: {loss.item()}")

    model.save_model()
