from __future__ import annotations

import os
from typing import ClassVar

import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from der_die_das.utils import DATA_DIR


class TransformerClassifier(nn.Module):
    def __init__(
        self, vocab_size: int, embed_dim: int, num_heads: int, num_layers: int, max_length: int, num_classes: int
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_length, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x: list) -> torch.Tensor:
        x = self.embedding(x) + self.positional_encoding[:, : x.size(1), :]
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)  # Global average pooling
        return self.fc(x)


class GermanNouns(Dataset):
    CHAR_TO_IDX: ClassVar = {char: idx for idx, char in enumerate("abcdefghijklmnopqrstuvwxyzäöüß-")}

    def __init__(self) -> None:
        df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
        test = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
        self.max_length = max(df["x"].str.len().max(), test["x"].str.len().max())

        self.words = self.encode_words(df["x"].values)
        self.labels = df["y"].values
        self.vocab_size = len(self.CHAR_TO_IDX)

    def encode_words(self, words: list[str]) -> list[list[int]]:
        encoded_words = []
        for word in words:
            encoded = [self.CHAR_TO_IDX[char] for char in word]
            encoded += [0] * (self.max_length - len(word))
            encoded = torch.tensor(encoded)
            encoded_words.append(encoded)

        return encoded_words

    def __len__(self) -> int:
        return len(self.words)

    def __getitem__(self, idx: int) -> tuple[list, int]:
        return self.words[idx], self.labels[idx]


def train() -> None:
    dataset = GermanNouns()
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    vocab_size = dataset.vocab_size
    embed_dim = 128
    num_heads = 8
    num_layers = 2
    max_length = dataset.max_length
    num_classes = len(set(dataset.labels))

    model = TransformerClassifier(vocab_size, embed_dim, num_heads, num_layers, max_length, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(50):
        for batch_words, batch_labels in dataloader:
            batch_words_tensor = torch.tensor(batch_words)
            batch_labels_tensor = torch.tensor(batch_labels)

            optimizer.zero_grad()
            outputs = model(batch_words_tensor)
            loss = criterion(outputs, batch_labels_tensor)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item()}")
