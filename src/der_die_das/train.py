from __future__ import annotations

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from der_die_das.model import TransformerClassifier
from der_die_das.utils import GermanNouns


def train() -> None:
    dataset = GermanNouns()
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = TransformerClassifier(
        vocab_size=dataset.vocab_size, max_length=dataset.max_length, num_classes=len(set(dataset.labels))
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    epoch_losses = []

    for epoch in range(2):
        running_loss = 0.0

        for batch_words, batch_labels in dataloader:
            optimizer.zero_grad()
            outputs = model(torch.tensor(batch_words))
            loss = criterion(outputs, torch.tensor(batch_labels))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        scheduler.step()

        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(dataloader)}")
        epoch_losses.append(running_loss / len(dataloader))

    model.save_model(epoch_losses)
