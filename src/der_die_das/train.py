from __future__ import annotations

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from der_die_das.model import TransformerClassifier
from der_die_das.utils import GermanNouns


class Config:
    def __init__(self, settings: dict) -> None:
        self.batch_size = settings.get("batch_size", 32)
        self.lr = settings.get("lr", 0.001)
        self.step_size = settings.get("step_size", 10)
        self.gamma = settings.get("gamma", 0.1)
        self.num_epochs = settings.get("num_epochs", 2)


def train(settings: dict) -> None:
    print(f"Using the following settings: {settings}")

    config = Config(settings)

    dataset = GermanNouns()
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    model = TransformerClassifier(
        vocab_size=dataset.vocab_size, max_length=dataset.max_length, num_classes=len(set(dataset.labels))
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)

    epoch_losses = []

    for epoch in range(config.num_epochs):
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
