from __future__ import annotations

from torch import nn, optim
from torch.utils.data import DataLoader, random_split

from der_die_das.model import Config, TransformerClassifier
from der_die_das.utils import NounsDataset


def train(settings: dict) -> None:
    language = settings.pop("language")

    config = Config(settings, language=language)
    print(config.__dict__)

    dataset = NounsDataset(language=language)

    eval_samples = int(len(dataset) * config.val_size)
    train_samples = len(dataset) - eval_samples

    print(f"Training samples: {train_samples}, Evaluation samples: {eval_samples}")

    train, val = random_split(dataset, [train_samples, eval_samples])
    train_loader = DataLoader(train, batch_size=config.batch_size, shuffle=True)
    eval_loader = DataLoader(val, batch_size=config.batch_size, shuffle=False)

    model = TransformerClassifier(
        vocab_size=dataset.vocab_size, max_length=dataset.max_length, num_classes=len(set(dataset.labels))
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)

    epoch_losses = []
    min_val_loss = float("inf")
    epoch_min_loss = 0

    for epoch in range(config.num_epochs):
        train_loss = 0.0

        for batch_words, batch_labels in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_words)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        scheduler.step()

        eval_loss = 0.0

        for batch_words, batch_labels in eval_loader:
            outputs = model(batch_words)
            loss = criterion(outputs, batch_labels)
            eval_loss += loss.item()

        val_loss_epoch = eval_loss / len(eval_loader) if len(eval_loader) > 0 else None
        if config.early_stop is not None:
            if val_loss_epoch < min_val_loss:
                min_val_loss = val_loss_epoch
                epoch_min_loss = 0
            else:
                epoch_min_loss += 1

            if epoch_min_loss == config.early_stop:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        epoch_losses.append((train_loss / len(train_loader), val_loss_epoch))

        print(f"Epoch {epoch + 1}, Loss: {train_loss / len(train_loader):.4f}, Val Loss: {val_loss_epoch:.4f}")

    model.save_model(epoch_losses, config)
