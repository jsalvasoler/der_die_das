from __future__ import annotations

import os

import torch
from matplotlib import pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

from der_die_das.model import TransformerClassifier
from der_die_das.utils import EVAL_DIR, MODEL_DIR, NounsDataset


def evaluate(model_timestamp: str | None = None) -> None:
    # Load the model
    model_dirs = os.listdir(MODEL_DIR)

    if model_timestamp is not None:
        assert len(model_timestamp) == len("YYYYMMDDHHMMSS"), "Model timestamp should have the format YYYYMMDDHHMMSS"
        try:
            model_dir = next(model_dir for model_dir in model_dirs if model_timestamp in model_dir)
        except StopIteration:
            print(f"No model found with timestamp {model_timestamp}")
            return
    else:
        models = sorted(model_dirs, reverse=True, key=lambda x: x.split("_")[1])
        model_dir = models[0]

    model_timestamp = model_dir.split("_")[1].split(".")[0]
    model_path = os.path.join(MODEL_DIR, model_dir, f"model_{model_timestamp}.pt")
    state_dict = torch.load(model_path)

    # Prepare the test dataset
    # read settings.txt to get the language
    with open(os.path.join(MODEL_DIR, model_dir, "settings.txt")) as f:
        settings = {line.split(":")[0]: line.split(":")[1].strip() for line in f.readlines()}
        language = settings["language"]
    dataset = NounsDataset(language=language, which="test")

    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    model = TransformerClassifier(
        vocab_size=dataset.vocab_size, max_length=dataset.max_length, num_classes=len(set(dataset.labels))
    )
    model.load_state_dict(state_dict)

    # Evaluation metrics
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_words, batch_labels in dataloader:
            outputs = model(batch_words)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="weighted")
    recall = recall_score(all_labels, all_preds, average="weighted")
    f1 = f1_score(all_labels, all_preds, average="weighted")

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # create directory for the model's evaluation
    eval_dir = os.path.join(EVAL_DIR, f"evaluation_{model_timestamp}")
    os.makedirs(eval_dir, exist_ok=True)

    cm = confusion_matrix(all_labels, all_preds)
    labels = ["der", "die", "das"] if language == "german" else ["el", "la"]
    display = ConfusionMatrixDisplay(cm, display_labels=labels)
    display.plot().figure_.savefig(os.path.join(eval_dir, f"confusion_matrix_{model_timestamp}.png"))

    # save metrics to file
    with open(os.path.join(eval_dir, f"metrics_{model_timestamp}.txt"), "w") as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")

    # save confusion matrix to the same file
    with open(os.path.join(eval_dir, f"metrics_{model_timestamp}.txt"), "ab") as f:
        f.write(b"\n")
        cm.tofile(f, sep=",")

    # plot the learning curve and save it
    model_dir = os.path.join(MODEL_DIR, f"model_{model_timestamp}")
    with open(os.path.join(model_dir, "epoch_losses.txt")) as f:
        epoch_losses = [line.strip().split(",") for line in f.readlines()]
        epoch_losses = [[float(train_loss), float(eval_loss)] for train_loss, eval_loss in epoch_losses]

    plt.clf()

    train_losses = [loss[0] for loss in epoch_losses]
    eval_losses = [loss[1] for loss in epoch_losses]

    # Plot the learning curve on two different axis
    fig, ax1 = plt.subplots()

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color="tab:blue")
    ax1.plot(range(1, len(train_losses) + 1), train_losses, color="tab:blue", label="Train Loss")
    ax1.plot(range(1, len(eval_losses) + 1), eval_losses, color="tab:orange", label="Eval Loss")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.legend(loc="upper left")

    plt.title("Learning Curve")
    plt.savefig(os.path.join(eval_dir, f"learning_curve_{model_timestamp}.png"))
