from __future__ import annotations

import os

import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

from der_die_das.train import TransformerClassifier
from der_die_das.utils import MODEL_DIR, GermanNouns


def evaluate(model_time_stamp: str | None = None) -> None:
    models = os.listdir(MODEL_DIR)
    models = [model for model in models if model.endswith(".pt")]

    if model_time_stamp is not None:
        assert len(model_time_stamp) == len("YYYYMMDDHHMMSS"), "Model timestamp should have the format YYYYMMDDHHMMSS"
        try:
            model = next(model for model in models if model_time_stamp in model)
        except StopIteration:
            print(f"No model found with timestamp {model_time_stamp}")
            return
    else:
        models = sorted(models, reverse=True, key=lambda x: x.split("_")[1])
        model = models[0]

    print(f"Evaluating model {model}")
    model_path = os.path.join(MODEL_DIR, model)
    state_dict = torch.load(model_path)

    # Prepare the test dataset
    dataset = GermanNouns(which="test")
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
            outputs = model(torch.tensor(batch_words))
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
