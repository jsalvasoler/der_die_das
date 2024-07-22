from __future__ import annotations

import os
from typing import ClassVar

import pandas as pd
import torch
from torch.utils.data import Dataset

this_directory = os.path.dirname(__file__)

DATA_DIR = os.path.join(this_directory, "..", "..", "data")
MODEL_DIR = os.path.join(this_directory, "..", "..", "models")


class GermanNouns(Dataset):
    CHAR_TO_IDX: ClassVar = {char: idx for idx, char in enumerate("abcdefghijklmnopqrstuvwxyzäöüß-")}

    def __init__(self, which: str = "train") -> None:
        assert which in ["train", "test"]

        train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
        test = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))

        self.max_length = max(train["x"].str.len().max(), test["x"].str.len().max())
        self.vocab_size = len(self.CHAR_TO_IDX)

        if which == "train":
            self.words = self.encode_words(train["x"].values)
            self.labels = train["y"].values
        else:
            self.words = self.encode_words(test["x"].values)
            self.labels = test["y"].values

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
