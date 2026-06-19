import numpy as np
import torch
from torch.utils.data import Dataset


class IndexedConcatDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        return img, label, idx


class AgeScoring:
    def __init__(self, n_samples: int):
        self.n_samples = n_samples
        self.age = np.zeros(n_samples, dtype=np.int32)
        self._epoch_correct = np.zeros(n_samples, dtype=np.int32)

    def update(self, indices, correct):
        idx = np.asarray(indices, dtype=np.int64)
        cor = np.asarray(correct, dtype=np.int32)
        self._epoch_correct[idx] += cor

    def epoch_end(self):
        self.age += self._epoch_correct
        self._epoch_correct[:] = 0

    def get(self):
        return self.age.copy()

    def save(self, path: str):
        np.save(path, self.age)

    @classmethod
    def load(cls, path: str, n_samples: int):
        obj = cls(n_samples)
        obj.age = np.load(path)
        return obj