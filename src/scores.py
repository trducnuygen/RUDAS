"""
utils/scores.py — Per-sample score accumulators.

Three signals, all computed during a single warm-up training run:

  AgeAccumulator
    age[i] += 1 each epoch that sample i is predicted correctly.
    Range: [0, E].  High = easy.

All accumulators operate on numpy arrays sized (N,) where N = len(train set).
They are updated in-place after each scoring pass and saved to disk.
"""

import time

import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional
from tqdm import tqdm

class AgeAccumulator:
    """
    Counts the number of epochs each sample is correctly predicted.
    """

    def __init__(self, n_samples):
        self.n_samples = n_samples
        self.age = np.zeros(n_samples, dtype=np.int32)

    def update(self, indices, correct):
        """
        indices : sample indices in the dataset
        correct : True if predicted correctly this epoch
        """
        self.age[indices] += correct.astype(np.int32)

    def get(self):
        return self.age.copy()

    def save(self, path):
        np.save(path, self.age)

    @classmethod
    def load(cls, path, n_samples):
        obj = cls(n_samples)
        obj.age = np.load(path)
        return obj


