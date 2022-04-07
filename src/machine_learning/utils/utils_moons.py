from typing import Tuple

import numpy as np
import torch
from sklearn import datasets
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


class MoonsSet(Dataset):
    """
    a vertical stack of coordinates of sklearn.datasets.make_moons(), without labels
    """

    def __init__(
        self, n_samples: int = 1000, n_moons: int = 1, noise_factor: float = 0.05
    ):
        """
        initializes MoonsSet
        :param n_samples: number of samples in 2 moons
        :param n_moons: number of pairs of moons to stack
        :param noise_factor: noise factor
        """
        subsets = []
        for _ in range(n_moons):
            subset, _ = datasets.make_moons(n_samples=n_samples, noise=noise_factor)
            subsets.append(subset)
        W = np.vstack(subsets).astype(np.float32)
        X = torch.from_numpy(W)
        self.X = X

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        x = self.X[index]
        return x, x

    def __len__(self) -> int:
        return len(self.X)


def load_moons_data(
    n_samples: int = 1000,
    n_moons_train: int = 5,
    n_moons_val: int = 2,
    n_moons_test: int = 1,
    noise_factor: float = 0.05,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    loads sklearn.datasets.make_moons() as (coordinates, coordinates), without labels
    :param n_samples: number of samples in 2 moons
    :param n_moons_train: number of pairs of moons for training
    :param n_moons_val: number of pairs of moons for validating
    :param n_moons_test: number of pairs of moons for testing
    :param noise_factor: noise factor
    :return: loaders (X_train_loader, X_val_loader, X_test_loader)
    """
    X_train = MoonsSet(n_samples, n_moons_train, noise_factor)
    X_val = MoonsSet(n_samples, n_moons_val, noise_factor)
    X_test = MoonsSet(n_samples, n_moons_test, noise_factor)
    X_train_loader = DataLoader(X_train, batch_size=n_samples, shuffle=False)
    X_val_loader = DataLoader(X_val, batch_size=n_samples, shuffle=False)
    X_test_loader = DataLoader(X_test, batch_size=n_samples, shuffle=False)
    return X_train_loader, X_val_loader, X_test_loader
