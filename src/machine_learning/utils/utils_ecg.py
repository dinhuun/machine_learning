from typing import Tuple

import numpy as np
import pandas as pd
import torch
from numpy import ndarray
from sklearn.model_selection import train_test_split
from tensorflow.python.framework.ops import disable_eager_execution
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

disable_eager_execution()


class ECGSet(Dataset):
    """
    ECG dataset
    """

    def __init__(self, X: Tensor):
        """
        initializes ECGSet
        :param X: tensors
        """
        self.X = X

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        x = self.X[index]
        return x, x

    def __len__(self) -> int:
        return len(self.X)


def load_ecg_data(filepath: str) -> Tuple[ndarray, ndarray]:
    """
    loads ECG dataset (ecgs, labels) where label 1 is normal and label 0 is anomalous
    :param filepath: where dataset is
    :return: ecgs and their labels
    """
    df = pd.read_csv(filepath, header=None)
    values = df.values
    X = values[:, :-1]
    Y = values[:, -1]
    return X, Y


def load_ecg_data_for_autoencoder(
    filepath: str,
    scaled: bool = False,
    val_size: int = 580,
    test_size: int = 290,
    batch_size: int = 64,
    drop_last: bool = False,
):
    """
    loads ECG dataset
    :param filepath: where dataset is
    :param scaled: whether to scale (x - min) / (max - min)
    :param val_size: val size
    :param test_size: test size
    :param batch_size: batch size
    :param drop_last: whether to drop last batch from dataloader, to avoid error of normalizing batch of size 1
    :return: normal train dataloader, normal val dataloader, normal test dataloader, anomalous dataloader
    """
    X, Y = load_ecg_data(filepath)
    Y = Y.astype(bool)
    normal_X = X[Y]
    anomalous_X = X[~Y]

    normal_X_temp, normal_X_test = train_test_split(normal_X, test_size=test_size)
    normal_X_train, normal_X_val = train_test_split(normal_X_temp, test_size=val_size)

    if scaled:
        min_ = np.min(normal_X_train)
        max_ = np.max(normal_X_train)
        spread = max_ - min_
        normal_X_train = (normal_X_train - min_) / spread
        normal_X_val = (normal_X_val - min_) / spread
        normal_X_test = (normal_X_test - min_) / spread
        anomalous_X = (anomalous_X - min_) / spread

    normal_X_train = torch.from_numpy(normal_X_train).to(torch.float32)
    normal_X_val = torch.from_numpy(normal_X_val).to(torch.float32)
    normal_X_test = torch.from_numpy(normal_X_test).to(torch.float32)
    anomalous_X = torch.from_numpy(anomalous_X).to(torch.float32)

    normal_train_set = ECGSet(normal_X_train)
    normal_val_set = ECGSet(normal_X_val)
    normal_test_set = ECGSet(normal_X_test)
    anomalous_set = ECGSet(anomalous_X)

    normal_X_train_loader = DataLoader(
        normal_train_set, batch_size=batch_size, drop_last=drop_last
    )
    normal_X_val_loader = DataLoader(normal_val_set, batch_size=batch_size)
    normal_X_test_loader = DataLoader(normal_test_set, batch_size=batch_size)
    anomalous_X_loader = DataLoader(anomalous_set, batch_size=batch_size)
    return (
        normal_X_train_loader,
        normal_X_val_loader,
        normal_X_test_loader,
        anomalous_X_loader,
    )
