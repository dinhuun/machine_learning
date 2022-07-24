# todo: implement k-valued factorization machine by using W(n_features, n_factors, k) and W(:, :, index)

from copy import deepcopy
from typing import Tuple

import numpy as np
import torch
from pydantic import BaseModel
from sklearn.metrics import accuracy_score
from torch import Tensor
from torch.nn import Linear, Module, Parameter

from machine_learning.classes import TrainConfigs
from machine_learning.strings import (
    activation_name_str,
    bce_loss_str,
    identity_str,
    n_factors_str,
    n_features_str,
    state_dict_str,
)
from machine_learning.utils.utils_networks import (
    init_activation,
    init_loss,
    init_optimizer,
)


class ModelConfigs(BaseModel):
    n_features: int = 1
    n_factors: int = 2
    activation_name: str = identity_str


class FactorizationMachine(Module):
    """
    factorization machine with degree-2 interaction
    as in paper Factorization Machines by Steffen Rendle
    as in notes Factorization Machines in factorization_machines.pdf
    """

    def __init__(
        self,
        n_features: int = 1,
        n_factors: int = 1,
        activation_name: str = identity_str,
    ):
        """
        initializes FactorizationMachine
        """
        super().__init__()
        self.n_features = n_features
        self.n_factors = n_factors
        self.activation_name = activation_name

        self.lin = Linear(n_features, 1)
        self.W = Parameter(torch.rand(n_features, n_factors), requires_grad=True)
        self.activation = init_activation(self.activation_name)

    def forward(self, x: Tensor) -> Tensor:
        """
        forwards input to output
        """
        out_0 = torch.matmul(x, self.W).pow(2).sum(1, keepdim=True)
        out_1 = torch.matmul(x.pow(2), self.W.pow(2)).sum(1, keepdim=True)
        interaction = (
            out_0 - out_1
        ) / 2  # works out to be interaction term in Factorization Machine, equation 1
        model_equation = self.activation(
            self.lin(x) + interaction,
        )  # bias term already in self.lin()
        return model_equation


def train(
    configs: TrainConfigs,
    model: FactorizationMachine,
    X_train: Tensor,
    Y_train: Tensor,
    X_val: Tensor,
    Y_val: Tensor,
    loss_name: str = bce_loss_str,
    verbose: bool = False,
    verbose_freq: int = 100,
) -> Tuple[FactorizationMachine, float, int]:
    """
    trains model to X_train, Y_train
    validates trained model to X_val, Y_val
    :param configs: training configs
    :param model: model
    :param X_train: train samples
    :param Y_train: train targets
    :param X_val: val samples
    :param Y_val: val targets
    :param loss_name: loss name
    :param verbose: whether to print train loss and val score over epochs
    :param verbose_freq: how often to print train loss and val score over epochs
    :return: best model, best score, best epoch
    """
    criterion = init_loss(loss_name)

    optimizer_configs = configs.optimizer_configs
    alpha = optimizer_configs.alpha
    eps = optimizer_configs.eps
    lr = optimizer_configs.lr
    momentum = optimizer_configs.momentum
    optimizer_name = optimizer_configs.optimizer_name
    weight_decay = optimizer_configs.weight_decay
    optimizer = init_optimizer(
        optimizer_name, model.parameters(), lr, momentum, alpha, eps, weight_decay
    )

    best_epoch = -1
    best_score = 0.0
    best_model = FactorizationMachine()
    n_epochs_without_improvement = 0
    for i in range(configs.n_epochs):
        model.train()
        Y_train_pred = model(X_train)

        loss = criterion(Y_train_pred, Y_train)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        model.eval()
        Y_val_pred = model(X_val)
        score = accuracy_score(np.around(Y_val_pred.detach().numpy()), Y_val)

        if verbose is True and i % verbose_freq == 0:
            print(f"epoch {i}, train loss {loss.item()}, val score {score}")

        if score > best_score:
            best_epoch = i
            best_score = score
            best_model = deepcopy(model)
            n_epochs_without_improvement = 0
        else:
            n_epochs_without_improvement += 1
            if n_epochs_without_improvement > configs.patience:
                print(f"ran out of patience at epoch {i}, best score {best_score}")
                break
    return best_model, best_score, best_epoch


def load(filepath: str) -> FactorizationMachine:
    """
    loads model from filepath
    """
    checkpoint = torch.load(filepath)
    state_dict = checkpoint.pop(state_dict_str)
    model = FactorizationMachine(**checkpoint)
    model.load_state_dict(state_dict)
    return model


def save(model: FactorizationMachine, filepath: str):
    """
    saves model to filepath
    """
    checkpoint = {
        n_features_str: model.n_features,
        n_factors_str: model.n_factors,
        activation_name_str: model.activation_name,
        state_dict_str: model.state_dict(),
    }
    torch.save(checkpoint, filepath)
