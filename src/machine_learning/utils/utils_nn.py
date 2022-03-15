from typing import Iterable
from warnings import warn

import numpy as np
import torch
from torch import Tensor
from torch.nn import (
    BCELoss,
    CrossEntropyLoss,
    Identity,
    KLDivLoss,
    L1Loss,
    LeakyReLU,
    Linear,
    Module,
    MSELoss,
    NLLLoss,
    Parameter,
    ReLU,
    Sigmoid,
    SiLU,
    Softmax,
    Tanh,
)
from torch.optim import SGD, Adam, AdamW, Optimizer, RMSprop

from machine_learning.strings import (
    adam_str,
    adamw_str,
    bce_loss_str,
    ce_loss_str,
    identity_str,
    kl_div_loss_str,
    l1_loss_str,
    l1_str,
    l2_str,
    leaky_relu_str,
    mse_loss_str,
    nll_loss_str,
    normal_str,
    ones_str,
    relu_str,
    rmsprop_str,
    sgd_str,
    sigmoid_str,
    small_str,
    softmax_str,
    swish_str,
    tanh_str,
    uniform_str,
    xavier_normal_str,
    xavier_uniform_str,
)
from machine_learning.utils.utils_misc import raise_not_implemented_error


def init_activation(  # type: ignore
    activation_name: str = identity_str, negative_slope: float = 0.01
) -> Module:
    if activation_name == identity_str:
        return Identity()
    elif activation_name == leaky_relu_str:
        return LeakyReLU(negative_slope)
    elif activation_name == relu_str:
        return ReLU()
    elif activation_name == sigmoid_str:
        return Sigmoid()
    elif activation_name == softmax_str:
        return Softmax(dim=1)
    elif activation_name == swish_str:
        return SiLU()
    elif activation_name == tanh_str:
        return Tanh()
    else:
        raise_not_implemented_error("activation", activation_name)


def init_loss(loss_name: str = bce_loss_str, reduction: str = "") -> Module:  # type: ignore
    if loss_name == bce_loss_str:
        return BCELoss()
    elif loss_name == ce_loss_str:
        return CrossEntropyLoss()
    elif loss_name == kl_div_loss_str:
        return KLDivLoss()
    elif loss_name == l1_loss_str:
        return L1Loss()
    elif loss_name == mse_loss_str:
        return MSELoss(reduction=reduction)
    elif loss_name == nll_loss_str:
        return NLLLoss()
    else:
        raise_not_implemented_error("loss", loss_name)


def init_optimizer(  # type: ignore
    optimizer_name: str,
    params: Iterable[Parameter],
    lr: float = 0.01,
    momentum: float = 0.01,
    alpha: float = 0.99,
    eps: float = 1e-08,
    weight_decay: float = 0.01,
) -> Optimizer:
    if optimizer_name == sgd_str:
        return SGD(params, lr, momentum, weight_decay=weight_decay)
    elif optimizer_name == adam_str:
        return Adam(params, lr, eps=eps, weight_decay=weight_decay)
    elif optimizer_name == adamw_str:
        return AdamW(params, lr, eps=eps, weight_decay=weight_decay)
    elif optimizer_name == rmsprop_str:
        return RMSprop(params, lr, alpha, eps, weight_decay, momentum)
    else:
        raise_not_implemented_error("optimizer", optimizer_name)


def init_weights(name: str = small_str):  # type: ignore
    if name == normal_str:
        return init_weights_normal
    elif name == ones_str:
        return init_weights_ones
    elif name == small_str:
        return init_weights_small
    elif name == uniform_str:
        return init_weights_uniform
    elif name == xavier_normal_str:
        return init_weights_xavier_normal
    elif name == xavier_uniform_str:
        return init_weights_xavier_uniform
    else:
        raise_not_implemented_error("weights initializer", name)


def init_weights_normal(module: Module):
    """
    initializes all linear layers' weights as N(0.0, 1 / sqrt(n_features))
    """
    if isinstance(module, Linear):
        n_features = module.in_features
        torch.nn.init.normal_(module.weight, 0.0, 1 / np.sqrt(n_features))
        module.bias.data.fill_(0)


def init_weights_ones(module: Module):
    if isinstance(module, Linear):
        torch.nn.init.ones_(module.weight)
        module.bias.data.fill_(1)


def init_weights_small(module: Module):
    if isinstance(module, Linear):
        n_features = module.in_features
        epsilon = 1.0 / np.sqrt(n_features)
        torch.nn.init.uniform_(module.weight, -epsilon, epsilon)
        module.bias.data.fill_(0)


def init_weights_uniform(module: Module):
    if isinstance(module, Linear):
        torch.nn.init.uniform_(module.weight, 0, 1)
        module.bias.data.fill_(0)


def init_weights_xavier_normal(module: Module):
    if isinstance(module, Linear):
        torch.nn.init.xavier_normal_(module.weight)
        module.bias.data.fill_(0)


def init_weights_xavier_uniform(module: Module):
    if isinstance(module, Linear):
        torch.nn.init.xavier_uniform_(module.weight)
        module.bias.data.fill_(0)


def regularizer(  # type: ignore
    regularizer_name: str, params: Iterable[Parameter], lamda: float = 0.0
) -> Tensor:
    lamda_t = torch.tensor(lamda)
    if regularizer_name is "":
        return torch.tensor(0.0)
    elif regularizer_name == l1_str:
        return lamda_t * sum(param.abs().sum() for param in params)
    elif regularizer_name == l2_str:
        warn(
            f"use OptimizerConfigs.weight_decay, torch.optim.Optimizer uses it to implement L2 regularization"
        )
        return lamda_t * sum(param.pow(2).sum() for param in params)
    else:
        raise_not_implemented_error("regularizer", regularizer_name)
