import pytest
from torch.nn import (
    BCELoss,
    CrossEntropyLoss,
    Identity,
    KLDivLoss,
    L1Loss,
    LeakyReLU,
    MSELoss,
    NLLLoss,
    ReLU,
    Sigmoid,
    SiLU,
    Softmax,
    Tanh,
)
from torch.optim import SGD, Adam, AdamW, RMSprop

from machine_learning.classes import TinyModel
from machine_learning.strings import (
    adam_str,
    adamw_str,
    bce_loss_str,
    ce_loss_str,
    identity_str,
    kl_div_loss_str,
    l1_loss_str,
    leaky_relu_str,
    mse_loss_str,
    nll_loss_str,
    not_implemented_str,
    relu_str,
    rmsprop_str,
    sgd_str,
    sigmoid_str,
    softmax_str,
    swish_str,
    tanh_str,
)
from machine_learning.utils.utils_networks import (
    init_activation,
    init_loss,
    init_optimizer,
)


def test_init_activation():
    """
    tests init_activation()
    """
    assert isinstance(init_activation(identity_str), Identity)
    assert isinstance(init_activation(leaky_relu_str), LeakyReLU)
    assert isinstance(init_activation(relu_str), ReLU)
    assert isinstance(init_activation(sigmoid_str), Sigmoid)
    assert isinstance(init_activation(softmax_str), Softmax)
    assert isinstance(init_activation(swish_str), SiLU)
    assert isinstance(init_activation(tanh_str), Tanh)
    with pytest.raises(NotImplementedError):
        init_activation(not_implemented_str)


def test_init_loss():
    """
    tests init_loss()
    """
    assert isinstance(init_loss(bce_loss_str), BCELoss)
    assert isinstance(init_loss(ce_loss_str), CrossEntropyLoss)
    assert isinstance(init_loss(kl_div_loss_str), KLDivLoss)
    assert isinstance(init_loss(l1_loss_str), L1Loss)
    assert isinstance(init_loss(mse_loss_str), MSELoss)
    assert isinstance(init_loss(nll_loss_str), NLLLoss)
    with pytest.raises(NotImplementedError):
        init_loss(not_implemented_str)


def test_init_optimizer():
    """
    tests init_optimizer()
    """
    model = TinyModel()
    assert isinstance(init_optimizer(sgd_str, model.parameters()), SGD)
    assert isinstance(init_optimizer(adam_str, model.parameters()), Adam)
    assert isinstance(init_optimizer(adamw_str, model.parameters()), AdamW)
    assert isinstance(init_optimizer(rmsprop_str, model.parameters()), RMSprop)
    with pytest.raises(NotImplementedError):
        init_optimizer(not_implemented_str, model.parameters())
