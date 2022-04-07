import numpy as np
import torch
from torch.distributions import MultivariateNormal

from machine_learning.flows.real_nvp import CouplingLayer, RealNVP, ScaleTranslate
from machine_learning.strings import sigmoid_str

x_dim = 2
h_dim = 4
z_dim = 2
activation_name = sigmoid_str
n_coupling_layers = 2

torch.manual_seed(0)
n = 10
m = 2
shape = (n, m)
x = torch.rand(shape)
z = torch.rand(shape)
mask = torch.arange(x_dim) % 2


def test_ScaleTranslate():
    """
    tests ScaleTranslate
    """
    # that it can initialize a scale, whose range is tanh (-1, 1)
    scale = ScaleTranslate(x_dim, h_dim, z_dim, activation_name, scale=True)
    s = scale(x).detach().numpy()
    assert np.all(s > -1)
    assert np.any(s < 0) or np.any(s > 0)
    assert np.all(s < 1)

    # that it can initialize a translate, whose range is sigmoid (0, 1)
    translate = ScaleTranslate(x_dim, h_dim, z_dim, activation_name, scale=False)
    t = translate(x).detach().numpy()
    assert np.all(t > 0)
    assert np.all(t < 1)


def test_CouplingLayer():
    """
    tests CouplingLayer
    """
    coupling_layer = CouplingLayer(x_dim, h_dim, z_dim, activation_name)
    # that its mask is as expected
    mask_np = coupling_layer.mask.numpy()
    expected_mask_np = mask.numpy()
    np.testing.assert_equal(mask_np, expected_mask_np)

    # that it forwards to a tuple as expected
    z_hat, log_det_J = coupling_layer.forward(x)
    assert z_hat.shape == (n, m)
    assert log_det_J.shape == (n,)

    # that it backwards as expected
    x_hat = coupling_layer.backward(z)
    assert x_hat.shape == (n, m)


def test_RealNVP():
    """
    tests RealNVP
    """
    # that it initializes as expected
    model = RealNVP(
        x_dim,
        h_dim,
        z_dim,
        activation_name,
        n_coupling_layers=n_coupling_layers,
    )
    assert len(model.coupling_layers) == n_coupling_layers
    assert model.loss == 0
    assert isinstance(model.p, MultivariateNormal)

    # that it forwards as expected
    z_hat = model(x)
    assert model.loss > 0
    assert z_hat.shape == (n, m)

    # that it backwards as expected
    x_hat = model(z)
    assert x_hat.shape == (n, m)
