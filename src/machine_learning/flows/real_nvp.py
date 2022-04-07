from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.distributions import MultivariateNormal
from torch.distributions.distribution import Distribution
from torch.nn import Linear, Module, ModuleList, Sequential, Tanh

from machine_learning.strings import leaky_relu_str
from machine_learning.utils.utils_nn import init_activation


class ScaleTranslate(Module):
    """
    either the scale map or the translate map in CouplingLayer
    """

    def __init__(
        self,
        x_dim: int,
        h_dim: int,
        z_dim: int,
        activation_name: str = leaky_relu_str,
        negative_slope: float = 0.01,
        scale: bool = False,
    ):
        """
        initializes ScaleTranslate
        :param x_dim: input dim
        :param h_dim: hidden dim
        :param z_dim: latent dim
        :param activation_name: activation name
        :param negative_slope: negative slope, if activation name is "leaky_relu"
        :param scale: whether this is the scale map or the translate map
        """
        super(ScaleTranslate, self).__init__()
        layers = [
            Linear(x_dim, h_dim),
            init_activation(activation_name, negative_slope),
            Linear(h_dim, h_dim),
            init_activation(activation_name, negative_slope),
            Linear(h_dim, z_dim),
        ]
        if scale:
            layers.append(Tanh())
        else:
            layers.append(init_activation(activation_name, negative_slope))
        self.model = Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """
        maps x to its image
        :param x: input
        :return: output
        """
        return self.model(x)


class CouplingLayer(Module):
    """
    coupling layer in RealNVP
    """

    def __init__(
        self,
        x_dim: int,
        h_dim: int,
        z_dim: int,
        activation_name: str = leaky_relu_str,
        negative_slope: float = 0.01,
        mask: Optional[Tensor] = None,
    ):
        """
        initializes CouplingLayer
        :param x_dim: input dim
        :param h_dim: hidden dim
        :param z_dim: latent dim
        :param activation_name: activation name
        :param negative_slope: negative slope, if activation name is "leaky_relu"
        :param mask: mask in RealNVP
        """
        super(CouplingLayer, self).__init__()
        self.scale = ScaleTranslate(
            x_dim, h_dim, z_dim, activation_name, negative_slope, True
        )
        self.translate = ScaleTranslate(
            x_dim, h_dim, z_dim, activation_name, negative_slope, False
        )

        if mask is None:
            mask = torch.arange(x_dim) % 2
        self.mask = mask

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        maps x through
        :param x: samples
        :return: images and log of determinant of Jacobian matrix
        """
        x_masked = self.mask * x
        s = self.scale(x_masked)
        t = self.translate(x_masked)
        z = x_masked + (1 - self.mask) * (x - t) * torch.exp(-s)
        log_det_J = -torch.sum((1 - self.mask) * s, 1)
        return z, log_det_J

    def backward(self, z: Tensor) -> Tensor:
        """
        maps z through
        :param z: images
        :return: samples
        """
        z_masked = self.mask * z
        s = self.scale(z_masked)
        t = self.translate(z_masked)
        x = z_masked + (1 - self.mask) * (z * torch.exp(s) + t)
        return x


class RealNVP(Module):
    """
    real NVP model in Density estimation using Real NVP by Laurent Dinh, Jascha Sohl-Dickstein, Samy Bengio
    """

    def __init__(
        self,
        x_dim: int,
        h_dim: int,
        z_dim: int,
        activation_name: str = leaky_relu_str,
        negative_slope: float = 0.01,
        n_coupling_layers: int = 2,
        p: Optional[Distribution] = None,
    ):
        """
        initializes RealNVP
        :param x_dim: input dim
        :param h_dim: hidden dim
        :param z_dim: latent dim
        :param activation_name: activation name
        :param negative_slope: negative slope, if activation name is "leaky_relu"
        :param n_coupling_layers: number of coupling layers
        :param p: latent distribution
        """
        super(RealNVP, self).__init__()
        mask = torch.arange(x_dim) % 2
        self.coupling_layers = ModuleList()
        for _ in range(n_coupling_layers):
            self.coupling_layers.append(
                CouplingLayer(
                    x_dim, h_dim, z_dim, activation_name, negative_slope, mask
                )
            )
            mask = 1 - mask
        self.loss = 0

        if p is None:
            p = MultivariateNormal(torch.zeros(z_dim), torch.eye(z_dim))
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        """
        maps samples to its latent samples
        :param x: samples
        :return: latent samples
        """
        sum_log_det_J = x.new_zeros(x.shape[0])
        for layer in self.coupling_layers:
            x, log_det_J = layer.forward(x)
            sum_log_det_J += log_det_J
        z = x
        self.loss = (-self.p.log_prob(z) - sum_log_det_J).mean()
        return z

    def backward(self, z: Tensor) -> Tensor:
        """
        maps latent samples to its samples
        :param z: latent samples
        :return: samples
        """
        for layer in reversed(self.coupling_layers):
            z = layer.backward(z)
        x = z
        return x
