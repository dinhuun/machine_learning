import torch
from torch import Tensor
from torch.distributions import Normal
from torch.nn import Linear, Module, Sequential


class VariationalAutoEncoder(Module):
    """
    a variational autoencoder
    """

    def __init__(
        self,
        encoder_activation_name: str,
        encoder_: Sequential,
        decoder_activation_name: str,
        decoder_: Sequential,
        z_dim: int,
    ):
        """
        initializes VariationalAutoEncoder with
            * encoder_ from autoencoder.convolutional_autoencoder.init_encoder()
            * decoder_ from autoencoder.convolutional_autoencoder.init_decoder()
        :param encoder_activation_name: encoder activation name
        :param encoder_: almost encoder, there is still a sampling step to get actual latent vector after encoder_()
        :param decoder_activation_name: decoder activation name
        :param decoder_: almost decoder, there is still a dense layer to decode actual latent vector before decoder_()
        :param z_dim: dimension of latent space
        """
        super(VariationalAutoEncoder, self).__init__()
        self.encoder_activation_name = encoder_activation_name
        self.encoder_ = encoder_
        self.decoder_activation_name = decoder_activation_name
        self.decoder_ = decoder_
        self.z_dim = z_dim
        self.layer_mu = Linear(4, z_dim)
        self.layer_sigma = Linear(4, z_dim)
        self.layer_z = Linear(z_dim, 4)
        self.p = Normal(0, 1)
        self.loss = 0

    def encode(self, x: Tensor) -> Tensor:
        w = self.encoder_(x)
        mu = self.layer_mu(w)
        sigma = torch.exp(self.layer_sigma(w))
        self.loss = (sigma**2 + mu**2 - torch.log(sigma) - 1 / 2).sum()
        z = mu + sigma * self.p.sample(mu.shape)
        return z

    def decode(self, z: Tensor):
        w = self.layer_z(z)  # z to w
        x = self.decoder_(w)
        return x

    def forward(self, x):
        return self.decode(self.encode(x))
