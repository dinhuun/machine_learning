from typing import List, Optional

import torch
from pydantic import BaseModel
from torch import Tensor
from torch.nn import BatchNorm1d, Dropout, Linear, Module, Sequential

from machine_learning.strings import (
    decoder_str,
    encoder_str,
    relu_str,
    sigmoid_str,
    state_dict_str,
)
from machine_learning.utils.utils_networks import init_activation


class FeedforwardCoderConfigs(BaseModel):
    """
    configs for Feedforward encoder, decoder
    """

    x_dim: int
    z_dim: int
    h_dims: Optional[List[int]]
    activation_name: str = relu_str
    batch_norm: bool = True
    dropout_prob: float = 0.0
    bias: bool = True


class FeedforwardAutoencoder(Module):
    """
    a feedforward autoencoder
    """

    def __init__(
        self,
        encoder_configs: FeedforwardCoderConfigs,
        encoder: Sequential,
        decoder_configs: FeedforwardCoderConfigs,
        decoder: Sequential,
    ):
        """
        initializes FeedforwardAutoencoder
        :param encoder_configs: encoder configs
        :param encoder: encoder
        :param decoder_configs: decoder configs
        :param decoder: decoder
        """
        super(FeedforwardAutoencoder, self).__init__()
        self.encoder_configs = encoder_configs
        self.encoder = encoder
        self.decoder_configs = decoder_configs
        self.decoder = decoder
        self.loss = 0

    def encode(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(z)

    def forward(self, x: Tensor) -> Tensor:
        return self.decode(self.encode(x))


def init_encoder(
    x_dim: int,
    z_dim: int,
    h_dims: Optional[List[int]] = None,
    activation_name: str = relu_str,
    batch_norm: bool = True,
    dropout_prob: float = 0.0,
    bias: bool = True,
) -> Sequential:
    """
    initializes FeedforwardAutoencoder encoder
    :param x_dim: input dim
    :param z_dim: latent dim
    :param h_dims: hidden dim
    :param activation_name: activation name
    :param batch_norm: whether to normalize batch
    :param dropout_prob: dropout probability
    :param bias: whether to use linear bias
    :return: encoder
    """
    modules: List[Module] = []
    if h_dims is None:
        modules.append(Linear(x_dim, z_dim, bias=bias))
        modules.append(init_activation(activation_name))
    else:
        # input layer
        modules.append(Linear(x_dim, h_dims[0], bias=bias))
        modules.append(init_activation(activation_name))
        # hidden layers
        for i in range(0, len(h_dims) - 1):
            if dropout_prob > 0:
                modules.append(Dropout(p=dropout_prob))
            modules.append(Linear(h_dims[i], h_dims[i + 1], bias=bias))
            modules.append(init_activation(activation_name))
            if batch_norm:
                modules.append(BatchNorm1d(h_dims[i + 1]))
        # output layer
        modules.append(Linear(h_dims[-1], z_dim, bias=bias))
        modules.append(init_activation(activation_name))
    encoder = Sequential(*modules)
    return encoder


def init_decoder(
    x_dim: int,
    z_dim: int,
    h_dims: Optional[List[int]] = None,
    activation_name: str = relu_str,
    batch_norm: bool = True,
    dropout_prob: float = 0.0,
    bias: bool = True,
) -> Sequential:
    """
    initializes FeedforwardAutoencoder decoder
    :param x_dim: input dim
    :param z_dim: latent dim
    :param h_dims: hidden dim
    :param activation_name: activation name
    :param batch_norm: whether to normalize batch
    :param dropout_prob: dropout probability
    :param bias: whether to use linear bias
    :return: decoder
    """
    modules: List[Module] = []
    if h_dims is None:
        modules.append(Linear(z_dim, x_dim, bias=bias))
        modules.append(init_activation(sigmoid_str))
    else:
        # input layer
        modules.append(Linear(z_dim, h_dims[-1], bias=bias))
        modules.append(init_activation(activation_name))
        # hidden layers
        for i in reversed(range(0, len(h_dims) - 1)):
            if dropout_prob > 0:
                modules.append(Dropout(p=dropout_prob))
            modules.append(Linear(h_dims[i + 1], h_dims[i], bias=bias))
            modules.append(init_activation(activation_name))
            if batch_norm:
                modules.append(BatchNorm1d(h_dims[i]))
        # output layer
        modules.append(Linear(h_dims[0], x_dim, bias=bias))
        modules.append(init_activation(sigmoid_str))
    decoder = Sequential(*modules)
    return decoder


def load(filepath: str) -> FeedforwardAutoencoder:
    """
    loads FeedforwardAutoencoder
    """
    checkpoint = torch.load(filepath)

    encoder_checkpoint = checkpoint[encoder_str]
    encoder_state_dict = encoder_checkpoint.pop(state_dict_str)
    encoder = init_encoder(**encoder_checkpoint)
    encoder.load_state_dict(encoder_state_dict)

    decoder_checkpoint = checkpoint[decoder_str]
    decoder_state_dict = decoder_checkpoint.pop(state_dict_str)
    decoder = init_decoder(**decoder_checkpoint)
    decoder.load_state_dict(decoder_state_dict)

    model = FeedforwardAutoencoder(
        FeedforwardCoderConfigs(**encoder_checkpoint),
        encoder,
        FeedforwardCoderConfigs(**decoder_checkpoint),
        decoder,
    )
    return model


def save(model: FeedforwardAutoencoder, filepath: str):
    """
    saves FeedforwardAutoencoder
    """
    encoder_checkpoint = model.encoder_configs.dict()
    encoder_checkpoint[state_dict_str] = model.encoder.state_dict()

    decoder_checkpoint = model.decoder_configs.dict()
    decoder_checkpoint[state_dict_str] = model.decoder.state_dict()

    checkpoint = {encoder_str: encoder_checkpoint, decoder_str: decoder_checkpoint}
    torch.save(checkpoint, filepath)
