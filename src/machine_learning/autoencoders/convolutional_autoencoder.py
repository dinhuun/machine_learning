import torch
from torch import Tensor
from torch.nn import Conv2d, ConvTranspose2d, MaxPool2d, Module, Sequential, Upsample

from machine_learning.strings import (
    activation_name_str,
    decoder_str,
    encoder_str,
    relu_str,
    same_str,
    sigmoid_str,
    state_dict_str,
)
from machine_learning.utils.utils_nn import init_activation


class ConvolutionalAutoencoder(Module):
    """
    a convolutional autoencoder
    """

    def __init__(
        self,
        encoder_activation_name: str,
        encoder: Sequential,
        decoder_activation_name: str,
        decoder: Sequential,
    ):
        """
        initializes ConvolutionalAutoencoder
        :param encoder_activation_name: encoder activation name
        :param encoder: encoder
        :param decoder_activation_name: decoder activation name
        :param decoder: decoder
        """
        super(ConvolutionalAutoencoder, self).__init__()
        self.encoder_activation_name = encoder_activation_name
        self.encoder = encoder
        self.decoder_activation_name = decoder_activation_name
        self.decoder = decoder
        self.loss = 0

    def encode(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(z)

    def forward(self, x: Tensor) -> Tensor:
        return self.decode(self.encode(x))


def init_encoder(activation_name: str = relu_str) -> Sequential:
    """
    initializes convolutional encoder in a rather specific way wrt to input size, kernel size, pool size, step size
    for dimensions to match
    :param activation_name: activation name
    :return: encoder
    """
    conv_kernel_size = 3
    pool_kernel_size = 2
    l_0 = Conv2d(1, 32, conv_kernel_size, padding=same_str)  # 32x28x28
    l_1 = init_activation(activation_name)  # 32x28x28
    l_2 = MaxPool2d(pool_kernel_size)  # 32x14x14
    l_3 = Conv2d(32, 16, conv_kernel_size, padding=same_str)  # 16x14x14
    l_4 = init_activation(activation_name)  # 16x14x14
    l_5 = MaxPool2d(pool_kernel_size)  # 16x7x7
    l_6 = Conv2d(16, 8, conv_kernel_size, padding=same_str)  # 8x7x7
    l_7 = init_activation(activation_name)  # 8x7x7
    l_8 = MaxPool2d(pool_kernel_size, ceil_mode=True)  # 8x4x4
    encoder = Sequential(l_0, l_1, l_2, l_3, l_4, l_5, l_6, l_7, l_8)
    return encoder


def init_decoder(activation_name: str = relu_str) -> Sequential:
    """
    initializes convolutional decoder in a rather specific way wrt to input size, kernel size, pool size, step size
    for dimensions to match
    :param activation_name: activation name
    :return: decoder
    """
    conv_kernel_size = 3
    pool_kernel_size = 2
    padding = int((conv_kernel_size - 1) / 2)
    l_0 = ConvTranspose2d(8, 16, conv_kernel_size, padding=padding)  # 16x4x4
    l_1 = init_activation(activation_name)  # 16x4x4
    l_2 = Upsample(scale_factor=pool_kernel_size)  # 16x8x8
    l_3 = ConvTranspose2d(16, 32, conv_kernel_size, padding=padding)  # 32x8x8
    l_4 = init_activation(activation_name)  # 32x8x8
    l_5 = Upsample(scale_factor=pool_kernel_size)  # 32x16x16
    l_6 = ConvTranspose2d(32, 1, conv_kernel_size, padding=padding + 1)  # 1x14x14
    l_7 = init_activation(activation_name)  # 1x14x14
    l_8 = Upsample(scale_factor=pool_kernel_size)  # 1x28x28
    l_9 = ConvTranspose2d(1, 1, conv_kernel_size, padding=padding)  # 1x28x28
    l_10 = init_activation(sigmoid_str)  # 1x28x28
    decoder = Sequential(l_0, l_1, l_2, l_3, l_4, l_5, l_6, l_7, l_8, l_9, l_10)
    return decoder


def load(filepath: str) -> ConvolutionalAutoencoder:
    """
    loads ConvolutionalAutoencoder
    """
    checkpoint = torch.load(filepath)

    encoder_checkpoint = checkpoint[encoder_str]
    encoder_activation_name = encoder_checkpoint[activation_name_str]
    encoder_state_dict = encoder_checkpoint[state_dict_str]
    encoder = init_encoder(encoder_activation_name)
    encoder.load_state_dict(encoder_state_dict)

    decoder_checkpoint = checkpoint[decoder_str]
    decoder_activation_name = decoder_checkpoint[activation_name_str]
    decoder_state_dict = decoder_checkpoint[state_dict_str]
    decoder = init_decoder(decoder_activation_name)
    decoder.load_state_dict(decoder_state_dict)

    model = ConvolutionalAutoencoder(
        encoder_activation_name, encoder, decoder_activation_name, decoder
    )
    return model


def save(model: ConvolutionalAutoencoder, filepath: str):
    """
    saves ConvolutionalAutoencoder
    """
    encoder_checkpoint = {
        activation_name_str: model.encoder_activation_name,
        state_dict_str: model.encoder.state_dict(),
    }
    decoder_checkpoint = {
        activation_name_str: model.decoder_activation_name,
        state_dict_str: model.decoder.state_dict(),
    }
    checkpoint = {encoder_str: encoder_checkpoint, decoder_str: decoder_checkpoint}
    torch.save(checkpoint, filepath)
