"""
from keras.io/guides/customizing_what_happens_in_fit
"""

from typing import Dict

import tensorflow as tf
from keras import Input, Model, Sequential
from keras.layers import (
    Conv2D,
    Conv2DTranspose,
    Dense,
    GlobalMaxPooling2D,
    LeakyReLU,
    Reshape,
)
from keras.losses import Loss
from keras.optimizers import Optimizer

from machine_learning.strings import (
    discriminator_str,
    generator_str,
    same_str,
    sigmoid_str,
)

X_dim = 28
Z_dim = 128


class GAN(Model):
    """
    a GAN to generate and discriminate MNIST images
    """

    def __init__(
        self,
        z_dim: int,
        generator: Sequential,
        discriminator: Sequential,
        optimizer_g: Optimizer,
        optimizer_d: Optimizer,
        loss_fn: Loss,
    ):
        """
        initializes GAN
        :param z_dim: latent dim
        :param generator: generator
        :param discriminator: discriminator
        :param optimizer_g: optimizer for generator
        :param optimizer_d: optimizer for discriminator
        :param loss_fn: loss function for both generator and discriminator
        """
        super(GAN, self).__init__()
        self.z_dim = z_dim
        self.generator = generator
        self.discriminator = discriminator
        self.optimizer_g = optimizer_g
        self.optimizer_d = optimizer_d
        self.loss_fn = loss_fn

    def train_step(self, X) -> Dict[str, float]:
        """
        customized train_step()
        :param X: train data
        :return: {"loss_d": discriminator loss, "loss_g": generator loss}
        """
        if isinstance(X, tuple):
            X = X[0]
        batch_size = tf.shape(X)[0]
        Y = tf.zeros((batch_size, 1))

        Z_d = tf.random.normal((batch_size, self.z_dim))
        X_hat_d = self.generator(Z_d)
        Y_hat_d = tf.ones((batch_size, 1))

        X_d = tf.concat([X_hat_d, X], axis=0)
        Y_d = tf.concat([Y_hat_d, Y], axis=0)
        Y_d += 0.05 * tf.random.uniform(tf.shape(Y_d))

        with tf.GradientTape() as tape:
            Y_d_pred = self.discriminator(X_d)
            loss_d = self.loss_fn(Y_d, Y_d_pred)
        grads_d = tape.gradient(loss_d, self.discriminator.trainable_weights)
        self.optimizer_d.apply_gradients(
            zip(grads_d, self.discriminator.trainable_weights)
        )

        Z_g = tf.random.normal((batch_size, self.z_dim))
        Y_g = tf.zeros((batch_size, 1))
        with tf.GradientTape() as tape:
            X_g = self.generator(Z_g)
            Y_g_pred = self.discriminator(X_g)
            loss_g = self.loss_fn(Y_g, Y_g_pred)
        grads_g = tape.gradient(loss_g, self.generator.trainable_weights)
        self.optimizer_g.apply_gradients(zip(grads_g, self.generator.trainable_weights))

        return {discriminator_str: loss_d, generator_str: loss_g}


def init_generator(
    z_dim: int = Z_dim, alpha: float = 0.2, activation_str: str = sigmoid_str
) -> Sequential:
    """
    initializes generator in a rather specific way wrt to input size, kernel size, pool size, step size
    :param z_dim: latent dim
    :param alpha: alpha for leaky relu layers
    :param activation_str: activation for last convolutional layer
    :return: initial generator
    """
    kernel_size = 7
    kernel_size_t = 4
    strides = (2, 2)
    layers = [
        Input((z_dim,)),
        Dense(kernel_size * kernel_size * z_dim),
        LeakyReLU(alpha),
        Reshape((kernel_size, kernel_size, z_dim)),
        Conv2DTranspose(
            z_dim, (kernel_size_t, kernel_size_t), strides=strides, padding=same_str
        ),
        LeakyReLU(alpha),
        Conv2DTranspose(
            z_dim, (kernel_size_t, kernel_size_t), strides=strides, padding=same_str
        ),
        LeakyReLU(alpha),
        Conv2D(
            1, (kernel_size, kernel_size), padding=same_str, activation=activation_str
        ),
    ]
    return Sequential(layers, name=generator_str)


def init_discriminator(z_dim: int = Z_dim, alpha: float = 0.2) -> Sequential:
    """
    initializes discriminator in a rather specific way wrt to input size, kernel size, pool size, step size
    :param z_dim: latent dim
    :param alpha: alpha for leaky relu layers
    :return: initial discriminator
    """
    kernel_size = 3
    strides = (2, 2)
    layers = [
        Input((X_dim, X_dim, 1)),
        Conv2D(64, (kernel_size, kernel_size), strides=strides, padding=same_str),
        LeakyReLU(alpha),
        Conv2D(z_dim, (kernel_size, kernel_size), strides=strides, padding=same_str),
        LeakyReLU(alpha),
        GlobalMaxPooling2D(),
        Dense(1),
    ]
    return Sequential(layers, name=discriminator_str)
