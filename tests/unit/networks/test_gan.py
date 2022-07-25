import tensorflow as tf
from tensorflow import keras

from machine_learning.networks.gan import (
    GAN,
    X_dim,
    Z_dim,
    init_discriminator,
    init_generator,
)
from machine_learning.strings import discriminator_str, generator_str

batch_size = 10
X = tf.random.normal((batch_size, X_dim, X_dim, 1))
Z = tf.random.normal((batch_size, Z_dim))


def test_GAN():
    """
    tests GAN
    """
    # that it initializes
    generator = init_generator()
    discriminator = init_discriminator()
    optimizer_g = keras.optimizers.Adam(learning_rate=0.0003)
    optimizer_d = keras.optimizers.Adam(learning_rate=0.0003)
    loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)
    model = GAN(Z_dim, generator, discriminator, optimizer_g, optimizer_d, loss_fn)

    # that it compiles
    model.compile()

    # that it takes train step
    losses = model.train_step(X)
    assert losses[generator_str] > 0
    assert losses[discriminator_str] > 0


def test_init_discriminator():
    """
    tests init_discriminator()
    """
    discriminator = init_discriminator()
    scores = discriminator(X)
    assert scores.shape == (batch_size, 1)


def test_init_generator():
    """
    tests init_generator()
    """
    generator = init_generator()
    X_hat = generator(Z)
    assert X_hat.shape == (batch_size, 28, 28, 1)
