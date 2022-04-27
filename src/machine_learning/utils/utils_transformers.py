import numpy as np
from numpy import ndarray


def get_angles(j_s: ndarray, i_s: ndarray, z_dim: int) -> ndarray:
    """
    gets angle for each sample index and each latent feature
    :param j_s: sample indices
    :param i_s: latent feature indices
    :param z_dim: latent dim
    :return: angles
    """
    angle_rates = 1 / np.power(10000, (2 * (i_s // 2)) / z_dim)
    return j_s * angle_rates


def encode_positions(n_samples: int, z_dim: int) -> ndarray:
    """
    encodes each position j in range(n_samples) as a z-dim latent vector
    :param n_samples: number of samples
    :param z_dim: latent dim
    :return: z-dim latent vectors
    """
    Z = get_angles(
        np.arange(n_samples)[:, np.newaxis], np.arange(z_dim)[np.newaxis, :], z_dim
    )
    # apply sin() to even encoding features
    Z[:, 0::2] = np.sin(Z[:, 0::2])
    # apply cos() to odd encoding features
    Z[:, 1::2] = np.cos(Z[:, 1::2])
    return Z
