from typing import Tuple, Union

from matplotlib import pyplot as plt
from numpy import ndarray
from torch import Tensor


def plot_mnist_images(
    images: Union[ndarray, Tensor], figsize: Tuple[int, int] = (12, 8)
):
    """
    plots (:, 1, 28, 28) arrays/tensors or (:, 784) arrays/tensors as (28, 28) images
    """
    plt.figure(figsize=figsize)
    for i, image in enumerate(images):
        ax = plt.subplot(1, len(images), i + 1)
        plt.imshow(image.reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()
