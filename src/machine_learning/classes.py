from pydantic import BaseModel
from torch import Tensor
from torch.nn import Linear, Module, ReLU, Softmax

from machine_learning.strings import adam_str


class OptimizerConfigs(BaseModel):
    alpha: float = 0.99  # smoothing factor
    eps: float = 1e-08  # denominator summand for numerical stability
    lamda: float = 0.0  # regularizer coefficient
    lr: float = 0.001  # learning rate
    momentum: float = 0.01  # momentum factor
    optimizer_name: str = adam_str
    regularizer_name: str = ""
    weight_decay: float = 0.0  # L2 coefficient


class Person:
    def __init__(self, age):
        self.age = age


class TinyModel(Module):
    """
    a tiny PyTorch model for tests
    """

    def __init__(self):
        """
        initializes TinyModel
        """
        super(TinyModel, self).__init__()
        self.linear_0 = Linear(10, 20)
        self.activation = ReLU()
        self.linear_1 = Linear(20, 10)
        self.softmax = Softmax()

    def forward(self, x: Tensor) -> Tensor:
        """
        forwards input to output
        :param x: input
        :return: output
        """
        x = self.linear_0(x)
        x = self.activation(x)
        x = self.linear_1(x)
        x = self.softmax(x)
        return x


class TrainConfigs(BaseModel):
    optimizer_configs: OptimizerConfigs = OptimizerConfigs()
    n_epochs: int = 100
    patience: int = 100
