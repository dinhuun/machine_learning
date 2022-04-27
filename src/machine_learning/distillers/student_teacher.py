from typing import Any, Dict, List, Tuple

from keras import Model
from numpy import ndarray
from tensorflow import GradientTape
from tensorflow.keras import Input, Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, LeakyReLU, MaxPooling2D
from tensorflow.keras.losses import Loss
from tensorflow.keras.optimizers import Optimizer
from tensorflow.nn import softmax


def init_student(
    input_shape: Tuple[int, int, int] = (32, 32, 3),
    output_dim: int = 10,
    alpha: float = 0.2,
) -> Sequential:
    """
    initializes convolutional model in a rather specific way wrt input size, kernel size, pool size, step size
    for dimension to match
    :param input_shape: input shape
    :param output_dim: output dim
    :param alpha: LeakyReLU alpha
    :return: model
    """
    conv_size = (3, 3)
    conv_strides = (2, 2)
    pool_size = (2, 2)
    pool_strides = (1, 1)
    layers = [
        Input(input_shape),
        Conv2D(64, conv_size, conv_strides, "same"),
        LeakyReLU(alpha),
        MaxPooling2D(pool_size, pool_strides, "same"),
        Conv2D(256, conv_size, conv_strides, "same"),
        LeakyReLU(alpha),
        MaxPooling2D(pool_size, pool_strides, "same"),
        Flatten(),
        Dense(output_dim),
    ]
    student = Sequential(layers, name="student")
    return student


def init_teacher(
    input_shape: Tuple[int, int, int] = (32, 32, 3),
    output_dim: int = 10,
    alpha: float = 0.2,
) -> Sequential:
    """
    initializes convolutional model in a rather specific way wrt input size, kernel size, pool size, step size
    for dimension to match
    :param input_shape: input shape
    :param output_dim: output dim
    :param alpha: LeakyReLU alpha
    :return: model
    """
    conv_size = (3, 3)
    conv_strides = (2, 2)
    pool_size = (2, 2)
    pool_strides = (1, 1)
    layers = [
        Input(input_shape),
        Conv2D(256, conv_size, conv_strides, "same"),
        LeakyReLU(alpha),
        MaxPooling2D(pool_size, pool_strides, "same"),
        Conv2D(512, conv_size, conv_strides, "same"),
        LeakyReLU(alpha),
        MaxPooling2D(pool_size, pool_strides, "same"),
        Flatten(),
        Dense(output_dim),
    ]
    teacher = Sequential(layers, name="teacher")
    return teacher


class StudentTeacher(Model):  # noqa
    """
    student-teacher model that uses larger teacher model to teach smaller student model
    attributes
        * student: to-be-trained smaller student model
        * teacher: pretrained larger teacher model
        * student_loss_fn: student loss function for difference between student prediction and ground truth,
                           such as sparse categorical cross entropy
        * distill_loss_fn: distillation loss function for difference between student soft prediction and
                           teacher soft prediction, such as KL divergence
        * alpha: weight for loss = self.alpha * student_loss + (1 - self.alpha) * distill_loss
        * temp: temperature for
            * student softmax function
            * teacher softmax function
            * gradient transformation gradients = [gradient * (self.temp ** 2) for gradient in gradients]
    """

    def __init__(
        self,
        student: Sequential,
        teacher: Sequential,
        student_loss_fn: Loss,
        distill_loss_fn: Loss,
        alpha: float = 0.3,
        temp: int = 7,
    ):
        """
        initializes StudentTeacher
        :param student: student model
        :param teacher: teacher model
        :param student_loss_fn: student loss function for difference between student prediction and ground truth,
                                such as sparse categorical cross entropy
        :param distill_loss_fn: distillation loss function for difference between student soft prediction and
                                teacher soft prediction, such as KL divergence
        :param alpha: weight for loss = alpha * student_loss + (1 - alpha) * distill_loss
        :param temp: temperature for
                        * student softmax function
                        * teacher softmax function
                        * gradient transformation gradients = [gradient * (self.temp ** 2) for gradient in gradients]
        """
        super(StudentTeacher, self).__init__()
        self.student = student
        self.teacher = teacher
        self.student_loss_fn = student_loss_fn
        self.distill_loss_fn = distill_loss_fn
        self.alpha = alpha
        self.temp = temp

    def compile(self, optimizer: Optimizer, metrics: List[Any]):  # noqa
        """
        implements Model.compile()
        :param optimizer: optimizer such as tensorflow.keras.optimizers.Adam()
        :param metrics: list of metrics such as keras.metrics.SparseCategoricalAccuracy()
        """
        super(StudentTeacher, self).compile(optimizer=optimizer, metrics=metrics)

    def train_step(self, data: Tuple[ndarray, ndarray]) -> Dict[str, float]:
        """
        implements Model.train_step()
        """
        X, Y = data
        teacher_Y_pred = self.teacher(X, training=False)
        teacher_softmax = softmax(teacher_Y_pred / self.temp, axis=1)

        with GradientTape() as tape:
            student_Y_pred = self.student(X, training=True)
            student_softmax = softmax(student_Y_pred / self.temp, axis=1)

            student_loss = self.student_loss_fn(Y, student_Y_pred)
            distill_loss = self.distill_loss_fn(teacher_softmax, student_softmax)
            loss = self.alpha * student_loss + (1 - self.alpha) * distill_loss

        trainable_variables = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))
        self.compiled_metrics.update_state(Y, student_Y_pred)

        results = {metric.name: metric.result() for metric in self.metrics}
        results["student_loss"] = student_loss
        results["distill_loss"] = distill_loss
        return results

    def test_step(self, data: Tuple[ndarray, ndarray]) -> Dict[str, float]:
        """
        implements Model.test_step()
        """
        X, Y = data
        student_Y_pred = self.student(X, training=False)
        student_loss = self.student_loss_fn(Y, student_Y_pred)
        self.compiled_metrics.update_state(Y, student_Y_pred)
        results = {metric.name: metric.result() for metric in self.metrics}
        results["student_loss"] = student_loss
        return results
