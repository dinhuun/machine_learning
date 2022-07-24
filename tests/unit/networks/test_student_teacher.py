import numpy as np
import tensorflow as tf
from keras.utils.layer_utils import count_params
from tensorflow import keras

from machine_learning.networks.student_teacher import (
    StudentTeacher,
    init_student,
    init_teacher,
)

n_samples = 100
sample_size = (32, 32, 3)
X_train = tf.convert_to_tensor(np.random.rand(n_samples, *sample_size))
Y_train = tf.convert_to_tensor(np.random.rand(n_samples, 1))


def test_init_student():
    """
    tests init_student()
    """
    # that it initializes model
    model = init_student()

    # that model has expected number of trainable weights and nontrainable weights
    assert count_params(model.trainable_weights) == 313354
    assert count_params(model.non_trainable_weights) == 0


def test_init_teacher():
    """
    tests init_teacher()
    """
    # that it initializes model
    model = init_teacher()

    # that model has expected number of trainable weights and nontrainable weights
    assert count_params(model.trainable_weights) == 1515018
    assert count_params(model.non_trainable_weights) == 0


def test_StudentTeacher():
    """
    tests StudentTeacher
    """
    # that it initializes
    student = init_student()
    teacher = init_teacher()
    student_loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    distill_loss_fn = keras.losses.KLDivergence()
    model = StudentTeacher(student, teacher, student_loss_fn, distill_loss_fn)

    # that it compiles
    model.compile(
        tf.keras.optimizers.Adam(), metrics=[keras.metrics.SparseCategoricalAccuracy()]
    )

    # that it takes train step
    train_results = model.train_step((X_train, Y_train))
    assert train_results["sparse_categorical_accuracy"] == 0.0
    assert train_results["student_loss"] > 0.0
    assert train_results["distill_loss"] > 0.0

    # that it takes test_step
    test_results = model.test_step((X_train, Y_train))
    assert test_results["sparse_categorical_accuracy"] == 0.0
    assert test_results["student_loss"] > 0.0
