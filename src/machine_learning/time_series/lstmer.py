from typing import Union

from keras.layers import LSTM, Dense
from keras.models import Sequential
from numpy import ndarray
from pydantic import BaseModel


class ModelConfigs(BaseModel):
    timesteps: int
    obs_as_timesteps: bool = True
    n_units: int
    n_lstm_layers: int
    stateful: bool = True
    batch_size: int = 100  # awkward model config


class TrainConfigs(BaseModel):
    loss_str: str = "mean_squared_error"
    optimizer_str: str = "adam"
    n_epochs: int = 100
    verbose: Union[int, str] = "auto"


def init_univariate_model(
    timesteps: int = 2,
    obs_as_timesteps: bool = True,
    n_units: int = 1,
    n_lstm_layers: int = 1,
    stateful: bool = True,
    batch_size: int = 100,
) -> Sequential:
    """
    initializes univariate model with some LSTM layers
    note: if stateful is True then batch_size must divide train_size
          so that all batches have the same size to keep memory/state
    :param timesteps: number of time steps
    :param obs_as_timesteps: whether each sequence is timesteps samples of 1 feature or 1 sample of timesteps features
    :param n_units: number of units
    :param n_lstm_layers: number of LSTM layers
    :param stateful: whether to keep memory/state from sample to sample in each batch during training
    :param batch_size: batch size
    :return: initial model
    """
    model = Sequential()
    model.timesteps = timesteps
    model.obs_as_timesteps = obs_as_timesteps
    model.n_units = n_units
    model.n_lstm_layers = n_lstm_layers
    model.stateful = stateful
    model.batch_size = batch_size

    if obs_as_timesteps is True:
        input_shape = (timesteps, 1)
    else:
        input_shape = (1, timesteps)
    if stateful is True:
        batch_input_shape = (batch_size, *input_shape)
        for i in range(n_lstm_layers):
            if i == n_lstm_layers - 1:
                lstm = LSTM(
                    n_units,
                    stateful=True,
                    batch_input_shape=batch_input_shape,
                    name=str(i),
                )
            else:
                lstm = LSTM(
                    n_units,
                    stateful=True,
                    batch_input_shape=batch_input_shape,
                    name=str(i),
                    return_sequences=True,
                )
            model.add(lstm)
    else:
        for i in range(n_lstm_layers):
            if i == n_lstm_layers - 1:
                lstm = LSTM(n_units, input_shape=input_shape, name=str(i))
            else:
                lstm = LSTM(
                    n_units, input_shape=input_shape, name=str(i), return_sequences=True
                )
            model.add(lstm)
    model.add(Dense(1))
    return model


def train(
    configs: TrainConfigs, model: Sequential, X_train: ndarray, Y_train: ndarray
) -> Sequential:
    """
    trains model to X_train, Y_train
    :param configs: training configs
    :param model: model
    :param X_train: train sequences, each of form [1, timesteps obs] or [timesteps obs, 1]
    :param Y_train: train targets
    :return: trained model
    """
    model.compile(loss=configs.loss_str, optimizer=configs.optimizer_str)
    shuffle = not model.stateful
    for _ in range(configs.n_epochs):
        model.fit(
            X_train,
            Y_train,
            batch_size=model.batch_size,
            shuffle=shuffle,
            verbose=configs.verbose,
        )
        model.reset_states()
    return model
