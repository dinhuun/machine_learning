import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch import from_numpy

from machine_learning.classes import OptimizerConfigs, TrainConfigs
from machine_learning.machines.factorization_machine import (
    FactorizationMachine,
    ModelConfigs,
    load,
    save,
    train,
)
from machine_learning.strings import sigmoid_str

n_samples = 2000
n_features = 5
centers = 2
random_state = 0
X, Y = make_blobs(n_samples, n_features, centers=centers, random_state=random_state)
X, Y = X.astype(np.float32), Y.astype(np.int32)

X_temp, X_test, Y_temp, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=random_state
)
X_train, X_val, Y_train, Y_val = train_test_split(
    X_temp, Y_temp, test_size=0.25, random_state=random_state
)
X_train_t = from_numpy(X_train)
Y_train_t = from_numpy(Y_train).view(-1, 1).float()
X_val_t = from_numpy(X_val)
Y_val_t = from_numpy(Y_val).view(-1, 1)
X_test_t = from_numpy(X_test)
Y_test_t = from_numpy(Y_test).view(-1, 1)

n_factors = 2
activation_name = sigmoid_str

model_configs = ModelConfigs(
    n_features=n_features,
    n_factors=n_factors,
    activation_name=activation_name,
)


lr = 0.01
optimizer_configs = OptimizerConfigs(lr=lr)

n_epochs = 1000
patience = 1000
train_configs = TrainConfigs(
    optimizer_configs=optimizer_configs,
    n_epochs=n_epochs,
    patience=patience,
)


def test_factorization_machine(tmp_path):
    # that it can initialize
    model = FactorizationMachine(**model_configs.dict())

    # that train() works
    best_model, best_score, best_epoch = train(
        train_configs, model, X_train_t, Y_train_t, X_val_t, Y_val_t
    )
    assert best_score > 0.0

    # that it beats random guess
    best_model.eval()
    Y_test_pred = best_model(X_test_t)
    Y_test_pred_np = Y_test_pred.detach().numpy()
    accuracy = accuracy_score(Y_test, Y_test_pred_np.astype(int))
    assert accuracy > 0.5

    # that save() and load() work
    tmp_dir = tmp_path / "tmp_dir"
    tmp_dir.mkdir()

    filepath = str(tmp_dir / "factorization_machine.pt")
    save(best_model, filepath)

    loaded_model = load(filepath)
    loaded_model.eval()
    Y_test_pred_loaded = loaded_model(X_test_t)
    Y_test_pred_loaded_np = Y_test_pred_loaded.detach().numpy()
    np.testing.assert_almost_equal(Y_test_pred_np, Y_test_pred_loaded_np)
