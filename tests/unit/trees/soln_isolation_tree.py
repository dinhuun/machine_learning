import numpy as np
from numpy.random import multivariate_normal

max_depth = 5

mu_normal = np.array([0, 0])
cov_normal = np.array([[1, 0], [0, 1]])
mu_anomalous = np.array([3, 3])
cov_anomalous = np.array([[1, 0], [0, 1]])

train_size = 1000
test_size = 10
X_train_normal = multivariate_normal(mu_normal, cov_normal, train_size)
Y_train_normal = np.ones(train_size)
X_test_normal = multivariate_normal(mu_normal, cov_normal, test_size)
Y_test_normal = np.ones(test_size)
X_test_anomalous = multivariate_normal(mu_anomalous, cov_anomalous, test_size)
Y_test_anomalous = np.zeros(test_size)
