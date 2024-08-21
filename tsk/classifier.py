from typing import Any

import numpy as np
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression

from tsk.clustering import CMeansClustering

from scipy.special import expit  # Sigmoid function


def compute_firing_level(data: np.ndarray, centers: int, delta: float) -> np.ndarray:
    """
    Compute firing strength using Gaussian model

    :param data: n_Samples * n_Features
    :param centers: data center，n_Clusters * n_Features
    :param delta: variance of each feature， n_Clusters * n_Features
    :return: firing strength
    """
    d = -(np.expand_dims(data, axis=2) - np.expand_dims(centers.T, axis=0)) ** 2 / (2 * delta.T)
    d = np.exp(np.sum(d, axis=1))
    d = np.fmax(d, np.finfo(np.float64).eps)
    return d / np.sum(d, axis=1, keepdims=True)


def apply_firing_level(x: np.ndarray, firing_levels: np.ndarray, order: int = 1) -> np.ndarray:
    """
    Convert raw input to tsk input, based on the provided firing levels

    :param x: (np.ndarray) Raw input
    :param firing_levels: (np.ndarray) Firing level for each rule
    :param order: (int) TSK order. Valid values are 0 and 1
    :return:
    """
    if order == 0:
        return firing_levels
    else:
        n = x.shape[0]
        firing_levels = np.expand_dims(firing_levels, axis=1)
        x = np.expand_dims(np.concatenate((x, np.ones([n, 1])), axis=1), axis=2)
        x = np.repeat(x, repeats=firing_levels.shape[1], axis=2)
        output = x * firing_levels
        output = output.reshape([n, -1])

        return output

class Classifier:
    def __init__(self, c: float = 1., max_iters: int = 10000, n_cluster: int = 2, order: int = 1):
        """
        Fuzzy classifier class for binary classification

        :param c: (float) c-coefficient for linear regressor estimator
        :param max_iters: (int) max iters for logistic regression fitting
        :param n_cluster: (int) Number of clusters
        :param order: (int) Order of the method. Valid values are 0 or 1
        """
        self._c = c
        self._max_iters = max_iters
        self._n_clusters = n_cluster
        self._order = order

        self._center = None
        self._variance = None
        self._regression = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> Any:
        """
        Fit a set of measurements to the model
        :param x: (np.ndarray) Vector with inputs
        :param y: (np.array) Vector with measurements (binary labels)
        :return: (Classifier) self
        """
        cluster = CMeansClustering(self._n_clusters).fit(x, y)

        self._center = cluster.center
        self._variance = cluster.delta

        mu_a = compute_firing_level(x, self._center, self._variance)
        computed_input = apply_firing_level(x, mu_a, self._order)

        # Logistic regression for binary classification
        self._regression = LogisticRegression(C=self._c, max_iter=self._max_iters)
        self._regression.fit(computed_input, y)

        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict to which class a given set of inputs belongs (binary classification)

        :param x: (np.ndarray) Input vector
        :return: (np.ndarray) Output vector with predicted classes (0 or 1)
        """
        firing_levels = compute_firing_level(x, self._center, self._variance)
        computed_input = apply_firing_level(x, firing_levels, self._order)

        # Decision function gives logit values
        logits = self._regression.decision_function(computed_input)

        # For binary classification, use sigmoid function and threshold at 0.5
        probabilities = expit(logits)  # Sigmoid function
        return (probabilities > 0.5).astype(int)  # Return 0 or 1 based on threshold