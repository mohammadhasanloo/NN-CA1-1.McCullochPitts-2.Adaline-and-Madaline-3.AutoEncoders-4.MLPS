"""Adaline: the adaptive linear neuron, trained by the Widrow-Hoff rule."""

from __future__ import annotations

import numpy as np


class Adaline:
    """A linear unit trained on the error before thresholding, not after.

    That is the whole difference from the perceptron. The perceptron updates on
    the sign of its output, so it learns nothing from a sample it happens to
    classify correctly by a hair. Adaline updates on the raw distance between the
    linear output and the target, which gives a smooth squared-error surface and
    a gradient everywhere on it.

    Being linear, it can only separate classes a straight line can separate.
    """

    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate
        self.weights: np.ndarray | None = None
        self.bias = 0.0

    def net_input(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=float) @ self.weights + self.bias

    def fit(self, x, y, epochs: int = 20) -> list[float]:
        """Online updates, one sample at a time. Returns loss per epoch."""
        x = np.atleast_2d(np.asarray(x, dtype=float))
        y = np.asarray(y, dtype=float).reshape(-1)
        if len(x) != len(y):
            raise ValueError(f"{len(x)} samples against {len(y)} labels")

        self.weights = np.zeros(x.shape[1])
        self.bias = 0.0

        history = []
        for _ in range(epochs):
            for sample, target in zip(x, y):
                error = target - (sample @ self.weights + self.bias)
                self.weights += self.learning_rate * error * sample
                self.bias += self.learning_rate * error
            residual = y - self.net_input(x)
            history.append(float((residual**2).sum() / 2))
        return history

    def predict(self, x) -> np.ndarray:
        """Bipolar output: +1 or -1."""
        if self.weights is None:
            raise RuntimeError("the model has not been fitted")
        return np.where(self.net_input(x) >= 0, 1, -1)

    def score(self, x, y) -> float:
        return float((self.predict(x) == np.asarray(y).reshape(-1)).mean())
