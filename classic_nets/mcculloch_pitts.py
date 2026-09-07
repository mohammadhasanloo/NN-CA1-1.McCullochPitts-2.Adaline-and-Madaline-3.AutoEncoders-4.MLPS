"""The McCulloch-Pitts neuron: the 1943 threshold unit.

No learning at all. Weights and a threshold are chosen by hand, and the unit
fires when the weighted sum reaches the threshold. It is the smallest thing that
computes a logical function, and a network of them computes any of them.
"""

from __future__ import annotations

import numpy as np


class ThresholdUnit:
    """One or more threshold neurons sharing an input.

    ``weights`` is (inputs, units), so a single call evaluates every unit at once
    and a layer is the same object as a neuron.
    """

    def __init__(self, weights, threshold: float = 1.0):
        self.weights = np.atleast_2d(np.asarray(weights, dtype=float))
        if self.weights.shape[0] == 1 and self.weights.shape[1] > 1:
            self.weights = self.weights.T
        self.threshold = float(threshold)

    @property
    def n_inputs(self) -> int:
        return self.weights.shape[0]

    @property
    def n_units(self) -> int:
        return self.weights.shape[1]

    def __call__(self, x) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x, dtype=float))
        if x.shape[1] != self.n_inputs:
            raise ValueError(f"expected {self.n_inputs} inputs, got {x.shape[1]}")
        return (x @ self.weights >= self.threshold).astype(int)

    def __repr__(self) -> str:
        return f"ThresholdUnit({self.n_inputs} -> {self.n_units}, threshold={self.threshold})"


def logical_and(n_inputs: int = 2) -> ThresholdUnit:
    """Fires only when every input is on."""
    return ThresholdUnit(np.ones((n_inputs, 1)), threshold=n_inputs)


def logical_or(n_inputs: int = 2) -> ThresholdUnit:
    """Fires when any input is on."""
    return ThresholdUnit(np.ones((n_inputs, 1)), threshold=1)


def logical_not() -> ThresholdUnit:
    """Inverts a single input.

    A negative weight and a threshold of zero: with the input off the sum is zero
    and reaches the threshold, with it on the sum is negative and does not.
    """
    return ThresholdUnit([[-1.0]], threshold=0)
