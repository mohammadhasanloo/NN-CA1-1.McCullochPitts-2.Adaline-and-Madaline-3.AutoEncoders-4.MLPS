"""Madaline: many Adalines with a fixed vote, trained by the MRI rule."""

from __future__ import annotations

import numpy as np


def sign(x):
    """Bipolar step. Zero counts as on, matching the Adaline convention."""
    return np.where(np.asarray(x) >= 0, 1, -1)


class Madaline:
    """A hidden layer of Adalines whose outputs are combined by a fixed OR.

    Only the hidden layer learns. The output weights are set so the unit fires
    when any hidden unit does, which is what lets the network carve out a region
    from several half-planes and separate classes no single line can.

    Training is the MRI rule. When the output is wrong, only the hidden units
    responsible are adjusted, and only just enough to flip them: for a missed
    positive that is the single unit closest to its own threshold, since it is
    the cheapest one to change.
    """

    def __init__(self, hidden_units: int = 3, learning_rate: float = 0.9,
                 vote: str = "and", max_epochs: int = 300,
                 rng: np.random.Generator | None = None):
        if hidden_units < 1:
            raise ValueError("need at least one hidden unit")
        if vote not in ("and", "or"):
            raise ValueError(f"vote must be 'and' or 'or', got {vote!r}")
        self.hidden_units = hidden_units
        self.vote = vote
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.rng = rng or np.random.default_rng()
        self.hidden_weights: np.ndarray | None = None
        self.epochs_used = 0

    def _initialise(self, n_features: int) -> None:
        self.hidden_weights = (self.rng.random((n_features, self.hidden_units)) - 0.5) / 2
        self.hidden_bias = (self.rng.random(self.hidden_units) - 0.5) / 2
        # Fixed output layer. The bias decides whether the vote is an AND or an OR.
        self.output_weights = np.ones(self.hidden_units) / self.hidden_units
        magnitude = (self.hidden_units - 1) / self.hidden_units
        self.output_bias = -magnitude if self.vote == "and" else magnitude

    def _hidden_input(self, sample: np.ndarray) -> np.ndarray:
        return sample @ self.hidden_weights + self.hidden_bias

    def forward(self, x) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x, dtype=float))
        hidden = sign(x @ self.hidden_weights + self.hidden_bias)
        return sign(hidden @ self.output_weights + self.output_bias)

    def fit(self, x, y) -> list[int]:
        """Train until no weight changes, or the epoch cap. Returns errors per epoch."""
        x = np.atleast_2d(np.asarray(x, dtype=float))
        y = np.asarray(y, dtype=float).reshape(-1)
        if len(x) != len(y):
            raise ValueError(f"{len(x)} samples against {len(y)} labels")

        self._initialise(x.shape[1])
        history = []

        for epoch in range(self.max_epochs):
            changed = False
            errors = 0
            for sample, target in zip(x, y):
                hidden_input = self._hidden_input(sample)
                hidden = sign(hidden_input)
                output = sign(hidden @ self.output_weights + self.output_bias)

                if output == target:
                    continue
                errors += 1
                changed = True

                # Which units are at fault depends on the vote. Where a single
                # unit must flip, the cheapest is the one nearest its threshold.
                if self.vote == "and":
                    if target == 1:
                        for k in np.where(hidden_input < 0)[0]:
                            self._nudge(int(k), sample, hidden_input[k], towards=1)
                    else:
                        k = int(np.argmin(np.abs(hidden_input)))
                        self._nudge(k, sample, hidden_input[k], towards=-1)
                else:
                    if target == 1:
                        k = int(np.argmin(np.abs(hidden_input)))
                        self._nudge(k, sample, hidden_input[k], towards=1)
                    else:
                        for k in np.where(hidden_input > 0)[0]:
                            self._nudge(int(k), sample, hidden_input[k], towards=-1)

            history.append(errors)
            self.epochs_used = epoch + 1
            if not changed:
                break
        return history

    def _nudge(self, unit: int, sample: np.ndarray, activation: float, towards: int) -> None:
        """Move one hidden unit's activation towards a target value.

        The step is divided by the squared norm of the augmented input. Without
        that normalisation the correction scales with the size of the input, so
        on unscaled data a large sample produces a large weight change, which
        produces a larger activation and a larger correction again. The weights
        diverge to infinity within a few epochs.
        """
        scale = 1.0 + float(sample @ sample)
        step = self.learning_rate * (towards - activation) / scale
        self.hidden_bias[unit] += step
        self.hidden_weights[:, unit] += step * sample

    def predict(self, x) -> np.ndarray:
        if self.hidden_weights is None:
            raise RuntimeError("the model has not been fitted")
        return self.forward(x)

    def score(self, x, y) -> float:
        return float((self.predict(x) == np.asarray(y).reshape(-1)).mean())
