"""Synthetic problems for showing what each model can and cannot separate."""

from __future__ import annotations

import numpy as np


def separable(n: int = 200, gap: float = 2.0, rng=None):
    """Two Gaussian blobs a straight line can divide."""
    rng = rng or np.random.default_rng(0)
    half = n // 2
    x = np.vstack([rng.normal(-gap, 0.7, (half, 2)), rng.normal(gap, 0.7, (half, 2))])
    return x, np.array([-1] * half + [1] * half)


def concentric(n: int = 300, inner: float = 1.0, outer: float = 3.0, rng=None):
    """An inner disc inside an outer ring. No single line separates these."""
    rng = rng or np.random.default_rng(0)
    half = n // 2
    angles = rng.uniform(0, 2 * np.pi, n)
    radii = np.concatenate([rng.normal(inner, 0.25, half), rng.normal(outer, 0.25, half)])
    x = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    return x, np.array([1] * half + [-1] * half)


def xor(n: int = 200, spread: float = 0.4, rng=None):
    """Four clusters labelled by the sign of the product of coordinates."""
    rng = rng or np.random.default_rng(0)
    centres = np.array([[1, 1], [-1, -1], [1, -1], [-1, 1]], dtype=float)
    labels = np.array([1, 1, -1, -1])
    per = n // 4
    x = np.vstack([rng.normal(c, spread, (per, 2)) for c in centres])
    return x, np.repeat(labels, per)
