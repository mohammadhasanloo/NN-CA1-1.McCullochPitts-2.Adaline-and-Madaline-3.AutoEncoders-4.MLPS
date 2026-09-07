"""Classical neural network models, implemented from scratch."""

from classic_nets.adaline import Adaline
from classic_nets.datasets import concentric, separable, xor
from classic_nets.madaline import Madaline, sign
from classic_nets.mcculloch_pitts import ThresholdUnit, logical_and, logical_not, logical_or

__all__ = [
    "Adaline",
    "Madaline",
    "ThresholdUnit",
    "concentric",
    "logical_and",
    "logical_not",
    "logical_or",
    "separable",
    "sign",
    "xor",
]
