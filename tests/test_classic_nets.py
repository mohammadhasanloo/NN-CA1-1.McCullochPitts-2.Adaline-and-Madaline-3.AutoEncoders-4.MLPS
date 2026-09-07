"""Tests for the threshold unit, Adaline and Madaline."""

from __future__ import annotations

import numpy as np
import pytest

from classic_nets.adaline import Adaline
from classic_nets.datasets import concentric, separable, xor
from classic_nets.madaline import Madaline, sign
from classic_nets.mcculloch_pitts import ThresholdUnit, logical_and, logical_not, logical_or

BINARY = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])


def test_and_fires_only_when_both_inputs_are_on():
    assert logical_and()(BINARY).ravel().tolist() == [0, 0, 0, 1]


def test_or_fires_when_any_input_is_on():
    assert logical_or()(BINARY).ravel().tolist() == [0, 1, 1, 1]


def test_not_inverts_a_single_input():
    assert logical_not()([[0], [1]]).ravel().tolist() == [1, 0]


def test_a_layer_evaluates_every_unit_at_once():
    layer = ThresholdUnit([[1, 1], [1, 1]], threshold=2)  # two AND units
    assert layer(BINARY).shape == (4, 2)


def test_threshold_unit_rejects_the_wrong_input_width():
    with pytest.raises(ValueError):
        logical_and()([[1, 1, 1]])


def test_xor_needs_two_layers_of_threshold_units():
    """A single threshold unit cannot compute XOR; composing three can."""
    or_gate, nand_gate = logical_or(), ThresholdUnit([[-1.0], [-1.0]], threshold=-1)
    hidden = np.hstack([or_gate(BINARY), nand_gate(BINARY)])
    assert logical_and()(hidden).ravel().tolist() == [0, 1, 1, 0]


def test_adaline_separates_a_linearly_separable_problem():
    x, y = separable(rng=np.random.default_rng(0))
    model = Adaline(learning_rate=0.01)
    model.fit(x, y, epochs=20)
    assert model.score(x, y) == 1.0


def test_adaline_loss_decreases():
    x, y = separable(rng=np.random.default_rng(0))
    history = Adaline(learning_rate=0.01).fit(x, y, epochs=20)
    assert history[-1] < history[0]


def test_adaline_cannot_separate_concentric_classes():
    """A linear unit has one straight boundary, and no line divides a ring from
    the disc inside it."""
    x, y = concentric(rng=np.random.default_rng(0))
    model = Adaline(learning_rate=0.01)
    model.fit(x, y, epochs=40)
    assert model.score(x, y) < 0.6


def test_adaline_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        Adaline().fit(np.zeros((3, 2)), np.zeros(2))


def test_predicting_before_fitting_is_an_error():
    with pytest.raises(RuntimeError):
        Adaline().predict(np.zeros((1, 2)))
    with pytest.raises(RuntimeError):
        Madaline().predict(np.zeros((1, 2)))


def test_sign_treats_zero_as_on():
    assert sign(np.array([-1.0, 0.0, 1.0])).tolist() == [-1, 1, 1]


def test_madaline_solves_what_adaline_cannot():
    """An AND over half-planes is their intersection, which is convex, and the
    inner disc of the concentric problem is exactly that."""
    x, y = concentric(rng=np.random.default_rng(0))
    model = Madaline(hidden_units=8, vote="and", rng=np.random.default_rng(1))
    model.fit(x, y)
    assert model.score(x, y) > 0.9


def test_a_fixed_vote_cannot_solve_xor_however_many_units():
    """XOR's positive class is two separate patches. Neither an intersection of
    half-planes nor a union of them is two patches, so no fixed vote expresses
    it. This needs a trained output layer."""
    x, y = xor(rng=np.random.default_rng(0))
    for vote in ("and", "or"):
        for units in (4, 16):
            model = Madaline(hidden_units=units, vote=vote, rng=np.random.default_rng(2))
            model.fit(x, y)
            assert model.score(x, y) < 0.9


def test_the_vote_decides_which_region_shape_is_expressible():
    """AND encloses a convex positive region; OR cannot on the same problem."""
    x, y = concentric(rng=np.random.default_rng(0))
    intersection = Madaline(hidden_units=8, vote="and", rng=np.random.default_rng(1))
    union = Madaline(hidden_units=8, vote="or", rng=np.random.default_rng(1))
    intersection.fit(x, y)
    union.fit(x, y)
    assert intersection.score(x, y) > union.score(x, y)


def test_an_unknown_vote_is_rejected():
    with pytest.raises(ValueError):
        Madaline(vote="xor")


def test_madaline_stops_once_no_weight_changes():
    x, y = separable(rng=np.random.default_rng(0))
    model = Madaline(hidden_units=4, vote="and", max_epochs=300,
                     rng=np.random.default_rng(3))
    history = model.fit(x, y)
    assert model.epochs_used < 300
    assert history[-1] == 0


def test_both_votes_handle_a_linearly_separable_problem():
    x, y = separable(rng=np.random.default_rng(0))
    for vote in ("and", "or"):
        model = Madaline(hidden_units=4, vote=vote, rng=np.random.default_rng(4))
        model.fit(x, y)
        assert model.score(x, y) == 1.0


def test_madaline_needs_at_least_one_hidden_unit():
    with pytest.raises(ValueError):
        Madaline(hidden_units=0)
