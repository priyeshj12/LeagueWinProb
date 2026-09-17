"""Shared fixtures."""

import pytest

from rift_oracle.model.registry import load_model
from rift_oracle.sim.synth import simulate_game


@pytest.fixture(scope="session")
def model():
    """The bundled baseline model."""
    loaded, _path = load_model()
    return loaded


@pytest.fixture(scope="session")
def game():
    """One deterministic simulated game."""
    return simulate_game(seed=1234)


@pytest.fixture(scope="session")
def dataset():
    """A feature dataset large enough for calibration to be measurable.

    Expected calibration error is a binned statistic, so on a small sample it
    measures binning noise rather than the model: forty games reports about
    0.15 for a model that is genuinely at 0.03. Five hundred games costs a
    few seconds and makes the number mean something.
    """
    from rift_oracle.sim.synth import simulate_dataset

    return simulate_dataset(n_games=500, seed=99)
