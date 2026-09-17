"""Shared fixtures."""

import pytest

from rift_oracle.model.registry import load_model
from rift_oracle.sim.synth import simulate_game


@pytest.fixture(autouse=True, scope="session")
def _isolated_home(tmp_path_factory):
    """Point every test at a throwaway state directory.

    Without this the suite reads and writes the real ~/.rift_oracle: it would
    pick up whatever API key the developer has configured, and the on-disk
    response cache would serve a previous run's HTTP responses to tests that
    assert a request was made. Both make tests pass or fail for reasons that
    have nothing to do with the code.
    """
    import os

    home = tmp_path_factory.mktemp("rift_oracle_home")
    previous = {
        name: os.environ.get(name)
        for name in ("RIFT_ORACLE_HOME", "RIOT_API_KEY", "RIOT_TOKEN", "RGAPI_KEY")
    }
    os.environ["RIFT_ORACLE_HOME"] = str(home)
    for name in ("RIOT_API_KEY", "RIOT_TOKEN", "RGAPI_KEY"):
        os.environ.pop(name, None)
    try:
        yield home
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


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
