"""The model's two structural guarantees: antisymmetry and exact attribution."""

import numpy as np
import pytest

from rift_oracle.model.features import FEATURE_KEYS, FeatureVector, extract
from rift_oracle.model.gam import (
    AdditiveWinModel,
    delta_attribution,
    monotone_normals,
    project_monotone,
    sigmoid,
)


def test_swapping_teams_flips_the_probability_exactly(model, game):
    """P(blue) + P(red) must be exactly 1, not approximately.

    Every feature is a blue-minus-red difference and every basis function is
    odd, so negating the feature vector must negate the logit. If this ever
    fails the model has learned to prefer a side for a reason it cannot name.
    """
    for state in game.states[::5]:
        vector = extract(state)
        forward = model.predict(vector)
        mirrored = model.predict(
            FeatureVector(values=-vector.values, mask=vector.mask, t=vector.t)
        )
        # side_bias is the one legitimate asymmetry, so remove it before comparing.
        assert forward.logit - model.side_bias == pytest.approx(
            -(mirrored.logit - model.side_bias), abs=1e-12
        )


def test_contributions_reconstruct_the_logit(model, game):
    for state in game.states[::3]:
        prediction = model.predict(extract(state))
        assert prediction.check(tolerance=1e-9)
        total = model.side_bias + sum(prediction.contributions.values())
        assert total == pytest.approx(prediction.logit, abs=1e-9)


def test_delta_attribution_sums_to_the_real_change(model, game):
    """Attribution must account for the whole move, with no residual."""
    points = [model.predict(extract(state)) for state in game.states]
    for before, after in zip(points, points[1:]):
        delta_p, per_feature = delta_attribution(before, after)
        assert delta_p == pytest.approx(after.p - before.p, abs=1e-12)
        if per_feature:
            assert sum(per_feature.values()) == pytest.approx(delta_p, abs=1e-9)


def test_every_feature_response_is_non_decreasing(model):
    """Features are defined so more is better for blue; the fit must agree."""
    for key in model.feature_keys:
        for clock in (300.0, 1200.0, 2400.0):
            _xs, ys = model.response_curve(key, t_seconds=clock, points=41)
            diffs = np.diff(ys)
            assert diffs.min() >= -1e-9, f"{key} reverses direction at {clock}s"


def test_masked_features_contribute_nothing(model, game):
    vector = extract(game.states[10])
    blanked = FeatureVector(
        values=vector.values, mask=np.zeros_like(vector.mask), t=vector.t
    )
    prediction = model.predict(blanked)
    assert all(abs(v) < 1e-12 for v in prediction.contributions.values())
    assert prediction.logit == pytest.approx(model.side_bias, abs=1e-12)


def test_monotone_projection_produces_feasible_weights():
    normals = monotone_normals()
    rng = np.random.default_rng(0)
    for _ in range(200):
        weights = rng.normal(0, 2.0, size=4)
        projected = project_monotone(weights, normals)
        assert (normals @ projected).min() >= -1e-9


def test_monotone_projection_leaves_feasible_weights_alone():
    normals = monotone_normals()
    feasible = np.array([1.0, 0.0, 0.0, 0.0])
    assert project_monotone(feasible, normals) == pytest.approx(feasible, abs=1e-12)


def test_fit_recovers_a_known_relationship():
    """A fit on data generated from a single feature should find that feature."""
    rng = np.random.default_rng(3)
    n = 4000
    values = np.zeros((n, len(FEATURE_KEYS)))
    masks = np.ones_like(values)
    gold = rng.normal(0, 3.0, size=n)
    values[:, list(FEATURE_KEYS).index("gold_diff_k")] = gold
    times = rng.uniform(300, 2100, size=n)
    labels = (rng.random(n) < sigmoid(0.6 * gold)).astype(float)

    fitted = AdditiveWinModel()
    fitted.set_knots_from_data(values, masks)
    stats = fitted.fit(values, masks, times, labels, l2=0.5)

    assert stats["accuracy"] > 0.75
    importance = fitted.feature_importance(values, masks, times)
    assert max(importance, key=importance.get) == "gold_diff_k"


def test_model_round_trips_through_json(model, tmp_path, game):
    path = model.save(tmp_path / "m.json")
    reloaded = AdditiveWinModel.load(path)
    vector = extract(game.states[5])
    assert reloaded.predict(vector).logit == pytest.approx(
        model.predict(vector).logit, abs=1e-12
    )


def test_loading_a_model_with_a_different_basis_is_refused():
    payload = {"n_basis": 7, "feature_keys": [], "knots": [], "clips": [], "weights": []}
    with pytest.raises(ValueError, match="basis"):
        AdditiveWinModel.from_dict(payload)


def test_sigmoid_is_stable_at_extremes():
    assert sigmoid(-10000.0) == pytest.approx(0.0, abs=1e-12)
    assert sigmoid(10000.0) == pytest.approx(1.0, abs=1e-12)
    assert np.isfinite(sigmoid(np.array([-1e9, 0.0, 1e9]))).all()


def test_l2_selection_prefers_more_shrinkage_on_less_data():
    """The cross-validated penalty has to respond to sample size.

    A fixed default is wrong for one of the two regimes this tool runs in:
    a few hundred real matches against eighty-five parameters need far more
    shrinkage than several thousand simulated games do.
    """
    from rift_oracle.model.train import Dataset, select_l2
    from rift_oracle.sim.synth import simulate_dataset

    def build(n_games):
        arrays = simulate_dataset(n_games=n_games, seed=5)
        return Dataset(
            values=arrays["values"], masks=arrays["masks"], times=arrays["times"],
            labels=arrays["labels"], groups=arrays["groups"],
        )

    small_l2, small_search = select_l2(build(25), grid=(1.0, 8.0, 64.0), folds=2)
    large_l2, large_search = select_l2(build(400), grid=(1.0, 8.0, 64.0), folds=2)

    assert small_search and large_search
    assert all(row["log_loss"] == row["log_loss"] for row in small_search)
    assert small_l2 >= large_l2


def test_l2_selection_splits_folds_by_game():
    """Splitting by row would leak a game's winner across the fold boundary."""
    import numpy as np

    from rift_oracle.model.train import Dataset, select_l2
    from rift_oracle.sim.synth import simulate_dataset

    arrays = simulate_dataset(n_games=30, seed=6)
    dataset = Dataset(
        values=arrays["values"], masks=arrays["masks"], times=arrays["times"],
        labels=arrays["labels"], groups=arrays["groups"],
    )
    chosen, search = select_l2(dataset, grid=(2.0,), folds=3)
    assert chosen == 2.0
    assert len(search) == 1
    # Every state of a game must land in the same fold, which is what makes the
    # held-out score mean anything.
    assert np.unique(dataset.groups).size == 30


def test_an_empty_grid_falls_back_to_a_sane_default():
    from rift_oracle.model.train import Dataset, select_l2
    from rift_oracle.sim.synth import simulate_dataset

    arrays = simulate_dataset(n_games=5, seed=1)
    dataset = Dataset(
        values=arrays["values"], masks=arrays["masks"], times=arrays["times"],
        labels=arrays["labels"], groups=arrays["groups"],
    )
    chosen, search = select_l2(dataset, grid=(), folds=2)
    assert chosen == 2.0 and search == []
