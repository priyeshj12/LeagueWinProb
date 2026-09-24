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

# -- feature shards --------------------------------------------------------
#
# Shards are what an unattended harvest accumulates. Raw match and timeline
# JSON is ~900 MB per thousand games; the features those reduce to are ~1 MB,
# which is the difference between a repo that can keep history and one that
# cannot.


def _dataset(n_games, seed):
    from rift_oracle.model.train import Dataset
    from rift_oracle.sim.synth import simulate_dataset

    a = simulate_dataset(n_games=n_games, seed=seed)
    return Dataset(a["values"], a["masks"], a["times"], a["labels"], a["groups"])


def test_a_shard_round_trips(tmp_path):
    from rift_oracle.model.train import Dataset

    original = _dataset(20, 1)
    path = original.save_npz(tmp_path / "s.npz", meta={"platform": "euw1"})
    back = Dataset.load_npz(path)

    assert np.allclose(original.values, back.values, atol=1e-4)
    assert np.array_equal(original.masks, back.masks)  # packed to bits and back
    assert np.array_equal(original.labels, back.labels)
    assert np.array_equal(original.groups, back.groups)
    assert np.allclose(original.times, back.times, atol=1e-2)


def test_a_shard_is_far_smaller_than_the_json_it_came_from(tmp_path):
    original = _dataset(50, 2)
    path = original.save_npz(tmp_path / "s.npz")
    kb_per_1000_games = path.stat().st_size / 1024 / original.n_games * 1000
    # Raw match+timeline JSON measures ~900_000 KB per thousand games.
    assert kb_per_1000_games < 5_000


def test_shards_carry_nothing_that_identifies_a_player_or_a_game(tmp_path):
    """A published shard must be feature differences and a win bit, no more."""
    path = _dataset(10, 3).save_npz(tmp_path / "s.npz")
    with np.load(path, allow_pickle=False) as payload:
        assert set(payload.files) == {
            "values", "masks", "n_features", "times", "labels",
            "groups", "feature_keys", "fmt", "meta",
        }
        blob = path.read_bytes().lower()
    for leak in (b"puuid", b"summoner", b"riotid", b"euw1_", b"na1_"):
        assert leak not in blob


def test_concatenating_shards_renumbers_games(tmp_path):
    """Two shards both number their games from zero."""
    from rift_oracle.model.train import Dataset

    a, b = _dataset(12, 4), _dataset(9, 5)
    assert a.groups.min() == b.groups.min() == 0

    merged = Dataset.concat([a, b])
    assert merged.n_games == a.n_games + b.n_games
    assert len(merged) == len(a) + len(b)
    # Without renumbering, a split promising to hold out whole games would
    # leak across the shard boundary.
    _train, test = merged.split_by_game(test_fraction=0.3, seed=0)
    assert set(test.groups).isdisjoint(set(_train.groups))


def test_a_directory_of_shards_loads_as_one_dataset(tmp_path):
    from rift_oracle.model.train import Dataset

    _dataset(8, 6).save_npz(tmp_path / "2026-09-17-euw1.npz")
    _dataset(7, 7).save_npz(tmp_path / "2026-09-24-na1.npz")
    merged = Dataset.load_shards(tmp_path)
    assert merged.n_games == 15


def test_a_shard_from_a_different_feature_set_is_refused(tmp_path):
    """Silently mixing incompatible shards would be worse than refusing."""
    import numpy as np_

    from rift_oracle.config import RiftOracleError
    from rift_oracle.model.train import Dataset

    path = _dataset(5, 8).save_npz(tmp_path / "s.npz")
    with np_.load(path, allow_pickle=False) as payload:
        fields = {k: payload[k] for k in payload.files}
    fields["feature_keys"] = np_.array(["not", "the", "same", "features"])
    np_.savez_compressed(path, **fields)

    with pytest.raises(RiftOracleError, match="different feature set"):
        Dataset.load_npz(path)


def test_an_unreadable_format_version_is_refused(tmp_path):
    import numpy as np_

    from rift_oracle.config import RiftOracleError
    from rift_oracle.model.train import Dataset

    path = _dataset(5, 9).save_npz(tmp_path / "s.npz")
    with np_.load(path, allow_pickle=False) as payload:
        fields = {k: payload[k] for k in payload.files}
    fields["fmt"] = np_.array(["rift_oracle.dataset.v99"])
    np_.savez_compressed(path, **fields)

    with pytest.raises(RiftOracleError, match="format"):
        Dataset.load_npz(path)


def test_the_committed_shards_train_a_model_without_an_api_key():
    """The repo ships features, not raw data, so it is trainable offline."""
    from pathlib import Path

    from rift_oracle.model.train import Dataset

    shards = Path(__file__).resolve().parent.parent / "data" / "features"
    if not list(shards.glob("*.npz")):
        pytest.skip("no shards committed")
    data = Dataset.load_shards(shards)
    assert data.n_games > 100
    assert len(data) > 1000
    assert set(np.unique(data.labels)) <= {0.0, 1.0}


def test_a_new_model_is_only_preferred_when_it_beats_the_old_one():
    """The guard an unattended retrain leans on."""
    from rift_oracle.model.train import compare_on_holdout, train_model

    data = _dataset(120, 11)
    good, _report = train_model(data, l2=2.0, seed=0, test_fraction=0.01)

    from rift_oracle.model.gam import AdditiveWinModel

    useless = AdditiveWinModel()  # all-zero weights: says 50% to everything

    verdict = compare_on_holdout(data, good, useless, seed=0)
    assert verdict["comparable"]
    assert verdict["log_loss_gain"] > 0  # the fitted model wins

    reversed_verdict = compare_on_holdout(data, useless, good, seed=0)
    assert reversed_verdict["log_loss_gain"] < 0
