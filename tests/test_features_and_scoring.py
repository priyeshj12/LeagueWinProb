"""Feature extraction, calibration metrics, the simulator, and the CLI."""


import numpy as np
import pytest

from rift_oracle.config import Settings, redact
from rift_oracle.game.scaling import champion_scale, comp_scale, ramp, scaling_edge
from rift_oracle.game.state import BLUE, RED
from rift_oracle.model import calibrate
from rift_oracle.model.features import (
    FEATURE_KEYS,
    LIVE_MASKED,
    SPEC_BY_KEY,
    apply_live_mask,
    extract,
    rank_prior_from_entries,
    rank_score,
)


# -- features --------------------------------------------------------------


def test_feature_keys_are_unique_and_specified():
    assert len(set(FEATURE_KEYS)) == len(FEATURE_KEYS)
    for key in FEATURE_KEYS:
        spec = SPEC_BY_KEY[key]
        assert spec.label and spec.noun and spec.typical > 0


def test_swapping_the_two_teams_negates_every_feature(game):
    """Antisymmetry has to hold at the extractor, not just at the model."""
    import copy

    for state in game.states[::7]:
        mirrored = copy.deepcopy(state)
        mirrored.blue, mirrored.red = mirrored.red, mirrored.blue
        mirrored.blue.team_id, mirrored.red.team_id = BLUE, RED
        for player in mirrored.blue.players:
            player.team = BLUE
        for player in mirrored.red.players:
            player.team = RED

        forward = extract(state)
        backward = extract(mirrored)
        observed = (forward.mask > 0) & (backward.mask > 0)
        assert forward.values[observed] == pytest.approx(
            -backward.values[observed], abs=1e-9
        )


def test_unobservable_features_are_masked_not_zeroed(game):
    state = game.states[5]
    for player in state.blue.players + state.red.players:
        player.xp = 0
        player.damage_to_champs = 0
        player.physical_damage_to_champs = 0
        player.magic_damage_to_champs = 0
    vector = extract(state)
    assert not vector.observed("xp_share")
    assert not vector.observed("dmg_share")
    assert "xp_share" not in vector.as_dict()


def test_live_mask_hides_exactly_the_timeline_only_features():
    mask = np.ones((3, len(FEATURE_KEYS)))
    masked = apply_live_mask(mask)
    hidden = {FEATURE_KEYS[i] for i in range(len(FEATURE_KEYS)) if masked[0, i] == 0}
    assert hidden == set(LIVE_MASKED)
    assert "gold_diff_k" not in hidden


def test_rank_scores_are_ordered():
    assert rank_score("IRON", "IV") < rank_score("GOLD", "IV") < rank_score("DIAMOND", "I")
    assert rank_score("GOLD", "I") > rank_score("GOLD", "IV")
    assert rank_score("CHALLENGER", "I", 1000) > rank_score("MASTER", "I", 0)


def test_rank_prior_is_antisymmetric_and_bounded():
    assert rank_prior_from_entries([20], [12]) == pytest.approx(
        -rank_prior_from_entries([12], [20])
    )
    assert abs(rank_prior_from_entries([32] * 5, [0] * 5)) <= 1.5
    assert rank_prior_from_entries([], [12]) == 0.0


# -- scaling ---------------------------------------------------------------


def test_ramp_runs_from_early_to_late():
    assert ramp(0) < -0.9
    assert ramp(22 * 60) == pytest.approx(0.0, abs=1e-9)
    assert ramp(45 * 60) > 0.9


def test_known_scalers_are_rated_correctly():
    assert champion_scale("Kayle") > champion_scale("Garen") > champion_scale("Lee Sin")
    assert champion_scale("Pantheon") < 0 < champion_scale("Vayne")


def test_unknown_champion_falls_back_rather_than_crashing():
    assert champion_scale("NotAChampion") == 0.0
    assert champion_scale("") == 0.0


def test_scaling_edge_flips_sign_with_the_clock():
    late = ["Kayle", "Vayne", "Veigar", "Nasus", "Kassadin"]
    early = ["Lee Sin", "Pantheon", "Renekton", "Draven", "Elise"]
    assert scaling_edge(late, early, 5 * 60) < 0  # early comp is ahead at 5 minutes
    assert scaling_edge(late, early, 40 * 60) > 0  # late comp is ahead at 40
    assert scaling_edge(late, early, 600) == pytest.approx(
        -scaling_edge(early, late, 600)
    )


def test_comp_scale_averages_its_members():
    assert comp_scale(["Kayle", "Kayle"]) == pytest.approx(champion_scale("Kayle"))


# -- scoring ---------------------------------------------------------------


def test_auc_of_a_perfect_ranker_is_one():
    labels = np.array([0, 0, 1, 1.0])
    assert calibrate.auc(labels, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)
    assert calibrate.auc(labels, np.array([0.9, 0.8, 0.2, 0.1])) == pytest.approx(0.0)


def test_auc_handles_ties_as_coin_flips():
    labels = np.array([0, 1.0])
    assert calibrate.auc(labels, np.array([0.5, 0.5])) == pytest.approx(0.5)


def test_auc_is_nan_when_one_class_is_missing():
    assert np.isnan(calibrate.auc(np.array([1.0, 1.0]), np.array([0.3, 0.7])))


def test_log_loss_and_brier_reward_the_truth():
    labels = np.array([1.0, 0.0])
    confident = np.array([0.99, 0.01])
    hedged = np.array([0.5, 0.5])
    assert calibrate.log_loss(labels, confident) < calibrate.log_loss(labels, hedged)
    assert calibrate.brier(labels, confident) < calibrate.brier(labels, hedged)


def test_log_loss_is_finite_at_the_boundaries():
    assert np.isfinite(calibrate.log_loss(np.array([1.0]), np.array([0.0])))


def test_isotonic_calibrator_is_monotone():
    rng = np.random.default_rng(1)
    raw = rng.random(2000)
    labels = (rng.random(2000) < raw**2).astype(float)
    fitted = calibrate.IsotonicCalibrator().fit(raw, labels)
    mapped = fitted.transform(np.linspace(0, 1, 50))
    assert np.all(np.diff(mapped) >= -1e-9)
    assert calibrate.log_loss(labels, fitted.transform(raw)) < calibrate.log_loss(labels, raw)


def test_isotonic_round_trips_through_json():
    fitted = calibrate.IsotonicCalibrator().fit(
        np.array([0.1, 0.5, 0.9]), np.array([0.0, 1.0, 1.0])
    )
    restored = calibrate.IsotonicCalibrator.from_dict(fitted.to_dict())
    assert restored.transform(np.array([0.5])) == pytest.approx(fitted.transform(np.array([0.5])))


def test_reliability_table_bins_cover_everything():
    rng = np.random.default_rng(2)
    p = rng.random(5000)
    labels = (rng.random(5000) < p).astype(float)
    rows = calibrate.reliability_table(labels, p, bins=10)
    assert sum(row["count"] for row in rows) == 5000
    for row in rows:
        assert abs(row["observed"] - row["mean_predicted"]) < 0.08


def test_the_bundled_model_is_calibrated(model, dataset):
    """The shipped model must mean what it says, not just rank correctly."""
    p = model.predict_proba(dataset["values"], dataset["masks"], dataset["times"])
    scores = calibrate.metrics(dataset["labels"], p)
    assert scores["auc"] > 0.75
    assert scores["ece"] < 0.05
    assert scores["log_loss"] < 0.60
    # Recalibrating a well-fit model should buy almost nothing.
    assert calibrate.calibration_gain(dataset["labels"], p)["gain"] < 0.02


# -- simulator -------------------------------------------------------------


def test_simulated_games_look_like_league():
    from rift_oracle.sim.synth import simulate_game

    rng = np.random.default_rng(5)
    durations, kills, blue_wins = [], [], 0
    for _ in range(120):
        sim = simulate_game(rng=rng)
        durations.append(sim.duration_s / 60.0)
        kills.append(sim.states[-1].blue.kills + sim.states[-1].red.kills)
        blue_wins += sim.winner == BLUE

    assert 24 < np.median(durations) < 36
    assert 25 < np.median(kills) < 70
    # The simulator has no side advantage, so the win rate must be near even.
    assert abs(blue_wins / 120 - 0.5) < 0.13


def test_simulated_games_are_internally_consistent(game):
    assert game.winner in (BLUE, RED)
    assert all(s.winner == game.winner for s in game.states)
    for state in game.states:
        assert len(state.blue.players) == len(state.red.players) == 5
        assert state.blue.alive <= 5 and state.red.alive <= 5
        assert state.blue.gold > 0 and state.red.gold > 0
    times = [s.t for s in game.states]
    assert times == sorted(times)


def test_simulation_is_reproducible():
    from rift_oracle.sim.synth import simulate_game

    a, b = simulate_game(seed=42), simulate_game(seed=42)
    assert a.winner == b.winner
    assert a.duration_s == b.duration_s
    assert a.blue_champions == b.blue_champions


# -- config ----------------------------------------------------------------


def test_api_keys_are_redacted_from_messages():
    text = "failed for key RGAPI-12345678-1234-1234-1234-123456789abc at /x"
    assert "RGAPI-12345678" not in redact(text)
    assert "RGAPI-<redacted>" in redact(text)


def test_settings_reject_an_unknown_platform():
    with pytest.raises(ValueError):
        Settings.load(platform="atlantis")


def test_settings_prefer_explicit_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("RIFT_ORACLE_HOME", str(tmp_path))
    monkeypatch.setenv("RIOT_API_KEY", "RGAPI-from-env")
    assert Settings.load().api_key == "RGAPI-from-env"
    assert Settings.load(api_key="RGAPI-explicit").api_key == "RGAPI-explicit"


def test_missing_key_message_explains_how_to_set_one(monkeypatch, tmp_path):
    from rift_oracle.config import MissingAPIKey

    monkeypatch.setenv("RIFT_ORACLE_HOME", str(tmp_path))
    monkeypatch.delenv("RIOT_API_KEY", raising=False)
    monkeypatch.delenv("RIOT_TOKEN", raising=False)
    monkeypatch.delenv("RGAPI_KEY", raising=False)
    with pytest.raises(MissingAPIKey, match="RIOT_API_KEY"):
        Settings.load().require_key()
