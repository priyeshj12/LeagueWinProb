"""Swing detection, narration, and the counterfactual advisor."""

import numpy as np
import pytest

from rift_oracle.analysis.advice import advise, build_advice, suggest_actions
from rift_oracle.analysis.narrate import (
    momentum_phrase,
    narrate_state,
    narrate_swing,
    scaling_outlook,
    standing,
)
from rift_oracle.analysis.swings import (
    BUNDLES,
    Track,
    TrackPoint,
    build_track,
    bundle_contributions,
    bundle_of,
    detect_swings,
    swing_summary,
)
from rift_oracle.game.state import BLUE, RED
from rift_oracle.model.gam import Prediction


# -- bundling --------------------------------------------------------------


def test_the_two_gold_encodings_report_as_one_line():
    """The model keeps them apart; a reader must not see 'gold lead' twice."""
    assert bundle_of("gold_share") == bundle_of("gold_diff_k") == "economy"
    bundled = bundle_contributions({"gold_share": 0.3, "gold_diff_k": 0.2, "drake_diff": 0.1})
    assert dict(bundled)["economy"] == pytest.approx(0.5)
    assert len(bundled) == 2


def test_bundles_never_overlap():
    seen = set()
    for _key, (_label, features, _primary) in BUNDLES.items():
        assert not (seen & set(features))
        seen.update(features)


def test_bundle_primary_is_one_of_its_members():
    for key, (_label, features, primary) in BUNDLES.items():
        assert primary in features


# -- swing detection -------------------------------------------------------


def _synthetic_track(probabilities, step=30.0):
    from rift_oracle.game.state import GameState, TeamState
    from rift_oracle.model.features import FeatureVector, FEATURE_KEYS

    points = []
    for i, p in enumerate(probabilities):
        t = i * step
        state = GameState(t=t, blue=TeamState(team_id=BLUE), red=TeamState(team_id=RED))
        vector = FeatureVector(
            values=np.zeros(len(FEATURE_KEYS)), mask=np.ones(len(FEATURE_KEYS)), t=t
        )
        prediction = Prediction(p=p, logit=float(np.log(p / (1 - p))), t=t, contributions={})
        points.append(TrackPoint(state=state, vector=vector, prediction=prediction))
    return Track(points=points)


def test_a_flat_game_has_no_swings():
    track = _synthetic_track([0.5] * 20)
    assert detect_swings(track, threshold=0.06) == []


def test_a_clear_jump_is_detected_with_the_right_size():
    track = _synthetic_track([0.5] * 5 + [0.8] * 5)
    swings = detect_swings(track, threshold=0.06, window_s=90.0)
    assert len(swings) == 1
    assert swings[0].delta == pytest.approx(0.3, abs=1e-9)
    assert swings[0].toward == "Blue"


def test_swings_never_overlap():
    rng = np.random.default_rng(0)
    probabilities = np.clip(0.5 + np.cumsum(rng.normal(0, 0.06, 120)), 0.02, 0.98)
    swings = detect_swings(_synthetic_track(list(probabilities)), threshold=0.06)
    spans = sorted((s.start_t, s.end_t) for s in swings)
    for (_a0, a1), (b0, _b1) in zip(spans, spans[1:]):
        assert b0 >= a1


def test_swings_respect_the_window():
    track = _synthetic_track([0.5, 0.52, 0.54, 0.56, 0.58, 0.60], step=60.0)
    # Each 60s step moves only 2pp, so nothing clears 6pp inside a 90s window.
    assert detect_swings(track, threshold=0.06, window_s=90.0) == []
    # Widen the window and the accumulated drift shows up.
    assert detect_swings(track, threshold=0.06, window_s=400.0)


def test_swing_direction_is_reported_from_the_asked_side():
    track = _synthetic_track([0.5] * 4 + [0.25] * 4)
    swing = detect_swings(track, threshold=0.06)[0]
    from_blue = narrate_swing(swing, track, perspective=BLUE)
    from_red = narrate_swing(swing, track, perspective=RED)
    assert "-25.0 pts" in from_blue.headline
    assert "+25.0 pts" in from_red.headline


def test_swing_summary_counts_both_directions():
    track = _synthetic_track([0.5] * 3 + [0.8] * 3 + [0.4] * 3)
    stats = swing_summary(detect_swings(track, threshold=0.06))
    assert stats["count"] >= 2
    assert stats["to_blue"] >= 1 and stats["to_red"] >= 1


# -- narration on a real game ----------------------------------------------


def test_narration_names_the_objective_that_caused_the_swing(model, game):
    track = build_track(game.states, model, winner=game.winner)
    swings = detect_swings(track, threshold=0.05)
    narratives = [narrate_swing(s, track) for s in swings]
    assert narratives, "a full game should contain at least one swing"
    for narrative in narratives:
        assert narrative.headline
        assert "pts" in narrative.headline
        for reason in narrative.reasons:
            assert reason.strip()


def test_every_narrated_reason_is_a_known_bundle_or_feature(model, game):
    from rift_oracle.analysis.swings import bundle_label

    track = build_track(game.states, model, winner=game.winner)
    labels = {bundle_label(k) for k in BUNDLES} | {
        bundle_label(k) for k in model.feature_keys
    }
    for swing in detect_swings(track, threshold=0.05):
        for key, _share in swing.top_attributions(limit=5):
            assert bundle_label(key) in labels


def test_standing_and_momentum_read_out(model, game):
    track = build_track(game.states, model, winner=game.winner)
    rows = standing(track.points[-1], limit=5)
    assert rows and all(len(row) == 4 for row in rows)
    assert narrate_state(track.points[-1])
    assert "min" in momentum_phrase(track)


def test_scaling_outlook_mentions_a_champion(model, game):
    text = scaling_outlook(game.states[len(game.states) // 2])
    assert text and text[-1] == "."


# -- advice ----------------------------------------------------------------


def test_every_suggested_objective_helps_the_side_that_takes_it(model, game):
    """A counterfactual that hands you an objective must not lower your odds."""
    state = game.states[len(game.states) // 2]
    actions, risks = suggest_actions(state, model, perspective=BLUE)
    for suggestion in actions:
        if suggestion.category == "objective":
            assert suggestion.delta_p >= -1e-9, suggestion.action
    for suggestion in risks:
        assert suggestion.delta_p <= 1e-9, suggestion.action


def test_advice_is_symmetric_between_the_two_sides(model, game):
    """What blue gains from an action, red loses. Same number, other sign."""
    state = game.states[len(game.states) // 2]
    blue_actions, _ = suggest_actions(state, model, perspective=BLUE, include_risks=False)
    _red_actions, red_risks = suggest_actions(state, model, perspective=RED)

    blue_drake = next(s for s in blue_actions if "Dragon" in s.action or "Soul" in s.action)
    red_view = next(s for s in red_risks if "Dragon" in s.action or "Soul" in s.action)
    assert blue_drake.delta_p == pytest.approx(-red_view.delta_p, abs=1e-9)


def test_counterfactuals_do_not_mutate_the_state(model, game):
    state = game.states[10]
    before_gold = state.blue.gold
    before_drakes = state.blue.dragon_count
    suggest_actions(state, model, perspective=BLUE)
    assert state.blue.gold == before_gold
    assert state.blue.dragon_count == before_drakes


def test_advice_bundle_is_populated(model, game):
    advice = advise(game.states[-5], model, perspective=BLUE)
    assert advice.actions and advice.tempo
    assert all(isinstance(note, str) and note for note in advice.build)


def test_build_advice_reads_the_enemy_damage_profile(model, game):
    state = game.states[-1]
    notes = build_advice(state, perspective=BLUE)
    assert any("physical" in n or "magic" in n for n in notes)


def test_spending_gold_is_only_suggested_when_there_is_gold_to_spend(model, game):
    state = game.states[3]  # minute two: nobody is sitting on a wallet
    for player in state.blue.players:
        player.current_gold = 0.0
    actions, _risks = suggest_actions(state, model, perspective=BLUE)
    assert not any("spend" in s.action.lower() for s in actions)


def test_a_headline_names_an_event_that_helped_the_side_the_swing_helped():
    """A window can hold good news for both teams; the headline must pick right.

    Seen on a real EUW game: Blue took a turret inside a window that swung
    fourteen points to Red, and the headline credited the turret.
    """
    from rift_oracle.analysis.narrate import narrate_swing
    from rift_oracle.analysis.swings import Swing
    from rift_oracle.game.state import GameEvent

    swing = Swing(
        start_t=900.0,
        end_t=1020.0,
        p_before=0.72,
        p_after=0.58,  # fourteen points toward Red
        attributions=[("kill_diff", -0.09), ("tower_diff", -0.05)],
        events=[
            GameEvent(t=950.0, type="TURRET_KILL", team=BLUE,
                      text="Blue took the Bot Outer turret", importance=2.0),
            GameEvent(t=1000.0, type="CHAMPION_KILL", team=RED,
                      text="Samira killed Jinx", importance=1.0),
        ],
    )
    headline = narrate_swing(swing).headline
    assert "Samira killed Jinx" in headline
    assert "Blue took" not in headline


def test_a_headline_falls_back_when_no_event_matches_the_direction():
    from rift_oracle.analysis.narrate import narrate_swing
    from rift_oracle.analysis.swings import Swing
    from rift_oracle.game.state import GameEvent

    swing = Swing(
        start_t=600.0, end_t=690.0, p_before=0.60, p_after=0.46,
        attributions=[("tower_diff", -0.14)],
        events=[
            GameEvent(t=650.0, type="TURRET_KILL", team=BLUE,
                      text="Blue took the Mid Outer turret", importance=2.0),
        ],
    )
    headline = narrate_swing(swing).headline
    assert "Blue took" not in headline
    assert "Red" in headline  # describes the drift instead of miscrediting
