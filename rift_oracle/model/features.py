"""Feature extraction: a GameState becomes a fixed, antisymmetric vector.

Every feature is a *difference* written from blue's point of view, so swapping
the two teams negates the whole vector. The model exploits that: because it is
built from odd basis functions, negating the input negates the logit, which
means ``P(blue wins) + P(red wins) == 1`` holds exactly rather than
approximately. There is no way for the model to prefer a side for a reason it
cannot name.

Each feature also carries an observability flag. The Live Client Data API
cannot see experience or damage dealt; a timeline can. Rather than imputing
zeros and pretending, the extractor returns a mask, and a masked feature
contributes exactly nothing to the logit. Training applies the same masks at
random (see :mod:`rift_oracle.model.train`) so the model is honest about what
it is like to run on partial information.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np

from rift_oracle.game.scaling import scaling_edge
from rift_oracle.game.state import GameState, expected_team_gold, expected_team_xp


@dataclass(frozen=True)
class FeatureSpec:
    """One model input: how to compute it, name it, and talk about it."""

    key: str
    label: str
    #: Short noun phrase used when narrating a swing ("the gold lead shifted").
    noun: str
    #: Units for display, e.g. ``"k gold"``. Empty when the number is a count.
    unit: str = ""
    #: Multiplier from model units back to something a player recognises.
    display_scale: float = 1.0
    #: False for features only a match timeline can supply.
    live_observable: bool = True
    #: Rough magnitude, used to place initial knots before any training data.
    typical: float = 1.0
    group: str = "economy"


FEATURES: Tuple[FeatureSpec, ...] = (
    FeatureSpec("gold_share", "Gold lead (relative)", "gold lead", "%", 100.0,
                True, 0.06, "economy"),
    FeatureSpec("gold_diff_k", "Gold lead (absolute)", "gold lead", "k", 1.0,
                True, 2.0, "economy"),
    FeatureSpec("xp_share", "Experience lead", "experience lead", "%", 100.0,
                False, 0.04, "economy"),
    FeatureSpec("level_diff", "Total level lead", "level lead", "lv", 1.0,
                True, 3.0, "economy"),
    FeatureSpec("cs_diff", "Creep score lead", "farm lead", "cs", 10.0,
                True, 2.0, "economy"),
    FeatureSpec("item_value_diff_k", "Completed item value", "item value", "k",
                1.0, True, 2.0, "economy"),

    FeatureSpec("kill_diff", "Kill lead", "kill lead", "", 1.0, True, 4.0, "combat"),
    FeatureSpec("alive_diff", "Champions alive", "bodies on the map", "", 1.0,
                True, 1.0, "combat"),
    FeatureSpec("respawn_diff", "Respawn timer edge", "respawn timers", "s", 10.0,
                True, 2.0, "combat"),
    FeatureSpec("dmg_share", "Damage share", "damage output", "%", 100.0,
                False, 0.05, "combat"),

    FeatureSpec("tower_diff", "Turret lead", "turret lead", "", 1.0, True, 2.0,
                "objectives"),
    FeatureSpec("plate_diff", "Turret plates", "plate gold", "", 1.0, True, 2.0,
                "objectives"),
    FeatureSpec("inhib_diff", "Open inhibitors", "open inhibitors", "", 1.0, True,
                1.0, "objectives"),
    FeatureSpec("drake_diff", "Dragon lead", "dragon control", "", 1.0, True, 1.5,
                "objectives"),
    FeatureSpec("soul_diff", "Dragon soul", "dragon soul", "", 1.0, True, 1.0,
                "objectives"),
    FeatureSpec("elder_diff", "Elder buff", "Elder buff", "", 1.0, True, 1.0,
                "objectives"),
    FeatureSpec("baron_diff", "Baron buff", "Baron buff", "", 1.0, True, 1.0,
                "objectives"),
    FeatureSpec("herald_diff", "Herald / grubs", "early objectives", "", 1.0, True,
                1.0, "objectives"),

    FeatureSpec("scaling_diff", "Composition scaling", "composition scaling", "",
                1.0, True, 0.2, "draft"),
    FeatureSpec("vision_diff", "Vision lead", "vision control", "", 10.0, True,
                2.0, "macro"),
    FeatureSpec("rank_diff", "Rank prior", "player skill", "", 1.0, True, 0.3,
                "draft"),
)

FEATURE_KEYS: Tuple[str, ...] = tuple(spec.key for spec in FEATURES)
FEATURE_INDEX: Dict[str, int] = {key: i for i, key in enumerate(FEATURE_KEYS)}
SPEC_BY_KEY: Dict[str, FeatureSpec] = {spec.key: spec for spec in FEATURES}

#: Features the Live Client Data API cannot supply.
LIVE_MASKED: Tuple[str, ...] = tuple(s.key for s in FEATURES if not s.live_observable)


@dataclass
class FeatureVector:
    """Feature values plus which of them were actually observed."""

    values: np.ndarray  # shape (n_features,)
    mask: np.ndarray  # shape (n_features,), 1.0 observed / 0.0 unknown
    t: float = 0.0

    def get(self, key: str) -> float:
        return float(self.values[FEATURE_INDEX[key]])

    def observed(self, key: str) -> bool:
        return bool(self.mask[FEATURE_INDEX[key]])

    def as_dict(self) -> Dict[str, float]:
        return {
            key: float(self.values[i])
            for i, key in enumerate(FEATURE_KEYS)
            if self.mask[i]
        }

    @property
    def minutes(self) -> float:
        return self.t / 60.0


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def extract(state: GameState, rank_prior: float = 0.0) -> FeatureVector:
    """Build the model input for one snapshot.

    ``rank_prior`` is an optional pre-game skill edge in logit units, positive
    for blue. It is zero unless ranked data was fetched for the lobby.
    """
    blue, red = state.blue, state.red
    t = float(state.t)
    values = np.zeros(len(FEATURE_KEYS), dtype=np.float64)
    mask = np.ones(len(FEATURE_KEYS), dtype=np.float64)

    def put(key: str, value: float, observed: bool = True) -> None:
        idx = FEATURE_INDEX[key]
        values[idx] = float(value) if observed else 0.0
        mask[idx] = 1.0 if observed else 0.0

    # -- economy ---------------------------------------------------------
    gold_diff = blue.gold - red.gold
    put("gold_share", _safe_div(gold_diff, expected_team_gold(t)))
    put("gold_diff_k", gold_diff / 1000.0)

    xp_known = (blue.xp + red.xp) > 0
    put("xp_share", _safe_div(blue.xp - red.xp, expected_team_xp(t)), xp_known)
    put("level_diff", blue.levels - red.levels)
    put("cs_diff", (blue.cs - red.cs) / 10.0)
    put("item_value_diff_k", (blue.item_value - red.item_value) / 1000.0)

    # -- combat ----------------------------------------------------------
    put("kill_diff", blue.kills - red.kills)
    put("alive_diff", blue.alive - red.alive)
    # Positive means red is the side sitting in the fountain.
    put("respawn_diff", (red.respawn_total_s - blue.respawn_total_s) / 10.0)

    damage_total = blue.damage_to_champs + red.damage_to_champs
    put(
        "dmg_share",
        _safe_div(blue.damage_to_champs - red.damage_to_champs, max(damage_total, 1.0)),
        damage_total > 1000,
    )

    # -- objectives ------------------------------------------------------
    put("tower_diff", blue.towers - red.towers)
    put("plate_diff", blue.turret_plates - red.turret_plates)
    # An inhibitor matters while it is *down*, not forever after it fell.
    put("inhib_diff", red.inhibitors_down - blue.inhibitors_down)
    put("drake_diff", blue.dragon_count - red.dragon_count)
    put("soul_diff", (1.0 if blue.has_soul else 0.0) - (1.0 if red.has_soul else 0.0))
    put("elder_diff", blue.buff_remaining("elder", t) - red.buff_remaining("elder", t))
    put("baron_diff", blue.buff_remaining("baron", t) - red.buff_remaining("baron", t))
    put("herald_diff", blue.heralds - red.heralds)

    # -- draft and macro -------------------------------------------------
    champions_known = any(p.champion not in ("", "Unknown") for p in blue.players)
    put(
        "scaling_diff",
        scaling_edge(blue.champions, red.champions, t) if champions_known else 0.0,
        champions_known,
    )
    put("vision_diff", (blue.vision - red.vision) / 10.0)
    put("rank_diff", rank_prior, abs(rank_prior) > 1e-9)

    return FeatureVector(values=values, mask=mask, t=t)


def extract_many(
    states: Sequence[GameState], rank_prior: float = 0.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised extraction: returns ``(values, masks, times)``."""
    if not states:
        empty = np.zeros((0, len(FEATURE_KEYS)))
        return empty, empty, np.zeros(0)
    vectors = [extract(state, rank_prior=rank_prior) for state in states]
    values = np.vstack([v.values for v in vectors])
    masks = np.vstack([v.mask for v in vectors])
    times = np.array([v.t for v in vectors], dtype=np.float64)
    return values, masks, times


def apply_live_mask(mask: np.ndarray) -> np.ndarray:
    """Zero out the features a live game cannot observe."""
    out = np.array(mask, dtype=np.float64, copy=True)
    for key in LIVE_MASKED:
        out[..., FEATURE_INDEX[key]] = 0.0
    return out


def display_value(key: str, value: float) -> str:
    """Render a feature value the way a player would say it."""
    spec = SPEC_BY_KEY[key]
    scaled = value * spec.display_scale
    if spec.unit == "%":
        return f"{scaled:+.1f}%"
    if spec.unit == "k":
        return f"{scaled:+.1f}k"
    if spec.unit == "s":
        return f"{scaled:+.0f}s"
    if spec.unit == "cs":
        return f"{scaled:+.0f} cs"
    if spec.unit == "lv":
        return f"{scaled:+.0f} lv"
    if abs(scaled - round(scaled)) < 1e-6:
        return f"{scaled:+.0f}"
    return f"{scaled:+.1f}"


def rank_score(tier: str, division: str = "I", lp: int = 0) -> float:
    """Map a ranked tier to a single number, roughly one unit per division."""
    tiers = {
        "IRON": 0, "BRONZE": 4, "SILVER": 8, "GOLD": 12, "PLATINUM": 16,
        "EMERALD": 20, "DIAMOND": 24, "MASTER": 28, "GRANDMASTER": 30,
        "CHALLENGER": 32,
    }
    divisions = {"IV": 0, "III": 1, "II": 2, "I": 3}
    base = tiers.get(str(tier).upper(), 12)
    if base < 28:
        base += divisions.get(str(division).upper(), 0)
    return float(base) + min(float(lp), 1500.0) / 400.0


def rank_prior_from_entries(
    blue_scores: Sequence[float], red_scores: Sequence[float]
) -> float:
    """Blue's skill edge from average rank, scaled to roughly ``[-1, 1]``.

    One division of average difference across a whole lobby is a real but
    modest edge, so the divisor keeps a full-tier gap near 0.25.
    """
    if not blue_scores or not red_scores:
        return 0.0
    blue_mean = sum(blue_scores) / len(blue_scores)
    red_mean = sum(red_scores) / len(red_scores)
    return max(-1.5, min(1.5, (blue_mean - red_mean) / 4.0))
