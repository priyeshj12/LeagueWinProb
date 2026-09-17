"""Finding the moments that decided the game, and what caused them.

A swing is a stretch of game time over which the win probability moved more
than it usually does. Detection is deliberately simple - a threshold on the
change across a sliding window, then a greedy pick of the largest
non-overlapping windows - because the interesting work is not finding the
swings but explaining them, and a clever detector would make the explanation
harder to trust rather than easier.

Attribution is exact. The model's logit is a sum of per-feature terms, so the
difference between two moments is the sum of the differences of those terms,
and :func:`~rift_oracle.model.gam.delta_attribution` rescales that split so it
sums to the real probability change. Nothing is approximated and nothing is
left over. The game events inside the window are then joined onto the features
they moved, which is what turns "the objectives term fell 0.4" into "Red took
Baron at 24:37".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from rift_oracle.game.state import GameEvent, GameState, format_clock
from rift_oracle.model.features import FeatureVector, SPEC_BY_KEY, extract
from rift_oracle.model.gam import AdditiveWinModel, Prediction, delta_attribution

#: Which event types can move which feature. Used to attach a cause to an
#: attributed feature rather than guessing from timing alone.
FEATURE_CAUSES: Dict[str, Tuple[str, ...]] = {
    "drake_diff": ("DRAGON_KILL",),
    "soul_diff": ("DRAGON_SOUL", "DRAGON_KILL"),
    "elder_diff": ("ELDER_KILL",),
    "baron_diff": ("BARON_KILL",),
    "herald_diff": ("HERALD_KILL",),
    "tower_diff": ("TURRET_KILL", "FIRST_TOWER"),
    "plate_diff": ("PLATE",),
    "inhib_diff": ("INHIBITOR_KILL",),
    "kill_diff": ("CHAMPION_KILL", "MULTIKILL", "ACE", "FIRST_BLOOD"),
    "alive_diff": ("CHAMPION_KILL", "ACE", "MULTIKILL"),
    "respawn_diff": ("CHAMPION_KILL", "ACE"),
    "item_value_diff_k": ("ITEM_COMPLETED",),
    "gold_share": ("CHAMPION_KILL", "TURRET_KILL", "PLATE", "BARON_KILL", "INHIBITOR_KILL"),
    "gold_diff_k": ("CHAMPION_KILL", "TURRET_KILL", "PLATE", "BARON_KILL", "INHIBITOR_KILL"),
}

#: Features that move continuously rather than in response to a single event.
BACKGROUND_FEATURES = frozenset(
    {"scaling_diff", "cs_diff", "xp_share", "level_diff", "dmg_share", "vision_diff", "rank_diff"}
)

#: Features the model keeps separate but a person thinks of as one thing.
#:
#: The model wants several encodings of the economy - a relative gold share and
#: an absolute gold lead - because they behave differently at minute ten and
#: minute thirty. A reader does not: they want one line that says "gold lead".
#: Reporting the raw features would print "gold lead" twice with two different
#: numbers, which reads as a bug. Bundles are summed for display only; the
#: underlying decomposition is untouched and still exact.
#:
#: Each entry is ``key -> (label, member features, the member whose raw value
#: is worth showing)``.
BUNDLES: Dict[str, Tuple[str, Tuple[str, ...], str]] = {
    "economy": ("gold lead", ("gold_share", "gold_diff_k"), "gold_diff_k"),
    "fight": ("the fight", ("kill_diff", "alive_diff", "respawn_diff"), "kill_diff"),
    "farm": ("farm and levels", ("cs_diff", "xp_share", "level_diff"), "cs_diff"),
}

_BUNDLE_OF: Dict[str, str] = {
    feature: key for key, (_label, features, _primary) in BUNDLES.items() for feature in features
}


def bundle_of(feature: str) -> str:
    """The reporting bundle a feature belongs to, or the feature itself."""
    return _BUNDLE_OF.get(feature, feature)


def bundle_label(key: str) -> str:
    """Display name for a bundle key or a bare feature key."""
    if key in BUNDLES:
        return BUNDLES[key][0]
    spec = SPEC_BY_KEY.get(key)
    return spec.noun if spec else key


def bundle_features(key: str) -> Tuple[str, ...]:
    return BUNDLES[key][1] if key in BUNDLES else (key,)


def bundle_primary(key: str) -> str:
    """The member feature whose raw value stands for the bundle on screen."""
    return BUNDLES[key][2] if key in BUNDLES else key


def bundle_contributions(contributions: Dict[str, float]) -> List[Tuple[str, float]]:
    """Sum a per-feature mapping into bundles, largest magnitude first."""
    totals: Dict[str, float] = {}
    for feature, value in contributions.items():
        totals[bundle_of(feature)] = totals.get(bundle_of(feature), 0.0) + value
    return sorted(totals.items(), key=lambda kv: abs(kv[1]), reverse=True)


@dataclass
class TrackPoint:
    """One evaluated moment: the state, its features, and the prediction."""

    state: GameState
    vector: FeatureVector
    prediction: Prediction

    @property
    def t(self) -> float:
        return self.state.t

    @property
    def p(self) -> float:
        return self.prediction.p


@dataclass
class Track:
    """A whole game, evaluated moment by moment."""

    points: List[TrackPoint] = field(default_factory=list)
    match_id: Optional[str] = None
    winner: Optional[int] = None
    source: str = "unknown"

    def __len__(self) -> int:
        return len(self.points)

    def __iter__(self):
        return iter(self.points)

    @property
    def times(self) -> List[float]:
        return [point.t for point in self.points]

    @property
    def probabilities(self) -> List[float]:
        return [point.p for point in self.points]

    @property
    def latest(self) -> Optional[TrackPoint]:
        return self.points[-1] if self.points else None

    def append(self, point: TrackPoint) -> None:
        self.points.append(point)

    def events(self) -> List[GameEvent]:
        return [event for point in self.points for event in point.state.events]

    def at(self, t: float) -> Optional[TrackPoint]:
        """The last point at or before ``t``."""
        best: Optional[TrackPoint] = None
        for point in self.points:
            if point.t <= t:
                best = point
            else:
                break
        return best

    def momentum(self, window_s: float = 120.0) -> float:
        """Probability change over the last ``window_s`` seconds."""
        if len(self.points) < 2:
            return 0.0
        now = self.points[-1]
        for point in reversed(self.points):
            if now.t - point.t >= window_s:
                return now.p - point.p
        return now.p - self.points[0].p

    def volatility(self) -> float:
        """Mean absolute step in probability, a proxy for how close the game is."""
        if len(self.points) < 2:
            return 0.0
        steps = [
            abs(self.points[i].p - self.points[i - 1].p) for i in range(1, len(self.points))
        ]
        return sum(steps) / len(steps)

    def peak(self) -> Tuple[float, float]:
        """Highest and lowest win probability blue ever held."""
        if not self.points:
            return 0.5, 0.5
        values = self.probabilities
        return max(values), min(values)


def build_track(
    states: Sequence[GameState],
    model: AdditiveWinModel,
    rank_prior: float = 0.0,
    winner: Optional[int] = None,
) -> Track:
    """Evaluate every state in a game."""
    vectors = [extract(state, rank_prior=rank_prior) for state in states]
    predictions = model.predict_series(vectors)
    track = Track(
        points=[
            TrackPoint(state=state, vector=vector, prediction=prediction)
            for state, vector, prediction in zip(states, vectors, predictions)
        ],
        match_id=states[0].match_id if states else None,
        winner=winner if winner is not None else (states[-1].winner if states else None),
        source=states[0].source if states else "unknown",
    )
    return track


@dataclass
class Swing:
    """A stretch of game time over which the odds moved meaningfully."""

    start_t: float
    end_t: float
    p_before: float
    p_after: float
    attributions: List[Tuple[str, float]] = field(default_factory=list)
    events: List[GameEvent] = field(default_factory=list)
    kind: str = "spike"  # spike | drift

    @property
    def delta(self) -> float:
        return self.p_after - self.p_before

    @property
    def magnitude(self) -> float:
        return abs(self.delta)

    @property
    def duration(self) -> float:
        return self.end_t - self.start_t

    @property
    def toward(self) -> str:
        return "Blue" if self.delta > 0 else "Red"

    def clock(self) -> str:
        if self.duration < 25.0:
            return format_clock(self.end_t)
        return f"{format_clock(self.start_t)}-{format_clock(self.end_t)}"

    def top_attributions(self, limit: int = 4) -> List[Tuple[str, float]]:
        """Bundled attributions, largest first. This is what gets reported."""
        bundled = bundle_contributions(dict(self.attributions))
        return [item for item in bundled[:limit] if abs(item[1]) > 0.002]

    def raw_attributions(self, limit: int = 8) -> List[Tuple[str, float]]:
        """Unbundled, per-feature attributions, for when the detail matters."""
        return [item for item in self.attributions[:limit] if abs(item[1]) > 0.001]

    def causes_for(self, key: str) -> List[GameEvent]:
        """Events in this window that plausibly moved a feature or bundle."""
        allowed: set = set()
        for feature in bundle_features(key):
            allowed.update(FEATURE_CAUSES.get(feature, ()))
        if not allowed:
            return []
        return [event for event in self.events if event.type in allowed]

    def is_background(self, key: str) -> bool:
        """True when nothing in this bundle is driven by a discrete event."""
        return all(feature in BACKGROUND_FEATURES for feature in bundle_features(key))


def detect_swings(
    track: Track,
    threshold: float = 0.06,
    window_s: float = 90.0,
    max_swings: int = 40,
) -> List[Swing]:
    """Find the non-overlapping windows where the odds moved most.

    Every window of up to ``window_s`` whose probability change clears
    ``threshold`` is a candidate. Candidates are taken largest first and any
    that overlap an already-taken window are dropped, so a single Baron does
    not get reported four times at four slightly different offsets.
    """
    points = track.points
    if len(points) < 2:
        return []

    candidates: List[Tuple[float, int, int]] = []
    for end in range(1, len(points)):
        for start in range(end - 1, -1, -1):
            span = points[end].t - points[start].t
            if span > window_s:
                break
            delta = points[end].p - points[start].p
            if abs(delta) >= threshold:
                candidates.append((abs(delta), start, end))

    candidates.sort(key=lambda item: item[0], reverse=True)

    taken: List[Tuple[int, int]] = []
    swings: List[Swing] = []

    for _magnitude, start, end in candidates:
        if len(swings) >= max_swings:
            break
        if any(not (end <= a or start >= b) for a, b in taken):
            continue
        taken.append((start, end))
        swings.append(_build_swing(track, start, end))

    swings.sort(key=lambda swing: swing.start_t)
    return swings


def _build_swing(track: Track, start: int, end: int) -> Swing:
    before = track.points[start]
    after = track.points[end]

    _delta_p, per_feature = delta_attribution(before.prediction, after.prediction)
    ranked = sorted(per_feature.items(), key=lambda kv: abs(kv[1]), reverse=True)

    events = [
        event
        for index in range(start + 1, end + 1)
        for event in track.points[index].state.events
        if event.type not in ("GAME_START", "MINIONS_SPAWNING")
    ]
    events.sort(key=lambda event: (-event.importance, event.t))

    duration = after.t - before.t
    kind = "spike" if (end - start <= 2 or duration <= 35.0) else "drift"

    return Swing(
        start_t=before.t,
        end_t=after.t,
        p_before=before.p,
        p_after=after.p,
        attributions=ranked,
        events=events,
        kind=kind,
    )


def biggest_swing(swings: Sequence[Swing]) -> Optional[Swing]:
    """The single moment that moved the game most."""
    return max(swings, key=lambda swing: swing.magnitude) if swings else None


def swing_summary(swings: Sequence[Swing]) -> Dict[str, float]:
    """Aggregate statistics over a game's swings."""
    if not swings:
        return {"count": 0, "largest": 0.0, "to_blue": 0, "to_red": 0}
    return {
        "count": len(swings),
        "largest": max(swing.magnitude for swing in swings),
        "to_blue": sum(1 for swing in swings if swing.delta > 0),
        "to_red": sum(1 for swing in swings if swing.delta < 0),
        "total_movement": sum(swing.magnitude for swing in swings),
    }


def feature_totals(track: Track) -> Dict[str, float]:
    """Net logit each feature contributed at the final evaluated moment."""
    if not track.points:
        return {}
    return dict(track.points[-1].prediction.contributions)


def contribution_history(track: Track, feature: str) -> List[Tuple[float, float]]:
    """``(time, contribution)`` for one feature, for plotting."""
    return [
        (point.t, point.prediction.contributions.get(feature, 0.0)) for point in track.points
    ]
