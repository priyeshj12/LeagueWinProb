"""Turning attributions into sentences a player would actually say.

The model produces a number per feature. That is the wrong unit for a human:
nobody wants to hear that the objectives term moved 0.43 logits. This module
takes each attributed feature, finds the events in the window that could have
moved it, and writes the line a person would write - the event if there is one,
the underlying quantity if there is not.

Wording is deliberately concrete. "Their composition keeps scaling" is not
useful without naming the champions doing the scaling; "the gold lead moved" is
not useful without the number.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from rift_oracle.analysis.swings import (
    Swing,
    Track,
    TrackPoint,
    bundle_contributions,
    bundle_label,
    bundle_primary,
)
from rift_oracle.game.scaling import biggest_scalers, crossover_minute
from rift_oracle.game.state import BLUE, GameEvent, GameState
from rift_oracle.model.features import SPEC_BY_KEY, display_value


@dataclass
class SwingNarrative:
    """A swing, written out."""

    swing: Swing
    headline: str
    reasons: List[str] = field(default_factory=list)
    clock: str = ""

    def as_lines(self) -> List[str]:
        lines = [f"{self.clock}  {self.headline}"]
        lines.extend(f"    {reason}" for reason in self.reasons)
        return lines


def points(value: float) -> str:
    """Render a probability change in percentage points."""
    return f"{value * 100:+.1f} pts"


def percent(value: float) -> str:
    return f"{value * 100:.0f}%"


def narrate_swing(
    swing: Swing,
    track: Optional[Track] = None,
    limit: int = 4,
    perspective: int = BLUE,
) -> SwingNarrative:
    """Explain one swing: what moved, and what caused it."""
    before_state = _state_at(track, swing.start_t)
    after_state = _state_at(track, swing.end_t)

    delta = swing.delta if perspective == BLUE else -swing.delta
    before = swing.p_before if perspective == BLUE else 1 - swing.p_before
    after = swing.p_after if perspective == BLUE else 1 - swing.p_after
    gainer = swing.toward if perspective == BLUE else ("Red" if swing.delta > 0 else "Blue")

    headline = _headline(swing, gainer, before, after, delta)

    reasons: List[str] = []
    for key, share in swing.top_attributions(limit=limit):
        signed = share if perspective == BLUE else -share
        detail = _feature_detail(key, swing, before_state, after_state)
        line = f"{bundle_label(key):<22} {points(signed):>10}"
        if detail:
            line += f"   {_clip(detail)}"
        reasons.append(line)

    leftovers = _uncredited_events(swing, limit=3)
    if leftovers:
        reasons.append(f"{'also in window':<22} {'':>10}   " + "; ".join(leftovers))

    return SwingNarrative(
        swing=swing, headline=headline, reasons=reasons, clock=swing.clock()
    )


def _headline(swing: Swing, gainer: str, before: float, after: float, delta: float) -> str:
    """One line naming the swing and, when possible, its single cause."""
    movement = f"{percent(before)} -> {percent(after)}"

    # Rank by the same bundles the body reports, or the headline can credit
    # one factor while the reasons underneath rank a different one first.
    ranked = swing.top_attributions(limit=4)
    top_key = ranked[0][0] if ranked else None

    # Headline from an event that actually moved the odds, choosing the most
    # notable one among the factors that carried the swing. Ranking purely by
    # contribution would headline a routine kill for the window in which a
    # team claimed dragon soul, because three combat features sum to more than
    # the soul feature alone. Requiring a real share of the move first is what
    # stops the loudest event in the window taking credit for something else.
    candidates = [
        event
        for key, share in ranked
        if abs(share) >= swing.magnitude * 0.2
        for event in swing.causes_for(key)
    ]
    if candidates:
        best = max(candidates, key=lambda event: (event.importance, event.t))
        return f"{best.text}  ({movement}, {points(delta)} {gainer})"

    if top_key:
        spec = SPEC_BY_KEY.get(top_key)
        label = spec.noun if spec else top_key
        verb = "drifted" if swing.kind == "drift" else "moved"
        return f"{label.capitalize()} {verb} toward {gainer}  ({movement}, {points(delta)})"

    if swing.kind == "drift":
        return f"Slow drift to {gainer}  ({movement}, {points(delta)})"
    return f"Swing to {gainer}  ({movement}, {points(delta)})"


def _clip(text: str, limit: int = 62) -> str:
    """Keep a detail clause on one line inside a report panel."""
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip(" ,;") + "..."


def _feature_detail(
    key: str,
    swing: Swing,
    before_state: Optional[GameState],
    after_state: Optional[GameState],
) -> str:
    """The evidence behind one attributed feature or bundle."""
    causes = swing.causes_for(key)
    if causes and not swing.is_background(key):
        return _summarise_events(causes, key)

    if before_state is None or after_state is None:
        return ""

    if key == "economy":
        gap = after_state.gold_diff() - before_state.gold_diff()
        return f"{gap:+,.0f}g net swing"
    if key == "fight":
        kills = (after_state.blue.kills - after_state.red.kills) - (
            before_state.blue.kills - before_state.red.kills
        )
        alive = f"{after_state.blue.alive}v{after_state.red.alive} on the map"
        return f"kill lead moved {kills:+d}, {alive}"
    if key == "farm":
        gap = (after_state.blue.cs - after_state.red.cs) - (
            before_state.blue.cs - before_state.red.cs
        )
        levels = (after_state.blue.levels - after_state.red.levels) - (
            before_state.blue.levels - before_state.red.levels
        )
        return f"farm gap moved {gap:+.0f} cs, team levels {levels:+.0f}"

    if key == "scaling_diff":
        return _scaling_detail(after_state)
    if key == "cs_diff":
        gap = (after_state.blue.cs - after_state.red.cs) - (
            before_state.blue.cs - before_state.red.cs
        )
        return f"farm gap moved {gap:+.0f} cs"
    if key in ("xp_share", "level_diff"):
        gap = (after_state.blue.levels - after_state.red.levels) - (
            before_state.blue.levels - before_state.red.levels
        )
        return f"team level gap moved {gap:+.0f}"
    if key == "dmg_share":
        return "damage output diverged in this window"
    if key == "vision_diff":
        return "vision control shifted"
    if key in ("gold_share", "gold_diff_k"):
        gap = after_state.gold_diff() - before_state.gold_diff()
        return f"{gap:+,.0f}g net swing"
    if key == "item_value_diff_k":
        gap = (after_state.blue.item_value - after_state.red.item_value) - (
            before_state.blue.item_value - before_state.red.item_value
        )
        return f"{gap:+,.0f}g of completed items"
    if key == "respawn_diff":
        gap = after_state.red.respawn_total_s - after_state.blue.respawn_total_s
        return f"{gap:+.0f}s of net respawn time"
    if key == "alive_diff":
        return (
            f"{after_state.blue.alive}v{after_state.red.alive} on the map"
        )
    if key == "rank_diff":
        return "pre-game rank prior"
    return ""


def _scaling_detail(state: GameState) -> str:
    """Name the champions responsible for a composition's scaling edge."""
    blue_late = biggest_scalers(state.blue.champions, limit=2)
    red_late = biggest_scalers(state.red.champions, limit=2)
    blue_mean = sum(v for _, v in blue_late) / max(len(blue_late), 1)
    red_mean = sum(v for _, v in red_late) / max(len(red_late), 1)

    if red_mean > blue_mean:
        names = ", ".join(name for name, _ in red_late)
        return f"Red scales harder ({names})"
    names = ", ".join(name for name, _ in blue_late)
    return f"Blue scales harder ({names})"


def _summarise_events(events: Sequence[GameEvent], key: str) -> str:
    """Condense a list of events into one clause."""
    if not events:
        return ""
    if len(events) == 1:
        return events[0].text

    kills = [event for event in events if event.type == "CHAMPION_KILL"]
    if len(kills) == len(events) and kills:
        by_team = Counter(event.team for event in kills)
        parts = []
        for team, count in by_team.most_common():
            side = "Blue" if team == BLUE else "Red"
            victims = [
                str(event.payload.get("victim_champion") or event.payload.get("victim") or "")
                for event in kills
                if event.team == team
            ]
            victims = [v for v in victims if v][:3]
            clause = f"{count} kill{'s' if count > 1 else ''} for {side}"
            if victims:
                clause += f" ({', '.join(victims)} down)"
            parts.append(clause)
        return "; ".join(parts)

    items = [event for event in events if event.type == "ITEM_COMPLETED"]
    if items and len(items) == len(events):
        return "; ".join(event.text for event in items[:2]) + (
            f" (+{len(items) - 2} more)" if len(items) > 2 else ""
        )

    head = "; ".join(event.text for event in events[:2])
    return head + (f" (+{len(events) - 2} more)" if len(events) > 2 else "")


def _uncredited_events(swing: Swing, limit: int = 3) -> List[str]:
    """Notable events in the window that no top attribution already named."""
    credited = set()
    for key, _share in swing.top_attributions(limit=6):
        for event in swing.causes_for(key):
            credited.add((event.t, event.text))

    out: List[str] = []
    for event in swing.events:
        if (event.t, event.text) in credited:
            continue
        if event.importance < 2.0:
            continue
        out.append(event.text)
        if len(out) >= limit:
            break
    return out


def _state_at(track: Optional[Track], t: float) -> Optional[GameState]:
    if track is None:
        return None
    point = track.at(t)
    return point.state if point else None


def standing(point: TrackPoint, limit: int = 8) -> List[Tuple[str, str, float, float]]:
    """Bundled standing decomposition: ``(key, label, logit, raw value)``.

    Shared by the terminal report, the live dashboard and the HTML report so
    all three can never disagree about what the model is weighing.
    """
    out: List[Tuple[str, str, float, float]] = []
    for key, contribution in bundle_contributions(point.prediction.contributions)[:limit]:
        if abs(contribution) < 1e-6:
            continue
        primary = bundle_primary(key)
        out.append((primary, bundle_label(key), contribution, point.vector.get(primary)))
    return out


def narrate_state(point: TrackPoint, perspective: int = BLUE, limit: int = 5) -> List[str]:
    """Explain why the odds are what they are right now."""
    lines: List[str] = []
    for primary, label, contribution, raw in standing(point, limit=limit):
        signed = contribution if perspective == BLUE else -contribution
        favours = "you" if signed > 0 else "them"
        lines.append(
            f"{label:<22} {display_value(primary, raw):>9}   "
            f"{signed:+.2f} logit, favours {favours}"
        )
    return lines


def momentum_phrase(track: Track, window_s: float = 120.0, perspective: int = BLUE) -> str:
    """A short verdict on which way the game is trending."""
    momentum = track.momentum(window_s=window_s)
    if perspective != BLUE:
        momentum = -momentum

    minutes = int(window_s // 60)
    unit = f"{minutes} min" if minutes >= 1 else f"{int(window_s)}s"

    if abs(momentum) < 0.015:
        return f"Flat over the last {unit}"
    direction = "your way" if momentum > 0 else "against you"
    strength = "hard" if abs(momentum) > 0.08 else "gently"
    return f"Trending {strength} {direction} ({points(momentum)} in {unit})"


def describe_outcome(track: Track, perspective: int = BLUE) -> str:
    """One line on how the model did, once the winner is known."""
    if track.winner is None or not track.points:
        return ""
    won = track.winner == perspective
    final = track.points[-1]
    stated = final.p if perspective == BLUE else 1 - final.p
    verdict = "won" if won else "lost"
    return (
        f"Final call {percent(stated)}; you {verdict}. "
        f"The model was {'right' if (stated > 0.5) == won else 'wrong'} at the buzzer."
    )


def scaling_outlook(state: GameState, perspective: int = BLUE) -> str:
    """Whether the clock is a friend or an enemy from here."""
    blue_late = biggest_scalers(state.blue.champions, limit=2)
    red_late = biggest_scalers(state.red.champions, limit=2)
    blue_mean = sum(v for _, v in blue_late) / max(len(blue_late), 1)
    red_mean = sum(v for _, v in red_late) / max(len(red_late), 1)

    ours, theirs = (blue_mean, red_mean) if perspective == BLUE else (red_mean, blue_mean)
    their_carries = red_late if perspective == BLUE else blue_late
    our_carries = blue_late if perspective == BLUE else red_late
    crossover = crossover_minute(state.blue.champions, state.red.champions)
    minutes = state.minutes

    if abs(ours - theirs) < 0.10:
        return "Neither composition out-scales the other; the clock is neutral."

    if theirs > ours:
        names = ", ".join(name for name, _ in their_carries)
        if crossover and minutes < crossover:
            return (
                f"They out-scale you ({names}). Their curve takes over around "
                f"{crossover:.0f}:00 - you have about "
                f"{max(0.0, crossover - minutes):.0f} minutes of tempo left."
            )
        return f"They out-scale you ({names}) and the game is past the crossover. Force it now."

    names = ", ".join(name for name, _ in our_carries)
    if crossover and minutes < crossover:
        return (
            f"You out-scale them ({names}), but not until about {crossover:.0f}:00. "
            "Survive the next few minutes and the game tilts your way."
        )
    return f"You out-scale them ({names}) and you are past the crossover. Play for time."
