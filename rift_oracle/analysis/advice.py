"""What to do about it: counterfactual advice from the same model.

Every suggestion here is a real counterfactual, not a rule of thumb. The
current game state is copied, one thing is changed - a dragon taken, an item
finished, an enemy caught out - and the model is asked again. The difference
between the two answers is the suggestion's value, in the same percentage
points the rest of the tool reports.

That has a property a hand-written advice table cannot have: it is automatically
situational. Baron is worth a lot when you can use it and little when you are
already closing; a fifth dragon is worth almost nothing but the fourth one is
worth a great deal because it carries the soul. Nothing here encodes those
rules. They fall out of asking the model.

The risk list is the same computation pointed the other way: what the enemy
gains from each of those actions, so the largest number tells you what to deny.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Callable, List, Tuple

from rift_oracle.game.state import BLUE, SOUL_AT, GameState, TeamState, other_team
from rift_oracle.model.features import extract
from rift_oracle.model.gam import AdditiveWinModel

#: Champions whose kit makes grievous wounds worth buying early.
HIGH_SUSTAIN = frozenset(
    {
        "Aatrox", "Vladimir", "Soraka", "Yuumi", "Dr. Mundo", "Warwick", "Sylas",
        "Swain", "Nasus", "Sett", "Zac", "Volibear", "Briar", "Fiora", "Olaf",
        "Trundle", "Samira", "Aphelios", "Senna", "Sion", "Irelia", "Gwen",
        "Mordekaiser", "Yorick", "Ryze", "Nilah", "Yone", "Master Yi", "Kayn",
    }
)

#: Items whose presence on an enemy signals a healing-heavy game.
LIFESTEAL_ITEMS = frozenset({3072, 3153, 3074, 6673, 6630, 6632, 3748, 6675, 6609})
#: Armor-stacking items, which is when penetration beats raw damage.
ARMOR_STACK_ITEMS = frozenset({3075, 3110, 3143, 3047, 3068, 3193, 8001})
MR_STACK_ITEMS = frozenset({3065, 3102, 3111, 3156, 3139, 3194, 4401})


@dataclass
class Suggestion:
    """One action, valued by how much it moves the odds."""

    action: str
    delta_p: float
    rationale: str = ""
    category: str = "objective"
    difficulty: str = "medium"

    @property
    def points(self) -> float:
        return self.delta_p * 100.0

    def __str__(self) -> str:
        return f"{self.action}: {self.points:+.1f} pts"


@dataclass
class Advice:
    """Everything the advisor has to say about one moment."""

    actions: List[Suggestion] = field(default_factory=list)
    risks: List[Suggestion] = field(default_factory=list)
    build: List[str] = field(default_factory=list)
    tempo: str = ""

    def top_actions(self, limit: int = 4) -> List[Suggestion]:
        return [s for s in self.actions if s.delta_p > 0.002][:limit]

    def top_risks(self, limit: int = 3) -> List[Suggestion]:
        return [s for s in self.risks if s.delta_p < -0.002][:limit]


# -- state mutations --------------------------------------------------------


def _clone(state: GameState) -> GameState:
    return copy.deepcopy(state)


def _take_dragon(state: GameState, team: int) -> None:
    side = state.team(team)
    side.dragons.append("Drake")
    if len(side.dragons) >= SOUL_AT:
        side.has_soul = True
        side.soul_type = side.soul_type or "Drake"


def _take_baron(state: GameState, team: int) -> None:
    side = state.team(team)
    side.barons += 1
    side.baron_taken_at = state.t
    for player in side.players:
        player.total_gold += 300


def _take_elder(state: GameState, team: int) -> None:
    side = state.team(team)
    side.elders += 1
    side.elder_taken_at = state.t


def _take_herald(state: GameState, team: int) -> None:
    state.team(team).heralds += 1.0


def _take_turret(state: GameState, team: int, weight: float) -> None:
    side = state.team(team)
    side.towers += weight
    side.towers_raw += 1
    for player in side.players:
        player.total_gold += 100


def _take_inhibitor(state: GameState, team: int) -> None:
    side = state.team(team)
    enemy = state.team(other_team(team))
    side.inhibitors_taken += 1
    enemy.inhibitors_down += 1


def _pick_off(state: GameState, team: int, count: int = 1) -> None:
    """Kill ``count`` living enemies, with gold, respawn timers and bodies."""
    side = state.team(team)
    enemy = state.team(other_team(team))
    from rift_oracle.game.timeline_adapter import death_timer

    victims = [p for p in enemy.players if not p.is_dead]
    victims.sort(key=lambda p: p.total_gold, reverse=True)  # the fed ones first

    for victim in victims[:count]:
        victim.is_dead = True
        victim.respawn_s = death_timer(victim.level, state.t)
        victim.deaths += 1
        if side.players:
            side.players[0].kills += 1
            side.players[0].total_gold += 300


def _spend_gold(state: GameState, team: int) -> float:
    """Convert unspent gold into completed items. Returns the gold converted."""
    side = state.team(team)
    converted = 0.0
    for player in side.players:
        # Only whole legendary items are a real power spike, so round down.
        spendable = int(player.current_gold // 1000) * 1000
        if spendable <= 0:
            continue
        player.item_value += spendable
        player.current_gold -= spendable
        converted += spendable
    return converted


# -- the advisor ------------------------------------------------------------


def _evaluate(
    state: GameState,
    model: AdditiveWinModel,
    perspective: int,
    rank_prior: float,
) -> float:
    """P(perspective team wins) for a state."""
    prediction = model.predict(extract(state, rank_prior=rank_prior))
    return prediction.p if perspective == BLUE else 1.0 - prediction.p


def _counterfactual(
    state: GameState,
    model: AdditiveWinModel,
    perspective: int,
    rank_prior: float,
    mutate: Callable[[GameState], None],
) -> float:
    """Probability change from applying one mutation."""
    baseline = _evaluate(state, model, perspective, rank_prior)
    candidate = _clone(state)
    mutate(candidate)
    return _evaluate(candidate, model, perspective, rank_prior) - baseline


def suggest_actions(
    state: GameState,
    model: AdditiveWinModel,
    perspective: int = BLUE,
    rank_prior: float = 0.0,
    include_risks: bool = True,
) -> Tuple[List[Suggestion], List[Suggestion]]:
    """Rank what you could do, and what they could do to you."""
    us = perspective
    them = other_team(perspective)
    our_side = state.team(us)
    their_side = state.team(them)
    minutes = state.minutes

    def value(mutate: Callable[[GameState], None]) -> float:
        return _counterfactual(state, model, us, rank_prior, mutate)

    actions: List[Suggestion] = []

    # -- dragons ---------------------------------------------------------
    next_drake = len(our_side.dragons) + 1
    drake_label = "Dragon Soul" if next_drake >= SOUL_AT else f"Dragon #{next_drake}"
    actions.append(
        Suggestion(
            action=f"Take {drake_label}",
            delta_p=value(lambda s: _take_dragon(s, us)),
            rationale=(
                "This one carries the soul."
                if next_drake >= SOUL_AT
                else f"You are on {len(our_side.dragons)} drakes."
            ),
            category="objective",
            difficulty="medium",
        )
    )

    if minutes >= 18.0:
        actions.append(
            Suggestion(
                action="Take Baron Nashor",
                delta_p=value(lambda s: _take_baron(s, us)),
                rationale="Baron is up and the buff converts directly into structures.",
                category="objective",
                difficulty="hard",
            )
        )

    if our_side.has_soul or their_side.has_soul:
        actions.append(
            Suggestion(
                action="Take Elder Dragon",
                delta_p=value(lambda s: _take_elder(s, us)),
                rationale="Elder's execute usually ends the next fight outright.",
                category="objective",
                difficulty="hard",
            )
        )

    if minutes < 20.0:
        actions.append(
            Suggestion(
                action="Take Rift Herald",
                delta_p=value(lambda s: _take_herald(s, us)),
                rationale="Herald turns into plates and tempo before it expires.",
                category="objective",
                difficulty="easy",
            )
        )

    # -- structures --------------------------------------------------------
    tier, weight = _next_turret_tier(our_side)
    actions.append(
        Suggestion(
            action=f"Take {_article(tier)} {tier} turret",
            delta_p=value(lambda s: _take_turret(s, us, weight)),
            rationale="Structures are the only lead that cannot be traded back.",
            category="objective",
            difficulty="easy" if tier == "outer" else "medium",
        )
    )

    if our_side.towers_raw >= 5 and minutes >= 15.0:
        actions.append(
            Suggestion(
                action="Break an inhibitor",
                delta_p=value(lambda s: _take_inhibitor(s, us)),
                rationale="Super minions hold the map for you while it is down.",
                category="objective",
                difficulty="hard",
            )
        )

    # -- fights ------------------------------------------------------------
    living_enemies = [p for p in their_side.players if not p.is_dead]
    if living_enemies:
        fed = max(living_enemies, key=lambda p: p.total_gold)
        actions.append(
            Suggestion(
                action=f"Catch out {fed.champion}",
                delta_p=value(lambda s: _pick_off(s, us, 1)),
                rationale=f"Their biggest wallet ({fed.total_gold:,}g, {fed.kda}).",
                category="fight",
                difficulty="medium",
            )
        )
    if len(living_enemies) >= 3:
        actions.append(
            Suggestion(
                action="Win the next teamfight 3-for-0",
                delta_p=value(lambda s: _pick_off(s, us, 3)),
                rationale="Three dead is enough time to take anything on the map.",
                category="fight",
                difficulty="hard",
            )
        )

    # -- shopping ----------------------------------------------------------
    unspent = our_side.unspent_gold
    if unspent >= 1200:
        delta = value(lambda s: _spend_gold(s, us))
        actions.append(
            Suggestion(
                action=f"Recall and spend {unspent:,.0f}g",
                delta_p=delta,
                rationale="Gold in your pocket does nothing. Gold in an item wins fights.",
                category="item",
                difficulty="easy",
            )
        )

    actions.sort(key=lambda s: s.delta_p, reverse=True)

    risks: List[Suggestion] = []
    if include_risks:
        risks = _enemy_risks(state, model, us, them, rank_prior, minutes)

    return actions, risks


def _enemy_risks(
    state: GameState,
    model: AdditiveWinModel,
    us: int,
    them: int,
    rank_prior: float,
    minutes: float,
) -> List[Suggestion]:
    """The same counterfactuals run for the enemy: what you must deny."""

    def value(mutate: Callable[[GameState], None]) -> float:
        return _counterfactual(state, model, us, rank_prior, mutate)

    their_side = state.team(them)
    risks: List[Suggestion] = []

    next_drake = len(their_side.dragons) + 1
    risks.append(
        Suggestion(
            action=f"They take {'Soul' if next_drake >= SOUL_AT else f'Dragon #{next_drake}'}",
            delta_p=value(lambda s: _take_dragon(s, them)),
            rationale="Contest or trade it for something on the other side of the map.",
            category="risk",
        )
    )
    if minutes >= 18.0:
        risks.append(
            Suggestion(
                action="They take Baron",
                delta_p=value(lambda s: _take_baron(s, them)),
                rationale="Ward it by 19:00 and do not face-check the pit.",
                category="risk",
            )
        )
    tier, weight = _next_turret_tier(their_side)
    risks.append(
        Suggestion(
            action=f"They take {_article(tier)} {tier} turret",
            delta_p=value(lambda s: _take_turret(s, them, weight)),
            rationale="Structures do not come back.",
            category="risk",
        )
    )
    our_living = [p for p in state.team(us).players if not p.is_dead]
    if our_living:
        risks.append(
            Suggestion(
                action="You get caught out solo",
                delta_p=value(lambda s: _pick_off(s, them, 1)),
                rationale="One bad ward-clear before an objective is the usual way this starts.",
                category="risk",
            )
        )

    risks.sort(key=lambda s: s.delta_p)
    return risks


def _article(word: str) -> str:
    return "an" if word[:1].lower() in "aeiou" else "a"


def _next_turret_tier(side: TeamState) -> Tuple[str, float]:
    """Which turret tier this side is realistically hitting next."""
    if side.towers_raw < 3:
        return "outer", 1.0
    if side.towers_raw < 6:
        return "inner", 1.6
    if side.towers_raw < 9:
        return "base", 2.2
    return "nexus", 2.6


# -- build advice -----------------------------------------------------------


def build_advice(state: GameState, perspective: int = BLUE) -> List[str]:
    """Itemisation notes from the enemy's damage profile and build."""
    from rift_oracle.riot.ddragon import (
        ANTIHEAL_ITEMS,
        ARMOR_ITEMS,
        ARMOR_PEN_ITEMS,
        MAGIC_PEN_ITEMS,
        MR_ITEMS,
    )

    us = state.team(perspective)
    them = state.team(other_team(perspective))
    notes: List[str] = []

    physical, magic = them.damage_split()
    if physical >= 0.68:
        names = ", ".join(list(ARMOR_ITEMS.values())[:3])
        notes.append(
            f"Their damage is {physical * 100:.0f}% physical - armor is the efficient buy ({names})."
        )
    elif magic >= 0.68:
        names = ", ".join(list(MR_ITEMS.values())[:3])
        notes.append(
            f"Their damage is {magic * 100:.0f}% magic - magic resist is the efficient buy ({names})."
        )
    else:
        notes.append(
            f"Their damage is mixed ({physical * 100:.0f}% physical / {magic * 100:.0f}% magic) - "
            "one resist item each, then health."
        )

    # Grievous wounds, if their comp or their build heals.
    healers = [c for c in them.champions if c in HIGH_SUSTAIN]
    lifesteal = [
        player.champion
        for player in them.players
        if set(player.items) & LIFESTEAL_ITEMS
    ]
    already_have = any(set(player.items) & set(ANTIHEAL_ITEMS) for player in us.players)
    if (healers or lifesteal) and not already_have:
        who = ", ".join(list(dict.fromkeys(healers + lifesteal))[:3])
        notes.append(
            f"Buy grievous wounds - {who} out-heals your burst without it "
            f"({', '.join(list(ANTIHEAL_ITEMS.values())[:3])})."
        )

    # Penetration, if they have started stacking resists against you.
    armor_stackers = sum(
        1 for player in them.players if set(player.items) & ARMOR_STACK_ITEMS
    )
    mr_stackers = sum(1 for player in them.players if set(player.items) & MR_STACK_ITEMS)
    our_physical, our_magic = us.damage_split()

    if armor_stackers >= 2 and our_physical > 0.55:
        notes.append(
            f"{armor_stackers} of them are stacking armor - "
            f"{', '.join(list(ARMOR_PEN_ITEMS.values())[:2])} beats more raw AD."
        )
    if mr_stackers >= 2 and our_magic > 0.55:
        notes.append(
            f"{mr_stackers} of them are building magic resist - "
            f"{', '.join(list(MAGIC_PEN_ITEMS.values())[:2])} beats more raw AP."
        )

    unspent = us.unspent_gold
    if unspent >= 1500:
        notes.append(f"Your team is sitting on {unspent:,.0f}g unspent. Back and buy.")

    return notes


def advise(
    state: GameState,
    model: AdditiveWinModel,
    perspective: int = BLUE,
    rank_prior: float = 0.0,
) -> Advice:
    """Full advice bundle for one moment."""
    from rift_oracle.analysis.narrate import scaling_outlook

    actions, risks = suggest_actions(state, model, perspective, rank_prior)
    return Advice(
        actions=actions,
        risks=risks,
        build=build_advice(state, perspective),
        tempo=scaling_outlook(state, perspective),
    )
