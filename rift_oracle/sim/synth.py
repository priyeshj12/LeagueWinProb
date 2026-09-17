"""A generative model of a Summoner's Rift game.

This exists for two reasons.

The obvious one is that ``rift-oracle demo`` has to work with no API key and no
network, and a canned replay would be a screenshot rather than a demonstration.

The important one is that a win-probability model needs *calibrated* targets,
and those are surprisingly hard to get from a small sample of real matches.
If games are simulated all the way to the nexus falling, the label is generated
by the same stochastic process that generates the features, so the conditional
win rate given a mid-game state is a real probability rather than an artefact
of how many games happened to be in the training set. A model fit on that is
calibrated by construction, and ``rift-oracle backtest`` confirms it.

The dynamics are built out of the mechanisms that actually decide League games:

* a latent skill gap that persists all match,
* composition scaling that grows with the clock,
* snowballing, where a lead generates more lead,
* bounties, which are the game's built-in negative feedback and the reason
  comebacks exist at all,
* and a siege hazard, so a game ends when one side has enough pressure rather
  than at a fixed time.

Nothing here is fit to real data. ``rift-oracle harvest`` and
``rift-oracle train --data`` replace it with real matches; this is the
cold-start prior and the thing that makes the demo honest about what the model
is doing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from rift_oracle.game.scaling import CHAMPION_SCALE, champion_scale, ramp
from rift_oracle.game.state import (
    BLUE,
    RED,
    SOUL_AT,
    GameEvent,
    GameState,
    PlayerState,
    TeamState,
)
from rift_oracle.model.features import extract

#: Items the simulator lets players complete, with their cost.
SIM_ITEMS: List[Tuple[int, str, int]] = [
    (3031, "Infinity Edge", 3450),
    (3089, "Rabadon's Deathcap", 3600),
    (3135, "Void Staff", 3000),
    (3036, "Lord Dominik's Regards", 3000),
    (3072, "Bloodthirster", 3400),
    (3153, "Blade of the Ruined King", 3200),
    (3078, "Trinity Force", 3333),
    (3157, "Zhonya's Hourglass", 3250),
    (3026, "Guardian Angel", 3200),
    (3068, "Sunfire Aegis", 2900),
    (3065, "Spirit Visage", 2900),
    (3075, "Thornmail", 2700),
    (3074, "Ravenous Hydra", 3300),
    (3115, "Nashor's Tooth", 3000),
    (3124, "Guinsoo's Rageblade", 3000),
    (3083, "Warmog's Armor", 3100),
]

DRAKE_TYPES = ["Infernal", "Ocean", "Mountain", "Cloud", "Hextech", "Chemtech"]
LANES = ["Top", "Mid", "Bot"]
POSITIONS = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]

#: Objective spawn schedule, in minutes.
FIRST_DRAKE_MIN = 5.0
DRAKE_RESPAWN_MIN = 5.0
BARON_SPAWN_MIN = 20.0
BARON_RESPAWN_MIN = 6.0
ELDER_AFTER_SOUL_MIN = 6.0
HERALD_MIN = 8.0

TICK_S = 30.0
MAX_GAME_MIN = 52.0


@dataclass
class SimulatedGame:
    """One complete simulated match."""

    states: List[GameState]
    winner: int
    duration_s: float
    blue_champions: List[str]
    red_champions: List[str]
    skill_gap: float
    seed: int = 0

    @property
    def events(self) -> List[GameEvent]:
        return [event for state in self.states for event in state.events]

    def summary(self) -> Dict[str, Any]:
        return {
            "match_id": f"SIM_{self.seed}",
            "queue_id": 420,
            "patch": "simulated",
            "duration_s": self.duration_s,
            "winner": self.winner,
            "blue_champions": list(self.blue_champions),
            "red_champions": list(self.red_champions),
        }


class _Side:
    """Mutable per-team accumulators used during a simulation."""

    def __init__(self, team_id: int, champions: List[str]) -> None:
        self.team_id = team_id
        self.champions = champions
        self.kills = 0
        self.deaths = 0
        self.assists = 0
        self.cs = [0.0] * 5
        self.gold = [500.0] * 5
        self.spent = [0.0] * 5
        self.levels = [1.0] * 5
        self.items: List[List[int]] = [[] for _ in range(5)]
        self.item_value = [0] * 5
        self.player_kills = [0] * 5
        self.player_deaths = [0] * 5
        self.player_assists = [0] * 5
        self.dead_until = [0.0] * 5
        self.damage = [0.0] * 5

        self.towers = 0.0
        self.towers_raw = 0
        self.plates = 0
        self.inhibitors = 0
        self.inhib_down_until = 0.0
        self.dragons: List[str] = []
        self.heralds = 0.0
        self.barons = 0
        self.elders = 0
        self.baron_taken_at: Optional[float] = None
        self.elder_taken_at: Optional[float] = None
        self.soul: Optional[str] = None
        self.vision = 0.0

    @property
    def total_gold(self) -> float:
        return float(sum(self.gold))

    def buff_active(self, kind: str, now: float) -> bool:
        taken = self.baron_taken_at if kind == "baron" else self.elder_taken_at
        duration = 180.0 if kind == "baron" else 150.0
        return taken is not None and (now - taken) < duration


def simulate_game(
    rng: Optional[np.random.Generator] = None,
    seed: Optional[int] = None,
    skill_gap: Optional[float] = None,
    emit_states: bool = True,
) -> SimulatedGame:
    """Simulate one match from the loading screen to the nexus exploding."""
    if rng is None:
        rng = np.random.default_rng(seed)
    if seed is None:
        seed = int(rng.integers(0, 2**31 - 1))

    champion_pool = list(CHAMPION_SCALE.keys())
    picks = rng.choice(len(champion_pool), size=10, replace=False)
    blue_champions = [champion_pool[i] for i in picks[:5]]
    red_champions = [champion_pool[i] for i in picks[5:]]

    blue = _Side(BLUE, blue_champions)
    red = _Side(RED, red_champions)

    # Latent, persistent skill difference between the two lobbies.
    delta = float(rng.normal(0.0, 0.42)) if skill_gap is None else float(skill_gap)
    scale_edge = (
        sum(champion_scale(c) for c in blue_champions) / 5.0
        - sum(champion_scale(c) for c in red_champions) / 5.0
    )

    noise = 0.0
    states: List[GameState] = []
    pending: List[GameEvent] = []

    next_drake = FIRST_DRAKE_MIN
    next_baron = BARON_SPAWN_MIN
    next_herald = HERALD_MIN
    elder_available_at: Optional[float] = None

    t = 0.0
    winner: Optional[int] = None

    while t < MAX_GAME_MIN * 60.0:
        t += TICK_S
        minutes = t / 60.0
        dt = TICK_S / 60.0

        # -- latent edge --------------------------------------------------
        # AR(1) noise so form comes and goes in streaks rather than flickering
        # per tick. The variance is deliberately large relative to the skill
        # gap: in real games an early lead is weak evidence about the result,
        # and a simulator whose leads never decay teaches the model to be far
        # more confident at ten minutes than the data supports.
        noise = 0.90 * noise + math.sqrt(1 - 0.90**2) * float(rng.normal(0.0, 0.85))
        gold_diff = blue.total_gold - red.total_gold
        snowball = 0.42 * math.tanh(gold_diff / 7000.0)
        edge = delta + 1.15 * scale_edge * ramp(t) + noise + snowball
        for side, sign in ((blue, 1.0), (red, -1.0)):
            if side.buff_active("baron", t):
                edge += 0.55 * sign
            if side.buff_active("elder", t):
                edge += 0.75 * sign
            if side.soul:
                edge += 0.28 * sign

        # -- farm ---------------------------------------------------------
        base_cs = (7.2 if minutes < 15 else 5.4) * dt
        for side, sign in ((blue, 1.0), (red, -1.0)):
            for i in range(5):
                role_rate = base_cs * (0.45 if i == 4 else 1.0)
                gain = max(0.0, role_rate * (1.0 + 0.10 * sign * edge) + rng.normal(0, 1.2))
                side.cs[i] += gain
                side.gold[i] += gain * 21.5
                side.gold[i] += 2.04 * TICK_S if t > 110 else 0.0
                side.levels[i] = min(
                    18.0, 1.0 + 17.0 * _level_curve(minutes, side.cs[i], side.player_kills[i])
                )
            # Vision tracks the side with tempo, but weakly and only once
            # trinkets are actually being used. Early ward score carries almost
            # no information about the outcome, and a simulator that leaks the
            # latent skill gap through it would teach the model to be confident
            # ninety seconds into a game.
            if minutes > 2.0:
                side.vision += max(
                    0.0, rng.normal(1.6 + 0.18 * sign * edge, 1.5)
                ) * dt * 5

        # -- fights -------------------------------------------------------
        # Tuned so a thirty minute game lands near the ranked average of about
        # forty five total kills, with fights getting denser as the game goes on.
        fight_rate = 0.25 + 0.025 * min(minutes, 35.0)
        n_kills = int(rng.poisson(max(fight_rate, 0.0)))
        for _ in range(n_kills):
            blue_wins_trade = rng.random() < _sigmoid(1.25 * edge)
            winner_side, loser_side = (blue, red) if blue_wins_trade else (red, blue)
            event = _resolve_kill(rng, winner_side, loser_side, t, minutes, gold_diff)
            pending.append(event)
            # Damage to champions only happens in fights. Accruing it as a
            # constant trickle would make the damage-share feature meaningful
            # at ninety seconds, when in a real game it is still zero.
            for side, share in ((winner_side, 1.0), (loser_side, 0.75)):
                for i in range(5):
                    side.damage[i] += max(0.0, rng.normal(420, 220)) * share

        # -- objectives ---------------------------------------------------
        if minutes >= next_drake:
            taker = blue if rng.random() < _sigmoid(1.05 * edge) else red
            if elder_available_at is not None and minutes >= elder_available_at:
                taker.elders += 1
                taker.elder_taken_at = t
                elder_available_at = minutes + ELDER_AFTER_SOUL_MIN
                pending.append(
                    GameEvent(t, "ELDER_KILL", taker.team_id,
                              f"{_side_name(taker.team_id)} killed Elder Dragon", 6.0, {})
                )
            else:
                drake = DRAKE_TYPES[int(rng.integers(0, len(DRAKE_TYPES)))]
                taker.dragons.append(drake)
                count = len(taker.dragons)
                pending.append(
                    GameEvent(t, "DRAGON_KILL", taker.team_id,
                              f"{_side_name(taker.team_id)} took {drake} Drake (#{count})",
                              2.0 + (1.5 if count >= 3 else 0.0),
                              {"dragon": drake, "count": count})
                )
                if count >= SOUL_AT and not taker.soul:
                    taker.soul = drake
                    elder_available_at = minutes + ELDER_AFTER_SOUL_MIN
                    pending.append(
                        GameEvent(t, "DRAGON_SOUL", taker.team_id,
                                  f"{_side_name(taker.team_id)} claimed the {drake} Dragon Soul",
                                  7.0, {"soul": drake})
                    )
            next_drake = minutes + DRAKE_RESPAWN_MIN

        if minutes >= next_herald and next_herald < 25.0:
            taker = blue if rng.random() < _sigmoid(1.0 * edge) else red
            taker.heralds += 1.0
            pending.append(
                GameEvent(t, "HERALD_KILL", taker.team_id,
                          f"{_side_name(taker.team_id)} took Rift Herald", 2.0, {})
            )
            next_herald = minutes + 8.0

        if minutes >= next_baron:
            if rng.random() < 0.38:  # Baron is often left standing for a while
                taker = blue if rng.random() < _sigmoid(1.3 * edge) else red
                taker.barons += 1
                taker.baron_taken_at = t
                for i in range(5):
                    taker.gold[i] += 300.0
                pending.append(
                    GameEvent(t, "BARON_KILL", taker.team_id,
                              f"{_side_name(taker.team_id)} killed Baron Nashor", 6.0, {})
                )
                next_baron = minutes + BARON_RESPAWN_MIN
            else:
                next_baron = minutes + 1.0

        # Structures. Both sides take turrets; the side with the edge takes
        # more of them. The rate is set so about eleven fall in a thirty minute
        # game, which is what the leading team needs to reach an inhibitor.
        if minutes >= 8.0:
            tower_rate = min(0.34, 0.13 + 0.007 * (minutes - 8.0))
            if rng.random() < tower_rate:
                blue_takes = rng.random() < _sigmoid(1.7 * edge)
                taker, loser = (blue, red) if blue_takes else (red, blue)
                event = _take_structure(rng, taker, loser, t, minutes)
                if event is not None:
                    pending.append(event)

        # Turret plates exist only before fourteen minutes. Fifteen are
        # available per side and a typical game sees ten or so fall, each worth
        # 160g. Drawing them one at a time as independent Bernoulli trials
        # matters: a coarser draw would make the plate count a near-noiseless
        # readout of which side is ahead and the model would over-weight it.
        if 8.0 <= minutes < 14.0:
            for _ in range(int(rng.poisson(0.9))):
                taker = blue if rng.random() < _sigmoid(0.6 * edge) else red
                if taker.plates >= 15:
                    continue
                taker.plates += 1
                for i in range(5):
                    taker.gold[i] += 32.0
                pending.append(
                    GameEvent(t, "PLATE", taker.team_id,
                              f"{_side_name(taker.team_id)} took a turret plate (+160g)",
                              0.4, {})
                )

        # -- shopping -----------------------------------------------------
        for side in (blue, red):
            for i in range(5):
                wallet = side.gold[i] - side.spent[i]
                if wallet > 2800 and len(side.items[i]) < 6 and rng.random() < 0.45:
                    item_id, item_name, cost = SIM_ITEMS[int(rng.integers(0, len(SIM_ITEMS)))]
                    if item_id in side.items[i]:
                        continue
                    side.items[i].append(item_id)
                    side.item_value[i] += cost
                    side.spent[i] += cost
                    pending.append(
                        GameEvent(t, "ITEM_COMPLETED", side.team_id,
                                  f"{side.champions[i]} completed {item_name} ({cost}g)",
                                  1.0 + cost / 3000.0,
                                  {"item": item_name, "item_id": item_id, "cost": cost,
                                   "champion": side.champions[i]})
                    )

        # -- state snapshot ------------------------------------------------
        if emit_states:
            states.append(_snapshot(blue, red, t, list(pending)))
            pending = []

        # -- does the game end here? --------------------------------------
        # A nexus needs an inhibitor down first, so ending is gated on
        # structural progress rather than on the clock alone. The hazard rises
        # with both the size of the lead and the length of the game, which is
        # what produces a realistic length distribution instead of every match
        # ending the moment somebody goes ahead.
        pressure = _siege_pressure(blue, red, t)
        leader = blue if pressure > 0 else red
        if minutes > 15.0 and (leader.inhibitors >= 1 or minutes > 42.0):
            base = 0.018 * (1.0 + max(0.0, minutes - 22.0) / 9.0)
            if rng.random() < min(base * math.exp(2.0 * abs(pressure)), 0.5):
                winner = BLUE if pressure > 0 else RED
                break

    if winner is None:
        pressure = _siege_pressure(blue, red, t)
        winner = BLUE if pressure >= 0 else RED

    if emit_states and states:
        final = states[-1]
        final.events.append(
            GameEvent(t, "GAME_END", winner, f"{_side_name(winner)} destroyed the nexus", 10.0,
                      {"winner": winner})
        )
        for state in states:
            state.winner = winner

    return SimulatedGame(
        states=states,
        winner=winner,
        duration_s=t,
        blue_champions=blue_champions,
        red_champions=red_champions,
        skill_gap=delta,
        seed=seed,
    )


def _resolve_kill(
    rng: np.random.Generator,
    winner_side: _Side,
    loser_side: _Side,
    t: float,
    minutes: float,
    gold_diff: float,
) -> GameEvent:
    """Award a kill, including the bounty that makes comebacks possible."""
    killer = int(rng.integers(0, 5))
    victim = int(rng.integers(0, 5))
    n_assists = int(rng.integers(0, 3))

    winner_side.kills += 1
    winner_side.player_kills[killer] += 1
    loser_side.deaths += 1
    loser_side.player_deaths[victim] += 1

    # Bounties: killing the side that is ahead pays more. This is the game's
    # own negative feedback, and without it simulated leads never reverse.
    behind = (winner_side.total_gold - loser_side.total_gold) < 0
    lead_size = abs(winner_side.total_gold - loser_side.total_gold)
    bounty = 300.0
    if behind:
        bounty += min(700.0, lead_size / 12.0)

    winner_side.gold[killer] += bounty
    for i in range(5):
        if i != killer and n_assists > 0:
            winner_side.gold[i] += 145.0
            winner_side.player_assists[i] += 1
            winner_side.assists += 1
            n_assists -= 1

    respawn = _death_timer(loser_side.levels[victim], t)
    loser_side.dead_until[victim] = t + respawn

    return GameEvent(
        t=t,
        type="CHAMPION_KILL",
        team=winner_side.team_id,
        text=f"{winner_side.champions[killer]} killed {loser_side.champions[victim]}"
        + (f" ({int(bounty)}g bounty)" if behind and bounty > 400 else ""),
        importance=1.0 + bounty / 1000.0,
        payload={
            "killer_champion": winner_side.champions[killer],
            "victim_champion": loser_side.champions[victim],
            "bounty": int(bounty),
            "shutdown": int(bounty - 300) if behind else 0,
        },
    )


def _take_structure(
    rng: np.random.Generator, taker: _Side, loser: _Side, t: float, minutes: float
) -> Optional[GameEvent]:
    """Take the next structure the taker is entitled to, outer turrets first."""
    lane = LANES[int(rng.integers(0, len(LANES)))]

    # An inhibitor sits behind three turrets in its lane, so five turrets
    # overall is a reasonable gate for the first one going down.
    if (
        taker.towers_raw >= 5
        and minutes > 16.0
        and taker.inhibitors < 3
        and rng.random() < 0.45
    ):
        taker.inhibitors += 1
        loser.inhib_down_until = t + 300.0
        for i in range(5):
            taker.gold[i] += 40.0
        return GameEvent(
            t, "INHIBITOR_KILL", taker.team_id,
            f"{_side_name(taker.team_id)} destroyed the {lane} inhibitor", 3.0, {"lane": lane}
        )

    if taker.towers_raw < 3:
        tier, weight = "Outer", 1.0
    elif taker.towers_raw < 6:
        tier, weight = "Inner", 1.6
    elif taker.towers_raw < 9:
        tier, weight = "Base", 2.2
    else:
        tier, weight = "Nexus", 2.6

    taker.towers_raw += 1
    taker.towers += weight
    for i in range(5):
        taker.gold[i] += 100.0
    return GameEvent(
        t, "TURRET_KILL", taker.team_id,
        f"{_side_name(taker.team_id)} took the {lane} {tier} turret",
        1.0 + weight * 0.5, {"lane": lane, "tier": tier}
    )


def _siege_pressure(blue: _Side, red: _Side, t: float) -> float:
    """Signed measure of how close a side is to ending the game."""
    gold = math.tanh((blue.total_gold - red.total_gold) / 9000.0)
    towers = (blue.towers - red.towers) / 8.0
    inhibs = 0.9 * (
        (1.0 if red.inhib_down_until > t else 0.0) - (1.0 if blue.inhib_down_until > t else 0.0)
    )
    buffs = 0.5 * (
        (1.0 if blue.buff_active("baron", t) else 0.0)
        - (1.0 if red.buff_active("baron", t) else 0.0)
    ) + 0.6 * (
        (1.0 if blue.buff_active("elder", t) else 0.0)
        - (1.0 if red.buff_active("elder", t) else 0.0)
    )
    return gold + towers + inhibs + buffs


def _snapshot(blue: _Side, red: _Side, t: float, events: List[GameEvent]) -> GameState:
    """Freeze the mutable simulator state into a GameState."""
    teams = {}
    for side in (blue, red):
        players = []
        for i in range(5):
            dead = side.dead_until[i] > t
            players.append(
                PlayerState(
                    participant_id=(1 if side.team_id == BLUE else 6) + i,
                    team=side.team_id,
                    champion=side.champions[i],
                    position=POSITIONS[i],
                    level=int(side.levels[i]),
                    kills=side.player_kills[i],
                    deaths=side.player_deaths[i],
                    assists=side.player_assists[i],
                    cs=int(side.cs[i]),
                    total_gold=int(side.gold[i]),
                    current_gold=max(0.0, side.gold[i] - side.spent[i]),
                    xp=int(side.levels[i] * 1100),
                    items=list(side.items[i]),
                    item_value=side.item_value[i],
                    completed_items=len(side.items[i]),
                    is_dead=dead,
                    respawn_s=max(0.0, side.dead_until[i] - t) if dead else 0.0,
                    damage_to_champs=int(side.damage[i]),
                    physical_damage_to_champs=int(side.damage[i] * 0.55),
                    magic_damage_to_champs=int(side.damage[i] * 0.45),
                    vision_score=side.vision / 5.0,
                )
            )
        state = TeamState(
            team_id=side.team_id,
            players=players,
            towers=side.towers,
            towers_raw=side.towers_raw,
            turret_plates=side.plates,
            inhibitors_taken=side.inhibitors,
            inhibitors_down=1 if side.inhib_down_until > t else 0,
            dragons=list(side.dragons),
            heralds=side.heralds,
            barons=side.barons,
            elders=side.elders,
            has_soul=bool(side.soul),
            soul_type=side.soul,
            baron_taken_at=side.baron_taken_at,
            elder_taken_at=side.elder_taken_at,
        )
        teams[side.team_id] = state

    return GameState(
        t=t,
        blue=teams[BLUE],
        red=teams[RED],
        match_id=None,
        queue_id=420,
        patch="simulated",
        source="sim",
        events=events,
    )


def simulate_dataset(
    n_games: int = 4000,
    seed: int = 7,
    progress: Optional[Any] = None,
) -> Dict[str, np.ndarray]:
    """Simulate many games and return stacked feature arrays.

    Returns ``values``, ``masks``, ``times``, ``labels`` (1 when blue won) and
    ``groups`` (a game index per row, so evaluation can split by game rather
    than by row and never score a game it partly trained on).
    """
    rng = np.random.default_rng(seed)
    all_values: List[np.ndarray] = []
    all_masks: List[np.ndarray] = []
    all_times: List[float] = []
    all_labels: List[float] = []
    all_groups: List[int] = []

    for game_index in range(n_games):
        game = simulate_game(rng=rng)
        label = 1.0 if game.winner == BLUE else 0.0
        for state in game.states:
            vector = extract(state)
            all_values.append(vector.values)
            all_masks.append(vector.mask)
            all_times.append(vector.t)
            all_labels.append(label)
            all_groups.append(game_index)
        if progress is not None:
            progress(game_index + 1, n_games)

    return {
        "values": np.vstack(all_values) if all_values else np.zeros((0, 1)),
        "masks": np.vstack(all_masks) if all_masks else np.zeros((0, 1)),
        "times": np.array(all_times, dtype=np.float64),
        "labels": np.array(all_labels, dtype=np.float64),
        "groups": np.array(all_groups, dtype=np.int64),
    }


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-35.0, min(35.0, z))))


def _level_curve(minutes: float, cs: float, kills: int) -> float:
    """Fraction of the way from level 1 to level 18.

    Weighted mostly by the clock, because most experience comes from waves that
    arrive whether or not anyone is there to take them; farm and kills account
    for the gap between a fed laner and a starved one.
    """
    progress = minutes / 25.0 * 0.75 + (cs / 220.0) * 0.20 + (kills / 8.0) * 0.05
    return max(0.0, min(1.0, progress))


def _death_timer(level: float, t: float) -> float:
    base = 10.0 + 2.5 * max(0.0, level - 6.0)
    minutes = t / 60.0
    factor = 1.0
    if minutes > 15:
        factor += (min(minutes, 30) - 15) * 0.017
    if minutes > 30:
        factor += (min(minutes, 45) - 30) * 0.012
    if minutes > 45:
        factor += (minutes - 45) * 0.058
    return base * factor


def _side_name(team_id: int) -> str:
    return "Blue" if team_id == BLUE else "Red"
