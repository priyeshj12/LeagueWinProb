"""Canonical game state.

The Live Client Data API and the Match-V5 timeline describe the same game with
completely different vocabularies. Both are adapted into the structures here so
that the model, the swing detector, and the advice engine only ever see one
shape. Anything a given source cannot observe stays ``None`` rather than being
guessed at, and the feature extractor treats ``None`` as "no evidence" instead
of as zero.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

BLUE = 100
RED = 200

#: Baron Nashor's buff duration in seconds.
BARON_BUFF_S = 180.0
#: Elder Dragon's execute buff duration in seconds.
ELDER_BUFF_S = 150.0
#: Dragons a team needs for its soul.
SOUL_AT = 4

#: Structural value of each turret tier, relative to an outer turret.
#: Inner and base turrets gate map control far more than outer ones, and the
#: nexus turrets mean the game is nearly over, so the weights are not linear.
TURRET_WEIGHT: Dict[str, float] = {
    "OUTER_TURRET": 1.0,
    "INNER_TURRET": 1.6,
    "BASE_TURRET": 2.2,
    "NEXUS_TURRET": 2.6,
    "UNKNOWN": 1.2,
}

#: Non-drake epic monsters and what each is worth relative to a Rift Herald.
HERALD_WEIGHT: Dict[str, float] = {
    "RIFTHERALD": 1.0,
    "HORDE": 0.25,  # a single voidgrub; six of them roughly equal a herald
    "ATAKHAN": 1.4,
}


def other_team(team: int) -> int:
    return RED if team == BLUE else BLUE


def team_name(team: int) -> str:
    return "Blue" if team == BLUE else "Red"


@dataclass
class PlayerState:
    """One participant at one instant."""

    participant_id: int
    team: int
    champion: str = "Unknown"
    champion_id: Optional[int] = None
    puuid: Optional[str] = None
    riot_id: Optional[str] = None
    position: str = ""

    level: int = 1
    kills: int = 0
    deaths: int = 0
    assists: int = 0
    cs: int = 0
    total_gold: int = 500
    current_gold: float = 0.0
    xp: int = 0

    items: List[int] = field(default_factory=list)
    item_value: int = 0
    completed_items: int = 0

    is_dead: bool = False
    respawn_s: float = 0.0

    damage_to_champs: int = 0
    physical_damage_to_champs: int = 0
    magic_damage_to_champs: int = 0
    true_damage_to_champs: int = 0
    vision_score: float = 0.0

    # Live-only combat stats, used for the build-advice heuristics.
    attack_damage: float = 0.0
    ability_power: float = 0.0
    armor: float = 0.0
    magic_resist: float = 0.0
    health_max: float = 0.0

    @property
    def label(self) -> str:
        return self.champion or (self.riot_id or f"P{self.participant_id}")

    @property
    def kda(self) -> str:
        return f"{self.kills}/{self.deaths}/{self.assists}"


@dataclass
class TeamState:
    """One side's aggregate state at one instant."""

    team_id: int
    players: List[PlayerState] = field(default_factory=list)

    # Objectives. ``towers`` is the weighted count this team has *taken*.
    towers: float = 0.0
    towers_raw: int = 0
    turret_plates: int = 0
    inhibitors_taken: int = 0
    inhibitors_down: int = 0  # of this team's own inhibitors, currently destroyed
    dragons: List[str] = field(default_factory=list)
    heralds: float = 0.0
    barons: int = 0
    elders: int = 0
    has_soul: bool = False
    soul_type: Optional[str] = None

    # Timestamps (seconds of game time) of the most recent buff pickups.
    baron_taken_at: Optional[float] = None
    elder_taken_at: Optional[float] = None

    # Pre-game prior.
    avg_rank_score: Optional[float] = None

    def sum(self, attr: str) -> float:
        return float(sum(getattr(p, attr, 0) or 0 for p in self.players))

    @property
    def gold(self) -> float:
        return self.sum("total_gold")

    @property
    def unspent_gold(self) -> float:
        return self.sum("current_gold")

    @property
    def xp(self) -> float:
        return self.sum("xp")

    @property
    def kills(self) -> int:
        return int(self.sum("kills"))

    @property
    def deaths(self) -> int:
        return int(self.sum("deaths"))

    @property
    def cs(self) -> int:
        return int(self.sum("cs"))

    @property
    def levels(self) -> int:
        return int(self.sum("level"))

    @property
    def item_value(self) -> float:
        return self.sum("item_value")

    @property
    def damage_to_champs(self) -> float:
        return self.sum("damage_to_champs")

    @property
    def vision(self) -> float:
        return self.sum("vision_score")

    @property
    def alive(self) -> int:
        return sum(1 for p in self.players if not p.is_dead)

    @property
    def respawn_total_s(self) -> float:
        return self.sum("respawn_s")

    @property
    def dragon_count(self) -> int:
        return len(self.dragons)

    @property
    def champions(self) -> List[str]:
        return [p.champion for p in self.players]

    def buff_remaining(self, kind: str, now_s: float) -> float:
        """Fraction of a Baron or Elder buff still active, in ``[0, 1]``.

        Returned as a fraction rather than a boolean because a Baron with 20
        seconds left is worth far less than one just taken, and the model
        should see that difference.
        """
        if kind == "baron":
            taken, duration = self.baron_taken_at, BARON_BUFF_S
        else:
            taken, duration = self.elder_taken_at, ELDER_BUFF_S
        if taken is None:
            return 0.0
        remaining = duration - (now_s - taken)
        if remaining <= 0:
            return 0.0
        return min(1.0, remaining / duration)

    def damage_split(self) -> Tuple[float, float]:
        """(physical share, magic share) of damage dealt to champions.

        Falls back to the live AD/AP stat lines early in the game, when nobody
        has dealt enough damage for the ratio to mean anything.
        """
        phys = self.sum("physical_damage_to_champs")
        magic = self.sum("magic_damage_to_champs")
        total = phys + magic
        if total > 2000:
            return phys / total, magic / total

        ad = self.sum("attack_damage")
        ap = self.sum("ability_power")
        if ad + ap > 0:
            # AP contributes less damage per point than AD does, empirically
            # around 0.75x once ratios and cooldowns are accounted for.
            weighted_ap = ap * 0.75
            denom = ad + weighted_ap
            return ad / denom, weighted_ap / denom
        return 0.5, 0.5


@dataclass
class GameEvent:
    """A single thing that happened, normalised across both data sources."""

    t: float  # seconds of game time
    type: str
    team: Optional[int] = None
    text: str = ""
    importance: float = 1.0
    payload: Dict[str, Any] = field(default_factory=dict)

    def clock(self) -> str:
        return format_clock(self.t)


@dataclass
class GameState:
    """A full snapshot of the game at time ``t``."""

    t: float
    blue: TeamState
    red: TeamState

    match_id: Optional[str] = None
    queue_id: Optional[int] = None
    patch: Optional[str] = None
    source: str = "unknown"  # live | timeline | sim | spectator
    map_terrain: Optional[str] = None
    events: List[GameEvent] = field(default_factory=list)

    #: Set only for finished games replayed from a timeline.
    winner: Optional[int] = None

    @property
    def minutes(self) -> float:
        return self.t / 60.0

    def team(self, team_id: int) -> TeamState:
        return self.blue if team_id == BLUE else self.red

    def clock(self) -> str:
        return format_clock(self.t)

    def player_by_participant(self, participant_id: int) -> Optional[PlayerState]:
        for side in (self.blue, self.red):
            for player in side.players:
                if player.participant_id == participant_id:
                    return player
        return None

    def player_by_champion(self, champion: str) -> Optional[PlayerState]:
        wanted = champion.strip().lower()
        for side in (self.blue, self.red):
            for player in side.players:
                if player.champion.lower() == wanted:
                    return player
        return None

    def gold_diff(self) -> float:
        return self.blue.gold - self.red.gold


def format_clock(seconds: float) -> str:
    """Seconds of game time as ``MM:SS``."""
    if seconds is None or (isinstance(seconds, float) and math.isnan(seconds)):
        return "--:--"
    seconds = max(0.0, float(seconds))
    return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"


def expected_team_gold(t_seconds: float) -> float:
    """Roughly how much gold one team has earned by time ``t``.

    Used to turn an absolute gold lead into a *relative* one. A 3k lead at ten
    minutes is close to decisive; the same 3k at thirty minutes is a skirmish.
    Dividing by this curve is what lets a single coefficient cover both.

    The constants come from the mean total-gold curve across ranked Summoner's
    Rift games: about 2.5k at minute zero (starting gold plus the first wave)
    growing near-linearly at about 1.55k per team per minute.
    """
    minutes = max(0.0, t_seconds / 60.0)
    return 2500.0 + 1550.0 * minutes


def expected_team_xp(t_seconds: float) -> float:
    """Companion curve to :func:`expected_team_gold` for experience."""
    minutes = max(0.0, t_seconds / 60.0)
    return 1200.0 + 1750.0 * minutes
