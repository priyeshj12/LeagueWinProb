"""Adapt the in-client Live Client Data API into the canonical game state.

The live endpoint and the Match-V5 timeline see different halves of the game.
The timeline knows exact gold, XP and damage but has to infer who is currently
dead. The live endpoint knows exactly who is dead and when they respawn, and it
lists every item every player holds - but it only reports *your* gold.

So team gold is estimated here, from two independent signals that are combined
because they fail in different places:

``spent``
    The sum of the ``price`` field over every item a player holds. This is
    exact for gold already converted into items, and it is the thing that
    actually matters for a fight, but it lags by the size of the player's
    wallet.

``earned``
    A closed-form income model: starting gold, passive income, creep score,
    takedowns, and objective gold. This tracks the wallet but cannot see
    bounties or gold Riot hands out for objectives we did not observe.

``spent`` is trusted as the floor, and the difference is used to estimate the
unspent wallet, clamped to a plausible range. The result is typically within a
few hundred gold per team, which is well inside the noise the model already
tolerates. :attr:`LiveGameTracker.gold_confidence` reports how good the
estimate looks so the UI can label it honestly.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from rift_oracle.game.state import (
    BLUE,
    MAX_PLATES_PER_SIDE,
    MAX_TURRET_WEIGHT,
    MAX_TURRETS_PER_SIDE,
    RED,
    SOUL_AT,
    TURRET_WEIGHT,
    GameEvent,
    GameState,
    PlayerState,
    TeamState,
    other_team,
)
from rift_oracle.riot.ddragon import DataDragon, default_ddragon

log = logging.getLogger(__name__)

# -- income model constants -------------------------------------------------

STARTING_GOLD = 500.0
#: Passive income is 20.4 gold per 10 seconds, starting at 1:50.
PASSIVE_GOLD_PER_S = 2.04
PASSIVE_START_S = 110.0
#: Blended gold per creep score, averaged over melee, caster, siege and camps.
GOLD_PER_CS = 21.5
#: Base champion takedown gold. Bounties and streak reductions move this, which
#: is part of why the estimate is blended with observed item value.
GOLD_PER_KILL = 300.0
GOLD_PER_ASSIST = 145.0
#: Team gold per structure, spread across the five players.
GOLD_PER_TURRET = 500.0
GOLD_PER_PLATE = 160.0
GOLD_PER_INHIB = 200.0
GOLD_PER_BARON = 300.0
#: Plausible bounds for a player's unspent wallet when it cannot be observed.
UNSPENT_MIN = 0.0
UNSPENT_MAX = 3200.0

ORDER = "ORDER"
CHAOS = "CHAOS"

_TURRET_RE = re.compile(r"Turret_T(?P<team>[12])_(?P<lane>[LCR])_(?P<index>\d+)", re.I)
_BARRACKS_RE = re.compile(r"Barracks_T(?P<team>[12])_(?P<lane>[LCR])", re.I)

#: Turret tier by lane and index. The numbering is not uniform across sides:
#: blue's inhibitor turrets are filed under the centre lane as 06 and 07.
_TURRET_TIER: Dict[Tuple[str, int], str] = {
    ("C", 1): "NEXUS_TURRET", ("C", 2): "NEXUS_TURRET",
    ("C", 3): "BASE_TURRET", ("C", 6): "BASE_TURRET", ("C", 7): "BASE_TURRET",
    ("C", 4): "INNER_TURRET", ("C", 5): "OUTER_TURRET",
    ("L", 1): "BASE_TURRET", ("L", 2): "INNER_TURRET", ("L", 3): "OUTER_TURRET",
    ("R", 1): "BASE_TURRET", ("R", 2): "INNER_TURRET", ("R", 3): "OUTER_TURRET",
}

_LANE_NAME = {"L": "Top", "C": "Mid", "R": "Bot"}

DRAGON_LABELS = {
    "fire": "Infernal", "water": "Ocean", "earth": "Mountain", "air": "Cloud",
    "hextech": "Hextech", "chemtech": "Chemtech", "elder": "Elder",
}


def team_from_side(side: str) -> int:
    """``ORDER`` is blue side, ``CHAOS`` is red side."""
    return BLUE if str(side).upper() == ORDER else RED


def side_from_team(team: int) -> str:
    return ORDER if team == BLUE else CHAOS


class LiveGameTracker:
    """Folds successive Live Client snapshots into a GameState sequence.

    The live endpoint is stateless: each poll returns the current scoreboard
    and the complete event log from the start of the game. This class keeps the
    derived state - objectives taken, buff timers, which events have already
    been reported - across polls.
    """

    def __init__(self, ddragon: Optional[DataDragon] = None) -> None:
        self.dd = ddragon or default_ddragon()
        self._objectives: Dict[int, Dict[str, Any]] = {
            BLUE: _blank_objectives(),
            RED: _blank_objectives(),
        }
        self._seen_event_ids: set = set()
        self._champion_team: Dict[str, int] = {}
        self.gold_confidence: float = 0.0
        self.game_over: bool = False
        self.result: Optional[str] = None

    # -- events -----------------------------------------------------------

    def ingest_events(self, raw_events: List[Dict[str, Any]]) -> List[GameEvent]:
        """Apply any events not seen before, returning the new ones."""
        fresh: List[GameEvent] = []
        for raw in sorted(raw_events, key=lambda e: float(e.get("EventTime", 0.0))):
            event_id = raw.get("EventID")
            key = event_id if event_id is not None else (
                raw.get("EventName"), raw.get("EventTime")
            )
            if key in self._seen_event_ids:
                continue
            self._seen_event_ids.add(key)
            event = self._translate(raw)
            if event is not None:
                fresh.append(event)
        return fresh

    def _team_of_champion(self, name: Optional[str]) -> Optional[int]:
        if not name:
            return None
        return self._champion_team.get(str(name))

    def _translate(self, raw: Dict[str, Any]) -> Optional[GameEvent]:
        name = str(raw.get("EventName") or "")
        t = float(raw.get("EventTime") or 0.0)

        if name == "ChampionKill":
            killer = raw.get("KillerName")
            victim = raw.get("VictimName")
            assisters = list(raw.get("Assisters") or [])
            team = self._team_of_champion(killer)
            if team is None:
                victim_team = self._team_of_champion(victim)
                team = other_team(victim_team) if victim_team is not None else None
            return GameEvent(
                t=t, type="CHAMPION_KILL", team=team,
                text=f"{killer} killed {victim}"
                + (f" (+{len(assisters)} assist{'s' if len(assisters) > 1 else ''})" if assisters else ""),
                importance=1.0,
                payload={"killer": killer, "victim": victim, "assists": assisters},
            )

        if name == "DragonKill":
            killer = raw.get("KillerName")
            team = self._team_of_champion(killer)
            dragon = str(raw.get("DragonType") or "").lower()
            label = DRAGON_LABELS.get(dragon, dragon.title() or "Drake")
            stolen = str(raw.get("Stolen", "False")).lower() == "true"
            if team is None:
                return None
            if dragon == "elder":
                self._objectives[team]["elders"] += 1
                self._objectives[team]["elder_taken_at"] = t
                return GameEvent(
                    t=t, type="ELDER_KILL", team=team,
                    text=f"{_side(team)} killed Elder Dragon" + (" (STOLEN)" if stolen else ""),
                    importance=6.0, payload={"stolen": stolen},
                )
            self._objectives[team]["dragons"].append(label)
            count = len(self._objectives[team]["dragons"])
            if count >= SOUL_AT and not self._objectives[team]["soul"]:
                self._objectives[team]["soul"] = label
            return GameEvent(
                t=t, type="DRAGON_KILL", team=team,
                text=f"{_side(team)} took {label} Drake (#{count})" + (" (STOLEN)" if stolen else ""),
                importance=2.0 + (1.5 if count >= 3 else 0.0),
                payload={"dragon": label, "count": count, "stolen": stolen},
            )

        if name == "BaronKill":
            team = self._team_of_champion(raw.get("KillerName"))
            if team is None:
                return None
            stolen = str(raw.get("Stolen", "False")).lower() == "true"
            self._objectives[team]["barons"] += 1
            self._objectives[team]["baron_taken_at"] = t
            return GameEvent(
                t=t, type="BARON_KILL", team=team,
                text=f"{_side(team)} killed Baron Nashor" + (" (STOLEN)" if stolen else ""),
                importance=6.0, payload={"stolen": stolen},
            )

        if name in ("HeraldKill", "HordeKill", "AtakhanKill"):
            team = self._team_of_champion(raw.get("KillerName"))
            if team is None:
                return None
            weight = {"HeraldKill": 1.0, "HordeKill": 0.25, "AtakhanKill": 1.4}[name]
            self._objectives[team]["heralds"] += weight
            pretty = {"HeraldKill": "Rift Herald", "HordeKill": "a Voidgrub", "AtakhanKill": "Atakhan"}[name]
            return GameEvent(
                t=t, type="HERALD_KILL", team=team,
                text=f"{_side(team)} took {pretty}", importance=1.0 + weight, payload={},
            )

        if name == "TurretKilled":
            owner, lane, tier = _parse_turret(str(raw.get("TurretKilled") or ""))
            if owner is None:
                return None
            taker = other_team(owner)
            self._objectives[taker]["towers"] += TURRET_WEIGHT.get(tier, 1.2)
            self._objectives[taker]["towers_raw"] += 1
            label = tier.replace("_TURRET", "").title()
            return GameEvent(
                t=t, type="TURRET_KILL", team=taker,
                text=f"{_side(taker)} took the {lane} {label} turret",
                importance=1.0 + TURRET_WEIGHT.get(tier, 1.2) * 0.5,
                payload={"lane": lane, "tier": tier},
            )

        if name == "InhibKilled":
            owner, lane = _parse_barracks(str(raw.get("InhibKilled") or ""))
            if owner is None:
                return None
            taker = other_team(owner)
            self._objectives[taker]["inhibitors"] += 1
            self._objectives[owner]["inhibitors_down"].append(t)
            return GameEvent(
                t=t, type="INHIBITOR_KILL", team=taker,
                text=f"{_side(taker)} destroyed the {lane} inhibitor",
                importance=3.0, payload={"lane": lane},
            )

        if name == "InhibRespawned":
            owner, lane = _parse_barracks(str(raw.get("InhibRespawned") or ""))
            if owner is not None:
                downs = self._objectives[owner]["inhibitors_down"]
                if downs:
                    downs.pop(0)
            return None

        if name == "FirstBrick":
            team = self._team_of_champion(raw.get("KillerName"))
            return GameEvent(
                t=t, type="FIRST_TOWER", team=team,
                text=f"{raw.get('KillerName')} drew first blood on a turret",
                importance=0.5, payload={},
            )

        if name == "FirstBlood":
            recipient = raw.get("Recipient")
            return GameEvent(
                t=t, type="FIRST_BLOOD", team=self._team_of_champion(recipient),
                text=f"First blood to {recipient}", importance=0.8, payload={},
            )

        if name == "Ace":
            team = team_from_side(str(raw.get("AcingTeam") or ORDER))
            return GameEvent(
                t=t, type="ACE", team=team,
                text=f"{_side(team)} aced the enemy team", importance=4.0, payload={},
            )

        if name == "Multikill":
            killer = raw.get("KillerName")
            streak = int(raw.get("KillStreak") or 2)
            label = {2: "Double Kill", 3: "Triple Kill", 4: "Quadra Kill", 5: "Penta Kill"}.get(
                streak, f"{streak}x Multikill"
            )
            return GameEvent(
                t=t, type="MULTIKILL", team=self._team_of_champion(killer),
                text=f"{killer}: {label}", importance=1.0 + streak * 0.5,
                payload={"streak": streak},
            )

        if name == "GameEnd":
            self.game_over = True
            self.result = str(raw.get("Result") or "")
            return GameEvent(
                t=t, type="GAME_END", team=None,
                text=f"Game over ({self.result})", importance=10.0,
                payload={"result": self.result},
            )

        if name in ("GameStart", "MinionsSpawning"):
            return GameEvent(
                t=t, type=name.upper(), team=None,
                text="Game start" if name == "GameStart" else "Minions spawning",
                importance=0.1, payload={},
            )
        return None

    # -- state ------------------------------------------------------------

    def build_state(self, payload: Dict[str, Any]) -> GameState:
        """Turn one ``allgamedata`` payload into a :class:`GameState`."""
        game_data = payload.get("gameData") or {}
        t = float(game_data.get("gameTime") or 0.0)
        all_players: List[Dict[str, Any]] = list(payload.get("allPlayers") or [])
        active = payload.get("activePlayer") or {}
        active_name = str(
            active.get("riotIdGameName") or active.get("summonerName") or ""
        )

        self._champion_team = {
            str(p.get("championName")): team_from_side(str(p.get("team", ORDER)))
            for p in all_players
        }

        raw_events = (payload.get("events") or {}).get("Events") or []
        new_events = self.ingest_events(raw_events)

        teams = {BLUE: TeamState(team_id=BLUE), RED: TeamState(team_id=RED)}
        estimates: List[Tuple[float, float]] = []

        for index, raw_player in enumerate(all_players, start=1):
            team_id = team_from_side(str(raw_player.get("team", ORDER)))
            player, spent, earned = self._build_player(
                raw_player, index, team_id, t, active, active_name
            )
            teams[team_id].players.append(player)
            estimates.append((spent, earned))

        self._apply_objective_state(teams, t)
        self.gold_confidence = _gold_confidence(estimates)

        return GameState(
            t=t,
            blue=teams[BLUE],
            red=teams[RED],
            queue_id=None,
            patch=None,
            source="live",
            map_terrain=game_data.get("mapTerrain"),
            events=new_events,
        )

    def _build_player(
        self,
        raw: Dict[str, Any],
        index: int,
        team_id: int,
        t: float,
        active: Dict[str, Any],
        active_name: str,
    ) -> Tuple[PlayerState, float, float]:
        scores = raw.get("scores") or {}
        items = raw.get("items") or []

        kills = int(scores.get("kills") or 0)
        deaths = int(scores.get("deaths") or 0)
        assists = int(scores.get("assists") or 0)
        cs = int(scores.get("creepScore") or 0)
        level = int(raw.get("level") or 1)

        item_ids = [int(i.get("itemID") or 0) for i in items if i.get("itemID")]
        spent = sum(
            float(i.get("price") or 0.0) * max(1, int(i.get("count") or 1)) for i in items
        )
        item_value = sum(self.dd.item_cost(i) for i in item_ids if self.dd.is_legendary(i))
        completed = sum(1 for i in item_ids if self.dd.is_legendary(i))

        objective_gold = self._objective_gold_share(team_id)
        earned = (
            STARTING_GOLD
            + max(0.0, t - PASSIVE_START_S) * PASSIVE_GOLD_PER_S
            + cs * GOLD_PER_CS
            + kills * GOLD_PER_KILL
            + assists * GOLD_PER_ASSIST
            + objective_gold
        )

        is_active = (
            active_name
            and str(raw.get("riotIdGameName") or raw.get("summonerName") or "") == active_name
        )
        if is_active and "currentGold" in active:
            unspent = float(active.get("currentGold") or 0.0)
        else:
            unspent = min(UNSPENT_MAX, max(UNSPENT_MIN, earned - spent))

        stats = active.get("championStats") or {} if is_active else {}

        player = PlayerState(
            participant_id=index,
            team=team_id,
            champion=str(raw.get("championName") or "Unknown"),
            riot_id=str(raw.get("riotIdGameName") or raw.get("summonerName") or ""),
            position=str(raw.get("position") or ""),
            level=level,
            kills=kills,
            deaths=deaths,
            assists=assists,
            cs=cs,
            total_gold=int(round(spent + unspent)),
            current_gold=unspent,
            xp=0,
            items=item_ids,
            item_value=int(item_value),
            completed_items=completed,
            is_dead=bool(raw.get("isDead")),
            respawn_s=float(raw.get("respawnTimer") or 0.0),
            vision_score=float(scores.get("wardScore") or 0.0),
            attack_damage=float(stats.get("attackDamage") or 0.0),
            ability_power=float(stats.get("abilityPower") or 0.0),
            armor=float(stats.get("armor") or 0.0),
            magic_resist=float(stats.get("magicResist") or 0.0),
            health_max=float(stats.get("maxHealth") or stats.get("healthMax") or 0.0),
        )
        return player, spent, earned

    def _objective_gold_share(self, team_id: int) -> float:
        """Objective gold credited to one player, i.e. the team total over five."""
        obj = self._objectives[team_id]
        total = (
            obj["towers_raw"] * GOLD_PER_TURRET
            + obj["plates"] * GOLD_PER_PLATE
            + obj["inhibitors"] * GOLD_PER_INHIB
            + obj["barons"] * GOLD_PER_BARON * 5
        )
        return total / 5.0

    def _apply_objective_state(self, teams: Dict[int, TeamState], t: float) -> None:
        for team_id, side in teams.items():
            obj = self._objectives[team_id]
            side.towers = min(obj["towers"], MAX_TURRET_WEIGHT)
            side.towers_raw = min(obj["towers_raw"], MAX_TURRETS_PER_SIDE)
            side.turret_plates = min(obj["plates"], MAX_PLATES_PER_SIDE)
            side.inhibitors_taken = obj["inhibitors"]
            side.inhibitors_down = len(obj["inhibitors_down"])
            side.dragons = list(obj["dragons"])
            side.heralds = obj["heralds"]
            side.barons = obj["barons"]
            side.elders = obj["elders"]
            side.baron_taken_at = obj["baron_taken_at"]
            side.elder_taken_at = obj["elder_taken_at"]
            side.soul_type = obj["soul"]
            side.has_soul = bool(obj["soul"]) or len(obj["dragons"]) >= SOUL_AT

    def active_team(self, payload: Dict[str, Any]) -> int:
        """Which side the player running this tool is on."""
        active = payload.get("activePlayer") or {}
        name = str(active.get("riotIdGameName") or active.get("summonerName") or "")
        for raw in payload.get("allPlayers") or []:
            candidate = str(raw.get("riotIdGameName") or raw.get("summonerName") or "")
            if candidate and candidate == name:
                return team_from_side(str(raw.get("team", ORDER)))
        return BLUE

    def active_champion(self, payload: Dict[str, Any]) -> Optional[str]:
        active = payload.get("activePlayer") or {}
        name = str(active.get("riotIdGameName") or active.get("summonerName") or "")
        for raw in payload.get("allPlayers") or []:
            candidate = str(raw.get("riotIdGameName") or raw.get("summonerName") or "")
            if candidate and candidate == name:
                return str(raw.get("championName") or "")
        return None


def _blank_objectives() -> Dict[str, Any]:
    return {
        "towers": 0.0, "towers_raw": 0, "plates": 0, "inhibitors": 0,
        "inhibitors_down": [], "dragons": [], "heralds": 0.0, "barons": 0,
        "elders": 0, "baron_taken_at": None, "elder_taken_at": None, "soul": None,
    }


def _parse_turret(raw: str) -> Tuple[Optional[int], str, str]:
    """``Turret_T1_C_05_A`` -> (owning team, lane label, tier)."""
    match = _TURRET_RE.search(raw)
    if not match:
        return None, "", "UNKNOWN"
    owner = BLUE if match.group("team") == "1" else RED
    lane_code = match.group("lane").upper()
    index = int(match.group("index"))
    tier = _TURRET_TIER.get((lane_code, index), "UNKNOWN")
    return owner, _LANE_NAME.get(lane_code, lane_code), tier


def _parse_barracks(raw: str) -> Tuple[Optional[int], str]:
    """``Barracks_T1_C1`` -> (owning team, lane label)."""
    match = _BARRACKS_RE.search(raw)
    if not match:
        return None, ""
    owner = BLUE if match.group("team") == "1" else RED
    return owner, _LANE_NAME.get(match.group("lane").upper(), "")


def _gold_confidence(estimates: List[Tuple[float, float]]) -> float:
    """How far the two independent gold estimates agree, in ``[0, 1]``.

    When the income model and the observed item spend disagree wildly, the
    wallet estimate is doing a lot of work and the UI should say so.
    """
    if not estimates:
        return 0.0
    gaps = []
    for spent, earned in estimates:
        denom = max(earned, 1.0)
        gaps.append(min(1.0, abs(earned - spent) / denom))
    mean_gap = sum(gaps) / len(gaps)
    return max(0.0, 1.0 - mean_gap)


def _side(team_id: int) -> str:
    return "Blue" if team_id == BLUE else "Red"
