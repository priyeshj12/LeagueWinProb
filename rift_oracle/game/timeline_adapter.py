"""Rebuild a game state sequence from a Match-V5 timeline.

The timeline gives one ``participantFrame`` snapshot per minute plus an exact,
timestamped event log. Neither alone is enough: the frames carry gold and XP
but no objectives, and the events carry objectives but no economy. This module
replays the events forward to maintain objective and inventory state, joins
that with the interpolated frames, and emits a :class:`GameState` at every
frame boundary and, optionally, at the exact instant of every major event.

That last part matters for attribution. If Baron falls at 24:37 and the frames
are a minute apart, a frame-only reconstruction reports the swing at 25:00 and
blames it on whatever else happened in that minute. Emitting a state at 24:37
pins the swing to the objective that actually caused it.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from rift_oracle.game.state import (
    BLUE,
    HERALD_WEIGHT,
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
    format_clock,
    other_team,
)
from rift_oracle.riot.ddragon import DataDragon, default_ddragon

log = logging.getLogger(__name__)

#: Base respawn wait in seconds by champion level (index 0 == level 1).
BASE_RESPAWN = [
    10.0, 10.0, 12.0, 12.0, 14.0, 16.0, 20.0, 25.0, 28.0,
    32.5, 35.0, 37.5, 40.0, 42.5, 45.0, 47.5, 50.0, 52.5,
]

#: Events that justify emitting an extra, interpolated game state.
MAJOR_EVENTS = frozenset(
    {
        "CHAMPION_KILL",
        "BUILDING_KILL",
        "ELITE_MONSTER_KILL",
        "DRAGON_SOUL_GIVEN",
        "TURRET_PLATE_DESTROYED",
        "GAME_END",
    }
)

DRAGON_LABELS = {
    "FIRE_DRAGON": "Infernal",
    "WATER_DRAGON": "Ocean",
    "EARTH_DRAGON": "Mountain",
    "AIR_DRAGON": "Cloud",
    "HEXTECH_DRAGON": "Hextech",
    "CHEMTECH_DRAGON": "Chemtech",
    "ELDER_DRAGON": "Elder",
}


def death_timer(level: int, t_seconds: float) -> float:
    """Respawn time for a champion of ``level`` dying at ``t_seconds``.

    Base wait scales with level; after fifteen minutes a time-increase factor
    compounds on top, which is why late-game deaths decide games and early ones
    rarely do. The tiers below match the in-game formula's breakpoints.
    """
    idx = max(0, min(int(level) - 1, len(BASE_RESPAWN) - 1))
    base = BASE_RESPAWN[idx]
    minutes = max(0.0, t_seconds / 60.0)

    factor = 0.0
    if minutes > 15.0:
        factor += (min(minutes, 30.0) - 15.0) * 0.017
    if minutes > 30.0:
        factor += (min(minutes, 45.0) - 30.0) * 0.012
    if minutes > 45.0:
        factor += (minutes - 45.0) * 0.058
    return base * (1.0 + factor)


def team_of_participant(participant_id: int) -> int:
    """Participants 1-5 are always blue, 6-10 always red."""
    return BLUE if int(participant_id) <= 5 else RED


def event_team(event: Dict[str, Any], *fields: str) -> Optional[int]:
    """Read a side out of an event, or ``None`` when it names no real side.

    Real timelines do not always name one. Riot records a Rift Herald or a
    voidgrub taken without a champion last-hit as ``killerTeamId: 300`` with no
    ``killerId`` at all, and ``DRAGON_SOUL_GIVEN`` ships ``teamId: 0``.

    Both used to land on blue, because ``int(event.get("teamId") or BLUE)``
    treats 0 as absent - so every dragon soul in every game was credited to
    blue side. Returning ``None`` and letting the caller skip or infer is the
    only honest handling: an unattributable objective belongs to nobody.
    """
    for field in fields:
        value = event.get(field)
        if value is not None and int(value) in (BLUE, RED):
            return int(value)

    killer = int(event.get("killerId") or 0)
    if killer:
        return team_of_participant(killer)
    return None


class _Inventory:
    """Tracks one participant's items so item value can be valued per frame."""

    __slots__ = ("items",)

    def __init__(self) -> None:
        self.items: List[int] = []

    def purchase(self, item_id: int) -> None:
        self.items.append(int(item_id))

    def remove(self, item_id: int) -> None:
        try:
            self.items.remove(int(item_id))
        except ValueError:
            pass

    def undo(self, before_id: int, after_id: int) -> None:
        """ITEM_UNDO reverses a purchase (``beforeId``) or a sale (``afterId``)."""
        if before_id:
            self.remove(before_id)
        if after_id:
            self.purchase(after_id)


class TimelineReplay:
    """Replays a match timeline into states, events, and a final result."""

    def __init__(
        self,
        match: Dict[str, Any],
        timeline: Dict[str, Any],
        ddragon: Optional[DataDragon] = None,
    ) -> None:
        self.match = match or {}
        self.timeline = timeline or {}
        self.dd = ddragon or default_ddragon()

        info = self.match.get("info", {})
        self.match_id = (self.match.get("metadata") or {}).get("matchId")
        self.queue_id = info.get("queueId")
        self.patch = str(info.get("gameVersion") or "")
        self.game_duration_s = _duration_seconds(info)

        self.participants = self._index_participants()
        self.winner = self._winner()

        tl_info = self.timeline.get("info", {})
        self.frames: List[Dict[str, Any]] = list(tl_info.get("frames") or [])
        self.frame_interval_ms = int(tl_info.get("frameInterval") or 60000)

        # Mutable replay state, advanced event by event.
        self._inventories: Dict[int, _Inventory] = {
            pid: _Inventory() for pid in self.participants
        }
        self._kills: Dict[int, int] = {pid: 0 for pid in self.participants}
        self._deaths: Dict[int, int] = {pid: 0 for pid in self.participants}
        self._assists: Dict[int, int] = {pid: 0 for pid in self.participants}
        self._wards: Dict[int, int] = {pid: 0 for pid in self.participants}
        self._last_death_at: Dict[int, Optional[float]] = {
            pid: None for pid in self.participants
        }
        self._last_death_level: Dict[int, int] = {pid: 1 for pid in self.participants}

        self._objectives: Dict[int, Dict[str, Any]] = {
            BLUE: _blank_objectives(),
            RED: _blank_objectives(),
        }
        self.events: List[GameEvent] = []

    # -- setup ------------------------------------------------------------

    def _index_participants(self) -> Dict[int, Dict[str, Any]]:
        table: Dict[int, Dict[str, Any]] = {}
        for entry in (self.match.get("info", {}) or {}).get("participants", []) or []:
            pid = entry.get("participantId")
            if pid is None:
                continue
            table[int(pid)] = entry

        if not table:
            # Timelines can be replayed without the match DTO; champion names
            # are then unavailable but everything numeric still works.
            for entry in (self.timeline.get("info", {}) or {}).get("participants", []) or []:
                pid = entry.get("participantId")
                if pid is None:
                    continue
                table[int(pid)] = {"participantId": int(pid), "puuid": entry.get("puuid")}
        return table

    def _winner(self) -> Optional[int]:
        for team in (self.match.get("info", {}) or {}).get("teams", []) or []:
            if team.get("win"):
                return int(team.get("teamId", BLUE))
        for frame in reversed(list((self.timeline.get("info", {}) or {}).get("frames") or [])):
            for event in frame.get("events") or []:
                if event.get("type") == "GAME_END" and event.get("winningTeam"):
                    return int(event["winningTeam"])
        return None

    def champion_of(self, participant_id: int) -> str:
        entry = self.participants.get(int(participant_id)) or {}
        name = entry.get("championName")
        if name:
            return str(name)
        champ_id = entry.get("championId")
        if champ_id:
            return self.dd.champion_name(int(champ_id))
        return f"P{participant_id}"

    # -- event application ------------------------------------------------

    def _apply_event(self, event: Dict[str, Any], t: float) -> Optional[GameEvent]:
        kind = event.get("type")
        handler = getattr(self, f"_on_{str(kind).lower()}", None)
        if handler is None:
            return None
        return handler(event, t)

    def _on_champion_kill(self, event: Dict[str, Any], t: float) -> GameEvent:
        killer = int(event.get("killerId") or 0)
        victim = int(event.get("victimId") or 0)
        assists = [int(a) for a in (event.get("assistingParticipantIds") or [])]

        if killer in self._kills:
            self._kills[killer] += 1
        if victim in self._deaths:
            self._deaths[victim] += 1
            self._last_death_at[victim] = t
        for pid in assists:
            if pid in self._assists:
                self._assists[pid] += 1

        scoring_team = team_of_participant(killer) if killer else other_team(
            team_of_participant(victim)
        )
        bounty = int(event.get("bounty") or 0)
        shutdown = int(event.get("shutdownBounty") or 0)

        killer_name = self.champion_of(killer) if killer else "An execution"
        victim_name = self.champion_of(victim)
        if killer:
            text = f"{killer_name} killed {victim_name}"
        else:
            text = f"{victim_name} died to turret/minions"
        if assists:
            text += f" (+{len(assists)} assist{'s' if len(assists) > 1 else ''})"
        if shutdown:
            text += f", {shutdown}g shutdown"

        return GameEvent(
            t=t,
            type="CHAMPION_KILL",
            team=scoring_team,
            text=text,
            importance=1.0 + (shutdown / 1000.0),
            payload={
                "killer": killer,
                "victim": victim,
                "assists": assists,
                "bounty": bounty,
                "shutdown": shutdown,
                "killer_champion": killer_name,
                "victim_champion": victim_name,
            },
        )

    def _on_building_kill(self, event: Dict[str, Any], t: float) -> Optional[GameEvent]:
        # ``teamId`` is the team that OWNED the destroyed building, so the
        # taker is the other side. A killer-based fallback would name the taker
        # directly, so invert it back before using it as the loser.
        named = event.get("teamId")
        if named is not None and int(named) in (BLUE, RED):
            loser = int(named)
        else:
            taker_guess = event_team(event)
            if taker_guess is None:
                return None
            loser = other_team(taker_guess)
        taker = other_team(loser)
        building = str(event.get("buildingType") or "")
        lane = str(event.get("laneType") or "").replace("_LANE", "").title()

        if building == "INHIBITOR_BUILDING":
            self._objectives[taker]["inhibitors"] += 1
            self._objectives[loser]["inhibitors_down"].append(t)
            text = f"{_side(taker)} destroyed the {lane} inhibitor"
            importance = 3.0
            kind = "INHIBITOR_KILL"
        else:
            tower = str(event.get("towerType") or "UNKNOWN")
            self._objectives[taker]["towers"] += TURRET_WEIGHT.get(tower, 1.2)
            self._objectives[taker]["towers_raw"] += 1
            label = tower.replace("_TURRET", "").title()
            text = f"{_side(taker)} took the {lane} {label} turret"
            importance = 1.0 + TURRET_WEIGHT.get(tower, 1.2) * 0.5
            kind = "TURRET_KILL"

        return GameEvent(
            t=t,
            type=kind,
            team=taker,
            text=text,
            importance=importance,
            payload={"lane": lane, "building": building, "tower": event.get("towerType")},
        )

    def _on_elite_monster_kill(self, event: Dict[str, Any], t: float) -> Optional[GameEvent]:
        killer_team = event_team(event, "killerTeamId")
        if killer_team is None:
            # Neutral kill with no attributable champion. Crediting a side here
            # would invent an objective lead out of a data quirk.
            return None
        monster = str(event.get("monsterType") or "")
        subtype = str(event.get("monsterSubType") or "")

        if monster == "DRAGON" and subtype == "ELDER_DRAGON":
            self._objectives[killer_team]["elders"] += 1
            self._objectives[killer_team]["elder_taken_at"] = t
            return GameEvent(
                t=t,
                type="ELDER_KILL",
                team=killer_team,
                text=f"{_side(killer_team)} killed Elder Dragon",
                importance=6.0,
                payload={"monster": "ELDER"},
            )

        if monster == "DRAGON":
            label = DRAGON_LABELS.get(subtype, subtype.replace("_DRAGON", "").title() or "Drake")
            self._objectives[killer_team]["dragons"].append(label)
            count = len(self._objectives[killer_team]["dragons"])
            text = f"{_side(killer_team)} took {label} Drake (#{count})"
            importance = 2.0 + (1.5 if count >= 3 else 0.0)
            return GameEvent(
                t=t,
                type="DRAGON_KILL",
                team=killer_team,
                text=text,
                importance=importance,
                payload={"dragon": label, "count": count},
            )

        if monster == "BARON_NASHOR":
            self._objectives[killer_team]["barons"] += 1
            self._objectives[killer_team]["baron_taken_at"] = t
            return GameEvent(
                t=t,
                type="BARON_KILL",
                team=killer_team,
                text=f"{_side(killer_team)} killed Baron Nashor",
                importance=6.0,
                payload={"monster": "BARON"},
            )

        weight = HERALD_WEIGHT.get(monster, 0.5)
        self._objectives[killer_team]["heralds"] += weight
        pretty = monster.replace("_", " ").title().replace("Riftherald", "Rift Herald")
        return GameEvent(
            t=t,
            type="HERALD_KILL",
            team=killer_team,
            text=f"{_side(killer_team)} took {pretty}",
            importance=1.0 + weight,
            payload={"monster": monster},
        )

    def _on_turret_plate_destroyed(self, event: Dict[str, Any], t: float) -> Optional[GameEvent]:
        named = event.get("teamId")
        if named is not None and int(named) in (BLUE, RED):
            loser = int(named)
        else:
            taker_guess = event_team(event)
            if taker_guess is None:
                return None
            loser = other_team(taker_guess)
        taker = other_team(loser)
        self._objectives[taker]["plates"] += 1
        lane = str(event.get("laneType") or "").replace("_LANE", "").title()
        return GameEvent(
            t=t,
            type="PLATE",
            team=taker,
            text=f"{_side(taker)} took a {lane} turret plate (+160g)",
            importance=0.5,
            payload={"lane": lane},
        )

    def _on_dragon_soul_given(self, event: Dict[str, Any], t: float) -> Optional[GameEvent]:
        # This event arrives with teamId 0, so the owner is inferred from who
        # actually has the four drakes by now.
        team = event_team(event, "teamId") or self._soul_owner()
        if team is None:
            return None
        name = str(event.get("name") or "").title()
        self._objectives[team]["soul"] = name or "Dragon"
        return GameEvent(
            t=t,
            type="DRAGON_SOUL",
            team=team,
            text=f"{_side(team)} claimed the {name} Dragon Soul",
            importance=7.0,
            payload={"soul": name},
        )

    def _soul_owner(self) -> Optional[int]:
        """Whichever side has enough drakes for the soul, if exactly one does."""
        qualified = [
            team
            for team in (BLUE, RED)
            if len(self._objectives[team]["dragons"]) >= SOUL_AT
        ]
        return qualified[0] if len(qualified) == 1 else None

    def _on_item_purchased(self, event: Dict[str, Any], t: float) -> Optional[GameEvent]:
        pid = int(event.get("participantId") or 0)
        item_id = int(event.get("itemId") or 0)
        if pid not in self._inventories or not item_id:
            return None
        self._inventories[pid].purchase(item_id)

        if not self.dd.is_legendary(item_id):
            return None
        cost = self.dd.item_cost(item_id)
        return GameEvent(
            t=t,
            type="ITEM_COMPLETED",
            team=team_of_participant(pid),
            text=f"{self.champion_of(pid)} completed {self.dd.item_name(item_id)} ({cost}g)",
            importance=1.0 + cost / 3000.0,
            payload={
                "participant": pid,
                "item_id": item_id,
                "item": self.dd.item_name(item_id),
                "cost": cost,
                "champion": self.champion_of(pid),
            },
        )

    def _on_item_sold(self, event: Dict[str, Any], t: float) -> None:
        pid = int(event.get("participantId") or 0)
        if pid in self._inventories:
            self._inventories[pid].remove(int(event.get("itemId") or 0))
        return None

    def _on_item_destroyed(self, event: Dict[str, Any], t: float) -> None:
        pid = int(event.get("participantId") or 0)
        if pid in self._inventories:
            self._inventories[pid].remove(int(event.get("itemId") or 0))
        return None

    def _on_item_undo(self, event: Dict[str, Any], t: float) -> None:
        pid = int(event.get("participantId") or 0)
        if pid in self._inventories:
            self._inventories[pid].undo(
                int(event.get("beforeId") or 0), int(event.get("afterId") or 0)
            )
        return None

    def _on_ward_placed(self, event: Dict[str, Any], t: float) -> None:
        pid = int(event.get("creatorId") or 0)
        if pid in self._wards:
            self._wards[pid] += 1
        return None

    def _on_level_up(self, event: Dict[str, Any], t: float) -> None:
        pid = int(event.get("participantId") or 0)
        if pid in self._last_death_level:
            self._last_death_level[pid] = int(event.get("level") or 1)
        return None

    def _on_game_end(self, event: Dict[str, Any], t: float) -> Optional[GameEvent]:
        winner = event_team(event, "winningTeam") or self.winner
        if winner is None:
            return None
        return GameEvent(
            t=t,
            type="GAME_END",
            team=winner,
            text=f"{_side(winner)} destroyed the nexus",
            importance=10.0,
            payload={"winner": winner},
        )

    # -- state construction ----------------------------------------------

    def _build_state(
        self, t: float, participant_frames: Dict[int, Dict[str, Any]], events: List[GameEvent]
    ) -> GameState:
        teams = {BLUE: TeamState(team_id=BLUE), RED: TeamState(team_id=RED)}

        for pid in sorted(self.participants):
            frame = participant_frames.get(pid) or {}
            entry = self.participants.get(pid) or {}
            team_id = int(entry.get("teamId") or team_of_participant(pid))
            level = int(frame.get("level") or 1)

            damage = frame.get("damageStats") or {}
            stats = frame.get("championStats") or {}
            inventory = self._inventories[pid].items
            item_value = sum(
                self.dd.item_cost(i) for i in inventory if self.dd.is_legendary(i)
            )
            completed = sum(1 for i in inventory if self.dd.is_legendary(i))

            dead, respawn = self._death_status(pid, t, level)

            player = PlayerState(
                participant_id=pid,
                team=team_id,
                champion=self.champion_of(pid),
                champion_id=entry.get("championId"),
                puuid=entry.get("puuid"),
                riot_id=entry.get("riotIdGameName") or entry.get("summonerName"),
                position=str(entry.get("teamPosition") or entry.get("individualPosition") or ""),
                level=level,
                kills=self._kills[pid],
                deaths=self._deaths[pid],
                assists=self._assists[pid],
                cs=int(frame.get("minionsKilled") or 0) + int(frame.get("jungleMinionsKilled") or 0),
                total_gold=int(frame.get("totalGold") or 500),
                current_gold=float(frame.get("currentGold") or 0.0),
                xp=int(frame.get("xp") or 0),
                items=list(inventory),
                item_value=item_value,
                completed_items=completed,
                is_dead=dead,
                respawn_s=respawn,
                damage_to_champs=int(damage.get("totalDamageDoneToChampions") or 0),
                physical_damage_to_champs=int(damage.get("physicalDamageDoneToChampions") or 0),
                magic_damage_to_champs=int(damage.get("magicDamageDoneToChampions") or 0),
                true_damage_to_champs=int(damage.get("trueDamageDoneToChampions") or 0),
                vision_score=float(self._wards[pid]),
                attack_damage=float(stats.get("attackDamage") or 0.0),
                ability_power=float(stats.get("abilityPower") or 0.0),
                armor=float(stats.get("armor") or 0.0),
                magic_resist=float(stats.get("magicResist") or 0.0),
                health_max=float(stats.get("healthMax") or 0.0),
            )
            teams[team_id].players.append(player)

        for team_id, side in teams.items():
            obj = self._objectives[team_id]
            enemy_obj = self._objectives[other_team(team_id)]
            # Clamped to what the map holds; see the caps in game.state for why.
            side.towers = min(obj["towers"], MAX_TURRET_WEIGHT)
            side.towers_raw = min(obj["towers_raw"], MAX_TURRETS_PER_SIDE)
            side.turret_plates = min(obj["plates"], MAX_PLATES_PER_SIDE)
            side.inhibitors_taken = obj["inhibitors"]
            side.inhibitors_down = _inhibs_down(obj["inhibitors_down"], t)
            side.dragons = list(obj["dragons"])
            side.heralds = obj["heralds"]
            side.barons = obj["barons"]
            side.elders = obj["elders"]
            side.baron_taken_at = obj["baron_taken_at"]
            side.elder_taken_at = obj["elder_taken_at"]
            side.soul_type = obj["soul"]
            side.has_soul = bool(obj["soul"]) or len(obj["dragons"]) >= SOUL_AT
            _ = enemy_obj  # kept for symmetry/readability of the loop

        return GameState(
            t=t,
            blue=teams[BLUE],
            red=teams[RED],
            match_id=self.match_id,
            queue_id=self.queue_id,
            patch=self.patch,
            source="timeline",
            events=events,
            winner=self.winner,
        )

    def _death_status(self, pid: int, t: float, level: int) -> Tuple[bool, float]:
        died_at = self._last_death_at.get(pid)
        if died_at is None:
            return False, 0.0
        timer = death_timer(level, died_at)
        elapsed = t - died_at
        if elapsed >= timer:
            return False, 0.0
        return True, max(0.0, timer - elapsed)

    # -- public API -------------------------------------------------------

    def run(self, resolution: str = "events") -> List[GameState]:
        """Replay the timeline into an ordered list of states.

        ``resolution`` is ``frames`` for one state per minute, or ``events``
        (the default) to also emit an interpolated state at the instant of
        every major event.
        """
        if not self.frames:
            return []

        states: List[GameState] = []
        pending: List[GameEvent] = []
        self.events = []

        frame_times = [float(f.get("timestamp", 0)) / 1000.0 for f in self.frames]
        frame_participants = [_participant_frames(f) for f in self.frames]

        for index, frame in enumerate(self.frames):
            frame_t = frame_times[index]
            raw_events = sorted(
                frame.get("events") or [], key=lambda e: int(e.get("timestamp") or 0)
            )

            for raw in raw_events:
                event_t = float(raw.get("timestamp") or 0) / 1000.0
                # Events are filed under the frame that closes after them, so
                # clamp into the interval this frame actually covers.
                event_t = max(frame_times[max(0, index - 1)], min(event_t, frame_t))
                game_event = self._apply_event(raw, event_t)
                if game_event is None:
                    continue
                self.events.append(game_event)
                pending.append(game_event)

                if resolution == "events" and raw.get("type") in MAJOR_EVENTS and index > 0:
                    interpolated = _interpolate(
                        frame_participants[index - 1],
                        frame_participants[index],
                        frame_times[index - 1],
                        frame_t,
                        event_t,
                    )
                    states.append(self._build_state(event_t, interpolated, list(pending)))
                    pending = []

            states.append(self._build_state(frame_t, frame_participants[index], list(pending)))
            pending = []

        states.sort(key=lambda s: s.t)
        return _dedupe(states)

    def summary(self) -> Dict[str, Any]:
        info = self.match.get("info", {})
        blue = [p for p in self.participants.values() if int(p.get("teamId", BLUE)) == BLUE]
        red = [p for p in self.participants.values() if int(p.get("teamId", BLUE)) == RED]
        return {
            "match_id": self.match_id,
            "queue_id": self.queue_id,
            "patch": self.patch,
            "duration_s": self.game_duration_s,
            "duration": format_clock(self.game_duration_s),
            "winner": self.winner,
            "blue_champions": [str(p.get("championName", "?")) for p in blue],
            "red_champions": [str(p.get("championName", "?")) for p in red],
            "game_creation": info.get("gameCreation"),
        }


def _blank_objectives() -> Dict[str, Any]:
    return {
        "towers": 0.0,
        "towers_raw": 0,
        "plates": 0,
        "inhibitors": 0,
        "inhibitors_down": [],  # timestamps at which our own inhibs fell
        "dragons": [],
        "heralds": 0.0,
        "barons": 0,
        "elders": 0,
        "baron_taken_at": None,
        "elder_taken_at": None,
        "soul": None,
    }


def _inhibs_down(timestamps: Sequence[float], now: float, respawn_s: float = 300.0) -> int:
    """How many of a team's own inhibitors are currently destroyed."""
    return sum(1 for ts in timestamps if now - ts < respawn_s)


def _participant_frames(frame: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    table: Dict[int, Dict[str, Any]] = {}
    for key, value in (frame.get("participantFrames") or {}).items():
        try:
            table[int(key)] = value
        except (TypeError, ValueError):
            continue
    return table


def _interpolate(
    before: Dict[int, Dict[str, Any]],
    after: Dict[int, Dict[str, Any]],
    t_before: float,
    t_after: float,
    t: float,
) -> Dict[int, Dict[str, Any]]:
    """Linearly interpolate the numeric fields of two participant frames."""
    span = max(t_after - t_before, 1e-6)
    alpha = max(0.0, min(1.0, (t - t_before) / span))
    merged: Dict[int, Dict[str, Any]] = {}

    for pid, late in after.items():
        early = before.get(pid, late)
        blended = dict(late)
        for key in ("totalGold", "currentGold", "xp", "minionsKilled", "jungleMinionsKilled", "level"):
            a, b = early.get(key), late.get(key)
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                value = a + (b - a) * alpha
                blended[key] = int(round(value)) if key != "currentGold" else value
        for group in ("damageStats", "championStats"):
            a_group, b_group = early.get(group) or {}, late.get(group) or {}
            if a_group and b_group:
                blended[group] = {
                    key: (
                        a_group.get(key, value) + (value - a_group.get(key, value)) * alpha
                        if isinstance(value, (int, float))
                        and isinstance(a_group.get(key, value), (int, float))
                        else value
                    )
                    for key, value in b_group.items()
                }
        merged[pid] = blended
    return merged


def _dedupe(states: List[GameState]) -> List[GameState]:
    """Collapse states that land on the same second, keeping their events."""
    out: List[GameState] = []
    for state in states:
        if out and abs(out[-1].t - state.t) < 0.25:
            seen = {(e.t, e.type, e.text) for e in out[-1].events}
            for event in state.events:
                if (event.t, event.type, event.text) not in seen:
                    out[-1].events.append(event)
            out[-1] = state if len(state.events) >= len(out[-1].events) else out[-1]
            continue
        out.append(state)
    return out


def _duration_seconds(info: Dict[str, Any]) -> float:
    duration = info.get("gameDuration")
    if duration is None:
        return 0.0
    duration = float(duration)
    # Before patch 11.20 gameDuration was in milliseconds. Games never run for
    # a hundred thousand seconds, so the magnitude disambiguates safely.
    if duration > 100000:
        return duration / 1000.0
    if info.get("gameEndTimestamp") is None and duration > 10000:
        return duration / 1000.0
    return duration


def _side(team_id: int) -> str:
    return "Blue" if team_id == BLUE else "Red"


def replay_match(
    match: Dict[str, Any],
    timeline: Dict[str, Any],
    resolution: str = "events",
    ddragon: Optional[DataDragon] = None,
) -> Tuple[List[GameState], TimelineReplay]:
    """Convenience wrapper returning both the states and the replay object."""
    replay = TimelineReplay(match, timeline, ddragon=ddragon)
    return replay.run(resolution=resolution), replay


def iter_events(states: Iterable[GameState]) -> Iterable[GameEvent]:
    for state in states:
        for event in state.events:
            yield event
