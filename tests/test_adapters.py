"""Adapters: the Riot payload semantics that are easy to get backwards."""

import pytest

from rift_oracle.game.live_adapter import (
    LiveGameTracker,
    _parse_barracks,
    _parse_turret,
    team_from_side,
)
from rift_oracle.game.state import BLUE, RED, expected_team_gold, format_clock
from rift_oracle.game.timeline_adapter import (
    TimelineReplay,
    death_timer,
    team_of_participant,
)
from rift_oracle.riot.routing import (
    platform_for_match_id,
    regional_route,
    resolve_platform,
)


# -- routing ---------------------------------------------------------------


def test_platform_aliases_resolve():
    assert resolve_platform("NA") == "na1"
    assert resolve_platform("euw") == "euw1"
    assert resolve_platform("KR") == "kr"


def test_unknown_platform_is_rejected_early():
    with pytest.raises(ValueError, match="unknown platform"):
        resolve_platform("mars1")


def test_regional_routing():
    assert regional_route("na1") == "americas"
    assert regional_route("euw1") == "europe"
    assert regional_route("kr") == "asia"
    assert regional_route("oc1") == "sea"


def test_platform_is_inferred_from_a_match_id():
    assert platform_for_match_id("NA1_5123456789") == "na1"
    assert platform_for_match_id("EUW1_1") == "euw1"
    assert platform_for_match_id("nonsense") is None


# -- timeline --------------------------------------------------------------


def test_participants_map_to_sides_by_id():
    assert [team_of_participant(i) for i in range(1, 6)] == [BLUE] * 5
    assert [team_of_participant(i) for i in range(6, 11)] == [RED] * 5


def test_death_timer_grows_with_level_and_clock():
    assert death_timer(1, 0) < death_timer(18, 0)
    # The time-increase factor only starts at fifteen minutes.
    assert death_timer(18, 600) == pytest.approx(death_timer(18, 0))
    assert death_timer(18, 1800) > death_timer(18, 900)
    assert death_timer(18, 3000) > death_timer(18, 1800) * 1.2


def _timeline(events, frames=3):
    """A minimal but structurally real Match-V5 timeline."""
    participant_frames = {
        str(i): {
            "participantId": i,
            "totalGold": 1000,
            "currentGold": 100,
            "xp": 500,
            "level": 5,
            "minionsKilled": 30,
            "jungleMinionsKilled": 0,
            "championStats": {"attackDamage": 60, "abilityPower": 0},
            "damageStats": {
                "totalDamageDoneToChampions": 500,
                "physicalDamageDoneToChampions": 400,
                "magicDamageDoneToChampions": 100,
            },
        }
        for i in range(1, 11)
    }
    # Riot files an event under the frame that CLOSES after it, so frame i
    # carries events in ((i-1)*60s, i*60s]. Binning with floor division instead
    # would file a 1:30 event under the 1:00 frame and the replay would clamp
    # it onto the boundary.
    def frame_of(timestamp):
        return max(0, -(-int(timestamp) // 60000))

    return {
        "info": {
            "frameInterval": 60000,
            "frames": [
                {
                    "timestamp": i * 60000,
                    "participantFrames": participant_frames,
                    "events": [e for e in events if frame_of(e["timestamp"]) == i],
                }
                for i in range(frames)
            ],
        }
    }


def _match(win_team=BLUE):
    return {
        "metadata": {"matchId": "NA1_1"},
        "info": {
            "queueId": 420,
            "gameVersion": "15.1.1",
            "gameDuration": 1800,
            "teams": [
                {"teamId": BLUE, "win": win_team == BLUE},
                {"teamId": RED, "win": win_team == RED},
            ],
            "participants": [
                {
                    "participantId": i,
                    "teamId": BLUE if i <= 5 else RED,
                    "championName": f"Champ{i}",
                    "puuid": f"p{i}",
                }
                for i in range(1, 11)
            ],
        },
    }


def test_building_kill_credits_the_team_that_did_not_own_it():
    """``teamId`` on BUILDING_KILL is the team that LOST the building."""
    timeline = _timeline(
        [
            {
                "timestamp": 60000,
                "type": "BUILDING_KILL",
                "teamId": RED,  # a red turret fell
                "buildingType": "TOWER_BUILDING",
                "towerType": "OUTER_TURRET",
                "laneType": "MID_LANE",
                "killerId": 1,
            }
        ]
    )
    states, _replay = _run(timeline)
    final = states[-1]
    assert final.blue.towers_raw == 1
    assert final.red.towers_raw == 0


def test_inhibitor_counts_as_down_only_while_it_is_down():
    timeline = _timeline(
        [
            {
                "timestamp": 30000,
                "type": "BUILDING_KILL",
                "teamId": RED,
                "buildingType": "INHIBITOR_BUILDING",
                "laneType": "MID_LANE",
                "killerId": 1,
            }
        ],
        frames=8,
    )
    states, _replay = _run(timeline)
    assert states[1].red.inhibitors_down == 1
    # Inhibitors respawn after five minutes.
    assert states[-1].red.inhibitors_down == 0
    assert states[-1].blue.inhibitors_taken == 1


def test_dragon_soul_and_elder_are_tracked_separately():
    events = [
        {
            "timestamp": 10000 + i * 1000,
            "type": "ELITE_MONSTER_KILL",
            "killerTeamId": BLUE,
            "monsterType": "DRAGON",
            "monsterSubType": "FIRE_DRAGON",
            "killerId": 1,
        }
        for i in range(4)
    ] + [
        {"timestamp": 20000, "type": "DRAGON_SOUL_GIVEN", "teamId": BLUE, "name": "Infernal"},
        {
            "timestamp": 30000,
            "type": "ELITE_MONSTER_KILL",
            "killerTeamId": BLUE,
            "monsterType": "DRAGON",
            "monsterSubType": "ELDER_DRAGON",
            "killerId": 1,
        },
    ]
    states, _replay = _run(_timeline(events))
    final = states[-1]
    assert final.blue.dragon_count == 4  # elder is not a soul point
    assert final.blue.has_soul
    assert final.blue.elders == 1


def test_baron_buff_decays_to_zero():
    timeline = _timeline(
        [
            {
                "timestamp": 1000,
                "type": "ELITE_MONSTER_KILL",
                "killerTeamId": BLUE,
                "monsterType": "BARON_NASHOR",
                "killerId": 1,
            }
        ],
        frames=6,
    )
    states, _replay = _run(timeline)
    assert states[1].blue.buff_remaining("baron", states[1].t) > 0.5
    assert states[-1].blue.buff_remaining("baron", states[-1].t) == 0.0


def test_champion_kill_updates_both_sides():
    timeline = _timeline(
        [
            {
                "timestamp": 60000,
                "type": "CHAMPION_KILL",
                "killerId": 1,
                "victimId": 6,
                "assistingParticipantIds": [2],
                "bounty": 300,
            }
        ]
    )
    states, replay = _run(timeline)
    final = states[-1]
    assert final.blue.kills == 1
    assert final.red.deaths == 1
    assert any(e.type == "CHAMPION_KILL" and e.team == BLUE for e in replay.events)


def test_events_produce_extra_states_at_their_own_timestamps():
    """An objective at 1:30 should be evaluated at 1:30, not at the next frame."""
    timeline = _timeline(
        [
            {
                "timestamp": 90000,
                "type": "ELITE_MONSTER_KILL",
                "killerTeamId": BLUE,
                "monsterType": "BARON_NASHOR",
                "killerId": 1,
            }
        ],
        frames=4,
    )
    at_events, _ = _run(timeline, resolution="events")
    at_frames, _ = _run(timeline, resolution="frames")
    assert len(at_events) > len(at_frames)
    assert any(abs(state.t - 90.0) < 1.0 for state in at_events)


# -- team ids that real timelines actually contain -------------------------
#
# These three shapes all appear in live EUW ranked data and all used to be
# mis-credited, because `int(event.get("teamId") or BLUE)` reads a literal 0
# as "absent" and falls through to blue.


def test_dragon_soul_with_team_id_zero_goes_to_whoever_has_the_drakes():
    """Riot ships DRAGON_SOUL_GIVEN with teamId 0, so the owner is inferred.

    Before this was handled, every dragon soul in every game was credited to
    blue side.
    """
    drakes = [
        {
            "timestamp": 10000 + i * 1000,
            "type": "ELITE_MONSTER_KILL",
            "killerTeamId": RED,
            "monsterType": "DRAGON",
            "monsterSubType": "AIR_DRAGON",
            "killerId": 6,
        }
        for i in range(4)
    ]
    soul = {"timestamp": 20000, "type": "DRAGON_SOUL_GIVEN", "teamId": 0, "name": "Cloud"}
    states, _replay = _run(_timeline(drakes + [soul]))
    final = states[-1]
    assert final.red.soul_type == "Cloud"
    assert final.red.has_soul
    assert not final.blue.has_soul


def test_dragon_soul_is_skipped_when_nobody_qualifies():
    soul = {"timestamp": 20000, "type": "DRAGON_SOUL_GIVEN", "teamId": 0, "name": "Cloud"}
    states, replay = _run(_timeline([soul]))
    assert states[-1].blue.soul_type is None
    assert states[-1].red.soul_type is None
    assert not any(e.type == "DRAGON_SOUL" for e in replay.events)


def test_a_neutral_epic_monster_kill_is_credited_to_nobody():
    """Herald and voidgrubs taken without a champion last-hit arrive as team 300.

    Riot supplies no killerId for these either, so there is nothing to infer
    from. Crediting a side would invent an objective lead out of a data quirk.
    """
    events = [
        {
            "timestamp": 60000,
            "type": "ELITE_MONSTER_KILL",
            "killerTeamId": 300,
            "monsterType": "RIFTHERALD",
        },
        {"timestamp": 61000, "type": "ELITE_MONSTER_KILL", "killerTeamId": 300,
         "monsterType": "HORDE"},
    ]
    states, replay = _run(_timeline(events))
    assert states[-1].blue.heralds == 0
    assert states[-1].red.heralds == 0
    assert not any(e.type == "HERALD_KILL" for e in replay.events)


def test_an_epic_monster_falls_back_to_the_killer_participant():
    """A missing killerTeamId is recoverable when a champion is named."""
    events = [
        {
            "timestamp": 60000,
            "type": "ELITE_MONSTER_KILL",
            "monsterType": "BARON_NASHOR",
            "killerId": 7,  # participant 7 is red side
        }
    ]
    states, _replay = _run(_timeline(events))
    assert states[-1].red.barons == 1
    assert states[-1].blue.barons == 0


def test_building_kill_without_a_usable_team_id_is_skipped():
    events = [
        {
            "timestamp": 60000,
            "type": "BUILDING_KILL",
            "teamId": 0,
            "buildingType": "TOWER_BUILDING",
            "towerType": "OUTER_TURRET",
            "laneType": "MID_LANE",
        }
    ]
    states, _replay = _run(_timeline(events))
    assert states[-1].blue.towers_raw == 0
    assert states[-1].red.towers_raw == 0


def test_building_kill_with_no_team_id_uses_the_killer():
    events = [
        {
            "timestamp": 60000,
            "type": "BUILDING_KILL",
            "buildingType": "TOWER_BUILDING",
            "towerType": "OUTER_TURRET",
            "laneType": "MID_LANE",
            "killerId": 2,  # blue took it, so red lost the building
        }
    ]
    states, _replay = _run(_timeline(events))
    assert states[-1].blue.towers_raw == 1
    assert states[-1].red.towers_raw == 0


def test_structural_counts_are_clamped_to_what_the_map_holds():
    """Riot's data sometimes exceeds the map.

    One real timeline records three nexus-turret kills, minutes apart, where
    the map has two. Clamping keeps every state physically possible, which the
    advice engine relies on because its counterfactuals mutate these counts.
    """
    from rift_oracle.game.state import MAX_TURRETS_PER_SIDE, MAX_TURRET_WEIGHT

    events = [
        {
            "timestamp": 1000 + i * 1000,
            "type": "BUILDING_KILL",
            "teamId": RED,
            "buildingType": "TOWER_BUILDING",
            "towerType": "NEXUS_TURRET",
            "laneType": "MID_LANE",
            "killerId": 1,
        }
        for i in range(20)  # far more than the map contains
    ]
    states, _replay = _run(_timeline(events, frames=3))
    final = states[-1]
    assert final.blue.towers_raw == MAX_TURRETS_PER_SIDE
    assert final.blue.towers <= MAX_TURRET_WEIGHT + 1e-9


def test_plate_counts_allow_what_the_current_map_actually_has():
    """Plating covers all nine lane turrets now, not just the three outer ones.

    An older cap of fifteen would have silently truncated most real games.
    """
    from rift_oracle.game.state import MAX_PLATES_PER_SIDE

    assert MAX_PLATES_PER_SIDE == 45
    events = [
        {"timestamp": 1000 + i * 100, "type": "TURRET_PLATE_DESTROYED",
         "teamId": RED, "laneType": "MID_LANE"}
        for i in range(30)
    ]
    states, _replay = _run(_timeline(events, frames=3))
    assert states[-1].blue.turret_plates == 30


def test_game_duration_in_milliseconds_is_normalised():
    match = _match()
    match["info"]["gameDuration"] = 1_800_000
    replay = TimelineReplay(match, _timeline([]))
    assert replay.game_duration_s == pytest.approx(1800.0)


def _run(timeline, resolution="events", win_team=BLUE):
    replay = TimelineReplay(_match(win_team), timeline)
    return replay.run(resolution=resolution), replay


# -- live client -----------------------------------------------------------


def test_order_is_blue_and_chaos_is_red():
    assert team_from_side("ORDER") == BLUE
    assert team_from_side("CHAOS") == RED


@pytest.mark.parametrize(
    "raw,owner,lane,tier",
    [
        ("Turret_T1_C_05_A", BLUE, "Mid", "OUTER_TURRET"),
        ("Turret_T1_C_04_A", BLUE, "Mid", "INNER_TURRET"),
        ("Turret_T2_L_03_A", RED, "Top", "OUTER_TURRET"),
        ("Turret_T2_R_02_A", RED, "Bot", "INNER_TURRET"),
        ("Turret_T1_C_01_A", BLUE, "Mid", "NEXUS_TURRET"),
        ("Turret_T1_C_07_A", BLUE, "Mid", "BASE_TURRET"),
    ],
)
def test_turret_names_decode(raw, owner, lane, tier):
    assert _parse_turret(raw) == (owner, lane, tier)


def test_unknown_turret_name_is_ignored_rather_than_miscredited():
    assert _parse_turret("Something_Else")[0] is None


def test_barracks_names_decode():
    assert _parse_barracks("Barracks_T1_C1") == (BLUE, "Mid")
    assert _parse_barracks("Barracks_T2_L1") == (RED, "Top")


def _live_payload(events=None, game_time=600.0):
    def player(name, team, index):
        return {
            "championName": name,
            "team": team,
            "level": 9,
            "isDead": False,
            "respawnTimer": 0.0,
            "position": "MIDDLE",
            "riotIdGameName": f"{name}Player",
            "summonerName": f"{name}Player",
            "scores": {"kills": 2, "deaths": 1, "assists": 3, "creepScore": 80, "wardScore": 9.5},
            "items": [{"itemID": 3031, "price": 3450, "count": 1, "slot": 0}],
        }

    return {
        "gameData": {"gameTime": game_time, "mapTerrain": "Infernal", "gameMode": "CLASSIC"},
        "activePlayer": {
            "riotIdGameName": "AhriPlayer",
            "summonerName": "AhriPlayer",
            "currentGold": 1234.0,
            "championStats": {"attackDamage": 70, "abilityPower": 200, "armor": 40},
        },
        "allPlayers": (
            [player(n, "ORDER", i) for i, n in enumerate(["Ahri", "Lee Sin", "Jinx", "Thresh", "Ornn"])]
            + [player(n, "CHAOS", i) for i, n in enumerate(["Zed", "Vi", "Ashe", "Lulu", "Sion"])]
        ),
        "events": {"Events": events or []},
    }


def test_live_state_splits_teams_and_finds_the_active_player():
    tracker = LiveGameTracker()
    state = tracker.build_state(_live_payload())
    assert len(state.blue.players) == 5 and len(state.red.players) == 5
    assert tracker.active_team(_live_payload()) == BLUE
    assert tracker.active_champion(_live_payload()) == "Ahri"
    assert state.t == pytest.approx(600.0)


def test_live_active_player_gold_is_exact_and_others_are_estimated():
    tracker = LiveGameTracker()
    state = tracker.build_state(_live_payload())
    me = next(p for p in state.blue.players if p.champion == "Ahri")
    other = next(p for p in state.blue.players if p.champion == "Jinx")
    assert me.current_gold == pytest.approx(1234.0)
    assert other.current_gold >= 0.0
    assert state.blue.gold > 0


def test_live_events_are_applied_once():
    tracker = LiveGameTracker()
    events = [
        {"EventID": 0, "EventName": "GameStart", "EventTime": 0.0},
        {
            "EventID": 1,
            "EventName": "DragonKill",
            "EventTime": 300.0,
            "KillerName": "Lee Sin",
            "DragonType": "Fire",
            "Stolen": "False",
        },
    ]
    first = tracker.build_state(_live_payload(events))
    assert first.blue.dragon_count == 1
    assert len(first.events) == 2

    # Polling again returns the same event log; it must not double-count.
    second = tracker.build_state(_live_payload(events))
    assert second.blue.dragon_count == 1
    assert second.events == []


def test_live_turret_event_credits_the_other_side():
    tracker = LiveGameTracker()
    tracker.build_state(_live_payload())  # learn the champion/team mapping
    state = tracker.build_state(
        _live_payload(
            [
                {
                    "EventID": 5,
                    "EventName": "TurretKilled",
                    "EventTime": 700.0,
                    "TurretKilled": "Turret_T2_C_05_A",
                    "KillerName": "Jinx",
                }
            ]
        )
    )
    assert state.blue.towers_raw == 1
    assert state.red.towers_raw == 0


def test_live_game_end_is_detected():
    tracker = LiveGameTracker()
    tracker.build_state(
        _live_payload([{"EventID": 9, "EventName": "GameEnd", "EventTime": 1800.0, "Result": "Win"}])
    )
    assert tracker.game_over
    assert tracker.result == "Win"


# -- misc ------------------------------------------------------------------


def test_expected_gold_curve_increases():
    assert expected_team_gold(0) < expected_team_gold(600) < expected_team_gold(1800)


def test_clock_formatting():
    assert format_clock(0) == "00:00"
    assert format_clock(95) == "01:35"
    assert format_clock(3600) == "60:00"
