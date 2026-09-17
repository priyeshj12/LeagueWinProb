"""Command line interface.

    rift-oracle live                    read the game you are in right now
    rift-oracle watch  "Name#TAG"       track a game in progress by Riot ID
    rift-oracle replay NA1_5123456789   replay a finished game and explain it
    rift-oracle demo                    full pipeline on a simulated game
    rift-oracle train                   fit the model
    rift-oracle harvest                 download real matches to train on
    rift-oracle backtest                score the model and check calibration
    rift-oracle doctor                  check keys, network and model

``live`` and ``demo`` need no API key; everything else does.
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from rich.console import Console
from rich.live import Live
from rich.text import Text

from rift_oracle import __version__
from rift_oracle.config import (
    MissingAPIKey,
    NoActiveGame,
    RiftOracleError,
    Settings,
    cache_dir,
    runs_dir,
)
from rift_oracle.game.state import BLUE, RED, GameState, format_clock

log = logging.getLogger("rift_oracle")


# -- shared plumbing --------------------------------------------------------


def _console(args: argparse.Namespace) -> Console:
    return Console(
        no_color=getattr(args, "no_color", False),
        force_terminal=None if not getattr(args, "no_color", False) else False,
        highlight=False,
    )


def _settings(args: argparse.Namespace, **extra: Any) -> Settings:
    return Settings.load(
        api_key=getattr(args, "api_key", None),
        platform=getattr(args, "platform", None),
        poll_interval=getattr(args, "poll", None),
        swing_threshold=getattr(args, "threshold", None),
        live_host=getattr(args, "live_host", None),
        live_cert=getattr(args, "live_cert", None),
        live_verify=True if getattr(args, "live_cert", None) else None,
        offline=getattr(args, "offline", None),
        **extra,
    )


def _load_model(args: argparse.Namespace, console: Console):
    from rift_oracle.model.registry import load_model

    model, path = load_model(getattr(args, "model", None))
    return model, path


def _write_html(
    path: Path, track, swings, summary, perspective, advice, model_meta, console: Console
) -> None:
    from rift_oracle.ui.html_report import render_html

    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        render_html(track, swings, summary, perspective, advice, model_meta), encoding="utf-8"
    )
    console.print(Text(f"  wrote {path}", style="dim"))


def _write_json(path: Path, payload: Dict[str, Any], console: Console) -> None:
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    console.print(Text(f"  wrote {path}", style="dim"))


def _track_payload(track, swings, summary, perspective: int) -> Dict[str, Any]:
    """Machine-readable dump of a finished analysis."""
    from rift_oracle.analysis.narrate import narrate_swing

    return {
        "summary": summary,
        "perspective": "blue" if perspective == BLUE else "red",
        "trace": [
            {
                "t": round(point.t, 1),
                "clock": format_clock(point.t),
                "p_blue": round(point.p, 5),
                "logit": round(point.prediction.logit, 5),
                "contributions": {
                    key: round(value, 5)
                    for key, value in point.prediction.contributions.items()
                    if abs(value) > 1e-9
                },
            }
            for point in track.points
        ],
        "swings": [
            {
                "start": round(swing.start_t, 1),
                "end": round(swing.end_t, 1),
                "clock": swing.clock(),
                "kind": swing.kind,
                "p_before": round(swing.p_before, 5),
                "p_after": round(swing.p_after, 5),
                "delta": round(swing.delta, 5),
                "headline": narrate_swing(swing, track, limit=1).headline,
                "attributions": [
                    {"feature": key, "delta_p": round(value, 5)}
                    for key, value in swing.top_attributions(limit=6)
                ],
                "events": [
                    {"t": round(e.t, 1), "type": e.type, "text": e.text} for e in swing.events[:12]
                ],
            }
            for swing in swings
        ],
    }


# -- live -------------------------------------------------------------------


def cmd_live(args: argparse.Namespace) -> int:
    """Read the game currently running on this machine."""
    from rift_oracle.analysis.advice import advise
    from rift_oracle.analysis.swings import detect_swings, Track, TrackPoint
    from rift_oracle.game.live_adapter import LiveGameTracker
    from rift_oracle.model.features import extract
    from rift_oracle.riot.ddragon import default_ddragon
    from rift_oracle.riot.live_client import GameNotRunning, LiveClient
    from rift_oracle.ui.dashboard import build_dashboard, waiting_panel

    console = _console(args)
    settings = _settings(args)
    model, model_path = _load_model(args, console)

    ddragon = default_ddragon(settings.locale, offline=settings.offline)
    client = LiveClient(settings)
    tracker = LiveGameTracker(ddragon)
    track = Track(source="live")

    console.print(
        waiting_panel(
            "Waiting for a League game...",
            "The Live Client Data API only answers once you are in-game "
            f"(polling {settings.live_host}). Press Ctrl-C to stop.",
        )
    )

    # Block until the client starts serving, so the tool can be launched from
    # champion select and simply come alive when the game does.
    deadline = time.monotonic() + float(args.wait)
    while not client.is_available():
        if time.monotonic() > deadline:
            console.print(
                Text(
                    f"No game found after {args.wait:.0f}s. Start a game, or raise --wait.",
                    style="yellow",
                )
            )
            return 1
        time.sleep(2.0)

    perspective = BLUE
    state: Optional[GameState] = None
    advice = None
    interrupted = False

    def _stop(_signum, _frame):
        nonlocal interrupted
        interrupted = True

    previous = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, _stop)

    try:
        with Live(console=console, refresh_per_second=4, screen=False) as live:
            while not interrupted:
                try:
                    payload = client.all_game_data()
                except GameNotRunning:
                    break
                except RiftOracleError as exc:
                    console.print(Text(str(exc), style="yellow"))
                    break

                state = tracker.build_state(payload)
                perspective = tracker.active_team(payload)
                vector = extract(state)
                prediction = model.predict(vector)
                track.append(TrackPoint(state=state, vector=vector, prediction=prediction))

                swings = detect_swings(
                    track, threshold=settings.swing_threshold, window_s=args.window
                )
                advice = advise(state, model, perspective=perspective)

                confidence = tracker.gold_confidence
                subtitle = (
                    f"gold is estimated from items and income "
                    f"(agreement {confidence * 100:.0f}%) - the client only reports your own"
                )
                live.update(
                    build_dashboard(
                        track, state, swings, advice, perspective,
                        width=console.width, subtitle=subtitle,
                    )
                )

                if tracker.game_over:
                    break
                time.sleep(settings.poll_interval)
    finally:
        signal.signal(signal.SIGINT, previous)

    if state is None or not track.points:
        console.print(Text("No game data was captured.", style="yellow"))
        return 1

    console.print()
    return _finish(
        args, console, track, state, perspective, model_path,
        summary={
            "match_id": "live game",
            "duration_s": state.t,
            "blue_champions": state.blue.champions,
            "red_champions": state.red.champions,
        },
        advice=advice,
    )


# -- replay -----------------------------------------------------------------


def cmd_replay(args: argparse.Namespace) -> int:
    """Replay a finished match and explain every swing."""
    from rift_oracle.analysis.advice import advise
    from rift_oracle.analysis.swings import build_track, detect_swings
    from rift_oracle.game.timeline_adapter import replay_match
    from rift_oracle.riot.client import RiotClient
    from rift_oracle.riot.ddragon import default_ddragon
    from rift_oracle.riot.routing import platform_for_match_id

    console = _console(args)
    settings = _settings(args)
    model, model_path = _load_model(args, console)

    match_id = args.match_id.strip()
    platform = args.platform or platform_for_match_id(match_id) or settings.platform
    settings.platform = platform

    client = RiotClient(settings)
    ddragon = default_ddragon(settings.locale, offline=settings.offline)

    with console.status(f"fetching {match_id}..."):
        match = client.match(match_id, platform=platform)
        timeline = client.timeline(match_id, platform=platform)

    states, replay = replay_match(
        match, timeline, resolution=args.resolution, ddragon=ddragon
    )
    if not states:
        console.print(Text(f"{match_id} has no usable timeline frames.", style="yellow"))
        return 1

    summary = replay.summary()
    perspective = _perspective_for(args, match, replay)

    track = build_track(states, model, winner=replay.winner)
    swings = detect_swings(track, threshold=args.threshold, window_s=args.window)
    advice = advise(states[-1], model, perspective=perspective) if args.advice else None

    return _finish(
        args, console, track, states[-1], perspective, model_path,
        summary=summary, advice=advice, swings=swings,
    )


def _perspective_for(args: argparse.Namespace, match: Dict[str, Any], replay) -> int:
    """Whose side to write the report from."""
    if getattr(args, "side", None) == "red":
        return RED
    if getattr(args, "side", None) == "blue":
        return BLUE

    wanted = getattr(args, "as_player", None)
    if wanted:
        needle = wanted.split("#")[0].strip().lower()
        for participant in (match.get("info", {}) or {}).get("participants", []) or []:
            names = {
                str(participant.get("riotIdGameName") or "").lower(),
                str(participant.get("summonerName") or "").lower(),
                str(participant.get("championName") or "").lower(),
            }
            if needle in names:
                return int(participant.get("teamId", BLUE))
    return BLUE


# -- watch ------------------------------------------------------------------


def cmd_watch(args: argparse.Namespace) -> int:
    """Track a game in progress via the spectator API, then explain it."""
    from rift_oracle.analysis.swings import Track, TrackPoint
    from rift_oracle.game.scaling import comp_scale, crossover_minute
    from rift_oracle.model.features import extract
    from rift_oracle.riot.client import RiotAPIError, RiotClient
    from rift_oracle.riot.ddragon import default_ddragon
    from rift_oracle.riot.routing import queue_name
    from rift_oracle.ui.dashboard import waiting_panel

    console = _console(args)
    settings = _settings(args)
    settings.require_key("watch")
    model, model_path = _load_model(args, console)

    client = RiotClient(settings)
    ddragon = default_ddragon(settings.locale, offline=settings.offline)

    with console.status(f"resolving {args.riot_id}..."):
        account = client.resolve_riot_id(args.riot_id)
        puuid = account["puuid"]
        display = f"{account.get('gameName', '?')}#{account.get('tagLine', '?')}"

    console.print(
        waiting_panel(
            f"Watching {display}",
            "The Riot Web API exposes the draft and the clock for a game in "
            "progress, but not live gold or kills - only the in-client Live "
            "Client Data API can see those, and only for the game you are "
            "playing. So this view tracks the composition-and-clock odds while "
            "the game runs, then pulls the full timeline and explains every "
            "swing the moment it ends.\n\n"
            "Run 'rift-oracle live' instead if this is your own game.",
        )
    )

    game = _await_game(client, puuid, settings, console, args)
    if game is None:
        return 1

    game_id = game.get("gameId")
    if args.game_id and int(args.game_id) != int(game_id or 0):
        console.print(
            Text(
                f"{display} is in game {game_id}, not {args.game_id}. Watching {game_id}.",
                style="yellow",
            )
        )

    blue_ids = [p for p in game.get("participants", []) if int(p.get("teamId", 100)) == BLUE]
    red_ids = [p for p in game.get("participants", []) if int(p.get("teamId", 100)) == RED]
    blue_champions = [ddragon.champion_name(int(p["championId"])) for p in blue_ids]
    red_champions = [ddragon.champion_name(int(p["championId"])) for p in red_ids]

    me = next((p for p in game.get("participants", []) if p.get("puuid") == puuid), None)
    perspective = int(me.get("teamId", BLUE)) if me else BLUE

    rank_prior = _rank_prior(client, blue_ids, red_ids, settings, console) if args.ranks else 0.0

    console.print()
    console.print(
        Text.assemble(
            ("  game ", "dim"), (str(game_id), "bold"),
            ("  ·  ", "dim"), (queue_name(game.get("gameQueueConfigId")), ""),
            ("  ·  you are on ", "dim"),
            ("Blue" if perspective == BLUE else "Red",
             "bold cyan" if perspective == BLUE else "bold red"),
        )
    )
    console.print(Text(f"  blue  {', '.join(blue_champions)}", style="cyan"))
    console.print(Text(f"  red   {', '.join(red_champions)}", style="red"))

    crossover = crossover_minute(blue_champions, red_champions)
    blue_scale, red_scale = comp_scale(blue_champions), comp_scale(red_champions)
    console.print(
        Text(
            f"  scaling: blue {blue_scale:+.2f}  red {red_scale:+.2f}"
            + (f"  ·  curves cross near {crossover:.0f}:00" if crossover else ""),
            style="dim",
        )
    )
    if rank_prior:
        console.print(Text(f"  rank prior: {rank_prior:+.2f} (blue)", style="dim"))

    _print_draft_projection(
        console, model, game, blue_champions, red_champions, rank_prior, perspective
    )
    console.print()

    track = Track(source="spectator")
    last_length = -1

    try:
        while True:
            state = _draft_state(
                game, blue_champions, red_champions, float(game.get("gameLength") or 0)
            )
            vector = extract(state, rank_prior=rank_prior)
            prediction = model.predict(vector)
            track.append(TrackPoint(state=state, vector=vector, prediction=prediction))

            p = prediction.p if perspective == BLUE else 1 - prediction.p
            length = float(game.get("gameLength") or 0)
            if int(length) != last_length:
                console.print(
                    f"  {format_clock(length)}  draft-and-clock odds for you: {p:.1%}"
                )
                last_length = int(length)

            time.sleep(max(settings.poll_interval, 10.0))
            try:
                game = client.active_game(puuid)
            except RiotAPIError as exc:
                if exc.status in (404, 403):
                    game = None
                else:
                    raise
            if game is None:
                console.print(Text("\n  game over - pulling the timeline...", style="dim"))
                break
    except KeyboardInterrupt:
        console.print(Text("\n  stopped watching.", style="dim"))
        if not args.then_replay:
            return 0

    if not args.then_replay:
        return 0

    match_id = f"{game_id_platform(settings.platform)}_{game_id}"
    return _replay_after_watch(args, console, client, match_id, model, model_path, perspective)


def _print_draft_projection(
    console: Console,
    model,
    game: Dict[str, Any],
    blue_champions: List[str],
    red_champions: List[str],
    rank_prior: float,
    perspective: int,
) -> None:
    """Show what the draft alone is worth across the length of a game.

    Spectator sees the draft and the clock and nothing else, so the live
    readout barely moves - which is honest but not much use on its own. What
    the draft *does* determine is how its value changes with time, and that is
    knowable up front: composition scaling is a deterministic function of the
    clock. Projecting it forward turns "50% and holding" into "50% now, and
    your draft is worth two more points by 35 minutes, so play for time".
    """
    from rift_oracle.model.features import extract
    from rift_oracle.ui.chart import winprob_chart

    marks = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
    curve = []
    for minute in marks:
        state = _draft_state(game, blue_champions, red_champions, minute * 60.0)
        prediction = model.predict(extract(state, rank_prior=rank_prior))
        p = prediction.p if perspective == BLUE else 1.0 - prediction.p
        curve.append((minute * 60.0, p))

    start, end = curve[0][1], curve[-1][1]
    drift = end - start
    console.print()
    console.print(
        Text("  draft outlook (composition and clock only, before a single minion dies)",
             style="dim")
    )
    console.print(
        winprob_chart(
            [t for t, _p in curve], [p for _t, p in curve],
            width=min(console.width - 4, 78), height=7,
        )
    )
    if abs(drift) < 0.01:
        verdict = "the clock favours neither draft"
    elif drift > 0:
        verdict = (
            f"your draft gains {drift * 100:.1f} points by 45:00 - the longer this "
            "goes, the better for you"
        )
    else:
        verdict = (
            f"your draft loses {abs(drift) * 100:.1f} points by 45:00 - win it early "
            "or it slips away"
        )
    console.print(Text(f"  {verdict}", style="italic"))


def game_id_platform(platform: str) -> str:
    """Match ids use the upper-cased platform id as their prefix."""
    return platform.upper()


def _replay_after_watch(
    args, console: Console, client, match_id: str, model, model_path, perspective: int
) -> int:
    from rift_oracle.analysis.advice import advise
    from rift_oracle.analysis.swings import build_track, detect_swings
    from rift_oracle.game.timeline_adapter import replay_match
    from rift_oracle.riot.client import RiotAPIError

    # Riot needs a short moment to publish a match after the nexus falls.
    for attempt in range(8):
        try:
            match = client.match(match_id)
            timeline = client.timeline(match_id)
            break
        except RiotAPIError as exc:
            if exc.status != 404 or attempt == 7:
                console.print(
                    Text(
                        f"  could not fetch {match_id} yet: {exc}\n"
                        f"  try: rift-oracle replay {match_id}",
                        style="yellow",
                    )
                )
                return 1
            wait = 15.0 * (attempt + 1)
            console.print(Text(f"  not published yet, retrying in {wait:.0f}s...", style="dim"))
            time.sleep(wait)
    else:  # pragma: no cover - loop always breaks or returns
        return 1

    states, replay = replay_match(match, timeline, resolution=args.resolution)
    if not states:
        console.print(Text("timeline had no frames", style="yellow"))
        return 1

    track = build_track(states, model, winner=replay.winner)
    swings = detect_swings(track, threshold=args.threshold, window_s=args.window)
    advice = advise(states[-1], model, perspective=perspective)
    return _finish(
        args, console, track, states[-1], perspective, model_path,
        summary=replay.summary(), advice=advice, swings=swings,
    )


def _await_game(client, puuid: str, settings, console: Console, args) -> Optional[Dict[str, Any]]:
    """Poll the spectator endpoint until the player is in a game."""
    from rift_oracle.riot.client import RiotAPIError

    deadline = time.monotonic() + float(args.wait)
    while True:
        try:
            game = client.active_game(puuid)
        except RiotAPIError as exc:
            if exc.status not in (404, 403):
                raise
            game = None
        if game:
            return game
        if time.monotonic() > deadline:
            console.print(
                Text(
                    f"Not in a game after {args.wait:.0f}s. "
                    "Start one, or raise --wait, or use 'replay' on a finished match.",
                    style="yellow",
                )
            )
            return None
        time.sleep(15.0)


def _rank_prior(client, blue_ids, red_ids, settings, console: Console) -> float:
    """Average ranked tier difference, as a prior in logit-ish units."""
    from rift_oracle.model.features import rank_prior_from_entries, rank_score

    def scores(participants) -> List[float]:
        out: List[float] = []
        for participant in participants:
            puuid = participant.get("puuid")
            if not puuid:
                continue
            try:
                entries = client.league_entries(puuid)
            except RiftOracleError:
                continue
            solo = next(
                (e for e in entries if e.get("queueType") == "RANKED_SOLO_5x5"),
                entries[0] if entries else None,
            )
            if solo:
                out.append(
                    rank_score(
                        solo.get("tier", "GOLD"),
                        solo.get("rank", "I"),
                        int(solo.get("leaguePoints") or 0),
                    )
                )
        return out

    with console.status("fetching ranks (10 requests)..."):
        return rank_prior_from_entries(scores(blue_ids), scores(red_ids))


def _draft_state(
    game: Dict[str, Any], blue_champions: List[str], red_champions: List[str], t: float
) -> GameState:
    """A game state carrying only what spectator can see: draft and clock."""
    from rift_oracle.game.state import PlayerState, TeamState

    def side(team_id: int, champions: List[str], offset: int) -> TeamState:
        return TeamState(
            team_id=team_id,
            players=[
                PlayerState(
                    participant_id=offset + i,
                    team=team_id,
                    champion=name,
                    total_gold=0,
                    current_gold=0.0,
                )
                for i, name in enumerate(champions)
            ],
        )

    return GameState(
        t=max(0.0, float(t)),
        blue=side(BLUE, blue_champions, 1),
        red=side(RED, red_champions, 6),
        match_id=str(game.get("gameId") or ""),
        queue_id=game.get("gameQueueConfigId"),
        source="spectator",
    )


# -- demo -------------------------------------------------------------------


def cmd_demo(args: argparse.Namespace) -> int:
    """Run the whole pipeline on a simulated game. No API key needed."""
    from rift_oracle.analysis.advice import advise
    from rift_oracle.analysis.swings import build_track, detect_swings
    from rift_oracle.sim.synth import simulate_game

    console = _console(args)
    model, model_path = _load_model(args, console)

    console.print(Text(f"simulating a ranked game (seed {args.seed})...", style="dim"))
    game = simulate_game(seed=args.seed)
    track = build_track(game.states, model, winner=game.winner)
    swings = detect_swings(track, threshold=args.threshold, window_s=args.window)
    perspective = BLUE if args.side != "red" else RED

    if args.replay_speed > 0:
        _animate(console, track, game, swings, model, perspective, args)

    advice = advise(game.states[-1], model, perspective=perspective)
    return _finish(
        args, console, track, game.states[-1], perspective, model_path,
        summary=game.summary(), advice=advice, swings=swings,
    )


def _animate(console, track, game, swings, model, perspective, args) -> None:
    """Play the simulated game back through the live dashboard."""
    from rift_oracle.analysis.advice import advise
    from rift_oracle.analysis.swings import detect_swings, Track
    from rift_oracle.ui.dashboard import build_dashboard

    partial = Track(source="sim")
    delay = 1.0 / max(args.replay_speed, 0.01)

    try:
        with Live(console=console, refresh_per_second=8, screen=False) as live:
            for index, point in enumerate(track.points):
                partial.append(point)
                if index % 2 and index < len(track.points) - 1:
                    continue
                current_swings = detect_swings(
                    partial, threshold=args.threshold, window_s=args.window
                )
                advice = advise(point.state, model, perspective=perspective)
                live.update(
                    build_dashboard(
                        partial, point.state, current_swings, advice, perspective,
                        width=console.width,
                        subtitle="simulated game - 'rift-oracle live' reads a real one",
                    )
                )
                time.sleep(delay)
    except KeyboardInterrupt:
        pass
    console.print()


# -- shared finish ----------------------------------------------------------


def _finish(
    args: argparse.Namespace,
    console: Console,
    track,
    state: GameState,
    perspective: int,
    model_path,
    summary: Dict[str, Any],
    advice=None,
    swings=None,
) -> int:
    """Render the post-game report and write whatever outputs were asked for."""
    from rift_oracle.analysis.swings import detect_swings
    from rift_oracle.ui.report import render_compact, render_report

    if swings is None:
        swings = detect_swings(
            track, threshold=args.threshold, window_s=args.window
        )

    if getattr(args, "compact", False):
        render_compact(track, swings, console, perspective)
    else:
        render_report(
            track, swings, summary, console, perspective,
            advice=advice, model_path=str(model_path), detail_limit=args.detail,
        )

    if getattr(args, "html", None):
        from rift_oracle.model.registry import model_info

        _write_html(
            Path(args.html), track, swings, summary, perspective, advice,
            model_info(Path(model_path)), console,
        )
    if getattr(args, "json_out", None):
        _write_json(
            Path(args.json_out), _track_payload(track, swings, summary, perspective), console
        )
    return 0


# -- train / harvest / backtest --------------------------------------------


def cmd_train(args: argparse.Namespace) -> int:
    from rift_oracle.model.registry import save_user_model
    from rift_oracle.model.train import (
        format_report,
        train_from_directory,
        train_synthetic,
    )

    console = _console(args)

    l2 = None if str(args.l2).lower() == "auto" else float(args.l2)

    if args.data:
        console.print(f"training on real matches in {args.data}")
        with console.status("replaying timelines..."):
            model, report = train_from_directory(
                Path(args.data), l2=l2, limit=args.limit, verbose=args.verbose
            )
    else:
        console.print(
            f"training on {args.games:,} simulated games "
            "(no API key or network needed; use --data to fit on real matches)"
        )
        with console.status("simulating and fitting..."):
            model, report = train_synthetic(
                n_games=args.games, seed=args.seed, l2=l2, verbose=args.verbose
            )

    console.print()
    console.print(format_report(report))

    console.print("\ntop factors by mean absolute contribution:")
    for key, value in sorted(report["importance"].items(), key=lambda kv: -kv[1])[:10]:
        console.print(f"  {key:<20} {value:.4f}")

    if args.out:
        path = model.save(Path(args.out))
    else:
        path = save_user_model(model)
    console.print(Text(f"\nsaved model to {path}", style="bold"))

    if args.report:
        _write_json(Path(args.report), report, console)
    return 0


def cmd_harvest(args: argparse.Namespace) -> int:
    """Download match + timeline pairs to train on."""
    import random

    from rift_oracle.riot.client import RiotAPIError, RiotClient

    console = _console(args)
    settings = _settings(args)
    settings.require_key("harvest")
    client = RiotClient(settings)

    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = _seed_players(client, args, console)
    if not seeds:
        console.print(
            Text("no seed players. Pass --riot-id, --tier, or --ladder.", style="yellow")
        )
        return 1

    rng = random.Random(args.seed)
    rng.shuffle(seeds)
    console.print(f"  {len(seeds)} seed players, sampling up to "
                  f"{args.per_player} games each")

    seen_matches = {path.name.split(".")[0] for path in out_dir.glob("*.match.json")}
    if seen_matches:
        console.print(Text(f"  {len(seen_matches)} matches already on disk", style="dim"))

    queue = None if args.queue == 0 else args.queue
    written = len(seen_matches)
    target = args.count
    failures = 0

    with console.status("harvesting...") as status:
        for puuid in seeds:
            if written >= target:
                break
            try:
                ids = client.match_ids(puuid, count=args.per_player, queue=queue)
            except RiotAPIError as exc:
                failures += 1
                log.debug("match ids failed for a player: %s", exc)
                continue

            for match_id in ids:
                if written >= target:
                    break
                if match_id in seen_matches:
                    continue
                seen_matches.add(match_id)

                match_path = out_dir / f"{match_id}.match.json"
                timeline_path = out_dir / f"{match_id}.timeline.json"
                try:
                    match = client.match(match_id)
                    timeline = client.timeline(match_id)
                except RiotAPIError as exc:
                    failures += 1
                    log.debug("%s: %s", match_id, exc)
                    continue

                match_path.write_text(json.dumps(match), encoding="utf-8")
                timeline_path.write_text(json.dumps(timeline), encoding="utf-8")
                written += 1
                status.update(
                    f"harvested {written}/{target} matches  "
                    f"({client.request_count} requests, {client.cache_hits} cached, "
                    f"{failures} skipped)"
                )

    console.print(
        f"\nwrote {written} match/timeline pairs to {out_dir} "
        f"({client.request_count} API requests, {failures} skipped)"
    )
    console.print(Text(f"next: rift-oracle train --data {out_dir}", style="dim"))
    return 0


#: A spread across the tiers most EUW ranked games are actually played in.
#: Seeding only from challenger would train the model on a population whose
#: games look nothing like the ones it will be asked about.
LADDER_SPREAD = [
    ("BRONZE", "II"), ("SILVER", "II"), ("GOLD", "II"), ("GOLD", "IV"),
    ("PLATINUM", "II"), ("PLATINUM", "IV"), ("EMERALD", "II"), ("EMERALD", "IV"),
    ("DIAMOND", "II"), ("DIAMOND", "IV"), ("MASTER", "I"),
]


def _seed_players(client, args: argparse.Namespace, console: Console) -> List[str]:
    """Collect seed puuids from explicit Riot IDs and from ladder tiers."""
    from rift_oracle.riot.client import RiotAPIError

    seeds: List[str] = []
    seen: set = set()

    for riot_id in args.riot_id:
        try:
            account = client.resolve_riot_id(riot_id)
        except RiftOracleError as exc:
            console.print(Text(f"  {riot_id}: {exc}", style="yellow"))
            continue
        if account["puuid"] not in seen:
            seen.add(account["puuid"])
            seeds.append(account["puuid"])
            console.print(f"  seed {riot_id}")

    tiers: List[Tuple[str, str]] = []
    if args.ladder:
        tiers.extend(LADDER_SPREAD)
    for spec in args.tier:
        tier, _, division = spec.partition(":")
        tiers.append((tier.upper(), (division or "I").upper()))

    for tier, division in tiers:
        try:
            entries = client.league_entries_by_tier(
                tier, division, page=args.page, platform=settings_platform(client)
            )
        except RiotAPIError as exc:
            console.print(Text(f"  {tier} {division}: {exc}", style="yellow"))
            continue
        added = 0
        for entry in entries[: args.per_tier]:
            puuid = entry.get("puuid")
            if puuid and puuid not in seen:
                seen.add(puuid)
                seeds.append(puuid)
                added += 1
        console.print(f"  seed {tier} {division}: {added} players")

    return seeds


def settings_platform(client) -> str:
    return client.platform


def cmd_backtest(args: argparse.Namespace) -> int:
    """Score the current model and check its calibration."""
    from rift_oracle.model import calibrate
    from rift_oracle.model.train import Dataset, load_harvested
    from rift_oracle.sim.synth import simulate_dataset

    console = _console(args)
    model, model_path = _load_model(args, console)

    if args.data:
        with console.status("loading matches..."):
            dataset = load_harvested(Path(args.data), limit=args.limit)
        source = f"{dataset.n_games} real matches from {args.data}"
    else:
        with console.status(f"simulating {args.games} games..."):
            arrays = simulate_dataset(n_games=args.games, seed=args.seed + 1000)
        dataset = Dataset(
            values=arrays["values"], masks=arrays["masks"], times=arrays["times"],
            labels=arrays["labels"], groups=arrays["groups"],
        )
        source = f"{dataset.n_games} freshly simulated games (held out from training)"

    if args.live_mask:
        dataset = dataset.with_live_masks()
        source += ", restricted to what a live game can observe"

    probabilities = model.predict_proba(dataset.values, dataset.masks, dataset.times)
    scores = calibrate.metrics(dataset.labels, probabilities)

    console.print(f"model: {model_path}")
    console.print(f"data:  {source}\n")
    console.print(
        f"  log loss {scores['log_loss']:.4f}   Brier {scores['brier']:.4f}   "
        f"AUC {scores['auc']:.4f}   accuracy {scores['accuracy'] * 100:.1f}%   "
        f"ECE {scores['ece'] * 100:.2f}%"
    )

    console.print("\naccuracy and confidence by game clock:")
    for row in calibrate.metrics_by_minute(dataset.labels, probabilities, dataset.times):
        console.print(
            f"  {row['from_min']:>3}-{row['to_min']:<3} min   n={row['n']:>7,}   "
            f"acc {row['accuracy'] * 100:5.1f}%   log loss {row['log_loss']:.3f}   "
            f"mean confidence {row['mean_confidence']:.2f}"
        )

    console.print("\ncalibration (a well-fit model tracks the diagonal):")
    for row in calibrate.reliability_table(dataset.labels, probabilities, bins=10):
        gap = row["observed"] - row["mean_predicted"]
        bar = "#" * max(1, int(row["count"] / max(len(dataset) / 60, 1)))
        console.print(
            f"  said {row['mean_predicted'] * 100:5.1f}%  actually "
            f"{row['observed'] * 100:5.1f}%  ({gap * 100:+5.1f})  n={row['count']:>6,}  {bar}"
        )

    gain = calibrate.calibration_gain(dataset.labels, probabilities)
    if gain["gain"] == gain["gain"]:
        verdict = (
            "already calibrated" if gain["gain"] < 0.005 else "recalibration would help"
        )
        console.print(f"\nisotonic recalibration gain: {gain['gain']:+.4f} nats ({verdict})")

    if args.json_out:
        _write_json(
            Path(args.json_out),
            {
                "model": str(model_path),
                "metrics": scores,
                "by_minute": calibrate.metrics_by_minute(
                    dataset.labels, probabilities, dataset.times
                ),
                "reliability": calibrate.reliability_table(dataset.labels, probabilities),
            },
            console,
        )
    return 0


# -- doctor / configure -----------------------------------------------------


def cmd_doctor(args: argparse.Namespace) -> int:
    """Check the key, the network, the client, and the model."""
    from rift_oracle.model.registry import find_model_path, model_info
    from rift_oracle.riot.client import RiotAPIError, RiotClient
    from rift_oracle.riot.ddragon import default_ddragon
    from rift_oracle.riot.live_client import LiveClient
    from rift_oracle.riot.routing import platform_host, regional_host, regional_route

    console = _console(args)
    settings = _settings(args)
    ok = True

    def line(label: str, good: bool, detail: str = "") -> None:
        nonlocal ok
        ok = ok and good
        mark = Text("  OK  ", style="bold green") if good else Text(" FAIL ", style="bold red")
        console.print(Text.assemble(mark, (f" {label:<26}", ""), (detail, "dim")))

    console.print(Text(f"rift_oracle {__version__}", style="bold"))
    console.print(Text(f"  platform {settings.platform}  "
                       f"({platform_host(settings.platform)}, {regional_host(settings.platform)})\n",
                       style="dim"))

    line(
        "API key present",
        bool(settings.api_key),
        f"...{settings.api_key[-4:]}" if settings.api_key else "set RIOT_API_KEY, or use live/demo",
    )

    if settings.api_key:
        client = RiotClient(settings)
        # Probe with champion-rotations, not spectator's featured-games: that
        # endpoint is not granted to development keys, so a health check built
        # on it reports a perfectly good key as rejected.
        try:
            client.champion_rotations()
            line(f"platform API ({settings.platform})", True, "champion-rotations answered")
        except RiotAPIError as exc:
            hint = " (development keys expire after 24h)" if exc.status == 403 else ""
            line(f"platform API ({settings.platform})", False, f"{exc}{hint}")
        except RiftOracleError as exc:
            line(f"platform API ({settings.platform})", False, str(exc))

        # The regional host is a separate routing value with its own limits and
        # its own grants, so it needs its own probe.
        region = regional_route(settings.platform)
        if args.riot_id:
            try:
                account = client.resolve_riot_id(args.riot_id)
                puuid = account["puuid"]
                line(
                    f"regional API ({region})", True,
                    f"{account.get('gameName')}#{account.get('tagLine')} resolved",
                )
                ids = client.match_ids(puuid, count=1, queue=420)
                line(
                    "ranked match history", True,
                    f"most recent solo-queue game: {ids[0]}" if ids
                    else "no ranked solo games on this account",
                )
                if ids:
                    client.timeline(ids[0])
                    line("match timeline", True, "downloadable, so replay will work")
                game = client.active_game(puuid)
                line(
                    "spectator", True,
                    f"in game {game.get('gameId')} right now" if game else "not in a game",
                )
            except RiotAPIError as exc:
                line(f"regional API ({region})", False, str(exc))
        else:
            line(
                f"regional API ({region})", True,
                "pass --riot-id 'Name#TAG' for an end-to-end check",
            )

    ddragon = default_ddragon(settings.locale, offline=settings.offline)
    champions = ddragon.champions()
    line(
        "Data Dragon metadata",
        bool(champions),
        f"{len(champions)} champions, patch {ddragon.version}" if champions
        else "unreachable; names and item costs fall back to a bundled table",
    )

    live = LiveClient(settings)
    running = live.is_available()
    line(
        "League client in-game",
        True,
        "game in progress - 'rift-oracle live' will attach" if running
        else "not in a game (this is fine unless you wanted 'live')",
    )

    model_path = find_model_path(getattr(args, "model", None))
    if model_path:
        meta = model_info(model_path)
        line(
            "Win-probability model",
            True,
            f"{model_path.name}, trained on {meta.get('trained_on', '?')} "
            f"({meta.get('n_games', '?')} games)",
        )
    else:
        line("Win-probability model", False, "run 'rift-oracle train'")

    console.print(Text(f"\n  cache {cache_dir()}\n  runs  {runs_dir()}", style="dim"))
    return 0 if ok else 1


def cmd_configure(args: argparse.Namespace) -> int:
    """Store the API key and default platform."""
    console = _console(args)
    settings = _settings(args)
    if args.api_key:
        settings.api_key = args.api_key
    path = settings.save()
    console.print(f"saved to {path}")
    console.print(
        Text(
            f"  platform {settings.platform}   "
            f"key {'set' if settings.api_key else 'not set'}",
            style="dim",
        )
    )
    return 0


def cmd_clear_cache(args: argparse.Namespace) -> int:
    from rift_oracle.riot.client import DiskCache

    console = _console(args)
    removed = DiskCache().clear()
    console.print(f"removed {removed} cached responses from {cache_dir()}")
    return 0


# -- argument parsing -------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rift-oracle",
        description=(
            "Live win-probability oracle for League of Legends: read the game "
            "state, get the odds, see when they swung and why."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--version", action="version", version=f"rift_oracle {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="log what it is doing")
    parser.add_argument("--no-color", action="store_true", help="plain output, no ANSI")

    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser, *, needs_key: bool = True) -> None:
        if needs_key:
            p.add_argument("--api-key", help="Riot API key (default: $RIOT_API_KEY)")
            p.add_argument("--platform", help="na1, euw1, kr, ... (default: na1)")
        p.add_argument("--model", help="path to a model json (default: the trained one)")
        p.add_argument(
            "--threshold", type=float, default=0.06,
            help="minimum probability move that counts as a swing (default 0.06)",
        )
        p.add_argument(
            "--window", type=float, default=90.0,
            help="seconds a swing may span (default 90)",
        )
        p.add_argument("--detail", type=int, default=5, help="swings to narrate in full")
        p.add_argument("--compact", action="store_true", help="one-line summary instead of the report")
        p.add_argument("--html", help="also write a standalone HTML report here")
        p.add_argument("--json-out", dest="json_out", help="also write machine-readable JSON here")

    # live
    p_live = sub.add_parser("live", help="read the game running on this machine (no key needed)")
    common(p_live, needs_key=False)
    p_live.add_argument("--poll", type=float, default=2.0, help="seconds between polls")
    p_live.add_argument("--wait", type=float, default=600.0, help="seconds to wait for a game")
    p_live.add_argument("--live-host", default=None, help="override https://127.0.0.1:2999")
    p_live.add_argument("--live-cert", default=None, help="riotgames.pem, to verify TLS properly")
    p_live.set_defaults(func=cmd_live)

    # watch
    p_watch = sub.add_parser("watch", help="track a game in progress by Riot ID")
    common(p_watch)
    p_watch.add_argument("riot_id", help="Name#TAG of a player in the game")
    p_watch.add_argument("--game-id", help="expected game id, to confirm you got the right one")
    p_watch.add_argument("--poll", type=float, default=20.0, help="seconds between polls")
    p_watch.add_argument("--wait", type=float, default=900.0, help="seconds to wait for a game")
    p_watch.add_argument("--ranks", action="store_true", help="fetch ranks for a skill prior (10 requests)")
    p_watch.add_argument("--resolution", choices=("frames", "events"), default="events")
    p_watch.add_argument(
        "--no-replay", dest="then_replay", action="store_false",
        help="do not auto-explain the game once it ends",
    )
    p_watch.set_defaults(func=cmd_watch, then_replay=True)

    # replay
    p_replay = sub.add_parser("replay", help="replay a finished match and explain every swing")
    common(p_replay)
    p_replay.add_argument("match_id", help="e.g. NA1_5123456789")
    p_replay.add_argument("--as-player", help="write the report from this player's side")
    p_replay.add_argument("--side", choices=("blue", "red"), help="write the report from this side")
    p_replay.add_argument(
        "--resolution", choices=("frames", "events"), default="events",
        help="'events' also samples at the instant of every objective (default)",
    )
    p_replay.add_argument(
        "--no-advice", dest="advice", action="store_false",
        help="skip the end-state suggestions",
    )
    p_replay.set_defaults(func=cmd_replay, advice=True)

    # demo
    p_demo = sub.add_parser("demo", help="run the whole pipeline on a simulated game (no key)")
    common(p_demo, needs_key=False)
    p_demo.add_argument("--seed", type=int, default=7, help="simulation seed")
    p_demo.add_argument("--side", choices=("blue", "red"), default="blue")
    p_demo.add_argument(
        "--replay-speed", type=float, default=0.0,
        help="frames per second to animate through the live dashboard (0 = skip)",
    )
    p_demo.set_defaults(func=cmd_demo)

    # train
    p_train = sub.add_parser("train", help="fit the win-probability model")
    p_train.add_argument("--data", help="directory of harvested matches (default: simulate)")
    p_train.add_argument("--games", type=int, default=6000, help="games to simulate")
    p_train.add_argument("--limit", type=int, help="cap on harvested matches to use")
    p_train.add_argument(
        "--l2", default="auto",
        help="ridge penalty, or 'auto' to cross-validate it (default: auto)",
    )
    p_train.add_argument("--seed", type=int, default=7)
    p_train.add_argument("--out", help="write the model here instead of the default location")
    p_train.add_argument("--report", help="write the training report as JSON")
    p_train.set_defaults(func=cmd_train)

    # harvest
    p_harvest = sub.add_parser("harvest", help="download matches and timelines to train on")
    p_harvest.add_argument("--api-key")
    p_harvest.add_argument("--platform")
    p_harvest.add_argument("--riot-id", action="append", default=[], help="seed player, repeatable")
    p_harvest.add_argument(
        "--tier", action="append", default=[],
        help="seed from a ranked tier, e.g. --tier EMERALD:II (repeatable)",
    )
    p_harvest.add_argument(
        "--ladder", action="store_true",
        help="seed from a spread of tiers across the ladder",
    )
    p_harvest.add_argument("--per-tier", type=int, default=60, help="players per tier")
    p_harvest.add_argument("--per-player", type=int, default=5, help="matches per player")
    p_harvest.add_argument("--page", type=int, default=1, help="ranked-entries page")
    p_harvest.add_argument("--count", type=int, default=200, help="matches to fetch")
    p_harvest.add_argument("--queue", type=int, default=420, help="queue id, 0 for any")
    p_harvest.add_argument("--out", default="matches", help="output directory")
    p_harvest.add_argument("--seed", type=int, default=0, help="shuffle seed")
    p_harvest.set_defaults(func=cmd_harvest)

    # backtest
    p_backtest = sub.add_parser("backtest", help="score the model and check calibration")
    p_backtest.add_argument("--model")
    p_backtest.add_argument("--data", help="directory of harvested matches")
    p_backtest.add_argument("--games", type=int, default=1500, help="games to simulate")
    p_backtest.add_argument("--limit", type=int)
    p_backtest.add_argument("--seed", type=int, default=7)
    p_backtest.add_argument(
        "--live-mask", action="store_true",
        help="score using only what a live game can observe",
    )
    p_backtest.add_argument("--json-out", dest="json_out")
    p_backtest.set_defaults(func=cmd_backtest)

    # doctor
    p_doctor = sub.add_parser("doctor", help="check keys, network, client and model")
    p_doctor.add_argument("--api-key")
    p_doctor.add_argument("--platform")
    p_doctor.add_argument("--model")
    p_doctor.add_argument(
        "--riot-id", help="Name#TAG to check the account, history and spectator paths"
    )
    p_doctor.set_defaults(func=cmd_doctor, riot_id=None)

    # configure
    p_conf = sub.add_parser("configure", help="save the API key and default platform")
    p_conf.add_argument("--api-key")
    p_conf.add_argument("--platform")
    p_conf.set_defaults(func=cmd_configure)

    # clear-cache
    p_cache = sub.add_parser("clear-cache", help="delete cached Riot API responses")
    p_cache.set_defaults(func=cmd_clear_cache)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    try:
        return int(args.func(args) or 0)
    except MissingAPIKey as exc:
        Console(stderr=True).print(Text(str(exc), style="yellow"))
        return 2
    except (NoActiveGame, RiftOracleError) as exc:
        Console(stderr=True).print(Text(f"error: {exc}", style="red"))
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
