"""The live terminal dashboard.

One screen, refreshed every couple of seconds: the current odds, the trace so
far, the scoreboard, what the model is reacting to, what to do about it, and
the swings that already happened. Everything on it is derived from the same
model evaluation, so nothing on the screen can disagree with anything else.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from rich.box import ROUNDED, SIMPLE
from rich.console import Group, RenderableType
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from rift_oracle.analysis.advice import Advice
from rift_oracle.analysis.narrate import momentum_phrase, narrate_swing, points, standing
from rift_oracle.analysis.swings import Swing, Track
from rift_oracle.game.state import BLUE, GameState, format_clock, other_team
from rift_oracle.model.features import display_value
from rift_oracle.ui.chart import winprob_chart

US_STYLE = "bold cyan"
THEM_STYLE = "bold red"


def _confidence_style(p: float) -> str:
    if p >= 0.70:
        return "bold green"
    if p >= 0.55:
        return "green"
    if p >= 0.45:
        return "yellow"
    if p >= 0.30:
        return "bright_red"
    return "bold red"


def odds_header(
    track: Track,
    state: GameState,
    perspective: int = BLUE,
    subtitle: str = "",
) -> Panel:
    """The big number, a two-sided bar, and which way it is moving."""
    point = track.latest
    p = 0.5 if point is None else (point.p if perspective == BLUE else 1 - point.p)

    bar_width = 44
    filled = int(round(p * bar_width))
    bar = Text()
    bar.append("#" * filled, style=US_STYLE)
    bar.append("#" * (bar_width - filled), style=THEM_STYLE)

    line = Text()
    line.append("  YOU ", style="dim")
    line.append(f"{p * 100:5.1f}%  ", style=_confidence_style(p))
    line.append_text(bar)
    line.append(f"  {(1 - p) * 100:5.1f}% ", style=_confidence_style(1 - p))
    line.append("THEM", style="dim")

    body: List[RenderableType] = [line]
    trend = momentum_phrase(track, window_s=120.0, perspective=perspective)
    body.append(Text("  " + trend, style="italic dim"))
    if subtitle:
        body.append(Text("  " + subtitle, style="dim"))

    title = f"rift_oracle  ·  {format_clock(state.t)}  ·  {state.source} game"
    return Panel(Group(*body), title=title, title_align="left", border_style="dim", box=ROUNDED)


def trace_panel(track: Track, swings: Sequence[Swing], width: int, perspective: int = BLUE) -> Panel:
    times = track.times
    probabilities = track.probabilities
    if perspective != BLUE:
        probabilities = [1 - p for p in probabilities]

    markers = [
        (swing.end_t, "^")
        for swing in sorted(swings, key=lambda s: s.magnitude, reverse=True)[:8]
    ]
    chart = winprob_chart(
        times,
        probabilities,
        width=max(28, width - 6),
        height=9,
        markers=markers,
        blue_style=US_STYLE,
        red_style=THEM_STYLE,
    )
    return Panel(chart, title="trace", title_align="left", border_style="dim", box=ROUNDED)


def scoreboard_panel(state: GameState, perspective: int = BLUE) -> Panel:
    """Team totals plus a per-player line for both sides."""
    us = state.team(perspective)
    them = state.team(other_team(perspective))

    totals = Table(box=SIMPLE, expand=True, pad_edge=False, show_edge=False)
    totals.add_column("", width=9, style="dim")
    totals.add_column("you", justify="right", width=9)
    totals.add_column("them", justify="right", width=9)
    totals.add_column("diff", justify="right", width=9)

    def row(label: str, ours: float, theirs: float, fmt: str = "{:,.0f}") -> None:
        delta = ours - theirs
        style = US_STYLE if delta > 0 else (THEM_STYLE if delta < 0 else "dim")
        totals.add_row(
            label,
            fmt.format(ours),
            fmt.format(theirs),
            Text(("+" if delta > 0 else "") + fmt.format(delta), style=style),
        )

    row("gold", us.gold, them.gold)
    row("kills", us.kills, them.kills)
    row("turrets", us.towers_raw, them.towers_raw)
    row("dragons", us.dragon_count, them.dragon_count)
    row("alive", us.alive, them.alive)
    row("items", us.item_value, them.item_value)

    badges = Text()
    for side, label, style in ((us, "you", US_STYLE), (them, "them", THEM_STYLE)):
        marks = []
        if side.has_soul:
            marks.append(f"{side.soul_type or ''} SOUL".strip())
        if side.buff_remaining("baron", state.t) > 0:
            marks.append(f"BARON {side.buff_remaining('baron', state.t) * 180:.0f}s")
        if side.buff_remaining("elder", state.t) > 0:
            marks.append(f"ELDER {side.buff_remaining('elder', state.t) * 150:.0f}s")
        if side.inhibitors_down:
            marks.append(f"{side.inhibitors_down} INHIB DOWN")
        if marks:
            badges.append(f"  {label}: ", style="dim")
            badges.append(", ".join(marks) + "\n", style=style)

    players = Table(box=SIMPLE, expand=True, pad_edge=False, show_edge=False)
    players.add_column("", width=14)
    players.add_column("lv", justify="right", width=3)
    players.add_column("k/d/a", justify="right", width=9)
    players.add_column("cs", justify="right", width=4)
    players.add_column("", width=8)

    for side, style in ((us, US_STYLE), (them, THEM_STYLE)):
        for player in sorted(side.players, key=lambda p: p.participant_id):
            status = ""
            if player.is_dead:
                status = f"dead {player.respawn_s:.0f}s"
            elif player.current_gold >= 1600:
                status = f"{player.current_gold:,.0f}g"
            players.add_row(
                Text(player.champion[:14], style=style),
                str(player.level),
                player.kda,
                str(player.cs),
                Text(status, style="yellow" if player.is_dead else "dim"),
            )

    body: List[RenderableType] = [totals]
    if badges.plain.strip():
        body.append(badges)
    body.append(players)

    return Panel(
        Group(*body), title="scoreboard", title_align="left", border_style="dim", box=ROUNDED
    )


def drivers_panel(track: Track, perspective: int = BLUE, limit: int = 7) -> Panel:
    """The factors currently pushing the number, largest first."""
    table = Table(box=SIMPLE, expand=True, pad_edge=False, show_edge=False)
    table.add_column("factor", width=20)
    table.add_column("now", justify="right", width=9)
    table.add_column("weight", justify="right", width=8)

    point = track.latest
    if point is not None:
        for primary, label, contribution, raw in standing(point, limit=limit):
            signed = contribution if perspective == BLUE else -contribution
            table.add_row(
                label,
                display_value(primary, raw),
                Text(f"{signed:+.2f}", style=US_STYLE if signed > 0 else THEM_STYLE),
            )

    return Panel(
        table, title="what is driving it", title_align="left", border_style="dim", box=ROUNDED
    )


def advice_panel(advice: Optional[Advice]) -> Panel:
    """Ranked actions and the single biggest thing to deny."""
    table = Table(box=SIMPLE, expand=True, pad_edge=False, show_edge=False)
    table.add_column("", width=2)
    table.add_column("do this", width=28)
    table.add_column("", justify="right", width=9)

    if advice is not None:
        for suggestion in advice.top_actions(limit=4):
            table.add_row(
                Text("+", style="bold green"),
                suggestion.action,
                Text(f"{suggestion.points:+.1f}", style="green"),
            )
        for suggestion in advice.top_risks(limit=2):
            table.add_row(
                Text("!", style="bold yellow"),
                suggestion.action,
                Text(f"{suggestion.points:+.1f}", style="yellow"),
            )

    body: List[RenderableType] = [table]
    if advice is not None and advice.build:
        notes = Text()
        for note in advice.build[:2]:
            notes.append("  * " + note + "\n", style="magenta")
        body.append(notes)
    if advice is not None and advice.tempo:
        body.append(Text("  " + advice.tempo, style="italic dim"))

    return Panel(
        Group(*body), title="how to move it", title_align="left", border_style="dim", box=ROUNDED
    )


def swings_panel(
    swings: Sequence[Swing], track: Track, perspective: int = BLUE, limit: int = 5
) -> Panel:
    """The most recent swings, newest first."""
    table = Table(box=SIMPLE, expand=True, pad_edge=False, show_edge=False)
    table.add_column("when", width=13)
    table.add_column("move", justify="right", width=9)
    table.add_column("why")

    recent = sorted(swings, key=lambda s: s.end_t, reverse=True)[:limit]
    for swing in recent:
        delta = swing.delta if perspective == BLUE else -swing.delta
        narrative = narrate_swing(swing, track, limit=1, perspective=perspective)
        table.add_row(
            swing.clock(),
            Text(points(delta), style=US_STYLE if delta > 0 else THEM_STYLE),
            narrative.headline.split("  (")[0],
        )

    if not recent:
        table.add_row("", "", Text("no swings yet - the game is still even", style="dim"))

    return Panel(
        table, title="swings so far", title_align="left", border_style="dim", box=ROUNDED
    )


def build_dashboard(
    track: Track,
    state: GameState,
    swings: Sequence[Swing],
    advice: Optional[Advice],
    perspective: int = BLUE,
    width: int = 110,
    subtitle: str = "",
) -> Layout:
    """Assemble the whole screen."""
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=6),
        Layout(name="middle", size=13),
        Layout(name="lower", size=11),
        Layout(name="swings", size=9),
    )
    layout["middle"].split_row(Layout(name="trace", ratio=3), Layout(name="score", ratio=2))
    layout["lower"].split_row(Layout(name="drivers", ratio=2), Layout(name="advice", ratio=3))

    layout["header"].update(odds_header(track, state, perspective, subtitle))
    layout["trace"].update(trace_panel(track, swings, width=int(width * 0.58), perspective=perspective))
    layout["score"].update(scoreboard_panel(state, perspective))
    layout["drivers"].update(drivers_panel(track, perspective))
    layout["advice"].update(advice_panel(advice))
    layout["swings"].update(swings_panel(swings, track, perspective))
    return layout


def waiting_panel(message: str, detail: str = "") -> Panel:
    body = Text(message, style="bold")
    if detail:
        body.append("\n\n" + detail, style="dim")
    return Panel(body, title="rift_oracle", border_style="dim", box=ROUNDED)
