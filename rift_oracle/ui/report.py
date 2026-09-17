"""Post-game report: the curve, the swings, and why each one happened."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from rich.box import ROUNDED, SIMPLE
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from rift_oracle.analysis.advice import Advice
from rift_oracle.analysis.narrate import narrate_swing, percent, points, standing
from rift_oracle.analysis.swings import Swing, Track, biggest_swing, swing_summary
from rift_oracle.game.state import BLUE, format_clock
from rift_oracle.model.features import display_value
from rift_oracle.ui.chart import winprob_chart

BLUE_STYLE = "bold cyan"
RED_STYLE = "bold red"


#: How many swings get a number on the chart and a row in the table.
SWING_LIMIT = 10


def side_style(team: int) -> str:
    return BLUE_STYLE if team == BLUE else RED_STYLE


def side_name(team: int) -> str:
    return "Blue" if team == BLUE else "Red"


def header_panel(summary: Dict[str, Any], track: Track, perspective: int = BLUE) -> Panel:
    """Match identity, teams, and the final result."""
    table = Table.grid(padding=(0, 2))
    table.add_column(style="dim", justify="right")
    table.add_column()

    match_id = summary.get("match_id") or track.match_id or "live game"
    table.add_row("match", str(match_id))

    duration = summary.get("duration_s")
    if duration:
        table.add_row("length", format_clock(float(duration)))

    queue = summary.get("queue_id")
    if queue:
        from rift_oracle.riot.routing import queue_name

        table.add_row("queue", queue_name(int(queue)))
    if summary.get("patch"):
        table.add_row("patch", str(summary["patch"]))

    blue_champions = summary.get("blue_champions") or []
    red_champions = summary.get("red_champions") or []
    if blue_champions:
        table.add_row(Text("blue", style=BLUE_STYLE), ", ".join(blue_champions))
    if red_champions:
        table.add_row(Text("red", style=RED_STYLE), ", ".join(red_champions))

    winner = track.winner if track.winner is not None else summary.get("winner")
    if winner is not None:
        won = winner == perspective
        table.add_row(
            "result",
            Text(
                f"{side_name(int(winner))} won" + ("  (you won)" if won else "  (you lost)"),
                style=side_style(int(winner)),
            ),
        )

    return Panel(table, title="rift_oracle", border_style="dim", box=ROUNDED)


def chart_panel(
    track: Track, swings: Sequence[Swing], width: int = 92, perspective: int = BLUE
) -> Panel:
    """The win-probability curve with the biggest swings marked on the axis."""
    times = track.times
    probabilities = track.probabilities
    if perspective != BLUE:
        probabilities = [1.0 - p for p in probabilities]

    ranked = sorted(swings, key=lambda s: s.magnitude, reverse=True)[:SWING_LIMIT]
    markers = [
        (swing.end_t, str(index + 1))
        for index, swing in enumerate(sorted(ranked, key=lambda s: s.start_t))
    ]

    chart = winprob_chart(
        times,
        probabilities,
        width=width - 6,
        height=13,
        markers=markers,
        blue_style=BLUE_STYLE if perspective == BLUE else RED_STYLE,
        red_style=RED_STYLE if perspective == BLUE else BLUE_STYLE,
    )
    label = "your win probability" if perspective != BLUE else "blue-side win probability"
    return Panel(chart, title=label, border_style="dim", box=ROUNDED)


def swing_table(
    swings: Sequence[Swing], track: Track, perspective: int = BLUE, limit: int = SWING_LIMIT
) -> Table:
    """One row per swing, largest first, with its headline cause."""
    table = Table(box=SIMPLE, expand=True, pad_edge=False)
    table.add_column("#", justify="right", style="dim", width=3)
    table.add_column("when", width=13)
    table.add_column("move", justify="right", width=10)
    table.add_column("odds", justify="right", width=13)
    table.add_column("why")

    ranked = sorted(swings, key=lambda s: s.magnitude, reverse=True)[:limit]
    order = {id(swing): i + 1 for i, swing in enumerate(sorted(ranked, key=lambda s: s.start_t))}

    for swing in ranked:
        delta = swing.delta if perspective == BLUE else -swing.delta
        before = swing.p_before if perspective == BLUE else 1 - swing.p_before
        after = swing.p_after if perspective == BLUE else 1 - swing.p_after
        style = BLUE_STYLE if delta > 0 else RED_STYLE

        narrative = narrate_swing(swing, track, limit=1, perspective=perspective)
        headline = narrative.headline.split("  (")[0]

        table.add_row(
            str(order[id(swing)]),
            swing.clock(),
            Text(points(delta), style=style),
            f"{percent(before)} -> {percent(after)}",
            headline,
        )
    return table


def swing_detail(
    swings: Sequence[Swing], track: Track, perspective: int = BLUE, limit: int = 5
) -> Group:
    """Full narration for the handful of swings that mattered most."""
    ranked = sorted(swings, key=lambda s: s.magnitude, reverse=True)[:limit]
    ranked.sort(key=lambda s: s.start_t)

    blocks: List[Any] = []
    for swing in ranked:
        narrative = narrate_swing(swing, track, limit=4, perspective=perspective)
        delta = swing.delta if perspective == BLUE else -swing.delta
        style = BLUE_STYLE if delta > 0 else RED_STYLE

        body = Text()
        body.append(narrative.headline + "\n", style=style)
        for reason in narrative.reasons:
            body.append("    " + reason + "\n", style="none")
        blocks.append(
            Panel(
                body,
                title=f"{narrative.clock}  [{swing.kind}]",
                title_align="left",
                border_style="dim",
                box=ROUNDED,
            )
        )
    return Group(*blocks)


def standing_table(track: Track, perspective: int = BLUE, limit: int = 8) -> Table:
    """What the model thinks at the final evaluated moment, and why."""
    table = Table(box=SIMPLE, expand=True, pad_edge=False)
    table.add_column("factor", width=24)
    table.add_column("value", justify="right", width=10)
    table.add_column("logit", justify="right", width=9)
    table.add_column("favours", width=8)
    table.add_column("share", justify="right", width=8)

    point = track.latest
    if point is None:
        return table

    rows = standing(point, limit=limit)
    total = sum(abs(value) for _p, _l, value, _r in rows) or 1.0

    for primary, label, contribution, raw in rows:
        signed = contribution if perspective == BLUE else -contribution
        table.add_row(
            label,
            display_value(primary, raw),
            f"{signed:+.3f}",
            Text("you", style=BLUE_STYLE) if signed > 0 else Text("them", style=RED_STYLE),
            f"{abs(contribution) / total * 100:.0f}%",
        )
    return table


def advice_panel(advice: Advice, width: int = 92) -> Panel:
    """Actions, risks, build notes, and the tempo call."""
    table = Table.grid(padding=(0, 1))
    table.add_column(width=3)
    table.add_column(width=34)
    table.add_column(justify="right", width=10)
    table.add_column()

    for suggestion in advice.top_actions(limit=5):
        table.add_row(
            Text("+", style="bold green"),
            suggestion.action,
            Text(f"{suggestion.points:+.1f} pts", style="green"),
            Text(suggestion.rationale, style="dim"),
        )
    for suggestion in advice.top_risks(limit=3):
        table.add_row(
            Text("!", style="bold yellow"),
            suggestion.action,
            Text(f"{suggestion.points:+.1f} pts", style="yellow"),
            Text(suggestion.rationale, style="dim"),
        )
    for note in advice.build:
        table.add_row(Text("*", style="bold magenta"), Text(note), "", "")

    body: List[Any] = [table]
    if advice.tempo:
        body.append(Text("\n" + advice.tempo, style="italic"))

    return Panel(
        Group(*body), title="how to move the number", border_style="dim", box=ROUNDED
    )


def render_report(
    track: Track,
    swings: Sequence[Swing],
    summary: Dict[str, Any],
    console: Console,
    perspective: int = BLUE,
    advice: Optional[Advice] = None,
    model_path: Optional[str] = None,
    detail_limit: int = 5,
) -> None:
    """Print the whole post-game report."""
    width = min(console.width, 110)

    console.print(header_panel(summary, track, perspective))
    console.print(chart_panel(track, swings, width=width, perspective=perspective))

    stats = swing_summary(swings)
    if stats["count"]:
        biggest = biggest_swing(swings)
        toward = biggest.toward if perspective == BLUE else (
            "Red" if biggest.delta > 0 else "Blue"
        )
        console.print(
            Text.assemble(
                ("  ", ""),
                (f"{stats['count']} swings", "bold"),
                ("  |  biggest ", "dim"),
                (f"{biggest.magnitude * 100:.1f} pts", "bold"),
                (f" at {biggest.clock()} toward {toward}", "dim"),
                ("  |  total movement ", "dim"),
                (f"{stats['total_movement'] * 100:.0f} pts", "bold"),
            )
        )
        console.print()
        console.print(
            Panel(
                swing_table(swings, track, perspective),
                title="swings, largest first",
                border_style="dim",
                box=ROUNDED,
            )
        )
        console.print(swing_detail(swings, track, perspective, limit=detail_limit))
    else:
        console.print(
            Panel(
                Text(
                    "No swing cleared the threshold. Lower it with --threshold 0.03 "
                    "to see the smaller movements.",
                    style="dim",
                ),
                border_style="dim",
                box=ROUNDED,
            )
        )

    console.print(
        Panel(
            standing_table(track, perspective),
            title="final accounting",
            border_style="dim",
            box=ROUNDED,
        )
    )

    if advice is not None:
        console.print(advice_panel(advice, width=width))

    if model_path:
        console.print(Text(f"  model: {model_path}", style="dim"))


def render_compact(track: Track, swings: Sequence[Swing], console: Console, perspective: int = BLUE) -> None:
    """A short, pipe-friendly summary instead of the full report."""
    point = track.latest
    if point is None:
        console.print("no states evaluated")
        return
    probability = point.p if perspective == BLUE else 1 - point.p
    console.print(
        f"{format_clock(point.t)}  win probability {probability:.1%}  "
        f"({len(swings)} swings, {len(track)} states)"
    )
    for swing in sorted(swings, key=lambda s: s.magnitude, reverse=True)[:5]:
        narrative = narrate_swing(swing, track, limit=1, perspective=perspective)
        console.print(f"  {swing.clock():>13}  {narrative.headline}")
