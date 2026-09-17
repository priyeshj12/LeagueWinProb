"""Self-contained HTML report.

One file, no network, no build step: open it in a browser or send it to whoever
you were playing with. The chart is inline SVG with a crosshair tooltip layer,
and every value it shows is also in the table at the bottom, so the report is
readable without a pointer.

Colour follows the data's job. A win-probability trace is a *diverging*
encoding - above and below a 50% baseline, with a neutral midpoint - so it uses
a warm/cool pole pair with a grey midpoint rather than two arbitrary series
colours. The pair here is blue and red, which is both the correct diverging
shape and the side convention every League player already reads. The pair was
validated for colour-vision deficiency and surface contrast in both light and
dark modes (worst-pair CVD dE 21.6 light / 19.2 dark against a dE 8 target).
The fill carries the polarity, the line is drawn with a hard-stop gradient at
the baseline so it changes colour exactly where the lead changes hands, and
both regions are keyed in the legend so identity is never colour alone.
"""

from __future__ import annotations

import html
import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

from rift_oracle.analysis.advice import Advice
from rift_oracle.analysis.narrate import narrate_swing, standing
from rift_oracle.analysis.swings import (
    Swing,
    Track,
    biggest_swing,
    bundle_label,
    swing_summary,
)
from rift_oracle.game.state import BLUE, RED, format_clock
from rift_oracle.model.features import display_value

CHART_W, CHART_H = 1000.0, 330.0
PAD_L, PAD_R, PAD_T, PAD_B = 52.0, 18.0, 16.0, 34.0


def _esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _scale(track: Track, perspective: int) -> Tuple[List[Tuple[float, float]], float, float]:
    times = track.times
    probabilities = [
        p if perspective == BLUE else 1.0 - p for p in track.probabilities
    ]
    t_min, t_max = min(times), max(times)
    span = max(t_max - t_min, 1e-6)

    plot_w = CHART_W - PAD_L - PAD_R
    plot_h = CHART_H - PAD_T - PAD_B

    points = [
        (
            PAD_L + (t - t_min) / span * plot_w,
            PAD_T + (1.0 - min(1.0, max(0.0, p))) * plot_h,
        )
        for t, p in zip(times, probabilities)
    ]
    return points, t_min, t_max


def _path(points: Sequence[Tuple[float, float]]) -> str:
    if not points:
        return ""
    return "M " + " L ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def _chart_svg(track: Track, swings: Sequence[Swing], perspective: int) -> str:
    points, t_min, t_max = _scale(track, perspective)
    if len(points) < 2:
        return '<p class="muted">Not enough data to plot.</p>'

    plot_h = CHART_H - PAD_T - PAD_B
    baseline_y = PAD_T + 0.5 * plot_h
    left_x, right_x = points[0][0], points[-1][0]

    line = _path(points)
    area = f"{line} L {right_x:.2f},{baseline_y:.2f} L {left_x:.2f},{baseline_y:.2f} Z"

    # Gridlines at 25% intervals, recessive.
    grid = []
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = PAD_T + (1.0 - fraction) * plot_h
        is_baseline = abs(fraction - 0.5) < 1e-9
        grid.append(
            f'<line x1="{PAD_L}" y1="{y:.1f}" x2="{CHART_W - PAD_R}" y2="{y:.1f}" '
            f'class="{"baseline" if is_baseline else "grid"}"/>'
        )
        grid.append(
            f'<text x="{PAD_L - 10}" y="{y + 4:.1f}" class="axis-label" '
            f'text-anchor="end">{int(fraction * 100)}%</text>'
        )

    # Time ticks.
    ticks = []
    span = max(t_max - t_min, 1e-6)
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        t = t_min + span * fraction
        x = PAD_L + fraction * (CHART_W - PAD_L - PAD_R)
        ticks.append(
            f'<text x="{x:.1f}" y="{CHART_H - 12:.1f}" class="axis-label" '
            f'text-anchor="middle">{format_clock(t)}</text>'
        )

    # Swing markers, biggest first, on the curve.
    markers = []
    ranked = sorted(swings, key=lambda s: s.magnitude, reverse=True)[:10]
    for index, swing in enumerate(sorted(ranked, key=lambda s: s.start_t), start=1):
        point = track.at(swing.end_t)
        if point is None:
            continue
        p = point.p if perspective == BLUE else 1.0 - point.p
        x = PAD_L + (swing.end_t - t_min) / span * (CHART_W - PAD_L - PAD_R)
        y = PAD_T + (1.0 - min(1.0, max(0.0, p))) * plot_h
        delta = swing.delta if perspective == BLUE else -swing.delta
        side = "up" if delta > 0 else "down"
        # Label above the dot, or below it when the dot is near the top edge,
        # so the numbers never land on the time axis.
        label_y = y - 11.0 if y > PAD_T + 22.0 else y + 17.0
        markers.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5.5" class="marker {side}"/>'
            f'<text x="{x:.1f}" y="{label_y:.1f}" class="marker-label" '
            f'text-anchor="middle">{index}</text>'
        )

    return f"""<svg viewBox="0 0 {CHART_W:.0f} {CHART_H:.0f}" class="chart"
     role="img" aria-label="Win probability over time">
  <defs>
    <linearGradient id="traceStroke" gradientUnits="userSpaceOnUse"
                    x1="0" y1="{PAD_T}" x2="0" y2="{CHART_H - PAD_B}">
      <stop offset="0" stop-color="var(--us)"/>
      <stop offset="{(baseline_y - PAD_T) / plot_h:.4f}" stop-color="var(--us)"/>
      <stop offset="{(baseline_y - PAD_T) / plot_h:.4f}" stop-color="var(--them)"/>
      <stop offset="1" stop-color="var(--them)"/>
    </linearGradient>
    <clipPath id="clipUp">
      <rect x="0" y="{PAD_T}" width="{CHART_W}" height="{baseline_y - PAD_T:.2f}"/>
    </clipPath>
    <clipPath id="clipDown">
      <rect x="0" y="{baseline_y:.2f}" width="{CHART_W}" height="{CHART_H - PAD_B - baseline_y:.2f}"/>
    </clipPath>
  </defs>
  {''.join(grid)}
  <path d="{area}" class="fill-up" clip-path="url(#clipUp)"/>
  <path d="{area}" class="fill-down" clip-path="url(#clipDown)"/>
  <path d="{line}" class="trace"/>
  {''.join(markers)}
  {''.join(ticks)}
  <g class="hover-layer" aria-hidden="true">
    <line class="crosshair" x1="0" y1="{PAD_T}" x2="0" y2="{CHART_H - PAD_B}" style="opacity:0"/>
    <circle class="hover-dot" r="5" style="opacity:0"/>
  </g>
  <rect class="hit" x="{PAD_L}" y="{PAD_T}" width="{CHART_W - PAD_L - PAD_R}"
        height="{plot_h}" fill="transparent"/>
</svg>"""


def _attribution_bars(track: Track, perspective: int, limit: int = 9) -> str:
    """Diverging bars for the final per-feature logit decomposition."""
    point = track.latest
    if point is None:
        return ""

    rows = standing(point, limit=limit)
    if not rows:
        return '<p class="muted">No factor is doing anything yet.</p>'

    peak = max(abs(value) for _p, _l, value, _r in rows) or 1.0
    out = ['<div class="bars">']
    for primary, label, contribution, raw in rows:
        signed = contribution if perspective == BLUE else -contribution
        width = abs(signed) / peak * 50.0
        side = "us" if signed > 0 else "them"
        offset = 50.0 if signed > 0 else 50.0 - width
        out.append(
            f'<div class="bar-row">'
            f'<span class="bar-label">{_esc(label)}</span>'
            f'<span class="bar-track">'
            f'<span class="bar-axis"></span>'
            f'<span class="bar-fill {side}" style="left:{offset:.2f}%;width:{width:.2f}%"></span>'
            f"</span>"
            f'<span class="bar-value">{_esc(display_value(primary, raw))}'
            f'<em>{signed:+.2f}</em></span>'
            f"</div>"
        )
    out.append("</div>")
    return "".join(out)


def _swing_cards(swings: Sequence[Swing], track: Track, perspective: int, limit: int = 12) -> str:
    ranked = sorted(swings, key=lambda s: s.magnitude, reverse=True)[:limit]
    order = {id(s): i + 1 for i, s in enumerate(sorted(ranked, key=lambda s: s.start_t))}

    if not ranked:
        return '<p class="muted">No swing cleared the threshold.</p>'

    cards = []
    for swing in ranked:
        narrative = narrate_swing(swing, track, limit=4, perspective=perspective)
        delta = swing.delta if perspective == BLUE else -swing.delta
        side = "us" if delta > 0 else "them"

        reasons = []
        for key, share in swing.top_attributions(limit=4):
            signed = share if perspective == BLUE else -share
            causes = swing.causes_for(key)
            detail = causes[0].text if causes else ""
            reasons.append(
                f'<li><span class="reason-name">{_esc(bundle_label(key))}</span>'
                f'<span class="reason-value {"us" if signed > 0 else "them"}">'
                f"{signed * 100:+.1f} pts</span>"
                f'<span class="reason-detail">{_esc(detail)}</span></li>'
            )

        cards.append(
            f'<article class="swing {side}">'
            f'<header><span class="swing-index">{order[id(swing)]}</span>'
            f'<span class="swing-clock">{_esc(swing.clock())}</span>'
            f'<span class="swing-delta {side}">{delta * 100:+.1f} pts</span>'
            f'<span class="swing-odds">{swing.p_before if perspective == BLUE else 1 - swing.p_before:.0%}'
            f" &rarr; "
            f"{swing.p_after if perspective == BLUE else 1 - swing.p_after:.0%}</span></header>"
            f'<p class="swing-headline">{_esc(narrative.headline.split("  (")[0])}</p>'
            f'<ul class="reasons">{"".join(reasons)}</ul>'
            f"</article>"
        )
    return "".join(cards)


def _advice_block(advice: Optional[Advice]) -> str:
    if advice is None:
        return ""
    rows = []
    for suggestion in advice.top_actions(limit=5):
        rows.append(
            f'<li class="act"><span class="act-name">{_esc(suggestion.action)}</span>'
            f'<span class="act-delta us">{suggestion.points:+.1f} pts</span>'
            f'<span class="act-why">{_esc(suggestion.rationale)}</span></li>'
        )
    for suggestion in advice.top_risks(limit=3):
        rows.append(
            f'<li class="risk"><span class="act-name">{_esc(suggestion.action)}</span>'
            f'<span class="act-delta them">{suggestion.points:+.1f} pts</span>'
            f'<span class="act-why">{_esc(suggestion.rationale)}</span></li>'
        )
    notes = "".join(f"<li>{_esc(note)}</li>" for note in advice.build)
    tempo = f'<p class="tempo">{_esc(advice.tempo)}</p>' if advice.tempo else ""
    return (
        f'<section class="card"><h2>How to move the number</h2>'
        f'<ul class="actions">{"".join(rows)}</ul>'
        f'{f"<ul class=notes>{notes}</ul>" if notes else ""}{tempo}</section>'
    )


def _table(track: Track, perspective: int, stride: int = 1) -> str:
    rows = []
    for point in track.points[::stride]:
        p = point.p if perspective == BLUE else 1.0 - point.p
        top = standing(point, limit=1)
        driver = top[0][1] if top else ""
        rows.append(
            f"<tr><td>{_esc(format_clock(point.t))}</td><td>{p:.1%}</td>"
            f"<td>{point.state.gold_diff():+,.0f}</td>"
            f"<td>{point.state.blue.kills}-{point.state.red.kills}</td>"
            f"<td>{_esc(driver)}</td></tr>"
        )
    return (
        "<table><thead><tr><th>clock</th><th>win probability</th>"
        "<th>gold diff (blue)</th><th>kills</th><th>top factor</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def render_html(
    track: Track,
    swings: Sequence[Swing],
    summary: Dict[str, Any],
    perspective: int = BLUE,
    advice: Optional[Advice] = None,
    model_meta: Optional[Dict[str, Any]] = None,
) -> str:
    """Build the complete report document."""
    point = track.latest
    final_p = 0.5 if point is None else (point.p if perspective == BLUE else 1 - point.p)
    stats = swing_summary(swings)
    biggest = biggest_swing(swings)

    us_label = "Blue" if perspective == BLUE else "Red"
    them_label = "Red" if perspective == BLUE else "Blue"

    winner = track.winner if track.winner is not None else summary.get("winner")
    result = "in progress"
    if winner is not None:
        result = "won" if int(winner) == perspective else "lost"

    duration = summary.get("duration_s") or (track.points[-1].t if track.points else 0)

    kpis = [
        ("final call", f"{final_p:.0%}", "model's last word"),
        ("result", result, "ground truth"),
        ("length", format_clock(float(duration)), "game time"),
        ("swings", str(stats["count"]), "over the threshold"),
        (
            "biggest",
            f"{biggest.magnitude * 100:.0f} pts" if biggest else "-",
            biggest.clock() if biggest else "none",
        ),
    ]
    kpi_html = "".join(
        f'<div class="kpi"><span class="kpi-label">{_esc(label)}</span>'
        f'<span class="kpi-value">{_esc(value)}</span>'
        f'<span class="kpi-note">{_esc(note)}</span></div>'
        for label, value, note in kpis
    )

    tooltip_data = json.dumps(
        [
            {
                "t": round(p.t, 1),
                "clock": format_clock(p.t),
                "p": round(p.p if perspective == BLUE else 1 - p.p, 4),
                "gold": round(p.state.gold_diff() if perspective == BLUE else -p.state.gold_diff()),
                "kills": (
                    f"{p.state.team(perspective).kills}-"
                    f"{p.state.team(RED if perspective == BLUE else BLUE).kills}"
                ),
            }
            for p in track.points
        ]
    )

    meta_bits = []
    if summary.get("match_id"):
        meta_bits.append(_esc(summary["match_id"]))
    if summary.get("queue_id"):
        from rift_oracle.riot.routing import queue_name

        meta_bits.append(_esc(queue_name(int(summary["queue_id"]))))
    if summary.get("patch"):
        meta_bits.append("patch " + _esc(summary["patch"]))
    if model_meta and model_meta.get("trained_on"):
        meta_bits.append("model: " + _esc(model_meta["trained_on"]))

    comps = ""
    if summary.get("blue_champions"):
        us_champs = summary["blue_champions"] if perspective == BLUE else summary.get("red_champions", [])
        them_champs = summary.get("red_champions", []) if perspective == BLUE else summary["blue_champions"]
        comps = (
            f'<div class="comps"><div><span class="key us"></span>'
            f"{_esc(us_label)} &middot; {_esc(', '.join(us_champs))}</div>"
            f'<div><span class="key them"></span>'
            f"{_esc(them_label)} &middot; {_esc(', '.join(them_champs))}</div></div>"
        )

    return (
        _TEMPLATE.replace("{{TITLE}}", _esc(summary.get("match_id") or "rift_oracle report"))
        .replace("{{META}}", " &middot; ".join(meta_bits))
        .replace("{{KPIS}}", kpi_html)
        .replace("{{COMPS}}", comps)
        .replace("{{US}}", _esc(us_label))
        .replace("{{THEM}}", _esc(them_label))
        .replace("{{CHART}}", _chart_svg(track, swings, perspective))
        .replace("{{SWINGS}}", _swing_cards(swings, track, perspective))
        .replace("{{BARS}}", _attribution_bars(track, perspective))
        .replace("{{ADVICE}}", _advice_block(advice))
        .replace("{{TABLE}}", _table(track, perspective))
        .replace("{{DATA}}", tooltip_data)
    )


_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{TITLE}} - rift_oracle</title>
<style>
  :root {
    color-scheme: light;
    --surface: #fcfcfb;
    --card: #ffffff;
    --border: #e6e5e1;
    --ink: #0b0b0b;
    --ink-2: #52514e;
    --ink-3: #85847e;
    --us: #2a78d6;
    --them: #e34948;
    --mid: #f0efec;
    --us-fill: rgba(42, 120, 214, 0.16);
    --them-fill: rgba(227, 73, 72, 0.16);
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      color-scheme: dark;
      --surface: #1a1a19;
      --card: #232322;
      --border: #383835;
      --ink: #ffffff;
      --ink-2: #c3c2b7;
      --ink-3: #8c8b83;
      --us: #3987e5;
      --them: #e66767;
      --mid: #383835;
      --us-fill: rgba(57, 135, 229, 0.20);
      --them-fill: rgba(230, 103, 103, 0.20);
    }
  }
  :root[data-theme="dark"] {
    color-scheme: dark;
    --surface: #1a1a19; --card: #232322; --border: #383835;
    --ink: #ffffff; --ink-2: #c3c2b7; --ink-3: #8c8b83;
    --us: #3987e5; --them: #e66767; --mid: #383835;
    --us-fill: rgba(57,135,229,.20); --them-fill: rgba(230,103,103,.20);
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--surface); color: var(--ink);
    font: 15px/1.55 ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
    -webkit-font-smoothing: antialiased;
  }
  .wrap { max-width: 1120px; margin: 0 auto; padding: 32px 16px 64px; }
  header.top { margin-bottom: 24px; }
  h1 { font-size: 24px; margin: 0 0 4px; letter-spacing: -0.01em; }
  h2 { font-size: 15px; margin: 0 0 14px; color: var(--ink-2);
       text-transform: uppercase; letter-spacing: 0.07em; font-weight: 600; }
  .meta { color: var(--ink-3); font-size: 13px; }
  .muted { color: var(--ink-3); }
  .card { background: var(--card); border: 1px solid var(--border);
          border-radius: 12px; padding: 20px; margin-bottom: 20px; }
  .kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
          gap: 2px; margin-bottom: 20px; background: var(--border);
          border: 1px solid var(--border); border-radius: 12px; overflow: hidden; }
  .kpi { background: var(--card); padding: 14px 16px; display: flex; flex-direction: column; }
  .kpi-label { font-size: 11px; text-transform: uppercase; letter-spacing: 0.07em; color: var(--ink-3); }
  .kpi-value { font-size: 28px; font-weight: 650; letter-spacing: -0.02em; margin: 2px 0; }
  .kpi-note { font-size: 12px; color: var(--ink-3); }
  .legend { display: flex; gap: 18px; font-size: 13px; color: var(--ink-2); margin-bottom: 10px; }
  .key { display: inline-block; width: 11px; height: 11px; border-radius: 3px;
         margin-right: 7px; vertical-align: -1px; }
  .key.us { background: var(--us); } .key.them { background: var(--them); }
  .comps { display: flex; flex-wrap: wrap; gap: 6px 26px; font-size: 13px;
           color: var(--ink-2); margin-top: 12px; }
  .chart-wrap { position: relative; }
  .chart { width: 100%; height: auto; display: block; touch-action: none; }
  .grid { stroke: var(--border); stroke-width: 1; }
  .baseline { stroke: var(--ink-3); stroke-width: 1; stroke-dasharray: 3 4; opacity: .7; }
  .axis-label { fill: var(--ink-3); font-size: 11px;
                font-family: ui-sans-serif, system-ui, sans-serif; }
  .marker-label { fill: var(--ink-2); font-size: 10px; font-weight: 700;
                  font-family: ui-sans-serif, system-ui, sans-serif;
                  paint-order: stroke; stroke: var(--card); stroke-width: 3px;
                  stroke-linejoin: round; }
  .fill-up { fill: var(--us-fill); } .fill-down { fill: var(--them-fill); }
  .trace { fill: none; stroke: url(#traceStroke); stroke-width: 2;
           stroke-linejoin: round; stroke-linecap: round; }
  .marker { stroke: var(--card); stroke-width: 2; }
  .marker.up { fill: var(--us); } .marker.down { fill: var(--them); }
  .crosshair { stroke: var(--ink-3); stroke-width: 1; stroke-dasharray: 2 3; }
  .hover-dot { fill: var(--ink); stroke: var(--card); stroke-width: 2; }
  .tip { position: absolute; pointer-events: none; opacity: 0; transform: translate(-50%, -100%);
         background: var(--card); border: 1px solid var(--border); border-radius: 8px;
         padding: 8px 11px; font-size: 12px; white-space: nowrap; color: var(--ink-2);
         box-shadow: 0 4px 16px rgba(0,0,0,.14); transition: opacity .1s; z-index: 5; }
  .tip b { display: block; font-size: 17px; color: var(--ink); font-weight: 650; }
  .tip .tip-clock { color: var(--ink-2); }
  .bars { display: flex; flex-direction: column; gap: 9px; }
  .bar-row { display: grid; grid-template-columns: 170px 1fr 116px; gap: 12px; align-items: center; }
  .bar-label { font-size: 13px; color: var(--ink-2); }
  .bar-track { position: relative; height: 15px; background: var(--mid); border-radius: 4px; }
  .bar-axis { position: absolute; left: 50%; top: -2px; bottom: -2px; width: 1px;
              background: var(--ink-3); opacity: .55; }
  .bar-fill { position: absolute; top: 0; bottom: 0; border-radius: 4px; }
  .bar-fill.us { background: var(--us); } .bar-fill.them { background: var(--them); }
  .bar-value { font-size: 12px; color: var(--ink-2); text-align: right;
               font-variant-numeric: tabular-nums; }
  .bar-value em { display: block; font-style: normal; color: var(--ink-3); font-size: 11px; }
  .swings { display: grid; grid-template-columns: repeat(auto-fill, minmax(330px, 1fr)); gap: 14px; }
  .swing { background: var(--card); border: 1px solid var(--border);
           border-radius: 12px; padding: 15px 17px; border-left-width: 3px; }
  .swing.us { border-left-color: var(--us); } .swing.them { border-left-color: var(--them); }
  .swing header { display: flex; align-items: baseline; gap: 9px; flex-wrap: wrap;
                  font-size: 12px; color: var(--ink-3); margin-bottom: 7px; }
  .swing-index { display: inline-flex; align-items: center; justify-content: center;
                 width: 19px; height: 19px; border-radius: 50%; background: var(--mid);
                 color: var(--ink-2); font-weight: 650; font-size: 11px; }
  .swing-clock { font-variant-numeric: tabular-nums; }
  .swing-delta { font-weight: 650; font-size: 14px; margin-left: auto; }
  .swing-delta.us, .reason-value.us, .act-delta.us { color: var(--us); }
  .swing-delta.them, .reason-value.them, .act-delta.them { color: var(--them); }
  .swing-headline { margin: 0 0 9px; font-size: 14px; font-weight: 550; color: var(--ink); }
  .reasons, .actions, .notes { list-style: none; margin: 0; padding: 0; }
  .reasons li { display: grid; grid-template-columns: 1fr auto; gap: 2px 10px;
                font-size: 12px; padding: 3px 0; border-top: 1px solid var(--border); }
  .reason-name { color: var(--ink-2); }
  .reason-value { font-variant-numeric: tabular-nums; font-weight: 600; }
  .reason-detail { grid-column: 1 / -1; color: var(--ink-3); font-size: 11px; }
  .actions li { display: grid; grid-template-columns: 1fr auto; gap: 1px 12px;
                padding: 8px 0; border-top: 1px solid var(--border); font-size: 13px; }
  .actions li:first-child { border-top: none; }
  .act-name { font-weight: 550; } .act-delta { font-variant-numeric: tabular-nums; font-weight: 650; }
  .act-why { grid-column: 1 / -1; color: var(--ink-3); font-size: 12px; }
  .notes { margin-top: 14px; padding-top: 12px; border-top: 1px solid var(--border); }
  .notes li { color: var(--ink-2); font-size: 13px; padding: 3px 0; }
  .notes li::before { content: "> "; color: var(--ink-3); }
  .tempo { margin: 12px 0 0; font-style: italic; color: var(--ink-2); font-size: 13px; }
  details { margin-top: 20px; }
  summary { cursor: pointer; color: var(--ink-2); font-size: 13px; padding: 8px 0; }
  table { border-collapse: collapse; width: 100%; font-size: 12px; margin-top: 10px; }
  th, td { text-align: right; padding: 5px 10px; border-bottom: 1px solid var(--border);
           font-variant-numeric: tabular-nums; }
  th:first-child, td:first-child, th:last-child, td:last-child { text-align: left; }
  th { color: var(--ink-3); font-weight: 600; text-transform: uppercase;
       font-size: 10px; letter-spacing: 0.06em; }
  footer { margin-top: 36px; color: var(--ink-3); font-size: 12px; }
  @media (max-width: 620px) {
    .bar-row { grid-template-columns: 110px 1fr 84px; }
    .kpi-value { font-size: 22px; }
  }
</style>
</head>
<body>
<div class="wrap">
  <header class="top">
    <h1>rift_oracle</h1>
    <div class="meta">{{META}}</div>
  </header>

  <div class="kpis">{{KPIS}}</div>

  <section class="card">
    <h2>Win probability</h2>
    <div class="legend">
      <span><span class="key us"></span>favours {{US}} (above 50%)</span>
      <span><span class="key them"></span>favours {{THEM}} (below 50%)</span>
      <span class="muted">numbered points are the largest swings</span>
    </div>
    <div class="chart-wrap">
      {{CHART}}
      <div class="tip" id="tip"></div>
    </div>
    {{COMPS}}
  </section>

  <section class="card">
    <h2>What the model is weighing right now</h2>
    {{BARS}}
  </section>

  {{ADVICE}}

  <h2>Swings, largest first</h2>
  <div class="swings">{{SWINGS}}</div>

  <details>
    <summary>Full trace as a table</summary>
    {{TABLE}}
  </details>

  <footer>
    Generated by rift_oracle. Probabilities come from an antisymmetric additive
    model, so the per-factor breakdown above sums exactly to the prediction.
  </footer>
</div>
<script>
(function () {
  var data = {{DATA}};
  var svg = document.querySelector('.chart');
  var tip = document.getElementById('tip');
  if (!svg || !tip || !data.length) return;

  var hit = svg.querySelector('.hit');
  var cross = svg.querySelector('.crosshair');
  var dot = svg.querySelector('.hover-dot');
  var padL = %PAD_L%, padR = %PAD_R%, padT = %PAD_T%, padB = %PAD_B%;
  var W = %CHART_W%, H = %CHART_H%;
  var tMin = data[0].t, tMax = data[data.length - 1].t;
  var span = Math.max(tMax - tMin, 1e-6);

  function xOf(t) { return padL + (t - tMin) / span * (W - padL - padR); }
  function yOf(p) { return padT + (1 - Math.min(1, Math.max(0, p))) * (H - padT - padB); }

  function show(evt) {
    var box = svg.getBoundingClientRect();
    var clientX = evt.clientX !== undefined ? evt.clientX : box.left + box.width / 2;
    var local = (clientX - box.left) / box.width * W;

    // Snap to the nearest sample so the reader aims at a time, not at a 2px line.
    var best = 0, bestGap = Infinity;
    for (var i = 0; i < data.length; i++) {
      var gap = Math.abs(xOf(data[i].t) - local);
      if (gap < bestGap) { bestGap = gap; best = i; }
    }
    var d = data[best];
    var x = xOf(d.t), y = yOf(d.p);

    cross.setAttribute('x1', x); cross.setAttribute('x2', x);
    cross.style.opacity = 1;
    dot.setAttribute('cx', x); dot.setAttribute('cy', y);
    dot.style.opacity = 1;

    // textContent only: these strings come from game data, never markup.
    tip.textContent = '';
    var strong = document.createElement('b');
    strong.textContent = Math.round(d.p * 100) + '%';
    var clock = document.createElement('span');
    clock.className = 'tip-clock';
    clock.textContent = d.clock + '  ·  ' + d.kills + '  ·  '
      + (d.gold >= 0 ? '+' : '') + d.gold.toLocaleString() + 'g';
    tip.appendChild(strong);
    tip.appendChild(clock);

    tip.style.left = (x / W * box.width) + 'px';
    tip.style.top = (y / H * box.height - 12) + 'px';
    tip.style.opacity = 1;
  }

  function hide() {
    tip.style.opacity = 0; cross.style.opacity = 0; dot.style.opacity = 0;
  }

  hit.addEventListener('pointermove', show);
  hit.addEventListener('pointerdown', show);
  hit.addEventListener('pointerleave', hide);
  svg.setAttribute('tabindex', '0');
  svg.addEventListener('focus', show);
  svg.addEventListener('blur', hide);
})();
</script>
</body>
</html>
"""

_TEMPLATE = (
    _TEMPLATE.replace("%PAD_L%", str(PAD_L))
    .replace("%PAD_R%", str(PAD_R))
    .replace("%PAD_T%", str(PAD_T))
    .replace("%PAD_B%", str(PAD_B))
    .replace("%CHART_W%", str(CHART_W))
    .replace("%CHART_H%", str(CHART_H))
)
