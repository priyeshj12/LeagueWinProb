"""Terminal plotting.

A win-probability trace is a line, and a line drawn in half-block characters
looks like a bar chart of a line. Braille gives four times the vertical
resolution and two times the horizontal for the same cell count, which is the
difference between seeing that the game swung and seeing where.

Each braille cell encodes a 2x4 dot grid, so a 60x12 chart is really a 120x48
canvas. Cells are coloured by which side the trace favours at that column, so
the shape of the game is legible before you read a single number.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from rich.text import Text

from rift_oracle.game.state import format_clock

#: Braille dot bit for each (column, row) position inside a cell.
_DOTS = (
    (0x01, 0x02, 0x04, 0x40),  # left column, rows top to bottom
    (0x08, 0x10, 0x20, 0x80),  # right column
)
_BRAILLE_BASE = 0x2800

_BLOCKS = " ▁▂▃▄▅▆▇█"


class BrailleCanvas:
    """A dot canvas that renders to braille characters."""

    def __init__(self, width: int, height: int) -> None:
        """``width`` and ``height`` are in *cells*; the dot grid is 2x4 bigger."""
        self.width = max(1, int(width))
        self.height = max(1, int(height))
        self.dot_width = self.width * 2
        self.dot_height = self.height * 4
        self._cells = [[0 for _ in range(self.width)] for _ in range(self.height)]

    def set(self, x: int, y: int) -> None:
        """Light the dot at dot-coordinates ``(x, y)``, origin top-left."""
        if not (0 <= x < self.dot_width and 0 <= y < self.dot_height):
            return
        self._cells[y // 4][x // 2] |= _DOTS[x % 2][y % 4]

    def line(self, x0: int, y0: int, x1: int, y1: int) -> None:
        """Bresenham line between two dot coordinates."""
        dx, dy = abs(x1 - x0), -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        error = dx + dy

        while True:
            self.set(x0, y0)
            if x0 == x1 and y0 == y1:
                return
            doubled = 2 * error
            if doubled >= dy:
                error += dy
                x0 += sx
            if doubled <= dx:
                error += dx
                y0 += sy

    def horizontal(self, y: int, char_every: int = 1) -> None:
        for x in range(0, self.dot_width, char_every):
            self.set(x, y)

    def rows(self) -> List[List[str]]:
        """Render to a grid of characters, one string per cell."""
        return [
            [chr(_BRAILLE_BASE + cell) if cell else " " for cell in row]
            for row in self._cells
        ]


def sparkline(values: Sequence[float], low: float = 0.0, high: float = 1.0) -> str:
    """A one-line block-character trace, for compact status rows."""
    if not values:
        return ""
    span = max(high - low, 1e-9)
    out = []
    for value in values:
        fraction = min(1.0, max(0.0, (float(value) - low) / span))
        out.append(_BLOCKS[min(len(_BLOCKS) - 1, int(fraction * (len(_BLOCKS) - 1) + 0.5))])
    return "".join(out)


def winprob_chart(
    times: Sequence[float],
    probabilities: Sequence[float],
    width: int = 72,
    height: int = 12,
    blue_label: str = "Blue",
    red_label: str = "Red",
    markers: Optional[Sequence[Tuple[float, str]]] = None,
    blue_style: str = "bold cyan",
    red_style: str = "bold red",
    axis_style: str = "dim",
) -> Text:
    """Render a win-probability trace with axes, a 50% line, and markers.

    ``markers`` are ``(time, character)`` pairs drawn on the axis, which is how
    swings and objectives get pinned to the timeline underneath the curve.
    """
    if len(times) < 2 or len(probabilities) < 2:
        return Text("(not enough data to plot yet)", style=axis_style)

    gutter = 6
    plot_width = max(10, width - gutter - 1)
    canvas = BrailleCanvas(plot_width, height)

    t_min, t_max = float(min(times)), float(max(times))
    t_span = max(t_max - t_min, 1e-6)

    def to_x(t: float) -> int:
        return int((float(t) - t_min) / t_span * (canvas.dot_width - 1))

    def to_y(p: float) -> int:
        clamped = min(1.0, max(0.0, float(p)))
        return int((1.0 - clamped) * (canvas.dot_height - 1))

    # The 50% reference line, dashed so it never reads as data.
    canvas.horizontal(to_y(0.5), char_every=4)

    points = [(to_x(t), to_y(p)) for t, p in zip(times, probabilities)]
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        canvas.line(x0, y0, x1, y1)

    # Colour each cell by the side the trace favours in that column.
    column_favours_blue: List[Optional[bool]] = [None] * canvas.width
    for (x, _y), probability in zip(points, probabilities):
        column_favours_blue[min(x // 2, canvas.width - 1)] = probability >= 0.5
    last = True
    for i, value in enumerate(column_favours_blue):
        if value is None:
            column_favours_blue[i] = last
        else:
            last = bool(value)

    rows = canvas.rows()
    text = Text()

    for row_index, row in enumerate(rows):
        fraction = 1.0 - (row_index / max(len(rows) - 1, 1))
        label = ""
        if row_index == 0:
            label = "100%"
        elif row_index == len(rows) - 1:
            label = "  0%"
        elif abs(fraction - 0.5) < (0.5 / max(len(rows) - 1, 1)):
            label = " 50%"
        text.append(f"{label:>{gutter - 1}} ", style=axis_style)
        text.append("|", style=axis_style)

        for col_index, char in enumerate(row):
            if char == " ":
                text.append(" ")
            else:
                style = blue_style if column_favours_blue[col_index] else red_style
                text.append(char, style=style)
        text.append("\n")

    # Axis with time ticks.
    text.append(" " * (gutter - 1) + "+", style=axis_style)
    axis = ["-"] * canvas.width
    tick_positions: List[Tuple[int, str]] = []
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        t = t_min + t_span * fraction
        column = min(int(fraction * (canvas.width - 1)), canvas.width - 1)
        axis[column] = "+"
        tick_positions.append((column, format_clock(t)))

    for t, char in markers or []:
        column = min(max(int((float(t) - t_min) / t_span * (canvas.width - 1)), 0), canvas.width - 1)
        axis[column] = char

    text.append("".join(axis) + "\n", style=axis_style)

    # Tick labels, placed under their ticks without overlapping.
    label_row = [" "] * (canvas.width + 8)
    for column, label in tick_positions:
        start = max(0, min(column - len(label) // 2, len(label_row) - len(label)))
        if all(ch == " " for ch in label_row[max(0, start - 1) : start + len(label) + 1]):
            label_row[start : start + len(label)] = list(label)
    text.append(" " * gutter + "".join(label_row).rstrip(), style=axis_style)

    return text


def bar(value: float, width: int = 20, low: float = 0.0, high: float = 1.0, fill: str = "=") -> str:
    """A plain ASCII progress bar, for values that need no colour."""
    span = max(high - low, 1e-9)
    filled = int(round(min(1.0, max(0.0, (value - low) / span)) * width))
    return fill * filled + "." * (width - filled)


def probability_bar(p: float, width: int = 40) -> Text:
    """A two-sided bar: blue's share on the left, red's on the right."""
    p = min(1.0, max(0.0, float(p)))
    blue_cells = int(round(p * width))
    text = Text()
    text.append("#" * blue_cells, style="bold cyan")
    text.append("#" * (width - blue_cells), style="bold red")
    return text
