"""End-to-end: the commands that need no API key, and the report renderers."""

import json
import re

import pytest

from rift_oracle.analysis.swings import build_track, detect_swings
from rift_oracle.cli import build_parser, main
from rift_oracle.game.state import BLUE


def test_parser_exposes_every_command():
    parser = build_parser()
    expected = {
        "live", "watch", "replay", "demo", "train",
        "harvest", "backtest", "doctor", "configure", "clear-cache",
    }
    actions = [a for a in parser._actions if a.dest == "command"]
    assert expected <= set(actions[0].choices)


def test_every_subcommand_has_a_handler():
    parser = build_parser()
    choices = [a for a in parser._actions if a.dest == "command"][0].choices
    for name, sub in choices.items():
        assert sub.get_default("func") is not None, name


def test_demo_runs_and_writes_both_outputs(tmp_path, capsys):
    html = tmp_path / "r.html"
    payload = tmp_path / "r.json"
    code = main(
        ["demo", "--seed", "5", "--compact", "--html", str(html), "--json-out", str(payload)]
    )
    assert code == 0
    assert html.is_file() and payload.is_file()

    out = capsys.readouterr().out
    assert "win probability" in out

    data = json.loads(payload.read_text())
    assert data["trace"] and data["summary"]["winner"] in (100, 200)
    for point in data["trace"]:
        assert 0.0 <= point["p_blue"] <= 1.0


def test_demo_full_report_renders(capsys):
    assert main(["demo", "--seed", "8", "--detail", "2"]) == 0
    out = capsys.readouterr().out
    assert "rift_oracle" in out
    assert "how to move the number" in out
    assert "final accounting" in out


def test_demo_from_the_red_side_inverts_the_odds(capsys):
    main(["demo", "--seed", "8", "--compact", "--side", "blue"])
    blue = _probability(capsys.readouterr().out)
    main(["demo", "--seed", "8", "--compact", "--side", "red"])
    red = _probability(capsys.readouterr().out)
    assert blue + red == pytest.approx(100.0, abs=0.2)


def _probability(text: str) -> float:
    match = re.search(r"win probability ([\d.]+)%", text)
    assert match, text
    return float(match.group(1))


def test_html_report_is_self_contained(tmp_path):
    html = tmp_path / "r.html"
    main(["demo", "--seed", "3", "--compact", "--html", str(html)])
    body = html.read_text(encoding="utf-8")

    assert body.startswith("<!DOCTYPE html>")
    # No external requests: the report has to work offline and in an email.
    assert not re.search(r'(src|href)="https?://', body)
    assert "<svg" in body and "prefers-color-scheme" in body
    # The table view keeps every value reachable without a pointer.
    assert "<table>" in body


def test_html_escapes_champion_names(tmp_path, model):
    """Names come from Riot payloads and go into markup, so they get escaped."""
    from rift_oracle.sim.synth import simulate_game
    from rift_oracle.ui.html_report import render_html

    game = simulate_game(seed=2)
    for player in game.states[-1].blue.players:
            player.champion = '<img src=x onerror=alert(1)>'
    track = build_track(game.states, model, winner=game.winner)
    body = render_html(
        track, detect_swings(track), {"blue_champions": ['<script>bad()</script>'],
                                      "red_champions": ["Ashe"]}, BLUE
    )
    assert "<script>bad()</script>" not in body
    assert "&lt;script&gt;" in body


def test_terminal_report_renders_without_swings(model):
    """A perfectly even game must not blow up the renderer."""
    from rich.console import Console

    from rift_oracle.sim.synth import simulate_game
    from rift_oracle.ui.report import render_report

    game = simulate_game(seed=21)
    track = build_track(game.states, model, winner=game.winner)
    console = Console(file=open("/dev/null", "w"), width=100)
    render_report(track, [], game.summary(), console, BLUE, advice=None)


def test_chart_handles_a_single_point():
    from rift_oracle.ui.chart import sparkline, winprob_chart

    assert "not enough data" in winprob_chart([0.0], [0.5]).plain
    assert sparkline([]) == ""
    assert len(sparkline([0.0, 0.5, 1.0])) == 3


def test_doctor_reports_without_a_key(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("RIFT_ORACLE_HOME", str(tmp_path))
    monkeypatch.delenv("RIOT_API_KEY", raising=False)
    monkeypatch.delenv("RIOT_TOKEN", raising=False)
    monkeypatch.delenv("RGAPI_KEY", raising=False)
    main(["doctor"])
    out = capsys.readouterr().out
    assert "API key present" in out
    assert "Win-probability model" in out


def test_commands_needing_a_key_fail_cleanly(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("RIFT_ORACLE_HOME", str(tmp_path))
    monkeypatch.delenv("RIOT_API_KEY", raising=False)
    monkeypatch.delenv("RIOT_TOKEN", raising=False)
    monkeypatch.delenv("RGAPI_KEY", raising=False)
    assert main(["replay", "NA1_1"]) == 2


def test_configure_round_trips(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("RIFT_ORACLE_HOME", str(tmp_path))
    assert main(["configure", "--api-key", "RGAPI-test-key", "--platform", "euw"]) == 0
    stored = json.loads((tmp_path / "config.json").read_text())
    assert stored["api_key"] == "RGAPI-test-key"
    assert stored["platform"] == "euw1"


def test_draft_projection_reflects_which_composition_scales(model, capsys):
    """Spectator only sees the draft, but the draft determines the whole curve.

    A hyper-scaling side should start behind and end ahead, crossing near the
    composition crossover; a mirror draft should stay flat.
    """
    from rich.console import Console

    from rift_oracle.cli import _draft_state, _print_draft_projection
    from rift_oracle.model.features import extract

    game = {"gameId": 1, "gameQueueConfigId": 420, "gameLength": 0}
    late = ["Kayle", "Vayne", "Veigar", "Nasus", "Kassadin"]
    early = ["Lee Sin", "Pantheon", "Renekton", "Draven", "Elise"]

    def odds_at(minute, blue, red):
        state = _draft_state(game, blue, red, minute * 60.0)
        return model.predict(extract(state)).p

    # Assert the shape, not a particular effect size: how much a draft is
    # worth is an empirical question and the answer differs between the
    # simulator and real ranked games. Fitting on real EUW data puts an extreme
    # scaling draft at about eight points of swing across a game, where the
    # simulator said seventeen. What must hold either way is that the edge
    # starts against the scaling side and ends with it.
    assert odds_at(5, late, early) < 0.50
    assert odds_at(40, late, early) > 0.50

    # For a fixed draft the projection must not go backwards. It used to:
    # scaling_diff already contains the clock, so interacting it with the clock
    # again made it quadratic in time and the curve dipped before it rose,
    # which reads as "your scaling composition got worse as the game moved
    # toward its power spike". See FeatureSpec.time_interacted.
    curve = [odds_at(m, late, early) for m in range(0, 46, 5)]
    assert all(a <= b + 1e-9 for a, b in zip(curve, curve[1:]))

    # It should cross even somewhere around the composition crossover, not at
    # the start or the end of the game.
    crossing = next(m for m, p in zip(range(0, 46, 5), curve) if p >= 0.5)
    assert 15 <= crossing <= 30

    mirror = odds_at(5, late, late), odds_at(40, late, late)
    assert abs(mirror[0] - mirror[1]) < 0.02

    console = Console(width=90, file=open("/dev/null", "w"))
    _print_draft_projection(console, model, game, late, early, 0.0, BLUE)
