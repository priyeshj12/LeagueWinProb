"""Champion power curves and team composition scaling.

A lot of League win probability is not visible in the scoreboard. A team can be
2k gold down at twenty minutes and still be favoured because its composition
gets stronger with every item, while the other side's advantage was rented from
a lane phase that is over.

Each champion carries a single ``scale`` rating in ``[-1, +1]``:

* ``-1`` the champion's power is front-loaded (Lee Sin, Draven, Pantheon),
* ``0``  the champion's relative power is roughly flat (most tanks),
* ``+1`` the champion needs items or levels before it matters (Kayle, Vayne,
  Nasus, Kassadin, Veigar).

Relative power is then ``scale * ramp(t)`` with ``ramp`` running from ``-1``
early to ``+1`` late. The team value is the mean over its five champions, and
the model sees the difference between the two teams' curves at the current
clock. That single feature is what produces lines like "their composition
overtakes yours around 28:00".
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from rift_oracle.riot.ddragon import DataDragon

#: Minute at which early-game and late-game champions are equally strong.
CROSSOVER_MIN = 22.0
#: How sharply the curve swings from early to late. Larger is more gradual.
RAMP_WIDTH_MIN = 9.0

#: Hand-rated scaling coefficients. Negative is early-game, positive is late.
#: Ratings reflect how a champion's *relative* strength moves across a game,
#: not how strong the champion is overall.
CHAMPION_SCALE: Dict[str, float] = {
    # Hyper-scaling
    "Kayle": 1.00, "Nasus": 0.95, "Veigar": 0.90, "Kassadin": 0.90,
    "Vayne": 0.85, "Jinx": 0.80, "Smolder": 0.85, "Aurelion Sol": 0.80,
    "Senna": 0.80, "Kindred": 0.70, "Twitch": 0.70, "Vladimir": 0.75,
    "Gangplank": 0.75, "Yuumi": 0.70, "Zeri": 0.65, "Sion": 0.65,
    "Cho'Gath": 0.60, "Thresh": 0.55, "Bard": 0.55, "Swain": 0.55,
    "Aphelios": 0.70, "Kog'Maw": 0.80, "Kai'Sa": 0.65, "Jayce": -0.35,
    "Ryze": 0.55, "Azir": 0.60, "Viktor": 0.65, "Karthus": 0.70,
    "Master Yi": 0.75, "Tryndamere": 0.65, "Yorick": 0.55, "Singed": 0.55,
    "Katarina": 0.50, "Akali": 0.40, "Fiora": 0.55, "Jax": 0.65,
    "Trundle": 0.45, "Mordekaiser": 0.45, "Kled": -0.30, "Ornn": 0.60,

    # Mid / flat
    "Ahri": 0.25, "Orianna": 0.40, "Syndra": 0.40, "Lux": 0.30,
    "Seraphine": 0.35, "Sona": 0.45, "Soraka": 0.35, "Janna": 0.30,
    "Lulu": 0.35, "Nami": 0.25, "Karma": -0.05, "Morgana": 0.20,
    "Zyra": 0.10, "Brand": 0.15, "Xerath": 0.30, "Ziggs": 0.35,
    "Vel'Koz": 0.30, "Malzahar": 0.30, "Anivia": 0.45, "Lissandra": 0.15,
    "Cassiopeia": 0.55, "Zoe": 0.10, "Neeko": 0.10, "Yone": 0.35,
    "Yasuo": 0.40, "Irelia": 0.35, "Camille": 0.30, "Gwen": 0.45,
    "Riven": 0.10, "Aatrox": -0.10, "Darius": -0.25, "Garen": 0.00,
    "Sett": -0.10, "Illaoi": -0.05, "Urgot": 0.05, "Volibear": 0.15,
    "Warwick": -0.05, "Udyr": 0.10, "Shyvana": 0.35, "Nocturne": 0.05,
    "Hecarim": 0.15, "Vi": -0.15, "Sejuani": 0.25, "Zac": 0.20,
    "Amumu": 0.25, "Maokai": 0.25, "Rammus": 0.20, "Malphite": 0.30,
    "Shen": 0.20, "Galio": 0.15, "Poppy": 0.05, "Gragas": 0.00,
    "Nautilus": 0.05, "Leona": -0.15, "Alistar": 0.05, "Braum": 0.10,
    "Taric": 0.30, "Rakan": 0.00, "Xayah": 0.45, "Rell": 0.05,
    "Milio": 0.30, "Renata Glasc": 0.20, "Briar": 0.10, "Naafiri": -0.20,
    "Hwei": 0.30, "K'Sante": 0.35, "Aurora": 0.30, "Ambessa": -0.10,
    "Mel": 0.30, "Nidalee": -0.25, "Rumble": 0.10, "Sylas": 0.10,
    "Taliyah": 0.05, "Qiyana": -0.25, "Akshan": 0.00, "Ekko": 0.20,
    "Fizz": 0.15, "Diana": 0.15, "Kennen": 0.30, "Teemo": 0.20,
    "Quinn": -0.35, "Corki": 0.35, "Ezreal": 0.30, "Sivir": 0.45,
    "Ashe": 0.30, "Varus": 0.30, "Jhin": 0.20, "Tristana": 0.35,
    "Miss Fortune": 0.10, "Lucian": -0.20, "Samira": 0.20, "Nilah": 0.45,
    "Kalista": -0.25, "Skarner": 0.20, "Wukong": -0.05, "Talon": -0.35,
    "Zed": -0.05, "Khazix": -0.10, "Kha'Zix": -0.10, "Rengar": -0.20,
    "Shaco": -0.20, "Evelynn": 0.15, "Graves": -0.20, "Twisted Fate": 0.25,
    "Heimerdinger": 0.20, "Annie": 0.00, "Ivern": 0.20, "Nunu & Willump": 0.10,
    "Nunu": 0.10, "Fiddlesticks": 0.30, "Jarvan IV": -0.20, "Xin Zhao": -0.25,
    "Olaf": -0.30, "Tahm Kench": 0.20, "Blitzcrank": -0.10, "Pyke": -0.20,
    "Vex": 0.15, "Gnar": 0.05, "Dr. Mundo": 0.35, "Zilean": 0.30,
    "Lillia": 0.20, "Rek'Sai": -0.35, "Elise": -0.45, "Pantheon": -0.55,
    "Renekton": -0.50, "Lee Sin": -0.50, "Draven": -0.45, "Caitlyn": -0.35,
    "Yunara": 0.55, "LeBlanc": -0.30, "Viego": 0.05, "Bel'Veth": 0.55,
    "Kayn": 0.35, "Nunu & Willump": 0.10,
}

#: Fallback by Riot role tag when a champion is not in the table above.
TAG_SCALE: Dict[str, float] = {
    "Marksman": 0.40,
    "Mage": 0.25,
    "Tank": 0.15,
    "Support": 0.15,
    "Fighter": -0.05,
    "Assassin": -0.20,
}


def ramp(t_seconds: float) -> float:
    """Early-to-late weighting in ``[-1, +1]``, crossing zero at CROSSOVER_MIN."""
    minutes = max(0.0, t_seconds / 60.0)
    return math.tanh((minutes - CROSSOVER_MIN) / RAMP_WIDTH_MIN)


def champion_scale(champion: str, ddragon: Optional[DataDragon] = None) -> float:
    """Scaling rating for a champion name, falling back to its role tags.

    The hand-rated table covers the roster as of this build. A champion
    released afterwards is not in it, so rather than silently rating a brand
    new hyper-carry as perfectly flat, the rating falls back to the mean of its
    Riot role tags from Data Dragon. That is a worse estimate than a hand
    rating and a much better one than zero.
    """
    if not champion:
        return 0.0
    name = champion.strip()
    if name in CHAMPION_SCALE:
        return CHAMPION_SCALE[name]

    normalised = _normalise(name)
    for key, value in CHAMPION_SCALE.items():
        if _normalise(key) == normalised:
            return value

    if ddragon is None:
        ddragon = _fallback_ddragon()
    if ddragon is not None:
        tags = ddragon.champion_tags(name)
        values = [TAG_SCALE[t] for t in tags if t in TAG_SCALE]
        if values:
            return sum(values) / len(values)
    return 0.0


_FALLBACK: List[Optional[DataDragon]] = [None]


def _fallback_ddragon() -> Optional[DataDragon]:
    """Shared Data Dragon for tag lookups, created on first unknown champion.

    Data Dragon caches its tables both in the instance and on disk, and returns
    an empty table rather than raising when it cannot reach the CDN, so an
    offline run costs one failed request and then stops trying.
    """
    if _FALLBACK[0] is None:
        from rift_oracle.riot.ddragon import default_ddragon

        _FALLBACK[0] = default_ddragon()
    return _FALLBACK[0]


def comp_scale(champions: Sequence[str], ddragon: Optional[DataDragon] = None) -> float:
    """Mean scaling rating for a five-champion composition."""
    values = [champion_scale(c, ddragon) for c in champions if c]
    if not values:
        return 0.0
    return sum(values) / len(values)


def comp_power(champions: Sequence[str], t_seconds: float, ddragon: Optional[DataDragon] = None) -> float:
    """Relative power of a composition at time ``t``, in roughly ``[-1, +1]``."""
    return comp_scale(champions, ddragon) * ramp(t_seconds)


def scaling_edge(
    blue_champions: Sequence[str],
    red_champions: Sequence[str],
    t_seconds: float,
    ddragon: Optional[DataDragon] = None,
) -> float:
    """Blue's composition advantage at time ``t``. Positive favours blue."""
    return comp_power(blue_champions, t_seconds, ddragon) - comp_power(
        red_champions, t_seconds, ddragon
    )


def crossover_minute(
    blue_champions: Sequence[str],
    red_champions: Sequence[str],
    ddragon: Optional[DataDragon] = None,
) -> Optional[float]:
    """Minute at which the scaling edge flips sides, if it ever does.

    Returns ``None`` when one composition scales better at every point in the
    game, which is the common case when both teams drafted similarly.
    """
    blue_scale = comp_scale(blue_champions, ddragon)
    red_scale = comp_scale(red_champions, ddragon)
    delta = blue_scale - red_scale
    if abs(delta) < 1e-6:
        return None
    # scaling_edge(t) = delta * ramp(t), and ramp crosses zero exactly once.
    # The sign therefore flips at the crossover minute for every non-zero delta.
    return CROSSOVER_MIN


def comp_profile(
    champions: Sequence[str], ddragon: Optional[DataDragon] = None
) -> Dict[str, float]:
    """Summary of a composition used by the advice engine.

    ``scale`` is the mean rating; ``spread`` says whether the team is uniformly
    mid-game or a mix of early bullies and late-game carries, which changes
    whether "play for tempo" or "play for time" is the right call.
    """
    values = [champion_scale(c, ddragon) for c in champions if c]
    if not values:
        return {"scale": 0.0, "spread": 0.0, "n": 0.0}
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return {"scale": mean, "spread": math.sqrt(var), "n": float(len(values))}


def biggest_scalers(
    champions: Sequence[str], ddragon: Optional[DataDragon] = None, limit: int = 2
) -> List[Tuple[str, float]]:
    """The champions pulling a composition latest, strongest first."""
    rated = [(c, champion_scale(c, ddragon)) for c in champions if c]
    rated.sort(key=lambda pair: pair[1], reverse=True)
    return rated[:limit]


def biggest_early(
    champions: Sequence[str], ddragon: Optional[DataDragon] = None, limit: int = 2
) -> List[Tuple[str, float]]:
    """The champions pulling a composition earliest, strongest first."""
    rated = [(c, champion_scale(c, ddragon)) for c in champions if c]
    rated.sort(key=lambda pair: pair[1])
    return rated[:limit]


def _normalise(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum())


def describe_matchup(
    blue_champions: Sequence[str],
    red_champions: Sequence[str],
    t_seconds: float,
    ddragon: Optional[DataDragon] = None,
) -> str:
    """One sentence on which composition the clock favours."""
    blue_scale = comp_scale(blue_champions, ddragon)
    red_scale = comp_scale(red_champions, ddragon)
    delta = blue_scale - red_scale
    minutes = t_seconds / 60.0

    if abs(delta) < 0.08:
        return "Both compositions scale about the same; the clock favours neither side."

    ahead, behind = ("Blue", "Red") if delta > 0 else ("Red", "Blue")
    strength = "far" if abs(delta) > 0.35 else "slightly"
    if minutes < CROSSOVER_MIN:
        return (
            f"{ahead} scales {strength} better, but the curve does not cross until "
            f"about {CROSSOVER_MIN:.0f}:00 - {behind} is on the clock until then."
        )
    return (
        f"{ahead} scales {strength} better and the game is past the "
        f"{CROSSOVER_MIN:.0f}:00 crossover, so every minute now favours {ahead}."
    )


def all_rated_champions() -> Iterable[str]:
    return CHAMPION_SCALE.keys()
