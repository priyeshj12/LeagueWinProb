"""Platform and regional routing tables for the Riot Web API.

Riot splits its endpoints across two host families:

* *platform* hosts (``na1``, ``euw1``, ...) serve per-shard data such as
  ``spectator-v5`` and ``summoner-v4``.
* *regional* hosts (``americas``, ``europe``, ``asia``, ``sea``) serve the
  account and match endpoints.

Every caller in this package names a platform; the regional host is derived.
"""

from __future__ import annotations

from typing import Dict, List, Optional

#: Platform routing value -> human label. Ordered roughly by population.
PLATFORMS: Dict[str, str] = {
    "na1": "North America",
    "euw1": "Europe West",
    "eun1": "Europe Nordic & East",
    "kr": "Korea",
    "br1": "Brazil",
    "jp1": "Japan",
    "la1": "Latin America North",
    "la2": "Latin America South",
    "oc1": "Oceania",
    "tr1": "Turkey",
    "ru": "Russia",
    "me1": "Middle East",
    "sg2": "Singapore / Malaysia / Indonesia",
    "tw2": "Taiwan",
    "vn2": "Vietnam",
    "ph2": "Philippines",
    "th2": "Thailand",
    "pbe1": "Public Beta Environment",
}

#: Platform -> regional routing value used by ``account-v1`` and ``match-v5``.
_REGIONAL: Dict[str, str] = {
    "na1": "americas",
    "br1": "americas",
    "la1": "americas",
    "la2": "americas",
    "pbe1": "americas",
    "euw1": "europe",
    "eun1": "europe",
    "tr1": "europe",
    "ru": "europe",
    "me1": "europe",
    "kr": "asia",
    "jp1": "asia",
    "tw2": "asia",
    "oc1": "sea",
    "sg2": "sea",
    "vn2": "sea",
    "ph2": "sea",
    "th2": "sea",
}

#: Common aliases people type instead of the platform routing value.
_ALIASES: Dict[str, str] = {
    "na": "na1",
    "nae": "na1",
    "euw": "euw1",
    "eune": "eun1",
    "eun": "eun1",
    "br": "br1",
    "jp": "jp1",
    "lan": "la1",
    "las": "la2",
    "oce": "oc1",
    "oc": "oc1",
    "tr": "tr1",
    "sg": "sg2",
    "tw": "tw2",
    "vn": "vn2",
    "ph": "ph2",
    "th": "th2",
    "me": "me1",
    "pbe": "pbe1",
}

#: Queue id -> label. Only the ones this tool actually models.
QUEUES: Dict[int, str] = {
    400: "Normal Draft",
    420: "Ranked Solo/Duo",
    430: "Normal Blind",
    440: "Ranked Flex",
    450: "ARAM",
    490: "Quickplay",
    700: "Clash",
    1700: "Arena",
}

#: Queues that share Summoner's Rift 5v5 dynamics, i.e. what the model is for.
SUPPORTED_QUEUES = (400, 420, 430, 440, 490, 700)

#: Ranked queues, the default training population.
RANKED_QUEUES = (420, 440)


def resolve_platform(value: str) -> str:
    """Normalise user input (``"NA"``, ``"na1"``, ``"euw"``) to a platform id.

    Raises ``ValueError`` when the value maps to no known platform so that a
    typo fails at argument-parse time instead of as a 404 twenty seconds later.
    """
    key = (value or "").strip().lower()
    if not key:
        raise ValueError("platform is empty")
    key = _ALIASES.get(key, key)
    if key not in PLATFORMS:
        known = ", ".join(sorted(PLATFORMS))
        raise ValueError(f"unknown platform {value!r}; expected one of: {known}")
    return key


def regional_route(platform: str) -> str:
    """Return the regional routing value (``americas`` / ``europe`` / ...)."""
    return _REGIONAL.get(resolve_platform(platform), "americas")


def platform_host(platform: str) -> str:
    """Return the fully qualified platform host."""
    return f"https://{resolve_platform(platform)}.api.riotgames.com"


def regional_host(platform: str) -> str:
    """Return the fully qualified regional host for a platform."""
    return f"https://{regional_route(platform)}.api.riotgames.com"


def platforms_in_region(region: str) -> List[str]:
    """All platforms routed through a given regional host."""
    return [p for p, r in _REGIONAL.items() if r == region]


def platform_for_match_id(match_id: str) -> Optional[str]:
    """Infer the platform from a match id such as ``NA1_5123456789``.

    Match ids are prefixed with the platform that hosted the game, which is how
    ``replay`` can work without the user passing ``--platform``.
    """
    if "_" not in match_id:
        return None
    prefix = match_id.split("_", 1)[0].lower()
    try:
        return resolve_platform(prefix)
    except ValueError:
        return None


def queue_name(queue_id: Optional[int]) -> str:
    if queue_id is None:
        return "Unknown queue"
    return QUEUES.get(queue_id, f"Queue {queue_id}")
