"""Data Dragon: champion and item metadata.

Data Dragon is Riot's static-content CDN. It needs no API key, so this module
works even in the keyless ``live`` and ``demo`` paths. Everything is cached on
disk, and every lookup degrades to a sensible default when the CDN is
unreachable, because a missing item name must never stop a live readout.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set

import requests

from rift_oracle.riot.client import DiskCache

log = logging.getLogger(__name__)

DDRAGON_BASE = "https://ddragon.leagueoflegends.com"
FALLBACK_VERSION = "15.18.1"

#: Item ids worth calling out by name even with no CDN access. These are the
#: components whose completion moves a game most, plus the objective-adjacent
#: consumables the advice engine reasons about.
_KNOWN_ITEMS: Dict[int, Dict[str, Any]] = {
    1001: {"name": "Boots", "gold": 300, "legendary": False},
    2003: {"name": "Health Potion", "gold": 50, "legendary": False},
    2055: {"name": "Control Ward", "gold": 75, "legendary": False},
    2031: {"name": "Refillable Potion", "gold": 150, "legendary": False},
    3006: {"name": "Berserker's Greaves", "gold": 1100, "legendary": False},
    3020: {"name": "Sorcerer's Shoes", "gold": 1100, "legendary": False},
    3047: {"name": "Plated Steelcaps", "gold": 1200, "legendary": False},
    3111: {"name": "Mercury's Treads", "gold": 1200, "legendary": False},
    3031: {"name": "Infinity Edge", "gold": 3450, "legendary": True},
    3036: {"name": "Lord Dominik's Regards", "gold": 3000, "legendary": True},
    3072: {"name": "Bloodthirster", "gold": 3400, "legendary": True},
    3074: {"name": "Ravenous Hydra", "gold": 3300, "legendary": True},
    3078: {"name": "Trinity Force", "gold": 3333, "legendary": True},
    3089: {"name": "Rabadon's Deathcap", "gold": 3600, "legendary": True},
    3094: {"name": "Rapid Firecannon", "gold": 2600, "legendary": True},
    3115: {"name": "Nashor's Tooth", "gold": 3000, "legendary": True},
    3124: {"name": "Guinsoo's Rageblade", "gold": 3000, "legendary": True},
    3135: {"name": "Void Staff", "gold": 3000, "legendary": True},
    3153: {"name": "Blade of the Ruined King", "gold": 3200, "legendary": True},
    3157: {"name": "Zhonya's Hourglass", "gold": 3250, "legendary": True},
    3026: {"name": "Guardian Angel", "gold": 3200, "legendary": True},
    3065: {"name": "Spirit Visage", "gold": 2900, "legendary": True},
    3068: {"name": "Sunfire Aegis", "gold": 2900, "legendary": True},
    3075: {"name": "Thornmail", "gold": 2700, "legendary": True},
    3083: {"name": "Warmog's Armor", "gold": 3100, "legendary": True},
    3110: {"name": "Frozen Heart", "gold": 2800, "legendary": True},
    3143: {"name": "Randuin's Omen", "gold": 2700, "legendary": True},
    3033: {"name": "Mortal Reminder", "gold": 3000, "legendary": True},
    3123: {"name": "Executioner's Calling", "gold": 800, "legendary": False},
    3076: {"name": "Bramble Vest", "gold": 900, "legendary": False},
    3916: {"name": "Oblivion Orb", "gold": 800, "legendary": False},
}

#: Item ids that reduce enemy healing, used by the advice engine.
ANTIHEAL_ITEMS: Dict[int, str] = {
    3123: "Executioner's Calling",
    3033: "Mortal Reminder",
    3076: "Bramble Vest",
    3075: "Thornmail",
    3916: "Oblivion Orb",
    3165: "Morellonomicon",
    3011: "Chemtech Putrifier",
}

#: Armor-heavy and MR-heavy defensive picks the advice engine recommends.
ARMOR_ITEMS: Dict[int, str] = {
    3047: "Plated Steelcaps",
    3075: "Thornmail",
    3110: "Frozen Heart",
    3143: "Randuin's Omen",
    3068: "Sunfire Aegis",
    3105: "Aegis of the Legion",
}
MR_ITEMS: Dict[int, str] = {
    3111: "Mercury's Treads",
    3065: "Spirit Visage",
    3102: "Banshee's Veil",
    3156: "Maw of Malmortius",
    3139: "Mercurial Scimitar",
    3105: "Aegis of the Legion",
}
#: Penetration items, recommended when the enemy stacks resistances.
ARMOR_PEN_ITEMS: Dict[int, str] = {
    3036: "Lord Dominik's Regards",
    3033: "Mortal Reminder",
    6694: "Serylda's Grudge",
    3142: "Youmuu's Ghostblade",
}
MAGIC_PEN_ITEMS: Dict[int, str] = {
    3135: "Void Staff",
    3020: "Sorcerer's Shoes",
    4645: "Shadowflame",
    3115: "Nashor's Tooth",
}

#: Gold value threshold above which finishing an item is treated as a spike.
LEGENDARY_GOLD_FLOOR = 2200


class DataDragon:
    """Lazy, cached accessor for Data Dragon champion and item tables."""

    def __init__(
        self,
        locale: str = "en_US",
        version: Optional[str] = None,
        cache: Optional[DiskCache] = None,
        offline: bool = False,
    ) -> None:
        self.locale = locale
        self.offline = offline
        self.cache = cache or DiskCache(ttl_s=7 * 24 * 3600)
        self._version = version
        self._champions: Optional[Dict[str, Any]] = None
        self._champ_by_key: Optional[Dict[int, Dict[str, Any]]] = None
        self._items: Optional[Dict[str, Any]] = None

    # -- fetching ---------------------------------------------------------

    def _fetch(self, url: str) -> Optional[Any]:
        hit = self.cache.get(url, immutable=False)
        if hit is not None:
            return hit
        if self.offline:
            return None
        try:
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:  # network, DNS, JSON - all non-fatal here
            log.debug("data dragon fetch failed for %s: %s", url, exc)
            return None
        self.cache.put(url, payload)
        return payload

    @property
    def version(self) -> str:
        if self._version:
            return self._version
        versions = self._fetch(f"{DDRAGON_BASE}/api/versions.json")
        if isinstance(versions, list) and versions:
            self._version = str(versions[0])
        else:
            self._version = FALLBACK_VERSION
        return self._version

    # -- champions --------------------------------------------------------

    def champions(self) -> Dict[str, Any]:
        """Champion table keyed by Data Dragon id (``"MissFortune"``)."""
        if self._champions is None:
            url = f"{DDRAGON_BASE}/cdn/{self.version}/data/{self.locale}/champion.json"
            payload = self._fetch(url) or {}
            self._champions = payload.get("data", {}) if isinstance(payload, dict) else {}
        return self._champions

    def _champions_by_key(self) -> Dict[int, Dict[str, Any]]:
        if self._champ_by_key is None:
            table: Dict[int, Dict[str, Any]] = {}
            for entry in self.champions().values():
                try:
                    table[int(entry.get("key"))] = entry
                except (TypeError, ValueError):
                    continue
            self._champ_by_key = table
        return self._champ_by_key

    def champion_name(self, champion_id: int) -> str:
        """Display name for a numeric champion id, or ``Champion <id>``."""
        entry = self._champions_by_key().get(int(champion_id))
        if entry:
            return str(entry.get("name") or entry.get("id") or f"Champion {champion_id}")
        return f"Champion {champion_id}"

    def champion_key_to_id(self, champion_id: int) -> Optional[str]:
        """Numeric id -> Data Dragon string id (``"MissFortune"``)."""
        entry = self._champions_by_key().get(int(champion_id))
        return str(entry["id"]) if entry and "id" in entry else None

    def champion_tags(self, name_or_id: str) -> List[str]:
        """Roles Riot assigns a champion (``["Marksman", "Assassin"]``)."""
        table = self.champions()
        entry = table.get(name_or_id)
        if entry is None:
            wanted = _normalise(name_or_id)
            for candidate in table.values():
                if _normalise(str(candidate.get("name", ""))) == wanted:
                    entry = candidate
                    break
        if not entry:
            return []
        tags = entry.get("tags") or []
        return [str(t) for t in tags]

    # -- items ------------------------------------------------------------

    def items(self) -> Dict[str, Any]:
        if self._items is None:
            url = f"{DDRAGON_BASE}/cdn/{self.version}/data/{self.locale}/item.json"
            payload = self._fetch(url) or {}
            self._items = payload.get("data", {}) if isinstance(payload, dict) else {}
        return self._items

    def item(self, item_id: int) -> Optional[Dict[str, Any]]:
        return self.items().get(str(int(item_id)))

    def item_name(self, item_id: int) -> str:
        entry = self.item(item_id)
        if entry and entry.get("name"):
            return str(entry["name"])
        known = _KNOWN_ITEMS.get(int(item_id))
        if known:
            return str(known["name"])
        return f"Item {item_id}"

    def item_cost(self, item_id: int) -> int:
        """Total gold cost, falling back to the bundled table then to 0."""
        entry = self.item(item_id)
        if entry:
            gold = entry.get("gold") or {}
            total = gold.get("total")
            if isinstance(total, (int, float)):
                return int(total)
        known = _KNOWN_ITEMS.get(int(item_id))
        return int(known["gold"]) if known else 0

    def is_legendary(self, item_id: int) -> bool:
        """True for a terminal, expensive, Summoner's Rift item.

        "Legendary" here means *finished*: something the player builds toward
        and that produces a real power spike on completion. Components, boots,
        consumables, trinkets, and anything that builds into something else are
        excluded so the swing narration does not celebrate a Long Sword.
        """
        entry = self.item(item_id)
        if entry is None:
            known = _KNOWN_ITEMS.get(int(item_id))
            return bool(known and known.get("legendary"))

        gold = entry.get("gold") or {}
        if not gold.get("purchasable", True):
            return False
        if int(gold.get("total") or 0) < LEGENDARY_GOLD_FLOOR:
            return False
        if entry.get("into"):
            return False  # still a component
        if entry.get("consumed") or entry.get("inStore") is False:
            return False
        tags: Set[str] = set(entry.get("tags") or [])
        if tags & {"Consumable", "Trinket", "Jungle", "Lane"}:
            return False
        maps = entry.get("maps") or {}
        if maps and maps.get("11") is False:  # 11 == Summoner's Rift
            return False
        return True

    def item_tags(self, item_id: int) -> Set[str]:
        entry = self.item(item_id)
        return set(entry.get("tags") or []) if entry else set()

    def item_stats(self, item_id: int) -> Dict[str, float]:
        entry = self.item(item_id)
        stats = entry.get("stats") if entry else None
        if not isinstance(stats, dict):
            return {}
        return {str(k): float(v) for k, v in stats.items() if isinstance(v, (int, float))}

    def warmup(self) -> bool:
        """Pre-fetch both tables. Returns True when metadata is available."""
        return bool(self.champions()) and bool(self.items())


def _normalise(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum())


_DEFAULT: Optional[DataDragon] = None


def default_ddragon(locale: str = "en_US", offline: bool = False) -> DataDragon:
    """Process-wide Data Dragon instance so the tables load at most once."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = DataDragon(locale=locale, offline=offline)
    return _DEFAULT
