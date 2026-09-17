"""Runtime configuration: API key discovery, paths, and tunables.

The API key is looked up in this order, first hit wins:

1. an explicit ``--api-key`` on the command line,
2. ``$RIOT_API_KEY`` (also accepts ``$RIOT_TOKEN`` and ``$RGAPI_KEY``),
3. ``~/.rift_oracle/config.json`` under the ``api_key`` key,
4. a ``.riot_api_key`` file next to the executable or in the home directory.

Nothing in this package ever writes the key back to disk unless the user runs
``rift-oracle configure``, and the key is redacted from every log line and
error message by :func:`redact`.
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

APP_NAME = "rift_oracle"

_KEY_ENV_VARS = ("RIOT_API_KEY", "RIOT_TOKEN", "RGAPI_KEY")
_KEY_PATTERN = re.compile(r"RGAPI-[0-9a-fA-F-]{8,}")


def app_dir() -> Path:
    """Per-user state directory, honouring ``$RIFT_ORACLE_HOME``."""
    override = os.environ.get("RIFT_ORACLE_HOME")
    if override:
        return Path(override).expanduser()
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
        if base:
            return Path(base) / "rift_oracle"
    return Path.home() / ".rift_oracle"


def config_path() -> Path:
    return app_dir() / "config.json"


def cache_dir() -> Path:
    return app_dir() / "cache"


def runs_dir() -> Path:
    return app_dir() / "runs"


def models_dir() -> Path:
    return app_dir() / "models"


def bundled_data_dir() -> Path:
    """Directory of data shipped inside the package (or the PyInstaller bundle)."""
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        bundled = Path(meipass) / "rift_oracle" / "data"
        if bundled.is_dir():
            return bundled
    return Path(__file__).resolve().parent / "data"


def redact(text: str) -> str:
    """Strip anything that looks like a Riot API key out of ``text``."""
    return _KEY_PATTERN.sub("RGAPI-<redacted>", text or "")


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _discover_key() -> Optional[str]:
    for var in _KEY_ENV_VARS:
        value = os.environ.get(var)
        if value and value.strip():
            return value.strip()

    stored = _read_json(config_path()).get("api_key")
    if isinstance(stored, str) and stored.strip():
        return stored.strip()

    candidates = [app_dir() / ".riot_api_key", Path.home() / ".riot_api_key"]
    exe_dir = Path(sys.argv[0]).resolve().parent if sys.argv and sys.argv[0] else None
    if exe_dir:
        candidates.insert(0, exe_dir / ".riot_api_key")
    for candidate in candidates:
        try:
            if candidate.is_file():
                value = candidate.read_text(encoding="utf-8").strip()
                if value:
                    return value
        except OSError:
            continue
    return None


@dataclass
class Settings:
    """Everything the rest of the package needs to know about the environment."""

    api_key: Optional[str] = None
    platform: str = "na1"
    locale: str = "en_US"

    # Live Client Data API (in-client, no key required).
    live_host: str = "https://127.0.0.1:2999"
    live_verify: bool = False
    live_cert: Optional[str] = None

    # Polling and sampling.
    poll_interval: float = 2.0
    swing_threshold: float = 0.06
    swing_window_s: float = 90.0

    # HTTP behaviour.
    request_timeout: float = 12.0
    max_retries: int = 4
    cache_ttl_s: int = 6 * 60 * 60
    offline: bool = False

    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, **overrides: Any) -> "Settings":
        """Build settings from the config file, environment, then overrides.

        ``None`` overrides are ignored so that unset CLI flags do not clobber a
        stored value.
        """
        stored = _read_json(config_path())
        settings = cls()

        for key in ("platform", "locale", "poll_interval", "swing_threshold", "live_host"):
            if key in stored and stored[key] is not None:
                setattr(settings, key, stored[key])

        settings.api_key = _discover_key()

        env_platform = os.environ.get("RIOT_PLATFORM") or os.environ.get("RIFT_ORACLE_PLATFORM")
        if env_platform:
            settings.platform = env_platform

        for key, value in overrides.items():
            if value is None:
                continue
            if hasattr(settings, key):
                setattr(settings, key, value)
            else:
                settings.extra[key] = value

        from rift_oracle.riot.routing import resolve_platform

        settings.platform = resolve_platform(settings.platform)
        return settings

    def require_key(self, what: str = "this command") -> str:
        """Return the API key or raise a message that says how to supply one."""
        if self.api_key:
            return self.api_key
        raise MissingAPIKey(
            f"{what} needs a Riot API key.\n"
            "  Set one with any of:\n"
            "    export RIOT_API_KEY=RGAPI-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx\n"
            "    rift-oracle configure --api-key RGAPI-...\n"
            "    rift-oracle <cmd> --api-key RGAPI-...\n"
            "  Keys are issued at https://developer.riotgames.com/.\n"
            "  Note: 'rift-oracle live' and 'rift-oracle demo' need no key at all."
        )

    def save(self) -> Path:
        """Persist the non-secret settings plus the key to the config file."""
        target = config_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "api_key": self.api_key,
            "platform": self.platform,
            "locale": self.locale,
            "poll_interval": self.poll_interval,
            "swing_threshold": self.swing_threshold,
            "live_host": self.live_host,
        }
        with target.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        try:
            os.chmod(target, 0o600)
        except OSError:
            pass
        return target


class RiftOracleError(Exception):
    """Base class for errors this tool reports without a traceback."""


class MissingAPIKey(RiftOracleError):
    pass


class NoActiveGame(RiftOracleError):
    pass
