"""Client for the in-client Live Client Data API.

While a game is running, the League client serves the full game state on
``https://127.0.0.1:2999/liveclientdata/``. It needs no API key, but it does
use a self-signed certificate, so requests either skip verification (the
default here, safe because the host is the loopback interface) or validate
against Riot's published root certificate.

Riot documents the certificate at
https://static.developer.riotgames.com/docs/lol/riotgames.pem - pass its path
via ``--live-cert`` to verify properly.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import requests

from rift_oracle.config import RiftOracleError, Settings

log = logging.getLogger(__name__)

RIOT_CERT_URL = "https://static.developer.riotgames.com/docs/lol/riotgames.pem"


class GameNotRunning(RiftOracleError):
    """The Live Client Data API is not answering on the loopback port."""


class LiveClient:
    """Reads the live game state out of the running League client."""

    def __init__(self, settings: Settings, session: Optional[requests.Session] = None) -> None:
        self.settings = settings
        self.base = settings.live_host.rstrip("/")
        self.session = session or requests.Session()
        self.session.trust_env = False  # never send loopback traffic through a proxy

        self._verify: Any
        if settings.live_cert:
            self._verify = settings.live_cert
        elif settings.live_verify:
            self._verify = True
        else:
            self._verify = False
            _silence_insecure_warning()

        # Live Client events carry a monotonically increasing EventID, which is
        # what lets the poller emit only what it has not seen yet.
        self._last_event_id = -1

    # -- raw endpoints ----------------------------------------------------

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        url = f"{self.base}/liveclientdata/{path.lstrip('/')}"
        try:
            response = self.session.get(
                url, params=params, verify=self._verify, timeout=self.settings.request_timeout
            )
        except requests.exceptions.SSLError as exc:
            raise RiftOracleError(
                "TLS verification failed against the League client.\n"
                "  The client uses a self-signed certificate. Either drop "
                "--live-verify, or download Riot's root certificate from\n"
                f"  {RIOT_CERT_URL}\n"
                "  and pass it with --live-cert /path/to/riotgames.pem"
            ) from exc
        except requests.RequestException as exc:
            raise GameNotRunning(
                "no game found on the Live Client Data API "
                f"({self.base}).\n"
                "  Start a League game and try again - the endpoint only exists "
                "once you are in-game (it is not served in champion select or the lobby).\n"
                "  If you are in a game and still see this, the client may be on a "
                "different port; pass --live-host https://127.0.0.1:<port>."
            ) from exc

        if response.status_code == 404:
            # Served while the game is loading, before data is populated.
            raise GameNotRunning(
                "the League client is up but has no game data yet (still loading?)"
            )
        if not response.ok:
            raise RiftOracleError(f"live client returned HTTP {response.status_code} for {path}")
        try:
            return response.json()
        except ValueError as exc:
            raise RiftOracleError(f"live client returned a non-JSON body for {path}") from exc

    def all_game_data(self) -> Dict[str, Any]:
        """Everything in one request: players, events, stats, active player."""
        payload = self._get("allgamedata")
        if not isinstance(payload, dict) or "gameData" not in payload:
            raise GameNotRunning("live client responded but the payload had no gameData")
        return payload

    def game_stats(self) -> Dict[str, Any]:
        return self._get("gamestats")

    def player_list(self) -> List[Dict[str, Any]]:
        return self._get("playerlist") or []

    def active_player(self) -> Dict[str, Any]:
        return self._get("activeplayer")

    def active_player_name(self) -> str:
        return self._get("activeplayername")

    def event_data(self) -> List[Dict[str, Any]]:
        payload = self._get("eventdata") or {}
        return payload.get("Events", []) if isinstance(payload, dict) else []

    def player_items(self, summoner_name: str) -> List[Dict[str, Any]]:
        return self._get("playeritems", {"summonerName": summoner_name}) or []

    # -- helpers ----------------------------------------------------------

    def is_available(self) -> bool:
        """True when a game is running and serving data."""
        try:
            self._get("gamestats")
            return True
        except RiftOracleError:
            return False

    def new_events(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter an event list down to the ones not yet returned by this client."""
        fresh = [e for e in events if int(e.get("EventID", -1)) > self._last_event_id]
        if fresh:
            self._last_event_id = max(int(e.get("EventID", -1)) for e in fresh)
        return fresh

    def reset_event_cursor(self) -> None:
        self._last_event_id = -1


_warning_silenced = False


def _silence_insecure_warning() -> None:
    """Suppress urllib3's InsecureRequestWarning for the loopback connection."""
    global _warning_silenced
    if _warning_silenced:
        return
    try:
        import urllib3

        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    except Exception:  # pragma: no cover - urllib3 internals vary by version
        log.debug("could not disable urllib3 insecure warning")
    _warning_silenced = True
