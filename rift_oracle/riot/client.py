"""Riot Web API client: rate limiting, retries, and an on-disk cache.

Development keys are limited to 20 requests/second and 100 requests/2 minutes,
and those budgets are enforced **per routing value** - ``euw1`` and ``europe``
each get their own. So does this client: it keeps one :class:`RateLimiter` per
host, which means resolving accounts and fetching ranks does not spend the
budget that match downloads need. Both windows are enforced locally, so the
client rarely sees a 429 at all, and when it does it honours ``Retry-After``
rather than guessing.

Match and timeline payloads are immutable once a game ends, so they are cached
on disk forever; everything else uses a short TTL or no cache at all.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence, Tuple

import requests

from rift_oracle.config import RiftOracleError, Settings, cache_dir, redact
from rift_oracle.riot.routing import (
    platform_host,
    regional_host,
    resolve_platform,
)

log = logging.getLogger(__name__)

RETRY_STATUS = frozenset({429, 500, 502, 503, 504})


class RiotAPIError(RiftOracleError):
    """An HTTP error from the Riot API, with the key scrubbed from the text."""

    def __init__(self, status: int, url: str, body: str = "") -> None:
        self.status = status
        self.url = redact(url)
        self.body = redact(body)[:400]
        super().__init__(self._describe())

    def _describe(self) -> str:
        hints = {
            400: "bad request - check the puuid / match id you passed",
            401: "no API key was sent",
            403: "key rejected: it is expired, revoked, or lacks this endpoint "
            "(development keys expire every 24h - regenerate at "
            "https://developer.riotgames.com/)",
            404: "not found (for spectator this usually means the player is not in a game)",
            415: "unsupported media type",
            429: "rate limited",
            500: "Riot server error",
            503: "Riot service unavailable",
        }
        hint = hints.get(self.status, "")
        suffix = f" - {hint}" if hint else ""
        return f"HTTP {self.status} from {self.url}{suffix}"


class RateLimiter:
    """Sliding-window limiter covering several (count, seconds) buckets.

    A server-imposed pause is tracked as its own deadline rather than by
    stuffing the windows with synthetic timestamps. Filling the windows would
    make a one-second ``Retry-After`` block for the length of the *longest*
    window, so a single 429 would stall the client for two minutes.
    """

    def __init__(self, limits: Sequence[Tuple[int, float]] = ((20, 1.0), (100, 120.0))) -> None:
        self._limits: List[Tuple[int, float, Deque[float]]] = [
            (count, window, deque()) for count, window in limits
        ]
        self._lock = threading.Lock()
        self._blocked_until = 0.0

    def acquire(self) -> None:
        """Block until a request may be issued, then record it."""
        while True:
            with self._lock:
                now = time.monotonic()
                wait = max(0.0, self._blocked_until - now)
                for count, window, stamps in self._limits:
                    while stamps and now - stamps[0] >= window:
                        stamps.popleft()
                    if len(stamps) >= count:
                        wait = max(wait, window - (now - stamps[0]) + 0.01)
                if wait <= 0:
                    for _count, _window, stamps in self._limits:
                        stamps.append(now)
                    return
            time.sleep(min(wait, 5.0))

    def penalise(self, seconds: float) -> None:
        """Hold off for ``seconds``, as the server asked."""
        with self._lock:
            self._blocked_until = max(
                self._blocked_until, time.monotonic() + max(seconds, 0.0)
            )


class DiskCache:
    """Tiny JSON cache. Immutable entries never expire; others use a TTL."""

    def __init__(self, root: Optional[Path] = None, ttl_s: int = 21600) -> None:
        self.root = Path(root) if root else cache_dir()
        self.ttl_s = ttl_s

    def _path(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.root / digest[:2] / f"{digest}.json"

    def get(self, key: str, immutable: bool = False) -> Optional[Any]:
        path = self._path(key)
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError:
            return None
        try:
            entry = json.loads(raw)
        except ValueError:
            return None
        if not immutable:
            age = time.time() - float(entry.get("stored_at", 0))
            if age > self.ttl_s:
                return None
        return entry.get("payload")

    def put(self, key: str, payload: Any) -> None:
        path = self._path(key)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(
                json.dumps({"stored_at": time.time(), "payload": payload}),
                encoding="utf-8",
            )
            tmp.replace(path)
        except OSError as exc:  # a broken cache must never break a request
            log.debug("cache write failed for %s: %s", path, exc)

    def clear(self) -> int:
        removed = 0
        if not self.root.exists():
            return 0
        for path in self.root.rglob("*.json"):
            try:
                path.unlink()
                removed += 1
            except OSError:
                pass
        return removed


class RiotClient:
    """Typed wrapper over the handful of endpoints this tool uses."""

    def __init__(self, settings: Settings, session: Optional[requests.Session] = None) -> None:
        self.settings = settings
        self.platform = settings.platform
        self.session = session or requests.Session()
        self.session.headers.update(
            {
                "Accept": "application/json",
                "User-Agent": "rift_oracle/1.0 (+https://github.com/priyeshj12/LeagueWinProb)",
            }
        )
        # One limiter per routing value, created on first use.
        self._limiters: Dict[str, RateLimiter] = {}
        self.cache = DiskCache(ttl_s=settings.cache_ttl_s)
        self.request_count = 0
        self.cache_hits = 0

    def limiter_for(self, url: str) -> RateLimiter:
        """The limiter guarding the routing value this URL belongs to."""
        return self._limiters.setdefault(routing_value(url), RateLimiter())

    # -- plumbing ---------------------------------------------------------

    def _headers(self) -> Dict[str, str]:
        key = self.settings.require_key("Riot Web API access")
        return {"X-Riot-Token": key}

    def get(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        *,
        cache: bool = False,
        immutable: bool = False,
        allow_404: bool = False,
    ) -> Any:
        """GET ``url``, returning parsed JSON (or ``None`` for an allowed 404)."""
        cache_key = url + "?" + json.dumps(params or {}, sort_keys=True)
        if cache:
            hit = self.cache.get(cache_key, immutable=immutable)
            if hit is not None:
                self.cache_hits += 1
                return hit

        if self.settings.offline:
            raise RiftOracleError(
                f"offline mode is on and {redact(url)} is not in the cache"
            )

        backoff = 1.0
        last_error: Optional[RiotAPIError] = None
        limiter = self.limiter_for(url)

        for attempt in range(self.settings.max_retries + 1):
            limiter.acquire()
            self.request_count += 1
            try:
                response = self.session.get(
                    url,
                    params=params,
                    headers=self._headers(),
                    timeout=self.settings.request_timeout,
                )
            except requests.RequestException as exc:
                last_error = RiotAPIError(0, url, str(exc))
                if attempt >= self.settings.max_retries:
                    raise RiftOracleError(
                        f"network error talking to {redact(url)}: {redact(str(exc))}"
                    ) from exc
                time.sleep(backoff)
                backoff = min(backoff * 2, 16.0)
                continue

            if response.status_code == 404 and allow_404:
                return None

            if response.status_code in RETRY_STATUS:
                retry_after = _retry_after(response, default=backoff)
                last_error = RiotAPIError(response.status_code, url, response.text)
                if response.status_code == 429:
                    limiter.penalise(retry_after)
                    log.warning(
                        "rate limited (%s); sleeping %.1fs",
                        response.headers.get("X-Rate-Limit-Type", "unknown"),
                        retry_after,
                    )
                if attempt >= self.settings.max_retries:
                    raise last_error
                time.sleep(retry_after)
                backoff = min(backoff * 2, 16.0)
                continue

            if not response.ok:
                raise RiotAPIError(response.status_code, url, response.text)

            try:
                payload = response.json()
            except ValueError as exc:
                raise RiftOracleError(
                    f"{redact(url)} returned a non-JSON body"
                ) from exc

            if cache:
                self.cache.put(cache_key, payload)
            return payload

        if last_error:
            raise last_error
        raise RiftOracleError(f"request to {redact(url)} failed")

    # -- account-v1 -------------------------------------------------------

    def account_by_riot_id(self, game_name: str, tag_line: str, platform: Optional[str] = None) -> Dict[str, Any]:
        """Resolve ``Name#TAG`` to an account record containing the puuid."""
        host = regional_host(platform or self.platform)
        url = f"{host}/riot/account/v1/accounts/by-riot-id/{_esc(game_name)}/{_esc(tag_line)}"
        return self.get(url, cache=True)

    def account_by_puuid(self, puuid: str, platform: Optional[str] = None) -> Dict[str, Any]:
        host = regional_host(platform or self.platform)
        return self.get(f"{host}/riot/account/v1/accounts/by-puuid/{_esc(puuid)}", cache=True)

    # -- summoner-v4 / league-v4 -----------------------------------------

    def summoner_by_puuid(self, puuid: str, platform: Optional[str] = None) -> Dict[str, Any]:
        host = platform_host(platform or self.platform)
        return self.get(f"{host}/lol/summoner/v4/summoners/by-puuid/{_esc(puuid)}", cache=True)

    def league_entries(self, puuid: str, platform: Optional[str] = None) -> List[Dict[str, Any]]:
        """Ranked entries for a puuid; empty list when unranked or unavailable."""
        host = platform_host(platform or self.platform)
        url = f"{host}/lol/league/v4/entries/by-puuid/{_esc(puuid)}"
        try:
            payload = self.get(url, cache=True, allow_404=True)
        except RiotAPIError as exc:
            if exc.status in (400, 403, 404):
                return []
            raise
        return payload or []

    APEX_TIERS = ("CHALLENGER", "GRANDMASTER", "MASTER")

    def league_entries_by_tier(
        self,
        tier: str,
        division: str = "I",
        queue: str = "RANKED_SOLO_5x5",
        page: int = 1,
        platform: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """A page of ranked entries at one tier and division.

        The apex tiers live behind their own endpoints and have no divisions,
        so they are routed differently and unwrapped to the same shape.
        """
        host = platform_host(platform or self.platform)
        tier = tier.upper()

        if tier in self.APEX_TIERS:
            url = f"{host}/lol/league/v4/{tier.lower()}leagues/by-queue/{_esc(queue)}"
            payload = self.get(url, cache=True) or {}
            return list(payload.get("entries") or [])

        url = f"{host}/lol/league/v4/entries/{_esc(queue)}/{tier}/{_esc(division.upper())}"
        return self.get(url, params={"page": max(1, int(page))}, cache=True) or []

    def champion_rotations(self, platform: Optional[str] = None) -> Dict[str, Any]:
        """A parameterless call every key can make, used as a reachability probe.

        ``spectator-v5/featured-games`` would be the obvious probe but is not
        granted to development keys, so a health check built on it reports a
        working key as broken.
        """
        host = platform_host(platform or self.platform)
        return self.get(f"{host}/lol/platform/v3/champion-rotations")

    # -- spectator-v5 -----------------------------------------------------

    def active_game(self, puuid: str, platform: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Current game for a puuid, or ``None`` when the player is not in one."""
        host = platform_host(platform or self.platform)
        url = f"{host}/lol/spectator/v5/active-games/by-summoner/{_esc(puuid)}"
        return self.get(url, allow_404=True)

    def featured_games(self, platform: Optional[str] = None) -> Dict[str, Any]:
        host = platform_host(platform or self.platform)
        return self.get(f"{host}/lol/spectator/v5/featured-games")

    # -- match-v5 ---------------------------------------------------------

    def match_ids(
        self,
        puuid: str,
        *,
        count: int = 20,
        start: int = 0,
        queue: Optional[int] = None,
        match_type: Optional[str] = None,
        start_time: Optional[int] = None,
        platform: Optional[str] = None,
    ) -> List[str]:
        host = regional_host(platform or self.platform)
        params: Dict[str, Any] = {"start": start, "count": max(1, min(count, 100))}
        if queue is not None:
            params["queue"] = queue
        if match_type:
            params["type"] = match_type
        if start_time is not None:
            params["startTime"] = start_time
        url = f"{host}/lol/match/v5/matches/by-puuid/{_esc(puuid)}/ids"
        return self.get(url, params=params) or []

    def match(self, match_id: str, platform: Optional[str] = None) -> Dict[str, Any]:
        host = regional_host(platform or self.platform)
        url = f"{host}/lol/match/v5/matches/{_esc(match_id)}"
        return self.get(url, cache=True, immutable=True)

    def timeline(self, match_id: str, platform: Optional[str] = None) -> Dict[str, Any]:
        host = regional_host(platform or self.platform)
        url = f"{host}/lol/match/v5/matches/{_esc(match_id)}/timeline"
        return self.get(url, cache=True, immutable=True)

    # -- convenience ------------------------------------------------------

    def resolve_riot_id(self, riot_id: str, platform: Optional[str] = None) -> Dict[str, Any]:
        """Accept ``Name#TAG`` (or a bare puuid) and return an account record."""
        value = riot_id.strip()
        if "#" not in value:
            if len(value) >= 70:  # puuids are 78 chars; be lenient
                return self.account_by_puuid(value, platform)
            raise RiftOracleError(
                f"{value!r} is not a Riot ID. Use the Name#TAG form, e.g. 'Faker#KR1'."
            )
        name, tag = value.rsplit("#", 1)
        if not name or not tag:
            raise RiftOracleError(f"{value!r} is not a valid Riot ID (expected Name#TAG)")
        return self.account_by_riot_id(name, tag, platform)

    def pick_platform(self, platform: Optional[str]) -> str:
        return resolve_platform(platform or self.platform)


def _retry_after(response: requests.Response, default: float) -> float:
    raw = response.headers.get("Retry-After")
    if raw:
        try:
            return max(float(raw), 0.5)
        except ValueError:
            pass
    return max(default, 1.0)


def routing_value(url: str) -> str:
    """The routing value a Riot URL belongs to (``euw1``, ``europe``, ...)."""
    from urllib.parse import urlparse

    host = urlparse(url).netloc
    return host.split(".", 1)[0].lower() if host else "unknown"


def _esc(value: str) -> str:
    from urllib.parse import quote

    return quote(str(value), safe="")


def iter_chunks(items: Iterable[Any], size: int) -> Iterable[List[Any]]:
    chunk: List[Any] = []
    for item in items:
        chunk.append(item)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk
