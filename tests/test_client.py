"""The Riot client: routing, rate limits, caching, and error handling.

Nothing here touches the network. Riot's rate limits are enforced per routing
value, so the per-host limiter split is worth testing directly - getting it
wrong either wastes half the budget or earns 429s.
"""

import time

import pytest
import requests

from rift_oracle.config import RiftOracleError, Settings
from rift_oracle.riot.client import (
    DiskCache,
    RateLimiter,
    RiotAPIError,
    RiotClient,
    routing_value,
)


class _FakeResponse:
    def __init__(self, status=200, payload=None, headers=None, text=""):
        self.status_code = status
        self._payload = payload if payload is not None else {}
        self.headers = headers or {}
        self.text = text

    @property
    def ok(self):
        return 200 <= self.status_code < 300

    def json(self):
        return self._payload


class _FakeSession:
    """Records requests and replays a queued list of responses."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []
        self.headers = {}

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append((url, params))
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _client(responses, **overrides):
    settings = Settings.load(api_key="RGAPI-test", platform="euw1", **overrides)
    return RiotClient(settings, session=_FakeSession(responses))


# -- routing ---------------------------------------------------------------


def test_routing_value_is_the_host_prefix():
    assert routing_value("https://euw1.api.riotgames.com/lol/x") == "euw1"
    assert routing_value("https://europe.api.riotgames.com/lol/y") == "europe"
    assert routing_value("not a url") == "unknown"


def test_each_routing_value_gets_its_own_limiter():
    """euw1 and europe have separate budgets, so they need separate limiters."""
    client = _client([])
    platform = client.limiter_for("https://euw1.api.riotgames.com/a")
    regional = client.limiter_for("https://europe.api.riotgames.com/b")
    assert platform is not regional
    assert client.limiter_for("https://euw1.api.riotgames.com/c") is platform


def test_match_endpoints_use_the_regional_host_and_spectator_the_platform():
    client = _client([_FakeResponse(payload={}), _FakeResponse(payload=None, status=404)])
    client.match("EUW1_1")
    client.active_game("puuid", )
    match_url, spectator_url = (call[0] for call in client.session.calls)
    assert routing_value(match_url) == "europe"
    assert routing_value(spectator_url) == "euw1"


# -- rate limiting ---------------------------------------------------------


def test_limiter_allows_a_burst_then_blocks():
    limiter = RateLimiter(limits=((3, 0.4),))
    started = time.monotonic()
    for _ in range(3):
        limiter.acquire()
    assert time.monotonic() - started < 0.1  # the burst is free

    limiter.acquire()  # the fourth has to wait for the window to roll
    assert time.monotonic() - started >= 0.35


def test_limiter_enforces_the_tighter_of_two_windows():
    limiter = RateLimiter(limits=((10, 0.2), (2, 0.5)))
    started = time.monotonic()
    for _ in range(3):
        limiter.acquire()
    # The per-second window would allow all three; the 2-per-0.5s window does not.
    assert time.monotonic() - started >= 0.45


def test_penalise_blocks_until_the_server_says_otherwise():
    limiter = RateLimiter(limits=((5, 0.3),))
    limiter.penalise(0.25)
    started = time.monotonic()
    limiter.acquire()
    assert time.monotonic() - started >= 0.2


def test_a_short_penalty_does_not_stall_for_the_long_window():
    """A one-second Retry-After must not block for the two-minute window.

    Recording the pause as synthetic timestamps inside the windows would make
    the next request wait ``window + seconds``, so a single 429 during a
    harvest would cost two minutes instead of one second.
    """
    limiter = RateLimiter(limits=((20, 1.0), (100, 120.0)))
    limiter.penalise(0.3)
    started = time.monotonic()
    limiter.acquire()
    waited = time.monotonic() - started
    assert 0.25 <= waited < 2.0


# -- requests --------------------------------------------------------------


def test_a_429_is_retried_after_the_header_says_to():
    client = _client(
        [
            _FakeResponse(429, headers={"Retry-After": "0.5"}, text="slow down"),
            _FakeResponse(200, payload={"ok": True}),
        ]
    )
    started = time.monotonic()
    assert client.get("https://europe.api.riotgames.com/x") == {"ok": True}
    assert time.monotonic() - started >= 0.4
    assert len(client.session.calls) == 2


def test_server_errors_are_retried_then_surfaced():
    client = _client([_FakeResponse(503) for _ in range(5)], max_retries=1)
    with pytest.raises(RiotAPIError) as caught:
        client.get("https://europe.api.riotgames.com/x")
    assert caught.value.status == 503
    assert len(client.session.calls) == 2  # one try plus one retry


def test_a_404_can_be_allowed_and_returns_none():
    client = _client([_FakeResponse(404)])
    assert client.get("https://euw1.api.riotgames.com/x", allow_404=True) is None


def test_active_game_returns_none_when_the_player_is_not_in_one():
    client = _client([_FakeResponse(404)])
    assert client.active_game("puuid") is None


def test_a_403_is_not_retried_and_explains_itself():
    client = _client([_FakeResponse(403, text="Forbidden")])
    with pytest.raises(RiotAPIError, match="expired, revoked"):
        client.get("https://euw1.api.riotgames.com/x")
    assert len(client.session.calls) == 1


def test_network_errors_are_retried_then_reported_without_the_key():
    """A key must never reach a log line, whatever shape it is."""
    from rift_oracle.config import remember_secret

    # A well-formed key (RGAPI- plus a UUID) and one the pattern cannot match.
    # Never put a live key here: this file is committed, and a secret in a test
    # fixture is a secret in the repository's history.
    well_formed = "RGAPI-00000000-1111-2222-3333-444444444444"
    malformed = "RGAPI-not-a-uuid-at-all"
    remember_secret(well_formed)
    remember_secret(malformed)

    for secret in (well_formed, malformed):
        client = _client(
            [requests.ConnectionError(f"dial tcp failed for {secret}")] * 3, max_retries=1
        )
        with pytest.raises(RiftOracleError) as caught:
            client.get("https://europe.api.riotgames.com/x")
        assert secret not in str(caught.value)
        assert "<redacted>" in str(caught.value)


def test_league_entries_tolerate_a_missing_or_forbidden_account():
    client = _client([_FakeResponse(404)])
    assert client.league_entries("puuid") == []


# -- ladder seeding --------------------------------------------------------


def test_apex_tiers_use_their_own_endpoint_and_unwrap_to_a_list():
    client = _client([_FakeResponse(payload={"entries": [{"puuid": "a"}, {"puuid": "b"}]})])
    entries = client.league_entries_by_tier("CHALLENGER")
    assert [e["puuid"] for e in entries] == ["a", "b"]
    url, _params = client.session.calls[0]
    # Pin the whole path: a substring check passes even with /league/v4 missing,
    # which is exactly the bug this asserts against.
    assert url.endswith("/lol/league/v4/challengerleagues/by-queue/RANKED_SOLO_5x5")


def test_normal_tiers_use_the_paged_entries_endpoint():
    client = _client([_FakeResponse(payload=[{"puuid": "a"}])])
    client.league_entries_by_tier("EMERALD", "II", page=3)
    url, params = client.session.calls[0]
    assert url.endswith("/lol/league/v4/entries/RANKED_SOLO_5x5/EMERALD/II")
    assert params == {"page": 3}


# -- caching ---------------------------------------------------------------


def test_immutable_entries_survive_an_expired_ttl(tmp_path):
    cache = DiskCache(root=tmp_path, ttl_s=0)
    cache.put("k", {"v": 1})
    time.sleep(0.01)
    assert cache.get("k") is None  # a normal entry has expired
    assert cache.get("k", immutable=True) == {"v": 1}  # a match never changes


def test_matches_are_served_from_the_cache_on_the_second_call(tmp_path):
    client = _client([_FakeResponse(payload={"metadata": {"matchId": "EUW1_1"}})])
    client.cache = DiskCache(root=tmp_path, ttl_s=3600)
    first = client.match("EUW1_1")
    second = client.match("EUW1_1")
    assert first == second
    assert len(client.session.calls) == 1
    assert client.cache_hits == 1


def test_a_corrupt_cache_entry_is_ignored_rather_than_raising(tmp_path):
    cache = DiskCache(root=tmp_path)
    cache.put("k", {"v": 1})
    path = cache._path("k")
    path.write_text("{not json", encoding="utf-8")
    assert cache.get("k") is None


def test_offline_mode_refuses_rather_than_reaching_out(tmp_path):
    client = _client([], offline=True)
    client.cache = DiskCache(root=tmp_path)
    with pytest.raises(RiftOracleError, match="offline"):
        client.get("https://europe.api.riotgames.com/x")
    assert client.session.calls == []


# -- riot id parsing -------------------------------------------------------


def test_riot_id_must_have_a_tag():
    client = _client([])
    with pytest.raises(RiftOracleError, match="Name#TAG"):
        client.resolve_riot_id("JustAName")


def test_riot_id_is_split_on_the_last_hash():
    client = _client([_FakeResponse(payload={"puuid": "p"})])
    client.resolve_riot_id("Name#With#TAG")
    url, _ = client.session.calls[0]
    assert url.endswith("/by-riot-id/Name%23With/TAG")
