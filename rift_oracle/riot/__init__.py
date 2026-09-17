"""Transport layer for Riot's public API and the in-client Live Client Data API."""

from rift_oracle.riot.routing import (
    PLATFORMS,
    QUEUES,
    platform_host,
    regional_host,
    regional_route,
    resolve_platform,
)

__all__ = [
    "PLATFORMS",
    "QUEUES",
    "platform_host",
    "regional_host",
    "regional_route",
    "resolve_platform",
]
