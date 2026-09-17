"""Interpretation: when the game swung, why it swung, and what to do now."""

from rift_oracle.analysis.advice import Suggestion, build_advice, suggest_actions
from rift_oracle.analysis.narrate import narrate_swing, narrate_state
from rift_oracle.analysis.swings import Swing, Track, TrackPoint, detect_swings

__all__ = [
    "Swing",
    "Track",
    "TrackPoint",
    "detect_swings",
    "narrate_swing",
    "narrate_state",
    "Suggestion",
    "suggest_actions",
    "build_advice",
]
