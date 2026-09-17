"""rift_oracle - live win-probability oracle for League of Legends.

The package is organised in four layers:

``rift_oracle.riot``
    Transport. Talks to the Riot Web API (match, spectator, account) and to the
    in-client Live Client Data API on ``127.0.0.1:2999``.

``rift_oracle.game``
    Normalisation. Turns either transport's payloads into the same
    :class:`~rift_oracle.game.state.GameState` sequence plus a stream of
    :class:`~rift_oracle.game.state.GameEvent` records.

``rift_oracle.model``
    Inference. An antisymmetric additive model maps a ``GameState`` to a win
    probability and, because it is additive, to an exact per-feature
    decomposition of the logit.

``rift_oracle.analysis`` / ``rift_oracle.ui``
    Interpretation and presentation: swing detection, causal narration,
    counterfactual advice, and the terminal / HTML front ends.
"""

__version__ = "1.0.0"

__all__ = ["__version__"]
