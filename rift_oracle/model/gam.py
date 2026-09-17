"""The win-probability model: an antisymmetric generalised additive model.

Two properties drive the design, and both come from what this tool has to do
rather than from what scores best on a leaderboard.

**It has to explain itself.** "Your odds dropped eleven points" is useless
without "because they finished Infinity Edge and took third drake". A gradient
boosted ensemble would predict slightly better and would not be able to say
that without a separate approximation layer on top. An additive model can, and
exactly: the logit is a plain sum of per-feature terms, so the change in the
logit between two moments is the sum of the changes in those terms. No
sampling, no approximation, no attribution that fails to add up.

**It must not prefer a side by accident.** Every feature is a blue-minus-red
difference, and every basis function here is odd - ``phi(-x) == -phi(x)``.
Negating the input therefore negates the logit, so ``P(blue) + P(red) == 1``
holds to floating point. A single fitted ``side_bias`` scalar carries the real,
small blue-side advantage, and it is the only thing in the model that can.

Per feature the basis is::

    x                          linear response
    sign(x) * relu(|x| - k1)   extra slope once the lead gets real
    sign(x) * relu(|x| - k2)   and again once it gets decisive
    x * tau                    how much the lead matters at this point in time

``tau`` runs from -1 early to +1.5 late, which is what lets a single model know
that 3k gold at ten minutes and 3k gold at thirty minutes are different facts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from rift_oracle.model.features import FEATURE_KEYS, FeatureVector, SPEC_BY_KEY

#: Basis functions per feature. Changing this invalidates saved models.
N_BASIS = 4
#: Clock at which ``tau`` crosses zero, in minutes.
TAU_CENTER_MIN = 20.0
TAU_SCALE_MIN = 20.0
TAU_MIN, TAU_MAX = -1.0, 1.5


def tau_of(t_seconds: np.ndarray | float) -> np.ndarray | float:
    """Normalised game clock used for every time interaction."""
    minutes = np.asarray(t_seconds, dtype=np.float64) / 60.0
    return np.clip((minutes - TAU_CENTER_MIN) / TAU_SCALE_MIN, TAU_MIN, TAU_MAX)


def monotone_normals(n_basis: int = N_BASIS) -> np.ndarray:
    """Half-space normals encoding "this feature's response never reverses".

    Every feature is a blue-minus-red difference written so that more is better
    for blue, so the fitted response must be non-decreasing in it. That is not a
    cosmetic preference: without the constraint, a feature that is highly
    correlated with another (kills with gold, say) picks up a negative
    coefficient conditional on its partner, and the model reports that getting
    kills lowered your odds. It is defensible statistics and useless advice.

    The response is piecewise linear with segment slopes ``w0 + tau*w3``,
    ``+ w1`` and ``+ w2``. Requiring each to be non-negative at both ends of the
    ``tau`` range gives six half-spaces ``a . w >= 0``. Diminishing returns are
    still allowed - a hinge may flatten the curve, it just may not turn it over.
    """
    rows = []
    for tau in (TAU_MIN, TAU_MAX):
        rows.append([1.0, 0.0, 0.0, tau])  # first segment
        rows.append([1.0, 1.0, 0.0, tau])  # after the first knot
        rows.append([1.0, 1.0, 1.0, tau])  # after the second knot
    normals = np.array(rows, dtype=np.float64)
    if n_basis != N_BASIS:  # pragma: no cover - guards a future basis change
        raise ValueError("monotone_normals is written for the 4-function basis")
    return normals


def project_monotone(weights: np.ndarray, normals: np.ndarray, iterations: int = 60) -> np.ndarray:
    """Project one feature's weights onto the monotone cone (Dykstra).

    Dykstra's algorithm converges to the true nearest point in the intersection
    of convex sets, unlike plain alternating projection, which matters inside a
    Newton loop: an inexact projection would make the step direction wrong
    rather than merely conservative.
    """
    w = np.array(weights, dtype=np.float64, copy=True)
    corrections = np.zeros((normals.shape[0], w.size), dtype=np.float64)
    squared = np.einsum("ij,ij->i", normals, normals)

    for _ in range(iterations):
        for i, normal in enumerate(normals):
            y = w + corrections[i]
            dot = float(normal @ y)
            w = y - (dot / squared[i]) * normal if dot < 0 else y
            corrections[i] = y - w
        if np.all(normals @ w >= -1e-11):
            break

    # Guarantee feasibility even if Dykstra ran out of iterations.
    for _ in range(50):
        violations = normals @ w
        worst = int(np.argmin(violations))
        if violations[worst] >= -1e-11:
            break
        w = w - (violations[worst] / squared[worst]) * normals[worst]
    return w


def sigmoid(z: np.ndarray | float) -> np.ndarray | float:
    """Numerically stable logistic function."""
    z = np.clip(np.asarray(z, dtype=np.float64), -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-z))


@dataclass
class Prediction:
    """One evaluation of the model, with its exact per-feature decomposition."""

    p: float
    logit: float
    t: float
    contributions: Dict[str, float] = field(default_factory=dict)
    side_bias: float = 0.0

    @property
    def percent(self) -> float:
        return self.p * 100.0

    def top_contributions(self, limit: int = 5) -> List[Tuple[str, float]]:
        """Features currently pushing the prediction hardest, either way."""
        ranked = sorted(
            self.contributions.items(), key=lambda kv: abs(kv[1]), reverse=True
        )
        return [(k, v) for k, v in ranked[:limit] if abs(v) > 1e-6]

    def check(self, tolerance: float = 1e-8) -> bool:
        """The decomposition must reconstruct the logit exactly."""
        total = self.side_bias + sum(self.contributions.values())
        return abs(total - self.logit) < tolerance


class AdditiveWinModel:
    """Fit, evaluate, explain, and serialise the win-probability model."""

    def __init__(
        self,
        feature_keys: Sequence[str] = FEATURE_KEYS,
        knots: Optional[np.ndarray] = None,
        clips: Optional[np.ndarray] = None,
        weights: Optional[np.ndarray] = None,
        side_bias: float = 0.0,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.feature_keys: List[str] = list(feature_keys)
        n = len(self.feature_keys)
        self.knots = (
            np.asarray(knots, dtype=np.float64)
            if knots is not None
            else self._default_knots()
        )
        self.clips = (
            np.asarray(clips, dtype=np.float64)
            if clips is not None
            else np.array([SPEC_BY_KEY[k].typical * 5.0 for k in self.feature_keys])
        )
        self.weights = (
            np.asarray(weights, dtype=np.float64)
            if weights is not None
            else np.zeros((n, N_BASIS), dtype=np.float64)
        )
        self.side_bias = float(side_bias)
        self.meta: Dict[str, Any] = dict(meta or {})
        # Zero for features that already contain the clock, which switches off
        # their time-interaction column without changing the parameter layout.
        self._time_interaction = np.array(
            [
                0.0 if SPEC_BY_KEY[key].time_interacted is False else 1.0
                for key in self.feature_keys
            ],
            dtype=np.float64,
        )[None, :]

    # -- basis ------------------------------------------------------------

    def _default_knots(self) -> np.ndarray:
        """Knots at one and two and a half typical magnitudes, before training."""
        return np.array(
            [
                [SPEC_BY_KEY[k].typical * 1.0, SPEC_BY_KEY[k].typical * 2.5]
                for k in self.feature_keys
            ],
            dtype=np.float64,
        )

    def basis(
        self, values: np.ndarray, masks: np.ndarray, times: np.ndarray
    ) -> np.ndarray:
        """Expand raw features into the design tensor, shape ``(n, f, N_BASIS)``.

        Masked features produce an all-zero row, which is what makes an
        unobserved feature contribute exactly nothing rather than a guess.
        """
        values = np.atleast_2d(np.asarray(values, dtype=np.float64))
        masks = np.atleast_2d(np.asarray(masks, dtype=np.float64))
        times = np.atleast_1d(np.asarray(times, dtype=np.float64))

        x = np.clip(values, -self.clips, self.clips) * masks
        sign = np.sign(x)
        magnitude = np.abs(x)

        k1 = self.knots[:, 0][None, :]
        k2 = self.knots[:, 1][None, :]
        tau = np.asarray(tau_of(times), dtype=np.float64)[:, None]

        design = np.empty((x.shape[0], x.shape[1], N_BASIS), dtype=np.float64)
        design[:, :, 0] = x
        design[:, :, 1] = sign * np.maximum(magnitude - k1, 0.0)
        design[:, :, 2] = sign * np.maximum(magnitude - k2, 0.0)
        design[:, :, 3] = x * tau * self._time_interaction
        design *= masks[:, :, None]
        return design

    # -- inference --------------------------------------------------------

    def contributions(
        self, values: np.ndarray, masks: np.ndarray, times: np.ndarray
    ) -> np.ndarray:
        """Per-feature logit contributions, shape ``(n, n_features)``."""
        design = self.basis(values, masks, times)
        return np.einsum("nfb,fb->nf", design, self.weights)

    def logits(
        self, values: np.ndarray, masks: np.ndarray, times: np.ndarray
    ) -> np.ndarray:
        return self.contributions(values, masks, times).sum(axis=1) + self.side_bias

    def predict_proba(
        self, values: np.ndarray, masks: np.ndarray, times: np.ndarray
    ) -> np.ndarray:
        """P(blue wins) for each row."""
        return np.asarray(sigmoid(self.logits(values, masks, times)))

    def predict(self, vector: FeatureVector) -> Prediction:
        """Evaluate one snapshot and return its full decomposition."""
        contributions = self.contributions(
            vector.values[None, :], vector.mask[None, :], np.array([vector.t])
        )[0]
        logit = float(contributions.sum() + self.side_bias)
        return Prediction(
            p=float(sigmoid(logit)),
            logit=logit,
            t=vector.t,
            contributions={
                key: float(contributions[i]) for i, key in enumerate(self.feature_keys)
            },
            side_bias=self.side_bias,
        )

    def predict_series(self, vectors: Sequence[FeatureVector]) -> List[Prediction]:
        if not vectors:
            return []
        values = np.vstack([v.values for v in vectors])
        masks = np.vstack([v.mask for v in vectors])
        times = np.array([v.t for v in vectors], dtype=np.float64)

        contributions = self.contributions(values, masks, times)
        logits = contributions.sum(axis=1) + self.side_bias
        probabilities = sigmoid(logits)

        return [
            Prediction(
                p=float(probabilities[i]),
                logit=float(logits[i]),
                t=float(times[i]),
                contributions={
                    key: float(contributions[i, j])
                    for j, key in enumerate(self.feature_keys)
                },
                side_bias=self.side_bias,
            )
            for i in range(len(vectors))
        ]

    # -- fitting ----------------------------------------------------------

    def set_knots_from_data(
        self, values: np.ndarray, masks: np.ndarray, quantiles: Tuple[float, float] = (0.60, 0.88)
    ) -> None:
        """Place knots and clip bounds at quantiles of each feature's magnitude.

        Data-driven knots matter: the point where an extra 1k gold stops being
        worth as much is an empirical fact about the game, not a guess.
        """
        n_features = values.shape[1]
        knots = np.zeros((n_features, 2), dtype=np.float64)
        clips = np.zeros(n_features, dtype=np.float64)

        for j in range(n_features):
            observed = np.abs(values[masks[:, j] > 0, j])
            observed = observed[observed > 1e-9]
            if observed.size < 50:
                typical = SPEC_BY_KEY[self.feature_keys[j]].typical
                knots[j] = [typical, typical * 2.5]
                clips[j] = typical * 5.0
                continue
            k1, k2 = np.quantile(observed, quantiles)
            # Keep the knots apart so the two hinges do not collapse into one.
            if k2 - k1 < 1e-6:
                k2 = k1 + max(1e-6, abs(k1) * 0.25 + 1e-3)
            knots[j] = [k1, k2]
            clips[j] = max(np.quantile(observed, 0.999), k2 * 1.25, 1e-6)

        self.knots = knots
        self.clips = clips

    def fit(
        self,
        values: np.ndarray,
        masks: np.ndarray,
        times: np.ndarray,
        labels: np.ndarray,
        *,
        l2: float = 2.0,
        sample_weight: Optional[np.ndarray] = None,
        fit_side_bias: bool = True,
        monotone: bool = True,
        max_iter: int = 80,
        tol: float = 1e-8,
        rel_tol: float = 1e-7,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Fit by iteratively reweighted least squares (Newton's method).

        With a few dozen parameters, Newton converges in a handful of passes and
        gives the exact optimum of the penalised likelihood, which matters here:
        a well-fit logistic model on a proper scoring rule is already calibrated,
        so no post-hoc squashing is needed for the probabilities to be honest.

        With ``monotone`` set, each step is projected back onto the cone where
        every feature's response is non-decreasing. The objective stays convex
        and the feasible set is convex, so projected Newton still converges;
        see :func:`monotone_normals` for why the constraint belongs here.
        """
        normals = monotone_normals() if monotone else None
        design = self.basis(values, masks, times)
        n_samples, n_features, _ = design.shape
        flat = design.reshape(n_samples, n_features * N_BASIS)

        if fit_side_bias:
            flat = np.hstack([flat, np.ones((n_samples, 1), dtype=np.float64)])

        labels = np.asarray(labels, dtype=np.float64).reshape(-1)
        weights = (
            np.ones(n_samples, dtype=np.float64)
            if sample_weight is None
            else np.asarray(sample_weight, dtype=np.float64).reshape(-1)
        )
        weights = weights / max(weights.mean(), 1e-12)

        n_params = flat.shape[1]
        beta = np.zeros(n_params, dtype=np.float64)

        # Penalise each coefficient in proportion to its column's spread, which
        # is the same as standardising the design and penalising uniformly.
        # Plain ridge penalises raw coefficients, and these columns differ by
        # orders of magnitude: a gold lead runs to tens of thousands of units
        # while composition scaling lives inside +/-0.6. To contribute the same
        # logit, the small-scale feature needs a coefficient ~50x larger, which
        # plain ridge punishes ~2500x harder - so heavy shrinkage silently
        # deletes the informative small-scale features first. That is exactly
        # what happened: a cross-validated penalty improved calibration while
        # flattening the draft projection to nothing.
        column_scale = flat.std(axis=0)
        column_scale[column_scale <= 1e-12] = 1.0
        penalty = float(l2) * column_scale**2
        if fit_side_bias:
            # The side bias is a single real effect; do not shrink it much.
            penalty[-1] = 1e-3

        history: List[float] = []
        for iteration in range(max_iter):
            eta = flat @ beta
            p = np.asarray(sigmoid(eta))
            # Floor the IRLS weights so a confident, correct fit cannot make the
            # Hessian singular late in optimisation.
            w = np.maximum(p * (1.0 - p), 1e-6) * weights

            gradient = flat.T @ (weights * (labels - p)) - penalty * beta
            hessian = (flat * w[:, None]).T @ flat + np.diag(penalty)

            try:
                step = np.linalg.solve(hessian, gradient)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]

            # Damped projected Newton. The projection happens *inside* the line
            # search, so the accepted point is both feasible and an improvement;
            # projecting afterwards would undo the search and stall the fit.
            objective = _penalised_loglik(flat, labels, weights, beta, penalty)
            candidate = beta

            # The Newton direction can point straight out of the feasible cone,
            # in which case no step along it improves anything. Falling back to
            # the gradient keeps the fit moving along the constraint surface.
            directions = [step]
            if normals is not None:
                directions.append(gradient / max(np.linalg.norm(gradient), 1e-12))

            for direction in directions:
                scale = 1.0
                for _ in range(14):
                    trial = beta + scale * direction
                    if normals is not None:
                        trial = _project_all(trial, normals, n_features, fit_side_bias)
                    if _penalised_loglik(flat, labels, weights, trial, penalty) > objective:
                        candidate = trial
                        break
                    scale *= 0.5
                if candidate is not beta:
                    break

            shift = float(np.max(np.abs(candidate - beta)))
            beta = candidate
            history.append(objective)
            if verbose:
                print(f"  iter {iteration:2d}  obj={objective:12.2f}  step={shift:.2e}")
            if shift < tol:
                break
            # Along a constraint surface the steps stay non-zero long after the
            # objective has stopped moving, so stop on the objective instead.
            if len(history) >= 2:
                gain = history[-1] - history[-2]
                if 0.0 <= gain < rel_tol * max(abs(history[-1]), 1.0):
                    break

        if fit_side_bias:
            self.side_bias = float(beta[-1])
            beta = beta[:-1]
        else:
            self.side_bias = 0.0
        self.weights = beta.reshape(n_features, N_BASIS)

        # A constrained optimum can sit slightly inside the unconstrained one,
        # which shrinks every contribution by roughly a constant factor and
        # shows up as a calibration error rather than a ranking error. Fitting a
        # scalar temperature and folding it into the weights fixes that: scaling
        # by a positive constant preserves both the monotonicity and the exact
        # additivity of the decomposition.
        temperature, offset = self._fit_temperature(values, masks, times, labels, weights)
        if abs(temperature - 1.0) > 1e-6 or abs(offset) > 1e-9:
            self.weights *= temperature
            self.side_bias = self.side_bias * temperature + offset

        probabilities = self.predict_proba(values, masks, times)
        return {
            "monotone": bool(monotone),
            "iterations": len(history),
            "final_objective": history[-1] if history else float("nan"),
            "n_samples": int(n_samples),
            "log_loss": float(_log_loss(labels, probabilities)),
            "brier": float(np.mean((probabilities - labels) ** 2)),
            "accuracy": float(np.mean((probabilities >= 0.5) == (labels >= 0.5))),
            "side_bias": self.side_bias,
        }

    def _fit_temperature(
        self,
        values: np.ndarray,
        masks: np.ndarray,
        times: np.ndarray,
        labels: np.ndarray,
        sample_weight: np.ndarray,
        max_iter: int = 40,
    ) -> Tuple[float, float]:
        """Fit ``logit -> a * logit + b`` by maximum likelihood.

        ``a`` is constrained positive so the rescale cannot flip any feature's
        direction, which is the whole point of having constrained the fit.
        """
        raw = self.logits(values, masks, times)
        design = np.column_stack([raw, np.ones_like(raw)])
        params = np.array([1.0, 0.0], dtype=np.float64)

        for _ in range(max_iter):
            p = np.asarray(sigmoid(design @ params))
            w = np.maximum(p * (1 - p), 1e-9) * sample_weight
            gradient = design.T @ (sample_weight * (labels - p))
            hessian = (design * w[:, None]).T @ design + np.eye(2) * 1e-8
            try:
                step = np.linalg.solve(hessian, gradient)
            except np.linalg.LinAlgError:
                break
            params = params + step
            if np.max(np.abs(step)) < 1e-10:
                break

        a = float(params[0])
        if not np.isfinite(a) or a <= 1e-3 or a > 20.0:
            return 1.0, 0.0
        b = float(params[1])
        return a, (b if np.isfinite(b) else 0.0)

    # -- introspection ----------------------------------------------------

    def feature_importance(
        self, values: np.ndarray, masks: np.ndarray, times: np.ndarray
    ) -> Dict[str, float]:
        """Mean absolute logit contribution per feature over a dataset."""
        contributions = np.abs(self.contributions(values, masks, times))
        return {
            key: float(contributions[:, i].mean())
            for i, key in enumerate(self.feature_keys)
        }

    def response_curve(
        self, key: str, t_seconds: float = 1500.0, span: Optional[float] = None, points: int = 41
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Logit response of one feature across its range at a fixed clock."""
        j = self.feature_keys.index(key)
        limit = float(span if span is not None else self.clips[j])
        xs = np.linspace(-limit, limit, points)

        values = np.zeros((points, len(self.feature_keys)), dtype=np.float64)
        masks = np.zeros_like(values)
        values[:, j] = xs
        masks[:, j] = 1.0
        times = np.full(points, float(t_seconds))
        return xs, self.contributions(values, masks, times)[:, j]

    # -- serialisation ----------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "format": "rift_oracle.additive.v1",
            "n_basis": N_BASIS,
            "tau": {
                "center_min": TAU_CENTER_MIN,
                "scale_min": TAU_SCALE_MIN,
                "min": TAU_MIN,
                "max": TAU_MAX,
            },
            "feature_keys": list(self.feature_keys),
            "knots": self.knots.tolist(),
            "clips": self.clips.tolist(),
            "weights": self.weights.tolist(),
            "side_bias": self.side_bias,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "AdditiveWinModel":
        if payload.get("n_basis") not in (None, N_BASIS):
            raise ValueError(
                f"model was saved with {payload.get('n_basis')} basis functions, "
                f"this build expects {N_BASIS}; retrain with 'rift-oracle train'"
            )
        return cls(
            feature_keys=payload.get("feature_keys", FEATURE_KEYS),
            knots=np.array(payload["knots"], dtype=np.float64),
            clips=np.array(payload["clips"], dtype=np.float64),
            weights=np.array(payload["weights"], dtype=np.float64),
            side_bias=float(payload.get("side_bias", 0.0)),
            meta=payload.get("meta", {}),
        )

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> "AdditiveWinModel":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def __repr__(self) -> str:
        trained = self.meta.get("trained_on", "untrained")
        return (
            f"AdditiveWinModel(features={len(self.feature_keys)}, "
            f"side_bias={self.side_bias:+.4f}, trained_on={trained})"
        )


def _project_all(
    beta: np.ndarray, normals: np.ndarray, n_features: int, has_bias: bool
) -> np.ndarray:
    """Project every feature block of a flat parameter vector."""
    out = np.array(beta, dtype=np.float64, copy=True)
    for j in range(n_features):
        start = j * N_BASIS
        out[start : start + N_BASIS] = project_monotone(
            out[start : start + N_BASIS], normals
        )
    return out


def _penalised_loglik(
    design: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    beta: np.ndarray,
    penalty: np.ndarray,
) -> float:
    eta = np.clip(design @ beta, -35.0, 35.0)
    # log(sigmoid(eta)) written to avoid overflow for large |eta|.
    loglik = np.sum(weights * (labels * eta - np.logaddexp(0.0, eta)))
    return float(loglik - 0.5 * np.sum(penalty * beta * beta))


def _log_loss(labels: np.ndarray, probabilities: np.ndarray) -> float:
    p = np.clip(probabilities, 1e-12, 1.0 - 1e-12)
    return float(-np.mean(labels * np.log(p) + (1.0 - labels) * np.log(1.0 - p)))


def delta_attribution(
    before: Prediction, after: Prediction
) -> Tuple[float, Dict[str, float]]:
    """Split the probability change between two moments across the features.

    The logit decomposition is exact and additive, so the change in each
    feature's term sums precisely to the change in the logit. Probability is a
    non-linear function of the logit, so the total probability change is
    computed exactly and then apportioned in proportion to each feature's share
    of the logit movement. Every returned attribution therefore sums to the real
    change, with no residual to explain away.
    """
    delta_logit = after.logit - before.logit
    delta_p = after.p - before.p

    per_feature = {
        key: after.contributions.get(key, 0.0) - before.contributions.get(key, 0.0)
        for key in set(before.contributions) | set(after.contributions)
    }
    per_feature = {k: v for k, v in per_feature.items() if abs(v) > 1e-12}

    total = sum(per_feature.values())
    if abs(total) < 1e-12 or abs(delta_logit) < 1e-12:
        return delta_p, {k: 0.0 for k in per_feature}

    scale = delta_p / total
    return delta_p, {key: value * scale for key, value in per_feature.items()}
