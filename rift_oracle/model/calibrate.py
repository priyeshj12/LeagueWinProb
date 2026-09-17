"""Calibration and scoring.

A win probability is only useful if it means what it says: of all the moments
the model called 70%, close to 70% should end in a win. Accuracy does not
measure that and can look fine while the numbers are badly wrong, so the
metrics here are the proper scoring rules plus an explicit reliability table.

:class:`IsotonicCalibrator` is available for the case where a model is fit on
one distribution and scored on another. A well-fit logistic model on its own
training distribution is already calibrated, so it is normally unused, and
``rift-oracle backtest`` reports whether applying it would help.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


def log_loss(labels: np.ndarray, probabilities: np.ndarray) -> float:
    p = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-12, 1 - 1e-12)
    y = np.asarray(labels, dtype=np.float64)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(labels: np.ndarray, probabilities: np.ndarray) -> float:
    return float(np.mean((np.asarray(probabilities) - np.asarray(labels)) ** 2))


def accuracy(labels: np.ndarray, probabilities: np.ndarray) -> float:
    return float(np.mean((np.asarray(probabilities) >= 0.5) == (np.asarray(labels) >= 0.5)))


def auc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    """Area under the ROC curve, via the rank-sum identity (ties averaged)."""
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    p = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    positives = int(np.sum(y >= 0.5))
    negatives = int(y.size - positives)
    if positives == 0 or negatives == 0:
        return float("nan")

    order = np.argsort(p, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, p.size + 1, dtype=np.float64)

    # Average ranks within groups of equal probability so ties score 0.5.
    sorted_p = p[order]
    start = 0
    for i in range(1, sorted_p.size + 1):
        if i == sorted_p.size or sorted_p[i] != sorted_p[start]:
            if i - start > 1:
                ranks[order[start:i]] = ranks[order[start:i]].mean()
            start = i

    rank_sum = float(np.sum(ranks[y >= 0.5]))
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def expected_calibration_error(
    labels: np.ndarray, probabilities: np.ndarray, bins: int = 20
) -> float:
    """Average gap between stated confidence and observed frequency."""
    table = reliability_table(labels, probabilities, bins=bins)
    total = sum(row["count"] for row in table)
    if total == 0:
        return float("nan")
    return float(
        sum(row["count"] * abs(row["mean_predicted"] - row["observed"]) for row in table)
        / total
    )


def reliability_table(
    labels: np.ndarray, probabilities: np.ndarray, bins: int = 20
) -> List[Dict[str, float]]:
    """Per-bin predicted vs observed win rate, for a calibration plot."""
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    p = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows: List[Dict[str, float]] = []

    for i in range(bins):
        low, high = edges[i], edges[i + 1]
        selected = (p >= low) & (p < high if i < bins - 1 else p <= high)
        count = int(np.sum(selected))
        if count == 0:
            continue
        rows.append(
            {
                "low": float(low),
                "high": float(high),
                "count": count,
                "mean_predicted": float(np.mean(p[selected])),
                "observed": float(np.mean(y[selected])),
            }
        )
    return rows


def metrics(labels: np.ndarray, probabilities: np.ndarray) -> Dict[str, float]:
    """The full scorecard for a set of predictions."""
    return {
        "n": int(np.asarray(labels).size),
        "log_loss": log_loss(labels, probabilities),
        "brier": brier(labels, probabilities),
        "accuracy": accuracy(labels, probabilities),
        "auc": auc(labels, probabilities),
        "ece": expected_calibration_error(labels, probabilities),
        "base_rate": float(np.mean(np.asarray(labels))),
    }


def metrics_by_minute(
    labels: np.ndarray,
    probabilities: np.ndarray,
    times: np.ndarray,
    buckets: Sequence[Tuple[float, float]] = (
        (0, 10), (10, 15), (15, 20), (20, 25), (25, 30), (30, 40), (40, 120),
    ),
) -> List[Dict[str, Any]]:
    """Scorecard sliced by game clock.

    Early-game predictions should be near a coin flip and late-game ones should
    be decisive; a model that is confident at four minutes is broken even if
    its overall log loss looks good.
    """
    minutes = np.asarray(times, dtype=np.float64) / 60.0
    rows: List[Dict[str, Any]] = []

    for low, high in buckets:
        selected = (minutes >= low) & (minutes < high)
        count = int(np.sum(selected))
        if count < 10:
            continue
        row: Dict[str, Any] = {"from_min": low, "to_min": high}
        row.update(metrics(np.asarray(labels)[selected], np.asarray(probabilities)[selected]))
        row["mean_confidence"] = float(
            np.mean(np.abs(np.asarray(probabilities)[selected] - 0.5)) * 2.0
        )
        rows.append(row)
    return rows


@dataclass
class IsotonicCalibrator:
    """Monotone probability remapping fit by pool-adjacent-violators."""

    x: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0]))
    y: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0]))

    def fit(self, probabilities: np.ndarray, labels: np.ndarray) -> "IsotonicCalibrator":
        p = np.asarray(probabilities, dtype=np.float64).reshape(-1)
        y = np.asarray(labels, dtype=np.float64).reshape(-1)
        if p.size == 0:
            return self

        order = np.argsort(p, kind="mergesort")
        p, y = p[order], y[order]

        # Pool adjacent violators: merge neighbouring blocks until the fitted
        # values are non-decreasing.
        values = list(y)
        weights = [1.0] * len(y)
        i = 0
        while i < len(values) - 1:
            if values[i] <= values[i + 1] + 1e-12:
                i += 1
                continue
            total_weight = weights[i] + weights[i + 1]
            pooled = (values[i] * weights[i] + values[i + 1] * weights[i + 1]) / total_weight
            values[i : i + 2] = [pooled]
            weights[i : i + 2] = [total_weight]
            if i > 0:
                i -= 1

        xs: List[float] = []
        ys: List[float] = []
        cursor = 0
        for value, weight in zip(values, weights):
            span = int(round(weight))
            xs.append(float(np.mean(p[cursor : cursor + span])))
            ys.append(float(value))
            cursor += span

        self.x = np.array(xs, dtype=np.float64)
        self.y = np.array(ys, dtype=np.float64)
        return self

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        if self.x.size < 2:
            return np.asarray(probabilities, dtype=np.float64)
        return np.clip(
            np.interp(np.asarray(probabilities, dtype=np.float64), self.x, self.y),
            1e-6,
            1 - 1e-6,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {"x": self.x.tolist(), "y": self.y.tolist()}

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "IsotonicCalibrator":
        return cls(
            x=np.array(payload.get("x", [0.0, 1.0]), dtype=np.float64),
            y=np.array(payload.get("y", [0.0, 1.0]), dtype=np.float64),
        )


def calibration_gain(
    labels: np.ndarray, probabilities: np.ndarray, folds: int = 4
) -> Dict[str, float]:
    """How much an isotonic remap would improve log loss, cross-validated.

    A number near zero means the raw model is already calibrated, which is the
    outcome to hope for. A large positive number means the fit is overconfident
    somewhere and the reliability table will show where.
    """
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    p = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if y.size < 200:
        return {"raw_log_loss": log_loss(y, p), "calibrated_log_loss": float("nan"), "gain": 0.0}

    rng = np.random.default_rng(0)
    assignment = rng.integers(0, folds, size=y.size)
    calibrated = np.empty_like(p)

    for fold in range(folds):
        test = assignment == fold
        train = ~test
        if np.sum(train) < 100 or np.sum(test) == 0:
            calibrated[test] = p[test]
            continue
        calibrated[test] = IsotonicCalibrator().fit(p[train], y[train]).transform(p[test])

    raw = log_loss(y, p)
    fixed = log_loss(y, calibrated)
    return {"raw_log_loss": raw, "calibrated_log_loss": fixed, "gain": raw - fixed}
