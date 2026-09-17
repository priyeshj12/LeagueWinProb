"""Training pipeline.

Two things here are less obvious than the fit itself.

**Splitting by game, not by row.** Consecutive states from one match are
enormously correlated - they share a winner and most of their features. A
random row split would put minute 20 of a game in train and minute 21 in test,
and every metric would come back flattering and wrong. Every split here is on
the game index.

**Training under the masks it will actually see.** Live games cannot observe
experience or damage dealt. If the model only ever saw complete feature
vectors, it would lean on those two and then quietly lose accuracy the moment
it ran on a real game. So each game enters the training set twice: once with
everything observed, once masked down to what the Live Client Data API can see.
The model learns to be right under both, and ``backtest --live-mask`` reports
what the restriction actually costs.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from rift_oracle.config import RiftOracleError
from rift_oracle.model import calibrate
from rift_oracle.model.features import FEATURE_KEYS, apply_live_mask, extract
from rift_oracle.model.gam import AdditiveWinModel

log = logging.getLogger(__name__)


@dataclass
class Dataset:
    """Stacked feature arrays plus the game each row came from."""

    values: np.ndarray
    masks: np.ndarray
    times: np.ndarray
    labels: np.ndarray
    groups: np.ndarray

    def __len__(self) -> int:
        return int(self.values.shape[0])

    @property
    def n_games(self) -> int:
        return int(np.unique(self.groups).size)

    def subset(self, selector: np.ndarray) -> "Dataset":
        return Dataset(
            values=self.values[selector],
            masks=self.masks[selector],
            times=self.times[selector],
            labels=self.labels[selector],
            groups=self.groups[selector],
        )

    def split_by_game(
        self, test_fraction: float = 0.2, seed: int = 0
    ) -> Tuple["Dataset", "Dataset"]:
        """Hold out whole games, never individual frames."""
        unique_games = np.unique(self.groups)
        rng = np.random.default_rng(seed)
        shuffled = rng.permutation(unique_games)
        n_test = max(1, int(len(shuffled) * test_fraction))
        test_games = set(shuffled[:n_test].tolist())

        is_test = np.array([g in test_games for g in self.groups])
        return self.subset(~is_test), self.subset(is_test)

    def with_live_masks(self) -> "Dataset":
        """A copy restricted to what a live game can observe."""
        return Dataset(
            values=self.values,
            masks=apply_live_mask(self.masks),
            times=self.times,
            labels=self.labels,
            groups=self.groups,
        )

    def augmented_for_training(self, live_weight: float = 1.0) -> Tuple["Dataset", np.ndarray]:
        """Stack the full-information and live-masked views of every row."""
        live = self.with_live_masks()
        combined = Dataset(
            values=np.vstack([self.values, live.values]),
            masks=np.vstack([self.masks, live.masks]),
            times=np.concatenate([self.times, live.times]),
            labels=np.concatenate([self.labels, live.labels]),
            groups=np.concatenate([self.groups, live.groups]),
        )
        weights = np.concatenate(
            [np.ones(len(self)), np.full(len(self), float(live_weight))]
        )
        return combined, weights


def dataset_from_games(
    games: Sequence[Tuple[Sequence[Any], int]],
    rank_priors: Optional[Sequence[float]] = None,
) -> Dataset:
    """Build a dataset from ``(states, winning_team_id)`` pairs."""
    from rift_oracle.game.state import BLUE

    values: List[np.ndarray] = []
    masks: List[np.ndarray] = []
    times: List[float] = []
    labels: List[float] = []
    groups: List[int] = []

    for index, (states, winner) in enumerate(games):
        if not states or winner is None:
            continue
        prior = float(rank_priors[index]) if rank_priors is not None else 0.0
        label = 1.0 if int(winner) == BLUE else 0.0
        for state in states:
            vector = extract(state, rank_prior=prior)
            values.append(vector.values)
            masks.append(vector.mask)
            times.append(vector.t)
            labels.append(label)
            groups.append(index)

    if not values:
        raise RiftOracleError("no usable states found; nothing to train on")

    return Dataset(
        values=np.vstack(values),
        masks=np.vstack(masks),
        times=np.array(times, dtype=np.float64),
        labels=np.array(labels, dtype=np.float64),
        groups=np.array(groups, dtype=np.int64),
    )


def train_model(
    dataset: Dataset,
    *,
    l2: float = 2.0,
    test_fraction: float = 0.2,
    seed: int = 0,
    live_weight: float = 1.0,
    fit_side_bias: bool = True,
    monotone: bool = True,
    verbose: bool = False,
) -> Tuple[AdditiveWinModel, Dict[str, Any]]:
    """Fit a model and score it on held-out games."""
    train, test = dataset.split_by_game(test_fraction=test_fraction, seed=seed)

    model = AdditiveWinModel(feature_keys=list(FEATURE_KEYS))
    model.set_knots_from_data(train.values, train.masks)

    augmented, weights = train.augmented_for_training(live_weight=live_weight)
    fit_stats = model.fit(
        augmented.values,
        augmented.masks,
        augmented.times,
        augmented.labels,
        l2=l2,
        sample_weight=weights,
        fit_side_bias=fit_side_bias,
        monotone=monotone,
        verbose=verbose,
    )

    report: Dict[str, Any] = {
        "fit": fit_stats,
        "n_games": dataset.n_games,
        "n_states": len(dataset),
        "train_games": train.n_games,
        "test_games": test.n_games,
    }

    if len(test):
        full_p = model.predict_proba(test.values, test.masks, test.times)
        live = test.with_live_masks()
        live_p = model.predict_proba(live.values, live.masks, live.times)

        report["test"] = calibrate.metrics(test.labels, full_p)
        report["test_live_masked"] = calibrate.metrics(live.labels, live_p)
        report["by_minute"] = calibrate.metrics_by_minute(test.labels, full_p, test.times)
        report["calibration"] = calibrate.calibration_gain(test.labels, full_p)
        report["reliability"] = calibrate.reliability_table(test.labels, full_p, bins=10)

    report["importance"] = model.feature_importance(
        dataset.values, dataset.masks, dataset.times
    )
    return model, report


def train_synthetic(
    n_games: int = 4000,
    seed: int = 7,
    l2: float = 2.0,
    progress: Optional[Callable[[int, int], None]] = None,
    verbose: bool = False,
) -> Tuple[AdditiveWinModel, Dict[str, Any]]:
    """Generate simulated matches and fit on them."""
    from rift_oracle.sim.synth import simulate_dataset

    arrays = simulate_dataset(n_games=n_games, seed=seed, progress=progress)
    dataset = Dataset(
        values=arrays["values"],
        masks=arrays["masks"],
        times=arrays["times"],
        labels=arrays["labels"],
        groups=arrays["groups"],
    )
    model, report = train_model(dataset, l2=l2, seed=seed, verbose=verbose)
    model.meta.update(
        {
            "trained_on": "simulated",
            "n_games": n_games,
            "seed": seed,
            "l2": l2,
            "note": (
                "Fit on simulated Summoner's Rift games. Run 'rift-oracle harvest' "
                "then 'rift-oracle train --data <dir>' to refit on real matches."
            ),
        }
    )
    return model, report


def load_harvested(
    directory: Path,
    limit: Optional[int] = None,
    resolution: str = "frames",
    progress: Optional[Callable[[int, int], None]] = None,
) -> Dataset:
    """Build a dataset from match/timeline pairs written by ``harvest``."""
    from rift_oracle.game.timeline_adapter import replay_match

    directory = Path(directory)
    if not directory.is_dir():
        raise RiftOracleError(f"no such directory: {directory}")

    match_files = sorted(directory.glob("*.match.json"))
    if limit:
        match_files = match_files[:limit]
    if not match_files:
        raise RiftOracleError(
            f"{directory} has no *.match.json files.\n"
            "  Populate it first: rift-oracle harvest --riot-id 'You#TAG' --count 200"
        )

    games: List[Tuple[List[Any], int]] = []
    skipped = 0

    for index, match_path in enumerate(match_files):
        timeline_path = match_path.with_name(
            match_path.name.replace(".match.json", ".timeline.json")
        )
        if not timeline_path.is_file():
            skipped += 1
            continue
        try:
            match = json.loads(match_path.read_text(encoding="utf-8"))
            timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
            states, replay = replay_match(match, timeline, resolution=resolution)
        except (OSError, ValueError) as exc:
            log.debug("skipping %s: %s", match_path.name, exc)
            skipped += 1
            continue

        if not states or replay.winner is None:
            skipped += 1
            continue
        # Remakes and very short games are not informative about win probability.
        if replay.game_duration_s and replay.game_duration_s < 8 * 60:
            skipped += 1
            continue
        games.append((states, replay.winner))
        if progress is not None:
            progress(index + 1, len(match_files))

    if not games:
        raise RiftOracleError(
            f"found {len(match_files)} match files in {directory} but none were usable "
            f"({skipped} skipped - missing timelines, remakes, or unparseable payloads)"
        )

    log.info("loaded %d games (%d skipped)", len(games), skipped)
    return dataset_from_games(games)


def train_from_directory(
    directory: Path,
    l2: float = 2.0,
    limit: Optional[int] = None,
    seed: int = 0,
    progress: Optional[Callable[[int, int], None]] = None,
    verbose: bool = False,
) -> Tuple[AdditiveWinModel, Dict[str, Any]]:
    """Fit on real harvested matches."""
    dataset = load_harvested(directory, limit=limit, progress=progress)
    model, report = train_model(dataset, l2=l2, seed=seed, verbose=verbose)
    model.meta.update(
        {
            "trained_on": "riot-matches",
            "n_games": dataset.n_games,
            "n_states": len(dataset),
            "source_dir": str(directory),
            "l2": l2,
        }
    )
    return model, report


def format_report(report: Dict[str, Any]) -> str:
    """Render a training report as plain text."""
    lines: List[str] = []
    fit = report.get("fit", {})
    lines.append(
        f"fit: {fit.get('n_samples', 0):,} rows over {report.get('train_games', 0):,} "
        f"games in {fit.get('iterations', 0)} Newton steps"
    )
    lines.append(f"side bias: {fit.get('side_bias', 0.0):+.4f} logit (blue-side edge)")

    test = report.get("test")
    if test:
        lines.append(
            f"held out {report.get('test_games', 0):,} games -> "
            f"log loss {test['log_loss']:.4f} | Brier {test['brier']:.4f} | "
            f"AUC {test['auc']:.4f} | acc {test['accuracy'] * 100:.1f}% | "
            f"ECE {test['ece'] * 100:.2f}%"
        )
    masked = report.get("test_live_masked")
    if masked:
        lines.append(
            f"live-masked (no XP/damage) -> log loss {masked['log_loss']:.4f} | "
            f"acc {masked['accuracy'] * 100:.1f}%"
        )
    gain = report.get("calibration", {}).get("gain")
    if gain is not None and gain == gain:
        verdict = "already calibrated" if gain < 0.002 else "would benefit from recalibration"
        lines.append(f"calibration: isotonic gain {gain:+.4f} nats ({verdict})")
    return "\n".join(lines)
