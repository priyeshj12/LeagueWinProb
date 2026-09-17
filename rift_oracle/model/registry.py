"""Finding and loading the win-probability model.

Lookup order, first hit wins:

1. an explicit ``--model`` path,
2. ``$RIFT_ORACLE_MODEL``,
3. the user's own trained model in the app directory,
4. the model bundled with the package (or inside the PyInstaller executable).

If none of those exist the caller gets a clear instruction to run
``rift-oracle train`` rather than silently scoring with zero weights, which
would report 50% for every state and look like it was working.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional, Tuple

from rift_oracle.config import RiftOracleError, bundled_data_dir, models_dir
from rift_oracle.model.gam import AdditiveWinModel

DEFAULT_MODEL_NAME = "winprob_model.json"
BUNDLED_MODEL_NAME = "baseline_model.json"


def user_model_path() -> Path:
    return models_dir() / DEFAULT_MODEL_NAME


def bundled_model_path() -> Path:
    return bundled_data_dir() / BUNDLED_MODEL_NAME


def find_model_path(explicit: Optional[str] = None) -> Optional[Path]:
    """Resolve which model file to use, or ``None`` when there is none."""
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise RiftOracleError(f"model file not found: {path}")
        return path

    env = os.environ.get("RIFT_ORACLE_MODEL")
    if env:
        path = Path(env).expanduser()
        if path.is_file():
            return path

    for candidate in (user_model_path(), bundled_model_path()):
        if candidate.is_file():
            return candidate
    return None


def load_model(explicit: Optional[str] = None) -> Tuple[AdditiveWinModel, Path]:
    """Load the model and report where it came from."""
    path = find_model_path(explicit)
    if path is None:
        raise RiftOracleError(
            "no win-probability model is available.\n"
            "  Train one (about a minute, no API key or network needed):\n"
            "      rift-oracle train\n"
            "  Or train on real matches you have harvested:\n"
            "      rift-oracle harvest --riot-id 'You#TAG' --count 200\n"
            "      rift-oracle train --data matches"
        )
    try:
        model = AdditiveWinModel.load(path)
    except (ValueError, KeyError, OSError) as exc:
        raise RiftOracleError(f"could not read the model at {path}: {exc}") from exc
    return model, path


def save_user_model(model: AdditiveWinModel) -> Path:
    return model.save(user_model_path())


def model_info(path: Path) -> dict:
    """Metadata block from a saved model, for ``rift-oracle doctor``."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload.get("meta", {})
