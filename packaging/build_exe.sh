#!/usr/bin/env bash
# Build the rift_oracle binary on Linux or macOS.
# PyInstaller does not cross-compile: run this on the platform you are
# targeting. For a Windows .exe, use packaging/build_exe.ps1 on Windows, or
# let the GitHub Actions workflow do it.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TRAIN_GAMES="${TRAIN_GAMES:-6000}"
MODEL="$ROOT/rift_oracle/data/baseline_model.json"

echo "==> installing rift_oracle and build dependencies"
python3 -m pip install --upgrade pip
python3 -m pip install -e ".[dev]"

if [[ "${RETRAIN:-0}" == "1" || ! -f "$MODEL" ]]; then
    echo "==> training the bundled model ($TRAIN_GAMES simulated games)"
    python3 -m rift_oracle train --games "$TRAIN_GAMES" --out "$MODEL"
else
    echo "==> using the existing bundled model (set RETRAIN=1 to refit)"
fi

echo "==> running tests"
python3 -m pytest -q

echo "==> building the executable"
python3 -m PyInstaller --noconfirm --clean packaging/rift_oracle.spec

BIN="$ROOT/dist/rift_oracle"
[[ -f "$BIN" ]] || { echo "build finished but $BIN is missing" >&2; exit 1; }

echo "==> built $BIN ($(du -h "$BIN" | cut -f1))"
"$BIN" --version
"$BIN" demo --seed 1 --compact
echo "==> smoke test passed"
