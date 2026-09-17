<#
.SYNOPSIS
    Build rift_oracle.exe on Windows.
.DESCRIPTION
    Installs the package with its dev extras, trains a model if none is
    bundled yet, and runs PyInstaller. The result lands in dist\rift_oracle.exe.
#>
[CmdletBinding()]
param(
    [int]$TrainGames = 6000,
    [switch]$Retrain
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

Write-Host "==> installing rift_oracle and build dependencies" -ForegroundColor Cyan
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

$model = Join-Path $root "rift_oracle\data\baseline_model.json"
if ($Retrain -or -not (Test-Path $model)) {
    Write-Host "==> training the bundled model ($TrainGames simulated games)" -ForegroundColor Cyan
    python -m rift_oracle train --games $TrainGames --out $model
} else {
    Write-Host "==> using the existing bundled model (pass -Retrain to refit)" -ForegroundColor DarkGray
}

Write-Host "==> running tests" -ForegroundColor Cyan
python -m pytest -q

Write-Host "==> building the executable" -ForegroundColor Cyan
python -m PyInstaller --noconfirm --clean packaging\rift_oracle.spec

$exe = Join-Path $root "dist\rift_oracle.exe"
if (-not (Test-Path $exe)) { throw "build finished but $exe is missing" }

$size = [math]::Round((Get-Item $exe).Length / 1MB, 1)
Write-Host "==> built $exe ($size MB)" -ForegroundColor Green

& $exe --version
& $exe demo --seed 1 --compact
Write-Host "==> smoke test passed" -ForegroundColor Green
