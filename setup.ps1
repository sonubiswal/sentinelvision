# SentinelVision -- Windows bootstrap (PowerShell 7+).
# Usage:  ./setup.ps1            # dev install (tests + lint)
#         ./setup.ps1 -Vision    # also install the ultralytics runtime

param(
    [switch]$Vision,
    [string]$Python = "py",
    [string]$Venv = ".venv"
)

$ErrorActionPreference = "Stop"

Write-Host "==> Creating virtual environment in $Venv" -ForegroundColor Cyan
& $Python -3.11 -m venv $Venv 2>$null
if ($LASTEXITCODE -ne 0) { & $Python -m venv $Venv }

$python = Join-Path $Venv "Scripts/python.exe"

Write-Host "==> Upgrading pip" -ForegroundColor Cyan
& $python -m pip install --upgrade pip wheel

$extras = "dev"
if ($Vision) { $extras = "dev,vision,runtime" }

Write-Host "==> Installing sentinelvision[$extras]" -ForegroundColor Cyan
& $python -m pip install -e ".[$extras]"

Write-Host "==> Running the test suite" -ForegroundColor Cyan
& $python -m pytest

Write-Host "==> Linting" -ForegroundColor Cyan
& $python -m ruff check .

Write-Host ""
Write-Host "Done. Try:" -ForegroundColor Green
Write-Host "  $python -m sentinelvision --list-classes"
Write-Host "  $python -m sentinelvision --source 0 --show"
