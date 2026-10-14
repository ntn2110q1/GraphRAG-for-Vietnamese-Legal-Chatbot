$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot
$pythonPath = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $pythonPath)) { throw "Run scripts\setup.ps1 first." }
& $pythonPath -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
