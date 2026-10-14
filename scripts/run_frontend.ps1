$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot
$pythonPath = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $pythonPath)) { throw "Run scripts\setup.ps1 first." }
& $pythonPath -m streamlit run frontend/app.py --server.address 127.0.0.1 --server.port 8501
