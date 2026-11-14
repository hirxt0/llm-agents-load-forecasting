$ErrorActionPreference = 'Stop'
$projectPython = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $projectPython)) {
    throw 'Run setup.ps1 first.'
}
& $projectPython (Join-Path $PSScriptRoot 'main.py') @args
exit $LASTEXITCODE
