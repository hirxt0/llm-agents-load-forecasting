$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$projectPython = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $projectPython)) {
    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCommand) {
        & $pythonCommand.Source -m venv .venv
    } elseif (Test-Path -LiteralPath 'C:/ProgramData/anaconda3/python.exe') {
        & 'C:/ProgramData/anaconda3/python.exe' -m venv .venv
    } else {
        & py -3 -m venv .venv
    }
    if ($LASTEXITCODE -ne 0) { throw 'Python environment creation failed.' }
}
& $projectPython -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
if (-not (Test-Path -LiteralPath '.env')) {
    Copy-Item -LiteralPath '.env.example' -Destination '.env'
}
Write-Host 'Ready. Run: ./run.ps1 --dry-run --split dev --limit 2'
