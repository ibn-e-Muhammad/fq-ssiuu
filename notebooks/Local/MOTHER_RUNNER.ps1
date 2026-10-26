# Run from any working directory. Installs Python through winget if necessary.
$ErrorActionPreference = 'Stop'
$runner = Join-Path $PSScriptRoot 'mother_runner.py'
$python = $null
foreach ($version in @('3.12', '3.11')) {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        $candidate = & py "-$version" -c 'import sys; print(sys.executable)' 2>$null
        if ($LASTEXITCODE -eq 0) { $python = $candidate; break }
    }
}
if (-not $python) {
    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
        throw 'Install Python 3.12 (python.org) or Windows App Installer, then rerun.'
    }
    & winget install --id Python.Python.3.12 --exact --scope user --silent --accept-package-agreements --accept-source-agreements
    if ($LASTEXITCODE -ne 0) { throw 'Python installation failed.' }
    $python = Join-Path $env:LOCALAPPDATA 'Programs\Python\Python312\python.exe'
    if (-not (Test-Path $python)) { throw 'Python installed in a custom location. Reopen PowerShell and rerun.' }
}
& $python $runner @args
exit $LASTEXITCODE
