$ErrorActionPreference = "Stop"

$repo = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repo

if (-not (Get-Command pixi -ErrorAction SilentlyContinue)) {
    Write-Host "Installing Pixi..."
    Invoke-RestMethod https://pixi.sh/install.ps1 | Invoke-Expression
    $env:Path = "$HOME\.pixi\bin;$env:Path"
}

if (-not (Get-Command pixi -ErrorAction SilentlyContinue)) {
    throw "Pixi was installed but is not on PATH. Open a new PowerShell window and rerun this script."
}

pixi install
if ($LASTEXITCODE -ne 0) { throw "pixi install failed" }

pixi run install
if ($LASTEXITCODE -ne 0) { throw "editable package installation failed" }

pixi run python -c "import torch, sapien, mani_skill, warehouse_sort; print('imports: OK'); print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())"
if ($LASTEXITCODE -ne 0) { throw "Python import smoke test failed" }

Write-Host ""
Write-Host "Windows environment is ready."
Write-Host "Next: pixi run python il/download_demos.py"
Write-Host "Then: pixi run python il/train.py method=dp_rgb demo_dir=all"
Write-Host "Note: native Windows trains offline; run simulator evaluation in WSL or Colab."
