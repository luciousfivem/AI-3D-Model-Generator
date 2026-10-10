<#
  Forge3D installer for Windows 10/11.

  Usage (from the project folder):
      powershell -ExecutionPolicy Bypass -File scripts\install.ps1

  Options:
      -CpuOnly       install the CPU build of PyTorch even if an NVIDIA GPU is present
      -SkipModels    do not download the default AI model now (it downloads on first use instead)
      -CudaIndex     PyTorch CUDA build to use: cu126 (default), cu128, cu124 or cu121.
                     The default works with any driver from the last couple of years; use cu121 only on very old drivers.
#>
param(
    [switch]$CpuOnly,
    [switch]$SkipModels,
    [string]$CudaIndex = "cu126"
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

function Step($message) { Write-Host "`n==> $message" -ForegroundColor Cyan }

# ---------------------------------------------------------------- 1. Python
# 3.12 has the widest wheel coverage for the libraries we use; 3.10-3.13 all work.
Step "Looking for Python 3.10 - 3.13"
$pyVersion = $null
foreach ($candidate in "3.12", "3.11", "3.13", "3.10") {
    try {
        & py "-$candidate" --version 2>$null | Out-Null
        if ($LASTEXITCODE -eq 0) { $pyVersion = $candidate; break }
    } catch { }
}
if (-not $pyVersion) {
    throw "No suitable Python was found. Install one with:`n    winget install Python.Python.3.12`nthen run this script again."
}
Write-Host "Using Python $pyVersion"

# ---------------------------------------------------------------- 2. virtual environment
Step "Creating the virtual environment (.venv)"
if (-not (Test-Path ".venv\Scripts\python.exe")) { & py "-$pyVersion" -m venv .venv }
$python = (Resolve-Path ".venv\Scripts\python.exe").Path
& $python -m pip install --upgrade pip

# ---------------------------------------------------------------- 3. PyTorch
$hasNvidia = $false
if (-not $CpuOnly) {
    try {
        $gpu = & nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>$null
        if ($LASTEXITCODE -eq 0 -and $gpu) { $hasNvidia = $true; Write-Host "Found GPU: $gpu" }
    } catch { }
}
if ($hasNvidia) {
    Step "Installing PyTorch with CUDA ($CudaIndex build, a large download)"
    & $python -m pip install torch torchvision --index-url "https://download.pytorch.org/whl/$CudaIndex"
} else {
    Write-Warning "No NVIDIA GPU/driver found (or -CpuOnly given): installing the CPU build of PyTorch. AI models will be very slow."
    & $python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
}

# ---------------------------------------------------------------- 4. everything else
Step "Installing Forge3D and the AI libraries"
& $python -m pip install -r requirements.txt -r requirements-ai.txt
# hy3dgen's own requirement list drags in a web demo (gradio, fastapi...) that this app does not use.
& $python -m pip install --no-deps hy3dgen==2.0.2
& $python -m pip install --no-deps -e .

# ---------------------------------------------------------------- 5. check + models
Step "Checking the installation"
& $python -m forge3d doctor

if (-not $SkipModels) {
    Step "Downloading the default AI model (about 4.7 GB, one time)"
    & $python -m forge3d download
}

Write-Host "`nDone. Start Forge3D with scripts\Forge3D.bat" -ForegroundColor Green
Write-Host "Optional: scripts\create_shortcut.ps1 puts a shortcut on your desktop."
