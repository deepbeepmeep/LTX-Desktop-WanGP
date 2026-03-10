# Windows development setup for LTX Desktop

$ErrorActionPreference = "Stop"

function Ok($msg)   { Write-Host "✓ $msg" -ForegroundColor Green }
function Fail($msg) { Write-Host "✗ $msg" -ForegroundColor Red; exit 1 }

# ── Pre-checks ──────────────────────────────────────────────────────
if (-not (Get-Command node -ErrorAction SilentlyContinue)) {
    Fail "node not found — install Node.js 18+ from https://nodejs.org/"
}
if (-not (Get-Command pnpm -ErrorAction SilentlyContinue)) {
    Fail "pnpm not found — install with: corepack enable && corepack prepare pnpm --activate"
}
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Fail "uv not found — install with: powershell -ExecutionPolicy ByPass -c 'irm https://astral.sh/uv/install.ps1 | iex'"
}
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Fail "git not found — install Git from https://git-scm.com/download/win"
}
Ok "node $(node -v)"
Ok "pnpm $(pnpm --version)"
Ok "uv   $(uv --version)"
Ok "git  $(git --version)"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = Split-Path -Parent $ScriptDir

# ── Wan2GP checkout ─────────────────────────────────────────────────
Write-Host "`nEnsuring Wan2GP checkout..."
& (Join-Path $ScriptDir "ensure-wan2gp.ps1")
if ($LASTEXITCODE -ne 0) { Fail "Wan2GP checkout setup failed" }
Ok "Wan2GP checkout ready"

# ── pnpm install ────────────────────────────────────────────────────
Write-Host "`nInstalling Node dependencies..."
Set-Location $ProjectDir
pnpm install
if ($LASTEXITCODE -ne 0) { Fail "pnpm install failed" }
Ok "pnpm install complete"

# ── uv sync ─────────────────────────────────────────────────────────
Write-Host "`nSetting up Python backend venv..."
Set-Location (Join-Path $ProjectDir "backend")
uv sync --extra dev
if ($LASTEXITCODE -ne 0) { Fail "uv sync failed" }
Ok "uv sync complete"

& (Join-Path $ScriptDir "ensure-wan2gp.ps1") `
    -InstallPythonDeps `
    -PythonExe (Join-Path $ProjectDir "backend\.venv\Scripts\python.exe")
if ($LASTEXITCODE -ne 0) { Fail "Wan2GP dependency install failed" }
Ok "Wan2GP Python dependencies installed"

# Verify torch + CUDA
Write-Host "`nVerifying PyTorch CUDA support..."
try {
    & .venv\Scripts\python.exe -c "import torch; cuda=torch.cuda.is_available(); print(f'CUDA available: {cuda}'); print(f'GPU: {torch.cuda.get_device_name(0)}') if cuda else None"
} catch {
    Write-Host "  Could not verify PyTorch — this is OK if setup is still downloading." -ForegroundColor DarkYellow
}

# ── ffmpeg check ────────────────────────────────────────────────────
Write-Host ""
if (Get-Command ffmpeg -ErrorAction SilentlyContinue) {
    $ffmpegVer = (ffmpeg -version 2>&1 | Select-Object -First 1)
    Ok "ffmpeg found: $ffmpegVer"
} else {
    Write-Host "⚠  ffmpeg not found — install with: winget install ffmpeg" -ForegroundColor Yellow
    Write-Host "   (imageio-ffmpeg bundled binary will be used as fallback)" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host "  Setup complete! Run the app with:  pnpm dev" -ForegroundColor Cyan
Write-Host "  Debug mode (with debugpy):         pnpm dev:debug" -ForegroundColor Cyan
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
