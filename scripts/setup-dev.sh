#!/usr/bin/env bash
# macOS development setup for LTX Desktop
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

ok()   { echo -e "${GREEN}✓${NC} $1"; }
fail() { echo -e "${RED}✗${NC} $1"; exit 1; }

# ── Pre-checks ──────────────────────────────────────────────────────
command -v node  >/dev/null 2>&1 || fail "node not found — install Node.js 18+"
command -v pnpm  >/dev/null 2>&1 || fail "pnpm not found — install with: corepack enable && corepack prepare pnpm --activate"
command -v uv    >/dev/null 2>&1 || fail "uv not found — install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
command -v git   >/dev/null 2>&1 || fail "git not found — install Git first"
ok "node $(node -v)"
ok "pnpm $(pnpm --version)"
ok "uv   $(uv --version)"
ok "git  $(git --version)"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# ── Wan2GP checkout ─────────────────────────────────────────
echo ""
echo "Ensuring Wan2GP checkout..."
bash "$SCRIPT_DIR/ensure-wan2gp.sh"
ok "Wan2GP checkout ready"

# ── pnpm install ────────────────────────────────────────────────────
echo ""
echo "Installing Node dependencies..."
cd "$PROJECT_DIR"
pnpm install
ok "pnpm install complete"

# ── uv sync ─────────────────────────────────────────────────────────
echo ""
echo "Setting up Python backend venv..."
cd "$PROJECT_DIR/backend"
uv sync --extra dev
ok "uv sync complete"

echo ""
echo "Wan2GP local bridge remains disabled on macOS; the checkout is cloned for repo parity and packaging."

# Verify torch + MPS
echo ""
echo "Verifying PyTorch MPS support..."
.venv/bin/python -c "import torch; mps=hasattr(torch.backends,'mps') and torch.backends.mps.is_available(); print(f'MPS available: {mps}')" || true

# ── ffmpeg check ────────────────────────────────────────────────────
echo ""
if command -v ffmpeg >/dev/null 2>&1; then
  ok "ffmpeg found: $(ffmpeg -version 2>&1 | head -1)"
else
  echo "⚠  ffmpeg not found — install with: brew install ffmpeg"
  echo "   (imageio-ffmpeg bundled binary will be used as fallback)"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Setup complete! Run the app with:  pnpm dev"
echo "  Debug mode (with debugpy):         pnpm dev:debug"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
