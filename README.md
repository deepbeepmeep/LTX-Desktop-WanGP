# LTX Desktop

LTX Desktop is an open-source desktop app for generating videos with LTX models - locally on supported Windows NVIDIA GPUs and on Linux source/dev setups backed by WanGP, with an API mode for unsupported hardware and macOS.

> **Status: Beta.** Expect breaking changes.
> Frontend architecture is under active refactor; large UI PRs may be declined for now (see [`CONTRIBUTING.md`](docs/CONTRIBUTING.md)).

**This LTX Desktop fork powered by WanGP reduces the VRAM requirements from 32 GB to 6 GB.**

Check the WanGP repo for more information (docs, Discord, and more): https://github.com/deepbeepmeep/Wan2GP

## Windows WanGP Quick Start

Use one of these two setup paths for local WanGP-backed generation on Windows.

Before running any `pnpm` command, make sure `pnpm` is installed and available in `PATH`.

Prerequisites:

- Node.js 18+ from https://nodejs.org/
- `pnpm`, usually enabled with Corepack:

```bash
corepack enable
corepack prepare pnpm@latest --activate
pnpm -v
```

If `corepack` is unavailable but Node.js is already installed, you can install `pnpm` with:

```bash
npm install -g pnpm
pnpm -v
```

### 1. Wan2GP not installed yet

`pnpm setup:dev:win` clones `Wan2GP/` into this repository, installs the backend dependencies, and prepares a plug-and-play local setup.

```bash
pnpm setup:dev:win
pnpm dev
```

The desktop backend will prefer the repo-local checkout at `.\Wan2GP`.

### 2. Wan2GP already installed elsewhere

If you already have a working Wan2GP checkout and want LTX Desktop to reuse it, do not keep a local `.\Wan2GP` subfolder in this repo.

`WANGP_ROOT` is an environment variable that must point to the root folder of your existing Wan2GP checkout, meaning the folder that contains `wgp.py`.

Examples:

- Windows `cmd.exe`:

```bat
set WANGP_ROOT=D:\Wan2GP
```

- Windows PowerShell:

```powershell
$env:WANGP_ROOT = "D:\Wan2GP"
```

Set `WANGP_ROOT` before running setup. `pnpm setup:dev:win` will then reuse that checkout and install its `requirements.txt` into the LTX Desktop backend venv.

```bash
set WANGP_ROOT=D:\Wan2GP
pnpm setup:dev:win
pnpm dev
```

If you prefer the manual path instead of `pnpm setup:dev:win`, install the external Wan2GP requirements into the backend venv yourself after `uv sync`:

```bash
set WANGP_ROOT=D:\Wan2GP
pnpm install
cd backend
uv sync --extra dev
uv pip install --python .venv\Scripts\python.exe -r %WANGP_ROOT%\requirements.txt
cd ..
pnpm dev
```

The backend still runs in LTX Desktop's own `backend/.venv` unless you explicitly override it with `LTX_BACKEND_PYTHON`.

If both are present, LTX Desktop uses the local `.\Wan2GP` checkout first and falls back to `WANGP_ROOT` only when no local subfolder is available.

If you want `WANGP_ROOT` to persist across new Windows terminals, you can set it permanently with:

- `cmd.exe`:

```bat
setx WANGP_ROOT D:\Wan2GP
```

- PowerShell:

```powershell
[Environment]::SetEnvironmentVariable("WANGP_ROOT", "D:\Wan2GP", "User")
```

## Linux WanGP Quick Start

Linux support in this fork currently targets running from source/dev with WanGP. This README does not claim an official packaged Linux release for the fork.

Prerequisites:

- Node.js 18+
- `pnpm`
- `uv`
- `git`
- NVIDIA GPU with CUDA support
- `ffmpeg`

If you already have a Wan2GP checkout, point `WANGP_ROOT` to it before setup:

```bash
export WANGP_ROOT=/path/to/Wan2GP
pnpm setup:dev:linux
pnpm dev
```

If `WANGP_ROOT` is not set, `pnpm setup:dev:linux` will prepare a repo-local `Wan2GP/` checkout for you.

<p align="center">
  <img src="images/gen-space.png" alt="Gen Space" width="70%">
</p>

<p align="center">
  <img src="images/video-editor.png" alt="Video Editor" width="70%">
</p>

<p align="center">
  <img src="images/timeline-gap-fill.png" alt="Timeline gap fill" width="70%">
</p>

## Features

- Text-to-video generation
- Image-to-video generation
- Audio-to-video generation
- Video edit generation (Retake)
- Video Editor Interface
- Video Editing Projects

## Local vs API mode

| Platform / hardware | Generation mode | Notes |
| --- | --- | --- |
| Windows + CUDA GPU with **as low as 6 GB VRAM with WanGP** | Local generation | Downloads model weights locally |
| Windows (no CUDA, low VRAM, or unknown VRAM) | API-only | **LTX API key required** |
| macOS (Apple Silicon builds) | API-only | **LTX API key required** |
| Linux + CUDA GPU + WanGP checkout | Local generation | Source/dev setup supported in this fork |
| Linux without WanGP bridge | API-only | **LTX API key required** |

In API-only mode, available resolutions/durations may be limited to what the API supports.

## System requirements

### Windows (local generation)

- Windows 10/11 (x64)
- NVIDIA GPU with CUDA support and as low as 6 GB VRAM with WanGP
- 16 GB+ RAM (32 GB recommended)
- Plenty of free disk space for model weights and outputs

### macOS (API-only)

- Apple Silicon (arm64)
- macOS 13+ (Ventura)
- Stable internet connection

### Linux (source/dev with WanGP)

- Modern x64 Linux distribution
- NVIDIA GPU with CUDA support
- `ffmpeg`
- A Wan2GP checkout available locally or via `WANGP_ROOT`

## Install

1. Windows: download the latest installer from GitHub Releases: [Releases](../../releases)
2. Linux: use the source/dev setup described in **Linux WanGP Quick Start** or **Development (quickstart)**
3. Launch **LTX Desktop** and complete first-run setup

## First run & data locations

LTX Desktop stores app data (settings, models, logs) in:

- **Windows:** `%LOCALAPPDATA%\LTXDesktop\`
- **macOS:** `~/Library/Application Support/LTXDesktop/`
- **Linux:** `$XDG_DATA_HOME/LTXDesktop/` or `~/.local/share/LTXDesktop/`

Model weights are downloaded into the `models/` subfolder (this can be large and may take time).

On first launch you may be prompted to review or accept model license terms (license text is fetched from Hugging Face and requires internet).

Text encoding: to generate videos you must configure text encoding:

- **LTX API key** (cloud text encoding) - **text encoding via the API is completely free** and highly recommended to speed up inference and save memory. Generate a free API key at the [LTX Console](https://console.ltx.video/). [Read more](https://ltx.io/model/model-blog/ltx-2-better-control-for-real-workflows).
- **Local Text Encoder** (extra download; enables fully local operation on supported Windows and Linux WanGP setups) - if you do not wish to generate an API key, you can encode text locally via the settings menu.

## API keys, cost, and privacy

### LTX API key

The LTX API is used for:

- **Cloud text encoding and prompt enhancement** - **free**; text encoding is highly recommended to speed up inference and save memory
- API-based video generations (required on macOS and on unsupported Windows/Linux hardware) - paid
- Retake - paid

An LTX API key is required in API-only mode, but optional on Windows and Linux local WanGP mode if you enable the Local Text Encoder.

Generate a free API key at the [LTX Console](https://console.ltx.video/). Text encoding is free; video generation API usage is paid. [Read more](https://ltx.io/model/model-blog/ltx-2-better-control-for-real-workflows).

When you use API-backed features, prompts and media inputs are sent to the API service. Your API key is stored locally in your app data folder - treat it like a secret.

### fal API key (optional)

Used for Z Image Turbo text-to-image generation in API mode. When enabled, image generation requests are sent to fal.ai.

Create an API key in the [fal dashboard](https://fal.ai/dashboard/keys).

### Gemini API key (optional)

Used for AI prompt suggestions. When enabled, prompt context and frames may be sent to Google Gemini.

## Architecture

LTX Desktop is split into three main layers:

- **Renderer (`frontend/`)**: TypeScript + React UI.
  Calls the local backend over HTTP at `http://localhost:8000`.
  Talks to Electron via the preload bridge (`window.electronAPI`).
- **Electron (`electron/`)**: TypeScript main process + preload.
  Owns app lifecycle and OS integration (file dialogs, native export via ffmpeg, starting/managing the Python backend).
  Security: renderer is sandboxed (`contextIsolation: true`, `nodeIntegration: false`).
- **Backend (`backend/`)**: Python + FastAPI local server.
  Orchestrates generation, model downloads, and GPU execution.
  Calls external APIs only when API-backed features are used.

```mermaid
graph TD
  UI["Renderer (React + TS)"] -->|HTTP: localhost:8000| BE["Backend (FastAPI + Python)"]
  UI -->|IPC via preload: window.electronAPI| EL["Electron main (TS)"]
  EL --> OS["OS integration (files, dialogs, ffmpeg, process mgmt)"]
  BE --> GPU["Local models + GPU (when supported)"]
  BE --> EXT["External APIs (only for API-backed features)"]
  EL --> DATA["App data folder (settings/models/logs)"]
  BE --> DATA
```

## Development (quickstart)

Prereqs:

- Node.js
- `uv` (Python package manager)
- Python 3.12+
- Git

Setup:

```bash
# macOS
pnpm setup:dev:mac

# Linux
pnpm setup:dev:linux

# Windows
pnpm setup:dev:win
```

On Windows and Linux, the WanGP-backed path uses the LTX Desktop backend venv plus either a repo-local `Wan2GP/` checkout or an existing checkout pointed to by `WANGP_ROOT`. A repo-local `Wan2GP/` remains directly usable on its own if you want to run Wan2GP from the subfolder.

Run:

```bash
pnpm dev
```

Debug:

```bash
pnpm dev:debug
```

`dev:debug` starts Electron with inspector enabled and starts the Python backend with `debugpy`.

Typecheck:

```bash
pnpm typecheck
```

Backend tests:

```bash
pnpm backend:test
```

Building installers:
- See [`INSTALLER.md`](docs/INSTALLER.md)

## Telemetry

LTX Desktop collects minimal, anonymous usage analytics (app version, platform, and a random installation ID) to help prioritize development. No personal information or generated content is collected. Analytics is enabled by default and can be disabled in **Settings > General > Anonymous Analytics**. See [`TELEMETRY.md`](docs/TELEMETRY.md) for details.

## Docs

- [`INSTALLER.md`](docs/INSTALLER.md) - building installers
- [`TELEMETRY.md`](docs/TELEMETRY.md) - telemetry and privacy
- [`backend/architecture.md`](backend/architecture.md) - backend architecture
- [`backend/WANGP_BACKEND.md`](backend/WANGP_BACKEND.md) - WanGP bridge configuration

## Contributing

See [`CONTRIBUTING.md`](docs/CONTRIBUTING.md).

## License

Apache-2.0 - see [`LICENSE.txt`](LICENSE.txt).

Third-party notices (including model licenses/terms): [`NOTICES.md`](NOTICES.md).

Model weights are downloaded separately and may be governed by additional licenses/terms.
