# Contributing to LTX Desktop

Thanks for taking the time to contribute!

## Getting started (development)

Prereqs:

- Node.js 18+
- `pnpm`
  Recommended Windows setup:

```bash
corepack enable
corepack prepare pnpm@latest --activate
pnpm -v
```

  Fallback if Corepack is unavailable:

```bash
npm install -g pnpm
pnpm -v
```

- `uv` (Python package manager)
- Python 3.12+
- Git

Setup:

```bash
pnpm setup:dev
```

On Windows, `pnpm setup:dev:win` installs Wan2GP Python dependencies into the backend venv so the desktop app can use the WanGP engine directly. If a repo-local `Wan2GP/` subfolder exists it is used first; otherwise, if `WANGP_ROOT` points to an existing checkout, that checkout is reused instead.

Run:

```bash
pnpm dev
```

Debug:

```bash
pnpm dev:debug
```

Typecheck:

```bash
pnpm typecheck
```

## What we accept right now

- Bug fixes and small improvements
- Documentation updates
- Small, targeted UI fixes

**Frontend policy:** the frontend is under active refactor. Please avoid large UI/state rewrites for now — open an issue first so we can align on the target direction.

## Proposing larger work

Before starting a larger change (especially frontend architecture/state), please open an issue with:

- The problem you’re trying to solve
- The proposed approach (1–2 paragraphs is fine)
- Scope (areas/files likely to change)
- Any UX or compatibility impact

Wait for maintainer alignment before investing in a major refactor.

## Checks

At minimum, run:

- Type checking:

```bash
pnpm typecheck
```

- Backend tests:

```bash
pnpm backend:test
```
