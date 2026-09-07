# Telemetry

LTX Desktop collects minimal, anonymous telemetry to help the team understand how the app is used and prioritize development.
No personal information, generated content, prompts, file paths, or IP-derived location data is collected or stored.

## Opting out

Analytics is enabled by default. You can disable it at any time in **Settings > General > Anonymous Analytics**. When disabled, no events are sent.

To disable telemetry before the first launch, create an `app_state.json` file in the app data folder with the following content:

```json
{ "analyticsEnabled": false }
```

App data folder locations:

- **Windows:** `%LOCALAPPDATA%\LTXDesktop\`
- **macOS:** `~/Library/Application Support/LTXDesktop/`
- **Linux:** `$XDG_DATA_HOME/LTXDesktop/` (default: `~/.local/share/LTXDesktop/`)

Your preference is respected immediately — no restart required.

## Implementation

The telemetry implementation is fully contained in [`electron/analytics.ts`](../electron/analytics.ts). Events are sent to an ingestion endpoint over HTTPS. No third-party analytics SDKs are used.
