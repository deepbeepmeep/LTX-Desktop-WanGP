import { randomUUID } from 'crypto';
import { app } from 'electron';
import { isDev } from './config';
import { readAppState, writeAppState } from './app-state';

const ANALYTICS_ENDPOINT = 'https://ltx-desktop.lightricks.com/v2/ingest';
const REQUEST_TIMEOUT_MS = 5000;
const MAX_RETRIES = 3;
const RETRY_DELAYS_MS = [1000, 3000, 10000]

export function getAnalyticsState(): { analyticsEnabled: boolean; installationId: string } {
  const state = readAppState()
  return {
    analyticsEnabled: state.analyticsEnabled !== false,
    installationId: state.installationId ?? '',
  }
}

export function setAnalyticsEnabled(enabled: boolean): void {
  const state = readAppState()
  state.analyticsEnabled = enabled
  // Generate installationId on first enable; persist forever after
  if (enabled && !state.installationId) {
    state.installationId = randomUUID()
  }
  writeAppState(state)
}

function delay(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms))
}

function isRetryable(status: number): boolean {
  return status === 429 || status >= 500
}

async function sendWithRetry(
  url: string,
  options: RequestInit,
): Promise<void> {
  for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
    try {
      const controller = new AbortController()
      const timeout = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS)
      const response = await fetch(url, { ...options, signal: controller.signal })
      clearTimeout(timeout)

      if (response.ok || !isRetryable(response.status)) return
    } catch (err) {
      console.warn('[analytics] request attempt failed:', err)
    }

    if (attempt < MAX_RETRIES) {
      await delay(RETRY_DELAYS_MS[attempt])
    }
  }
}

export async function sendAnalyticsEvent(
  eventName: string,
  extraDetails?: Record<string, unknown> | null,
): Promise<void> {
  try {
    // Skip analytics in dev builds
    if (isDev) return;

    const state = readAppState()
    if (state.analyticsEnabled === false) return

    // Generate installationId on first send
    if (!state.installationId) {
      state.installationId = randomUUID()
      writeAppState(state)
    }

    const platformNames: Record<string, string> = { darwin: 'mac', win32: 'windows', linux: 'linux' }
    const platform = platformNames[process.platform] ?? process.platform
    const now = Date.now()

    const payload = {
      events: [
        {
          subject: eventName,
          eventId: randomUUID(),
          eventTimestamp: now,
          event: {
            app_version: app.getVersion(),
            device_timestamp: now,
            installation_id: state.installationId,
            platform,
            extra_details: extraDetails ? JSON.stringify(extraDetails) : null,
          },
        },
      ],
    }

    // Fire-and-forget with retries — never throws
    void sendWithRetry(ANALYTICS_ENDPOINT, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    })
  } catch (err) {
    console.error('[analytics] failed to send event:', err)
  }
}
