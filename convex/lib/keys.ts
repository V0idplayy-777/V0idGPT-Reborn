// Shared API key lifecycle rules.
//
// A key that has not been used for INACTIVITY_DEACTIVATION_DAYS is deactivated
// automatically. Deactivation is reversible: the owner reactivates the key from
// the dashboard. Revocation stays permanent.

export const INACTIVITY_DEACTIVATION_DAYS = 14;
export const DAY_MS = 24 * 60 * 60 * 1000;
export const INACTIVITY_DEACTIVATION_MS = INACTIVITY_DEACTIVATION_DAYS * DAY_MS;

export type KeyLifecycleFields = {
  createdAt: number;
  lastUsedAt?: number;
  lastActiveAt?: number;
  deactivatedAt?: number;
  revokedAt?: number;
};

export type KeyStatus = "active" | "deactivated" | "revoked";

/**
 * The moment the idle countdown last restarted: the latest of the key's last
 * use, its last manual activation, or its creation. `lastActiveAt` is written
 * on use and on reactivation; older keys predate the field, so fall back to
 * `lastUsedAt` and then `createdAt`.
 */
export function lastActivityAt(key: KeyLifecycleFields) {
  return Math.max(key.lastActiveAt ?? 0, key.lastUsedAt ?? 0, key.createdAt);
}

/** Timestamp at which an unused key becomes (or became) inactive. */
export function inactivityDeadline(key: KeyLifecycleFields) {
  return lastActivityAt(key) + INACTIVITY_DEACTIVATION_MS;
}

/**
 * True when the key has passed the inactivity window. Checked on every request
 * so enforcement never depends on the sweep cron having already run.
 */
export function isInactive(key: KeyLifecycleFields, now = Date.now()) {
  return now >= inactivityDeadline(key);
}

/** Milliseconds left before automatic deactivation (0 once the window closes). */
export function msUntilDeactivation(key: KeyLifecycleFields, now = Date.now()) {
  return Math.max(0, inactivityDeadline(key) - now);
}

/**
 * Effective status, treating a key that is past its inactivity deadline as
 * already deactivated even if the cron has not stamped it yet.
 */
export function keyStatus(key: KeyLifecycleFields, now = Date.now()): KeyStatus {
  if (key.revokedAt !== undefined) return "revoked";
  if (key.deactivatedAt !== undefined || isInactive(key, now)) return "deactivated";
  return "active";
}

/**
 * When a key was (or should be recorded as) deactivated: the moment the
 * inactivity window closed, rather than the moment the sweep noticed.
 */
export function effectiveDeactivatedAt(key: KeyLifecycleFields, now = Date.now()) {
  if (key.deactivatedAt !== undefined) return key.deactivatedAt;
  return isInactive(key, now) ? inactivityDeadline(key) : null;
}

/** The exact clicks that turn a deactivated key back on. */
export function reactivationSteps(dashboardUrl?: string | null) {
  return [
    `Sign in to the v0id dashboard${dashboardUrl ? ` (${dashboardUrl})` : ""} as the owner of this key.`,
    "Go to the API keys panel.",
    'Find this key in the list and press the "Activate" button on it.',
  ];
}

/**
 * Error text returned to any caller using a key that inactivity switched off.
 * It states the cause and the steps to turn the key back on.
 */
export function deactivatedKeyMessage(options: {
  days?: number;
  dashboardUrl?: string | null;
  prefix?: string | null;
} = {}) {
  const days = options.days ?? INACTIVITY_DEACTIVATION_DAYS;
  const keyLabel = options.prefix ? ` (${options.prefix})` : "";
  const where = options.dashboardUrl ? ` at ${options.dashboardUrl}` : "";
  return (
    `This API key${keyLabel} is deactivated because it went ${days} days without being used. ` +
    `To switch it back on, the owner of the key signs in to the v0id dashboard${where}, opens the API keys panel, ` +
    `finds this key, and presses the "Activate" button shown on it. ` +
    "The same key then works again immediately — no new key and no code changes. " +
    `Each request resets the ${days}-day inactivity timer.`
  );
}
