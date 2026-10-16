export const FREE_DAILY_LIMIT = 5_000;
export const PREMIUM_DAILY_LIMIT = 500_000;
export const MAX_ACTIVE_KEYS = 10;
export const ACTIVE_MODEL_ID = "v0idgpt";

export function limitForTier(tier: "free" | "premium") {
  return tier === "premium" ? PREMIUM_DAILY_LIMIT : FREE_DAILY_LIMIT;
}

export function utcDateKey(now = Date.now()) {
  return new Date(now).toISOString().slice(0, 10);
}
