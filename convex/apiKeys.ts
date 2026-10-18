import { getAuthUserId } from "@convex-dev/auth/server";
import { v } from "convex/values";
import { createApiToken, sha256Hex } from "./lib/crypto.js";
import {
  INACTIVITY_DEACTIVATION_DAYS,
  effectiveDeactivatedAt,
  inactivityDeadline,
  isInactive,
  keyStatus,
  lastActivityAt,
} from "./lib/keys.js";
import { MAX_ACTIVE_KEYS } from "./lib/limits.js";
import { internalMutation, mutation, query } from "./_generated/server.js";

// Keep each sweep bounded so one cron run can never read an unbounded table.
const SWEEP_BATCH = 400;

export const list = query({
  args: {},
  handler: async (ctx) => {
    const userId = await getAuthUserId(ctx);
    if (!userId) throw new Error("Sign in to manage API keys.");

    const keys = await ctx.db
      .query("apiKeys")
      .withIndex("by_user_created", (q) => q.eq("userId", userId))
      .order("desc")
      .take(25);

    const now = Date.now();
    return keys.map((key) => ({
      id: key._id,
      name: key.name,
      prefix: key.prefix,
      createdAt: key.createdAt,
      lastUsedAt: key.lastUsedAt ?? null,
      revokedAt: key.revokedAt ?? null,
      // Status is computed, so a key that passed its deadline before the sweep
      // ran still reads as deactivated here and at the API edge.
      status: keyStatus(key, now),
      deactivatedAt: effectiveDeactivatedAt(key, now),
      deactivatedReason: key.deactivatedReason ?? (isInactive(key, now) ? "inactivity" : null),
      lastActivityAt: lastActivityAt(key),
      deactivatesAt: inactivityDeadline(key),
      inactivityDays: INACTIVITY_DEACTIVATION_DAYS,
    }));
  },
});

export const create = mutation({
  args: { name: v.string() },
  handler: async (ctx, args) => {
    const userId = await getAuthUserId(ctx);
    if (!userId) throw new Error("Sign in to create an API key.");

    const name = args.name.trim();
    if (!name || name.length > 48) {
      throw new Error("Give this key a name between 1 and 48 characters.");
    }

    const allKeys = await ctx.db
      .query("apiKeys")
      .withIndex("by_user", (q) => q.eq("userId", userId))
      .collect();
    // Deactivated keys still occupy a slot because they can be switched back on
    // at any time. Revoking is the way to free one up.
    if (allKeys.filter((key) => key.revokedAt === undefined).length >= MAX_ACTIVE_KEYS) {
      throw new Error(
        `You can have up to ${MAX_ACTIVE_KEYS} keys at once. Deactivated keys still count, so revoke one you no longer need.`,
      );
    }

    const token = createApiToken();
    const now = Date.now();
    const id = await ctx.db.insert("apiKeys", {
      userId,
      name,
      keyHash: await sha256Hex(token),
      prefix: `${token.slice(0, 13)}…`,
      createdAt: now,
      lastActiveAt: now,
    });

    // The raw secret is returned once and is never persisted in Convex.
    return { id, token, prefix: `${token.slice(0, 13)}…` };
  },
});

export const revoke = mutation({
  args: { id: v.id("apiKeys") },
  handler: async (ctx, args) => {
    const userId = await getAuthUserId(ctx);
    if (!userId) throw new Error("Sign in to revoke an API key.");

    const key = await ctx.db.get(args.id);
    if (!key || key.userId !== userId) throw new Error("API key not found.");
    if (key.revokedAt === undefined) {
      await ctx.db.patch(args.id, { revokedAt: Date.now() });
    }
    return { ok: true };
  },
});

/**
 * Turns an inactivity-deactivated key back on. Only the owner can call this,
 * and the idle countdown restarts from the moment of reactivation.
 */
export const activate = mutation({
  args: { id: v.id("apiKeys") },
  handler: async (ctx, args) => {
    const userId = await getAuthUserId(ctx);
    if (!userId) throw new Error("Sign in to activate an API key.");

    const key = await ctx.db.get(args.id);
    if (!key || key.userId !== userId) throw new Error("API key not found.");
    if (key.revokedAt !== undefined) {
      throw new Error("Revoked keys cannot be reactivated. Create a new key instead.");
    }

    const now = Date.now();
    if (key.deactivatedAt === undefined && !isInactive(key, now)) {
      return { ok: true as const, alreadyActive: true as const, deactivatesAt: inactivityDeadline(key) };
    }

    await ctx.db.patch(args.id, {
      deactivatedAt: undefined,
      deactivatedReason: undefined,
      reactivatedAt: now,
      // Restart the inactivity window from this moment.
      lastActiveAt: now,
    });
    return {
      ok: true as const,
      alreadyActive: false as const,
      deactivatesAt: now + INACTIVITY_DEACTIVATION_DAYS * 24 * 60 * 60 * 1000,
    };
  },
});

/**
 * Cron target: deactivates every live key whose last use is older than the
 * inactivity window. Requests also deactivate lazily, so this exists to keep
 * the dashboard honest for keys that are simply never called again.
 */
export const deactivateIdleKeys = internalMutation({
  args: {},
  handler: async (ctx) => {
    const now = Date.now();
    const cutoff = now - INACTIVITY_DEACTIVATION_DAYS * 24 * 60 * 60 * 1000;

    // Keys missing `lastActiveAt` sort before every timestamp, so they are
    // included here and get backfilled below.
    const candidates = await ctx.db
      .query("apiKeys")
      .withIndex("by_live_activity", (q) =>
        q.eq("revokedAt", undefined).eq("deactivatedAt", undefined).lt("lastActiveAt", cutoff),
      )
      .take(SWEEP_BATCH);

    let deactivated = 0;
    let backfilled = 0;
    for (const key of candidates) {
      if (isInactive(key, now)) {
        await ctx.db.patch(key._id, {
          // Record when the key actually went idle, not when the sweep ran.
          deactivatedAt: inactivityDeadline(key),
          deactivatedReason: "inactivity",
          lastActiveAt: lastActivityAt(key),
        });
        deactivated += 1;
      } else if (key.lastActiveAt !== lastActivityAt(key)) {
        // Older keys predate `lastActiveAt`; stamp them so later sweeps skip
        // them through the index instead of rereading them every hour.
        await ctx.db.patch(key._id, { lastActiveAt: lastActivityAt(key) });
        backfilled += 1;
      }
    }

    return { scanned: candidates.length, deactivated, backfilled };
  },
});
