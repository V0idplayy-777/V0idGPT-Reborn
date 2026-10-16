import { v } from "convex/values";
import { internalMutation, internalQuery } from "./_generated/server.js";
import { ACTIVE_MODEL_ID, limitForTier, utcDateKey } from "./lib/limits.js";

export const checkApiKey = internalQuery({
  args: { keyHash: v.string() },
  handler: async (ctx, args) => {
    const key = await ctx.db
      .query("apiKeys")
      .withIndex("by_hash", (q) => q.eq("keyHash", args.keyHash))
      .unique();
    if (!key || key.revokedAt !== undefined) return null;

    const profile = await ctx.db
      .query("profiles")
      .withIndex("by_user", (q) => q.eq("userId", key.userId))
      .unique();
    const tier = profile?.tier ?? "free";
    const today = utcDateKey();
    const usage = await ctx.db
      .query("usageDaily")
      .withIndex("by_user_date", (q) => q.eq("userId", key.userId).eq("date", today))
      .unique();
    return { tier, limit: limitForTier(tier), used: usage?.requests ?? 0 };
  },
});

export const authorizeAndCount = internalMutation({
  args: { keyHash: v.string(), modelId: v.string() },
  handler: async (ctx, args) => {
    if (args.modelId !== ACTIVE_MODEL_ID) return { ok: false as const, reason: "model_not_found" as const };

    const key = await ctx.db
      .query("apiKeys")
      .withIndex("by_hash", (q) => q.eq("keyHash", args.keyHash))
      .unique();
    if (!key || key.revokedAt !== undefined) return { ok: false as const, reason: "unauthorized" as const };

    const profile = await ctx.db
      .query("profiles")
      .withIndex("by_user", (q) => q.eq("userId", key.userId))
      .unique();
    const tier = profile?.tier ?? "free";
    const limit = limitForTier(tier);
    const now = Date.now();
    const date = utcDateKey(now);
    const current = await ctx.db
      .query("usageDaily")
      .withIndex("by_user_date", (q) => q.eq("userId", key.userId).eq("date", date))
      .unique();

    if ((current?.requests ?? 0) >= limit) {
      return {
        ok: false as const,
        reason: "rate_limited" as const,
        used: current?.requests ?? limit,
        limit,
        tier,
        date,
      };
    }

    if (current) {
      await ctx.db.patch(current._id, {
        requests: current.requests + 1,
        updatedAt: now,
      });
    } else {
      await ctx.db.insert("usageDaily", {
        userId: key.userId,
        date,
        requests: 1,
        promptTokens: 0,
        completionTokens: 0,
        updatedAt: now,
      });
    }

    // Avoid writing the key document on every hot request while still keeping
    // last-used status reasonably fresh for the dashboard.
    if (key.lastUsedAt === undefined || now - key.lastUsedAt > 60_000) {
      await ctx.db.patch(key._id, { lastUsedAt: now });
    }

    return {
      ok: true as const,
      userId: key.userId,
      keyId: key._id,
      tier,
      limit,
      used: (current?.requests ?? 0) + 1,
      date,
    };
  },
});

export const recordTokens = internalMutation({
  args: {
    userId: v.id("users"),
    date: v.string(),
    promptTokens: v.number(),
    completionTokens: v.number(),
  },
  handler: async (ctx, args) => {
    const current = await ctx.db
      .query("usageDaily")
      .withIndex("by_user_date", (q) => q.eq("userId", args.userId).eq("date", args.date))
      .unique();
    if (!current) return { ok: false };
    await ctx.db.patch(current._id, {
      promptTokens: current.promptTokens + Math.max(0, Math.floor(args.promptTokens)),
      completionTokens: current.completionTokens + Math.max(0, Math.floor(args.completionTokens)),
      updatedAt: Date.now(),
    });
    return { ok: true };
  },
});
