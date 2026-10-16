import { getAuthUserId } from "@convex-dev/auth/server";
import { mutation, query } from "./_generated/server.js";
import { limitForTier, utcDateKey } from "./lib/limits.js";

export const ensureProfile = mutation({
  args: {},
  handler: async (ctx) => {
    const userId = await getAuthUserId(ctx);
    if (!userId) throw new Error("Sign in to continue.");

    const existing = await ctx.db
      .query("profiles")
      .withIndex("by_user", (q) => q.eq("userId", userId))
      .unique();
    if (existing) return existing._id;

    const now = Date.now();
    return await ctx.db.insert("profiles", {
      userId,
      tier: "free",
      createdAt: now,
      updatedAt: now,
      subscriptionStatus: "free",
    });
  },
});

export const get = query({
  args: {},
  handler: async (ctx) => {
    const userId = await getAuthUserId(ctx);
    if (!userId) return null;

    const [user, profile] = await Promise.all([
      ctx.db.get(userId),
      ctx.db
        .query("profiles")
        .withIndex("by_user", (q) => q.eq("userId", userId))
        .unique(),
    ]);
    const tier = profile?.tier ?? "free";
    const limit = limitForTier(tier);
    const now = Date.now();
    const today = utcDateKey(now);
    const usageToday = await ctx.db
      .query("usageDaily")
      .withIndex("by_user_date", (q) => q.eq("userId", userId).eq("date", today))
      .unique();

    const startDate = utcDateKey(now - 6 * 24 * 60 * 60 * 1000);
    const history = await ctx.db
      .query("usageDaily")
      .withIndex("by_user_date", (q) => q.eq("userId", userId).gte("date", startDate))
      .order("asc")
      .take(7);
    const requestsByDay = new Map(history.map((entry) => [entry.date, entry.requests]));
    const chart = Array.from({ length: 7 }, (_, index) => {
      const date = utcDateKey(now - (6 - index) * 24 * 60 * 60 * 1000);
      return { date, requests: requestsByDay.get(date) ?? 0 };
    });

    const activeKeys = await ctx.db
      .query("apiKeys")
      .withIndex("by_user", (q) => q.eq("userId", userId))
      .collect();

    return {
      user: {
        name: user?.name ?? "Developer",
        email: user?.email ?? null,
        image: user?.image ?? null,
      },
      tier,
      subscriptionStatus: profile?.subscriptionStatus ?? "free",
      limit,
      used: usageToday?.requests ?? 0,
      remaining: Math.max(0, limit - (usageToday?.requests ?? 0)),
      chart,
      activeKeys: activeKeys.filter((key) => key.revokedAt === undefined).length,
      billingCustomerReady: Boolean(profile?.stripeCustomerId),
    };
  },
});
