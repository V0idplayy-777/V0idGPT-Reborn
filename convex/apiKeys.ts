import { getAuthUserId } from "@convex-dev/auth/server";
import { v } from "convex/values";
import { createApiToken, sha256Hex } from "./lib/crypto.js";
import { MAX_ACTIVE_KEYS } from "./lib/limits.js";
import { mutation, query } from "./_generated/server.js";

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

    return keys.map((key) => ({
      id: key._id,
      name: key.name,
      prefix: key.prefix,
      createdAt: key.createdAt,
      lastUsedAt: key.lastUsedAt ?? null,
      revokedAt: key.revokedAt ?? null,
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
    if (allKeys.filter((key) => key.revokedAt === undefined).length >= MAX_ACTIVE_KEYS) {
      throw new Error(`You can have up to ${MAX_ACTIVE_KEYS} active API keys.`);
    }

    const token = createApiToken();
    const now = Date.now();
    const id = await ctx.db.insert("apiKeys", {
      userId,
      name,
      keyHash: await sha256Hex(token),
      prefix: `${token.slice(0, 13)}…`,
      createdAt: now,
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
