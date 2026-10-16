import { authTables } from "@convex-dev/auth/server";
import { defineSchema, defineTable } from "convex/server";
import { v } from "convex/values";

export default defineSchema({
  ...authTables,
  profiles: defineTable({
    userId: v.id("users"),
    tier: v.union(v.literal("free"), v.literal("premium")),
    createdAt: v.number(),
    updatedAt: v.number(),
    stripeCustomerId: v.optional(v.string()),
    stripeSubscriptionId: v.optional(v.string()),
    subscriptionStatus: v.optional(v.string()),
  }).index("by_user", ["userId"]),
  apiKeys: defineTable({
    userId: v.id("users"),
    name: v.string(),
    keyHash: v.string(),
    prefix: v.string(),
    createdAt: v.number(),
    lastUsedAt: v.optional(v.number()),
    revokedAt: v.optional(v.number()),
  })
    .index("by_hash", ["keyHash"])
    .index("by_user", ["userId"])
    .index("by_user_created", ["userId", "createdAt"]),
  usageDaily: defineTable({
    userId: v.id("users"),
    date: v.string(),
    requests: v.number(),
    promptTokens: v.number(),
    completionTokens: v.number(),
    updatedAt: v.number(),
  }).index("by_user_date", ["userId", "date"]),
  stripeEvents: defineTable({
    eventId: v.string(),
    type: v.string(),
    processedAt: v.number(),
  }).index("by_event", ["eventId"]),
});
