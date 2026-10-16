import { getAuthUserId } from "@convex-dev/auth/server";
import { v } from "convex/values";
import { anyApi } from "convex/server";
import { action, internalMutation, internalQuery } from "./_generated/server.js";

const internal: any = anyApi;

function stripeConfiguration() {
  const secretKey = process.env.STRIPE_SECRET_KEY;
  const priceId = process.env.STRIPE_PREMIUM_PRICE_ID;
  const siteUrl = process.env.SITE_URL?.replace(/\/$/, "");
  if (!secretKey || !priceId || !siteUrl) {
    throw new Error("Premium billing is not configured yet. Add the Stripe keys and SITE_URL to the Convex deployment.");
  }
  return { secretKey, priceId, siteUrl };
}

async function stripePost(secretKey: string, path: string, params: URLSearchParams) {
  const response = await fetch(`https://api.stripe.com/v1/${path}`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${secretKey}`,
      "Content-Type": "application/x-www-form-urlencoded",
    },
    body: params.toString(),
  });
  const body = (await response.json()) as { url?: string; error?: { message?: string } };
  if (!response.ok || !body.url) {
    // Keep provider response details out of user-facing errors and logs.
    throw new Error("Stripe could not start the billing session. Please try again later.");
  }
  return body.url;
}

export const getBillingInfo = internalQuery({
  args: { userId: v.id("users") },
  handler: async (ctx, args) => {
    const [user, profile] = await Promise.all([
      ctx.db.get(args.userId),
      ctx.db
        .query("profiles")
        .withIndex("by_user", (q) => q.eq("userId", args.userId))
        .unique(),
    ]);
    return {
      email: user?.email ?? undefined,
      tier: profile?.tier ?? "free",
      customerId: profile?.stripeCustomerId,
    };
  },
});

export const createCheckout = action({
  args: {},
  handler: async (ctx) => {
    const userId = await getAuthUserId(ctx);
    if (!userId) throw new Error("Sign in before upgrading.");

    const { secretKey, priceId, siteUrl } = stripeConfiguration();
    const billing = await ctx.runQuery(internal.billing.getBillingInfo, { userId });
    if (billing.tier === "premium") return { url: `${siteUrl}/?billing=active` };

    const params = new URLSearchParams({
      mode: "subscription",
      "line_items[0][price]": priceId,
      "line_items[0][quantity]": "1",
      success_url: `${siteUrl}/?billing=success`,
      cancel_url: `${siteUrl}/?billing=cancelled`,
      client_reference_id: userId,
      "metadata[userId]": userId,
      "subscription_data[metadata][userId]": userId,
    });
    if (billing.customerId) params.set("customer", billing.customerId);
    else if (billing.email) params.set("customer_email", billing.email);

    const url = await stripePost(secretKey, "checkout/sessions", params);
    return { url };
  },
});

export const createPortal = action({
  args: {},
  handler: async (ctx) => {
    const userId = await getAuthUserId(ctx);
    if (!userId) throw new Error("Sign in to manage billing.");
    const { secretKey, siteUrl } = stripeConfiguration();
    const billing = await ctx.runQuery(internal.billing.getBillingInfo, { userId });
    if (!billing.customerId) throw new Error("No billing account is linked to this profile yet.");
    const params = new URLSearchParams({
      customer: billing.customerId,
      return_url: `${siteUrl}/?billing=return`,
    });
    return { url: await stripePost(secretKey, "billing_portal/sessions", params) };
  },
});

export const applyStripeEvent = internalMutation({
  args: {
    eventId: v.string(),
    eventType: v.string(),
    userId: v.optional(v.id("users")),
    status: v.string(),
    customerId: v.optional(v.string()),
    subscriptionId: v.optional(v.string()),
  },
  handler: async (ctx, args) => {
    const seen = await ctx.db
      .query("stripeEvents")
      .withIndex("by_event", (q) => q.eq("eventId", args.eventId))
      .unique();
    if (seen) return { ok: true, duplicate: true };

    if (args.userId && (await ctx.db.get(args.userId))) {
      const existing = await ctx.db
        .query("profiles")
        .withIndex("by_user", (q) => q.eq("userId", args.userId!))
        .unique();
      const now = Date.now();
      const tier = args.status === "active" || args.status === "trialing" ? "premium" : "free";
      if (existing) {
        await ctx.db.patch(existing._id, {
          tier,
          subscriptionStatus: args.status,
          stripeCustomerId: args.customerId ?? existing.stripeCustomerId,
          stripeSubscriptionId: args.subscriptionId,
          updatedAt: now,
        });
      } else {
        await ctx.db.insert("profiles", {
          userId: args.userId,
          tier,
          subscriptionStatus: args.status,
          stripeCustomerId: args.customerId,
          stripeSubscriptionId: args.subscriptionId,
          createdAt: now,
          updatedAt: now,
        });
      }
    }

    await ctx.db.insert("stripeEvents", {
      eventId: args.eventId,
      type: args.eventType,
      processedAt: Date.now(),
    });
    return { ok: true, duplicate: false };
  },
});
