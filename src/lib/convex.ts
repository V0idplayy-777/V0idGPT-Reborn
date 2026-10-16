import { ConvexReactClient } from "convex/react";

const configuredUrl = import.meta.env.VITE_CONVEX_URL as string | undefined;
export const convexUrl = (configuredUrl || "https://tough-pelican-388.convex.cloud").replace(/\/$/, "");
export const convexSiteUrl = (
  (import.meta.env.VITE_CONVEX_SITE_URL as string | undefined) ||
  convexUrl.replace(/\.convex\.cloud$/i, ".convex.site")
).replace(/\/$/, "");

export const convex = new ConvexReactClient(convexUrl);
