import type { AuthConfig } from "convex/server";

const authConfig: AuthConfig = {
  providers: [
    {
      domain: process.env.CONVEX_SITE_URL!,
      applicationID: "convex",
    },
  ],
};

export default authConfig;
