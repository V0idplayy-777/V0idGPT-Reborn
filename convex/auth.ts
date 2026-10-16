import GitHub from "@auth/core/providers/github";
import { convexAuth } from "@convex-dev/auth/server";

// GitHub OAuth keeps onboarding to a single, familiar step. Configure
// AUTH_GITHUB_ID and AUTH_GITHUB_SECRET in the Convex deployment before launch.
export const { auth, signIn, signOut, store, isAuthenticated } = convexAuth({
  providers: [GitHub],
});
