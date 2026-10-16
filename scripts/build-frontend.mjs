import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

// Convex supplies the selected deployment's public URL. Only the parent
// deployment process needs the administrative key, never the frontend build.
const env = { ...process.env };
delete env.CONVEX_DEPLOY_KEY;
const result = spawnSync(process.platform === "win32" ? "npm.cmd" : "npm", ["run", "build"], {
  cwd: fileURLToPath(new URL("../", import.meta.url)),
  env,
  stdio: "inherit",
});
if (result.error) {
  console.error(result.error.message);
}
process.exitCode = result.status ?? 1;
