import { spawnSync } from "node:child_process";
import { fileURLToPath, pathToFileURL } from "node:url";
import path from "node:path";

const scriptPath = fileURLToPath(import.meta.url);
const repoRoot = path.resolve(path.dirname(scriptPath), "..");

/**
 * Deploy the Convex schema and functions to the deployment selected by the
 * server-side deploy key. Safe for app startup: the key is never passed to the
 * browser, printed, or copied into a file.
 */
export function provisionConvex({ required = false, development = false, build = false } = {}) {
  const deployKey = process.env.CONVEX_DEPLOY_KEY?.trim();
  if (!deployKey) {
    const message =
      "Convex auto-provision skipped: set CONVEX_DEPLOY_KEY in the server environment to deploy the schema and functions. Do not use a VITE_ prefix.";
    if (required) throw new Error(message);
    console.warn(message);
    return false;
  }

  const cliPath = path.join(repoRoot, "node_modules", "convex", "bin", "main.js");
  console.log("Provisioning Convex schema and functions for the deployment selected by CONVEX_DEPLOY_KEY…");
  const args = development
    ? ["dev", "--once", "--typecheck", "enable", "--codegen", "enable"]
    : ["deploy", "--yes", "--typecheck", "enable", "--codegen", "enable"];
  if (build) {
    if (development) throw new Error("Release builds cannot use development provisioning.");
    args.push("--cmd", "node scripts/build-frontend.mjs", "--cmd-url-env-var-name", "VITE_CONVEX_URL");
  }
  const result = spawnSync(
    process.execPath,
    [cliPath, ...args],
    {
      cwd: repoRoot,
      env: process.env,
      stdio: "inherit",
    },
  );

  if (result.error) throw result.error;
  if (result.status !== 0) {
    throw new Error(`Convex provisioning failed${result.status === null ? "" : ` (exit ${result.status})`}.`);
  }
  console.log("Convex schema and functions are up to date.");
  return true;
}

if (process.argv[1] && pathToFileURL(path.resolve(process.argv[1])).href === pathToFileURL(scriptPath).href) {
  try {
    provisionConvex({ required: process.argv.includes("--required"), build: process.argv.includes("--build") });
  } catch (error) {
    console.error(error instanceof Error ? error.message : "Convex provisioning failed.");
    process.exitCode = 1;
  }
}
