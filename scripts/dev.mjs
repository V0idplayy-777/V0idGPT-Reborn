import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { provisionConvex } from "./provision-convex.mjs";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

try {
  if (process.env.V0ID_SKIP_CONVEX_AUTOPROVISION !== "1") {
    provisionConvex({ required: false });
  } else {
    console.log("Convex auto-provision skipped by V0ID_SKIP_CONVEX_AUTOPROVISION=1.");
  }
} catch (error) {
  console.error(error instanceof Error ? error.message : "Convex provisioning failed.");
  process.exit(1);
}

// Never pass the privileged deploy key into the Vite/web-server process.
const viteEnvironment = { ...process.env };
delete viteEnvironment.CONVEX_DEPLOY_KEY;

const vitePath = path.join(repoRoot, "node_modules", "vite", "bin", "vite.js");
const vite = spawn(process.execPath, [vitePath, "--host", "0.0.0.0", ...process.argv.slice(2)], {
  cwd: repoRoot,
  env: viteEnvironment,
  stdio: "inherit",
});

vite.on("error", (error) => {
  console.error(`Could not start Vite: ${error.message}`);
  process.exitCode = 1;
});
vite.on("exit", (code, signal) => {
  if (signal) process.kill(process.pid, signal);
  else process.exitCode = code ?? 1;
});

for (const signal of ["SIGINT", "SIGTERM"]) {
  process.on(signal, () => {
    if (!vite.killed) vite.kill(signal);
  });
}
