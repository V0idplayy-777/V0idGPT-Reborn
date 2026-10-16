import { loadModel } from "./model.mjs";

const model = await loadModel();
const report = model.verify();
for (const c of report.checks) console.log("[" + c.status.toUpperCase().padEnd(4) + "] " + c.name + ": " + c.detail);
console.log(report.ok ? "OK" : "FAILED");
if (typeof process !== "undefined" && !report.ok) process.exitCode = 1;
