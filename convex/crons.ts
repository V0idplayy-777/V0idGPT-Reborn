import { cronJobs } from "convex/server";
import { internal } from "./_generated/api.js";

const crons = cronJobs();

// Sweep hourly so a key is deactivated soon after it crosses 14 days without
// use. Requests also check the deadline inline, so the API never serves a key
// that is past it, even between runs.
crons.interval("deactivate idle api keys", { hours: 1 }, internal.apiKeys.deactivateIdleKeys, {});

export default crons;
