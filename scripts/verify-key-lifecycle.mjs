// Verifies the API key inactivity policy without a live Convex deployment.
//
// It loads the real Convex handlers (convex/apiKeys.ts and convex/api.ts) and
// runs them against an in-memory database that mimics Convex's index queries,
// Convex value ordering (undefined sorts first), and schema validation.
//
//   npm run keys:verify

import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { registerHooks } from "node:module";
import { dirname, resolve as resolvePath } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

// Convex source uses ESM ".js" specifiers that point at ".ts" files. Node's
// type stripping does not rewrite those, so map them here.
registerHooks({
  resolve(specifier, context, nextResolve) {
    if (specifier.startsWith(".") && specifier.endsWith(".js") && context.parentURL) {
      const parentPath = fileURLToPath(context.parentURL);
      const candidateJs = resolvePath(dirname(parentPath), specifier);
      if (!existsSync(candidateJs) && existsSync(candidateJs.replace(/\.js$/, ".ts"))) {
        return { url: pathToFileURL(candidateJs.replace(/\.js$/, ".ts")).href, shortCircuit: true };
      }
    }
    return nextResolve(specifier, context);
  },
});

const schemaModule = await import("../convex/schema.ts");
const keyPolicy = await import("../convex/lib/keys.ts");
const apiKeys = await import("../convex/apiKeys.ts");
const apiFunctions = await import("../convex/api.ts");

const { INACTIVITY_DEACTIVATION_DAYS, DAY_MS } = keyPolicy;
const schema = schemaModule.default;

/* ------------------------------------------------------------------ */
/* Minimal Convex database emulator                                     */
/* ------------------------------------------------------------------ */

const tableIndexes = new Map();
const tableValidators = new Map();
for (const [name, table] of Object.entries(schema.tables)) {
  tableIndexes.set(name, new Map((table.indexes ?? []).map((index) => [index.indexDescriptor, index.fields])));
  tableValidators.set(name, table.validator?.json ?? null);
}

/** Convex sorts undefined before numbers, numbers before strings. */
function rank(value) {
  if (value === undefined) return 0;
  if (typeof value === "number") return 1;
  if (typeof value === "string") return 2;
  return 3;
}

function compareValues(a, b) {
  const rankA = rank(a);
  const rankB = rank(b);
  if (rankA !== rankB) return rankA - rankB;
  if (rankA === 0) return 0;
  if (a === b) return 0;
  return a < b ? -1 : 1;
}

function checkFieldType(fieldType, value, path) {
  switch (fieldType.type) {
    case "number":
      assert.equal(typeof value, "number", `${path} must be a number`);
      return;
    case "string":
    case "id":
      assert.equal(typeof value, "string", `${path} must be a string`);
      return;
    case "boolean":
      assert.equal(typeof value, "boolean", `${path} must be a boolean`);
      return;
    case "literal":
      assert.equal(value, fieldType.value, `${path} must equal ${JSON.stringify(fieldType.value)}`);
      return;
    case "union": {
      const matched = fieldType.value.some((option) => {
        try {
          checkFieldType(option, value, path);
          return true;
        } catch {
          return false;
        }
      });
      assert.ok(matched, `${path} does not match any union member`);
      return;
    }
    default:
      return; // objects/arrays/any are not used by the tables under test
  }
}

/** Rejects writes Convex itself would reject: unknown or wrongly typed fields. */
function validateDocument(table, doc) {
  const validator = tableValidators.get(table);
  if (!validator || validator.type !== "object") return;
  const fields = validator.value;
  for (const [field, value] of Object.entries(doc)) {
    if (field.startsWith("_")) continue;
    assert.ok(fields[field], `${table}.${field} is not in the schema`);
    if (value === undefined) {
      assert.ok(fields[field].optional, `${table}.${field} is required and cannot be undefined`);
      continue;
    }
    checkFieldType(fields[field].fieldType, value, `${table}.${field}`);
  }
  for (const [field, spec] of Object.entries(fields)) {
    if (!spec.optional) assert.ok(doc[field] !== undefined, `${table}.${field} is required`);
  }
}

class FakeDb {
  constructor() {
    this.docs = new Map();
    this.counter = 0;
  }

  async insert(table, fields) {
    validateDocument(table, fields);
    const id = `${table}:${++this.counter}`;
    const doc = { _id: id, _creationTime: Date.now(), _table: table, ...fields };
    this.docs.set(id, doc);
    return id;
  }

  async get(id) {
    const doc = this.docs.get(id);
    return doc ? { ...doc } : null;
  }

  async patch(id, fields) {
    const doc = this.docs.get(id);
    assert.ok(doc, `patch on missing document ${id}`);
    const next = { ...doc };
    for (const [field, value] of Object.entries(fields)) {
      // Convex removes a field when it is patched with undefined.
      if (value === undefined) delete next[field];
      else next[field] = value;
    }
    const { _id, _creationTime, _table, ...rest } = next;
    validateDocument(doc._table, rest);
    this.docs.set(id, next);
  }

  query(table) {
    return new FakeQuery(this, table);
  }

  all(table) {
    return [...this.docs.values()].filter((doc) => doc._table === table).map((doc) => ({ ...doc }));
  }
}

class FakeQuery {
  constructor(db, table) {
    this.db = db;
    this.table = table;
    this.indexFields = [];
    this.constraints = [];
    this.direction = "asc";
  }

  withIndex(indexName, rangeBuilder) {
    const fields = tableIndexes.get(this.table)?.get(indexName);
    assert.ok(fields, `${this.table} has no index named ${indexName}`);
    this.indexFields = fields;
    if (rangeBuilder) {
      const constraints = this.constraints;
      let cursor = 0;
      const builder = {
        eq(field, value) {
          assert.equal(field, fields[cursor], `index ${indexName}: expected field ${fields[cursor]}, got ${field}`);
          cursor += 1;
          constraints.push({ op: "eq", field, value });
          return builder;
        },
      };
      for (const op of ["gt", "gte", "lt", "lte"]) {
        builder[op] = (field, value) => {
          assert.equal(field, fields[cursor], `index ${indexName}: expected field ${fields[cursor]}, got ${field}`);
          constraints.push({ op, field, value });
          return builder;
        };
      }
      rangeBuilder(builder);
    }
    return this;
  }

  order(direction) {
    this.direction = direction;
    return this;
  }

  #rows() {
    let rows = this.db.all(this.table);
    for (const constraint of this.constraints) {
      rows = rows.filter((row) => {
        const comparison = compareValues(row[constraint.field], constraint.value);
        if (constraint.op === "eq") return comparison === 0;
        if (constraint.op === "gt") return comparison > 0;
        if (constraint.op === "gte") return comparison >= 0;
        if (constraint.op === "lt") return comparison < 0;
        return comparison <= 0;
      });
    }
    const sortFields = this.indexFields.length > 0 ? this.indexFields : ["_creationTime"];
    rows.sort((left, right) => {
      for (const field of sortFields) {
        const comparison = compareValues(left[field], right[field]);
        if (comparison !== 0) return this.direction === "desc" ? -comparison : comparison;
      }
      return 0;
    });
    return rows;
  }

  async collect() {
    return this.#rows();
  }

  async take(count) {
    return this.#rows().slice(0, count);
  }

  async unique() {
    const rows = this.#rows();
    assert.ok(rows.length <= 1, `unique() matched ${rows.length} rows in ${this.table}`);
    return rows[0] ?? null;
  }
}

/* ------------------------------------------------------------------ */
/* Test helpers                                                         */
/* ------------------------------------------------------------------ */

function ctxFor(db, userId) {
  return {
    db,
    auth: {
      getUserIdentity: async () => (userId ? { subject: `${userId}|session`, tokenIdentifier: "test" } : null),
    },
  };
}

const run = (fn, ctx, args = {}) => fn._handler(ctx, args);

let currentNow = Date.UTC(2026, 0, 1, 12, 0, 0);
const realNow = Date.now;
Date.now = () => currentNow;
const advanceDays = (days) => {
  currentNow += days * DAY_MS;
};

const results = [];
async function test(name, fn) {
  try {
    await fn();
    results.push({ name, ok: true });
    console.log(`  ok   ${name}`);
  } catch (error) {
    results.push({ name, ok: false, error });
    console.log(`  FAIL ${name}\n       ${error instanceof Error ? error.message : error}`);
  }
}

async function seedKey(db, { userId = "users:1", name = "Test key", createdAt = Date.now(), ...rest } = {}) {
  const id = await db.insert("apiKeys", {
    userId,
    name,
    keyHash: `hash-${name}-${createdAt}`,
    prefix: "v0id_abc1234…",
    createdAt,
    ...rest,
  });
  return (await db.get(id));
}

console.log(`\nAPI key inactivity policy (${INACTIVITY_DEACTIVATION_DAYS}-day window)\n`);

await test("a brand new key is active and reports its deactivation date", async () => {
  const db = new FakeDb();
  const ctx = ctxFor(db, "users:1");
  await run(apiKeys.create, ctx, { name: "Fresh key" });
  const [listed] = await run(apiKeys.list, ctx);
  assert.equal(listed.status, "active");
  assert.equal(listed.deactivatedAt, null);
  assert.equal(listed.deactivatesAt, Date.now() + INACTIVITY_DEACTIVATION_DAYS * DAY_MS);
  assert.equal(listed.inactivityDays, INACTIVITY_DEACTIVATION_DAYS);
});

await test("a key used 13 days ago still authorizes and resets the timer", async () => {
  const db = new FakeDb();
  const key = await seedKey(db, { createdAt: Date.now() - 40 * DAY_MS, lastUsedAt: Date.now(), lastActiveAt: Date.now() });
  advanceDays(13);
  const result = await run(apiFunctions.authorizeAndCount, ctxFor(db), { keyHash: key.keyHash, modelId: "v0idgpt" });
  assert.equal(result.ok, true);
  const stored = await db.get(key._id);
  assert.equal(stored.deactivatedAt, undefined);
  assert.equal(stored.lastActiveAt, Date.now(), "a request restarts the inactivity window");
});

await test("a key unused for 14 days is refused and stamped deactivated", async () => {
  const db = new FakeDb();
  const lastUse = Date.now();
  const key = await seedKey(db, { createdAt: lastUse - DAY_MS, lastUsedAt: lastUse, lastActiveAt: lastUse });
  advanceDays(INACTIVITY_DEACTIVATION_DAYS);
  const result = await run(apiFunctions.authorizeAndCount, ctxFor(db), { keyHash: key.keyHash, modelId: "v0idgpt" });
  assert.equal(result.ok, false);
  assert.equal(result.reason, "key_deactivated");
  assert.equal(result.inactivityDays, INACTIVITY_DEACTIVATION_DAYS);
  const stored = await db.get(key._id);
  assert.equal(stored.deactivatedAt, lastUse + INACTIVITY_DEACTIVATION_DAYS * DAY_MS);
  assert.equal(stored.deactivatedReason, "inactivity");
  assert.equal(stored.revokedAt, undefined, "deactivation must not revoke the key");
});

await test("a never-used key deactivates 14 days after it was created", async () => {
  const db = new FakeDb();
  const key = await seedKey(db, { name: "Unused", createdAt: Date.now() });
  advanceDays(INACTIVITY_DEACTIVATION_DAYS + 1);
  const result = await run(apiFunctions.authorizeAndCount, ctxFor(db), { keyHash: key.keyHash, modelId: "v0idgpt" });
  assert.equal(result.reason, "key_deactivated");
});

await test("checkApiKey reports deactivation to GET /v1/models", async () => {
  const db = new FakeDb();
  const key = await seedKey(db, { createdAt: Date.now() - 30 * DAY_MS, lastUsedAt: Date.now() - 20 * DAY_MS });
  const access = await run(apiFunctions.checkApiKey, ctxFor(db), { keyHash: key.keyHash });
  assert.equal(access.status, "deactivated");
  assert.equal(access.prefix, key.prefix);
  assert.equal(access.inactivityDays, INACTIVITY_DEACTIVATION_DAYS);
});

await test("the hourly sweep deactivates idle keys and leaves fresh ones alone", async () => {
  const db = new FakeDb();
  const now = Date.now();
  const idle = await seedKey(db, { name: "Idle", createdAt: now - 60 * DAY_MS, lastUsedAt: now - 20 * DAY_MS, lastActiveAt: now - 20 * DAY_MS });
  const fresh = await seedKey(db, { name: "Fresh", createdAt: now - 60 * DAY_MS, lastUsedAt: now - DAY_MS, lastActiveAt: now - DAY_MS });
  const revoked = await seedKey(db, { name: "Revoked", createdAt: now - 60 * DAY_MS, lastActiveAt: now - 60 * DAY_MS, revokedAt: now - 5 * DAY_MS });

  const summary = await run(apiKeys.deactivateIdleKeys, ctxFor(db), {});
  assert.equal(summary.deactivated, 1);
  assert.equal((await db.get(idle._id)).deactivatedAt, now - 20 * DAY_MS + INACTIVITY_DEACTIVATION_DAYS * DAY_MS);
  assert.equal((await db.get(fresh._id)).deactivatedAt, undefined);
  assert.equal((await db.get(revoked._id)).deactivatedAt, undefined, "revoked keys are left untouched");
});

await test("the sweep backfills lastActiveAt on keys created before the field existed", async () => {
  const db = new FakeDb();
  const legacyUse = Date.now() - 2 * DAY_MS;
  const legacy = await seedKey(db, { name: "Legacy", createdAt: Date.now() - 90 * DAY_MS, lastUsedAt: legacyUse });
  const summary = await run(apiKeys.deactivateIdleKeys, ctxFor(db), {});
  assert.equal(summary.deactivated, 0);
  assert.equal(summary.backfilled, 1);
  assert.equal((await db.get(legacy._id)).lastActiveAt, legacyUse);
});

await test("the sweep is idempotent", async () => {
  const db = new FakeDb();
  await seedKey(db, { createdAt: Date.now() - 90 * DAY_MS, lastActiveAt: Date.now() - 90 * DAY_MS });
  const first = await run(apiKeys.deactivateIdleKeys, ctxFor(db), {});
  const second = await run(apiKeys.deactivateIdleKeys, ctxFor(db), {});
  assert.equal(first.deactivated, 1);
  assert.equal(second.deactivated, 0);
  assert.equal(second.scanned, 0, "already deactivated keys drop out of the index range");
});

await test("the owner reactivates a key from the dashboard and it works again", async () => {
  const db = new FakeDb();
  const owner = ctxFor(db, "users:1");
  const key = await seedKey(db, { createdAt: Date.now() - 90 * DAY_MS, lastActiveAt: Date.now() - 90 * DAY_MS });
  await run(apiKeys.deactivateIdleKeys, ctxFor(db), {});

  const beforeList = await run(apiKeys.list, owner);
  assert.equal(beforeList[0].status, "deactivated");
  assert.equal(beforeList[0].deactivatedReason, "inactivity");

  const activation = await run(apiKeys.activate, owner, { id: key._id });
  assert.equal(activation.ok, true);
  assert.equal(activation.deactivatesAt, Date.now() + INACTIVITY_DEACTIVATION_DAYS * DAY_MS);

  const afterList = await run(apiKeys.list, owner);
  assert.equal(afterList[0].status, "active");
  assert.equal(afterList[0].deactivatedAt, null);

  const result = await run(apiFunctions.authorizeAndCount, ctxFor(db), { keyHash: key.keyHash, modelId: "v0idgpt" });
  assert.equal(result.ok, true, "the same secret works again after activation");
});

await test("a reactivated key survives the next sweep", async () => {
  const db = new FakeDb();
  const owner = ctxFor(db, "users:1");
  const key = await seedKey(db, { createdAt: Date.now() - 90 * DAY_MS, lastActiveAt: Date.now() - 90 * DAY_MS });
  await run(apiKeys.deactivateIdleKeys, ctxFor(db), {});
  await run(apiKeys.activate, owner, { id: key._id });
  const summary = await run(apiKeys.deactivateIdleKeys, ctxFor(db), {});
  assert.equal(summary.deactivated, 0);
  assert.equal((await db.get(key._id)).deactivatedAt, undefined);
});

await test("only the owner can activate a key", async () => {
  const db = new FakeDb();
  const key = await seedKey(db, { userId: "users:1", createdAt: Date.now() - 90 * DAY_MS });
  await assert.rejects(() => run(apiKeys.activate, ctxFor(db, "users:2"), { id: key._id }), /API key not found/);
  await assert.rejects(() => run(apiKeys.activate, ctxFor(db, null), { id: key._id }), /Sign in/);
});

await test("revoked keys cannot be reactivated", async () => {
  const db = new FakeDb();
  const key = await seedKey(db, { createdAt: Date.now() - 2 * DAY_MS, revokedAt: Date.now() });
  await assert.rejects(() => run(apiKeys.activate, ctxFor(db, "users:1"), { id: key._id }), /cannot be reactivated/);
});

await test("activating an already active key is a harmless no-op", async () => {
  const db = new FakeDb();
  const owner = ctxFor(db, "users:1");
  const key = await seedKey(db, { createdAt: Date.now(), lastActiveAt: Date.now() });
  const activation = await run(apiKeys.activate, owner, { id: key._id });
  assert.equal(activation.alreadyActive, true);
  assert.equal((await db.get(key._id)).reactivatedAt, undefined);
});

await test("the dashboard separates active keys from deactivated ones", async () => {
  const db = new FakeDb();
  const now = Date.now();
  await seedKey(db, { name: "Live", createdAt: now, lastActiveAt: now });
  await seedKey(db, { name: "Idle", createdAt: now - 90 * DAY_MS, lastActiveAt: now - 90 * DAY_MS });
  await seedKey(db, { name: "Gone", createdAt: now - 90 * DAY_MS, lastActiveAt: now, revokedAt: now });
  const dashboard = await import("../convex/dashboard.ts");
  const overview = await run(dashboard.get, ctxFor(db, "users:1"));
  assert.equal(overview.activeKeys, 1);
  assert.equal(overview.deactivatedKeys, 1);
  assert.equal(overview.inactivityDays, INACTIVITY_DEACTIVATION_DAYS);
});

await test("the API error explains the cause and how to reactivate", async () => {
  const message = keyPolicy.deactivatedKeyMessage({ dashboardUrl: "https://v0id.example/#api-keys", prefix: "v0id_abc1234…" });
  assert.match(message, new RegExp(`${INACTIVITY_DEACTIVATION_DAYS} days without being used`));
  assert.match(message, /"Activate"/);
  assert.match(message, /API keys panel/);
  assert.match(message, /https:\/\/v0id\.example\/#api-keys/);
  const steps = keyPolicy.reactivationSteps("https://v0id.example/#api-keys");
  assert.equal(steps.length, 3);
});

Date.now = realNow;

const failed = results.filter((result) => !result.ok);
console.log(`\n${results.length - failed.length}/${results.length} checks passed\n`);
if (failed.length > 0) process.exit(1);
