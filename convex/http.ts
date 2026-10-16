import { anyApi, httpRouter } from "convex/server";
import { auth } from "./auth.js";

const internal: any = anyApi;
import { httpAction } from "./_generated/server.js";
import { sha256Hex } from "./lib/crypto.js";
import { ACTIVE_MODEL_ID } from "./lib/limits.js";

const http = httpRouter();
auth.addHttpRoutes(http);

const MAX_BODY_BYTES = 16 * 1024;
const API_KEY_PATTERN = /^v0id_[a-f0-9]{64}$/;
const encoder = new TextEncoder();

class ApiError extends Error {
  status: number;
  code: string;
  extraHeaders: Record<string, string>;

  constructor(status: number, code: string, message: string, extraHeaders: Record<string, string> = {}) {
    super(message);
    this.status = status;
    this.code = code;
    this.extraHeaders = extraHeaders;
  }
}

function json(data: unknown, status = 200, extraHeaders: Record<string, string> = {}) {
  return new Response(JSON.stringify(data), {
    status,
    headers: {
      "content-type": "application/json; charset=utf-8",
      "cache-control": "no-store",
      ...extraHeaders,
    },
  });
}

function cors(response: Response) {
  const headers = new Headers(response.headers);
  headers.set("access-control-allow-origin", "*");
  headers.set("access-control-allow-methods", "GET, POST, OPTIONS");
  headers.set("access-control-allow-headers", "authorization, content-type");
  headers.set("access-control-max-age", "600");
  return new Response(response.body, { status: response.status, statusText: response.statusText, headers });
}

const preflight = httpAction(async () => cors(new Response(null, { status: 204 })));

function apiAction(handler: (ctx: any, request: Request) => Promise<Response>) {
  return httpAction(async (ctx, request) => {
    try {
      return cors(await handler(ctx, request));
    } catch (error) {
      if (error instanceof ApiError) {
        return cors(json({ error: { message: error.message, type: "v0id_error", code: error.code } }, error.status, error.extraHeaders));
      }
      return cors(json({ error: { message: "The model could not process this request.", type: "server_error", code: "inference_failed" } }, 500));
    }
  });
}

function presentedKey(request: Request) {
  const header = request.headers.get("authorization") ?? "";
  const match = /^Bearer\s+(\S+)$/i.exec(header);
  const token = match?.[1];
  if (!token || !API_KEY_PATTERN.test(token)) {
    throw new ApiError(401, "invalid_api_key", "A valid v0id API key is required.");
  }
  return token;
}

async function readJson(request: Request) {
  const contentLength = Number(request.headers.get("content-length") ?? "0");
  if (contentLength > MAX_BODY_BYTES) {
    throw new ApiError(413, "request_too_large", "Request bodies must be 16 KB or smaller.");
  }
  const body = await request.text();
  if (encoder.encode(body).length > MAX_BODY_BYTES) {
    throw new ApiError(413, "request_too_large", "Request bodies must be 16 KB or smaller.");
  }
  try {
    const value: unknown = JSON.parse(body);
    if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("not an object");
    return value as Record<string, unknown>;
  } catch {
    throw new ApiError(400, "invalid_json", "Request body must be a JSON object.");
  }
}

function dayReset() {
  const now = new Date();
  const next = Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate() + 1);
  return Math.floor(next / 1000);
}

function quotaHeaders(limit: number, used: number) {
  const reset = dayReset();
  return {
    "x-ratelimit-limit": String(limit),
    "x-ratelimit-remaining": String(Math.max(0, limit - used)),
    "x-ratelimit-reset": String(reset),
  };
}

async function validateModelAccess(ctx: any, request: Request) {
  const keyHash = await sha256Hex(presentedKey(request));
  const access = await ctx.runQuery(internal.api.checkApiKey, { keyHash });
  if (!access) throw new ApiError(401, "invalid_api_key", "This API key is invalid or has been revoked.");
  return access as { tier: "free" | "premium"; limit: number; used: number };
}

function parseGenerationOptions(body: Record<string, unknown>) {
  if (body.model !== undefined && typeof body.model !== "string") {
    throw new ApiError(400, "invalid_model", "model must be a string.");
  }
  const model = typeof body.model === "string" ? body.model : ACTIVE_MODEL_ID;
  if (model !== ACTIVE_MODEL_ID) {
    throw new ApiError(404, "model_not_found", `Model '${model}' is not available. The only active model is '${ACTIVE_MODEL_ID}'.`);
  }
  if (body.stream === true) {
    throw new ApiError(400, "streaming_not_supported", "Streaming is not available for this model yet. Set stream to false or omit it.");
  }

  const maxTokens = body.max_tokens ?? body.maxTokens ?? 40;
  const temperature = body.temperature ?? 0.7;
  const topK = body.top_k ?? body.topK ?? 50;
  const seed = body.seed;
  if (typeof maxTokens !== "number" || !Number.isInteger(maxTokens) || maxTokens < 1 || maxTokens > 128) {
    throw new ApiError(400, "invalid_max_tokens", "max_tokens must be an integer from 1 to 128.");
  }
  if (typeof temperature !== "number" || !Number.isFinite(temperature) || temperature < 0 || temperature > 2) {
    throw new ApiError(400, "invalid_temperature", "temperature must be a number from 0 to 2.");
  }
  if (typeof topK !== "number" || !Number.isInteger(topK) || topK < 1 || topK > 320) {
    throw new ApiError(400, "invalid_top_k", "top_k must be an integer from 1 to 320.");
  }
  if (seed !== undefined && (typeof seed !== "number" || !Number.isInteger(seed) || seed < 0 || seed > 2_147_483_647)) {
    throw new ApiError(400, "invalid_seed", "seed must be a non-negative 32-bit integer.");
  }

  let stop: string[] | undefined;
  if (body.stop !== undefined) {
    if (typeof body.stop === "string") stop = [body.stop];
    else if (Array.isArray(body.stop) && body.stop.every((item) => typeof item === "string")) stop = body.stop as string[];
    else throw new ApiError(400, "invalid_stop", "stop must be a string or an array of strings.");
    if (stop.length > 4 || stop.some((item) => item.length === 0 || item.length > 100)) {
      throw new ApiError(400, "invalid_stop", "Provide up to four non-empty stop strings, each 100 characters or fewer.");
    }
  }

  return {
    model,
    maxTokens: Number(maxTokens),
    temperature: Number(temperature),
    topK: Number(topK),
    seed: seed === undefined ? undefined : Number(seed),
    stop,
  };
}

function chatPrompt(messagesValue: unknown) {
  if (!Array.isArray(messagesValue) || messagesValue.length === 0 || messagesValue.length > 32) {
    throw new ApiError(400, "invalid_messages", "messages must be a non-empty array with at most 32 entries.");
  }
  const messages = messagesValue.map((message) => {
    if (!message || typeof message !== "object" || Array.isArray(message)) {
      throw new ApiError(400, "invalid_messages", "Each message must contain a role and string content.");
    }
    const item = message as Record<string, unknown>;
    if (!["system", "user", "assistant"].includes(String(item.role)) || typeof item.content !== "string") {
      throw new ApiError(400, "invalid_messages", "Only system, user, and assistant messages with string content are supported.");
    }
    return { role: String(item.role), content: item.content.slice(0, 4_000) };
  });
  if (!messages.some((message) => message.role === "user")) {
    throw new ApiError(400, "invalid_messages", "Include at least one user message.");
  }

  // The exported checkpoint was trained on short q:/a: text patterns, rather
  // than chat-role tokens. This lightweight adapter keeps the endpoint familiar
  // while formatting prompts in the model's learned style.
  const transcript = messages
    .filter((message) => message.role !== "system")
    .map((message) => `${message.role === "assistant" ? "a" : "q"}: ${message.content}`)
    .join("\n");
  const system = messages.filter((message) => message.role === "system").map((message) => message.content).join("\n");
  return `${system ? `${system}\n` : ""}${transcript}\na:`;
}

async function runInference(ctx: any, request: Request, prompt: string, body: Record<string, unknown>, mode: "chat" | "generate") {
  const options = parseGenerationOptions(body);
  const keyHash = await sha256Hex(presentedKey(request));
  const quota = await ctx.runMutation(internal.api.authorizeAndCount, {
    keyHash,
    modelId: options.model,
  });

  if (!quota.ok) {
    if (quota.reason === "unauthorized") throw new ApiError(401, "invalid_api_key", "This API key is invalid or has been revoked.");
    if (quota.reason === "model_not_found") throw new ApiError(404, "model_not_found", "The requested model is not available.");
    const headers: Record<string, string> = quotaHeaders(quota.limit, quota.used);
    headers["retry-after"] = String(Math.max(1, dayReset() - Math.floor(Date.now() / 1000)));
    throw new ApiError(429, "daily_limit_exceeded", `Daily request limit reached (${quota.limit.toLocaleString()} requests). Resets at 00:00 UTC.`, headers);
  }

  const output = await ctx.runAction(internal.inference.generate, {
    prompt,
    maxTokens: options.maxTokens,
    temperature: options.temperature,
    topK: options.topK,
    ...(options.seed === undefined ? {} : { seed: options.seed }),
    ...(options.stop === undefined ? {} : { stop: options.stop }),
  });
  await ctx.runMutation(internal.api.recordTokens, {
    userId: quota.userId,
    date: quota.date,
    promptTokens: output.promptTokens,
    completionTokens: output.completionTokens,
  });

  const headers = quotaHeaders(quota.limit, quota.used);
  const created = Math.floor(Date.now() / 1000);
  if (mode === "chat") {
    return json(
      {
        id: `chatcmpl-${crypto.randomUUID()}`,
        object: "chat.completion",
        created,
        model: ACTIVE_MODEL_ID,
        choices: [{ index: 0, message: { role: "assistant", content: output.text }, finish_reason: output.finishReason === "length" ? "length" : "stop" }],
        usage: {
          prompt_tokens: output.promptTokens,
          completion_tokens: output.completionTokens,
          total_tokens: output.promptTokens + output.completionTokens,
        },
      },
      200,
      headers,
    );
  }

  const includePrompt = body.include_prompt === true;
  return json(
    {
      id: `gen-${crypto.randomUUID()}`,
      object: "text_completion",
      created,
      model: ACTIVE_MODEL_ID,
      choices: [{ index: 0, text: includePrompt ? `${prompt}${output.text}` : output.text, finish_reason: output.finishReason === "length" ? "length" : "stop" }],
      usage: {
        prompt_tokens: output.promptTokens,
        completion_tokens: output.completionTokens,
        total_tokens: output.promptTokens + output.completionTokens,
      },
    },
    200,
    headers,
  );
}

const health = httpAction(async () =>
  cors(json({ status: "ok", service: "v0id-inference", model: ACTIVE_MODEL_ID })),
);

const models = apiAction(async (ctx, request) => {
  if (request.method !== "GET") throw new ApiError(405, "method_not_allowed", "Use GET for this endpoint.");
  const access = await validateModelAccess(ctx, request);
  return json(
    {
      object: "list",
      data: [
        {
          id: ACTIVE_MODEL_ID,
          object: "model",
          owned_by: "v0id",
          tier: "free",
          context_length: 64,
          status: "available",
        },
      ],
      tier: access.tier,
      daily_limit: access.limit,
    },
    200,
    quotaHeaders(access.limit, access.used),
  );
});

const chatCompletions = apiAction(async (ctx, request) => {
  if (request.method !== "POST") throw new ApiError(405, "method_not_allowed", "Use POST for this endpoint.");
  const body = await readJson(request);
  const prompt = chatPrompt(body.messages);
  return await runInference(ctx, request, prompt, body, "chat");
});

const generate = apiAction(async (ctx, request) => {
  if (request.method !== "POST") throw new ApiError(405, "method_not_allowed", "Use POST for this endpoint.");
  const body = await readJson(request);
  if (typeof body.prompt !== "string" || body.prompt.trim().length === 0) {
    throw new ApiError(400, "invalid_prompt", "prompt must be a non-empty string.");
  }
  if (body.prompt.length > 12_000) throw new ApiError(400, "prompt_too_long", "prompt must be 12,000 characters or shorter.");
  return await runInference(ctx, request, body.prompt, body, "generate");
});

const webhook = httpAction(async (ctx, request) => {
  const secret = process.env.STRIPE_WEBHOOK_SECRET;
  if (!secret) return json({ error: "Stripe webhook is not configured." }, 503);

  const signatureHeader = request.headers.get("stripe-signature") ?? "";
  const parts = signatureHeader.split(",").map((part) => part.split("=", 2));
  const timestamp = parts.find(([key]) => key === "t")?.[1];
  const signatures = parts.filter(([key]) => key === "v1").map(([, value]) => value);
  if (!timestamp || signatures.length === 0 || !/^\d+$/.test(timestamp)) {
    return json({ error: "Invalid Stripe signature." }, 400);
  }
  const timestampNumber = Number(timestamp);
  if (Math.abs(Math.floor(Date.now() / 1000) - timestampNumber) > 300) {
    return json({ error: "Expired Stripe signature." }, 400);
  }

  const body = await request.text();
  if (encoder.encode(body).length > 1_000_000) return json({ error: "Webhook body is too large." }, 413);
  const key = await crypto.subtle.importKey(
    "raw",
    encoder.encode(secret),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign"],
  );
  const digest = new Uint8Array(await crypto.subtle.sign("HMAC", key, encoder.encode(`${timestamp}.${body}`)));
  const expected = Array.from(digest, (byte) => byte.toString(16).padStart(2, "0")).join("");
  const valid = signatures.some((signature) => {
    if (signature.length !== expected.length) return false;
    let mismatch = 0;
    for (let index = 0; index < expected.length; index++) mismatch |= expected.charCodeAt(index) ^ signature.charCodeAt(index);
    return mismatch === 0;
  });
  if (!valid) return json({ error: "Invalid Stripe signature." }, 400);

  let event: { id: string; type: string; data?: { object?: Record<string, unknown> } };
  try {
    event = JSON.parse(body) as typeof event;
  } catch {
    return json({ error: "Invalid webhook JSON." }, 400);
  }
  if (typeof event.id !== "string" || typeof event.type !== "string" || !event.id || !event.type) {
    return json({ error: "Incomplete Stripe event." }, 400);
  }

  const object = event.data?.object ?? {};
  let userId: string | undefined;
  let status = "free";
  let customerId: string | undefined;
  let subscriptionId: string | undefined;

  if (event.type === "checkout.session.completed" || event.type === "checkout.session.async_payment_succeeded") {
    userId = typeof object.client_reference_id === "string" ? object.client_reference_id : undefined;
    if (!userId && object.metadata && typeof object.metadata === "object") {
      const metadata = object.metadata as Record<string, unknown>;
      userId = typeof metadata.userId === "string" ? metadata.userId : undefined;
    }
    status = object.mode === "subscription" && (object.payment_status === "paid" || event.type.endsWith("async_payment_succeeded")) ? "active" : "free";
    customerId = typeof object.customer === "string" ? object.customer : undefined;
    subscriptionId = typeof object.subscription === "string" ? object.subscription : undefined;
  } else if (event.type.startsWith("customer.subscription.")) {
    const metadata = object.metadata && typeof object.metadata === "object" ? object.metadata as Record<string, unknown> : {};
    userId = typeof metadata.userId === "string" ? metadata.userId : undefined;
    status = typeof object.status === "string" ? object.status : "free";
    customerId = typeof object.customer === "string" ? object.customer : undefined;
    subscriptionId = typeof object.id === "string" ? object.id : undefined;
    if (event.type === "customer.subscription.deleted") status = "canceled";
  }

  await ctx.runMutation(internal.billing.applyStripeEvent, {
    eventId: event.id,
    eventType: event.type,
    ...(userId ? { userId: userId as never } : {}),
    status,
    ...(customerId ? { customerId } : {}),
    ...(subscriptionId ? { subscriptionId } : {}),
  });
  return json({ received: true });
});

http.route({ path: "/health", method: "GET", handler: health });
http.route({ path: "/health", method: "OPTIONS", handler: preflight });
http.route({ path: "/v1/models", method: "GET", handler: models });
http.route({ path: "/v1/models", method: "OPTIONS", handler: preflight });
http.route({ path: "/v1/chat/completions", method: "POST", handler: chatCompletions });
http.route({ path: "/v1/chat/completions", method: "OPTIONS", handler: preflight });
http.route({ path: "/v1/generate", method: "POST", handler: generate });
http.route({ path: "/v1/generate", method: "OPTIONS", handler: preflight });
http.route({ path: "/stripe/webhook", method: "POST", handler: webhook });

export default http;
