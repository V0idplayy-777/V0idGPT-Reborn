# v0id Inference Console

A single-page control room for the v0id inference API. It uses **Convex for the backend** and the exported checkpoint in `v0idgpt/javascript` as the **only model**. There are no external model providers or model fallbacks.

## What is included

- GitHub sign-in via Convex Auth.
- One-page dashboard for the v0idgpt model, API keys, UTC daily usage, quickstart snippets, and plan details.
- One-time API key display; only a SHA-256 digest and a short prefix are stored.
- OpenAI-style `POST /v1/chat/completions`, direct `POST /v1/generate`, and authenticated `GET /v1/models` HTTP Actions.
- Atomic per-account quotas in Convex: 5,000 accepted inference requests per UTC day on Free and 500,000 on Premium. All API keys on the same account share the allowance. Rejected requests at the limit return HTTP 429 and do not increment usage.
- Optional Stripe Checkout, customer portal, and a signature-verified webhook for the $5/month Premium entitlement. Premium currently raises the request allowance only; no premium-only model is listed.
- Live seven-day usage chart and immediate key revocation through Convex subscriptions.

The model is a small experimental checkpoint with a 64-token context and 320-token vocabulary. It is not a substitute for a production reasoning model. The dashboard and API make those limits visible rather than routing elsewhere.

## Local setup

Requirements: Node.js 20.19+ (Node 22 recommended).

```bash
npm install
cp .env.example .env.local
npm run dev
```

The frontend defaults to the Convex deployment URL configured for this project. `VITE_CONVEX_URL` and `VITE_CONVEX_SITE_URL` can override it for another environment.

### Safe first-run Convex provisioning

The Convex schema and functions are code, so they must be deployed before a browser dashboard can call them. The web client cannot safely create its own backend tables: that would require exposing an administrator deploy key to every browser user. Instead, `npm run dev` now checks for a **server-side** `CONVEX_DEPLOY_KEY` and, when present, runs an idempotent Convex deployment before starting Vite. The key is removed from Vite's environment and is never bundled or written to disk. If no key is set, development starts normally with a clear message that provisioning was skipped.

### Automatic GitHub releases

The workflow in `.github/workflows/release.yml` runs on pushes to `main` or manually from GitHub Actions. It reads your existing repository Actions secret **`CONVEX_DEPLOY_KEY`**; no secret belongs in frontend code or a `VITE_` variable.

Each run:

1. Installs dependencies and runs `npm run build:release`.
2. Uses Convex's deployment command to build with the selected deployment's public URL, then deploy the schema, indexes, and functions. This provisions missing tables, including Convex Auth tables, without dummy records or wiping existing data. Incompatible schema changes fail rather than automatically deleting data.
3. Uploads `frontend-dist` only after both the build and backend deployment succeed. Publish that artifact with your frontend host; this workflow does not itself configure hosting.

Use a production deployment key for production releases. The frontend's `VITE_CONVEX_URL` is supplied by Convex, so the build targets the deployment selected by the key rather than the hardcoded fallback URL. The deploy key is removed from the frontend build process's environment.

GitHub secrets are available to Actions, not to visitors' browsers or ordinary local terminals. Tables are automatically provisioned at release time, not on page load. If backend resources are subsequently removed, rerun the workflow to restore the schema/functions. OAuth and billing environment variables still need the setup below.

For an explicit deployment from a trusted terminal with `CONVEX_DEPLOY_KEY` already exported:

```bash
npm run convex:deploy
# Or deploy and build a frontend targeting the same deployment:
npm run build:release
```

Local `npm run dev` retains development provisioning using `convex dev --once` when a key is present. Release commands use `convex deploy` instead. Do not put the key in `.env.local` or commit it. Deployments must run on a network that can reach Convex.

## Authentication setup

1. Create a GitHub OAuth App.
2. Set its **Authorization callback URL** to `https://tough-pelican-388.convex.site/api/auth/callback/github` (or the matching `.convex.site` URL for your deployment).
3. Add the OAuth app's client ID and secret as Convex deployment variables `AUTH_GITHUB_ID` and `AUTH_GITHUB_SECRET`. Do not use `VITE_` prefixes for secrets.
4. Convex Auth uses the deployment's `CONVEX_SITE_URL` for its issuer and callback routes.

The sign-in button intentionally reports setup guidance when GitHub OAuth is not configured; it never creates an unauthenticated account with access to keys or usage.

## Premium billing setup (optional)

Create a Stripe product with a **$5 monthly recurring price**, then add these variables to the Convex deployment:

- `STRIPE_SECRET_KEY`
- `STRIPE_PREMIUM_PRICE_ID`
- `STRIPE_WEBHOOK_SECRET`
- `SITE_URL` — the public dashboard origin, without a trailing slash

Configure the Stripe webhook endpoint as `https://tough-pelican-388.convex.site/stripe/webhook` and subscribe to:

- `checkout.session.completed`
- `checkout.session.async_payment_succeeded`
- `customer.subscription.created`
- `customer.subscription.updated`
- `customer.subscription.deleted`

The webhook verifies Stripe's timestamped HMAC signature and processes each event idempotently. Premium status is only granted by a verified Stripe event; the browser cannot promote its own account. If Stripe is not configured, the Free console and inference API continue to work.

## API

Base URL: `https://tough-pelican-388.convex.site`

### `POST /v1/chat/completions`

```http
Authorization: Bearer v0id_<64 lowercase hex characters>
Content-Type: application/json
```

```json
{
  "model": "v0idgpt",
  "messages": [{ "role": "user", "content": "Say hello." }],
  "max_tokens": 40,
  "temperature": 0.7,
  "top_k": 50,
  "seed": 42
}
```

Returns an OpenAI-style `choices[0].message.content` and prompt/completion token counts. Supported roles are `system`, `user`, and `assistant`; message content must be plain text. `max_tokens` is 1–128, `temperature` is 0–2, and `top_k` is 1–320. Streaming is not supported. The model receives its recent context only; longer prompts are truncated by the exported tokenizer.

### `POST /v1/generate`

```json
{
  "model": "v0idgpt",
  "prompt": "q: What is a tensor?\\na:",
  "max_tokens": 40,
  "temperature": 0.7
}
```

Returns a text-completion-style `choices[0].text`. Both inference routes return rate-limit headers: `x-ratelimit-limit`, `x-ratelimit-remaining`, and `x-ratelimit-reset` (Unix seconds). Daily limits reset at 00:00 UTC.

### `GET /v1/models`

Requires a valid bearer key and lists exactly one model: `v0idgpt`. It does not consume an inference request.

## Checks

```bash
npm run build
npm run model:verify
```
