import { useEffect, useState, type FormEvent, type ReactNode } from "react";
import {
  Activity,
  ArrowDownRight,
  ArrowRight,
  ArrowUpRight,
  BookOpen,
  Check,
  CheckCircle2,
  ChevronDown,
  CircleHelp,
  Clock3,
  Code2,
  Copy,
  CreditCard,
  ExternalLink,
  Gauge,
  KeyRound,
  LoaderCircle,
  LockKeyhole,
  LogOut,
  Menu,
  Plus,
  ShieldCheck,
  Sparkles,
  Terminal,
  Trash2,
  X,
  Zap,
} from "lucide-react";
import { useAction, useMutation, useQuery } from "convex/react";
import { useAuthActions, useConvexAuth } from "@convex-dev/auth/react";
import { api } from "../convex/_generated/api";
import { convexSiteUrl } from "./lib/convex";

type Tier = "free" | "premium";
type BackendStatus = "checking" | "ready" | "offline";
type DashboardData = {
  user: { name: string; email: string | null; image: string | null };
  tier: Tier;
  subscriptionStatus: string;
  limit: number;
  used: number;
  remaining: number;
  chart: { date: string; requests: number }[];
  activeKeys: number;
  billingCustomerReady: boolean;
};
type ApiKey = {
  id: string;
  name: string;
  prefix: string;
  createdAt: number;
  lastUsedAt: number | null;
  revokedAt: number | null;
};
type ToastMessage = { text: string; kind?: "success" | "error" | "info" };
type TabId = "javascript" | "python" | "curl";
type NewKey = { token: string; prefix: string; name: string };

const MODEL_ID = "v0idgpt";
const FREE_LIMIT = 5_000;
const PREMIUM_LIMIT = 500_000;

function numberFormat(value: number) {
  return new Intl.NumberFormat("en-NZ").format(value);
}

function shortDate(value: string) {
  return new Date(`${value}T12:00:00Z`).toLocaleDateString("en-NZ", { weekday: "short" });
}

function relativeTime(timestamp: number | null | undefined) {
  if (!timestamp) return "Never used";
  const minutes = Math.max(0, Math.floor((Date.now() - timestamp) / 60_000));
  if (minutes < 1) return "Used just now";
  if (minutes < 60) return `Used ${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `Used ${hours}h ago`;
  return `Used ${Math.floor(hours / 24)}d ago`;
}

async function copyText(text: string) {
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(text);
    return;
  }
  const field = document.createElement("textarea");
  field.value = text;
  field.style.position = "fixed";
  field.style.opacity = "0";
  document.body.appendChild(field);
  field.select();
  document.execCommand("copy");
  field.remove();
}

function BrandMark({ compact = false }: { compact?: boolean }) {
  return (
    <div className={`brand-mark${compact ? " brand-mark--compact" : ""}`} aria-label="v0id">
      <span className="brand-symbol"><span /></span>
      {!compact && <span className="brand-word">v0id<span className="brand-word__dot">.</span></span>}
    </div>
  );
}

function QuotaRing({ used, limit, signedIn }: { used: number; limit: number; signedIn: boolean }) {
  const fraction = signedIn ? Math.min(used / Math.max(1, limit), 1) : 0;
  const circumference = 2 * Math.PI * 34;
  return (
    <div className="quota-ring" aria-label={signedIn ? `${numberFormat(used)} of ${numberFormat(limit)} requests used` : "Sign in to view usage"}>
      <svg viewBox="0 0 84 84" aria-hidden="true">
        <circle className="quota-ring__track" cx="42" cy="42" r="34" />
        <circle
          className="quota-ring__value"
          cx="42"
          cy="42"
          r="34"
          strokeDasharray={`${circumference}`}
          strokeDashoffset={`${circumference * (1 - fraction)}`}
        />
      </svg>
      <div className="quota-ring__center"><Gauge size={17} strokeWidth={1.7} /></div>
    </div>
  );
}

function UsageChart({ data, ready }: { data: DashboardData["chart"]; ready: boolean }) {
  const max = Math.max(1, ...data.map((item) => item.requests));
  return (
    <div className={`usage-chart${ready ? "" : " usage-chart--empty"}`}>
      <div className="usage-chart__plot" role="img" aria-label={ready ? "Daily requests over the last seven days" : "Usage history appears after your first inference request"}>
        {data.map((item, index) => {
          const height = item.requests === 0 ? 4 : Math.max(8, (item.requests / max) * 74);
          return (
            <div className="usage-chart__column" key={item.date}>
              <div className="usage-chart__bar-wrap">
                <div className="usage-chart__bar" style={{ height: `${ready ? height : 4}px`, opacity: item.requests ? 1 : 0.25 }} />
              </div>
              <span className="usage-chart__day">{shortDate(item.date)}</span>
              {item.requests > 0 && <span className="usage-chart__tooltip">{numberFormat(item.requests)}</span>}
              {index === 6 && ready && <span className="usage-chart__today">today</span>}
            </div>
          );
        })}
      </div>
      {!ready && <div className="usage-chart__empty-note">Your first request starts the chart.</div>}
    </div>
  );
}

const snippets: Record<TabId, { label: string; language: string; code: string }> = {
  javascript: {
    label: "JavaScript",
    language: "javascript",
    code: `const response = await fetch("${convexSiteUrl}/v1/chat/completions", {\n  method: "POST",\n  headers: {\n    "Authorization": "Bearer YOUR_V0ID_API_KEY",\n    "Content-Type": "application/json"\n  },\n  body: JSON.stringify({\n    model: "v0idgpt",\n    messages: [{ role: "user", content: "Say hello." }],\n    max_tokens: 40\n  })\n});\n\nconst result = await response.json();\nconsole.log(result.choices[0].message.content);`,
  },
  python: {
    label: "Python",
    language: "python",
    code: `import requests\n\nresponse = requests.post(\n    "${convexSiteUrl}/v1/chat/completions",\n    headers={"Authorization": "Bearer YOUR_V0ID_API_KEY"},\n    json={\n        "model": "v0idgpt",\n        "messages": [{"role": "user", "content": "Say hello."}],\n        "max_tokens": 40,\n    },\n    timeout=30,\n)\nprint(response.json()["choices"][0]["message"]["content"])`,
  },
  curl: {
    label: "cURL",
    language: "bash",
    code: `curl ${convexSiteUrl}/v1/chat/completions \\\n  -H "Authorization: Bearer YOUR_V0ID_API_KEY" \\\n  -H "Content-Type: application/json" \\\n  -d '{\n    "model": "v0idgpt",\n    "messages": [{"role": "user", "content": "Say hello."}],\n    "max_tokens": 40\n  }'`,
  },
};

function CodeSnippet({ onNotify }: { onNotify: (message: ToastMessage) => void }) {
  const [tab, setTab] = useState<TabId>("javascript");
  const current = snippets[tab];
  const handleCopy = async () => {
    try {
      await copyText(current.code);
      onNotify({ text: "Code sample copied", kind: "success" });
    } catch {
      onNotify({ text: "Couldn't access the clipboard. Select and copy the snippet instead.", kind: "error" });
    }
  };
  return (
    <section className="panel code-panel" id="quickstart">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">QUICKSTART</p>
          <h2>One request. That's it.</h2>
          <p className="panel-subtitle">Drop in a key and start building.</p>
        </div>
        <div className="heading-icon heading-icon--purple"><Code2 size={17} /></div>
      </div>
      <div className="code-tabs" role="tablist" aria-label="Code language">
        {(Object.keys(snippets) as TabId[]).map((id) => (
          <button
            className={`code-tabs__tab${tab === id ? " is-active" : ""}`}
            key={id}
            type="button"
            role="tab"
            aria-selected={tab === id}
            onClick={() => setTab(id)}
          >
            {snippets[id].label}
          </button>
        ))}
        <button className="code-copy" type="button" onClick={handleCopy} aria-label="Copy code sample"><Copy size={14} /> Copy</button>
      </div>
      <pre className="code-block"><code data-language={current.language}>{current.code}</code></pre>
      <div className="code-footnote"><span className="status-dot status-dot--lime" /> OpenAI-style response shape <span className="code-footnote__divider">·</span> Text generation only</div>
    </section>
  );
}

function ModelCard() {
  return (
    <section className="panel model-panel" id="models">
      <div className="panel-heading panel-heading--compact">
        <div>
          <p className="eyebrow">MODEL CATALOG <span className="eyebrow-count">01</span></p>
          <h2>One model. On purpose.</h2>
        </div>
        <div className="heading-icon heading-icon--blue"><CpuIcon /></div>
      </div>
      <article className="model-card">
        <div className="model-card__top">
          <div className="model-avatar"><span>v</span><i>0</i></div>
          <div className="model-card__identity">
            <div className="model-card__titleline"><h3>v0idgpt</h3><span className="badge badge--default">DEFAULT</span></div>
            <p>v0id · language model · text generation</p>
          </div>
          <span className="model-status"><span className="status-dot status-dot--lime" /> Available</span>
        </div>
        <div className="model-card__facts">
          <div><span>Context</span><strong>64 tokens</strong></div>
          <div><span>Vocabulary</span><strong>320 tokens</strong></div>
          <div><span>Access</span><strong>Free tier</strong></div>
        </div>
        <div className="model-card__note"><Sparkles size={14} /><span>Small experimental checkpoint. Best for quick tests, not production-grade reasoning.</span></div>
      </article>
      <div className="model-panel__foot"><span>More models will appear here when they're ready.</span><span className="model-panel__foot-mark">01 / 01</span></div>
    </section>
  );
}

function CpuIcon() {
  return <Activity size={17} />;
}

function PlanCard({
  tier,
  isAuthenticated,
  onUpgrade,
  onManage,
  busy,
}: {
  tier: Tier;
  isAuthenticated: boolean;
  onUpgrade: () => void;
  onManage: () => void;
  busy: boolean;
}) {
  const premium = tier === "premium";
  return (
    <section className={`panel plan-panel${premium ? " plan-panel--premium" : ""}`} id="plan">
      <div className="plan-panel__orb" />
      <div className="plan-panel__topline">
        <span className={`badge ${premium ? "badge--premium" : "badge--free"}`}>{premium ? "PREMIUM" : "FREE"}</span>
        <span className="plan-panel__price"><strong>$5</strong><span>/ month</span></span>
      </div>
      <h2>{premium ? "More room to move." : "Room to grow."}</h2>
      <p className="plan-panel__description">
        {premium
          ? "You're on the higher daily allowance. Manage your subscription any time."
          : "A simple upgrade for bigger workloads. Same model today; new models will be added later."}
      </p>
      <div className="plan-panel__quota"><Zap size={15} /><strong>{premium ? "500,000" : "5,000"}</strong> requests / day</div>
      {premium ? (
        <button className="button button--secondary plan-button" type="button" onClick={onManage} disabled={busy}>
          {busy ? <LoaderCircle className="spin" size={16} /> : <CreditCard size={16} />}
          Manage subscription <ExternalLink size={13} className="button-tail" />
        </button>
      ) : (
        <button className="button button--lime plan-button" type="button" onClick={onUpgrade} disabled={busy}>
          {busy ? <LoaderCircle className="spin" size={16} /> : <ArrowUpRight size={16} />}
          {isAuthenticated ? "Upgrade to premium" : "Sign in to upgrade"}
        </button>
      )}
      <p className="plan-panel__fineprint">Premium increases your quota only. No premium-only models are listed yet.</p>
    </section>
  );
}

function DocsModal({ onClose }: { onClose: () => void }) {
  return (
    <div className="modal-backdrop" role="presentation" onMouseDown={(event) => event.target === event.currentTarget && onClose()}>
      <section className="docs-modal" role="dialog" aria-modal="true" aria-labelledby="docs-heading">
        <div className="modal-topline">
          <div className="modal-title-icon"><BookOpen size={17} /></div>
          <div><p className="eyebrow">API REFERENCE</p><h2 id="docs-heading">Everything you need to call v0id.</h2></div>
          <button className="icon-button modal-close" type="button" onClick={onClose} aria-label="Close documentation"><X size={18} /></button>
        </div>
        <div className="docs-modal__body">
          <div className="docs-callout"><ShieldCheck size={17} /><span>Keep API keys on a trusted server. Never ship a secret key in browser code or a public app bundle.</span></div>
          <div className="docs-section">
            <h3><span>01</span> Authentication</h3>
            <p>Send the key in the <code>Authorization</code> header as a bearer token. Keys are shown once, stored as hashes, and can be revoked from this page.</p>
            <pre className="docs-inline-code">Authorization: Bearer YOUR_V0ID_API_KEY</pre>
          </div>
          <div className="docs-section">
            <h3><span>02</span> Generate a chat completion</h3>
            <p>Use the OpenAI-style endpoint below. Only <code>v0idgpt</code> is available right now. Responses are JSON; streaming is not supported.</p>
            <pre className="docs-inline-code">POST {convexSiteUrl}/v1/chat/completions</pre>
            <pre className="docs-inline-code">{`{\n  "model": "v0idgpt",\n  "messages": [{ "role": "user", "content": "Say hello." }],\n  "max_tokens": 40,\n  "temperature": 0.7\n}`}</pre>
          </div>
          <div className="docs-section docs-section--grid">
            <div><h3><span>03</span> Limits</h3><p>Free: 5,000 accepted inference requests per UTC day. Premium: 500,000. All keys on an account share one allowance. At the limit, requests return <code>429</code> until 00:00 UTC.</p></div>
            <div><h3><span>04</span> Model boundaries</h3><p>The checkpoint has a 64-token context and 320-token vocabulary. Prompts are truncated to the most recent context. Treat it as an experimental model.</p></div>
          </div>
          <div className="docs-section">
            <h3><span>05</span> Errors & usage</h3>
            <p>Errors include a machine-readable <code>error.code</code>. Successful requests return token counts, and responses include <code>x-ratelimit-limit</code>, <code>x-ratelimit-remaining</code>, and <code>x-ratelimit-reset</code> headers.</p>
          </div>
        </div>
        <div className="modal-footer"><span>One page. No maze of menus.</span><button className="button button--secondary" type="button" onClick={onClose}>Got it <Check size={15} /></button></div>
      </section>
    </div>
  );
}

function RevealKeyModal({ keyInfo, onClose, onNotify }: { keyInfo: NewKey; onClose: () => void; onNotify: (message: ToastMessage) => void }) {
  const [copied, setCopied] = useState(false);
  const copyKey = async () => {
    try {
      await copyText(keyInfo.token);
      setCopied(true);
      onNotify({ text: "API key copied to clipboard", kind: "success" });
    } catch {
      onNotify({ text: "Couldn't copy the key. Select it and copy manually.", kind: "error" });
    }
  };
  return (
    <div className="modal-backdrop" role="presentation">
      <section className="key-modal" role="dialog" aria-modal="true" aria-labelledby="key-modal-title">
        <div className="key-modal__success"><CheckCircle2 size={19} /><span>Key created</span></div>
        <button className="icon-button modal-close" type="button" onClick={onClose} aria-label="Close key dialog"><X size={18} /></button>
        <h2 id="key-modal-title">Copy your new key.</h2>
        <p className="panel-subtitle">This secret is only shown once. Store it somewhere safe before closing.</p>
        <label className="field-label" htmlFor="new-api-key">{keyInfo.name}</label>
        <div className="secret-field"><input id="new-api-key" value={keyInfo.token} readOnly onFocus={(event) => event.currentTarget.select()} /><button className="button button--lime" type="button" onClick={copyKey}>{copied ? <Check size={15} /> : <Copy size={15} />}{copied ? "Copied" : "Copy key"}</button></div>
        <div className="key-modal__warning"><LockKeyhole size={15} /><span>Only the prefix <code>{keyInfo.prefix}</code> is saved in your key list.</span></div>
        <button className="button button--secondary key-modal__done" type="button" onClick={onClose}>I've saved it <ArrowRight size={15} /></button>
      </section>
    </div>
  );
}

function Toast({ message, onDismiss }: { message: ToastMessage; onDismiss: () => void }) {
  return (
    <div className={`toast toast--${message.kind ?? "info"}`} role="status">
      {message.kind === "success" ? <CheckCircle2 size={17} /> : message.kind === "error" ? <CircleHelp size={17} /> : <Activity size={17} />}
      <span>{message.text}</span><button className="toast__close" type="button" onClick={onDismiss} aria-label="Dismiss"><X size={15} /></button>
    </div>
  );
}

function PanelHeading({ eyebrow, title, subtitle, icon, iconTone = "green", trailing }: {
  eyebrow: string;
  title: string;
  subtitle?: string;
  icon: ReactNode;
  iconTone?: "green" | "purple" | "blue";
  trailing?: ReactNode;
}) {
  return (
    <div className="panel-heading">
      <div><p className="eyebrow">{eyebrow}</p><h2>{title}</h2>{subtitle && <p className="panel-subtitle">{subtitle}</p>}</div>
      {trailing ?? <div className={`heading-icon heading-icon--${iconTone}`}>{icon}</div>}
    </div>
  );
}

export default function App() {
  const { isLoading: authLoading, isAuthenticated } = useConvexAuth();
  const { signIn, signOut } = useAuthActions();
  const overview = useQuery(api.dashboard.get, isAuthenticated ? {} : "skip") as DashboardData | null | undefined;
  const keyList = useQuery(api.apiKeys.list, isAuthenticated ? {} : "skip") as ApiKey[] | undefined;
  const ensureProfileMutation = useMutation(api.dashboard.ensureProfile);
  const createKeyMutation = useMutation(api.apiKeys.create);
  const revokeKeyMutation = useMutation(api.apiKeys.revoke);
  const createCheckout = useAction(api.billing.createCheckout);
  const createPortal = useAction(api.billing.createPortal);
  const [keyName, setKeyName] = useState("Personal development");
  const [creatingKey, setCreatingKey] = useState(false);
  const [billingBusy, setBillingBusy] = useState(false);
  const [revoking, setRevoking] = useState<string | null>(null);
  const [newKey, setNewKey] = useState<NewKey | null>(null);
  const [showDocs, setShowDocs] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [toast, setToast] = useState<ToastMessage | null>(null);
  const [signingIn, setSigningIn] = useState(false);
  const [backendStatus, setBackendStatus] = useState<BackendStatus>("checking");
  const [healthCheckSeq, setHealthCheckSeq] = useState(0);
  const signedIn = isAuthenticated && !authLoading;
  useEffect(() => {
    if (isAuthenticated) void ensureProfileMutation({}).catch(() => undefined);
  }, [isAuthenticated, ensureProfileMutation]);
  useEffect(() => {
    const controller = new AbortController();
    setBackendStatus("checking");
    fetch(`${convexSiteUrl}/health`, { signal: controller.signal })
      .then((response) => {
        if (!response.ok) throw new Error("Convex health endpoint is unavailable.");
        return response.json() as Promise<{ status?: string }>;
      })
      .then((result) => setBackendStatus(result.status === "ok" ? "ready" : "offline"))
      .catch(() => {
        if (!controller.signal.aborted) setBackendStatus("offline");
      });
    return () => controller.abort();
  }, [healthCheckSeq]);
  const gatewayDot = backendStatus === "ready" ? "status-dot--lime" : backendStatus === "offline" ? "status-dot--red" : "status-dot--amber";
  const tier: Tier = overview?.tier ?? "free";
  const limit = overview?.limit ?? FREE_LIMIT;
  const used = overview?.used ?? 0;
  const usagePercent = Math.min(100, Math.round((used / Math.max(1, limit)) * 100));
  const modelEndpoint = `${convexSiteUrl}/v1/chat/completions`;
  const accountName = overview?.user?.name?.trim() || overview?.user?.email?.split("@")[0] || "Developer";
  const avatarLetter = accountName.charAt(0).toUpperCase();
  const hasChartData = Boolean(overview?.chart.some((day) => day.requests > 0));
  const chartData = overview?.chart ?? Array.from({ length: 7 }, (_, index) => ({
    date: new Date(Date.now() - (6 - index) * 24 * 60 * 60 * 1000).toISOString().slice(0, 10),
    requests: 0,
  }));

  const notify = (message: ToastMessage) => {
    setToast(message);
    window.setTimeout(() => setToast((current) => current === message ? null : current), 4200);
  };

  const handleSignIn = async () => {
    setSigningIn(true);
    try {
      await signIn("github", { redirectTo: "/" });
    } catch {
      notify({ text: "GitHub sign-in isn't configured yet. Add the GitHub OAuth keys to your Convex deployment.", kind: "error" });
      setSigningIn(false);
    }
  };

  const handleCreateKey = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!signedIn) {
      notify({ text: "Sign in to create and manage API keys.", kind: "info" });
      return;
    }
    setCreatingKey(true);
    try {
      const result = await createKeyMutation({ name: keyName }) as { token: string; prefix: string };
      setNewKey({ token: result.token, prefix: result.prefix, name: keyName.trim() });
      setKeyName("Personal development");
      notify({ text: "Your key is ready. Copy it before closing this window.", kind: "success" });
    } catch (error) {
      notify({ text: error instanceof Error ? error.message : "Couldn't create an API key.", kind: "error" });
    } finally {
      setCreatingKey(false);
    }
  };

  const handleRevoke = async (key: ApiKey) => {
    if (!window.confirm(`Revoke “${key.name}”? Any app using it will stop working immediately.`)) return;
    setRevoking(key.id);
    try {
      await revokeKeyMutation({ id: key.id as never });
      notify({ text: `${key.name} was revoked.`, kind: "success" });
    } catch (error) {
      notify({ text: error instanceof Error ? error.message : "Couldn't revoke that key.", kind: "error" });
    } finally {
      setRevoking(null);
    }
  };

  const handleUpgrade = async () => {
    if (!signedIn) {
      await handleSignIn();
      return;
    }
    setBillingBusy(true);
    try {
      const result = await createCheckout({}) as { url: string };
      window.location.assign(result.url);
    } catch (error) {
      notify({ text: error instanceof Error ? error.message : "Couldn't open Stripe checkout.", kind: "error" });
      setBillingBusy(false);
    }
  };

  const handleManagePlan = async () => {
    setBillingBusy(true);
    try {
      const result = await createPortal({}) as { url: string };
      window.location.assign(result.url);
    } catch (error) {
      notify({ text: error instanceof Error ? error.message : "Couldn't open billing settings.", kind: "error" });
      setBillingBusy(false);
    }
  };

  const handleSignOut = async () => {
    try {
      await signOut();
      notify({ text: "Signed out.", kind: "success" });
    } catch {
      notify({ text: "Couldn't sign out. Please try again.", kind: "error" });
    }
  };

  const handleCopyEndpoint = async () => {
    try {
      await copyText(modelEndpoint);
      notify({ text: "API endpoint copied", kind: "success" });
    } catch {
      notify({ text: "Couldn't copy the endpoint.", kind: "error" });
    }
  };

  return (
    <div className="app-shell">
      <aside className={`sidebar${mobileMenuOpen ? " sidebar--open" : ""}`}>
        <div className="sidebar__brand"><BrandMark /><span className="sidebar__edition">INFERENCE CONSOLE</span></div>
        <div className="sidebar__workspace">
          <div className="workspace-avatar">v</div>
          <div className="workspace-copy"><strong>v0id workspace</strong><span>Personal project</span></div>
          <ChevronDown size={14} className="workspace-chevron" />
        </div>
        <div className="sidebar__group-label">WORKSPACE</div>
        <nav className="sidebar-nav" aria-label="Workspace navigation">
          <a className="sidebar-nav__item is-active" href="#overview" onClick={() => setMobileMenuOpen(false)}><Activity size={16} /><span>Overview</span><span className="nav-current" /></a>
          <a className="sidebar-nav__item" href="#api-keys" onClick={() => setMobileMenuOpen(false)}><KeyRound size={16} /><span>API keys</span>{signedIn && <span className="nav-count">{overview?.activeKeys ?? 0}</span>}</a>
          <button className="sidebar-nav__item" type="button" onClick={() => { setShowDocs(true); setMobileMenuOpen(false); }}><BookOpen size={16} /><span>Documentation</span><ExternalLink size={13} className="nav-external" /></button>
          <a className="sidebar-nav__item" href="#models" onClick={() => setMobileMenuOpen(false)}><CpuIcon /><span>Models</span><span className="nav-count nav-count--quiet">1</span></a>
          <a className="sidebar-nav__item" href="#plan" onClick={() => setMobileMenuOpen(false)}><CreditCard size={16} /><span>Plan & billing</span></a>
        </nav>
        <div className="sidebar__bottom">
          <div className="sidebar-status"><span className={`status-dot ${gatewayDot}`} /><div><strong>API gateway</strong><span>{backendStatus === "ready" ? "Convex endpoint reachable" : backendStatus === "checking" ? "Checking Convex endpoint" : "Backend setup required"}</span></div><span className={`sidebar-status__pulse ${gatewayDot}`} /></div>
          <a className="sidebar-help" href="#quickstart" onClick={() => setMobileMenuOpen(false)}><CircleHelp size={15} />Need a hand? <span>Quickstart <ArrowRight size={12} /></span></a>
          {signedIn && (
            <div className="sidebar-user">
              {overview?.user?.image ? <img className="sidebar-user__avatar" src={overview.user.image} alt="" /> : <div className="sidebar-user__avatar sidebar-user__avatar--initial">{avatarLetter}</div>}
              <div className="sidebar-user__copy"><strong>{accountName}</strong><span>{overview?.user?.email ?? "Signed in"}</span></div>
              <button className="icon-button sidebar-user__logout" type="button" onClick={handleSignOut} title="Sign out" aria-label="Sign out"><LogOut size={15} /></button>
            </div>
          )}
        </div>
      </aside>

      <div className="main-shell">
        <header className="topbar">
          <button className="icon-button mobile-menu-button" type="button" onClick={() => setMobileMenuOpen((open) => !open)} aria-label="Open navigation"><Menu size={19} /></button>
          <div className="breadcrumbs"><span>v0id</span><span className="breadcrumb-slash">/</span><strong>Overview</strong><span className="topbar__environment"><span className={`status-dot ${gatewayDot}`} />{backendStatus === "ready" ? "ONLINE" : backendStatus === "checking" ? "CHECK" : "SETUP"}</span></div>
          <div className="topbar__right">
            <button className="topbar-docs" type="button" onClick={() => setShowDocs(true)}><BookOpen size={15} /> API docs <ArrowUpRight size={12} /></button>
            {signedIn ? (
              <div className={`tier-chip tier-chip--${tier}`}><span className="tier-chip__spark">✳</span>{tier === "premium" ? "PREMIUM" : "FREE"}<span className="tier-chip__divider" />{numberFormat(limit)}/day</div>
            ) : (
              <button className="button button--lime topbar-signin" type="button" onClick={handleSignIn} disabled={signingIn || authLoading}>{signingIn ? <LoaderCircle size={15} className="spin" /> : <span className="github-mark">GH</span>} Sign in</button>
            )}
          </div>
        </header>

        <main id="overview" className="content-area">
          <section className="welcome-row">
            <div>
              <div className="welcome-kicker"><span className="welcome-kicker__line" /> DEVELOPER CONSOLE <span className="welcome-kicker__date">· {new Date().toLocaleDateString("en-NZ", { month: "short", day: "numeric", year: "numeric" })}</span></div>
              <h1>{signedIn ? <>Good to have you, <span>{accountName.split(" ")[0]}.</span></> : <>Inference, <span>without the noise.</span></>}</h1>
              <p className="welcome-description">{signedIn ? "Your model, keys, and usage — all in one place." : "A clean home for your API keys, usage, and the v0id model."}</p>
            </div>
            <div className="welcome-actions">
              <span className="welcome-live"><span className={`status-dot ${gatewayDot}`} /> {backendStatus === "ready" ? signedIn ? "Workspace live" : "Ready when you are" : backendStatus === "checking" ? "Checking backend" : "Backend needs setup"}</span>
              {!signedIn && <button className="button button--secondary welcome-login" type="button" onClick={handleSignIn} disabled={signingIn || authLoading}>{signingIn ? <LoaderCircle size={15} className="spin" /> : <span className="github-mark">GH</span>} Continue with GitHub</button>}
            </div>
          </section>

          {backendStatus === "offline" && (
            <section className="backend-notice" role="status">
              <div className="backend-notice__icon"><CircleHelp size={17} /></div>
              <div className="backend-notice__copy"><strong>Convex isn't reachable yet.</strong><p>The app can provision its schema and functions at server startup when <code>CONVEX_DEPLOY_KEY</code> is set server-side. Or run <code>npm run convex:deploy</code> in a secure environment. The browser never receives the deploy key.</p></div>
              <button className="button button--secondary backend-notice__retry" type="button" onClick={() => setHealthCheckSeq((value) => value + 1)}>Check again <ArrowRight size={13} /></button>
            </section>
          )}

          <section className="metric-grid" aria-label="Usage overview">
            <article className="panel metric-card metric-card--quota">
              <div className="metric-card__top"><span className="eyebrow">TODAY'S USAGE</span><span className="metric-card__icon metric-card__icon--lime"><Gauge size={16} /></span></div>
              <div className="quota-layout"><QuotaRing used={used} limit={limit} signedIn={signedIn} /><div className="quota-copy"><div className="metric-number">{signedIn ? numberFormat(used) : "—"}<span> / {numberFormat(limit)}</span></div><p>{signedIn ? "requests used today" : "requests per UTC day"}</p><div className="quota-progress"><span style={{ width: `${signedIn ? usagePercent : 0}%` }} /></div><small>{signedIn ? `${usagePercent}% of your daily allowance` : "Sign in to load live usage"}</small></div></div>
              <div className="metric-card__foot"><span><Clock3 size={13} /> Resets at 00:00 UTC</span><span className="metric-card__remaining">{signedIn ? `${numberFormat(overview?.remaining ?? limit)} left` : `${tier === "premium" ? "Premium" : "Free"} tier`}</span></div>
            </article>

            <article className="panel metric-card metric-card--history">
              <div className="metric-card__top"><span className="eyebrow">REQUESTS · 7 DAYS</span><span className="metric-card__icon metric-card__icon--blue"><Activity size={16} /></span></div>
              <div className="history-total">{signedIn ? numberFormat(chartData.reduce((total, item) => total + item.requests, 0)) : "—"}<span className="history-total__label">total calls</span></div>
              <UsageChart data={chartData} ready={signedIn && hasChartData} />
              <div className="metric-card__foot"><span><span className="chart-legend-dot" /> Daily inference requests</span><span>{signedIn ? "UTC" : "Live after sign-in"}</span></div>
            </article>

            <article className="panel metric-card metric-card--api">
              <div className="metric-card__top"><span className="eyebrow">YOUR API</span><span className="metric-card__icon metric-card__icon--purple"><Terminal size={16} /></span></div>
              <div className="api-endpoint-label">BASE ENDPOINT</div>
              <button className="endpoint-copy" type="button" onClick={handleCopyEndpoint} title="Copy endpoint"><span>{convexSiteUrl.replace("https://", "")}</span><Copy size={14} /></button>
              <div className="api-route"><span className="method-pill">POST</span><code>/v1/chat/completions</code></div>
              <div className="metric-card__foot"><span><ShieldCheck size={13} /> API-key protected</span><button className="foot-link" type="button" onClick={() => setShowDocs(true)}>Read docs <ArrowRight size={12} /></button></div>
            </article>
          </section>

          <section className="dashboard-grid">
            <div className="dashboard-grid__left">
              <section className="panel keys-panel" id="api-keys">
                <PanelHeading
                  eyebrow="ACCESS MANAGEMENT"
                  title="API keys"
                  subtitle={signedIn ? "Create, copy once, and revoke in a click." : "Sign in to create and manage keys."}
                  icon={<KeyRound size={17} />}
                  trailing={<span className="keys-count">{signedIn ? `${overview?.activeKeys ?? 0} ACTIVE` : "PRIVATE"}</span>}
                />
                <form className="create-key-form" onSubmit={handleCreateKey}>
                  <label htmlFor="key-name" className="visually-hidden">Name your API key</label>
                  <div className="key-input-wrap"><KeyRound size={15} /><input id="key-name" value={keyName} onChange={(event) => setKeyName(event.target.value)} maxLength={48} placeholder="Name your API key" disabled={!signedIn || creatingKey} /></div>
                  <button className="button button--lime create-key-button" type="submit" disabled={!signedIn || creatingKey || keyName.trim().length === 0}>
                    {creatingKey ? <LoaderCircle size={15} className="spin" /> : <Plus size={16} />}
                    Create key
                  </button>
                </form>
                <div className="key-list">
                  {signedIn && keyList === undefined ? (
                    <div className="key-empty key-empty--loading"><LoaderCircle size={17} className="spin" /> Loading your keys…</div>
                  ) : signedIn && keyList && keyList.length > 0 ? keyList.map((key) => (
                    <article className={`key-row${key.revokedAt ? " key-row--revoked" : ""}`} key={key.id}>
                      <div className="key-row__icon"><KeyRound size={16} /></div>
                      <div className="key-row__main"><div className="key-row__name"><strong>{key.name}</strong>{key.revokedAt ? <span className="badge badge--revoked">REVOKED</span> : <span className="key-row__active"><span className="status-dot status-dot--lime" /> ACTIVE</span>}</div><div className="key-row__meta"><code>{key.prefix}</code><span>·</span><span>Created {new Date(key.createdAt).toLocaleDateString("en-NZ", { day: "numeric", month: "short" })}</span><span>·</span><span>{key.revokedAt ? "No longer valid" : relativeTime(key.lastUsedAt)}</span></div></div>
                      {!key.revokedAt && <button className="icon-button key-row__revoke" type="button" onClick={() => handleRevoke(key)} disabled={revoking === key.id} aria-label={`Revoke ${key.name}`} title="Revoke key">{revoking === key.id ? <LoaderCircle size={16} className="spin" /> : <Trash2 size={15} />}</button>}
                    </article>
                  )) : (
                    <div className="key-empty"><div className="key-empty__icon"><LockKeyhole size={18} /></div><div><strong>{signedIn ? "No API keys yet" : "Your keys stay private"}</strong><p>{signedIn ? "Create your first key above. The secret is only shown once." : "Sign in with GitHub to create a key and unlock live usage."}</p></div>{!signedIn && <button className="text-button" type="button" onClick={handleSignIn}>Sign in <ArrowRight size={13} /></button>}</div>
                  )}
                </div>
                <div className="keys-panel__foot"><ShieldCheck size={13} /><span>Keys are stored as SHA-256 hashes. Revocation takes effect immediately.</span><span className="keys-panel__cap">UP TO 10 ACTIVE</span></div>
              </section>
              <CodeSnippet onNotify={notify} />
            </div>

            <div className="dashboard-grid__right">
              <ModelCard />
              <PlanCard tier={tier} isAuthenticated={signedIn} onUpgrade={handleUpgrade} onManage={handleManagePlan} busy={billingBusy} />
              <section className="security-note"><div className="security-note__icon"><ShieldCheck size={15} /></div><div><strong>Built to stay simple.</strong><p>Each account has one shared daily quota. No hidden model routing and no surprise overages.</p></div><span className="security-note__arrow"><ArrowDownRight size={14} /></span></section>
            </div>
          </section>

          <footer className="page-footer"><div><BrandMark compact /><span>Independent inference, built with intent.</span></div><div className="page-footer__links"><button type="button" onClick={() => setShowDocs(true)}>API reference</button><span>·</span><a href="#models">One model, for now</a><span>·</span><span>UTC daily limits</span></div><span className="footer-version">CONSOLE 0.1.0</span></footer>
        </main>
      </div>

      {mobileMenuOpen && <button className="mobile-scrim" aria-label="Close navigation" type="button" onClick={() => setMobileMenuOpen(false)} />}
      {showDocs && <DocsModal onClose={() => setShowDocs(false)} />}
      {newKey && <RevealKeyModal keyInfo={newKey} onClose={() => setNewKey(null)} onNotify={notify} />}
      {toast && <Toast message={toast} onDismiss={() => setToast(null)} />}
    </div>
  );
}
