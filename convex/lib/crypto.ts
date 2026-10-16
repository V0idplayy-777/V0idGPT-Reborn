const encoder = new TextEncoder();

function toHex(bytes: Uint8Array) {
  return Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0")).join("");
}

export async function sha256Hex(value: string) {
  const digest = await crypto.subtle.digest("SHA-256", encoder.encode(value));
  return toHex(new Uint8Array(digest));
}

export function createApiToken() {
  const bytes = new Uint8Array(32);
  crypto.getRandomValues(bytes);
  return `v0id_${toHex(bytes)}`;
}
