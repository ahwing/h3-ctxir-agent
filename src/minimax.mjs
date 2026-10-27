// Official MiniMax H3-Context-IR API arm ("api").
//   POST /v2/h3_context_ir -> task_id; GET /v2/query/video_generation/{task_id} -> content.prompt
// Docs: https://platform.minimax.io/docs/api-reference/video-generation-v2-h3-context-ir
// The key is read at runtime from MINIMAX_API_KEY or a 0600 file; it is never logged or written.
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

export const KEY_FILE = process.env.MINIMAX_KEY_FILE || path.join(os.homedir(), '.config/h3-ctxir/minimax.key');
const BASE = process.env.MINIMAX_BASE || 'https://api.minimaxi.com';
// Context-IR price (USD per M tokens), pricing/overview 2026-10
export const PRICE = {in: 0.9, out: 3.6};
const MIME = {'.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png', '.webp': 'image/webp', '.mp4': 'video/mp4', '.mov': 'video/quicktime', '.mp3': 'audio/mp3', '.wav': 'audio/wav'};

/** Inline a local file as a data URI (S3 presigned URLs from us-west are often unreachable from MiniMax, error 2013). */
export function dataUri(file) {
  const mime = MIME[path.extname(file).toLowerCase()];
  if (!mime) throw new Error(`no mime for ${file}`);
  return `data:${mime};base64,${fs.readFileSync(file).toString('base64')}`;
}

export function apiKey() {
  if (process.env.MINIMAX_API_KEY) return process.env.MINIMAX_API_KEY.trim();
  if (!fs.existsSync(KEY_FILE)) throw new Error(`no MiniMax key: set MINIMAX_API_KEY or create ${KEY_FILE} (chmod 600)`);
  const mode = fs.statSync(KEY_FILE).mode & 0o077;
  if (mode) throw new Error(`${KEY_FILE} is readable by group/others; run chmod 600`);
  return fs.readFileSync(KEY_FILE, 'utf8').trim();
}

const ROLE = {image: 'reference_image', video: 'reference_video', audio: 'reference_audio'};

/** Map a case to the H3ContextIRReq body. `urls` are public (presigned) URLs, one per condition. */
export function buildIRRequest(c, urls) {
  if (!c.text?.trim()) throw new Error('Context-IR needs a non-empty text item');
  const conds = c.conditions || [];
  if (urls.length !== conds.length) throw new Error('one url per condition');
  const content = [{type: 'text', text: c.text}];
  conds.forEach((x, i) => {
    const role = x.role || (c.mode === 'i2va' ? 'first_frame' : ROLE[x.type]);
    const t = `${x.type}_url`;
    content.push({type: t, [t]: {url: urls[i]}, role});
  });
  const frame = content.some((x) => /_frame$/.test(x.role || ''));
  const ref = content.some((x) => /^reference_/.test(x.role || ''));
  if (frame && ref) throw new Error('first/last frame and reference roles cannot be mixed');
  return {
    model: 'MiniMax-H3',
    content,
    duration: Math.min(15, Math.max(4, Math.round(c.duration))),
    ratio: !c.aspect || c.aspect === 'auto' ? 'adaptive' : c.aspect,
  };
}

async function call(method, p, body) {
  const r = await fetch(`${BASE}${p}`, {
    method,
    headers: {Authorization: `Bearer ${apiKey()}`, 'Content-Type': 'application/json'},
    body: body && JSON.stringify(body),
    signal: AbortSignal.timeout(60_000),
  });
  const text = await r.text();
  let j;
  try {
    j = JSON.parse(text);
  } catch {
    throw new Error(`MiniMax ${method} ${p}: HTTP ${r.status} non-JSON: ${text.slice(0, 200)}`);
  }
  const br = j.base_resp;
  if (!r.ok || (br && br.status_code && br.status_code !== 0)) throw new Error(`MiniMax ${method} ${p}: HTTP ${r.status} ${br ? `${br.status_code} ${br.status_msg}` : text.slice(0, 200)}`);
  return j;
}

/** Pull the task out of a query response (tolerates {task:{...}} or a bare task). */
export function parseTask(j) {
  const t = j.task || j;
  const usage = t.usage || {};
  const pin = usage.prompt_tokens || 0;
  const pout = usage.completion_tokens || 0;
  return {
    id: t.id || t.task_id,
    status: t.status,
    prompt: t.content?.prompt ?? null,
    error: t.error || null,
    usage,
    usd: +((pin * PRICE.in + pout * PRICE.out) / 1e6).toFixed(4),
  };
}

export async function officialIR(c, urls, {log = () => {}, pollMs = 5000, timeoutMs = 15 * 60_000} = {}) {
  const t0 = Date.now();
  const created = await call('POST', '/v2/h3_context_ir', buildIRRequest(c, urls));
  const id = created.task_id || created.task?.id;
  if (!id) throw new Error(`no task_id in create response: ${JSON.stringify(created).slice(0, 200)}`);
  log(`context-ir task ${id}`);
  while (Date.now() - t0 < timeoutMs) {
    await new Promise((r) => setTimeout(r, pollMs));
    const t = parseTask(await call('GET', `/v2/query/video_generation/${id}`));
    if (t.status === 'succeeded') {
      if (!t.prompt) throw new Error(`task ${id} succeeded without content.prompt`);
      return {...t, id, sec: +((Date.now() - t0) / 1000).toFixed(1)};
    }
    if (t.status === 'failed') throw new Error(`task ${id} failed: ${t.error?.code} ${t.error?.message}`);
  }
  throw new Error(`task ${id} timed out after ${timeoutMs / 1000}s`);
}
