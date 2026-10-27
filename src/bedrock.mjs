// Thin Bedrock Converse wrapper: text + image blocks, returns text, usage and an estimated USD cost.
import fs from 'node:fs';
import crypto from 'node:crypto';
import {BedrockRuntimeClient, ConverseCommand} from '@aws-sdk/client-bedrock-runtime';
import {SignatureV4} from '@smithy/signature-v4';
import {defaultProvider} from '@aws-sdk/credential-provider-node';

export const MODELS = {
  writer: process.env.IR_WRITER_MODEL || 'us.anthropic.claude-opus-5-5',
  critic: process.env.IR_CRITIC_MODEL || 'us.anthropic.claude-sonnet-5-5',
  judge: process.env.JUDGE_MODEL || 'us.openai.gpt-6-sol', // video judge: sees frames + ASR only, never the prompt
  // The IR-text judge reads the prompts, so it must be from a different vendor than the writer.
  irJudge: process.env.IR_JUDGE_MODEL || process.env.JUDGE_MODEL || 'us.openai.gpt-6-sol',
};
// List prices USD / 1M tokens, estimates only.
// Claude: Anthropic list 2026-10 (Bedrock assumed equal); cache write 1.25x, read 0.1x input.
// GPT-6.1 Sol/Terra: Bedrock model cards (US CRIS, short context): cache write 1.25x, read 0.05x.
// Other GPT-6 rows: third-party summaries 2026-10 (+10% US CRIS), cache multipliers assumed.
const PRICE = {
  'us.anthropic.claude-opus-5-5': {in: 4, out: 20},
  'us.anthropic.claude-sonnet-5-5': {in: 2, out: 10},
  'us.openai.gpt-6-sol': {in: 2.2, out: 11, cw: 1, cr: 0.1},
  'us.openai.gpt-6.1-sol': {in: 2.2, out: 11, cw: 1.25, cr: 0.05},
  'us.openai.gpt-6.1-terra': {in: 1.1, out: 5.5, cw: 1.25, cr: 0.05},
  'us.openai.gpt-6-astra': {in: 11, out: 55, cw: 1, cr: 0.1},
  'us.openai.gpt-6-luna': {in: 0.11, out: 0.55, cw: 1, cr: 0.1},
};
export const usd = (model, u) => {
  const p = PRICE[model] || {in: 5, out: 15};
  const cw = u?.cacheWriteInputTokens || 0;
  const cr = u?.cacheReadInputTokens || 0;
  // Converse inputTokens excludes cached tokens, so the three input buckets add up.
  return ((u?.inputTokens || 0) * p.in + cw * p.in * (p.cw ?? 1.25) + cr * p.in * (p.cr ?? 0.1) + (u?.outputTokens || 0) * p.out) / 1e6;
};

// Prompt caching (Converse cachePoint). On by default; H3_PROMPT_CACHE=0 disables it.
export const CACHE_ON = process.env.H3_PROMPT_CACHE !== '0';
/** Cache breakpoint block; put it right after the last stable block (e.g. the media). */
export const cachePoint = () => (CACHE_ON ? [{cachePoint: {type: 'default'}}] : []);

let client;
const getClient = () => (client ??= new BedrockRuntimeClient({region: process.env.BEDROCK_REGION || 'us-east-1', maxAttempts: 6}));

export const img = (file) => ({image: {format: /\.png$/i.test(file) ? 'png' : 'jpeg', source: {bytes: fs.readFileSync(file)}}});
export const txt = (text) => ({text});

// --- OpenAI GPT prompt caching -------------------------------------------------
// Converse only has an implicit breakpoint at the END of the prompt, so a changing tail
// (plan/write/repair asks) never reads back (measured: 0 reads in rgpt). The Responses API
// supports explicit breakpoints; measured on gpt-6.1-terra a breakpoint on the system text
// reads back every time, a breakpoint after image blocks never does. So GPT + CACHE_ON goes
// through Responses with one breakpoint on the system prompt.
const sha256 = class {
  constructor(k) {
    this.h = k ? crypto.createHmac('sha256', k) : crypto.createHash('sha256');
  }
  update(d) {
    this.h.update(d);
  }
  async digest() {
    return new Uint8Array(this.h.digest());
  }
};
let signer;
const BP = {prompt_cache_breakpoint: {mode: 'explicit'}};

/** Converse-style blocks -> Responses API input (exported for tests). */
export function toResponsesBody({model, system, blocks, maxTokens, cacheKey}) {
  const part = (b) =>
    b.image ? {type: 'input_image', image_url: `data:image/${b.image.format};base64,${Buffer.from(b.image.source.bytes).toString('base64')}`} : {type: 'input_text', text: b.text};
  const input = [];
  if (system) input.push({type: 'message', role: 'developer', content: [{type: 'input_text', text: system, ...BP}]});
  input.push({type: 'message', role: 'user', content: blocks.filter((b) => !b.cachePoint).map(part)});
  return {model, prompt_cache_key: cacheKey, prompt_cache_options: {mode: 'explicit'}, max_output_tokens: maxTokens, input};
}

/** Responses usage -> Converse-style usage (inputTokens excludes cached buckets). */
export function fromResponsesUsage(u = {}) {
  const cr = u.input_tokens_details?.cached_tokens || 0;
  const cw = u.input_tokens_details?.cache_write_tokens || 0;
  return {inputTokens: (u.input_tokens || 0) - cr - cw, outputTokens: u.output_tokens || 0, totalTokens: u.total_tokens || 0, cacheReadInputTokens: cr, cacheWriteInputTokens: cw};
}

async function responsesCall(body) {
  const region = process.env.BEDROCK_REGION || 'us-east-1';
  const host = `bedrock-runtime.${region}.amazonaws.com`;
  signer ??= new SignatureV4({service: 'bedrock', region, credentials: defaultProvider(), sha256});
  const payload = JSON.stringify(body);
  for (let attempt = 0; ; attempt++) {
    const s = await signer.sign({method: 'POST', protocol: 'https:', hostname: host, path: '/openai/v1/responses', headers: {host, 'content-type': 'application/json'}, body: payload});
    const r = await fetch(`https://${host}${s.path}`, {method: 'POST', headers: s.headers, body: payload});
    const j = await r.json().catch(() => ({}));
    if (r.ok) return j;
    if (attempt < 5 && (r.status === 429 || r.status >= 500)) {
      await new Promise((ok) => setTimeout(ok, 2000 * 2 ** attempt));
      continue;
    }
    throw new Error(`Responses ${r.status}: ${JSON.stringify(j.error || j).slice(0, 300)}`);
  }
}

/** One call. `ledger` (optional array) receives {model, purpose, usage, usd, sec}. */
export async function converse({model, system, content, maxTokens = 16000, purpose = '', ledger}) {
  const t0 = Date.now();
  // Claude: native Converse cachePoint. GPT: Responses API with a system breakpoint.
  const claude = /anthropic\.claude/.test(model);
  let blocks = typeof content === 'string' ? [{text: content}] : content;
  if (!claude && CACHE_ON && /openai\.gpt/.test(model)) {
    const cacheKey = `h3ctx:${crypto.createHash('sha256').update(system || '').digest('hex').slice(0, 16)}`;
    const j = await responsesCall(toResponsesBody({model, system, blocks, maxTokens, cacheKey}));
    const text = (j.output || []).filter((o) => o.type === 'message').flatMap((o) => o.content || []).map((c) => c.text || '').join('');
    const usage = fromResponsesUsage(j.usage);
    const rec = {model, purpose, usage, usd: +usd(model, usage).toFixed(4), sec: +((Date.now() - t0) / 1000).toFixed(1), stopReason: j.incomplete_details?.reason || j.status, api: 'responses'};
    ledger?.push(rec);
    return {text, ...rec};
  }
  if (!claude) blocks = blocks.filter((b) => !b.cachePoint);
  const r = await getClient().send(
    new ConverseCommand({
      modelId: model,
      system: system ? [{text: system}, ...(claude ? cachePoint() : [])] : undefined,
      messages: [{role: 'user', content: blocks}],
      inferenceConfig: {maxTokens},
    }),
  );
  const text = (r.output?.message?.content || []).map((b) => b.text || '').join('');
  const rec = {model, purpose, usage: r.usage, usd: +usd(model, r.usage).toFixed(4), sec: +((Date.now() - t0) / 1000).toFixed(1), stopReason: r.stopReason};
  ledger?.push(rec);
  return {text, ...rec};
}

/** First JSON object in a reply (tolerates fences / prose). */
export function parseJson(text) {
  const s = String(text || '').replace(/```(?:json)?/g, '');
  const a = s.indexOf('{');
  const b = s.lastIndexOf('}');
  if (a < 0 || b <= a) throw new Error(`no JSON object in reply: ${s.slice(0, 160)}`);
  return JSON.parse(s.slice(a, b + 1));
}
