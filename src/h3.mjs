// H3-Base client for the two single-GPU Ref2VA SGLang servers on the G7e (local SSM forwards 39010 / 39011).
// The G7e has no S3 permissions, so media go to S3 here and the request carries presigned GET URLs.
import crypto from 'node:crypto';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {S3Client, PutObjectCommand, HeadObjectCommand, GetObjectCommand} from '@aws-sdk/client-s3';
import {getSignedUrl} from '@aws-sdk/s3-request-presigner';

// Staging bucket: H3CTX_BUCKET, else ~/.config/h3-ctxir/bucket (kept out of the repo).
const BUCKET_FILE = path.join(os.homedir(), '.config/h3-ctxir/bucket');
export const BUCKET = process.env.H3CTX_BUCKET || (fs.existsSync(BUCKET_FILE) ? fs.readFileSync(BUCKET_FILE, 'utf8').trim() : '');
export const PREFIX = 'h3ctx/';
export const h3Urls = () => (process.env.H3_URLS || 'http://127.0.0.1:39010,http://127.0.0.1:39011').split(',').map((u) => u.trim().replace(/\/+$/, '')).filter(Boolean);
// Same sampler settings the Studio pipeline uses with the merged Turbo 4-step LoRA (shared/h3sglang.mjs).
export const SAMPLER = {num_inference_steps: 5, flow_shift: 12.0, audio_flow_shift: 3.0};
const CT = {'.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png', '.mp4': 'video/mp4', '.mp3': 'audio/mpeg', '.wav': 'audio/wav'};

let s3;
const S3 = () => (s3 ??= new S3Client({region: 'us-east-1'}));

/** Upload (idempotent, content-addressed) and return a presigned GET URL. */
export async function stage(file, expiresIn = 6 * 3600) {
  if (!BUCKET) throw new Error('set H3CTX_BUCKET (or ~/.config/h3-ctxir/bucket) to an S3 bucket for staging media');
  const ext = path.extname(file).toLowerCase();
  const body = fs.readFileSync(file);
  const Key = `${PREFIX}${crypto.createHash('sha256').update(body).digest('hex')}${ext}`;
  const exists = await S3().send(new HeadObjectCommand({Bucket: BUCKET, Key})).then(() => true, () => false);
  if (!exists) await S3().send(new PutObjectCommand({Bucket: BUCKET, Key, Body: body, ContentType: CT[ext] || 'application/octet-stream'}));
  return getSignedUrl(S3(), new GetObjectCommand({Bucket: BUCKET, Key}), {expiresIn});
}

/** Request body. Every arm of a case gets the identical body except `prompt`. */
export function buildRequest({prompt, uris, conditions, duration, aspect, seed}) {
  if (!prompt || typeof prompt !== 'string') throw new Error('prompt required');
  if (uris.length !== conditions.length) throw new Error('one uri per condition');
  if (!conditions.length) throw new Error('ref2va needs at least one condition');
  const sec = Math.min(15, Math.max(4, duration));
  return {
    model: 'MiniMaxAI/MiniMax-H3',
    task: 'ref2va',
    prompt,
    seconds: sec,
    conditions: conditions.map((c, i) => ({type: c.type, uri: uris[i], role: 'reference'})),
    target: {short_edge: 768, aspect_ratio: aspect || 'auto', duration_seconds: sec},
    num_outputs_per_prompt: 1,
    ...SAMPLER,
    seed,
  };
}

export async function health(u) {
  try {
    return (await fetch(`${u}/health`, {signal: AbortSignal.timeout(3000)})).ok;
  } catch {
    return false;
  }
}

/** Submit, poll until done, download. Returns {id, mp4, sec, server, inferenceSec, peakMb}. */
export async function generate(u, body, outFile, {timeoutMs = 30 * 60_000, log = () => {}} = {}) {
  const t0 = Date.now();
  const r = await fetch(`${u}/v1/videos`, {method: 'POST', headers: {'content-type': 'application/json'}, body: JSON.stringify(body), signal: AbortSignal.timeout(60_000)});
  const d = await r.json().catch(() => ({}));
  if (!r.ok || !d.id) throw new Error(`submit ${r.status}: ${JSON.stringify(d).slice(0, 400)}`);
  for (;;) {
    if (Date.now() - t0 > timeoutMs) throw new Error(`timeout ${d.id}`);
    await new Promise((res) => setTimeout(res, 5000));
    const s = await (await fetch(`${u}/v1/videos/${d.id}`, {signal: AbortSignal.timeout(20_000)})).json().catch(() => ({}));
    if (s.status === 'failed') throw new Error(`failed ${d.id}: ${JSON.stringify(s.error).slice(0, 400)}`);
    if (s.status !== 'completed') continue;
    const c = await fetch(`${u}/v1/videos/${d.id}/content`, {signal: AbortSignal.timeout(180_000)});
    if (!c.ok) throw new Error(`content ${c.status}`);
    fs.writeFileSync(outFile, Buffer.from(await c.arrayBuffer()));
    log(`done ${path.basename(outFile)} on ${u} in ${((Date.now() - t0) / 1000).toFixed(0)}s`);
    return {id: d.id, mp4: outFile, sec: +((Date.now() - t0) / 1000).toFixed(1), server: u, inferenceSec: s.inference_time_s, peakMb: s.peak_memory_mb};
  }
}
