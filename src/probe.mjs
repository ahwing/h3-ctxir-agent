// Media probe: the agent's eyes and ears. Turns each condition file into things a text+image LLM can use:
//   image -> resized JPEG
//   video -> duration, soundtrack flag, N timestamped frames, ASR of the soundtrack, loudness
//   audio -> duration, ASR, loudness
// Also used by the harness QC (ffprobe / full decode / frame std) and the judge (frame sheet + ASR of outputs).
import {execFile} from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {promisify} from 'node:util';

const run = promisify(execFile);
const ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
export const ASR_PY = process.env.ASR_PY || path.join(path.resolve(path.dirname(new URL(import.meta.url).pathname), '..'), '.venv/bin/python');

export async function ffprobe(file) {
  const {stdout} = await run('ffprobe', ['-v', 'error', '-show_entries', 'format=duration:stream=codec_type,codec_name,width,height,r_frame_rate,sample_rate,channels', '-of', 'json', file]);
  const j = JSON.parse(stdout);
  const v = j.streams.find((s) => s.codec_type === 'video');
  const a = j.streams.find((s) => s.codec_type === 'audio');
  return {
    duration: +j.format?.duration || 0,
    video: v ? {codec: v.codec_name, width: v.width, height: v.height, fps: v.r_frame_rate} : null,
    audio: a ? {codec: a.codec_name, sampleRate: +a.sample_rate, channels: a.channels} : null,
  };
}

/** Mean / max volume in dB (ffmpeg volumedetect). */
export async function loudness(file) {
  const {stderr} = await run('ffmpeg', ['-v', 'info', '-nostats', '-i', file, '-vn', '-af', 'volumedetect', '-f', 'null', '-'], {maxBuffer: 1 << 24});
  const mean = /mean_volume: (-?[\d.]+) dB/.exec(stderr);
  const max = /max_volume: (-?[\d.]+) dB/.exec(stderr);
  return {meanDb: mean ? +mean[1] : null, maxDb: max ? +max[1] : null};
}

export async function asr(file, {model = process.env.ASR_MODEL || 'small'} = {}) {
  const {stdout} = await run(ASR_PY, [path.join(ROOT, 'src/asr.py'), file, '--model', model], {maxBuffer: 1 << 24, timeout: 600_000});
  return JSON.parse(stdout.trim().split('\n').pop());
}

export async function resizeImage(file, outFile, maxEdge = 1280) {
  await run('ffmpeg', ['-v', 'error', '-y', '-i', file, '-vf', `scale='min(${maxEdge},iw)':-2`, '-q:v', '3', '-frames:v', '1', outFile]);
  return outFile;
}

/** n evenly spaced frames (at (i+0.5)/n of the duration). Returns [{t, file}]. */
export async function frames(file, n, outDir, {width = 640, duration} = {}) {
  const d = duration ?? (await ffprobe(file)).duration;
  fs.mkdirSync(outDir, {recursive: true});
  const out = [];
  for (let i = 0; i < n; i++) {
    const t = +((d * (i + 0.5)) / n).toFixed(2);
    const f = path.join(outDir, `f${String(i).padStart(2, '0')}.jpg`);
    await run('ffmpeg', ['-v', 'error', '-y', '-ss', String(t), '-i', file, '-frames:v', '1', '-vf', `scale=${width}:-2`, '-q:v', '4', f]);
    out.push({t, file: f});
  }
  return out;
}

/** Contact sheet (cols x rows), tiles evenly spaced in time, left-to-right then top-to-bottom.
 * (This ffmpeg build has no drawtext, so the tile times are returned instead of burned in.) */
export async function contactSheet(file, outFile, {cols = 3, rows = 3, tileW = 448} = {}) {
  const d = (await ffprobe(file)).duration;
  const n = cols * rows;
  const fps = n / Math.max(0.5, d);
  await run('ffmpeg', ['-v', 'error', '-y', '-i', file, '-vf', `fps=${fps.toFixed(4)},scale=${tileW}:-2,tile=${cols}x${rows}`, '-frames:v', '1', '-q:v', '4', outFile]);
  return outFile;
}

/** Probe one condition for the agent. */
export async function probeCondition(cond, absFile, workDir) {
  const base = path.join(workDir, `cond_${path.basename(absFile).replace(/\W+/g, '_')}`);
  fs.mkdirSync(base, {recursive: true});
  const info = await ffprobe(absFile);
  if (cond.type === 'image') {
    return {type: 'image', width: info.video?.width, height: info.video?.height, image: await resizeImage(absFile, path.join(base, 'img.jpg'))};
  }
  const hasAudio = !!info.audio;
  const res = {type: cond.type, duration: +info.duration.toFixed(2), hasAudio};
  if (cond.type === 'video') {
    res.width = info.video?.width;
    res.height = info.video?.height;
    res.frames = await frames(absFile, Math.min(10, Math.max(6, Math.round(info.duration))), path.join(base, 'frames'), {duration: info.duration});
  }
  if (hasAudio) {
    res.loudness = await loudness(absFile);
    res.asr = await asr(absFile);
  }
  return res;
}

/** Spatial std of grayscale 160x90 frames (blank ~2-3, grey mush ~10, normal 24-47; see aws-notes H3 note). */
export async function frameStd(file, n = 6) {
  const d = (await ffprobe(file)).duration;
  const vals = [];
  for (let i = 0; i < n; i++) {
    const t = ((d * (i + 0.5)) / n).toFixed(2);
    const {stdout} = await run('ffmpeg', ['-v', 'error', '-ss', t, '-i', file, '-frames:v', '1', '-vf', 'scale=160:90,format=gray', '-f', 'rawvideo', '-'], {encoding: 'buffer', maxBuffer: 1 << 20});
    const px = [...stdout];
    const m = px.reduce((a, b) => a + b, 0) / px.length;
    vals.push(Math.sqrt(px.reduce((a, b) => a + (b - m) ** 2, 0) / px.length));
  }
  return {min: +Math.min(...vals).toFixed(2), mean: +(vals.reduce((a, b) => a + b, 0) / n).toFixed(2)};
}

/** Output QC: streams, full decode with zero errors, not blank. */
export async function qcClip(file) {
  const p = await ffprobe(file);
  const {stderr} = await run('ffmpeg', ['-v', 'error', '-i', file, '-f', 'null', '-'], {maxBuffer: 1 << 24}).catch((e) => ({stderr: String(e.stderr || e.message)}));
  const decodeErrors = stderr.split('\n').filter(Boolean).length;
  const std = p.video ? await frameStd(file) : {min: 0, mean: 0};
  const pass = !!p.video && !!p.audio && decodeErrors === 0 && std.mean >= 15;
  return {pass, duration: +p.duration.toFixed(3), video: p.video, audio: p.audio, decodeErrors, frameStd: std, bytes: fs.statSync(file).size};
}
