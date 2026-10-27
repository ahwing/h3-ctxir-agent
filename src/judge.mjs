// Scoring: deterministic dialogue checks on the output audio + a blind LLM judge from a different vendor (GPT-6).
import {converse, img, txt, parseJson, MODELS} from './bedrock.mjs';

export const normSpeech = (s) => String(s || '').toLowerCase().normalize('NFKC').replace(/[\p{P}\p{S}\s]+/gu, '');

/** Semi-global edit distance: cost of finding `needle` anywhere inside `hay` (free prefix/suffix in hay). */
export function infixDistance(needle, hay) {
  const a = [...needle];
  const b = [...hay];
  let prev = new Array(b.length + 1).fill(0);
  for (let i = 1; i <= a.length; i++) {
    const cur = [i];
    for (let j = 1; j <= b.length; j++) cur[j] = Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1));
    prev = cur;
  }
  return Math.min(...prev);
}

/** Per expected line: 1 - CER of its best match inside the ASR transcript (1 = spoken exactly). */
export function dialogueScore(expectLines, asrText) {
  const hay = normSpeech(asrText);
  const per = expectLines.map((l) => {
    const n = normSpeech(l);
    return {line: l, acc: n.length ? +Math.max(0, 1 - infixDistance(n, hay) / n.length).toFixed(3) : 1};
  });
  return {per, mean: per.length ? +(per.reduce((s, p) => s + p.acc, 0) / per.length).toFixed(3) : null};
}

/** Deterministic shuffle so the judge never sees arms in a fixed order. */
export function shuffled(arr, seedStr) {
  let h = 2166136261;
  for (const ch of seedStr) h = Math.imul(h ^ ch.charCodeAt(0), 16777619) >>> 0;
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i--) {
    h = Math.imul(h ^ (h >>> 13), 1103515245) >>> 0;
    const j = h % (i + 1);
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

const CRITERIA = ['instruction_following', 'reference_fidelity', 'visual_quality', 'motion_camera', 'audio_dialogue', 'overall'];

const VIDEO_SYS = `You are a strict, blind evaluator of AI-generated video clips. Several candidate clips were made from the SAME user instruction and the SAME reference assets; only the hidden text prompt differs. You see each clip as a 3x3 contact sheet (9 frames evenly spaced in time, read left-to-right, top-to-bottom) plus an ASR transcript of its audio.
Score each candidate 1-10 on: instruction_following (does it do what the user asked, including requested lines, actions, camera), reference_fidelity (subjects, scenes and styles from the references kept as the user wanted), visual_quality (artifacts, deformation, blur, fake text), motion_camera (requested motion/camera, temporal coherence), audio_dialogue (requested speech present and correct, no unwanted speech, sensible sound), overall.
Judge only what you can see/read. Use the full scale. Reply with ONE JSON object: {"candidates": {"A": {"instruction_following": n, ..., "overall": n, "notes": "..."}, ...}, "ranking": ["best letter", ...]}`;

/**
 * @param {{text:string, refBlocks:any[], clips:{key:string, sheet:string, asrText:string, meanDb:number|null}[], seed:string, ledger?:any[]}} o
 * @returns {{byKey: Record<string, any>, ranking: string[], letters: Record<string,string>, raw: string}}
 */
export async function judgeVideos({text, refBlocks, clips, seed, ledger}) {
  const order = shuffled(clips, seed);
  const letters = {};
  const content = [txt(`USER INSTRUCTION:\n${text}\n\nREFERENCE ASSETS:`), ...refBlocks, txt('CANDIDATES:')];
  order.forEach((c, i) => {
    const L = String.fromCharCode(65 + i);
    letters[L] = c.key;
    content.push(txt(`Candidate ${L} - audio: mean ${c.meanDb ?? '?'} dB; ASR transcript: "${c.asrText || '(no speech)'}"`), img(c.sheet));
  });
  const r = await converse({model: MODELS.judge, system: VIDEO_SYS, content, maxTokens: 6000, purpose: 'judge-video', ledger});
  const j = parseJson(r.text);
  return parseJudge(j, letters, r.text);
}

/** Map letter-keyed judge JSON back to arm keys; validates scores. */
export function parseJudge(j, letters, raw = '') {
  const byKey = {};
  for (const [L, key] of Object.entries(letters)) {
    const s = j.candidates?.[L];
    if (!s) throw new Error(`judge omitted candidate ${L}`);
    const out = {notes: s.notes || ''};
    for (const k of CRITERIA) {
      const v = Number(s[k]);
      if (!Number.isFinite(v) || v < 1 || v > 10) throw new Error(`judge score ${L}.${k} invalid: ${s[k]}`);
      out[k] = v;
    }
    byKey[key] = out;
  }
  const ranking = (j.ranking || []).map((L) => letters[L]).filter(Boolean);
  return {byKey, ranking, letters, raw};
}

const IR_SYS = `You compare candidate prompts written for the MiniMax H3 video model. Each was produced from the same user instruction and assets by a context-understanding stage that must follow the official prompt guide (given). You do not know who wrote which.
Score each 1-10 on: intent_fidelity (keeps everything the user asked, adds nothing contradictory), guide_compliance (structure, labels, timing, <d> dialogue, English), visual_specificity (concrete, renderable detail), audio_design (soundscape and music useful and consistent), overall.
Reply with ONE JSON object: {"candidates": {"A": {"intent_fidelity": n, "guide_compliance": n, "visual_specificity": n, "audio_design": n, "overall": n, "notes": "..."}, ...}, "ranking": [...]}`;
const IR_CRITERIA = ['intent_fidelity', 'guide_compliance', 'visual_specificity', 'audio_design', 'overall'];

export async function judgeIRs({text, guideText, refBlocks, irs, seed, ledger}) {
  const order = shuffled(irs, seed);
  const letters = {};
  const content = [txt(`OFFICIAL GUIDE:\n${guideText}\n\nUSER INSTRUCTION:\n${text}\n\nREFERENCE ASSETS:`), ...refBlocks];
  order.forEach((c, i) => {
    const L = String.fromCharCode(65 + i);
    letters[L] = c.key;
    content.push(txt(`Candidate ${L}:\n<<<\n${c.ir}\n>>>`));
  });
  const r = await converse({model: MODELS.irJudge, system: IR_SYS, content, maxTokens: 6000, purpose: 'judge-ir', ledger});
  const j = parseJson(r.text);
  const byKey = {};
  for (const [L, key] of Object.entries(letters)) {
    const s = j.candidates?.[L];
    if (!s) throw new Error(`IR judge omitted ${L}`);
    byKey[key] = Object.fromEntries([...IR_CRITERIA.map((k) => [k, Number(s[k])]), ['notes', s.notes || '']]);
  }
  return {byKey, ranking: (j.ranking || []).map((L) => letters[L]).filter(Boolean), letters};
}
