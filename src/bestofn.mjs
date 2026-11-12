// Best-of-N with a cheap, deterministic verifier: generate a clip, check it (QC + ASR dialogue + voice-reference
// leak), and re-roll the seed only when the check fails. No LLM judge in the loop, so it can run on every request.
import {leakShare, LEAK_SHARE} from './validate.mjs';

export const DIALOGUE_MIN = +(process.env.BESTOFN_DIALOGUE_MIN || 0.85);

/**
 * @param {{qc:{pass:boolean}, asr?:{text:string}, dialogue?:{mean:number|null}, forbidden?:string[]}} clip  scored clip record
 * @param {{expect_dialogue?:string[], text?:string}} c  the case / request
 * @param {{label:string,text:string}[]} [refSpeech]  ASR of voice-reference audio (from buildIR)
 * @returns {{pass:boolean, score:number, reasons:string[]}}  score orders clips: higher is better
 */
export function verifyClip(clip, c, refSpeech = []) {
  const reasons = [];
  if (!clip?.qc?.pass) return {pass: false, score: -1, reasons: ['qc_failed']};
  const wantDialogue = (c.expect_dialogue || []).length > 0;
  const acc = wantDialogue ? clip.dialogue?.mean ?? 0 : 1;
  if (wantDialogue && acc < DIALOGUE_MIN) reasons.push(`dialogue_${acc}`);
  if (clip.forbidden?.length) reasons.push(`forbidden_${clip.forbidden.join('/')}`);
  const said = clip.asr?.text || '';
  const leak = Math.max(0, ...refSpeech.map((r) => leakShare(r.text, said, c.text)));
  if (leak >= LEAK_SHARE) reasons.push(`voice_ref_leak_${leak}`);
  const pass = !reasons.length;
  return {pass, score: +((pass ? 1 : 0) + acc - leak - (clip.forbidden?.length ? 1 : 0)).toFixed(3), reasons};
}

/** Pick from already-generated attempts: the first that passes, else the best score (earliest wins ties). */
export function pickBest(attempts) {
  if (!attempts.length) return null;
  return attempts.find((a) => a.verdict.pass) || attempts.reduce((a, b) => (b.verdict.score > a.verdict.score ? b : a));
}

/**
 * Online loop. Injected I/O keeps it testable: generate(seed) -> file, score(file) -> clip record.
 * Stops at the first passing clip; otherwise returns the best of all `seeds`.
 */
export async function generateBestOfN({c, seeds, refSpeech = [], generate, score, log = () => {}}) {
  const attempts = [];
  for (const seed of seeds) {
    const file = await generate(seed);
    const clip = await score(file);
    const verdict = verifyClip(clip, c, refSpeech);
    attempts.push({seed, file, clip, verdict});
    log(`[best-of-n] ${c.id} s${seed}: ${verdict.pass ? 'pass' : `fail (${verdict.reasons.join(', ')})`}`);
    if (verdict.pass) break;
  }
  return {chosen: pickBest(attempts), attempts};
}

/**
 * Offline estimate from an existing run: for each arm, simulate "stop at the first passing seed, up to N" over the
 * seeds that were already generated and judged, and read the judge's score of the clip that would have shipped.
 * @param {{id:string}[]} cases
 * @param {Record<string, {clips:Record<string,any>, judge:Record<string,{byKey:Record<string,{overall:number}>}>}>} scoresByCase
 * @param {number[]} seeds  in generation order
 * @param {Record<string, {label:string,text:string}[]>} [refSpeechByCase]
 */
export function simulate(cases, scoresByCase, seeds, refSpeechByCase = {}, {usable = 7} = {}) {
  const arms = {};
  for (const c of cases) {
    const sc = scoresByCase[c.id];
    if (!sc) continue;
    const armKeys = [...new Set(Object.keys(sc.clips).map((f) => f.split('__')[1]))];
    for (const arm of armKeys) {
      const tries = seeds
        .map((seed) => {
          const f = `${c.id}__${arm}__s${seed}.mp4`;
          const overall = sc.judge?.[seed]?.byKey?.[arm]?.overall;
          return sc.clips[f] && Number.isFinite(overall) ? {seed, overall, verdict: verifyClip(sc.clips[f], c, refSpeechByCase[c.id])} : null;
        })
        .filter(Boolean);
      if (!tries.length) continue;
      const a = (arms[arm] ??= {byN: {}, verifier: {tp: 0, fp: 0, fn: 0, tn: 0}});
      for (const t of tries) {
        const good = t.overall >= usable;
        a.verifier[t.verdict.pass ? (good ? 'tp' : 'fp') : good ? 'fn' : 'tn']++;
      }
      for (let n = 1; n <= tries.length; n++) {
        const pool = tries.slice(0, n);
        const firstPass = pool.findIndex((t) => t.verdict.pass);
        const chosen = pickBest(pool);
        const b = (a.byN[n] ??= {shipped: [], attempts: []});
        b.shipped.push(chosen.overall);
        b.attempts.push(firstPass >= 0 ? firstPass + 1 : n);
      }
    }
  }
  const mean = (xs) => +(xs.reduce((s, x) => s + x, 0) / xs.length).toFixed(2);
  return Object.fromEntries(
    Object.entries(arms).map(([arm, a]) => [
      arm,
      {
        verifier: a.verifier,
        byN: Object.fromEntries(Object.entries(a.byN).map(([n, b]) => [n, {cases: b.shipped.length, meanOverall: mean(b.shipped), usableRate: +(b.shipped.filter((x) => x >= usable).length / b.shipped.length).toFixed(2), worst: Math.min(...b.shipped), meanAttempts: mean(b.attempts)}])),
      },
    ]),
  );
}
