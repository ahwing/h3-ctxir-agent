// The agent that stands in for the closed H3-Context-IR service.
//   probe media -> plan (JSON: intent, asset roles, speakers, verbatim lines, shot plan)
//   -> write IR per the official guide -> deterministic validator + independent critic -> repair (<= maxRepairs)
// Writer/repair: Claude Opus; critic: Claude Sonnet (fresh context, sees the media too).
import fs from 'node:fs';
import path from 'node:path';
import {computeLabels} from './labels.mjs';
import {probeCondition} from './probe.mjs';
import {converse, img, txt, parseJson, MODELS, cachePoint} from './bedrock.mjs';
import {validateIR, errorsOf} from './validate.mjs';

const ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
const guide = (f) => fs.readFileSync(path.join(ROOT, 'guides', f), 'utf8');

const MODE_TEXT = {
  ref2va: 'full-reference mode (Ref2VA). Output the six sections of the full-reference guide.',
  t2va: 'T2VA (text only). Output the three core fields of the base guide, no instruction line.',
  i2va: 'I2VA (first frame). Output the exact I2VA instruction line, a blank line, then the three core fields of the base guide.',
};

// Opt-in (H3_DIALOGUE_SPLIT=1): one sentence per <d>, learned from the official API. r3 showed no dialogue gain
// (0.91 vs 0.91) and a lower overall score, so it is off by default.
export const DIALOGUE_SPLIT = process.env.H3_DIALOGUE_SPLIT === '1';
const SPLIT_RULE = DIALOGUE_SPLIT ? '7. Spoken lines: put exactly ONE sentence in each <d>...</d> block. When the user gives several sentences, write them as consecutive <d> blocks for the same speaker, each introduced by "(Sx) says," / "(Sx) continues," with one short visible action or expression change between them. After the last block, state explicitly that the voice stops right after the final word and what the mouth/face does next (e.g. "Immediately after the final word, he closes his lips into a calm smile.").\n' : '';

export function systemPrompt(mode) {
  const g = mode === 'ref2va' ? `${guide('VIDEO_PROMPT_WRITING_GUIDE_ref_en.md')}\n\n---\n\nShared base guide:\n${guide('VIDEO_PROMPT_WRITING_GUIDE_base_en.md')}` : guide('VIDEO_PROMPT_WRITING_GUIDE_base_en.md');
  return `You are the context-understanding stage of the MiniMax H3 video system (the role of the hosted H3-Context-IR service).
You read the user's free-form multimodal instruction plus every reference asset, understand how text, images, video and audio relate to the target video, and serialize that understanding into the Context Intermediate Representation that H3-Base reads as its prompt.

Hard rules:
1. Follow the official guide below exactly: section names, order, label syntax, shot timing notation, speaker IDs, <d>[Language] ...</d> dialogue.
2. The REFERENCE LABEL TABLE you are given is how H3-Base itself numbers the assets. Use ONLY those labels. Users often count assets their own way ("图1", "audio 1", "reference audio 1" = the first file of that type they uploaded). Map the user's words to the real label from the table; e.g. if a video with a soundtrack comes first, the first audio FILE is <Audio 2>, not <Audio 1>.
3. Keep every user-provided spoken line verbatim in its original language inside <d>. Never translate or paraphrase it. Never carry over the words of a reference audio that is only a voice-timbre reference.
4. Everything else is written in English, concrete and visual. You may add missing visual, motion, camera and sound details that serve the user's intent, but never contradict the user or the assets.
5. The timeline must fit the target duration exactly. Cuts must lie strictly inside it.
6. Describe only what H3-Base can render in this one clip.
${SPLIT_RULE}${SPLIT_RULE ? 8 : 7}. Output only the IR text. No markdown fences, no commentary.

Official guide:
${g}`;
}

const fmtAsr = (a) => (a?.segments?.length ? `ASR (${a.language}, p=${a.language_prob}): ${a.segments.map((s) => `[${s.start}-${s.end}s] ${s.text}`).join(' ')}` : 'ASR: no intelligible speech detected (music, ambience or silence)');

/** Build the multimodal user content shared by plan / write / critic calls. */
export function mediaContent(c, probes, labels) {
  const blocks = [];
  const table = labels.map((l) => `${l.label} = condition #${l.conditionIndex + 1} (${c.conditions[l.conditionIndex].type}), the ${l.source}`).join('\n') || '(no reference assets)';
  blocks.push(txt(`USER INSTRUCTION (verbatim):\n${c.text}\n\nTARGET: mode ${c.mode}; duration ${c.duration}s; aspect ${c.aspect}; 24 fps; stereo audio.\n\nREFERENCE LABEL TABLE (authoritative, H3-Base numbering):\n${table}`));
  c.conditions.forEach((cond, i) => {
    const p = probes[i];
    const mine = labels.filter((l) => l.conditionIndex === i).map((l) => l.label).join(' + ');
    if (p.type === 'image') {
      blocks.push(txt(`Condition #${i + 1} -> ${mine}: image ${p.width}x${p.height}${cond.role ? `, role ${cond.role}` : ''}`));
      blocks.push(img(p.image));
    } else if (p.type === 'video') {
      blocks.push(txt(`Condition #${i + 1} -> ${mine}: video ${p.width}x${p.height}, ${p.duration}s, soundtrack: ${p.hasAudio ? `yes (mean ${p.loudness?.meanDb} dB). ${fmtAsr(p.asr)}` : 'none'}. Frames follow, each preceded by its timestamp.`));
      for (const f of p.frames) blocks.push(txt(`t=${f.t}s`), img(f.file));
    } else {
      blocks.push(txt(`Condition #${i + 1} -> ${mine}: audio ${p.duration}s, mean ${p.loudness?.meanDb} dB, max ${p.loudness?.maxDb} dB. ${fmtAsr(p.asr)} (You cannot hear it; rely on the ASR, loudness and the user's words for what it is.)`));
    }
  });
  return blocks;
}

const PLAN_ASK = `First, plan. Reply with ONE JSON object:
{"intent": "<one sentence>", "user_label_mapping": [{"user_said": "...", "label": "<real label>"}],
 "asset_roles": [{"label": "<Picture 1>", "role": "...", "relationship": "fully_preserved|partially_preserved|attribute_transfer|weak_reference|fully_copy|partially_copy|reference|unused", "why": "..."}],
 "task_types": ["reference generation", ...], "speakers": [{"id": "S1", "who": "...", "voice": "..."}],
 "dialogue_verbatim": ["..."], "shots": [{"n": 1, "start": 0.0, "end": 5.0, "content": "..."}],
 "sound": "...", "music": "...", "risks": ["things H3 might get wrong here"]}`;

const writeAsk = (plan) => `Your plan:\n${JSON.stringify(plan, null, 1)}\n\nNow write the final Context-IR, following the guide and the plan. Output only the IR.`;

const CRITIC_SYS = `You audit a MiniMax H3 Context-IR (the prompt H3-Base will read) against the user's instruction, the reference assets and the label table.
Find real problems only: wrong or swapped reference labels, a user request that was dropped or changed, invented content that contradicts the assets, dialogue not verbatim, voice-reference words wrongly spoken, timing that cannot fit, an asset used in the wrong role (copy vs reference), anything H3 cannot render in one clip.
Reply with ONE JSON object: {"verdict": "pass"|"fix", "issues": [{"severity": "high"|"low", "what": "...", "fix": "..."}]}`;

const repairAsk = (ir, issues) => `Your current IR:\n<<<\n${ir}\n>>>\n\nProblems found (validator + independent critic):\n${issues.map((i, n) => `${n + 1}. [${i.level || i.severity}] ${i.msg || `${i.what} -> ${i.fix}`}`).join('\n')}\n\nRewrite the complete IR fixing every problem; keep everything that was right. Output only the IR.`;

const clean = (s) => s.trim().replace(/^```[a-z]*\n?|```$/g, '').trim();

/**
 * Build a Context-IR for one case.
 * @returns {{ir, labels, plan, rounds:[{ir, issues, critic}], ledger, probes}}
 */
export async function buildIR(c, {workDir, maxRepairs = 2, useCritic = true, log = () => {}} = {}) {
  const ledger = [];
  fs.mkdirSync(workDir, {recursive: true});
  const probes = [];
  for (const cond of c.conditions) probes.push(await probeCondition(cond, path.resolve(ROOT, 'cases', cond.file), workDir));
  const labels = computeLabels(c.conditions, probes.map((p) => ({hasAudio: p.hasAudio})));
  log(`labels: ${labels.map((l) => l.label).join(' ') || '-'}`);
  const system = systemPrompt(c.mode);
  const media = mediaContent(c, probes, labels);
  const ctx = {mode: c.mode, duration: c.duration, labels, expectDialogue: c.expect_dialogue};

  const planR = await converse({model: MODELS.writer, system, content: [...media, ...cachePoint(), txt(PLAN_ASK)], maxTokens: 6000, purpose: 'plan', ledger});
  let plan;
  try {
    plan = parseJson(planR.text);
  } catch {
    plan = {raw: planR.text};
  }
  let ir = clean((await converse({model: MODELS.writer, system, content: [...media, ...cachePoint(), txt(MODE_TEXT[c.mode]), txt(writeAsk(plan))], maxTokens: 8000, purpose: 'write', ledger})).text);
  const rounds = [];
  for (let r = 0; ; r++) {
    const issues = validateIR(ir, ctx);
    let critic = null;
    if (useCritic) {
      const cr = await converse({model: MODELS.critic, system: CRITIC_SYS, content: [...media, ...cachePoint(), txt(`CONTEXT-IR TO AUDIT:\n<<<\n${ir}\n>>>`)], maxTokens: 3000, purpose: 'critic', ledger});
      try {
        critic = parseJson(cr.text);
      } catch {
        critic = {verdict: 'pass', issues: [], parseError: cr.text.slice(0, 200)};
      }
    }
    const high = (critic?.issues || []).filter((i) => i.severity === 'high');
    rounds.push({ir, issues, critic});
    log(`round ${r}: ${errorsOf(issues).length} validator errors, ${issues.length - errorsOf(issues).length} warnings, critic ${critic?.verdict ?? '-'} (${high.length} high)`);
    if ((!errorsOf(issues).length && !high.length) || r >= maxRepairs) break;
    const fixList = [...issues, ...(critic?.issues || [])];
    ir = clean((await converse({model: MODELS.writer, system, content: [...media, ...cachePoint(), txt(repairAsk(ir, fixList))], maxTokens: 8000, purpose: `repair${r + 1}`, ledger})).text);
  }
  // keep the round with the fewest validator errors, then fewest critic highs (latest wins ties)
  const score = (x) => errorsOf(x.issues).length * 10 + (x.critic?.issues || []).filter((i) => i.severity === 'high').length;
  const best = rounds.reduce((a, b) => (score(b) <= score(a) ? b : a));
  return {ir: best.ir, labels, plan, rounds, ledger, probes: probes.map(({frames, image, ...p}) => p)};
}
