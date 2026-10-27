// Deterministic checks of a Context-IR against the official MiniMax H3 prompt guides
// (guides/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md and _base_en.md). Errors trigger an agent repair round; warnings are
// passed to the repair round too but never block.

export const REF_SECTIONS = ['subject_definitions', 'summary', 'retention_analysis', 'detailed_description', 'overall_soundscape', 'non_diegetic_music'];
export const BASE_SECTIONS = ['integrated_multimodal_description', 'overall_soundscape', 'non_diegetic_music'];
const TASK_TYPES = new Set(['keyframe completion', 'reference generation', 'video editing', 'video continuation', 'audio reuse', 'audio reference']);
const VIS_MARKERS = new Set(['fully_preserved', 'partially_preserved', 'attribute_transfer', 'weak_reference']);
const AUD_MARKERS = new Set(['fully_copy', 'partially_copy', 'reference', 'weak_reference']);
export const I2VA_LINE = 'For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.';

/** Split "name:\n body" / "name: body" sections. Returns {order:[names], map:{name: body}}. */
export function splitSections(ir, names) {
  const re = new RegExp(`^(${names.join('|')}):[ \\t]*`, 'gm');
  const hits = [...ir.matchAll(re)];
  const map = {};
  const order = [];
  hits.forEach((m, i) => {
    const end = i + 1 < hits.length ? hits[i + 1].index : ir.length;
    map[m[1]] = ir.slice(m.index + m[0].length, end).trim();
    order.push(m[1]);
  });
  return {order, map, preamble: hits.length ? ir.slice(0, hits[0].index).trim() : ir.trim()};
}

const norm = (s) => String(s).replace(/\s+/g, '').replace(/[“”"]/g, '');
const CJK = /[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af]/;
/** Text with <d>..</d> blocks and "quoted visible text" removed (those may stay in the original language). */
const stripAllowed = (s) => s.replace(/<d>[\s\S]*?<\/d>/g, ' ').replace(/"[^"\n]*"/g, ' ').replace(/“[^”\n]*”/g, ' ');
const words = (s) => (stripAllowed(s).match(/[A-Za-z][A-Za-z'-]*/g) || []).length;

/** "MM:SS.mmm" -> seconds */
const ts = (m, s) => +m * 60 + +s;

export function checkShots(body, duration, issues) {
  const shots = [...body.matchAll(/\[Shot (\d+)\]([^\n]{0,60})/g)];
  if (!shots.length) return issues.push({level: 'error', code: 'no_shots', msg: 'no [Shot N] markers'});
  let prev = 0;
  shots.forEach((m, i) => {
    const n = +m[1];
    if (n !== i + 1) issues.push({level: 'error', code: 'shot_order', msg: `shot markers must be sequential; found [Shot ${n}] at position ${i + 1}`});
    const t = /^\s*At (\d{2}):(\d{2}(?:\.\d{1,3})?)/.exec(m[2]);
    if (i === 0) {
      if (t) issues.push({level: 'error', code: 'shot1_time', msg: '[Shot 1] must not carry a timestamp'});
      return;
    }
    if (!t) return issues.push({level: 'error', code: 'shot_time_missing', msg: `[Shot ${n}] must start with "At MM:SS.mmm,"`});
    const sec = ts(t[1], t[2]);
    if (sec <= prev) issues.push({level: 'error', code: 'shot_time_order', msg: `[Shot ${n}] cut time ${sec}s is not after the previous cut`});
    if (sec >= duration) issues.push({level: 'error', code: 'shot_time_range', msg: `[Shot ${n}] cut time ${sec}s is outside the ${duration}s video`});
    prev = sec;
  });
}

export function checkDialogue(ir, expect, issues) {
  const blocks = [...ir.matchAll(/<d>([\s\S]*?)<\/d>/g)].map((m) => m[1]);
  blocks.forEach((b) => {
    if (!/^\[[A-Za-z][A-Za-z -]*\]\s/.test(b)) issues.push({level: 'error', code: 'd_lang', msg: `<d> block must start with a [Language] tag: <d>${b.slice(0, 40)}`});
    // One sentence per <d> (learned from the official Context-IR API, r2): split multi-sentence lines into
    // consecutive <d> blocks with a small action beat between them; H3-Base drops/garbles tails of long blocks.
    const body = b.replace(/^\[[^\]]*\]\s*/, '').trim();
    const n = body.replace(/\b(Mr|Mrs|Ms|Dr|St|Jr|Sr|vs|etc|e\.g|i\.e)\./g, '$1').split(/(?<=[.!?])\s+(?=["“'A-Z0-9])|(?<=[。！？])/).filter((s) => s.replace(/[\s.!?。！？"”]/g, '')).length;
    if (n > 1 && process.env.H3_DIALOGUE_SPLIT === '1') issues.push({level: 'error', code: 'd_multi_sentence', msg: `<d> holds ${n} sentences; give each sentence its own <d> block with a short action beat between them: <d>${b.slice(0, 60)}`});
  });
  const joined = norm(blocks.map((b) => b.replace(/^\[[^\]]*\]/, '')).join(''));
  for (const line of expect || []) {
    if (!joined.includes(norm(line))) issues.push({level: 'error', code: 'dialogue_lost', msg: `requested line not found verbatim inside <d>: ${line}`});
  }
  if ((ir.match(/<d>/g) || []).length !== (ir.match(/<\/d>/g) || []).length) issues.push({level: 'error', code: 'd_unbalanced', msg: '<d> and </d> counts differ'});
}

function checkEnglish(name, body, issues) {
  const rest = stripAllowed(body);
  if (CJK.test(rest)) issues.push({level: 'error', code: 'not_english', msg: `${name} contains non-English text outside <d> or quoted on-screen text: "${rest.match(/.{0,20}[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af]+.{0,20}/)?.[0]}"`});
}

/**
 * @param {string} ir
 * @param {{mode:'ref2va'|'t2va'|'i2va', duration:number, labels:{label:string,source:string}[], expectDialogue?:string[]}} ctx
 * @returns {{level:'error'|'warn', code:string, msg:string}[]}
 */
export function validateIR(ir, ctx) {
  const issues = [];
  if (typeof ir !== 'string' || ir.trim().length < 50) return [{level: 'error', code: 'empty', msg: 'IR is empty'}];
  if (/```/.test(ir)) issues.push({level: 'error', code: 'fence', msg: 'remove markdown code fences'});
  const assetLabels = new Set(ctx.labels.map((l) => l.label));
  const used = new Set(ir.match(/<(Picture|Video|Audio) \d+>/g) || []);
  for (const l of used) if (!assetLabels.has(l)) issues.push({level: 'error', code: 'label_undefined', msg: `${l} does not exist in the request (available: ${[...assetLabels].join(', ') || 'none'})`});

  if (ctx.mode === 'ref2va') {
    const {order, map, preamble} = splitSections(ir, REF_SECTIONS);
    REF_SECTIONS.forEach((s) => !map[s] && issues.push({level: 'error', code: 'section_missing', msg: `missing section ${s}:`}));
    if (order.join() !== REF_SECTIONS.filter((s) => order.includes(s)).join()) issues.push({level: 'error', code: 'section_order', msg: `sections out of order: ${order.join(', ')}`});
    if (preamble) issues.push({level: 'warn', code: 'preamble', msg: 'text before subject_definitions'});
    for (const l of ctx.labels) {
      if (l.source === 'soundtrack of the video file') continue; // a video's soundtrack may legitimately go unused
      if (!used.has(l.label)) issues.push({level: 'error', code: 'label_unused', msg: `${l.label} (${l.source}) is never referenced`});
    }
    const defs = map.subject_definitions || '';
    const subj = new Set(ir.match(/<Subject \d+>/g) || []);
    for (const s of subj) if (!defs.includes(`${s} is`) && !defs.includes(`${s} (`)) issues.push({level: 'error', code: 'subject_undefined', msg: `${s} is used but not defined in subject_definitions`});
    const sum = map.summary || '';
    const pre = /^\[([^\]]+)\]/.exec(sum);
    if (!pre) issues.push({level: 'error', code: 'summary_prefix', msg: 'summary must start with a [task type] prefix'});
    else pre[1].split(' + ').forEach((t) => !TASK_TYPES.has(t.trim()) && issues.push({level: 'error', code: 'summary_type', msg: `unknown task type "${t.trim()}"`}));
    const newInSummary = (sum.match(/<(Subject|Picture|Video|Audio) \d+>/g) || []).filter((l) => !defs.includes(l));
    if (newInSummary.length) issues.push({level: 'warn', code: 'summary_new_label', msg: `summary introduces labels not in subject_definitions: ${[...new Set(newInSummary)].join(', ')}`});
    const ret = map.retention_analysis || '';
    if (/\(S\d+(,S\d+)*\)/.test(ret)) issues.push({level: 'error', code: 'speaker_in_retention', msg: 'do not write (Sx) in retention_analysis'});
    ret.split('\n').filter((l) => /^<(Subject|Picture|Video|Audio) \d+>/.test(l.trim())).forEach((line) => {
      const isAudio = /^<Audio/.test(line.trim());
      const m = /:\s*([a-z_]+)\s*-/.exec(line);
      const ok = m && (isAudio ? AUD_MARKERS : VIS_MARKERS).has(m[1]);
      if (!ok) issues.push({level: 'error', code: 'retention_marker', msg: `bad relationship marker in: ${line.trim().slice(0, 80)}`});
    });
    const det = map.detailed_description || '';
    if (det && !/^[^[]{10,}\[Shot 1\]/.test(det)) issues.push({level: 'warn', code: 'style_opening', msg: 'detailed_description should open with one or two style sentences before [Shot 1]'});
    checkShots(det, ctx.duration, issues);
    const w = words(det);
    if (w < 200) issues.push({level: 'error', code: 'too_short', msg: `detailed_description has ${w} English words; target 350-500`});
    else if (w < 300 || w > 650) issues.push({level: 'warn', code: 'length', msg: `detailed_description has ${w} English words; target 350-500`});
    REF_SECTIONS.forEach((s) => map[s] && checkEnglish(s, map[s], issues));
  } else {
    const {order, map, preamble} = splitSections(ir, BASE_SECTIONS);
    BASE_SECTIONS.forEach((s) => !map[s] && issues.push({level: 'error', code: 'section_missing', msg: `missing section ${s}:`}));
    if (order.join() !== BASE_SECTIONS.filter((s) => order.includes(s)).join()) issues.push({level: 'error', code: 'section_order', msg: `sections out of order: ${order.join(', ')}`});
    if (ctx.mode === 'i2va' && preamble !== I2VA_LINE) issues.push({level: 'error', code: 'i2va_instruction', msg: `first line must be exactly: ${I2VA_LINE}`});
    if (ctx.mode === 't2va' && preamble) issues.push({level: 'error', code: 't2va_preamble', msg: 'T2VA starts directly with integrated_multimodal_description:'});
    const body = map.integrated_multimodal_description || '';
    if (!/^\[Shot 1\]/.test(body)) issues.push({level: 'error', code: 'shot1_start', msg: 'integrated_multimodal_description must start with [Shot 1]'});
    checkShots(body, ctx.duration, issues);
    const w = words(body);
    if (w < 120) issues.push({level: 'error', code: 'too_short', msg: `integrated_multimodal_description has only ${w} English words`});
    BASE_SECTIONS.forEach((s) => map[s] && checkEnglish(s, map[s], issues));
  }
  checkDialogue(ir, ctx.expectDialogue, issues);
  return issues;
}

export const errorsOf = (issues) => issues.filter((i) => i.level === 'error');
