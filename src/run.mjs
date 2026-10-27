// A/B harness: cases x arms (raw | agent | official) x seeds on the G7e Ref2VA servers.
//   arms: raw (user text) | agent (ours) | official (doc gold IR) | api (official Context-IR API)
//   node src/run.mjs --run r1 [--cases a,b] [--seeds 2] [--arms raw,agent,official,api] [--phase ir|gen|score|report|all] [--dry]
// Every phase is resumable: finished artifacts under runs/<run>/ are reused.
import fs from 'node:fs';
import path from 'node:path';
import {buildIR} from './agent.mjs';
import {computeLabels} from './labels.mjs';
import {validateIR, errorsOf} from './validate.mjs';
import {stage, buildRequest, generate, health, h3Urls} from './h3.mjs';
import {probeCondition, qcClip, contactSheet, asr, loudness} from './probe.mjs';
import {judgeVideos, judgeIRs, dialogueScore} from './judge.mjs';
import {img, txt} from './bedrock.mjs';
import {writeReport} from './report.mjs';
import {officialIR, dataUri} from './minimax.mjs';

const ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
const args = Object.fromEntries(process.argv.slice(2).reduce((a, x, i, all) => (x.startsWith('--') ? [...a, [x.slice(2), all[i + 1] && !all[i + 1].startsWith('--') ? all[i + 1] : true]] : a), []));
const RUN = path.join(ROOT, 'runs', args.run || 'dev');
const PHASE = args.phase || 'all';
const SEEDS = [1001, 2002, 3003, 4004, 5005, 6006].slice(0, +(args.seeds || 2));
const ARMS = String(args.arms || 'raw,agent,official,api').split(',');
const ALL = JSON.parse(fs.readFileSync(path.join(ROOT, 'cases/cases.json'), 'utf8'));
const CASES = args.cases ? ALL.filter((c) => String(args.cases).split(',').includes(c.id)) : ALL;
const log = (...m) => console.log(new Date().toISOString().slice(11, 19), ...m);
const rj = (f) => JSON.parse(fs.readFileSync(f, 'utf8'));
const wj = (f, o) => (fs.mkdirSync(path.dirname(f), {recursive: true}), fs.writeFileSync(f, JSON.stringify(o, null, 1)));
const ledgerFile = path.join(RUN, 'ledger.json');
const ledger = fs.existsSync(ledgerFile) ? rj(ledgerFile) : [];
const saveLedger = () => wj(ledgerFile, ledger);

/** Prompt text per arm for a case (null = arm not available for this case). */
function armPrompt(c, arm) {
  if (arm === 'raw') return c.text;
  if (arm === 'official') return c.official_ir || null;
  if (arm === 'agent') return rj(path.join(RUN, 'ir', `${c.id}.json`)).ir;
  // agent_v1 = the agent IR from an earlier run (--prev r2), for before/after comparisons of agent changes
  if (arm === 'agent_v1') {
    const f = path.join(ROOT, 'runs', String(args.prev || 'r2'), 'ir', `${c.id}.json`);
    return fs.existsSync(f) ? rj(f).ir : null;
  }
  // ir_<run> = the agent IR written in another run (e.g. one run per writer model), so many writers share one A/B
  if (arm.startsWith('ir_')) {
    const f = path.join(ROOT, 'runs', arm.slice(3), 'ir', `${c.id}.json`);
    return fs.existsSync(f) ? rj(f).ir : null;
  }
  if (arm === 'api') {
    const f = path.join(RUN, 'ir_api', `${c.id}.json`);
    return fs.existsSync(f) ? rj(f).ir : null;
  }
  throw new Error(`unknown arm ${arm}`);
}

async function phaseIR() {
  for (const c of CASES) {
    const f = path.join(RUN, 'ir', `${c.id}.json`);
    if (fs.existsSync(f)) {
      log(`[ir] ${c.id} cached`);
      continue;
    }
    log(`[ir] ${c.id} (${c.mode})`);
    const before = ledger.length;
    const out = await buildIR(c, {workDir: path.join(RUN, 'work', c.id), log: (m) => log(`  ${c.id}: ${m}`)});
    ledger.push(...out.ledger.map((x) => ({...x, case: c.id})));
    saveLedger();
    const official = c.official_ir ? validateIR(c.official_ir, {mode: c.mode, duration: c.duration, labels: out.labels, expectDialogue: c.expect_dialogue}) : null;
    wj(f, {case: c.id, ir: out.ir, labels: out.labels, plan: out.plan, rounds: out.rounds, probes: out.probes, officialValidator: official, usd: +ledger.slice(before).reduce((s, x) => s + x.usd, 0).toFixed(3)});
    log(`  ${c.id}: final IR ${out.ir.length} chars, ${errorsOf(out.rounds.at(-1).issues).length} errors in last round`);
  }
}

async function phaseIRApi({dry}) {
  if (!ARMS.includes('api')) return;
  for (const c of CASES) {
    const f = path.join(RUN, 'ir_api', `${c.id}.json`);
    if (fs.existsSync(f)) {
      log(`[ir-api] ${c.id} cached`);
      continue;
    }
    if (dry) {
      log(`[ir-api] ${c.id} (dry: skipped)`);
      continue;
    }
    log(`[ir-api] ${c.id} (${c.mode})`);
    try {
      const urls = (c.conditions || []).map((x) => dataUri(path.join(ROOT, 'cases', x.file)));
      const t = await officialIR(c, urls, {log: (m) => log(`  ${c.id}: ${m}`)});
      ledger.push({purpose: 'minimax-context-ir', case: c.id, usd: t.usd, usage: t.usage});
      saveLedger();
      // labels need the media probe (soundtrack detection); reuse the agent IR's labels when present
      const agentF = path.join(RUN, 'ir', `${c.id}.json`);
      const labels = fs.existsSync(agentF) ? rj(agentF).labels : null;
      wj(f, {case: c.id, ir: t.prompt, task: t.id, usage: t.usage, usd: t.usd, sec: t.sec, validator: labels ? validateIR(t.prompt, {mode: c.mode, duration: c.duration, labels, expectDialogue: c.expect_dialogue}) : null});
      log(`  ${c.id}: api IR ${t.prompt.length} chars, $${t.usd}, ${t.sec}s`);
    } catch (e) {
      log(`[ir-api] ${c.id} FAIL: ${e.message}`);
    }
  }
}

async function phaseGen({dry}) {
  const jobs = [];
  for (const c of CASES.filter((x) => x.video_ab)) {
    const uris = dry ? c.conditions.map((x) => `s3://dry/${x.file}`) : await Promise.all(c.conditions.map((x) => stage(path.join(ROOT, 'cases', x.file))));
    for (const arm of ARMS) {
      const prompt = armPrompt(c, arm);
      if (!prompt) continue;
      for (const seed of SEEDS) {
        const out = path.join(RUN, 'clips', `${c.id}__${arm}__s${seed}.mp4`);
        const body = buildRequest({prompt, uris, conditions: c.conditions, duration: c.duration, aspect: c.aspect, seed});
        wj(out.replace(/\.mp4$/, '.request.json'), {...body, conditions: body.conditions.map((x, i) => ({...x, uri: c.conditions[i].file}))});
        if (!fs.existsSync(out)) jobs.push({c, arm, seed, out, body});
      }
    }
  }
  log(`[gen] ${jobs.length} clips to make${dry ? ' (dry: requests written, nothing submitted)' : ''}`);
  if (dry || !jobs.length) return;
  const servers = [];
  for (const u of h3Urls()) if (await health(u)) servers.push(u);
  if (!servers.length) throw new Error(`no healthy H3 server in ${h3Urls().join(', ')}`);
  log(`[gen] servers: ${servers.join(', ')}`);
  // Interleave arms so a spot interruption or a busy GPU does not bias one arm.
  jobs.sort((a, b) => a.seed - b.seed || a.c.id.localeCompare(b.c.id) || a.arm.localeCompare(b.arm));
  const failures = [];
  await Promise.all(servers.map(async (u) => {
    while (jobs.length) {
      const j = jobs.shift();
      try {
        const g = await generate(u, j.body, j.out, {log});
        wj(j.out.replace(/\.mp4$/, '.gen.json'), g);
      } catch (e) {
        failures.push({clip: path.basename(j.out), error: String(e.message || e)});
        log(`[gen] FAIL ${path.basename(j.out)}: ${e.message}`);
      }
    }
  }));
  wj(path.join(RUN, 'gen_failures.json'), failures);
}

async function refBlocks(c) {
  const blocks = [];
  for (const [i, cond] of c.conditions.entries()) {
    const p = await probeCondition(cond, path.join(ROOT, 'cases', cond.file), path.join(RUN, 'judge_refs', c.id));
    if (p.type === 'image') blocks.push(txt(`Reference ${i + 1}: image`), img(p.image));
    else if (p.type === 'video') blocks.push(txt(`Reference ${i + 1}: video ${p.duration}s, soundtrack ASR: "${p.asr?.segments?.map((s) => s.text).join(' ') || 'none'}". Four frames:`), ...p.frames.filter((_, k) => k % Math.ceil(p.frames.length / 4) === 0).map((f) => img(f.file)));
    else blocks.push(txt(`Reference ${i + 1}: audio ${p.duration}s, ASR: "${p.asr?.segments?.map((s) => s.text).join(' ') || 'no speech'}"`));
  }
  return blocks;
}

async function phaseScore() {
  const clipDir = path.join(RUN, 'clips');
  for (const c of CASES) {
    const sf = path.join(RUN, 'scores', `${c.id}.json`);
    const prev = fs.existsSync(sf) ? rj(sf) : {};
    const out = {case: c.id, clips: prev.clips || {}, judge: prev.judge || {}, irJudge: prev.irJudge || null};
    const refs = await refBlocks(c);
    if (c.video_ab && fs.existsSync(clipDir)) {
      for (const f of fs.readdirSync(clipDir).filter((x) => x.startsWith(`${c.id}__`) && x.endsWith('.mp4'))) {
        if (out.clips[f]) continue;
        const file = path.join(clipDir, f);
        const qc = await qcClip(file);
        const a = await asr(file, {model: process.env.OUT_ASR_MODEL || 'medium'});
        const asrText = a.segments.filter((s) => s.no_speech < 0.6).map((s) => s.text).join(' ');
        const sheet = await contactSheet(file, file.replace(/\.mp4$/, '.sheet.jpg'));
        out.clips[f] = {qc, asr: {language: a.language, text: asrText}, loud: await loudness(file), dialogue: dialogueScore(c.expect_dialogue || [], asrText), forbidden: (c.forbid_speech || []).filter((w) => asrText.includes(w)), sheet};
        log(`[score] ${f}: qc ${qc.pass ? 'ok' : 'FAIL'} std ${qc.frameStd.mean} dlg ${out.clips[f].dialogue.mean} asr "${asrText.slice(0, 60)}"`);
        wj(sf, out);
      }
      for (const seed of SEEDS) {
        if (out.judge[seed]) continue;
        const clips = Object.entries(out.clips).filter(([f, v]) => f.endsWith(`__s${seed}.mp4`) && v.qc.pass).map(([f, v]) => ({key: f.split('__')[1], sheet: v.sheet, asrText: v.asr.text, meanDb: v.loud.meanDb}));
        if (clips.length < 2) continue;
        out.judge[seed] = await judgeVideos({text: c.text, refBlocks: refs, clips, seed: `${c.id}:${seed}`, ledger});
        saveLedger();
        log(`[judge] ${c.id} s${seed}: ranking ${out.judge[seed].ranking.join(' > ')}`);
        wj(sf, out);
      }
    }
    const hasAgent = fs.existsSync(path.join(RUN, 'ir', `${c.id}.json`));
    const irKeys = [...(hasAgent ? ['agent'] : []), ...ARMS.filter((k) => ['agent_v1', 'official', 'api'].includes(k) || k.startsWith('ir_'))];
    const irs = irKeys.map((key) => ({key, ir: armPrompt(c, key)})).filter((x) => x.ir);
    if (irs.length >= 2 && !out.irJudge) {
      const g = fs.readFileSync(path.join(ROOT, 'guides', c.mode === 'ref2va' ? 'VIDEO_PROMPT_WRITING_GUIDE_ref_en.md' : 'VIDEO_PROMPT_WRITING_GUIDE_base_en.md'), 'utf8');
      // two passes with swapped order to cancel position bias
      out.irJudge = [await judgeIRs({text: c.text, guideText: g, refBlocks: refs, irs, seed: `${c.id}:ir1`, ledger}), await judgeIRs({text: c.text, guideText: g, refBlocks: refs, irs: [...irs].reverse(), seed: `${c.id}:ir2`, ledger})];
      saveLedger();
      log(`[judge-ir] ${c.id}: ${out.irJudge.map((j) => j.ranking.join('>')).join(' | ')}`);
    }
    wj(sf, out);
  }
}

const t0 = Date.now();
if (['ir', 'all'].includes(PHASE)) await phaseIR();
if (['ir', 'irapi', 'all'].includes(PHASE)) await phaseIRApi({dry: !!args.dry});
if (['gen', 'all'].includes(PHASE)) await phaseGen({dry: !!args.dry});
if (['score', 'all'].includes(PHASE) && !args.dry) await phaseScore();
if (['report', 'all', 'score'].includes(PHASE) && !args.dry) writeReport(RUN, {cases: CASES, arms: ARMS, seeds: SEEDS, ledger});
log(`done in ${((Date.now() - t0) / 1000).toFixed(0)}s; Bedrock spend so far $${ledger.reduce((s, x) => s + x.usd, 0).toFixed(2)} (list-price estimate)`);
