// report.json + report.md from runs/<run>/{ir,scores,clips,ledger.json}
import fs from 'node:fs';
import path from 'node:path';
import {errorsOf} from './validate.mjs';

const mean = (xs) => (xs.length ? +(xs.reduce((a, b) => a + b, 0) / xs.length).toFixed(2) : null);
const rj = (f) => JSON.parse(fs.readFileSync(f, 'utf8'));

export const USABLE = 7;

export function summarize(run, {cases, arms, seeds, ledger}) {
  const rep = {run: path.basename(run), at: new Date().toISOString(), cases: {}, overall: {}, irText: {}, cost: {}};
  const pooled = {}; // arm -> {overall:[], wins:0, n:0, dlg:[]}
  for (const c of cases) {
    const irF = path.join(run, 'ir', `${c.id}.json`);
    const sF = path.join(run, 'scores', `${c.id}.json`);
    const ir = fs.existsSync(irF) ? rj(irF) : null;
    const sc = fs.existsSync(sF) ? rj(sF) : null;
    const row = {mode: c.mode, videoAB: c.video_ab, agentIR: ir && {rounds: ir.rounds.length, finalErrors: errorsOf(ir.rounds.at(-1).issues).length, firstErrors: errorsOf(ir.rounds[0].issues).length, usd: ir.usd, labels: ir.labels.map((l) => l.label)}, officialValidatorErrors: ir?.officialValidator ? errorsOf(ir.officialValidator).map((i) => i.code) : null, arms: {}};
    if (sc) {
      for (const arm of arms) {
        const clips = Object.entries(sc.clips).filter(([f]) => f.split('__')[1] === arm);
        if (!clips.length) continue;
        const judged = seeds.map((s) => sc.judge[s]).filter(Boolean);
        const scores = judged.map((j) => j.byKey[arm]).filter(Boolean);
        const wins = judged.filter((j) => j.ranking[0] === arm).length;
        const a = {
          clips: clips.length,
          qcPass: clips.filter(([, v]) => v.qc.pass).length,
          frameStd: mean(clips.map(([, v]) => v.qc.frameStd.mean)),
          dialogueAcc: c.expect_dialogue?.length ? mean(clips.map(([, v]) => v.dialogue.mean)) : null,
          forbiddenSpeech: clips.filter(([, v]) => v.forbidden?.length).length,
          judge: Object.fromEntries(['instruction_following', 'reference_fidelity', 'visual_quality', 'motion_camera', 'audio_dialogue', 'overall'].map((k) => [k, mean(scores.map((s) => s[k]))])),
          wins,
          judged: judged.length,
          asr: clips.map(([f, v]) => `${f.split('__')[2].replace('.mp4', '')}: ${v.asr.text || '(none)'}`),
        };
        row.arms[arm] = a;
        const p = (pooled[arm] ??= {overall: [], instr: [], ref: [], wins: 0, n: 0, dlg: []});
        p.overall.push(...scores.map((s) => s.overall));
        p.instr.push(...scores.map((s) => s.instruction_following));
        p.ref.push(...scores.map((s) => s.reference_fidelity));
        p.wins += wins;
        p.n += judged.length;
        if (a.dialogueAcc != null) p.dlg.push(a.dialogueAcc);
      }
      if (sc.irJudge) {
        rep.irText[c.id] = Object.fromEntries(Object.keys(sc.irJudge[0].byKey).map((k) => [k, {overall: mean(sc.irJudge.map((j) => j.byKey[k].overall)), intent: mean(sc.irJudge.map((j) => j.byKey[k].intent_fidelity)), guide: mean(sc.irJudge.map((j) => j.byKey[k].guide_compliance)), firstPlace: sc.irJudge.filter((j) => j.ranking[0] === k).length, passes: sc.irJudge.length, notes: sc.irJudge.map((j) => j.byKey[k].notes)}]));
      }
    }
    rep.cases[c.id] = row;
  }
  rep.overall = Object.fromEntries(Object.entries(pooled).map(([k, p]) => [k, {meanOverall: mean(p.overall), firstPlace: `${p.wins}/${p.n}`, meanDialogueAcc: mean(p.dlg), instr: mean(p.instr), ref: mean(p.ref),
    // stability ("draw rate"): share of judged clips scoring >= USABLE overall, and the worst clip
    usableRate: p.overall.length ? +(p.overall.filter((x) => x >= USABLE).length / p.overall.length).toFixed(2) : null, worst: p.overall.length ? Math.min(...p.overall) : null,
    // spread across seeds/cases: lower = fewer re-rolls needed
    sd: p.overall.length > 1 ? +Math.sqrt(p.overall.reduce((a, x) => a + (x - mean(p.overall)) ** 2, 0) / (p.overall.length - 1)).toFixed(2) : null, n: p.overall.length}]));
  const by = {};
  for (const x of ledger) by[x.purpose.replace(/\d+$/, '')] = +((by[x.purpose.replace(/\d+$/, '')] || 0) + x.usd).toFixed(3);
  rep.cost = {bedrockUsdEstimate: +ledger.reduce((s, x) => s + x.usd, 0).toFixed(2), byPurpose: by, calls: ledger.length};
  const gens = fs.existsSync(path.join(run, 'clips')) ? fs.readdirSync(path.join(run, 'clips')).filter((f) => f.endsWith('.gen.json')).map((f) => rj(path.join(run, 'clips', f))) : [];
  rep.cost.gpu = {clips: gens.length, meanInferenceSec: mean(gens.map((g) => g.inferenceSec).filter(Number.isFinite)), totalWallSec: +gens.reduce((s, g) => s + g.sec, 0).toFixed(0)};
  return rep;
}

export function toMarkdown(rep) {
  const L = [`# H3 Context-IR agent A/B - run ${rep.run}`, '', `Generated ${rep.at}. Judge: blind GPT-6 on 3x3 contact sheets + ASR (arms shuffled per seed). Scores 1-10.`, '', '## Pooled (video cases)', '', `| arm | mean overall | judge 1st place | mean dialogue accuracy | instr | ref | usable (>=${USABLE}) | worst | sd | n |`, '|---|---|---|---|---|---|---|---|---|---|'];
  for (const [k, v] of Object.entries(rep.overall)) L.push(`| ${k} | ${v.meanOverall} | ${v.firstPlace} | ${v.meanDialogueAcc ?? '-'} | ${v.instr} | ${v.ref} | ${v.usableRate} | ${v.worst} | ${v.sd} | ${v.n} |`);
  L.push('', '## Per case', '');
  for (const [id, r] of Object.entries(rep.cases)) {
    L.push(`### ${id} (${r.mode}${r.videoAB ? '' : ', IR text only'})`, '');
    if (r.agentIR) L.push(`Agent IR: labels ${r.agentIR.labels.join(' ') || '-'}; validator errors first draft ${r.agentIR.firstErrors} -> final ${r.agentIR.finalErrors} after ${r.agentIR.rounds} round(s); $${r.agentIR.usd}.${r.officialValidatorErrors ? ` Official IR validator errors: ${r.officialValidatorErrors.join(', ') || 'none'}.` : ''}`, '');
    if (Object.keys(r.arms).length) {
      L.push('| arm | QC | frame std | dialogue acc | instr | ref | visual | motion | audio | overall | 1st |', '|---|---|---|---|---|---|---|---|---|---|---|');
      for (const [a, v] of Object.entries(r.arms)) L.push(`| ${a} | ${v.qcPass}/${v.clips} | ${v.frameStd} | ${v.dialogueAcc ?? '-'}${v.forbiddenSpeech ? ` (${v.forbiddenSpeech} leaked)` : ''} | ${v.judge.instruction_following} | ${v.judge.reference_fidelity} | ${v.judge.visual_quality} | ${v.judge.motion_camera} | ${v.judge.audio_dialogue} | ${v.judge.overall} | ${v.wins}/${v.judged} |`);
      L.push('');
      for (const [a, v] of Object.entries(r.arms)) L.push(`- ${a} ASR: ${v.asr.join(' / ')}`);
      L.push('');
    }
    if (rep.irText[id]) {
      const t = rep.irText[id];
      const passes = Object.values(t)[0].passes || 2;
      L.push(`IR text judge (${passes} passes, order shuffled): ${Object.entries(t).map(([k, v]) => `${k} overall ${v.overall} (intent ${v.intent}, guide ${v.guide}, 1st ${v.firstPlace}/${passes})`).join(' vs ')}.`, '');
    }
  }
  L.push('## Cost', '', `Model spend (Bedrock list-price estimate + MiniMax Context-IR): $${rep.cost.bedrockUsdEstimate} over ${rep.cost.calls} calls: ${JSON.stringify(rep.cost.byPurpose)}. GPU: ${rep.cost.gpu.clips} clips, mean inference ${rep.cost.gpu.meanInferenceSec}s.`);
  return L.join('\n');
}

export function writeReport(run, opts) {
  const rep = summarize(run, opts);
  fs.writeFileSync(path.join(run, 'report.json'), JSON.stringify(rep, null, 1));
  fs.writeFileSync(path.join(run, 'report.md'), toMarkdown(rep));
  return rep;
}
