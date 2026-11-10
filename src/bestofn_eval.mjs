// Offline best-of-N estimate on an existing run: how would "re-roll the seed until the cheap verifier passes"
// have changed the clip that ships? Uses the run's judge scores (and a second judge's, with --tag).
//   node src/bestofn_eval.mjs --run rgpt [--tag anthropic] [--cases-file f]
// Writes runs/<run>/bestofn.{json,md}. No GPU and no LLM calls (only local ASR of the reference audio).
import fs from 'node:fs';
import path from 'node:path';
import {simulate} from './bestofn.mjs';
import {computeLabels} from './labels.mjs';
import {probeCondition} from './probe.mjs';
import {refSpeechOf} from './agent.mjs';

const ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
const args = Object.fromEntries(process.argv.slice(2).reduce((a, x, i, all) => (x.startsWith('--') ? [...a, [x.slice(2), all[i + 1] && !all[i + 1].startsWith('--') ? all[i + 1] : true]] : a), []));
if (!args.run) throw new Error('--run is required');
const RUN = path.join(ROOT, 'runs', args.run);
const casesFile = args['cases-file'] ? path.resolve(args['cases-file']) : fs.existsSync(path.join(RUN, 'cases.json')) ? path.join(RUN, 'cases.json') : path.join(ROOT, 'cases/cases.json');
const CASES = JSON.parse(fs.readFileSync(casesFile, 'utf8'));
const rj = (f) => JSON.parse(fs.readFileSync(f, 'utf8'));

const scores = {};
const scoresAlt = {};
const refSpeech = {};
let seeds = new Set();
for (const c of CASES) {
  const f = path.join(RUN, 'scores', `${c.id}.json`);
  if (!fs.existsSync(f)) continue;
  scores[c.id] = rj(f);
  Object.keys(scores[c.id].judge || {}).forEach((s) => seeds.add(+s));
  if (args.tag) {
    const g = path.join(RUN, `scores_${args.tag}`, `${c.id}.json`);
    if (fs.existsSync(g)) scoresAlt[c.id] = {...scores[c.id], judge: rj(g).judge};
  }
  const probes = [];
  for (const cond of c.conditions) probes.push(cond.type === 'image' ? {type: 'image'} : await probeCondition(cond, path.resolve(ROOT, 'cases', cond.file), path.join(RUN, 'judge_refs', c.id)));
  refSpeech[c.id] = refSpeechOf(computeLabels(c.conditions, probes.map((p) => ({hasAudio: p.hasAudio}))), probes);
}
seeds = [...seeds].sort((a, b) => a - b);
const res = {run: args.run, seeds, refSpeech, judgeA: simulate(CASES, scores, seeds, refSpeech)};
if (args.tag) res.judgeB = simulate(CASES, scoresAlt, seeds, refSpeech);
fs.writeFileSync(path.join(RUN, 'bestofn.json'), JSON.stringify(res, null, 1));

const L = [`# Best-of-N offline estimate - run ${args.run}`, '', `Policy: generate seeds ${seeds.join(', ')} in order, stop at the first clip the verifier passes (QC, dialogue accuracy, no voice-reference leak), else ship the best-scoring one. N=1 is today's single draw.`, ''];
for (const [k, sim] of Object.entries({A: res.judgeA, B: res.judgeB}).filter(([, v]) => v)) {
  L.push(`## Judge ${k}${k === 'B' ? ` (${args.tag})` : ''}`, '', '| arm | N | cases | mean overall | usable (>=7) | worst | mean attempts | verifier TP/FP/FN/TN |', '|---|---|---|---|---|---|---|---|');
  for (const [arm, a] of Object.entries(sim)) {
    for (const [n, b] of Object.entries(a.byN)) L.push(`| ${arm} | ${n} | ${b.cases} | ${b.meanOverall} | ${b.usableRate} | ${b.worst} | ${b.meanAttempts} | ${n === '1' ? `${a.verifier.tp}/${a.verifier.fp}/${a.verifier.fn}/${a.verifier.tn}` : ''} |`);
  }
  L.push('');
}
L.push('Verifier counts compare "verifier pass" with "judge overall >= 7" over every judged clip: FP = passed but judged unusable, FN = rejected but judged usable.');
fs.writeFileSync(path.join(RUN, 'bestofn.md'), L.join('\n'));
console.log(L.join('\n'));
