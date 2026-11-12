// Re-score an existing run's clips with a second video judge (another vendor) and compare the two judges.
//   node src/rejudge.mjs --run rgpt --model us.anthropic.claude-opus-5-5 \
//     --writers agent_v1=anthropic,ir_g61sol=openai,ir_g61terra=openai,api=minimax [--cases-file f] [--tag claude]
// Reads runs/<run>/scores/*.json (judge A = the run's own judge), writes runs/<run>/scores_<tag>/*.json,
// runs/<run>/judges_<tag>.{json,md}. Resumable: judged seeds are skipped.
import fs from 'node:fs';
import path from 'node:path';
import {judgeVideos, refBlocksFor} from './judge.mjs';
import {MODELS} from './bedrock.mjs';
import {compareJudges, compareToMarkdown, vendorOf} from './judges.mjs';

const ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
const args = Object.fromEntries(process.argv.slice(2).reduce((a, x, i, all) => (x.startsWith('--') ? [...a, [x.slice(2), all[i + 1] && !all[i + 1].startsWith('--') ? all[i + 1] : true]] : a), []));
if (!args.run) throw new Error('--run is required');
const RUN = path.join(ROOT, 'runs', args.run);
const MODEL = args.model || 'us.anthropic.claude-opus-5-5';
const MODEL_A = args['model-a'] || MODELS.judge;
const TAG = args.tag || vendorOf(MODEL);
const casesFile = args['cases-file'] ? path.resolve(args['cases-file']) : fs.existsSync(path.join(RUN, 'cases.json')) ? path.join(RUN, 'cases.json') : path.join(ROOT, 'cases/cases.json');
const CASES = JSON.parse(fs.readFileSync(casesFile, 'utf8'));
const writerVendor = {raw: 'none', official: 'none', api: 'minimax', ...Object.fromEntries(String(args.writers || '').split(',').filter(Boolean).map((kv) => kv.split('=')))};
const rj = (f) => JSON.parse(fs.readFileSync(f, 'utf8'));
const wj = (f, o) => (fs.mkdirSync(path.dirname(f), {recursive: true}), fs.writeFileSync(f, JSON.stringify(o, null, 1)));
const log = (...m) => console.log(new Date().toISOString().slice(11, 19), ...m);
const ledgerFile = path.join(RUN, `ledger_${TAG}.json`);
const ledger = fs.existsSync(ledgerFile) ? rj(ledgerFile) : [];

const A = {};
const B = {};
for (const c of CASES) {
  const sf = path.join(RUN, 'scores', `${c.id}.json`);
  if (!fs.existsSync(sf)) continue;
  const sc = rj(sf);
  if (!sc.judge || !Object.keys(sc.judge).length) continue;
  const outF = path.join(RUN, `scores_${TAG}`, `${c.id}.json`);
  const out = fs.existsSync(outF) ? rj(outF) : {case: c.id, model: MODEL, judge: {}};
  let refs = null;
  for (const seed of Object.keys(sc.judge)) {
    if (out.judge[seed]) continue;
    const clips = Object.entries(sc.clips)
      .filter(([f, v]) => f.endsWith(`__s${seed}.mp4`) && v.qc.pass)
      .map(([f, v]) => ({key: f.split('__')[1], sheet: v.sheet, asrText: v.asr.text, meanDb: v.loud.meanDb}));
    if (clips.length < 2) continue;
    // condition files resolve against cases/ (absolute paths in external case lists are kept as is)
    refs ??= await refBlocksFor(c, path.join(ROOT, 'cases'), path.join(RUN, 'judge_refs', c.id));
    // a different shuffle seed than judge A, so position bias does not line up between the two judges
    out.judge[seed] = await judgeVideos({text: c.text, refBlocks: refs, clips, seed: `${c.id}:${seed}:${TAG}`, ledger, model: MODEL});
    wj(outF, out);
    wj(ledgerFile, ledger);
    log(`[rejudge ${TAG}] ${c.id} s${seed}: ${out.judge[seed].ranking.join(' > ')}  (A: ${sc.judge[seed].ranking.join(' > ')})`);
  }
  A[c.id] = sc.judge;
  B[c.id] = out.judge;
}
const cmp = compareJudges(A, B, {modelA: MODEL_A, modelB: MODEL, writerVendor});
cmp.usd = +ledger.reduce((s, x) => s + x.usd, 0).toFixed(2);
wj(path.join(RUN, `judges_${TAG}.json`), cmp);
fs.writeFileSync(path.join(RUN, `judges_${TAG}.md`), `${compareToMarkdown(args.run, cmp)}\n\nJudge B spend: $${cmp.usd} (list-price estimate).\n`);
log(`done: top-1 agreement ${cmp.top1Agreement}, spearman ${cmp.spearmanClips}, $${cmp.usd}`);
