// Comparing two video judges on the same clips: per-arm means, top-1 agreement, rank correlation, and a
// "cross-vendor" score that only counts the judge from a different vendor than the arm's writer.

export const vendorOf = (model) => (/anthropic|claude/.test(model) ? 'anthropic' : /openai|gpt/.test(model) ? 'openai' : /amazon|nova/.test(model) ? 'amazon' : 'other');

/** Average ranks (ties share the mean rank), 1 = smallest. */
function ranks(xs) {
  const idx = xs.map((x, i) => [x, i]).sort((a, b) => a[0] - b[0]);
  const r = new Array(xs.length);
  for (let i = 0; i < idx.length; ) {
    let j = i;
    while (j + 1 < idx.length && idx[j + 1][0] === idx[i][0]) j++;
    for (let k = i; k <= j; k++) r[idx[k][1]] = (i + j) / 2 + 1;
    i = j + 1;
  }
  return r;
}

/** Spearman rank correlation of two equal-length score lists (null when undefined). */
export function spearman(a, b) {
  if (a.length !== b.length || a.length < 3) return null;
  const ra = ranks(a);
  const rb = ranks(b);
  const m = (ra.length + 1) / 2;
  let num = 0;
  let da = 0;
  let db = 0;
  for (let i = 0; i < ra.length; i++) {
    num += (ra[i] - m) * (rb[i] - m);
    da += (ra[i] - m) ** 2;
    db += (rb[i] - m) ** 2;
  }
  return da && db ? +(num / Math.sqrt(da * db)).toFixed(2) : null;
}

/**
 * @param {Record<string, Record<string,{byKey:Record<string,{overall:number}>, ranking:string[]}>>} A  case -> seed -> judge result (judge A)
 * @param {typeof A} B  same shape for judge B
 * @param {{modelA:string, modelB:string, writerVendor:Record<string,string>, usable?:number}} o
 *   writerVendor: arm -> vendor of the model that wrote that arm's prompt ('none' for raw, 'minimax' for api)
 */
export function compareJudges(A, B, {modelA, modelB, writerVendor = {}, usable = 7}) {
  const vA = vendorOf(modelA);
  const vB = vendorOf(modelB);
  const arms = {};
  const pairsA = [];
  const pairsB = [];
  let top1Same = 0;
  let seedsBoth = 0;
  for (const [cid, seedsA] of Object.entries(A)) {
    for (const [seed, ja] of Object.entries(seedsA)) {
      const jb = B[cid]?.[seed];
      if (!jb) continue;
      seedsBoth++;
      if (ja.ranking[0] && ja.ranking[0] === jb.ranking[0]) top1Same++;
      for (const [arm, sa] of Object.entries(ja.byKey)) {
        const sb = jb.byKey[arm];
        if (!sb) continue;
        pairsA.push(sa.overall);
        pairsB.push(sb.overall);
        const x = (arms[arm] ??= {a: [], b: [], cross: [], winsA: 0, winsB: 0, n: 0});
        x.a.push(sa.overall);
        x.b.push(sb.overall);
        const wv = writerVendor[arm] || 'none';
        // cross-vendor: drop the judge that shares the writer's vendor; keep both when neither does
        const keep = [...(wv !== vA ? [sa.overall] : []), ...(wv !== vB ? [sb.overall] : [])];
        x.cross.push(keep.length ? keep.reduce((s, v) => s + v, 0) / keep.length : (sa.overall + sb.overall) / 2);
      }
      for (const [arm, x] of Object.entries(arms)) {
        if (!ja.byKey[arm] || !jb.byKey[arm]) continue;
        x.n++;
        if (ja.ranking[0] === arm) x.winsA++;
        if (jb.ranking[0] === arm) x.winsB++;
      }
    }
  }
  const mean = (xs) => (xs.length ? +(xs.reduce((s, v) => s + v, 0) / xs.length).toFixed(2) : null);
  const use = (xs) => (xs.length ? +(xs.filter((v) => v >= usable).length / xs.length).toFixed(2) : null);
  const perArm = Object.fromEntries(
    Object.entries(arms).map(([arm, x]) => [arm, {n: x.a.length, writer: writerVendor[arm] || 'none', meanA: mean(x.a), meanB: mean(x.b), meanBoth: mean(x.a.map((v, i) => (v + x.b[i]) / 2)), crossVendor: mean(x.cross), usableA: use(x.a), usableB: use(x.b), firstA: `${x.winsA}/${x.n}`, firstB: `${x.winsB}/${x.n}`}]),
  );
  const order = (k) => Object.entries(perArm).sort((p, q) => q[1][k] - p[1][k]).map(([a]) => a);
  return {modelA, modelB, seeds: seedsBoth, top1Agreement: seedsBoth ? +(top1Same / seedsBoth).toFixed(2) : null, spearmanClips: spearman(pairsA, pairsB), clips: pairsA.length, perArm, rankA: order('meanA'), rankB: order('meanB'), rankCross: order('crossVendor')};
}

export function compareToMarkdown(run, cmp) {
  const L = [`# Judge cross-check - run ${run}`, '', `Judge A: ${cmp.modelA} · Judge B: ${cmp.modelB} · ${cmp.clips} clips over ${cmp.seeds} case-seeds.`, '', `- Top-1 agreement (same winner per case-seed): ${cmp.top1Agreement}`, `- Spearman of per-clip overall scores: ${cmp.spearmanClips}`, `- Arm order by A: ${cmp.rankA.join(' > ')}`, `- Arm order by B: ${cmp.rankB.join(' > ')}`, `- Arm order, cross-vendor only: ${cmp.rankCross.join(' > ')}`, '', '| arm | writer vendor | n | overall A | overall B | mean A+B | cross-vendor | usable A | usable B | 1st A | 1st B |', '|---|---|---|---|---|---|---|---|---|---|---|'];
  for (const [arm, v] of Object.entries(cmp.perArm)) L.push(`| ${arm} | ${v.writer} | ${v.n} | ${v.meanA} | ${v.meanB} | ${v.meanBoth} | ${v.crossVendor} | ${v.usableA} | ${v.usableB} | ${v.firstA} | ${v.firstB} |`);
  L.push('', 'cross-vendor = mean of the judges whose vendor differs from the writer of that arm (both judges for raw / api).');
  return L.join('\n');
}
