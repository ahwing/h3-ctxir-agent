import test from 'node:test';
import assert from 'node:assert/strict';
import {validateIR, errorsOf, leakShare, retentionMarkers, normSay} from '../src/validate.mjs';
import {verifyClip, pickBest, generateBestOfN, simulate} from '../src/bestofn.mjs';
import {compareJudges, spearman, vendorOf} from '../src/judges.mjs';
import {refSpeechOf} from '../src/agent.mjs';

const REF_AUDIO = '今天的天气真不错 我们下午去公园走走吧 顺便买点水果回来';
const USER = '画面里穿浅灰色毛衣的女人端起茶杯，笑着用音频1的声音说：“大家快趁热吃吧。”';
const ref2va = (marker, line) => `subject_definitions:
<Subject 1> is the woman in the light grey sweater from <Picture 1>.

summary:
[reference generation + audio reference] <Subject 1> lifts a teacup and speaks with the voice of <Audio 1>.

retention_analysis:
<Picture 1>: fully_preserved - keep her face and sweater.
<Audio 1>: ${marker} - voice timbre only.

detailed_description:
Warm realistic home dining scene with soft window light. [Shot 1] ${'The woman lifts the teacup and smiles warmly at the family around the table. '.repeat(30)}She (S1) says: <d>[Chinese] ${line}</d>

overall_soundscape:
Soft clinking of dishes.

non_diegetic_music:
N/A`;
const ctx = {mode: 'ref2va', duration: 5, labels: [{label: '<Picture 1>', source: 'image file', type: 'image'}, {label: '<Audio 1>', source: 'audio file', type: 'audio'}], expectDialogue: [], refSpeech: [{label: '<Audio 1>', text: REF_AUDIO}], userText: USER};
const codesOf = (issues, level) => issues.filter((i) => i.level === level).map((i) => i.code);

test('leak check: a timbre-only reference audio must not have its words spoken in <d>', () => {
  assert.ok(codesOf(validateIR(ref2va('reference', '今天的天气真不错，我们下午去公园走走吧。'), ctx), 'error').includes('voice_ref_leak'));
  assert.ok(!codesOf(validateIR(ref2va('reference', '大家快趁热吃吧。'), ctx), 'error').includes('voice_ref_leak'), 'the user line is fine');
  const copy = validateIR(ref2va('fully_copy', '今天的天气真不错，我们下午去公园走走吧。'), ctx);
  assert.ok(codesOf(copy, 'warn').includes('voice_ref_leak') && !codesOf(copy, 'error').includes('voice_ref_leak'), 'copied audio only warns');
  assert.ok(!codesOf(validateIR(ref2va('reference', '今天的天气真不错，我们下午去公园走走吧。'), {...ctx, refSpeech: undefined}), 'error').includes('voice_ref_leak'), 'no ASR -> no check');
});

test('leak share ignores words the user typed and survives Traditional/Simplified flips', () => {
  assert.equal(normSay('大家快趁熱吃吧！'), '大家快趁热吃吧');
  assert.ok(leakShare(REF_AUDIO, '今天的天氣真不錯 我們下午去公園走走吧') >= 0.5);
  assert.equal(leakShare(REF_AUDIO, '大家快趁热吃吧'), 0);
  assert.equal(leakShare('我们下午去公园', '我们下午去公园', '我们下午去公园'), 0, 'user asked for these exact words');
  assert.ok(leakShare('Follow the wind, live free', 'follow the wind live free and smile') >= 0.5);
  assert.deepEqual(retentionMarkers(ref2va('weak_reference', 'x')), {'<Picture 1>': 'fully_preserved', '<Audio 1>': 'weak_reference'});
});

test('refSpeechOf keeps audio labels with speech, including video soundtracks', () => {
  const labels = [{label: '<Audio 1>', type: 'audio', conditionIndex: 0}, {label: '<Video 1>', type: 'video', conditionIndex: 0}, {label: '<Audio 2>', type: 'audio', conditionIndex: 1}, {label: '<Picture 1>', type: 'image', conditionIndex: 2}];
  const probes = [{asr: {segments: [{text: 'hello there friend'}]}}, {asr: {segments: []}}, {}];
  assert.deepEqual(refSpeechOf(labels, probes), [{label: '<Audio 1>', text: 'hello there friend'}]);
});

const clip = (o = {}) => ({qc: {pass: true}, asr: {text: '大家快趁热吃吧'}, dialogue: {mean: 1}, forbidden: [], ...o});
const C = {id: 'v', text: USER, expect_dialogue: ['大家快趁热吃吧。']};

test('verifier: QC, dialogue accuracy, forbidden words and voice leak all fail a clip', () => {
  assert.equal(verifyClip(clip(), C).pass, true);
  assert.deepEqual(verifyClip(clip({qc: {pass: false}}), C).reasons, ['qc_failed']);
  assert.equal(verifyClip(clip({dialogue: {mean: 0.6}}), C).pass, false);
  assert.equal(verifyClip(clip({forbidden: ['天气']}), C).pass, false);
  const leak = verifyClip(clip({asr: {text: '今天的天气真不错 我们下午去公园走走吧 大家快趁热吃吧'}}), C, [{label: '<Audio 1>', text: REF_AUDIO}]);
  assert.ok(leak.reasons.some((r) => r.startsWith('voice_ref_leak')));
  assert.equal(verifyClip(clip({dialogue: {mean: null}}), {id: 'x', text: 'no lines'}).pass, true, 'no expected dialogue -> QC only');
});

test('best-of-N stops at the first passing seed and otherwise keeps the best', async () => {
  const plan = {1: clip({dialogue: {mean: 0.5}}), 2: clip(), 3: clip()};
  const gen = [];
  const r = await generateBestOfN({c: C, seeds: [1, 2, 3], generate: async (s) => (gen.push(s), s), score: async (s) => plan[s]});
  assert.deepEqual(gen, [1, 2]);
  assert.equal(r.chosen.seed, 2);
  const bad = await generateBestOfN({c: C, seeds: [1, 2], generate: async (s) => s, score: async (s) => clip({dialogue: {mean: s === 2 ? 0.8 : 0.4}})});
  assert.equal(bad.chosen.seed, 2);
  assert.equal(pickBest([]), null);
});

test('offline simulation reads the judge score of the clip that would have shipped', () => {
  const scores = {v: {clips: {'v__raw__s1.mp4': clip({dialogue: {mean: 0.3}}), 'v__raw__s2.mp4': clip()}, judge: {1: {byKey: {raw: {overall: 4}}}, 2: {byKey: {raw: {overall: 8}}}}}};
  const sim = simulate([C], scores, [1, 2]);
  assert.equal(sim.raw.byN[1].meanOverall, 4);
  assert.equal(sim.raw.byN[2].meanOverall, 8);
  assert.equal(sim.raw.byN[2].meanAttempts, 2);
  assert.deepEqual(sim.raw.verifier, {tp: 1, fp: 0, fn: 0, tn: 1});
});

test('judge comparison: agreement, rank correlation and cross-vendor score', () => {
  assert.equal(spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1);
  assert.equal(spearman([1, 2, 3, 4], [4, 3, 2, 1]), -1);
  assert.equal(vendorOf('us.anthropic.claude-opus-5-5'), 'anthropic');
  assert.equal(vendorOf('us.openai.gpt-6-sol'), 'openai');
  const A = {c: {1: {byKey: {opus: {overall: 7}, gpt: {overall: 9}}, ranking: ['gpt', 'opus']}}};
  const B = {c: {1: {byKey: {opus: {overall: 9}, gpt: {overall: 7}}, ranking: ['opus', 'gpt']}}};
  const r = compareJudges(A, B, {modelA: 'us.openai.gpt-6-sol', modelB: 'us.anthropic.claude-opus-5-5', writerVendor: {opus: 'anthropic', gpt: 'openai'}});
  assert.equal(r.top1Agreement, 0);
  assert.equal(r.perArm.opus.crossVendor, 7, 'opus is only scored by the OpenAI judge');
  assert.equal(r.perArm.gpt.crossVendor, 7, 'gpt is only scored by the Anthropic judge');
  assert.equal(r.perArm.opus.meanBoth, 8);
});

test('judge comparison: a harsher judge is calibrated on neutral arms before cross-vendor scoring', () => {
  // judge B scores everything 1 point lower; the raw arm (no writer) reveals that
  const A = {c: {1: {byKey: {raw: {overall: 5}, opus: {overall: 8}, gpt: {overall: 8}}, ranking: ['opus']}}};
  const B = {c: {1: {byKey: {raw: {overall: 4}, opus: {overall: 7}, gpt: {overall: 7}}, ranking: ['opus']}}};
  const r = compareJudges(A, B, {modelA: 'us.openai.gpt-6-sol', modelB: 'us.anthropic.claude-opus-5-5', writerVendor: {opus: 'anthropic', gpt: 'openai'}});
  assert.equal(r.harshnessOffsetBminusA, -1);
  assert.equal(r.perArm.gpt.crossVendor, 7, 'uncalibrated: looks worse than opus');
  assert.equal(r.perArm.gpt.crossCalibrated, 8, 'calibrated: same as opus');
  assert.equal(r.perArm.opus.crossCalibrated, 8);
});

test('validator still passes clean IRs with no reference speech (no regressions)', () => {
  assert.deepEqual(errorsOf(validateIR(ref2va('reference', '大家快趁热吃吧。'), ctx)).map((i) => i.code), []);
});
