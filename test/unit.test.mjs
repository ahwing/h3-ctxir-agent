import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {computeLabels} from '../src/labels.mjs';
import {validateIR, errorsOf, I2VA_LINE} from '../src/validate.mjs';
import {buildRequest, SAMPLER} from '../src/h3.mjs';
import {parseJudge, dialogueScore, infixDistance, shuffled, normSpeech} from '../src/judge.mjs';
import {parseJson, usd, cachePoint, CACHE_ON, toResponsesBody, fromResponsesUsage} from '../src/bedrock.mjs';
import {systemPrompt, mediaContent} from '../src/agent.mjs';

const ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
const CASES = JSON.parse(fs.readFileSync(path.join(ROOT, 'cases/cases.json'), 'utf8'));
const byId = (id) => CASES.find((c) => c.id === id);
const codes = (issues) => errorsOf(issues).map((i) => i.code);

test('labels follow H3-Base numbering: a video soundtrack takes an Audio label before its Video label', () => {
  const l = computeLabels([{type: 'video'}, {type: 'audio'}], [{hasAudio: true}, {}]);
  assert.deepEqual(l.map((x) => x.label), ['<Audio 1>', '<Video 1>', '<Audio 2>']);
  const m = computeLabels([{type: 'image'}, {type: 'video'}, {type: 'image'}], [{}, {hasAudio: false}, {}]);
  assert.deepEqual(m.map((x) => x.label), ['<Picture 1>', '<Video 1>', '<Picture 2>']);
  assert.throws(() => computeLabels([{type: 'gif'}]));
});

test('validator passes all three official Context-IR examples (no errors)', () => {
  for (const c of CASES.filter((x) => x.official_ir)) {
    const labels = computeLabels(c.conditions, c.conditions.map((x) => ({hasAudio: x.type === 'video'})));
    assert.deepEqual(codes(validateIR(c.official_ir, {mode: c.mode, duration: c.duration, labels, expectDialogue: c.expect_dialogue})), [], c.id);
  }
});

test('validator catches the typical raw-prompt and agent mistakes', () => {
  const c = byId('gold_ref2va');
  const labels = computeLabels(c.conditions, [{hasAudio: true}, {}]);
  const ctx = {mode: 'ref2va', duration: 5, labels, expectDialogue: c.expect_dialogue};
  assert.ok(codes(validateIR(c.text, ctx)).includes('section_missing'), 'raw text has no sections');
  const bad = c.official_ir.replaceAll('<Audio 2>', '<Audio 3>').replace('Follow the wind, live free.', 'Follow the wind.').replace('[Shot 1] The shot begins', '[Shot 1] At 00:01.000, The shot begins');
  const k = codes(validateIR(bad, ctx));
  for (const want of ['label_undefined', 'label_unused', 'dialogue_lost', 'shot1_time']) assert.ok(k.includes(want), `${want} in ${k}`);
  const zh = c.official_ir.replace('realistic photographic style', '写实风格');
  assert.ok(codes(validateIR(zh, ctx)).includes('not_english'));
  const late = c.official_ir.replace('through the end of the video.', 'through the end. [Shot 2] At 00:06.000, the camera cuts to the lamb.');
  assert.ok(codes(validateIR(late, ctx)).includes('shot_time_range'));
});

test('validator: base modes', () => {
  const i2 = byId('gold_i2va');
  const ctx = {mode: 'i2va', duration: 8, labels: computeLabels(i2.conditions, [{}]), expectDialogue: []};
  assert.ok(i2.official_ir.startsWith(I2VA_LINE));
  assert.ok(codes(validateIR(i2.official_ir.replace(I2VA_LINE, 'Start from the image.'), ctx)).includes('i2va_instruction'));
  const t2 = byId('gold_t2va');
  assert.ok(codes(validateIR(`Hello.\n\n${t2.official_ir}`, {mode: 't2va', duration: 10, labels: [], expectDialogue: []})).includes('t2va_preamble'));
  assert.ok(codes(validateIR(t2.official_ir, {mode: 't2va', duration: 4, labels: [], expectDialogue: []})).includes('shot_time_range'), '4.5s cut in a 4s clip');
});

test('dialogue <d> needs a language tag and Chinese lines must stay verbatim', () => {
  const ir = 'integrated_multimodal_description: [Shot 1] Live-action, a man (S1) says: <d>这座塔已经站了九百年。</d> ' + 'word '.repeat(130) + '\n\noverall_soundscape: Wind.\n\nnon_diegetic_music: N/A';
  const k = codes(validateIR(ir, {mode: 't2va', duration: 5, labels: [], expectDialogue: ['这座塔已经站了九百年。']}));
  assert.ok(k.includes('d_lang'));
  assert.ok(!k.includes('dialogue_lost'));
  assert.ok(!k.includes('not_english'), 'Chinese inside <d> is allowed');
});

test('buildRequest: identical body across arms except prompt; Studio sampler settings', () => {
  const c = byId('zh_two_images');
  const base = {uris: ['u1', 'u2'], conditions: c.conditions, duration: 5, aspect: '16:9', seed: 7};
  const a = buildRequest({...base, prompt: 'A'});
  const b = buildRequest({...base, prompt: 'B'});
  assert.deepEqual({...a, prompt: 0}, {...b, prompt: 0});
  assert.equal(a.task, 'ref2va');
  assert.equal(a.num_inference_steps, SAMPLER.num_inference_steps);
  assert.deepEqual(a.conditions, [{type: 'image', uri: 'u1', role: 'reference'}, {type: 'image', uri: 'u2', role: 'reference'}]);
  assert.equal(buildRequest({...base, prompt: 'x', duration: 30}).target.duration_seconds, 15);
  assert.throws(() => buildRequest({...base, prompt: 'x', uris: ['u1']}));
});

test('judge parsing maps letters back to arms and rejects bad scores', () => {
  const j = {candidates: {A: {instruction_following: 8, reference_fidelity: 7, visual_quality: 6, motion_camera: 7, audio_dialogue: 9, overall: 8, notes: 'ok'}, B: {instruction_following: 3, reference_fidelity: 4, visual_quality: 6, motion_camera: 5, audio_dialogue: 2, overall: 3}}, ranking: ['A', 'B']};
  const p = parseJudge(j, {A: 'agent', B: 'raw'});
  assert.equal(p.byKey.agent.overall, 8);
  assert.deepEqual(p.ranking, ['agent', 'raw']);
  assert.throws(() => parseJudge({...j, candidates: {A: j.candidates.A}}, {A: 'agent', B: 'raw'}), /omitted/);
  assert.throws(() => parseJudge({candidates: {A: {...j.candidates.A, overall: 11}}}, {A: 'agent'}), /invalid/);
  assert.deepEqual(parseJson('Sure:\n```json\n{"a":1}\n```'), {a: 1});
});

test('dialogue score: semi-global CER, punctuation and case ignored', () => {
  assert.equal(infixDistance('abc', 'xxabcxx'), 0);
  assert.equal(normSpeech('Follow the wind, live free.'), 'followthewindlivefree');
  const s = dialogueScore(['大家快趁热吃吧。'], '嗯，大家快趁热吃吧！好');
  assert.equal(s.mean, 1);
  assert.ok(dialogueScore(['Follow the wind, live free.'], 'fallow the wind live free').mean > 0.9);
  assert.ok(dialogueScore(['这座塔已经站了九百年。'], '今天天气很好').mean < 0.3);
  assert.equal(dialogueScore([], 'x').mean, null);
});

test('shuffle is deterministic per seed and a permutation', () => {
  const a = shuffled([1, 2, 3], 'case:1001');
  assert.deepEqual(a, shuffled([1, 2, 3], 'case:1001'));
  assert.deepEqual([...a].sort(), [1, 2, 3]);
});

test('agent prompt embeds the official guide and the label table', () => {
  assert.match(systemPrompt('ref2va'), /retention_analysis/);
  assert.match(systemPrompt('t2va'), /integrated_multimodal_description/);
  const c = byId('gold_ref2va');
  const labels = computeLabels(c.conditions, [{hasAudio: true}, {}]);
  const probes = [{type: 'video', width: 1, height: 1, duration: 6, hasAudio: true, loudness: {meanDb: -20}, asr: {segments: []}, frames: []}, {type: 'audio', duration: 8.5, loudness: {meanDb: -18, maxDb: -1}, asr: {language: 'en', language_prob: 0.9, segments: [{start: 0, end: 2, text: 'hi'}]}}];
  const t = mediaContent(c, probes, labels).map((b) => b.text || '').join('\n');
  assert.match(t, /<Audio 1> = condition #1 \(video\), the soundtrack of the video file/);
  assert.match(t, /<Audio 2> = condition #2 \(audio\)/);
});

test('official Context-IR request: roles, ratio, duration, no frame/ref mix', async () => {
  const {buildIRRequest, parseTask} = await import('../src/minimax.mjs');
  const cases = JSON.parse(fs.readFileSync(path.join(ROOT, 'cases/cases.json'), 'utf8'));
  const byId = Object.fromEntries(cases.map((c) => [c.id, c]));
  const r = buildIRRequest(byId.gold_ref2va, ['https://x/v.mp4', 'https://x/a.mp3']);
  assert.equal(r.model, 'MiniMax-H3');
  assert.equal(r.ratio, 'adaptive');
  assert.deepEqual(r.content.map((x) => x.role || x.type), ['text', 'reference_video', 'reference_audio']);
  assert.equal(r.content[1].video_url.url, 'https://x/v.mp4');
  const i = buildIRRequest(byId.gold_i2va, ['https://x/i.png']);
  assert.equal(i.content[1].role, 'first_frame');
  assert.equal(buildIRRequest(byId.gold_t2va, []).content.length, 1);
  assert.throws(() => buildIRRequest({...byId.gold_t2va, text: ' '}, []), /non-empty/);
  assert.throws(() => buildIRRequest({mode: 'ref2va', text: 'x', duration: 5, conditions: [{type: 'image', role: 'first_frame'}, {type: 'audio'}]}, ['a', 'b']), /cannot be mixed/);
  const t = parseTask({task: {id: '1', status: 'succeeded', content: {prompt: 'P'}, usage: {prompt_tokens: 1e6, completion_tokens: 1e6}}});
  assert.equal(t.prompt, 'P');
  assert.equal(t.usd, 4.5);
});

test('one sentence per <d> block (opt-in H3_DIALOGUE_SPLIT=1)', async () => {
  process.env.H3_DIALOGUE_SPLIT = '1';
  const {checkDialogue} = await import('../src/validate.mjs');
  const codes = (ir) => {
    const iss = [];
    checkDialogue(ir, [], iss);
    return iss.map((i) => i.code);
  };
  assert.deepEqual(codes('<d>[English] Follow the wind, live free. Leave worries behind, enjoy the moment.</d>'), ['d_multi_sentence']);
  assert.deepEqual(codes('<d>[Chinese] 今天天气真好。我们走吧。</d>'), ['d_multi_sentence']);
  assert.deepEqual(codes('<d>[English] Follow the wind, live free.</d> He smiles. <d>[English] Leave worries behind, enjoy the moment.</d>'), []);
  assert.deepEqual(codes('<d>[English] Mr. Smith, the 3.5 version is ready!</d> <d>[Chinese] 大家快趁热吃吧。</d>'), []);
  delete process.env.H3_DIALOGUE_SPLIT;
});

test('usd prices cache write/read buckets and cachePoint helper', () => {
  const m = 'us.anthropic.claude-opus-5-5';
  // 1M plain in ($4) + 1M cache write ($5) + 1M cache read ($0.4) + 1M out ($20)
  assert.equal(+usd(m, {inputTokens: 1e6, cacheWriteInputTokens: 1e6, cacheReadInputTokens: 1e6, outputTokens: 1e6}).toFixed(4), 29.4);
  assert.equal(usd(m, {inputTokens: 1e6}), 4);
  assert.deepEqual(cachePoint(), CACHE_ON ? [{cachePoint: {type: 'default'}}] : []);
});

test('usd uses GPT prices with 1x cache write and per-model cache read', () => {
  // 6.1-sol US CRIS (model card): $2.2 in, $2.75 cache write, $0.11 cache read, $11 out
  assert.equal(+usd('us.openai.gpt-6.1-sol', {inputTokens: 1e6, cacheWriteInputTokens: 1e6, cacheReadInputTokens: 1e6, outputTokens: 1e6}).toFixed(4), 16.06);
  assert.equal(+usd('us.openai.gpt-6-luna', {inputTokens: 1e6, outputTokens: 1e6}).toFixed(4), 0.66);
});

test('GPT Responses body: system breakpoint, images as data URLs, cachePoint dropped', () => {
  const b = toResponsesBody({model: 'us.openai.gpt-6.1-terra', system: 'GUIDE', maxTokens: 50, cacheKey: 'k',
    blocks: [{image: {format: 'jpeg', source: {bytes: Buffer.from('ab')}}}, {cachePoint: {type: 'default'}}, {text: 'ask'}]});
  assert.equal(b.prompt_cache_options.mode, 'explicit');
  assert.deepEqual(b.input[0].content[0], {type: 'input_text', text: 'GUIDE', prompt_cache_breakpoint: {mode: 'explicit'}});
  assert.deepEqual(b.input[1].content, [{type: 'input_image', image_url: 'data:image/jpeg;base64,YWI='}, {type: 'input_text', text: 'ask'}]);
  const u = fromResponsesUsage({input_tokens: 7353, output_tokens: 6, total_tokens: 7359, input_tokens_details: {cached_tokens: 5242, cache_write_tokens: 1958}});
  assert.deepEqual([u.inputTokens, u.cacheReadInputTokens, u.cacheWriteInputTokens], [153, 5242, 1958]);
});
