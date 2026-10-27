# h3-ctxir-agent

用一个前置 agent 补上 MiniMax H3 闭源的前段 **H3-Context-IR**，让开源的 **H3-Base** 出片更好、少出错。

## 目标

MiniMax H3 = Context-IR（闭源：把用户的话 + 图/视频/音频理解成结构化 prompt）+ H3-Base（开源：出视频）+ 2K 放大（闭源）。
本项目只做第一段：

1. 提高出片效果：贴合用户意图、运镜、画面、声音。
2. 减少出错：台词念错/漏念、说话人认错、素材标签对错（如 `<Audio 1>` / `<Audio 2>`）。
3. 对准客户两个痛点：指令遵循 / 人物保持 / 细节不足；抽卡率高（出片不稳）。

比法固定为 A/B：**只换 prompt，其他全不变**（同素材、同种子、同一台 H3）。分数来自真实生成的视频盲评 + ASR 台词听写，不只看 prompt 文字。

## 怎么工作

```
用户的话 + 素材
  -> probe     抽帧、取音频、ASR 听写
  -> labels    算出 H3 的素材标签（<Picture n> / <Video n> / <Audio n>）
  -> plan      大模型先出 JSON 方案（意图、素材角色、说话人、台词、分镜）
  -> write     按官方写法指南（guides/）写 Context-IR
  -> validate  规则检查 + critic 新上下文审查 -> repair（最多 2 轮）
  -> H3-Base   ref2va / t2va / i2va 出视频
```

写手可换：Claude Opus 5.5（默认）、Claude Sonnet 5.5、GPT-6.1 Terra / Sol（Bedrock）。
Prompt Cache 默认开：Claude 走 Converse `cachePoint`，GPT 走 Responses API 的显式断点。

## 已测出的效果

rgpt（4 个 ref2va 案例 x 5 种子，每组 20 条真视频，盲评满分 10）：

| 写法 | 总分 | 指令遵循 | 人物保持 | 画面细节 | 能用率(>=7) | 最差 | 写一次 prompt |
|---|---|---|---|---|---|---|---|
| 原话（不用 agent） | 5.25 | 4.65 | 7.70 | 7.85 | 20% | 3 | $0 |
| Claude Opus 5.5 | 8.00 | 7.90 | 8.75 | 8.55 | 90% | 6 | ~$0.22 |
| GPT-6.1 Terra | 7.95 | 7.75 | 8.85 | 8.55 | 85% | 6 | ~$0.04（开缓存后） |
| GPT-6.1 Sol | 7.85 | 7.65 | 8.85 | 8.60 | 90% | 5 | ~$0.126（未开缓存） |
| 官方 Context-IR 接口 | 8.15 | 8.25 | 8.50 | 8.30 | 95% | 6 | ~$0.036 |

结论：

- agent 比原话好很多：总分 +2.7，指令遵循 +3，能用率 20% -> 85-90%。
- agent 和官方接口差不多（差 0.3 分内）。agent 画面和运镜略强；官方台词更准（r3：0.95 比 0.91）。
- 写手选 Opus 或 GPT-6.1 Terra。Terra 开缓存后约 $0.04/案例，接近官方接口价。
- Sonnet 5.5 便宜但效果掉到 7.47（rsonnet），不推荐。
- 学官方的"一句台词一个 `<d>`"没变好（r3 总分 8.60 -> 7.67），默认关（`H3_DIALOGUE_SPLIT=1` 打开）。
- 弱点：认对说话人（"图 + 声音像谁"案例，官方 9.0，自写 agent 6.4-8.0）。
- prompt 修不了的失败：G7e 显存被别的任务占时会 CUDA OOM，要靠重试或排期。

每轮报告在 `runs/<run>/report.md`（本地生成，不进仓库）。

## 接到 H3 上

### 1. 准备 H3-Base 服务

- 在 GPU 机上起 H3-Base 的 SGLang 服务（本项目用 G7e，每张 GPU 一个 ref2va 服务）。
- 本机通过 SSM 端口转发到 `127.0.0.1:39010` / `39011`，或用 `H3_URLS` 指向你的地址（逗号分隔）。
- 健康检查：`GET <url>/health` 返回 200。
- H3 机上没有 S3 权限时，素材由本机传 S3，请求里带 presigned GET URL（必须用 `H3CTX_BUCKET` 或 `~/.config/h3-ctxir/bucket` 指定桶）。

### 2. 准备本机

```bash
npm ci
python3 -m venv .venv && .venv/bin/pip install -r requirements-asr.txt   # ASR 听写
# 需要 ffmpeg / ffprobe，以及能调 Bedrock 的 AWS 凭证（BEDROCK_REGION，默认 us-east-1）
npm test
```

### 3. 在自己的流程里用

把用户的话和素材交给 `buildIR`，拿到的 `ir` 就是给 H3-Base 的 `prompt`：

```js
import {buildIR} from './src/agent.mjs';
import {stage, buildRequest, generate, h3Urls} from './src/h3.mjs';

// Same shape as an entry in cases/cases.json; condition files resolve relative to cases/
const c = {
  id: 'mine', mode: 'ref2va', duration: 5, aspect: '16:9',
  text: '让图1里的男人站在图2的木塔前，转身面对镜头说：“这座塔已经站了九百年。”',
  conditions: [{type: 'image', file: 'media/pinkman.jpg'}, {type: 'image', file: 'media/pagoda_ext.jpg'}],
  expect_dialogue: ['这座塔已经站了九百年。'], // optional, used by the validator
};
const {ir, labels, ledger} = await buildIR(c, {workDir: 'runs/mine/work'});

const uris = await Promise.all(c.conditions.map((x) => stage(`cases/${x.file}`)));
const body = buildRequest({prompt: ir, uris, conditions: c.conditions, duration: c.duration, aspect: c.aspect, seed: 42});
await generate(h3Urls()[0], body, 'out.mp4');
```

- 素材顺序就是标签顺序；视频自带的声音会先占一个 `<Audio n>`，`labels` 里能看到实际编号。
- `ledger` 记着每次模型调用的 token 和花费。
- 案例格式照 `cases/cases.json` 写。

### 4. 跑 A/B 对比

```bash
node src/run.mjs --run r5 --cases gold_ref2va,zh_voice_timbre --seeds 5 \
  --arms raw,agent,api --phase all --prev r2
```

- `--arms`：`raw` 原话、`agent` 本项目、`official` 文档示例、`api` 官方 Context-IR 接口（需 MiniMax key，放在 `~/.config/h3-ctxir/minimax.key`，国内站 key 走 `api.minimaxi.com`）。
- `--phase`：`ir` 只写 prompt、`gen` 出视频、`score` 盲评、`report` 出报告。
- `--prev rN`：把上一版 agent 当 `agent_v1` 组同种子对比。
- 跑前确认 GPU 机空闲，否则会排队或 OOM。

### 常用开关

| 环境变量 | 作用 | 默认 |
|---|---|---|
| `IR_WRITER_MODEL` | 写手 | `us.anthropic.claude-opus-5-5` |
| `IR_CRITIC_MODEL` | 审查 | `us.anthropic.claude-sonnet-5-5` |
| `JUDGE_MODEL` / `IR_JUDGE_MODEL` | 评委（要和写手不同厂商） | `us.openai.gpt-6-sol` |
| `H3_PROMPT_CACHE` | `0` 关提示词缓存 | 开 |
| `H3_DIALOGUE_SPLIT` | `1` 开拆句台词规则 | 关 |
| `H3_URLS` | H3 服务地址 | `http://127.0.0.1:39010,http://127.0.0.1:39011` |

## 局限

- 只能写文字版 prompt，进不了官方 Context-IR 的内部格式；2K 放大段也没开源。
- 评委只看 9 帧拼图 + 台词听写，看不全动作，也没评音色。
- 每组 15-20 次评比，差 0.3 分以内的不算定论。
- 测 `api` 组时素材会发到 MiniMax 服务器。
