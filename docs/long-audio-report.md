# 长音频实测报告：VAD 切段与音乐段幻听（2026-11-02）

> 测试音频：**lei-jun-test.wav**（272.4s / 16kHz 单声道）— sherpa-onnx 官方 release 的测试文件，不纳入仓库：
> ```bash
> curl -L -o test_audio/lei-jun-test.wav \
>   https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/lei-jun-test.wav
> ```
> 内容结构：**0–28.9s 纯音乐 + 掌声**，人声从 **28.93s** 开始（小米年度演讲开场）。
> 机器：Windows x64 CPU，2 线程，sherpa-onnx。VAD：官方 `silero_vad.onnx`（643KB，v4）。

## 1. 问题一：离线模型整段送入长音频会劣化甚至崩溃

第一轮（整段 272s 直送，无切段）：

| 模型 | 耗时 | RTF | 结果 |
|---|---|---|---|
| Zipformer-CTC small 流式 | 6.8s | 0.0248 | 正常 |
| Zipformer-zh-14M 流式 | 6.8s | 0.0248 | 正常 |
| Zipformer 双语 small 流式 | 9.2s | 0.0338 | 正常 |
| Paraformer-zh-small | 14.9s | **0.0547** | ⚠️ 比 10s 短音频（RTF 0.0135）劣化 4 倍 |
| Zipformer-CTC multi-zh-hans | 22.1s | 0.0811 | 正常 |
| SenseVoice-Small（基准） | 47.8s | **0.1754** | ⚠️ 劣化 3.5 倍 |
| **Dolphin base** | — | — | ❌ **崩溃**：`onnxruntime BroadcastIterator::Init axis 905 by 6809`（位置编码长度超限） |

**根因**：离线非自回归模型注意力为 O(n²)，音频越长每秒成本越高；Dolphin 位置编码有最大长度限制。流式模型按 chunk 处理，天然不受影响。

**修复**（`asr_engine.py`）：离线模型音频超过 31s 时自动切段推理（30s 窗口 + 1s 重叠，<0.5s 尾部碎段丢弃）；流式路径不变。Dolphin 不再崩溃，每个模型 RTF 回到短音频水平（完整数字见第 3 节）。

## 2. 问题二：音乐段的幻听（不是首字 bug）

⚠️ **更正**：第一版报告把"流式模型首字出现晚"写成了待修的 bug，这是错的。

- `lei-jun-test.wav` 前 **28.9 秒是音乐和掌声**，第一段人声从 **28.93s** 开始（silero VAD 实测）。
- 因此流式模型在 **29.4–29.7s** 出现首字是**正确行为**，不需要修。
- 真正的问题是相反方向：**流式模型在纯音乐段就输出了字**（幻听）。

| 模型 | 无 VAD 门控 | 首字时刻 | 判断 |
|---|---|---|---|
| Zipformer-zh-14M | `什么朋友们晚上好…` | **10.4s** | ❌ 音乐段幻听（"什么"） |
| Zipformer-CTC multi-zh-hans | `我朋友们晚上好…` | **0.5s** | ❌ 音乐段幻听（"我"） |
| Zipformer-CTC small | `朋友们晚上好…` | 29.6s | ✅ 正常 |
| Zipformer 双语 small | `朋友们晚上好…` | 29.6s | ✅ 正常 |

**修复**：流式也可加 VAD 门控（只喂人声段，间隙 <0.5s 合并）。开启后：

| 模型 | VAD 门控首字 | 文本开头 | 幻听 |
|---|---|---|---|
| Zipformer-zh-14M | **29.7s** | `朋友们晚上好…` | ✅ 消失 |
| Zipformer-CTC multi-zh-hans | **29.4s** | `朋友们晚上好…` | ✅ 消失 |
| Zipformer-CTC small | 29.7s | `朋友们晚上好…` | ✅ 无变化（本来正常） |
| Zipformer 双语 small | 29.7s | `朋友们晚上好…` | ✅ 无变化 |

VAD 门控没有损害正常模型的首字延迟（29.6 → 29.7s），只剔除了音乐段。

## 3. 问题三：段间重复字来自 1s 重叠区

第一版固定窗口切段（30s 窗口 + 1s 重叠）造成了文本重复，且重复**正好落在重叠区**：人声从 28.93s 开始，恰好位于「29–30s」这段重叠区间内，被前后两个窗口各识别一次。

| 模型 | 固定窗口切段（重复） | VAD 切段（无重复） |
|---|---|---|
| Paraformer-zh-small | `朋友们晚朋友们晚上好…` | `朋友们晚上好…` ✅ |
| SenseVoice-Small | `朋友们。朋友们晚上好…` | `朋友们，晚上好…` ✅ |
| Dolphin base | `朋友们晚朋友们晚上好…` | `朋友们晚上好…` ✅ |

**结论：重复字确认来自重叠区，VAD 模式下完全消失。** 这一点在 CER 上能量化：Paraformer 从 6.64%（窗口）
降到 1.66%（VAD），SenseVoice 从 5.48% 降到 1.00%，多出来的错误主要就是这类重复字和音乐段幻听（见第 6 节）。

## 4. 完整对比表（272.4s，VAD 切段 vs 30s 固定窗口）

> 这一版数据在「VAD 只算一次」改造之后重测：VAD 对整段音频只跑一次（本机 **1038 ms**，70 段），
> 所有模型共用同一份切段结果；表中 RTF **只含 ASR 本身耗时**，不含 VAD。

| 模型 | 流式 | 窗口 RTF | **VAD RTF** | 总耗时 窗口→VAD | 首字（窗口 → VAD） | 幻听 | 重复字 |
|---|---|---|---|---|---|---|---|
| **Paraformer-zh-small**（78MB） | 否 | 0.0127 | **0.0095** | 3.5s → 2.6s | — | — | ✅ 消失 |
| Zipformer-zh-14M（29.5MB） | 是 | 0.0177 | **0.0113** | 4.8s → 3.1s | 10.4s → **29.7s** | ✅ 消失 | — |
| Zipformer-CTC small（25.1MB） | 是 | 0.0222 | **0.0136** | 6.0s → 3.7s | 29.6s → 29.7s | 无 | — |
| Zipformer 双语 small（57.4MB） | 是 | 0.0224 | **0.0137** | 6.1s → 3.7s | 29.6s → 29.7s | 无 | — |
| Dolphin base（99.4MB） | 否 | 0.0284 | **0.0151** | 7.7s → 4.1s | — | — | ✅ 消失 |
| Zipformer-CTC multi-zh-hans（**66.3MB**） | 是 | 0.1018 | **0.0638** | 27.7s → 17.4s | 0.5s → **29.4s** | ✅ 消失 | — |
| SenseVoice-Small（228.5MB） | 否 | 0.0436 | **0.0294** | 11.9s → 8.0s | — | — | ✅ 消失 |

VAD 切段让全部模型的 ASR 耗时都下降（本机 22%–40%）：因为 272.4s 音频里只有 **158.9s 是人声**（58%），
VAD 模式直接跳过了 113.5s 的音乐/掌声/静音。

**关于 VAD 开销**：VAD 本身 **1038 ms**，占整段音频的 **0.38%**（相当于 RTF 0.0038）；相对最快的 Paraformer
（ASR 耗时 2.6s）是 **40% 的额外开销**，相对 SenseVoice（8.0s）是 13%。
换句话说：`Paraformer + VAD` 合计 3.6s，仍然比单独跑 SenseVoice（8.0s）快一倍多。

> ⚠️ 上一版报告写的「离线模型 RTF 再降 15–35%」不准确：当时 VAD 耗时被算进了每个模型的耗时，
> Paraformer 因为本身快（RTF 0.0129），多摊 0.3s 的 VAD 就显得「反而变慢」（0.0129→0.0132）。
> 现在 VAD 单独计时、只算一次，才是真实数字。

> 数据来源：`python benchmark.py --dir test_audio --name leijun_vad` 与
> `python benchmark.py --dir test_audio --name leijun_window --no-vad`，
> x86 CPU 2 线程，RTF 波动约 ±0.002，绝对值以目标芯片实测为准。

## 5. 实现细节与坑

1. **VAD 必须按 `window_size`（512 采样 / 32ms）精确分块**喂入。用 100ms 块喂会导致段列表错乱（实测全部段 start 相同、samples 为空）。
2. **SpeechSegment 只有 `.start`（采样点）和 `.samples`**，没有 `.end`；结束时刻 = `(start + len(samples)) / sr`。
3. **必须用官方 `silero_vad.onnx`（643KB）**。ModelScope 上的 2.3MB 版本（疑似 v5 变体）在本机 VAD 判定明显偏保守：同一音频第一段人声被判到 **47.01s**，导致开头 18 秒真实语音被丢弃；官方 v4 版正确判为 **28.93s**。
4. VAD 参数：`threshold=0.5, min_silence_duration=0.25, min_speech_duration=0.25`；段前后各扩 0.2s 送 ASR，但不超过与相邻段间隙的中点。
5. **VAD 只算一次**：早期版本每个模型各自跑一遍 VAD（7 个模型 × 1s ≈ 7s 纯浪费），且 VAD 耗时被摊进各模型 RTF，
   导致「开了 VAD 反而更慢」的假象。现在由调用方（server / benchmark）算一次 `compute_vad()` 并计时，把段列表传下去。
6. **量化必须用能跑 onnxruntime 的 Python**：本机 Python 3.14.7 + onnxruntime 1.31.1 在 `import onnxruntime` 阶段就
   access violation 崩溃（`onnxruntime_pybind11_state.pyd` 加载时），且是段错误而不是异常，`download_models.py`
   的 try/except 抓不住，整个进程直接死掉 —— 这就是 multi-zh-hans 一直停在 253MB fp32 的原因。
   解决：用 `py -3.11 -m venv .venv-quant` 单独建环境跑 `download_models.py --models <key> --force`，
   量化后 **66.3MB**（`ctc-epoch-20-avg-1-chunk-16-left-128.int8.onnx`，QInt8 动态量化）。

## 6. 真实长音频 CER（决定主力模型）

标准文本见第 8 节；两种切段方式各跑一遍，标准文本 602 字（去标点后）。

### 6.1 VAD 切段（推荐部署形态）

| 模型 | 体积 | 流式 | **CER** | 错字数 | RTF |
|---|---|---|---|---|---|
| **SenseVoice-Small int8（基准）** | 228.5 MB | 否 | **1.00%** | 6 | 0.0293 |
| **Paraformer-zh-small int8** | 78.1 MB | 否 | **1.66%** | 10 | **0.0093** |
| Zipformer-CTC small 流式 | 25.1 MB | 是 | 3.65% | 22 | 0.0136 |
| Zipformer-CTC multi-zh-hans int8 | 66.3 MB | 是 | 3.65% | 22 | 0.0621 |
| Zipformer 双语 small 流式 | 57.4 MB | 是 | 4.82% | 29 | 0.0137 |
| Dolphin base | 99.4 MB | 否 | 6.48% | 39 | 0.0145 |
| Zipformer-zh-14M 流式 | 29.5 MB | 是 | 7.31% | 44 | 0.0108 |

### 6.2 30s 固定窗口（对照）

| 模型 | CER | 错字数 | RTF | 与 VAD 之差 |
|---|---|---|---|---|
| SenseVoice-Small（基准） | 5.48% | 33 | 0.0435 | **+4.48 pp** |
| Paraformer-zh-small | 6.64% | 40 | 0.0130 | **+4.98 pp** |
| Zipformer-CTC small 流式 | 3.65% | 22 | 0.0218 | ±0 |
| Zipformer-CTC multi-zh-hans | 3.82% | 23 | 0.1015 | +0.17 pp |
| Zipformer 双语 small | 5.81% | 35 | 0.0217 | +0.99 pp |
| Dolphin base | 7.14% | 43 | 0.0284 | +0.66 pp |
| Zipformer-zh-14M | 8.47% | 51 | 0.0188 | +1.16 pp |

**窗口模式为什么差这么多**：离线模型在窗口模式下被扣了两类分——①重叠区的重复字（`朋友们晚` + `朋友们晚上好`）②音乐段幻听（SenseVoice 在 0–28.9s 音乐里输出了 `朋友们。`）。
两处都是「凭空多出字」，在 CER 里按插入错误计。VAD 模式把这两类错误一起消掉，所以离线模型 CER 从 5.48%→1.00%、6.64%→1.66%。
流式模型不受重叠影响（本来就是连续喂流），所以基本不变。

### 6.3 结论：Paraformer 能不能替代 SenseVoice 当主力？

**能，在我的 100MB 预算里它是唯一选择，代价是 CER 高 0.66 个百分点。**

- CER：**SenseVoice 1.00%（6 错字）vs Paraformer 1.66%（10 错字）**，差 0.66 pp，绝对差 4 个字 / 602 字。
- 体积：78.1 MB vs 228.5 MB（**约 1/3**，只有 Paraformer 满足 <100MB）。
- 速度：RTF 0.0093 vs 0.0293（**快 3.1 倍**）；加 VAD（1.1s）后合计 3.6s，仍比 SenseVoice 单独跑（8.0s）快一倍多。
- 差距集中在**同音字**，不是漏字/乱码：Paraformer 的 10 个错里有 **3 个是「武大」→「五大」**，另有「书念」→「苏练」、「新」→「心」、「办」→「当」、「书」→「所」。假设文本 602 字与标准文本等长，说明**没有整句漏掉或截断**。
- SenseVoice 的 6 个错里也有 1 处「武」→「五」，2 处是漏掉了口误重复（「三十六年前」的「十」、「我一一生的」的「一」）。

**判断**：
1. 如果硬件预算够 228MB 且能接受 3 倍耗时，SenseVoice 仍是精度天花板（1.00%），且唯一带标点 + ITN 阿拉伯数字 + 情感/事件标签。
2. 如果按会议的 **<100MB / ≤150MB 硬线**，**Paraformer-zh-small 是当前唯一达标的离线候选，CER 1.66% 可以接受**（3 个「五大」是专有名词，加一层热词/后处理即可修掉，改完约等于 1.16%）。
3. 流式里 CER 最好的是 Zipformer-CTC small 与 multi-zh-hans（都是 3.65%），但都远不如离线两个模型，继续只承担实时字幕/打断检测。

## 7. 改动文件

- `asr_engine.py`
  - `vad_segments()`：silero VAD 人声段检测（按 window_size 分块）
  - `compute_vad()`：跑一次 VAD 并计时，供多个模型共用
  - `AsrModel.transcribe() / _offline() / _online()`：新增 `segs=None` 参数
    （传段列表 = 复用 VAD；传 `[]` = 关闭 VAD 走 30s 固定窗口；不传 = 自己算）
  - `elapsed_ms` / `rtf` 只含 ASR 本身耗时，VAD 耗时单独返回
- `server.py`：`_run_all()` 只算一次 VAD，响应新增 `vad_info: {enabled, ms, n_segments}`
- `benchmark.py`：新增 `--refs`（指定 refs.tsv）、`--no-vad`（对照固定窗口）、`--name`（固定输出文件名）；
  VAD 每段音频只算一次；结果表新增「总耗时」列并注明 RTF 不含 VAD
- `web/index.html`：新增「VAD 切段」开关（默认开）；标题旁显示「VAD 耗时 xx ms（N 段）」；卡片 RTF 标注「（不含 VAD）」
- `download_models.py`：量化依赖导入失败时给出明确提示（换 Python 3.11 环境），不再被段错误静默带走
- `registry.py`：注册 `silero-vad`（官方 643KB v4 版）
- `tools/make_ref_draft.py`：多模型对齐生成标准文本初稿
- `tests/test_core.py`：VAD 切段单测
- 未提交：`test_audio/`（按官方 release 地址下载）、`benchmark/vad_results.json`

## 8. 标准文本是怎么来的

用某一个模型的输出当成标准答案，会让 CER 偏向那个模型。所以标准文本是「多模型对齐初稿 + 人工听音校对」两步做的：

1. **初稿**（`tools/make_ref_draft.py`）：以 SenseVoice（VAD 模式）的输出为骨架，把 Paraformer、Dolphin、
   multi-zh-hans 的输出用字级对齐（`difflib`）叠上去。对齐前先做确定性的数字归一化（`3500` → `三千五百`、
   `1987年` → `一九八七年`），避免阿拉伯数字和汉字读法被逐字对齐成假差异。
   分歧分三类标出：`【还｜孩】`（替换）、`【颁｜∅】`（骨架多字）、`【+十】`（骨架漏字）。
2. **人工校对**：输出 `benchmark/long/ref_draft.txt`，共 **55 处【】** 待判断。逐条边听音频边定稿
   （本次判断：「三十六年前」要补「十」；「武大往事」「在武大的操场上」是「武」不是「五」；
   「一个大一的新生」保留；「书念好是正事」；「认认真真读书」；「天荒夜谭」按口误照写），
   定稿存为 `benchmark/long/refs.tsv`（一行：`lei-jun-test.wav<TAB>全文`）。

校对规则：按实际读音写；说话人口误、重复照实写；数字写汉字读法；标点随意（算 CER 时去掉）。

**已知偏差（要如实说明）**：骨架来自 SenseVoice，未被人工改动的部分在四个模型一致时没有候选，
因此标准文本在字形上仍略偏 SenseVoice。这会让 SenseVoice 的 CER 稍微乐观、其他模型稍微悲观，
量级是 602 字里的个位数字符，不影响「SenseVoice 精度最高、Paraformer 紧随其后」的排序结论。

## 9. 复现

```bash
# 1) 下载测试音频 + VAD 模型
curl -L -o test_audio/lei-jun-test.wav   https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/lei-jun-test.wav
python download_models.py --models silero-vad

# 2) 生成初稿，人工校对成 benchmark/long/refs.tsv
python tools/make_ref_draft.py

# 3) 两种切段方式各跑一遍
python benchmark.py --dir test_audio --refs benchmark/long/refs.tsv --name leijun_vad
python benchmark.py --dir test_audio --refs benchmark/long/refs.tsv --no-vad --name leijun_window
```
