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

**结论：重复字确认来自重叠区，VAD 模式下完全消失。**

## 4. 完整对比表（272.4s，VAD 切段 vs 整段/固定窗口）

| 模型 | 流式 | 无 VAD RTF | **VAD RTF** | 首字（无 VAD → VAD） | 幻听 | 重复字 |
|---|---|---|---|---|---|---|
| **Paraformer-zh-small**（78MB） | 否 | 0.0129 | **0.0132** | — | — | ✅ 消失 |
| Zipformer-zh-14M（29.5MB） | 是 | 0.0173 | **0.0153** | 10.4s → **29.7s** | ✅ 消失 | — |
| Zipformer-CTC small（25.1MB） | 是 | 0.0220 | **0.0175** | 29.6s → 29.7s | 无 | — |
| Zipformer 双语 small（57.4MB） | 是 | 0.0217 | **0.0172** | 29.6s → 29.7s | 无 | — |
| Dolphin base（99.4MB） | 否 | 0.0283 | **0.0187** | — | — | ✅ 消失 |
| Zipformer-CTC multi-zh-hans（253MB） | 是 | 0.0683 | **0.0452** | 0.5s → **29.4s** | ✅ 消失 | — |
| SenseVoice-Small（228.5MB） | 否 | 0.0434 | **0.0331** | — | — | ✅ 消失 |

VAD 切段对全部模型都有收益：离线模型 RTF 再降 15–35%（免去音乐段计算），流式模型降 20–35%（不处理非人声）。VAD 本身开销很小（272s 音频约 0.3s）。

> 表中数字取 `python benchmark_vad.py` 的单次运行，x86 CPU 2 线程下 RTF 波动约 ±0.002，横向比较可看，绝对值以目标芯片实测为准。

## 5. 实现细节与坑

1. **VAD 必须按 `window_size`（512 采样 / 32ms）精确分块**喂入。用 100ms 块喂会导致段列表错乱（实测全部段 start 相同、samples 为空）。
2. **SpeechSegment 只有 `.start`（采样点）和 `.samples`**，没有 `.end`；结束时刻 = `(start + len(samples)) / sr`。
3. **必须用官方 `silero_vad.onnx`（643KB）**。ModelScope 上的 2.3MB 版本（疑似 v5 变体）在本机 VAD 判定明显偏保守：同一音频第一段人声被判到 **47.01s**，导致开头 18 秒真实语音被丢弃；官方 v4 版正确判为 **28.93s**。
4. VAD 参数：`threshold=0.5, min_silence_duration=0.25, min_speech_duration=0.25`；段前后各扩 0.2s 送 ASR。

## 6. 结论

1. **端侧常驻场景必须配 VAD**——不只是为了切段控 RTF，更是为了消除音乐/噪声段的幻听。两件事一次解决。
2. 用 VAD 切段后，长音频 RTF 排序：**Paraformer-zh-small（0.0132，78MB）< Zipformer-zh-14M（0.0153，29.5MB）< Zipformer 双语 small（0.0172）≈ Zipformer-CTC small（0.0175）< Dolphin（0.0187）< SenseVoice（0.0331）< multi-zh-hans（0.0452）**。
3. 总选型结论不变：**主力转写仍是 Paraformer-zh-small int8**（最快 + 离线 + 准确）；流式 Zipformer 承担实时字幕/打断，且应加 VAD 门控。
4. 待办：VAD 段可再按语义合并（当前间隙 >0.5s 就切，短句会被切碎）；分角色场景下 VAD + 声纹聚类的增量方案仍需在目标芯片实测。

## 7. 改动文件

- `asr_engine.py`
  - `vad_segments()`：silero VAD 人声段检测（按 window_size 分块）
  - `AsrModel._offline()`：VAD 切段优先，无 VAD 时退回固定窗口，短音频整段直送
  - `AsrModel._online()`：可选 VAD 门控（`use_vad`）
- 未提交 `test_audio/`（测试音频按官方 release 地址下载）

详见总选型报告 [selection.md](selection.md)。
