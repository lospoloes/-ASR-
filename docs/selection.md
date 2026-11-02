# 端侧语音转写（ASR）模型选型调研

> 背景：2026-10-29 会议《语音转文本模型相关工作讨论》。陪伴型机器人需要 **24 小时常驻监听、本地缓存转写文本**，未唤醒时也要保留上下文。端侧语音转写是整个交互体系的基础。会议确定 **以 SenseVoice 为性能基准**，表现不如它的候选全部淘汰。
>
> 调研日期：2026-11-02 · 推理框架：[sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) 2.0.1 · 实测机器：云端 x86 Intel Xeon 2.1GHz、4 vCPU（**不是**机器人目标芯片，RTF 只能横向比较）

## 0. 结论

| 场景 | 推荐 | 体积 | 理由 |
|---|---|---|---|
| **主力转写（精度优先）** | **Paraformer-zh-small int8** | 78 MB | 测试集 CER 4.99%，**略好于 SenseVoice 基准（5.57%）**，体积约为基准的 1/3，速度约为基准的 3 倍（RTF 0.020 vs 0.060）。四川、天津、河南方言的真人录音转写质量和 SenseVoice 相当。缺点是离线模型，需要配合 VAD 切段。 |
| **实时字幕 / 打断检测（流式）** | **Zipformer 流式中英双语 small** | 58 MB | 流式模型里精度最高（CER 13.2%），能处理中英混说，0.9 秒左右出现首字。 |
| **常驻监听整体方案** | Silero VAD + 双路：流式 Zipformer 出实时字幕 → VAD 切段后 Paraformer-small 出最终文本 | 约 137 MB | 两个模型都在 150MB 以内。流式路负责打断和实时反馈，离线路负责写入本地缓存的最终文本。 |
| 分角色 | pyannote-seg-3.0 + 3D-Speaker ERes2Net | 39 MB | 已集成进 demo。在双人对话测试中，A/B/A/B 四段全部分对，时间戳也准确。 |
| 方言 / 粤语补充 | Dolphin base（99 MB），或保留 SenseVoice | — | 粤语真人录音只有 SenseVoice 和 Dolphin 能正确转写，其余候选全部失败。 |

**哪些候选被淘汰（按会议规则：不如 SenseVoice 的淘汰）**：流式候选在 CER 上都明显不如 SenseVoice（13%–24% 对 5.6%），单独作为最终转写都不合格，只适合承担实时字幕和打断检测。离线候选中，只有 Paraformer-zh-small 达到基准。

> ⚠️ 测试集是用 TTS 合成的语音（见第 4 节），比真人语音"干净"，CER 偏乐观。结论要用真人录音复核，demo 页面可以直接录音对比。

---

## 1. 选型标准

- **硬标准**：量化后 ≤150 MB，并且 CPU 可跑（优先选 sherpa-onnx 支持的模型）。
- **加分项**（满足任意一项且表现特别出色，也可以入选）：流式、<100 MB、中英混说、方言。
- 体积以官方文档里 `ls -lh` 的 int8 文件大小为准，并注明出处。查不到的信息写"未核实"。
- 会议要求：Sense Voice 作为基准，表现不如它的候选全部淘汰。

## 2. 候选总表

体积出处：如无特别说明，均来自 k2-fsa/sherpa 官方文档 `docs/source/onnx/...` 中的 `ls -lh` 输出（文档站 https://k2-fsa.github.io/sherpa/onnx/ ，源文件在 GitHub k2-fsa/sherpa）。

### 2.1 进入 demo 的 7 个

| # | 模型 | int8 体积（出处） | 流式 | 公开中文 CER（测试集 / 出处） | 许可证 | 链接 |
|---|---|---|---|---|---|---|
| 1 | **SenseVoice-Small** int8（2024-07-17，基准） | 228 MB（sense-voice/pretrained：`model.int8.onnx 228M`） | 否 | 约 2.9%（AISHELL-1 test）/ 约 7.6%（WenetSpeech test_net）/ 约 7.4%（test_meeting）。数值是从[官方 README](https://github.com/FunAudioLLM/SenseVoice) 的柱状图 `image/asr_results1.png` 目测读出的，对应 PyTorch 原模型，不是 int8 | FunASR Model License（README 写明官方澄清可商用，需要署名） | [文档](https://k2-fsa.github.io/sherpa/onnx/sense-voice/pretrained.html) |
| 2 | **Zipformer-CTC small 流式** zh int8（2025-04-01） | 25 MB（online-ctc/zipformer-ctc-models：`model.int8.onnx 25M`） | 是 | 未核实（官方文档没有给出） | Apache-2.0（icefall/sherpa-onnx 项目；模型卡未核实） | [文档](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-ctc/zipformer-ctc-models.html) |
| 3 | **Zipformer 流式 zh-14M**（2023-02-23） | 文档：全 int8 共 24.5 MB（encoder 21M + decoder 1.8M + joiner 1.7M）；**实际可用配置 29.9 MB**（decoder 必须用 fp32 的 7.2M，见第 5 节） | 是 | 未核实 | Apache-2.0（icefall；模型卡未核实） | [文档](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-transducer/zipformer-transducer-models.html) |
| 4 | **Zipformer 流式中英双语 small**（2023-02-16） | 文档：全 int8 共 47.5 MB（encoder 41M + decoder 3.4M + joiner 3.1M）；**实际可用配置 58 MB**（decoder 用 fp32 的 14M） | 是 | 未核实（社区贡献的模型，用内部数据训练） | 未核实 | 同上 |
| 5 | **Zipformer-CTC 流式 multi-zh-hans**（2023-12-13） | 文档：int8 68 MB；**当前 GitHub 压缩包只有 fp32（252 MB）**，下载脚本会自动做动态量化，量化后 66 MB | 是 | 未核实（icefall multi_zh-hans RESULTS.md 没有单列这个版本；同系列 large 流式版 CTC 在 AISHELL-1 test 上为 1.97%） | Apache-2.0（icefall；模型卡未核实） | [文档](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-ctc/zipformer-ctc-models.html) |
| 6 | **Paraformer-zh-small** int8（2024-03-09） | 79 MB（offline-paraformer：`model.int8.onnx 79M`；参考起点实测值 82MB） | 否 | 未核实（small 版没有公开 CER；大版 Paraformer-zh 在 AISHELL-1 上约 1.9%，读自 SenseVoice README 柱状图，不能直接类比） | FunASR Model License（未逐条核实） | [文档](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-paraformer/paraformer-models.html) |
| 7 | **Dolphin base** CTC int8（2025-04-02） | 99 MB（Dolphin/pretrained：`model.int8.onnx 99M`，total 100M；参考起点记 104MB） | 否 | 未核实（官方 README 没有 base 版的中文 CER 表） | Apache-2.0（[Dolphin README](https://github.com/DataoceanAI/Dolphin)） | [文档](https://k2-fsa.github.io/sherpa/onnx/Dolphin/pretrained.html) |

### 2.2 调研过但没进 demo 的

| 模型 | int8 体积（出处） | 流式 | 公开 CER | 没进 demo 的原因 |
|---|---|---|---|---|
| WeNet WenetSpeech（`csukuangfj/sherpa-onnx-zh-wenet-wenetspeech`） | 参考起点实测 83 MB（**未核实**：调研环境访问不了 HF，文件列表查不到） | 可流式 | 未核实 | 只有 HF 一个下载源，文件名未核实。已作为实验项写进注册表：`python download_models.py --models wenet-wenetspeech`，下载失败不影响 demo |
| Zipformer 流式 zh int8（2025-06-30，large） | 160 MB（encoder 154M + decoder 4.9M + joiner 1.0M） | 是 | AISHELL-1 test：Transducer 流式 1.91%、CTC 流式 1.97%；WenetSpeech test_net 8.54% / test_meeting 7.91%（icefall `egs/multi_zh-hans/ASR/RESULTS.md`，zipformer large） | 超过 150MB 硬线（只超了约 10MB），但公开 CER 是所有流式模型里最好的。**如果硬件允许，建议优先补测** |
| Zipformer 流式 zh xlarge int8（2025-06-30） | 728 MB+ | 是 | 见 RESULTS.md | 体积太大 |
| SenseVoice int8（2025-09-09 粤语增强版） | 226 MB | 否 | 未核实 | 文档说明不支持标点，体积和基准相同，没有额外价值 |
| Paraformer-zh int8（2024-03-09，大版） | 217–232 MB | 否 | 约 1.9%（AISHELL-1，读自 SenseVoice README 柱状图） | 超过 150MB |
| Paraformer 川渝方言 int8（2025-10-07） | 227 MB | 否 | 未核实 | 超过 150MB；方言需求可以先用 small 版和 Dolphin 覆盖 |
| Paraformer 流式中英（online-paraformer） | encoder 158M + decoder 68M | 是 | 未核实 | 超过 150MB |
| Dolphin small int8 | 239 MB | 否 | 未核实 | 超过 150MB |
| Whisper tiny / small | tiny.en：encoder 12M + decoder 105M（int8） | 否 | Whisper-Small 在 AISHELL-1 上约 10%（读自 SenseVoice README 柱状图） | 中文明显差于基准 |
| FunASR-Nano int8（2025-12-30） | 约 948 MB | 否 | 未核实 | 体积太大（LLM 架构） |
| Qwen3-ASR 0.6B int8 | 约 937 MB | 否 | 未核实 | 体积太大 |

## 3. 体积和下载情况（本次实测）

`python download_models.py` 默认下载的 7 个 ASR 模型和 2 个分角色模型全部成功（来源：GitHub Release。调研环境访问不了 HuggingFace，所以 HF 路径没有实测，脚本会在 HF 失败后自动回退到 GitHub）。

| key | 磁盘体积（实际加载的文件，含 tokens） | 说明 |
|---|---|---|
| sensevoice | 228.5 MB | |
| zipformer-ctc-small-zh | 25.1 MB | |
| zipformer-zh-14m | 29.5 MB | int8 encoder/joiner + fp32 decoder |
| zipformer-bilingual-small | 57.4 MB | 同上。压缩包里还带有 `64/`、`96/` 两个变体子目录，脚本会优先使用顶层文件 |
| zipformer-ctc-multi-zh-hans | 66.3 MB | 本地量化得到（`onnxruntime.quantization.quantize_dynamic`，QInt8） |
| paraformer-zh-small | 78.1 MB | |
| dolphin-base | 99.4 MB | |
| pyannote-seg | 1.5 MB | 分角色：分段模型 |
| speaker-embedding | 37.8 MB | 分角色：3D-Speaker ERes2Net base（文档没有列体积，这是实测值） |

HF 仓库名：`csukuangfj/sherpa-onnx-paraformer-zh-small-2024-03-09`、`csukuangfj/sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23`、`csukuangfj/sherpa-onnx-zh-wenet-wenetspeech` 这三个能在官方文档里找到对应写法。其余仓库名按同一命名规则推断，**未核实**。如果 HF 下载失败，脚本会自动回退到 GitHub。

## 4. 实测结果

### 4.1 测试方法

- 测试集：`benchmark/sentences.tsv` 里的 20 句陪伴机器人场景语句，覆盖指令、提醒、时间、数字、电话、日期、长句、闲聊、情绪、人名、中英混说、计算等。由 `make_test_audio.py` 调用本地 TTS 生成，音频已经提交在 `benchmark/testset/`。
  - **melo 集**（主测试集）：MeloTTS zh_en，单音色，音质较好。
  - **aishell3 集**（鲁棒性测试）：VITS aishell3 的 6 个不同音色，音质明显较差，用来模拟口齿不清、发音偏差的情况。
- CER：按字计算编辑距离，去掉标点和空格。**做了数字归一化**，把 `9点`、`123000`、`30%` 等阿拉伯数字转成汉字读法再比较，避免 SenseVoice 的 ITN 输出（阿拉伯数字）被误判为错误。
- 流式模型按 100ms 一块送入音频，模拟实时输入；"首字出现"是第一个非空中间结果对应的音频时刻（包含音频开头 0.25s 的静音）。
- 完整逐句结果见 `benchmark/results/*.md`。

### 4.2 melo 集（20 句，71.4s，2 线程）

| 模型 | 体积 MB | 流式 | **CER** | RTF（2 线程） | RTF（1 线程） | 首字出现(s) | 输出阿拉伯数字 |
|---|---|---|---|---|---|---|---|
| Paraformer-zh-small | 78.1 | 否 | **4.99%** | 0.020 | 0.028 | — | 否 |
| SenseVoice-Small（基准） | 228.5 | 否 | 5.57% | 0.060 | 0.099 | — | **是** |
| Zipformer 流式中英双语 small | 57.4 | 是 | 13.20% | 0.048 | 0.056 | 0.86 | 否 |
| Dolphin base | 99.4 | 否 | 13.78% | 0.035 | 0.050 | — | 否 |
| Zipformer-CTC 流式 multi-zh-hans | 66.3 | 是 | 19.94% | 0.151 | 0.170 | 0.91 | 否 |
| Zipformer-CTC small 流式 | 25.1 | 是 | 21.70% | 0.038 | 0.044 | 0.83 | 否 |
| Zipformer 流式 zh-14M | 29.5 | 是 | 24.34% | 0.037 | 0.041 | 0.98 | 否 |

### 4.3 aishell3 集（音质差的多音色，2 线程）

| 模型 | CER |
|---|---|
| SenseVoice-Small（基准） | **12.32%** |
| Paraformer-zh-small | 13.20% |
| Zipformer 流式中英双语 small | 34.90% |
| Dolphin base | 36.66% |
| Zipformer-CTC 流式 multi-zh-hans | 42.52% |
| Zipformer-CTC small 流式 | 42.82% |
| Zipformer 流式 zh-14M | 51.03% |

在音质差的语音上，SenseVoice 和 Paraformer-small 基本持平，其余模型的 CER 都翻倍了。**鲁棒性方面，Paraformer-small 和 SenseVoice 同属第一档。**

### 4.4 真人录音定性对比（各模型官方压缩包自带的 test_wavs，没有标准答案）

| 录音 | 表现 |
|---|---|
| 四川话 / 天津话 / 河南话（paraformer 包自带） | Paraformer-small 和 SenseVoice 的输出通顺，而且两者高度一致（四川话、河南话两条几乎一字不差）；Dolphin 次之；流式小模型错误较多 |
| 粤语（sensevoice 包自带 yue.wav） | **只有 SenseVoice 和 Dolphin 正确**（Dolphin 输出繁体字），其余候选全部转写失败 |
| 中英混说（"yesterday was 星期一，today is Tuesday…"） | Paraformer-small 输出看起来完全正确；SenseVoice 基本正确；中文专用模型（CTC small、zh-14M、multi-zh-hans）会把英文输出成 `<unk>` 或音译汉字 |
| 普通话（sensevoice 包自带 zh.wav） | 多数模型输出"开放时间早上九点至下午五点"，SenseVoice 输出"开饭时间早上9点至下午5点"。这条录音没有标准答案，无法判断谁对 |

### 4.5 分角色实测

`benchmark/testset/dialog_2spk.wav`：用两个 aishell3 音色交替说 4 句，共 10.1 秒。

| 标准答案 | demo 输出 |
|---|---|
| 0.00–1.78 A | 0.05–1.50 说话人1 |
| 2.38–4.38 B | 2.39–4.25 说话人2 |
| 4.98–6.99 A | 5.06–6.92 说话人1 |
| 7.59–9.76 B | 7.61–9.53 说话人2 |

说话人数自动判断为 2，4 段全部分对。处理 10 秒音频的分离耗时约 0.66 秒。注意：这是**离线**分离，处理的是整段音频；常驻场景要做成按 VAD 段做声纹聚类的增量方案（见第 7 节）。

## 5. 调研中发现的坑（重要）

1. **Transducer 模型（zh-14M、中英双语 small）不能用 int8 decoder**：用 int8 decoder 时输出会被严重截断。例如"二零二六年十月二十九日我们开了一个会"只输出了"嗯开"。换成 fp32 decoder 后输出就正常了。官方 wasm demo 也是用 fp32 decoder 搭配 int8 encoder 和 int8 joiner。所以这两个模型的实际体积比"全 int8"多出 5–10MB（47.5→58MB、24.5→30MB）。
2. **multi-zh-hans 流式 CTC 的官方压缩包当前只有 fp32 文件**（252MB）。文档列出的 68MB int8 文件不在包里。下载脚本会用 onnxruntime 动态量化补上（66MB）。不过这个模型 RTF 偏高（0.15），可能和自行量化有关。
3. **数字格式**：SenseVoice（开启 use_itn）输出阿拉伯数字，比如"7点半"、"30%"、"13800138000"；其余 6 个模型都输出汉字数字。如果下游需要阿拉伯数字，非 SenseVoice 模型要另外加一个 ITN（sherpa-onnx 支持 `rule_fsts` 规则文件，或者用 WeTextProcessing）。
4. **"首字出现"约 0.8–1.0 秒**，其中包含 0.25s 的前置静音和模型的 chunk 延迟。流式 Zipformer 的 chunk 是 16 帧（约 0.32s）。如果需要更低延迟，要找 chunk 更小的导出版本。
5. **SenseVoice 和 Paraformer 都不支持流式**。常驻监听场景下要用 VAD 切段，等一句话说完再识别。按实测 RTF 0.02–0.06 计算，5 秒的句子在说完后 0.1–0.3 秒内就能出结果。

## 6. 音色定制（TTS）初步调研

TTS 不在本次 demo 范围内，只做了初步摸底。测试集就是用其中两个模型生成的。

| 模型 | 体积（出处） | 音色 | 定制方式 | 备注 |
|---|---|---|---|---|
| VITS aishell3 | 29 MB（tts/pretrained_models/vits：`model.onnx 29M`） | 174 个 | 按 sid 选择预置音色 | 实测音质偏差，ASR 识别它合成的语音时 CER 翻倍 |
| VITS MeloTTS zh_en | 163 MB（同上） | 1 个 | 只能微调 | 中英混读，音质好 |
| Matcha zh-baker | 声学模型 71–72M + 声码器 vocos 51M（tts/pretrained_models/matcha） | 1 个 | 可以微调 | |
| ZipVoice-distill zh-en | 未核实（文档里有下载命令，没列体积） | 任意 | **零样本声音克隆**（参考音频 + 参考文本） | sherpa-onnx 已经支持，最贴合"音色定制"需求，**建议下一步重点测试** |
| Kokoro 多语种 | 310–330 MB | 多个 | 预置音色 | 超出体积上限 |

## 7. 常驻监听架构建议（对应会议中的"本地小脑调度"）

```
麦克风 ──► Silero VAD（<1MB）──┬─► 流式 Zipformer 双语 small（58MB）── 实时字幕 / 打断检测 / 唤醒词前置
                              │
                              └─► 一句话结束 ─► Paraformer-zh-small（78MB）── 最终文本 ─► 本地缓存（带时间戳、说话人）
                                                  │
                                                  └─► 分角色：3D-Speaker 声纹（38MB）对每句话提取向量并做增量聚类
```

- 总体积约 175MB（各模型分别都 ≤150MB）。内存和算力消耗以目标芯片实测为准。
- 如果想精简成一个模型：只用 VAD + Paraformer-small（78MB）。代价是没有实时字幕，打断检测要等一句话说完才能判断。

## 8. 下一步

1. **用真人录音复核**：用 demo 页面录同事的真实说话（包括远场、带噪声、方言），导出 CSV；或者把录音放进一个目录，写好 `refs.tsv`，然后运行 `python benchmark.py --dir 目录`。
2. **补测 Zipformer 流式 zh large int8（160MB）**：它的公开 CER 是流式模型里最好的，只超出硬线 10MB。
3. **在目标芯片（机器人主控）上测 RTF 和内存**：官方文档给出 SenseVoice int8 在 RK3588 上 1 线程的 RTF：A76 核 0.099，A55 核 0.436。
4. 测试 ZipVoice 零样本声音克隆，作为音色定制方案。
5. 如果需要模型微调或训练资源，按会议约定提交申请。
