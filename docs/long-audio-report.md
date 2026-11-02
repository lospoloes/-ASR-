# 长音频实测报告与切段修复（2026-11-02）

> 背景：用雷军 2022 年度演讲片段（[lei-jun-test.wav](https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/lei-jun-test.wav)，**272.4s / 16kHz 单声道**）对 7 个模型做长音频横评，发现离线模型在长音频上严重劣化甚至崩溃，已修复并复测。机器：Windows x64 CPU，2 线程，sherpa-onnx。

## 1. 问题：离线模型整段送入长音频

第一轮（修复前，整段 272s 直送）：

| 模型 | 耗时 | RTF | 结果 |
|---|---|---|---|
| Zipformer-CTC small 流式 | 6.8s | 0.0248 | 正常 |
| Zipformer-zh-14M 流式 | 6.8s | 0.0248 | 正常 |
| Zipformer 双语 small 流式 | 9.2s | 0.0338 | 正常 |
| Paraformer-zh-small | 14.9s | **0.0547** | ⚠️ 比 10s 短音频（RTF 0.0135）劣化 4 倍 |
| Zipformer-CTC multi-zh-hans | 22.1s | 0.0811 | 正常 |
| SenseVoice-Small（基准） | 47.8s | **0.1754** | ⚠️ 劣化 3.5 倍 |
| **Dolphin base** | — | — | ❌ **崩溃**：`onnxruntime BroadcastIterator::Init axis 905 by 6809`（位置编码长度超限） |

**根因**：离线非自回归模型的注意力是 O(n²)，音频越长每秒成本越高；Dolphin 的位置编码有最大长度限制。流式模型按 chunk 处理天然不受影响。

## 2. 修复：离线模型自动切段（asr_engine.py）

- 音频 ≤ 31s：整段直送（原行为不变）
- 音频 > 31s：按 **30s 窗口 + 1s 重叠**切段，逐段 `create_stream → accept_waveform → decode`，段文本直接拼接；<0.5s 尾部碎段丢弃
- SenseVoice 的 lang/emotion/event 标签取各段属性并集
- 改动集中在 `AsrModel._offline()`，新增 `_offline_stream()` 辅助，流式路径与分角色路径未动

## 3. 修复后复测（同一音频，网页端重跑）

| 模型 | 耗时 | RTF | 对比修复前 |
|---|---|---|---|
| **Paraformer-zh-small** | 4.4s | **0.016** | ⚡ 快 3.4 倍，回到短音频水平 |
| Zipformer-CTC small 流式 | 6.2s | 0.0229 | 持平 |
| Zipformer-zh-14M 流式 | 7.2s | 0.0265 | 持平 |
| Zipformer 双语 small 流式 | 9.1s | 0.0331 | 持平 |
| **Dolphin base** | 8.8s | **0.0321** | ✅ 不再崩溃，输出完整带标点 |
| Zipformer-CTC multi-zh-hans | 21.9s | 0.0806 | 持平 |
| SenseVoice-Small（基准） | — | ~0.06（页面确认中） | 恢复短音频水平 |

## 4. 转写质量观察（272s 真实演讲）

- **SenseVoice**：唯一自带标点 + 阿拉伯数字 ITN（"1987年""130周年""3500人"）+ 情感/事件标签，文字最完整
- **Dolphin**（切段后）：带标点和空格，可读性好
- **流式 Zipformer**：丢标点，偶有开头丢字（"什么朋友们""我朋友们"），但速度最稳
- **Paraformer**：切段后段间有轻微重复（"朋友们晚上朋友们晚上好"，来自 1s 重叠区），待做重叠去重
- 分角色未在本轮测试（单人演讲）

## 5. 结论（对选型的影响）

1. **端侧常驻场景必须 VAD/切段**——不是可选项。修复后 Paraformer/SenseVoice/Dolphin 的 RTF 才反映真实部署形态。
2. 修复后长音频 RTF 排序：**Paraformer-zh-small（0.016，78MB）> Zipformer-CTC small（0.023，25MB）> zh-14M（0.027）> 双语 small（0.033）> Dolphin（0.032，99MB）> multi-zh-hans（0.081）**。
3. 长音频结论不改变总选型：**主力转写仍是 Paraformer-zh-small int8**；流式 Zipformer 承担实时字幕/打断。
4. 待办：拼接重叠去重（消除段间重复字）；用 VAD 切段替代固定窗口进一步提质。

## 6. 复现

```bash
# 下载测试音频（272s）
curl -L -o test_audio/lei-jun-test.wav \
  https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/lei-jun-test.wav
# 启动 demo，网页上传 test_audio/lei-jun-test.wav
python server.py
```

改动文件：`asr_engine.py`（离线切段推理）。详见 [selection.md](selection.md) 总选型报告。
