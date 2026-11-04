# 持续监听场景：SenseVoice vs Paraformer

- 音频以 0.1s 为单位流式喂给 `live_engine.LiveSession`（与网页 /live 同一套逻辑），VAD 实时切 chunk，两个模型拿到完全相同的 chunk
- **真实延迟** = 用户实际说完 → 该模型出字（= VAD 判停等待 + 本模型解码，按单独部署计）；只在句尾精确已知的合成对话（已裁掉 TTS 首尾静音）上统计
- CER：字级编辑距离，去标点空格，数字归一化
- 最长 chunk 15.0s

## 静音阈值 0.3s

| 数据 | chunk 数 | 平均/最长 chunk | 一句被切成多段 | 一段含多句 | SenseVoice-Small int8（基准） CER | SenseVoice-Small int8（基准） 真实延迟（平均/P90） | Paraformer-zh-small int8 CER | Paraformer-zh-small int8 真实延迟（平均/P90） |
|---|---|---|---|---|---|---|---|---|
| 合成对话 session1.wav | 13 | 1.5s / 3.8s | 4 | 0 | 4.08% | 586 / 819 ms | 4.08% | 499 / 693 ms |
| 合成对话 session2.wav | 12 | 1.1s / 2.2s | 3 | 0 | 4.55% | 560 / 651 ms | 9.09% | 494 / 582 ms |
| 合成对话 session3.wav | 10 | 1.6s / 3.2s | 2 | 0 | 0.00% | 507 / 590 ms | 0.00% | 432 / 514 ms |
| 双人交替对话 dialog_2spk | 4 | 1.8s / 2.0s | 0 | 0 | 34.21% | — | 44.74% | — |
| 雷军演讲 272s（真人） | 65 | 2.5s / 8.8s | — | — | 1.33% | — | 1.99% | — |

合计：SenseVoice-Small int8（基准） CER **3.17%**、真实延迟平均 **551 ms**；Paraformer-zh-small int8 CER **4.41%**、真实延迟平均 **475 ms**

## 静音阈值 0.5s

| 数据 | chunk 数 | 平均/最长 chunk | 一句被切成多段 | 一段含多句 | SenseVoice-Small int8（基准） CER | SenseVoice-Small int8（基准） 真实延迟（平均/P90） | Paraformer-zh-small int8 CER | Paraformer-zh-small int8 真实延迟（平均/P90） |
|---|---|---|---|---|---|---|---|---|
| 合成对话 session1.wav | 9 | 2.3s / 3.9s | 1 | 0 | 4.08% | 794 / 1039 ms | 4.08% | 690 / 896 ms |
| 合成对话 session2.wav | 9 | 1.6s / 5.4s | 2 | 1 | 4.55% | 746 / 957 ms | 7.58% | 649 / 817 ms |
| 合成对话 session3.wav | 7 | 2.6s / 5.4s | 1 | 2 | 1.25% | 716 / 791 ms | 1.25% | 597 / 639 ms |
| 双人交替对话 dialog_2spk | 4 | 1.8s / 2.0s | 0 | 0 | 34.21% | — | 44.74% | — |
| 雷军演讲 272s（真人） | 53 | 3.1s / 10.6s | — | — | 1.16% | — | 1.50% | — |

合计：SenseVoice-Small int8（基准） CER **3.17%**、真实延迟平均 **756 ms**；Paraformer-zh-small int8 CER **4.07%**、真实延迟平均 **650 ms**

## 静音阈值 0.8s

| 数据 | chunk 数 | 平均/最长 chunk | 一句被切成多段 | 一段含多句 | SenseVoice-Small int8（基准） CER | SenseVoice-Small int8（基准） 真实延迟（平均/P90） | Paraformer-zh-small int8 CER | Paraformer-zh-small int8 真实延迟（平均/P90） |
|---|---|---|---|---|---|---|---|---|
| 合成对话 session1.wav | 6 | 3.8s / 7.9s | 1 | 2 | 4.08% | 1063 / 1401 ms | 3.06% | 923 / 1155 ms |
| 合成对话 session2.wav | 4 | 4.3s / 7.8s | 0 | 2 | 4.55% | 1102 / 1480 ms | 4.55% | 926 / 1201 ms |
| 合成对话 session3.wav | 6 | 3.1s / 6.1s | 1 | 2 | 1.25% | 988 / 1166 ms | 2.50% | 843 / 942 ms |
| 双人交替对话 dialog_2spk | 4 | 1.9s / 2.7s | 0 | 0 | 36.84% | — | 39.47% | — |
| 雷军演讲 272s（真人） | 39 | 4.4s / 17.5s | — | — | 1.33% | — | 1.33% | — |

合计：SenseVoice-Small int8（基准） CER **3.39%**、真实延迟平均 **1047 ms**；Paraformer-zh-small int8 CER **3.51%**、真实延迟平均 **895 ms**

