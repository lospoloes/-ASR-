# 持续监听场景：真人双人对话（MagicData-RAMC）

- 数据：MagicData-RAMC dev 集，两人面对面自由聊天、手机录制（16kHz 单声道），人工转写到句，带起止时间和说话人。`python tools/fetch_ramc.py` 下载并截取；许可证 CC BY-NC-ND 4.0，音频和文本只在本机，不提交
- 音频以 0.1s 为单位流式喂给 `live_engine.LiveSession`（与网页 /live 同一套逻辑），VAD 实时切 chunk，两个模型拿到完全相同的 chunk
- **真实延迟** = 人工标注的句子结束时刻 → 该模型出字（= VAD 判停等待 + 本模型解码，按单独部署计）；只统计 VAD 正好在这句话结尾切开的句子（括号里是句数），和下一句挨得太近、被并进同一个 chunk 的不计
- **一段含两人**：一个 chunk 里同时有两个说话人的话（交给大模型时分不清是谁说的）
- CER：字级编辑距离，去标点空格，数字归一化
- 最长 chunk 15.0s；机器：Darwin arm64（Apple M4 Pro），每个模型 2 线程

## 静音阈值 0.3s

| 数据 | chunk 数 | 平均/最长 chunk | 一句被切成多段 | 一段含多句 | 一段含两人 | SenseVoice-Small int8（基准） CER | SenseVoice-Small int8（基准） 真实延迟（平均/P90） | Paraformer-zh-small int8 CER | Paraformer-zh-small int8 真实延迟（平均/P90） |
|---|---|---|---|---|---|---|---|---|---|
| 真人双人对话 CTS-CN-F2F-2019-11-15-1434_13-190 | 43 | 3.5s / 11.6s | 6 | 10 | 9 | 6.36% | 412 / 625 ms（36 句） | 5.41% | 366 / 545 ms（36 句） |

合计：SenseVoice-Small int8（基准） CER **6.36%**、真实延迟平均 **412 ms**；Paraformer-zh-small int8 CER **5.41%**、真实延迟平均 **366 ms**

## 静音阈值 0.5s

| 数据 | chunk 数 | 平均/最长 chunk | 一句被切成多段 | 一段含多句 | 一段含两人 | SenseVoice-Small int8（基准） CER | SenseVoice-Small int8（基准） 真实延迟（平均/P90） | Paraformer-zh-small int8 CER | Paraformer-zh-small int8 真实延迟（平均/P90） |
|---|---|---|---|---|---|---|---|---|---|
| 真人双人对话 CTS-CN-F2F-2019-11-15-1434_13-190 | 28 | 5.6s / 16.2s | 0 | 11 | 7 | 5.82% | 573 / 675 ms（28 句） | 5.41% | 509 / 648 ms（28 句） |

合计：SenseVoice-Small int8（基准） CER **5.82%**、真实延迟平均 **573 ms**；Paraformer-zh-small int8 CER **5.41%**、真实延迟平均 **509 ms**

## 静音阈值 0.8s

| 数据 | chunk 数 | 平均/最长 chunk | 一句被切成多段 | 一段含多句 | 一段含两人 | SenseVoice-Small int8（基准） CER | SenseVoice-Small int8（基准） 真实延迟（平均/P90） | Paraformer-zh-small int8 CER | Paraformer-zh-small int8 真实延迟（平均/P90） |
|---|---|---|---|---|---|---|---|---|---|
| 真人双人对话 CTS-CN-F2F-2019-11-15-1434_13-190 | 17 | 9.7s / 18.5s | 2 | 12 | 8 | 5.68% | 651 / 917 ms（14 句） | 5.82% | 560 / 862 ms（14 句） |

合计：SenseVoice-Small int8（基准） CER **5.68%**、真实延迟平均 **651 ms**；Paraformer-zh-small int8 CER **5.82%**、真实延迟平均 **560 ms**

