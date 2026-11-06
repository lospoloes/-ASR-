# PROGRESS — 声纹注册 · 主人识别

## 目标 / 顺序 / 最大风险（≤10 行）

- 目标：持续监听 ASR 里做声纹注册 + 主人识别，每条最终结果加 `owner` 字段（true/false/null），转写内容不动。
- 基线：live-demo-mac @ 8499d19，21 tests OK；模型齐（ERes2Net zh-cn 37.8MB 默认，TitaNet en 38.4MB 仅对比）。
- 基线参考数字（benchmark/results/speaker_id.md，数据在 D:/projects/lospoloes-proj/test_audio/speakers）：整条 24/24，3s 段 89%，EER 3.1% @ 阈值 0.41。
- 顺序：Task 0 核对 → Task 1 eval 工具 → Task 2 注册+判定模块 → Task 3 接入持续监听 → Task 4 模型对比 → Task 5 调参+报告。
- 最大风险：整条/EER 复现依赖本机 D:/projects/lospoloes-proj/test_audio/speakers 数据（本仓库 test_audio/speakers/ 不存在，见 BLOCKED.md）；跨句累积（speaker-roles 未合入 live-demo-mac）不做，写 BLOCKED.md。
- 铁律：判卷文件不碰；不新增依赖；声纹/注册音频不提交（进 .gitignore）；不推送；每项要有真实命令输出 + 反向验证红→绿。
