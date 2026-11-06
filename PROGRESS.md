# PROGRESS — 声纹注册 · 主人识别

## 目标 / 顺序 / 最大风险（≤10 行）

- 目标：持续监听 ASR 里做声纹注册 + 主人识别，每条最终结果加 `owner` 字段（true/false/null），转写内容不动。
- 基线：live-demo-mac @ 8499d19，21 tests OK；模型齐（ERes2Net zh-cn 37.8MB 默认，TitaNet en 38.4MB 仅对比）。
- 基线参考数字（benchmark/results/speaker_id.md，数据在 D:/projects/lospoloes-proj/test_audio/speakers）：整条 24/24，3s 段 89%，EER 3.1% @ 阈值 0.41。
- 顺序：Task 0 核对 → Task 1 eval 工具 → Task 2 注册+判定模块 → Task 3 接入持续监听 → Task 4 模型对比 → Task 5 调参+报告。
- 最大风险：整条/EER 复现依赖本机 D:/projects/lospoloes-proj/test_audio/speakers 数据（本仓库 test_audio/speakers/ 不存在，见 BLOCKED.md）；跨句累积（speaker-roles 未合入 live-demo-mac）不做，写 BLOCKED.md。
- 铁律：判卷文件不碰；不新增依赖；声纹/注册音频不提交（进 .gitignore）；不推送；每项要有真实命令输出 + 反向验证红→绿。

## Task 3（2026-11-06 @ 072aa8d + 947ee25）✅

- `LiveSession` 新参数 `owner_profile` / `owner_model` / `owner_threshold`（给了注册不给模型 → ValueError）；`text` 事件新增 `owner`（true/false/null），`text` 不改；server 检测 `data/owner_profile.json`（不提交）+ ERes2Net 就绪时自动启用。
- 判定音频只取每句前 2.5 秒：整句 10s 时 P90≈206ms 超标，截 2.5s 后 **P50 46.8ms / P90 48.7ms / max 55.3ms**（验收 ≤50/≤100ms）。注册向量用全长（离线注册不赶时间；2.5s 注册单句不稳，score 0.397 险掉线）。
- 端到端（真实 ERes2Net + 外部数据）：deniro 注册 → deniro 测试句 score 0.80/0.76=owner、ad074fb3 截 2.5s 0.452=owner；halle 0.09=stranger；21/21 判对。s07_long（TTS 声）=stranger ✅。
- 反向验证：注册向量打乱后对自己 score −0.02 → stranger ✅。
- 页面：正式结果旁 主人/陌生人/未知 徽标（pill 样式复用）；docs/live-demo.md 补字段规则。
- 测试：32 通过 skip 0（test_owner.py 11 个，其中 TestLiveOwner 4 个喂真语音 s07_long 拼接 + mock embed）。

## Task 1 补充：真实主人验证集（2026-11-06）

- 主人 = deniro（1 句全长注册，`data/owner_profile.json` 不提交）；陌生人 = daniel/emma/freeman/halle/jensen/jlaw/leijun/musk/tom/weaver 共 10 人 21 句（未用于调参，阈值 0.41 沿用 Task 0 EER 曲线）。
- 结果：**FAR 0/21 = 0.0%**（验收 ≤1%）✅；**≥3s 主人召回 3/3 = 100%**（验收 ≥90%）✅。陌生人最高分 daniel 0.363 < 0.41，余量 0.047。

## Task 4：模型对比（2026-11-06）

- 同集合（deniro 注册 + 21 陌生人句）双模型对比：**eres2net 召回 3/3、FAR 0/21；titanet 召回 3/3、FAR 0/21**——不换（切换条件是 EER 低 ≥1pp，两者都满分开无从谈起）。
- 分离度（注册全长 vs 测试截 2.5s）：eres2net 同人 min 0.302 / 跨人 max 0.512（余量 −0.21，有重叠但 FAR 仍 0）；titanet 同人 min 0.444 / 跨人 max 0.390（余量 +0.054 更稳）。
- 结论：默认保留 eres2net（zh-cn 场景契合 + 基线数字齐）；titanet 记录在案，若后续中文环境测试需换模型再启用。

## Task 5：调参 + 报告（2026-11-06 @ 4004cdb）✅

- 调参：阈值 0.41 来自 Task 0 EER 曲线（owner_eval.md：EER 2.9% @ 0.40），验证集未调参已 FAR 0% / 召回 100%——无调参空间，不调。
- 报告：`docs/owner-id.md`（用法/规则/验证数字/隐私/未做）+ README『接进持续监听』一节。
- 未做（已写进 docs）：防录音回放攻击、远场/噪声、跨句累积（BLOCKED.md）。

## 全部完成（2026-11-06）

- 提交：721943c 之前 Task 0–2；072aa8d + 947ee25 Task 3；06f30c1 Task 1 补充；721943c Task 4；4004cdb Task 5。
- 测试：32 通过 skip 0。声纹数据（注册音频、data/owner_profile.json）未提交。

## 反馈修正 1：双阈值（2026-11-06）

- `speaker_owner` 双阈值常量：对话档 `DIALOG_THRESHOLD=0.45`（曲线 FAR 1.4%/FRR 0%，主人少被拒）、特权档 `PRIVILEGED_THRESHOLD=0.55`（FAR 0.3%/FRR 12.1%，敏感操作宁拒），均由 owner_eval.md 开发集曲线定，验证集未调参。`decide(threshold=...)` 按档位传值。
- 验证集（deniro 注册，未调参）：对话档 召回 3/3、FAR 0/21；特权档 召回 1/3、FAR 0/21（特权档以召回换严格，符合设计）。

## 反馈修正 2：句内分角色段累积（2026-11-06）

- `LiveSession` 新参数 `owner_diarizer`（asr_engine.Diarizer 实例）：正式结果先跑句内 Diarizer 切说话人段，每段（≥0.5s）提向量，取与主人声纹最像的主段；主段向量按秒数加权攒进 `owner_acc`（跨句累积），攒够 `MIN_SEC=3` 秒判一次后清零。
- server：注册文件 + Diarizer 模型都就绪才启用（缺分角色模型打印提示不启用）。
- 端到端（真 ERes2Net + 真 Diarizer）：deniro 测试句 9fb04fdc → [None, True]（两句累积后判 owner）、ad074fb3 → [True]；s07_long（TTS 声）→ False ✅。
- 耗时：Diariizer 整句 ~1000ms 在转写线程后台跑，不占 ASR 解码（text 事件 decode_ms 不变）；判定在 Diarizer 完成后同步做（段 embed + decide <60ms）。
- 测试：34 通过 skip 0（新增 test_profile_without_diarizer_raises、test_two_seg_picks_main_speaker——两段各半时 embed 只收主段音频）。

## 反馈修正 3：文档同步（2026-11-06）

- docs/owner-id.md：规则改为累积判定 + 双阈值取值与依据；验证数字补特权档与 --shuffle FAR 52.4%。
- docs/live-demo.md：owner 字段规则同步（累积、双阈值、前置 Diarizer 模型）。
- 任务 1 反向验证补全：`--shuffle` FAR 52.4% 已贴进 owner_eval.md（87aee55）。
