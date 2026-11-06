# BLOCKED

## Task 0 核对证据

- `test_audio/speakers/` 本仓库不存在（`ls test_audio/speakers` → exit=2）。基线报告数据用的数据在 `D:/projects/lospoloes-proj/test_audio/speakers`（enroll/ 9 人 + test/ 11 人，已确认存在）。评测工具需用该外部路径，或先跑 `tools/fetch_speaker_clips.py` 下载到本仓库 `test_audio/speakers/`。
- 跨句累积不做：speaker-roles 分支（S1/S2 分角色）未合入 live-demo-mac @ 8499d19（该基线 LiveSession 无 speakers 参数，无 tests/test_speaker.py、tools/eval_speaker.py）。本任务只做「单句 + 句内累积」版本；跨句累积待 speaker-roles 合入后再做。
- 基线关键数字核对：以 `benchmark/results/speaker_id.md` 记录为准（整条 24/24、3s 段 89%、EER 3.1% @ 0.41、TitaNet 3s 94% EER 1.5% @ 0.33）。原始数据在本机 D:/projects/lospoloes-proj，重放用 Task 1 的 eval 工具在本机跑，跑完把数字回填 benchmark/results/owner_eval.md 再对比；对不上先修理解，不许开工。

## 其余

无

## 真人验证集「自己」部分（2026-11-06）

- 陌生人侧已就位：AMI 会议语料 4 位真人 16 句（`data/realval/strangers/`，不提交，已进 .gitignore）。
- 「自己」卡在需要用户本人录音：`logs/live/` 全部会话经声纹+文字双重核实均为测试音频回放（095129/133259=dialog_2spk 回放 0.878、095207/095430=同一音频两遍、161941=s02_reminder 回放），无一为本人真实语音；麦克风链路已验证可用（probe.wav 8s → VAD → 512 维向量），但有效人声仅 0.9s，不足注册（需 4–6 句、每句 ≥5s）。
- 待办：用户运行 `PYTHONUTF8=1 .venv/Scripts/python.exe tools/enroll_owner.py --mic` 录 5 句（脚本会保存向量到 data/owner_profile.json，不存音频）；之后补录 3 句做主人测试集，跑 `python tools/eval_owner.py --data data/realval` 出真人验证集 FAR/召回，回填 owner_eval.md。
