# 声纹注册 · 主人识别

持续监听 ASR 的可选能力：注册主人声纹后，每条正式转写结果带 `owner` 字段（true 主人 / false 陌生人 / null 判不了），转写内容不动。

## 用法

```bash
# 1. 注册（音频任意格式，建议 2 句以上、每句 ≥5 秒；不提交）
python tools/enroll_owner.py --name 老板 --files a.wav b.wav

# 2. 重启 server：检测到 data/owner_profile.json + ERes2Net 模型就自动启用
# 3. 页面正式结果旁显示 主人 / 陌生人 / 未知 徽标；不注册就是普通模式，无 owner 字段
```

## 规则

- **<3 秒句一律 null**（不认人，防止短句误判；`MIN_SEC=3.0`）；
- 判定音频只取每句**前 2.5 秒**（长句提速：10s 句 P90≈206ms 超标，2.5s 后 P90≈49ms，验收 ≤50/≤100ms）；
- 注册向量用**全长**音频（离线注册不赶时间；截 2.5s 注册单句不稳）；
- 阈值 0.41 来自 EER 曲线（benchmark/results/owner_eval.md：EER 2.9% @ 0.40），**未在验证集上调参**。

## 验证（真实数据，deniro 注册）

- FAR 0/21 = 0%（陌生人 10 人 21 句，验收 ≤1%）；
- ≥3s 主人召回 3/3 = 100%（验收 ≥90%）；
- 反向：注册向量打乱后对自己 score −0.02 → stranger；
- 引擎对比：ERes2Net 与 TitaNet 同集合都召回 3/3、FAR 0/21，保留 ERes2Net（zh-cn 契合）。

## 隐私

注册音频与 `data/owner_profile.json` 属生物信息：**不提交、不推送**（已在 `.gitignore`）；文档只放统计数字。

## 未做

- 防伪造（录音回放攻击）未做——陌生人高分句 daniel 0.363 距阈值 0.41 余量 0.047，回放攻击不受此保护；
- 远场/噪声场景未测（数据全为近讲）；
- 跨句累积（多句联合判定）不做（speaker-roles 未合入本仓库，见 BLOCKED.md）。
