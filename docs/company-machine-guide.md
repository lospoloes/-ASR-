# 公司电脑（Windows）操作手册

> 本文档是给公司电脑上的操作者（人或 Claude Code）看的，按顺序执行即可。
> 当前选定模型：**SenseVoice-Small int8 + Paraformer-zh-small int8**。其余候选已从项目中移除，不用再管。

## 0. 更新代码

```bat
cd C:\work\asr-demo
git pull origin main
```

如果之前的 `.venv` 是用 Python 3.14 建的，先删掉重建：onnxruntime 在 3.14 下 `import` 时会直接崩溃（int8 量化那一步）。

```bat
rmdir /s /q .venv
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

没装 3.12 的话用 `py -3.11`。可以用 `py -0` 查看本机装了哪些 Python 版本。

## 1. 下载模型

```bat
.venv\Scripts\python.exe download_models.py
```

默认只下载 SenseVoice、Paraformer、Silero VAD、分角色模型（pyannote + 3D-Speaker），约 350MB。已下载的会自动跳过。

可选：
```bat
.venv\Scripts\python.exe download_models.py --models speaker-embedding-en --no-diarization
```
```bat
.venv\Scripts\python.exe download_models.py --models tts-melo-zh-en,punct-ct-transformer --no-diarization
```
第一条是英文声纹模型，用于名人声纹实验的对照；第二条是 TTS 和标点模型，只在重新生成测试集或试标点模型时才需要。

## 2. 单元测试

```bat
.venv\Scripts\python.exe -m unittest discover -s tests
```

应全部通过（8 条）。

## 3. 网页 demo（语义切段）

```bat
.venv\Scripts\python.exe server.py
```

浏览器打开 http://127.0.0.1:8000 ：
- 「切段」默认是「语义」，结果下方会逐句列出断句依据（完整句 / 未说完 / 句首承接）和等待时长；
- 可以切到「VAD」「固定窗口」做对比；
- 可以试着说"我想……（停一下）……明天去公园"，看会不会被切成两句。

批量评测（结果写到 `benchmark/results/semantic.md`）：
```bat
.venv\Scripts\python.exe benchmark_semantic.py
```

## 3.5 持续监听 demo（双模型并行实时转写）

这次更新新增了依赖 `websockets`，先装一下：

```bat
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe server.py
```

浏览器打开 http://127.0.0.1:8000/live ：
- 点「开始监听」，两个人正常聊天，每说完一句，表格里就多一行（左 SenseVoice、右 Paraformer），底部统计各自的延迟；
- 没人陪聊时点「用音频文件模拟」，选 `benchmark\semantic\session1.wav` 等文件；
- 可以对比切段静音阈值 0.3 / 0.5 / 0.8 秒的效果；
- 「导出转写」「导出事件 JSON」可以保存结果。

**请录一段两个人的真实聊天**（3–5 分钟，正常语速，可以有插话），把导出的 JSON 和转写放进 `benchmark/results/`，并在 `docs/live-demo.md` 第 3 节补一段"真人对话实测"（两个模型各自的明显错误、延迟统计）。录音本身不要提交。

批量评测（结果写到 `benchmark/results/live.md`）：
```bat
.venv\Scripts\python.exe live_sim.py
```

## 4. 名人声纹实验（马斯克、黄仁勋）

### 4.1 安装工具

```bat
winget install ffmpeg
.venv\Scripts\python.exe -m pip install yt-dlp
```

装完 ffmpeg 后**重开一个终端**，再运行 `ffmpeg -version` 确认能找到。

### 4.2 填清单

打开 `benchmark/speakers/clips.tsv`（Tab 分隔，可以用 VS Code 编辑），把最上面 `<填链接>` 的 4 行填好：

| 人名 | 用途 | 链接 | 开始 | 结束 |
|---|---|---|---|---|
| musk | enroll | 马斯克视频 A | 1:05 | 1:15 |
| musk | test | 马斯克视频 B（**不能和 A 是同一个视频**） | 3:20 | 3:40 |
| jensen | enroll | 黄仁勋视频 C | … | … |
| jensen | test | 黄仁勋视频 D（可以选他讲中文的片段，测跨语种） | … | … |

选片段的要求：
- **只有他一个人在说话**：不要有主持人插话，也不要有背景音乐、掌声；
- 注册片段约 10 秒，测试片段 10–30 秒；
- 每人可以多加几行 test（来自不同视频），结果更可靠；
- 想再加中文名人（比如罗永浩、董宇辉），照同样格式加行即可。人名用英文或拼音，它会作为目录名。

清单里其余行（6 位好莱坞演员、雷军、陌生人）是已验证能下载的公开音频，保留即可。

### 4.3 下载并截取

```bat
.venv\Scripts\python.exe tools\fetch_speaker_clips.py
```

- 音频会放进 `test_audio\speakers\enroll|test\<人名>\`，来源自动记录到 `test_audio\speakers\sources.tsv`；
- 某行失败不影响其他行，失败原因会打印出来；
- 如果 YouTube 报"Sign in to confirm you're not a bot"，加上浏览器登录态重试：
  ```bat
  .venv\Scripts\python.exe tools\fetch_speaker_clips.py --cookies-from-browser edge
  ```
- 同一个人的注册和测试用了同一个链接时，会打印 ⚠ 警告（雷军那两行是已知情况，可以忽略）。

### 4.4 运行实验

```bat
.venv\Scripts\python.exe speaker_id.py
```

结果写到 `benchmark/results/speaker_id.md`，内容包括：
1. 每条测试音频认成了谁、和每个人的相似度；陌生人应显示"陌生人"；
2. 切成 1/2/3/5 秒小段的准确率；
3. 推荐阈值（EER）；
4. 拼接场景：所有人约 10 秒依次拼接，先分角色再认人，按时长统计准确率。

**重点看**：马斯克、黄仁勋两人的整条识别是否正确、相似度多少（中文声纹模型对英文人声的表现）；黄仁勋讲中文的片段能否认出（跨语种）。

### 4.5 提交

- **只提交结果文件和清单**：`benchmark/results/speaker_id.md`、`benchmark/speakers/clips.tsv`；
- **不要提交音频**：`test_audio/` 已在 .gitignore 里，名人音频涉及版权和声纹隐私；
- 在 `docs/speaker-id.md` 第 2 节表格下补一段「马斯克 / 黄仁勋 结果」，然后：

```bat
git add benchmark/results/speaker_id.md benchmark/speakers/clips.tsv docs/speaker-id.md
git commit -m "Speaker ID: add Musk and Jensen Huang results"
git push origin main
```

## 5. 常见问题

| 现象 | 处理 |
|---|---|
| `import onnxruntime` 时进程直接退出、没有报错 | Python 3.14 的问题，按第 0 节用 3.12 / 3.11 重建 .venv |
| `ffmpeg 不是内部或外部命令` | 装完 ffmpeg 要重开终端；还不行就把 ffmpeg 的 bin 目录加进 PATH |
| yt-dlp 报 403 / 需要登录 | 加 `--cookies-from-browser edge`（或 chrome）；还不行就先 `.venv\Scripts\python.exe -m pip install -U yt-dlp` 升级 |
| B 站链接下载失败 | 同上；也可以先手动下载视频，把清单里的链接换成本地文件路径 |
| 识别结果全是"陌生人" | 检查 enroll 目录是否有对应人名的音频；或用 `--threshold 0.4` 调低阈值重跑 |
| 持续监听页提示"无法连接后端 WebSocket" | 没装 `websockets`：`.venv\Scripts\python.exe -m pip install websockets` 后重启 server.py |
| 中文路径报错 | 项目放在不含中文、空格的路径下（如 `C:\work\asr-demo`） |
