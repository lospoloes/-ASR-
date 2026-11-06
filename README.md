# 端侧语音转写（ASR）模型对比 Demo

用于陪伴机器人端侧语音转写。项目用 sherpa-onnx 在本地 CPU 上加载已选定的两个模型：**SenseVoice-Small int8** 和 **Paraformer-zh-small int8**。网页上按住空格说一句话，两个模型转写同一段音频，同一个页面里能看到结果、耗时、RTF、CER，以及语义切段和分角色的结果。

- 选型调研、实测数据和推荐结论：**[docs/selection.md](docs/selection.md)**
- 长音频实测、VAD 切段与音乐段幻听：**[docs/long-audio-report.md](docs/long-audio-report.md)**
- 支持分角色（说话人分离）、流式模型的逐步输出、批量 benchmark，以及用 TTS 生成带标准答案的测试集

## 快速开始

需要 Python 3.10–3.13（**不要用 3.14**：onnxruntime 在 3.14 下会直接崩溃），以及 2GB 左右的磁盘空间。**项目路径里不要有中文或空格**，否则 onnxruntime 在 Windows 上可能读不到模型。

### Windows（公司电脑）

公司电脑的完整操作步骤（含名人声纹实验）见 **[docs/company-machine-guide.md](docs/company-machine-guide.md)**。

最省事的方法是双击 `run_windows.bat`。它会依次完成建虚拟环境、装依赖、下载模型、启动服务。

手动执行的步骤如下（cmd 或 PowerShell）：

```bat
git clone <本仓库地址> C:\work\asr-demo
cd C:\work\asr-demo
py -3 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python download_models.py
python server.py
```

如果 PowerShell 提示无法执行脚本，可以不激活虚拟环境，直接用 `.venv\Scripts\python.exe download_models.py` 和 `.venv\Scripts\python.exe server.py`。

### macOS / Linux

可以直接运行 `./run_mac.sh`，也可以手动执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python download_models.py
python server.py
```

### 打开页面

`server.py` 启动后会自动打开 **http://127.0.0.1:8000**。第一次录音时浏览器会请求麦克风权限，点允许即可。

> 必须用 `127.0.0.1` 或 `localhost` 访问。用局域网 IP 访问时，浏览器会禁止页面使用麦克风；这种情况下仍然可以用"上传 WAV"功能。

## 持续监听（双模型并行实时转写）

打开 **http://127.0.0.1:8000/live**，点「开始监听」后不用按键，直接聊天：VAD 自动把每句话切成 chunk，同一个 chunk 分别交给 SenseVoice 和 Paraformer 转写，文字左右两栏持续出现，并统计各自的延迟。一句话还没说完就实时出字（「实时出字」默认每 0.5 秒，可选关 / 0.25 / 0.4 / 0.5 / 0.6 / 1 秒）：浅灰是正在说的这一段，深灰是句内停顿处已经锁定的段，说完后整句重新识别一遍，换成黑色的正式结果。没人陪聊时可以用「用音频文件模拟」。每次监听的转写事件和录音会存到 `logs/live/`（只在本机，不提交；`--no-save-audio` 不存录音）。

```bash
python live_sim.py          # 持续聊天场景下分别评测两个模型（CER / 真实延迟 / 不同静音阈值），输出 benchmark/results/live.md
python tools/fetch_ramc.py  # 下载一段真人双人自由聊天（MagicData-RAMC，约 60MB，截 3 分钟，带人工转写；不提交）
python live_sim.py --real   # 在这段真人对话上评测，输出 benchmark/results/live_real.md
python tools/fetch_ramc.py --count 3   # 共下 3 段整段对话（94 分钟，约 180MB；不提交）
python tools/eval_vad.py    # 在这 3 段的人工标注上比较 VAD 参数，输出 benchmark/results/vad_eval.md
python tools/replay_live.py test_audio/ramc/CTS-CN-F2F-2019-11-15-1434_13-190.wav   # 按真实速度回放给 /ws/live（要先起 server.py），
                            # 对比不同的实时出字间隔（默认不开 / 0.4 / 1 秒），输出 benchmark/results/live_realtime.md
```

方案、评测结果和结论见 [docs/live-demo.md](docs/live-demo.md)；给下游（Voice Agent OS 的 V4）的事件接口见其中第 5 节。

## 下载模型

```bash
python download_models.py                 # 默认：SenseVoice + Paraformer + VAD + 分角色模型，约 350MB
python download_models.py --list          # 查看全部可选模型
python download_models.py --hf-mirror     # 国内网络推荐：走 hf-mirror.com
python download_models.py --models sensevoice,paraformer-zh-small   # 只下指定模型
python download_models.py --tts           # 额外下载 TTS（生成测试音频用，约 200MB）
```

- 每个模型优先从 **HuggingFace** 下载，失败后自动回退到 **GitHub Release**。已下载完整的模型会跳过。
- 如果两个来源都下载不了：加 `--source github --gh-proxy <GitHub加速前缀>` 再试；或者手动把官方压缩包解压到 `models/<key>/`。demo 会自动跳过缺失的模型。
- 模型文件保存在 `models/` 目录，已加入 `.gitignore`，不会被提交。

## 页面功能

| 区块 | 内容 |
|---|---|
| 录音区 | 按住 **空格**（或按住圆形按钮）说话，松开后开始识别。可以填参考文本来计算 CER，可选"数字归一化"（让 `9点` 和 `九点` 视为相同）。支持上传 WAV、回放录音、用上一段录音重跑、保存录音 |
| 结果卡 | 每个模型一张：转写文本（填了参考文本时，用红色和黄色标出错字）、耗时、RTF、CER；流式模型还会显示首字出现时间、每 100ms 块的计算耗时和逐步输出；SenseVoice 额外显示语种、情感和事件标签 |
| 分角色 | 勾选后，会对整段音频做说话人分离，然后逐段转写，显示"说话人1 / 说话人2…" |
| 需求对照表 | 对每个模型逐条列出体积（≤150 / <100MB）、流式、精度（本次会话的平均 CER，与 SenseVoice 基准比较）、分角色、音色定制的满足状态 |
| 参数卡 | 本机实测体积和文档体积（带出处）、流式还是离线、语言、数字输出格式、公开 CER（带测试集和出处）、许可证、本机平均 RTF |
| 会话记录 | 每次录音、每个模型的结果都会记下来，并计算平均值；可以导出 CSV |

## 批量测试

```bash
python download_models.py --tts      # 只有重新生成测试集时才需要
python make_test_audio.py            # 用 TTS 生成 benchmark/testset/（仓库里已经有一份）
python benchmark.py                  # 所有已下载模型 × 测试集，输出 benchmark/results/<时间>.md 和 .csv
python benchmark.py --dir D:\my_wavs --threads 1   # 用自己的真人录音测，目录里放 refs.tsv（文件名<TAB>参考文本）

# 长音频：VAD 切段 vs 整段对比（需要先下载 silero_vad.onnx）
curl -L -o test_audio/lei-jun-test.wav \
  https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/lei-jun-test.wav
python benchmark_vad.py test_audio/lei-jun-test.wav
```

**长音频必须配 VAD**：离线模型音频超过 31s 时会自动按 VAD 人声段切段推理（没有 VAD 模型时退回 30s 固定窗口切段），见 [docs/long-audio-report.md](docs/long-audio-report.md)。

## 语义切段

网页上「切段」默认选「语义」：VAD 先细切，再根据识别出的文字判断这句话说完没有（「我想……」「然后……」算没说完），决定等多久静音再断句。结果下方会列出每一句的断句依据和等待时长。

```bash
python benchmark_semantic.py         # 语义切段 vs 纯声学切段，输出 benchmark/results/semantic.md
```

实现和评测见 [docs/semantic-seg.md](docs/semantic-seg.md)。

## 声纹注册 + 说话人识别（名人声音实验）

先告诉系统"这是谁的声音"（注册），再让它认人。名人音频涉及版权和生物特征，**只放在 `test_audio/`（不会被提交）**。

1. 准备工具：需要 ffmpeg 和 yt-dlp。Windows 见 [公司电脑操作手册](docs/company-machine-guide.md) 第 4.1 节；Mac：

```bash
brew install ffmpeg
python3 -m pip install yt-dlp
```

2. 在 `benchmark/speakers/clips.tsv` 里填好每段音频：人名、用途（enroll 注册 / test 测试）、视频链接、起止时间。注册和测试**必须来自不同的视频**。清单里已经填好了 6 位好莱坞演员和雷军的公开音频，马斯克、黄仁勋两人留空待填。

3. 自动下载并截取，放进 `test_audio/speakers/`（不会被提交），来源记录到 `sources.tsv`：

```bash
python tools/fetch_speaker_clips.py
```

4. 运行：

```bash
python download_models.py --models speaker-embedding-en --no-diarization   # 可选：英文声纹模型做对照
python speaker_id.py
```

输出 `benchmark/results/speaker_id.md`：整条识别、1/2/3/5 秒片段的准确率、推荐阈值（等错误率 EER），以及多人拼接音频里"先分角色、再认人"的结果。已做过一轮实验（6 位好莱坞演员 + 雷军），结论见 [docs/speaker-id.md](docs/speaker-id.md)。

### 接进持续监听（主人识别）

注册自己的声音后，持续监听的每条正式结果带 `owner` 字段（true 主人 / false 陌生人 / null 判不了），页面显示徽标：

```bash
python tools/enroll_owner.py --name 老板 --files 我的录音1.wav 我的录音2.wav   # 存 data/owner_profile.json，不提交
# 重启 server 即自动启用；规则与验证数字见 [docs/owner-id.md](docs/owner-id.md)
```

## 常用参数

```bash
python server.py --threads 1                  # 模拟端侧的单核性能
python server.py --models sensevoice,paraformer-zh-small,zipformer-bilingual-small
python server.py --port 8080 --no-browser
python server.py --no-save-audio              # 持续监听不保存录音（默认存到 logs/live/，不提交）
python -m unittest discover -s tests          # 单元测试（不需要模型）
```

## 文件说明

| 文件 | 作用 |
|---|---|
| `registry.py` | 模型注册表：下载地址、文件匹配规则、文档体积和出处、公开 CER、许可证 |
| `download_models.py` | 下载模型：HF 优先，失败回退 GitHub；必要时自动做 int8 量化 |
| `asr_engine.py` | sherpa-onnx 加载和推理（离线 / 流式 / 说话人分离 / VAD 切段），以及 CER 和数字归一化 |
| `server.py` | FastAPI 后端，提供 `/api/models` 和 `/api/transcribe` |
| `web/index.html` | 前端页面（单文件，没有外部依赖） |
| `make_test_audio.py` / `benchmark.py` | 生成测试集 / 批量评测（`--refs` 指定参考文本，`--no-vad` 对照固定窗口） |
| `benchmark_vad.py` | 长音频 VAD 切段 vs 整段的对比测评 |
| `semantic_seg.py` / `benchmark_semantic.py` | 语义切段 / 语义切段评测 |
| `make_semantic_testset.py` | 生成语义切段测试集（句内思考停顿 + 长短不一的句间停顿，句子边界已知） |
| `vad_stream.py` | 流式 VAD：逐帧跑 Silero（与 sherpa-onnx 逐帧等价），给出每帧人声概率，切句、按句内停顿切段，长度上限是硬切 |
| `live_engine.py` | 持续监听核心：流式 VAD 切句 → 每个模型分别转写，实时出字分蹦字 / 段锁定 / 正式结果三层 → 事件回调（与网页无关，可搬到端侧） |
| `web/live.html` / `live_sim.py` | 持续监听页面（`/live`）/ 持续聊天场景下的双模型评测 |
| `speaker_id.py` | 声纹注册 + 说话人识别实验 |
| `tools/fetch_speaker_clips.py` / `benchmark/speakers/clips.tsv` | 按清单下载、截取说话人音频 |
| `tools/fetch_ramc.py` | 下载真人双人对话（MagicData-RAMC）并截取，生成带时间戳和说话人的标准文本（`live_sim.py --real` 用）；`--count N` 下前 N 段整段对话 |
| `tools/eval_vad.py` | 在 RAMC 整段对话的人工标注上评测 VAD 参数：开口 / 判停延迟、漏检、误检、混说话人、切段与算力 |
| `tools/replay_live.py` | 按真实速度把 WAV 推给 `/ws/live`，对比不同的实时出字间隔：开口后多久见字、屏幕落后多少、最长多久没有新字、正式结果慢了多少、多花多少算力 |
| `tools/make_ref_draft.py` | 多模型字级对齐，生成带【】的标准文本初稿（供人工听音校对） |
| `benchmark/long/` | 长音频标准文本初稿与定稿（`refs.tsv`） |
| `benchmark/` | 测试句、TTS 生成的测试音频、评测结果 |
| `docs/selection.md` | 选型调研报告 |
| `docs/long-audio-report.md` | 长音频实测、VAD 切段、音乐段幻听分析 |

## 常见问题

- **页面显示"无可用模型"**：先运行 `python download_models.py`，然后重启 `server.py`。
- **麦克风打不开**：确认地址是 `http://127.0.0.1:8000`，并检查 Windows 设置 → 隐私 → 麦克风里是否允许了浏览器访问。
- **下载一直失败**：试试 `--hf-mirror`，或者用 `--source github --gh-proxy` 加速前缀。如果公司网络有代理，先设置 `HTTPS_PROXY` 环境变量再运行。
- **8000 端口被占用**：`python server.py --port 8001`。
