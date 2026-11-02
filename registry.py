"""模型注册表：download_models.py 和 server.py 共用。

每个模型条目说明：
- files:   角色 -> glob 模式列表（按优先级），在 models/<key>/ 下递归查找，优先 int8
- hf_repo / hf_allow: HuggingFace 仓库与要下载的文件模式（hf_verified=False 表示仓库名未能在调研时核实）
- gh_url:  GitHub Release 压缩包（地址均来自 k2-fsa/sherpa 官方文档），作为 HF 失败时的回退
- doc_size_mb / size_source: 文档中的 int8 文件体积及出处
- public_cer: 公开 CER 及测试集、出处；查不到写“未核实”
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"

GH = "https://github.com/k2-fsa/sherpa-onnx/releases/download"
DOC = "https://k2-fsa.github.io/sherpa/onnx"

ASR_MODELS: list[dict] = [
    {
        "key": "sensevoice",
        "name": "SenseVoice-Small int8（基准）",
        "loader": "sense_voice",
        "streaming": False,
        "files": {"model": ["model.int8.onnx", "model.onnx"], "tokens": ["tokens.txt"]},
        "hf_repo": "csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17",
        "hf_verified": False,
        "hf_allow": ["model.int8.onnx", "tokens.txt"],
        "gh_url": f"{GH}/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17.tar.bz2",
        "doc_size_mb": 228,
        "size_source": "sherpa 文档 sense-voice/pretrained.html（ls -lh: model.int8.onnx 228M）",
        "languages": "中/英/日/韩/粤，自动语种识别",
        "number_format": "阿拉伯数字（开启 use_itn，会议实测）",
        "public_cer": "约 2.9%（AISHELL-1 test）/ 约 7.6%（WenetSpeech test_net），读自官方 README 柱状图，PyTorch 原模型，非 int8",
        "license": "FunASR Model License（官方澄清可商用，需署名）",
        "link": f"{DOC}/sense-voice/pretrained.html",
        "extras": "输出语种/情感/声学事件标签，带标点",
        "default": True,
        "default_reason": "会议定的性能基准",
    },
    {
        "key": "zipformer-ctc-small-zh",
        "name": "Zipformer-CTC small 流式 int8",
        "loader": "online_zipformer2_ctc",
        "streaming": True,
        "files": {"model": ["model.int8.onnx", "model.onnx"], "tokens": ["tokens.txt"]},
        "hf_repo": "csukuangfj/sherpa-onnx-streaming-zipformer-small-ctc-zh-int8-2025-04-01",
        "hf_verified": False,
        "hf_allow": ["model.int8.onnx", "tokens.txt", "bbpe.model"],
        "gh_url": f"{GH}/asr-models/sherpa-onnx-streaming-zipformer-small-ctc-zh-int8-2025-04-01.tar.bz2",
        "doc_size_mb": 25,
        "size_source": "sherpa 文档 online-ctc/zipformer-ctc-models.html（model.int8.onnx 25M）",
        "languages": "中文",
        "number_format": "汉字数字（测试集实测）",
        "public_cer": "未核实（官方文档未给出）",
        "license": "Apache-2.0（icefall/sherpa-onnx；模型卡未核实）",
        "link": f"{DOC}/pretrained_models/online-ctc/zipformer-ctc-models.html",
        "extras": "",
        "default": True,
        "default_reason": "最小的流式中文模型之一",
    },
    {
        "key": "zipformer-zh-14m",
        "name": "Zipformer 流式 zh-14M int8",
        "loader": "online_transducer",
        "streaming": True,
        "files": {
            "encoder": ["encoder*.int8.onnx", "encoder*.onnx"],
            # 实测：int8 decoder 会导致 transducer 输出被截断，必须用 fp32 decoder（官方示例也是这样配的）
            "decoder": ["decoder-epoch-99-avg-1.onnx", "decoder.onnx"],
            "joiner": ["joiner*.int8.onnx", "joiner*.onnx"],
            "tokens": ["tokens.txt"],
        },
        "hf_repo": "csukuangfj/sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23",
        "hf_verified": True,
        "hf_allow": ["encoder*.int8.onnx", "joiner*.int8.onnx", "decoder-epoch-99-avg-1.onnx", "tokens.txt"],
        "gh_url": f"{GH}/asr-models/sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23.tar.bz2",
        "doc_size_mb": 29.9,
        "size_source": "sherpa 文档 online-transducer（enc int8 21M + dec fp32 7.2M + joiner int8 1.7M；全 int8 为 24.5M，但 int8 decoder 实测会截断输出）",
        "languages": "中文（WenetSpeech 训练）",
        "number_format": "汉字数字（测试集实测）",
        "public_cer": "未核实",
        "license": "Apache-2.0（icefall；模型卡未核实）",
        "link": f"{DOC}/pretrained_models/online-transducer/zipformer-transducer-models.html",
        "extras": "14M 参数，超低资源",
        "default": True,
        "default_reason": "下限参照：极小体积能做到什么程度",
    },
    {
        "key": "zipformer-bilingual-small",
        "name": "Zipformer 流式中英双语 small int8",
        "loader": "online_transducer",
        "streaming": True,
        "files": {
            "encoder": ["encoder*.int8.onnx", "encoder*.onnx"],
            # 实测：int8 decoder 会导致 transducer 输出被截断，必须用 fp32 decoder（官方示例也是这样配的）
            "decoder": ["decoder-epoch-99-avg-1.onnx", "decoder.onnx"],
            "joiner": ["joiner*.int8.onnx", "joiner*.onnx"],
            "tokens": ["tokens.txt"],
        },
        "hf_repo": "csukuangfj/sherpa-onnx-streaming-zipformer-small-bilingual-zh-en-2023-02-16",
        "hf_verified": False,
        "hf_allow": ["encoder*.int8.onnx", "joiner*.int8.onnx", "decoder-epoch-99-avg-1.onnx", "tokens.txt", "bpe.model"],
        "gh_url": f"{GH}/asr-models/sherpa-onnx-streaming-zipformer-small-bilingual-zh-en-2023-02-16.tar.bz2",
        "doc_size_mb": 58.1,
        "size_source": "sherpa 文档 online-transducer（enc int8 41M + dec fp32 14M + joiner int8 3.1M；全 int8 为 47.5M，但 int8 decoder 实测会截断输出）",
        "languages": "中英混说",
        "number_format": "汉字数字（测试集实测）",
        "public_cer": "未核实（社区贡献，内部数据训练）",
        "license": "未核实（社区贡献模型）",
        "link": f"{DOC}/pretrained_models/online-transducer/zipformer-transducer-models.html",
        "extras": "",
        "default": True,
        "default_reason": "流式 + 中英混说 + <50MB",
    },
    {
        "key": "zipformer-ctc-multi-zh-hans",
        "name": "Zipformer-CTC 流式 multi-zh-hans int8",
        "loader": "online_zipformer2_ctc",
        "streaming": True,
        "files": {"model": ["ctc*.int8.onnx", "*.int8.onnx", "ctc*.onnx"], "tokens": ["tokens.txt"]},
        "hf_repo": "csukuangfj/sherpa-onnx-streaming-zipformer-ctc-multi-zh-hans-2023-12-13",
        "hf_verified": False,
        "hf_allow": ["*.int8.onnx", "tokens.txt", "bpe.model"],
        "gh_url": f"{GH}/asr-models/sherpa-onnx-streaming-zipformer-ctc-multi-zh-hans-2023-12-13.tar.bz2",
        "doc_size_mb": 68,
        "size_source": "sherpa 文档 online-ctc 列出 int8 68M；但当前 GitHub 压缩包只含 fp32（252MB），下载脚本会用 onnxruntime 动态量化生成 int8（实测 66MB）",
        "quantize": {"src": "ctc-epoch-20-avg-1-chunk-16-left-128.onnx",
                     "dst": "ctc-epoch-20-avg-1-chunk-16-left-128.int8.onnx"},
        "languages": "中文（14k 小时多数据集）",
        "number_format": "汉字数字（测试集实测）",
        "public_cer": "未核实（icefall multi_zh-hans RESULTS.md 未单列该版本）",
        "license": "Apache-2.0（icefall；模型卡未核实）",
        "link": f"{DOC}/pretrained_models/online-ctc/zipformer-ctc-models.html",
        "extras": "训练数据量大（14k h），流式中精度候选",
        "default": True,
        "default_reason": "流式 + <100MB + 训练数据量大",
    },
    {
        "key": "paraformer-zh-small",
        "name": "Paraformer-zh-small int8",
        "loader": "paraformer",
        "streaming": False,
        "files": {"model": ["model.int8.onnx", "model.onnx"], "tokens": ["tokens.txt"]},
        "hf_repo": "csukuangfj/sherpa-onnx-paraformer-zh-small-2024-03-09",
        "hf_verified": True,
        "hf_allow": ["model.int8.onnx", "tokens.txt"],
        "gh_url": f"{GH}/asr-models/sherpa-onnx-paraformer-zh-small-2024-03-09.tar.bz2",
        "doc_size_mb": 79,
        "size_source": "sherpa 文档 offline-paraformer（model.int8.onnx 79M；第三方实测记 82MB）",
        "languages": "中英，支持河南/天津/四川等方言（文档）",
        "number_format": "汉字数字（测试集实测）",
        "public_cer": "未核实（small 版无公开 CER；大版 Paraformer-zh AISHELL-1 约 1.9%，不可直接类比）",
        "license": "FunASR Model License（未逐条核实）",
        "link": f"{DOC}/pretrained_models/offline-paraformer/paraformer-models.html",
        "extras": "无时间戳",
        "default": True,
        "default_reason": "<100MB 离线 + 方言",
    },
    {
        "key": "dolphin-base",
        "name": "Dolphin base CTC int8",
        "loader": "dolphin_ctc",
        "streaming": False,
        "files": {"model": ["model.int8.onnx", "model.onnx"], "tokens": ["tokens.txt"]},
        "hf_repo": "csukuangfj/sherpa-onnx-dolphin-base-ctc-multi-lang-int8-2025-04-02",
        "hf_verified": False,
        "hf_allow": ["model.int8.onnx", "tokens.txt"],
        "gh_url": f"{GH}/asr-models/sherpa-onnx-dolphin-base-ctc-multi-lang-int8-2025-04-02.tar.bz2",
        "doc_size_mb": 99,
        "size_source": "sherpa 文档 Dolphin/pretrained.html（model.int8.onnx 99M，total 100M）",
        "languages": "40 种东方语言 + 22 种中文方言",
        "number_format": "汉字数字（测试集实测）；粤语输出繁体字",
        "public_cer": "未核实（官方 README 无 base 版中文 CER 表）",
        "license": "Apache-2.0（Dolphin 官方 README）",
        "link": f"{DOC}/Dolphin/pretrained.html",
        "extras": "方言覆盖最广",
        "default": True,
        "default_reason": "方言能力突出，体积刚好 ~100MB",
    },
    {
        "key": "wenet-wenetspeech",
        "name": "WeNet WenetSpeech 流式 int8（实验）",
        "loader": "online_wenet_ctc",
        "streaming": True,
        "files": {"model": ["model-streaming.int8.onnx", "*streaming*.int8.onnx"], "tokens": ["tokens.txt"]},
        "hf_repo": "csukuangfj/sherpa-onnx-zh-wenet-wenetspeech",
        "hf_verified": True,
        "hf_allow": ["*streaming*.int8.onnx", "tokens.txt"],
        "gh_url": None,
        "doc_size_mb": 83,
        "size_source": "第三方实测 83MB（参考起点提供，HF 文件列表调研时无法访问，未核实）",
        "languages": "中文",
        "number_format": "未核实",
        "public_cer": "未核实",
        "license": "Apache-2.0（WeNet；模型卡未核实）",
        "link": f"{DOC}/pretrained_models/wenet/all-models.html",
        "extras": "仓库内文件名未核实，只有 HF 一个来源",
        "default": False,
        "default_reason": "可选，见 selection.md",
    },
]

# 分角色（说话人分离）用的辅助模型
DIARIZATION_MODELS: list[dict] = [
    {
        "key": "pyannote-seg",
        "name": "pyannote segmentation-3.0 int8",
        "files": {"model": ["model.int8.onnx", "model.onnx"]},
        "hf_repo": "csukuangfj/sherpa-onnx-pyannote-segmentation-3-0",
        "hf_verified": False,
        "hf_allow": ["model.int8.onnx", "model.onnx"],
        "gh_url": f"{GH}/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2",
        "doc_size_mb": 1.5,
        "size_source": "sherpa 文档 speaker-diarization/models.html（model.int8.onnx 1.5M）",
    },
    {
        "key": "speaker-embedding",
        "name": "3D-Speaker ERes2Net base zh-cn",
        "files": {"model": ["3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx", "*.onnx"]},
        "hf_repo": "csukuangfj/speaker-embedding-models",
        "hf_verified": False,
        "hf_allow": ["3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"],
        "gh_url": f"{GH}/speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx",
        "doc_size_mb": None,
        "size_source": "未核实（文档未列体积）",
    },
]


# TTS：用于 make_test_audio.py 生成带标准答案的测试音频（也作为“音色定制”方向的初步验证）
TTS_MODELS: list[dict] = [
    {
        "key": "tts-aishell3",
        "name": "VITS aishell3（174 个音色）",
        "files": {"model": ["model.onnx"], "lexicon": ["lexicon.txt"], "tokens": ["tokens.txt"]},
        "rule_fsts": ["phone.fst", "date.fst", "number.fst"],
        "hf_repo": "csukuangfj/vits-icefall-zh-aishell3",
        "hf_verified": False,
        "hf_allow": ["model.onnx", "lexicon.txt", "tokens.txt", "*.fst"],
        "gh_url": f"{GH}/tts-models/vits-icefall-zh-aishell3.tar.bz2",
        "doc_size_mb": 29,
        "size_source": "sherpa 文档 tts/pretrained_models/vits.html（model.onnx 29M）",
    },
    {
        "key": "tts-melo-zh-en",
        "name": "VITS MeloTTS zh_en（中英，1 个音色）",
        "files": {"model": ["model.onnx"], "lexicon": ["lexicon.txt"], "tokens": ["tokens.txt"]},
        "rule_fsts": ["date.fst", "number.fst"],
        "hf_repo": "csukuangfj/vits-melo-tts-zh_en",
        "hf_verified": False,
        "hf_allow": ["model.onnx", "lexicon.txt", "tokens.txt", "*.fst"],
        "gh_url": f"{GH}/tts-models/vits-melo-tts-zh_en.tar.bz2",
        "doc_size_mb": 163,
        "size_source": "sherpa 文档 tts/pretrained_models/vits.html（model.onnx 163M）",
    },
]


# VAD：长音频切段 / 流式门控（离线模型长音频必须配 VAD，见 docs/long-audio-report.md）
VAD_MODELS: list[dict] = [
    {
        "key": "silero-vad",
        "name": "Silero VAD（人声检测）",
        "files": {"model": ["silero_vad.onnx"]},
        "hf_repo": "csukuangfj/silero-vad",
        "hf_verified": False,
        "hf_allow": ["silero_vad.onnx"],
        "gh_url": f"{GH}/asr-models/silero_vad.onnx",
        "doc_size_mb": 0.6,
        "size_source": "实测 643KB（必须是官方 v4 版；ModelScope 的 2.3MB 变体判定过保守，会把掌声下的人声判丢）",
    },
]


def all_specs() -> list[dict]:
    return ASR_MODELS + DIARIZATION_MODELS + TTS_MODELS + VAD_MODELS


def get_spec(key: str) -> dict:
    for s in all_specs():
        if s["key"] == key:
            return s
    raise KeyError(key)


def resolve_files(spec: dict, models_dir: Path = MODELS_DIR) -> dict[str, Path] | None:
    """在 models/<key>/ 下按优先级找到每个角色的文件；缺任意一个返回 None。"""
    d = models_dir / spec["key"]
    if not d.is_dir():
        return None
    out: dict[str, Path] = {}
    for role, patterns in spec["files"].items():
        found = None
        for pat in patterns:
            # 先找顶层，再递归（部分压缩包带 64/、96/ 等变体子目录，不能误选）
            hits = sorted(p for p in d.glob(pat) if p.is_file())
            if not hits:
                hits = sorted(p for p in d.rglob(pat) if p.is_file() and "test_wavs" not in p.parts)
            if hits:
                found = hits[0]
                break
        if found is None:
            return None
        out[role] = found
    return out


def disk_size_mb(files: dict[str, Path] | None) -> float | None:
    if not files:
        return None
    # tokens/bpe 文件也算进去，和“模型包体积”口径一致
    return round(sum(p.stat().st_size for p in set(files.values())) / 1024 / 1024, 1)
