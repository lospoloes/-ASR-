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


# 英文声纹模型（可选，与中文 3D-Speaker 对照；名人声纹实验里 Musk / 黄仁勋 说英文）
DIARIZATION_MODELS.append({
    "key": "speaker-embedding-en",
    "name": "NeMo TitaNet small（英文声纹，可选）",
    "files": {"model": ["nemo_en_titanet_small.onnx"]},
    "hf_repo": "csukuangfj/speaker-embedding-models",
    "hf_verified": False,
    "hf_allow": ["nemo_en_titanet_small.onnx"],
    "gh_url": f"{GH}/speaker-recongition-models/nemo_en_titanet_small.onnx",
    "doc_size_mb": None,
    "size_source": "未核实（文档未列体积）",
    "optional": True,
})


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


# 标点：给没有标点的 Paraformer 输出加标点，可选（语义切段可以不用它，只靠规则）
PUNCT_MODELS: list[dict] = [
    {
        "key": "punct-ct-transformer",
        "name": "CT-Transformer 中英标点 int8",
        "files": {"model": ["model.int8.onnx"]},
        "hf_repo": "csukuangfj/sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12-int8",
        "hf_verified": False,
        "hf_allow": ["model.int8.onnx"],
        "gh_url": f"{GH}/punctuation-models/sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12-int8.tar.bz2",
        "doc_size_mb": 72,
        "size_source": "sherpa 文档 punctuation/pretrained_models.html（model.int8.onnx 72M）",
    },
]


def all_specs() -> list[dict]:
    return ASR_MODELS + DIARIZATION_MODELS + TTS_MODELS + VAD_MODELS + PUNCT_MODELS


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
