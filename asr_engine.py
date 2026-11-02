"""sherpa-onnx 模型加载与推理封装。"""
from __future__ import annotations

import io
import re
import time
import unicodedata
import wave

import numpy as np

import registry

SAMPLE_RATE = 16000
VAD_PATH = None  # 由 server/asr_engine 初始化时探测；指向 silero_vad.onnx


# ---------------------------------------------------------------- 音频工具
def read_wav(data: bytes) -> tuple[np.ndarray, int]:
    """读取 PCM WAV（8/16/32 bit，任意声道/采样率），返回 float32 单声道 [-1,1]。"""
    with wave.open(io.BytesIO(data), "rb") as w:
        ch, width, sr, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
        raw = w.readframes(n)
    if width == 2:
        x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 4:
        x = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    elif width == 1:
        x = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"不支持的位深: {width * 8} bit")
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return np.ascontiguousarray(x, dtype=np.float32), sr


def resample(x: np.ndarray, sr: int, target: int = SAMPLE_RATE) -> np.ndarray:
    if sr == target or len(x) == 0:
        return x
    n = int(round(len(x) * target / sr))
    t_old = np.arange(len(x)) / sr
    t_new = np.arange(n) / target
    return np.interp(t_new, t_old, x).astype(np.float32)


# ---------------------------------------------------------------- CER
_PUNCT_RE = re.compile(r"\s+")
_ZH_DIGITS = "零一二三四五六七八九"


def _int_to_zh(n: int) -> str:
    """整数转中文读法（< 1 亿），如 15 -> 十五，305 -> 三百零五。"""
    if n == 0:
        return "零"
    units = ["", "十", "百", "千"]

    def four(x: int) -> str:
        out, zero = "", False
        for i in range(3, -1, -1):
            d = x // (10 ** i) % 10
            if d == 0:
                zero = bool(out)
            else:
                if zero:
                    out += "零"
                    zero = False
                out += _ZH_DIGITS[d] + units[i]
        return out

    hi, lo = divmod(n, 10000)
    s = (four(hi) + "万" if hi else "") + ("零" if hi and 0 < lo < 1000 else "") + (four(lo) if lo else "")
    return s[1:] if s.startswith("一十") else s


def arabic_to_zh(text: str) -> str:
    """把阿拉伯数字改写成汉字读法，用于 CER 归一化（让 ITN 输出和汉字输出可比）。
    规则：年份（后接“年”）、以 0 开头或 ≥9 位（电话号码等）的数字串逐位读；小数点读“点”；其余按数值读。"""
    text = re.sub(r"(\d+(?:\.\d+)?)%", r"百分之\1", text)

    def rep(m: re.Match) -> str:
        num, frac = m.group(1), m.group(2)
        after = m.string[m.end():m.end() + 1]
        if after == "年" or (num.startswith("0") and len(num) > 1) or len(num) >= 9:
            out = "".join(_ZH_DIGITS[int(c)] for c in num)
        else:
            out = _int_to_zh(int(num))
        if frac:
            out += "点" + "".join(_ZH_DIGITS[int(c)] for c in frac)
        return out

    return re.sub(r"(\d+)(?:\.(\d+))?", rep, text)


# ---------------------------------------------------------------- VAD
_vad = None

def _get_vad():
    """加载 silero VAD（懒加载，找不到模型文件返回 None）。"""
    global _vad
    if _vad is not None:
        return _vad
    import os
    import sherpa_onnx as so
    # download_models.py 把模型放在 models/silero-vad/；兼容手动放在 models/silero_vad.onnx 的旧布局
    p = VAD_PATH
    if not p:
        files = registry.resolve_files(registry.get_spec("silero-vad"))
        legacy = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "silero_vad.onnx")
        p = str(files["model"]) if files else legacy
    if not os.path.exists(p):
        _vad = False
        return _vad
    cfg = so.VadModelConfig()
    cfg.silero_vad.model = p
    cfg.silero_vad.threshold = 0.5
    cfg.silero_vad.min_silence_duration = 0.25
    cfg.silero_vad.min_speech_duration = 0.25
    cfg.sample_rate = SAMPLE_RATE
    _vad = so.VoiceActivityDetector(cfg, buffer_size_in_seconds=60)
    return _vad

def vad_segments(samples: np.ndarray, sr: int = SAMPLE_RATE) -> list[tuple[float, float]]:
    """返回 [(start_sec, end_sec), ...] 人声段；VAD 不可用时返回 []。"""
    v = _get_vad()
    if not v:
        return []
    v.reset()
    cfg = v.config
    ws = cfg.silero_vad.window_size  # silero 要求按 512 采样（32ms）精确分块
    segs = []
    for i in range(0, len(samples) - ws + 1, ws):
        v.accept_waveform(samples[i:i + ws])
        while not v.empty():
            seg = v.front
            segs.append((seg.start / sr, (seg.start + len(seg.samples)) / sr))
            v.pop()
    v.flush()
    while not v.empty():
        seg = v.front
        segs.append((seg.start / sr, (seg.start + len(seg.samples)) / sr))
        v.pop()
    v.reset()
    return segs


def normalize_text(s: str, norm_numbers: bool = False) -> str:
    s = unicodedata.normalize("NFKC", s)
    if norm_numbers:
        s = arabic_to_zh(s).replace("幺", "一").replace("两", "二")
    s = s.lower()
    s = "".join(ch for ch in s if not unicodedata.category(ch).startswith(("P", "S")))
    return _PUNCT_RE.sub("", s)


def edit_distance(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1]


def cer(ref: str, hyp: str, norm_numbers: bool = True) -> float | None:
    r, h = normalize_text(ref, norm_numbers), normalize_text(hyp, norm_numbers)
    if not r:
        return None
    return edit_distance(r, h) / len(r)


# ---------------------------------------------------------------- 模型
class AsrModel:
    def __init__(self, spec: dict, files: dict, num_threads: int = 2):
        self.spec = spec
        self.files = files
        self.num_threads = num_threads
        self.recognizer = None
        self.load_seconds = None

    @property
    def key(self) -> str:
        return self.spec["key"]

    def load(self):
        import sherpa_onnx as so

        f = {k: str(v) for k, v in self.files.items()}
        t0 = time.perf_counter()
        loader = self.spec["loader"]
        nt = self.num_threads
        if loader == "sense_voice":
            r = so.OfflineRecognizer.from_sense_voice(
                model=f["model"], tokens=f["tokens"], num_threads=nt, use_itn=True, language="auto")
        elif loader == "paraformer":
            r = so.OfflineRecognizer.from_paraformer(paraformer=f["model"], tokens=f["tokens"], num_threads=nt)
        elif loader == "dolphin_ctc":
            r = so.OfflineRecognizer.from_dolphin_ctc(model=f["model"], tokens=f["tokens"], num_threads=nt)
        elif loader == "online_zipformer2_ctc":
            r = so.OnlineRecognizer.from_zipformer2_ctc(tokens=f["tokens"], model=f["model"], num_threads=nt)
        elif loader == "online_transducer":
            r = so.OnlineRecognizer.from_transducer(
                tokens=f["tokens"], encoder=f["encoder"], decoder=f["decoder"], joiner=f["joiner"], num_threads=nt)
        elif loader == "online_wenet_ctc":
            r = so.OnlineRecognizer.from_wenet_ctc(tokens=f["tokens"], model=f["model"], num_threads=nt)
        else:
            raise ValueError(f"未知 loader: {loader}")
        self.recognizer = r
        self.load_seconds = time.perf_counter() - t0

    # 离线：优先 VAD 切段（人声段，避免音乐段幻听 + 消除重叠重复字），无 VAD 时按固定窗口切段
    CHUNK_SEC = 30.0      # 每段时长（固定窗口模式）
    CHUNK_OVERLAP = 1.0   # 段间重叠（固定窗口模式）

    def _offline(self, samples: np.ndarray, sr: int, use_vad: bool = True) -> dict:
        r = self.recognizer
        dur = len(samples) / sr
        segs = vad_segments(samples, sr) if use_vad else []
        if segs:
            # VAD 模式：逐人声段识别（前后各扩 0.2s），记录时间戳
            pad = 0.2
            texts, cuts = [], []
            for i, (a, b) in enumerate(segs):
                # 前后各扩最多 0.2s，但不越过与相邻段间隙的中点，避免相邻段重叠导致重复字
                lo = a - pad if i == 0 else max(a - pad, (segs[i - 1][1] + a) / 2)
                hi = b + pad if i == len(segs) - 1 else min(b + pad, (b + segs[i + 1][0]) / 2)
                s0, s1 = max(0, int(lo * sr)), min(len(samples), int(hi * sr))
                if s1 - s0 < sr // 5:
                    continue
                res = self._offline_stream(r, sr, samples[s0:s1])
                if res["text"]:
                    texts.append(res["text"])
                    cuts.append({"start": round(a, 2), "end": round(b, 2)})
            out = {"text": "".join(texts), "vad": True, "n_segments": len(cuts), "cuts": cuts}
            for k in ("lang", "emotion", "event"):
                if texts and k in res:
                    out[k] = res[k]
            return out
        if dur <= self.CHUNK_SEC + self.CHUNK_OVERLAP:
            return self._offline_stream(r, sr, samples)
        # 固定窗口模式（无 VAD）
        step = int(sr * (self.CHUNK_SEC - self.CHUNK_OVERLAP))
        win = int(sr * self.CHUNK_SEC)
        texts = []
        attrs = {}
        for start in range(0, len(samples), step):
            seg = samples[start:start + win]
            if len(seg) < sr // 2:  # 丢弃 <0.5s 的尾部碎段
                break
            res = self._offline_stream(r, sr, seg)
            if res["text"]:
                texts.append(res["text"])
            for k in ("lang", "emotion", "event"):
                if k in res:
                    attrs[k] = res[k]
        out = {"text": "".join(texts), "vad": False}
        out.update(attrs)
        return out

    def _offline_stream(self, r, sr: int, samples: np.ndarray) -> dict:
        s = r.create_stream()
        s.accept_waveform(sr, samples)
        r.decode_stream(s)
        res = s.result
        out = {"text": res.text.strip()}
        for attr in ("lang", "emotion", "event"):
            v = getattr(res, attr, "") or ""
            v = v.strip("<|>") if isinstance(v, str) else v
            if v:
                out[attr] = v
        return out

    # 流式：按 100ms 小块喂入，模拟实时场景，记录首字出现的音频时刻；
    # use_vad=True 时用 VAD 门控（只喂人声区，避免音乐段幻听）
    def _online(self, samples: np.ndarray, sr: int, chunk_sec: float = 0.1, use_vad: bool = True) -> dict:
        r = self.recognizer
        s = r.create_stream()
        step = int(sr * chunk_sec)
        first_partial_at = None
        partials = []
        last = ""
        chunk_costs = []
        # VAD 门控：把人声段合并（间隙 <0.5s 视为连续），只喂人声区
        gate = None
        if use_vad:
            segs = vad_segments(samples, sr)
            if segs:
                merged = []
                for a, b in segs:
                    if merged and a - merged[-1][1] < 0.5:
                        merged[-1][1] = b
                    else:
                        merged.append([a, b])
                gate = merged
        gate_i = 0
        for start in range(0, len(samples), step):
            t = (start + step) / sr
            if gate is not None:
                while gate_i < len(gate) and t > gate[gate_i][1]:
                    gate_i += 1
                if gate_i >= len(gate) or t < gate[gate_i][0]:
                    continue  # 非人声区，不喂
            c0 = time.perf_counter()
            s.accept_waveform(sr, samples[start:start + step])
            while r.is_ready(s):
                r.decode_stream(s)
            chunk_costs.append(time.perf_counter() - c0)
            text = r.get_result(s)
            if text != last:
                at = min(start + step, len(samples)) / sr
                if first_partial_at is None and text.strip():
                    first_partial_at = at
                partials.append({"t": round(at, 2), "text": text})
                last = text
        s.accept_waveform(sr, np.zeros(int(0.66 * sr), dtype=np.float32))  # 尾部补静音，冲出最后几帧
        s.input_finished()
        while r.is_ready(s):
            r.decode_stream(s)
        text = r.get_result(s).strip()
        return {
            "text": text,
            "first_partial_at": first_partial_at,
            "partials": partials[-30:],
            "chunk_ms_avg": round(1000 * float(np.mean(chunk_costs)), 2) if chunk_costs else None,
            "chunk_ms_max": round(1000 * float(np.max(chunk_costs)), 2) if chunk_costs else None,
            "vad": gate is not None,
        }

    def transcribe(self, samples: np.ndarray, sr: int = SAMPLE_RATE, use_vad: bool = True) -> dict:
        dur = len(samples) / sr
        t0 = time.perf_counter()
        out = (self._online(samples, sr, use_vad=use_vad) if self.spec["streaming"]
               else self._offline(samples, sr, use_vad=use_vad))
        elapsed = time.perf_counter() - t0
        out.update({
            "key": self.key,
            "elapsed_ms": round(elapsed * 1000, 1),
            "rtf": round(elapsed / dur, 4) if dur > 0 else None,
            "has_arabic_digits": bool(re.search(r"[0-9]", out["text"])),
        })
        return out


class Diarizer:
    def __init__(self, seg_path, emb_path, num_threads: int = 2):
        import sherpa_onnx as so

        self._so = so
        self.seg_path, self.emb_path, self.num_threads = str(seg_path), str(emb_path), num_threads
        self._cache = {}

    def _get(self, num_speakers: int, threshold: float):
        key = (num_speakers, threshold)
        if key not in self._cache:
            so = self._so
            cfg = so.OfflineSpeakerDiarizationConfig(
                segmentation=so.OfflineSpeakerSegmentationModelConfig(
                    pyannote=so.OfflineSpeakerSegmentationPyannoteModelConfig(model=self.seg_path),
                    num_threads=self.num_threads),
                embedding=so.SpeakerEmbeddingExtractorConfig(model=self.emb_path, num_threads=self.num_threads),
                clustering=so.FastClusteringConfig(num_clusters=num_speakers, threshold=threshold),
                min_duration_on=0.3,
                min_duration_off=0.5,
            )
            if not cfg.validate():
                raise RuntimeError("说话人分离配置校验失败，请检查模型文件")
            self._cache[key] = so.OfflineSpeakerDiarization(cfg)
        return self._cache[key]

    def run(self, samples16k: np.ndarray, num_speakers: int = -1, threshold: float = 0.5):
        sd = self._get(num_speakers, threshold)
        t0 = time.perf_counter()
        result = sd.process(samples16k).sort_by_start_time()
        segs = [{"start": round(x.start, 2), "end": round(x.end, 2), "speaker": int(x.speaker)} for x in result]
        return segs, (time.perf_counter() - t0) * 1000


def find_downloaded():
    """返回 {key: files} 已下载且文件齐全的 ASR 模型。"""
    out = {}
    for spec in registry.ASR_MODELS:
        files = registry.resolve_files(spec)
        if files:
            out[spec["key"]] = files
    return out
