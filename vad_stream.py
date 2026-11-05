"""流式 VAD：逐帧跑 Silero，给出每帧的人声概率，把语音切成「句」和「段」。

和 sherpa-onnx 的 VoiceActivityDetector 用同一个 silero_vad.onnx、同一套判定规则，默认参数下逐帧等价
（tests/test_core.py 有对照测试；177 秒真人对话上 27/27 段起止一个采样点都不差）。多出三样东西：
  - 每帧的人声概率（sherpa 的 Python 接口只给"是不是在说话"）；
  - 句内短停顿切「段」：给实时出字用，每段单独反复识别，段越短越省算力；
  - 真正的长度上限：段 / 句到上限就硬切，切在最近 2 秒里概率最低的那一帧
    （sherpa 的上限是软的：超长后只是把判停条件放宽，实测切出过 16.2 秒的段）。

用法:
    vad = SileroStream(min_silence=0.5, sentence_max=15, piece_gap=0.2, piece_max=8)
    for fr in vad.accept(pcm):     # pcm：16kHz 单声道 float32，长度任意；每 512 采样（32ms）一帧
        fr.end, fr.prob, fr.events
    vad.flush()                    # 结束时把没说完的那句收尾，返回事件

事件 Event.kind（位置都是采样数，从会话开始算）:
    start  一句开始。at = 起点（往回推到真正开口处）
           reason = speech（有人开口）/ max_len（上一句超长被切开，人还在接着说）
    cut    句内切段。at = 切点，下一段从这里开始；piece = 刚切下来的是第几段（从 1 数）；
           hi = 刚切下来的这段识别到哪里为止（pause：判定时刻，切点之后都是静音；max_len：就是切点）
           reason = pause（停顿 ≥ piece_gap）/ max_len（段长到 piece_max）
    end    一句结束。start / at = 这句的起止
           reason = silence（静音 ≥ min_silence）/ max_len（句长到 sentence_max）/ flush（调用 flush 时还没说完）
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

SR = 16000
WIN = 512   # silero v4 每次吃 512 采样（32ms）


@dataclass
class Event:
    kind: str            # start / cut / end
    at: int              # start：句起点；cut：切点；end：句终点
    reason: str
    start: int = -1      # end：这句的起点
    piece: int = 0       # cut：刚切下来的是第几段
    hi: int = -1         # cut：刚切下来的这段识别到哪里为止


@dataclass
class Frame:
    end: int             # 这一帧结束的位置（= 已处理的采样数）
    prob: float          # 这一帧的人声概率
    events: list = field(default_factory=list)


class SileroModel:
    """silero_vad.onnx（v4：输入 x / h / c）逐帧推理，自己保存 LSTM 状态。"""

    def __init__(self, path: str | None = None, threads: int = 1):
        import onnxruntime as ort
        if path is None:
            import asr_engine
            path = asr_engine.vad_model_path()
        if not path:
            raise RuntimeError("没有 VAD 模型：python download_models.py")
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        names = [i.name for i in self.sess.get_inputs()]
        if not {"h", "c"} <= set(names):
            raise RuntimeError(f"不支持的 Silero 模型（输入 {names}）：只支持 v4（x / h / c）")
        self._x = next(n for n in names if n not in ("h", "c", "sr"))
        self._sr = np.array(SR, np.int64) if "sr" in names else None
        self.reset()

    def reset(self) -> None:
        self.h = np.zeros((2, 1, 64), np.float32)
        self.c = np.zeros((2, 1, 64), np.float32)

    def __call__(self, frame: np.ndarray) -> float:
        feed = {self._x: np.asarray(frame, np.float32).reshape(1, WIN), "h": self.h, "c": self.c}
        if self._sr is not None:
            feed["sr"] = self._sr
        prob, self.h, self.c = self.sess.run(None, feed)
        return float(prob.ravel()[0])


class Segmenter:
    """人声概率 → 句 / 段。纯 Python，不跑模型：同一串概率可以换参数反复重放（tools/eval_vad.py）。

    判定规则照抄 sherpa-onnx（silero-vad-model.cc 的 IsSpeech + voice-activity-detector.cc）：
      - 概率 > threshold 的帧持续 min_speech 秒才算开口；开口以后概率 > threshold − 0.15 都算还在说；
      - 概率掉到 neg_threshold（默认 threshold − 0.15）以下开始计静音（中途概率超过 threshold 才清零），
        满 min_silence 秒算这句说完；
      - 句起点 = 确认开口那一帧 − 2 帧 − min_speech；句终点 = 判定说完那一帧 − min_silence。
    在此之上加两级切分（参数为 0 = 不启用）：
      - 段：静音计到 piece_gap 秒（还没到 min_silence）、这段已经 ≥ piece_min 秒，就在已静音部分的中点切开；
            段长到 piece_max 秒强制切，切在最近 cut_window 秒里概率最低的那一帧；
      - 句：句长到 sentence_max 秒强制切（切点规则同上），切点之后算新的一句。
    """

    def __init__(self, threshold: float = 0.5, min_speech: float = 0.25, min_silence: float = 0.5,
                 sentence_max: float = 0.0, piece_gap: float = 0.0, piece_min: float = 1.0,
                 piece_max: float = 0.0, cut_window: float = 2.0, neg_threshold: float | None = None):
        self.threshold = float(threshold)
        # 开口以后概率高于它都算还在说、不计静音。sherpa / Silero 官方固定为 threshold − 0.15；
        # 有底噪的房间里说完话概率要一秒多才降到 0.35 以下，判停跟着晚（tools/eval_vad.py），所以可调
        self.neg = self.threshold - 0.15 if neg_threshold is None else min(float(neg_threshold), self.threshold)
        self.min_speech = int(min_speech * SR)
        self.min_silence = int(min_silence * SR)
        self.sentence_max = int(sentence_max * SR)
        self.piece_gap = int(piece_gap * SR)
        self.piece_min = int(piece_min * SR)
        self.piece_max = int(piece_max * SR)
        self.cut_window = int(cut_window * SR)
        self.cur = 0                 # 已处理的采样数
        self._ts = self._te = 0      # sherpa 的 temp_start / temp_end（帧终点位置）
        self._triggered = False
        self._floor = 0              # 上一句的终点：新句起点不早于它
        self.start = -1              # 当前这句的起点；-1 = 没在说话
        self.piece = 0               # 当前是这句的第几段
        self.piece_start = -1        # 当前这段的起点
        self.voiced_from = -1        # 当前这段里第一帧人声（概率 > threshold）的起点；-1 = 这段还没人声
        self._pause_cut = -1         # 已经按哪一次静音（_te）切过段
        self._hist: deque = deque(maxlen=max(1, self.cut_window // WIN))   # 最近的 (帧终点, 概率)

    @property
    def in_speech(self) -> bool:
        """和 sherpa 的 is_speech_detected() 一致。"""
        return self.start != -1

    @property
    def silence(self) -> int:
        """正在计的静音有多少采样（没在计为 0）。"""
        return self.cur - self._te if self._te else 0

    def step(self, p: float) -> list[Event]:
        """喂一帧（512 采样）的人声概率，返回这一帧产生的事件。"""
        self.cur += WIN
        self._hist.append((self.cur, p))
        ev: list[Event] = []
        if self._is_speech(p):
            if self.start == -1:
                self._open(max(self.cur - 2 * WIN - self.min_speech, self._floor), "speech", ev)
                self.voiced_from = max(self._ts - WIN, self.start)
            else:
                self._cuts(ev)
        elif self.start != -1:
            self._close(self.cur - self.min_silence, "silence", ev)
        if self.start != -1 and self.voiced_from == -1 and p > self.threshold:
            self.voiced_from = self.cur - WIN
        return ev

    def flush(self) -> list[Event]:
        """收尾：正在说的那句截到当前位置（和 sherpa 的 flush 一致）。"""
        ev: list[Event] = []
        if self.start != -1:
            self._close(self.cur, "flush", ev)
        return ev

    # ------------------------------------------------------------ 内部
    def _is_speech(self, p: float) -> bool:
        """sherpa-onnx SileroVadModel::IsSpeech 逐句照搬（分支顺序也不能动）。"""
        thr, cur = self.threshold, self.cur
        if p > thr and self._te:
            self._te = 0
        if p > thr and self._ts == 0:
            self._ts = cur
            return False
        if p > thr and self._ts and not self._triggered:
            if cur - self._ts < self.min_speech:
                return False
            self._triggered = True
            return True
        if p < thr and not self._triggered:
            self._ts = self._te = 0
            return False
        if p > self.neg and self._triggered:
            return True
        if p > thr and not self._triggered:    # sherpa 里有这一支，实际走不到
            self._triggered = True
            return True
        if self._triggered:                    # 走到这里 p ≤ neg（sherpa 写作 p < thr，neg < thr 时等价；neg = thr 时 p == thr 也算静音）
            if self._te == 0:
                self._te = cur
            if cur - self._te < self.min_silence:
                return True
            self._ts = self._te = 0
            self._triggered = False
            return False
        return False

    def _open(self, at: int, reason: str, ev: list) -> None:
        self.start = self.piece_start = at
        self.piece = 1
        self.voiced_from = -1
        ev.append(Event("start", at, reason))

    def _close(self, at: int, reason: str, ev: list) -> None:
        ev.append(Event("end", at, reason, start=self.start))
        self._floor = at
        self.start = self.piece_start = self.voiced_from = -1
        self.piece = 0

    def _cuts(self, ev: list) -> None:
        cur = self.cur
        if self.sentence_max and cur - self.start >= self.sentence_max:
            at = self._lowest(max(cur - self.cut_window, self.piece_start))
            self._close(at, "max_len", ev)
            self._open(at, "max_len", ev)
            self.voiced_from = self._voiced_after(at)
            return
        if (self.piece_gap and self._te and self._te != self._pause_cut and cur - self._te >= self.piece_gap
                and self.voiced_from != -1 and self._te - WIN - self.piece_start >= self.piece_min):
            self._pause_cut = self._te
            self._split((self._te - WIN + cur) // 2, cur, "pause", ev)
            return
        if self.piece_max and cur - self.piece_start >= self.piece_max:
            at = self._lowest(max(cur - self.cut_window, self.piece_start + WIN))
            self._split(at, at, "max_len", ev)

    def _split(self, at: int, hi: int, reason: str, ev: list) -> None:
        ev.append(Event("cut", at, reason, piece=self.piece, hi=hi))
        self.piece += 1
        self.piece_start = at
        self.voiced_from = self._voiced_after(at)

    def _lowest(self, lo: int) -> int:
        """[lo, 当前] 里概率最低的那一帧的中点（一样低取最晚的，少切几刀）；一帧都没有就切在 lo。"""
        best = None
        for e, q in self._hist:
            if e - WIN >= lo and (best is None or q <= best[1]):
                best = (e, q)
        return lo if best is None else best[0] - WIN // 2

    def _voiced_after(self, at: int) -> int:
        return next((e - WIN for e, q in self._hist if e - WIN >= at and q > self.threshold), -1)


class SileroStream:
    """Silero 逐帧推理 + Segmenter，外加自动增益（只作用于 VAD 的输入，不改送去识别的音频）。"""

    def __init__(self, model_path: str | None = None, agc: bool = True, threads: int = 1, **seg_kwargs):
        self.model = SileroModel(model_path, threads)
        self.seg = Segmenter(**seg_kwargs)
        self.agc = agc
        self.gain = 1.0
        self._peak = 0.0
        self._rest = np.zeros(0, np.float32)

    def accept(self, pcm: np.ndarray) -> list[Frame]:
        """喂入 16kHz 单声道 float32 音频（长度任意），返回凑满的每一帧。"""
        pcm = np.asarray(pcm, dtype=np.float32)
        if self.agc and len(pcm):
            # 轻声 / 远场时 silero 的人声概率贴近阈值，会把整句判丢（实测 -40dBFS 的一句话在 Int16 输入下被丢掉，
            # 放大 1.3 倍即恢复；纯噪声放大 10 倍也不误触发）。峰值半衰期约 5s
            self._peak = max(float(np.abs(pcm).max()), self._peak * 0.5 ** (len(pcm) / SR / 5.0))
            self.gain = float(np.clip(0.7 / max(self._peak, 0.07), 1.0, 10.0))
        x = np.concatenate([self._rest, np.clip(pcm * self.gain, -1.0, 1.0)])
        n = len(x) // WIN * WIN
        frames = []
        for i in range(0, n, WIN):
            p = self.model(x[i:i + WIN])
            ev = self.seg.step(p)
            frames.append(Frame(self.seg.cur, p, ev))
        self._rest = x[n:]
        return frames

    def flush(self) -> list[Event]:
        return self.seg.flush()


def frame_probs(x: np.ndarray, model_path: str | None = None, agc: bool = True, block: int = 1600) -> np.ndarray:
    """整段音频的逐帧人声概率。按 block 采样一块喂入，自动增益和实时链路（0.1s 一块）一致。"""
    s = SileroStream(model_path, agc=agc)
    probs: list[float] = []
    for i in range(0, len(x), block):
        probs += [f.prob for f in s.accept(x[i:i + block])]
    return np.asarray(probs, np.float32)


def replay(probs, **seg_kwargs) -> tuple[list[tuple[int, int, str]], list[Event]]:
    """拿一串逐帧概率按给定参数重放切分（最后自动 flush）。返回 [(句起点, 句终点, 结束原因)] 和全部 cut 事件。"""
    seg = Segmenter(**seg_kwargs)
    sents: list[tuple[int, int, str]] = []
    cuts: list[Event] = []
    for p in probs:
        for e in seg.step(float(p)):
            if e.kind == "end":
                sents.append((e.start, e.at, e.reason))
            elif e.kind == "cut":
                cuts.append(e)
    sents += [(e.start, e.at, e.reason) for e in seg.flush()]
    return sents, cuts
