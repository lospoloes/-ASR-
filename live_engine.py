"""持续监听：流式 VAD 切 chunk → 同一个 chunk 分别交给每个 ASR 模型转写 → 事件回调。

和网页无关，可以直接搬到端侧常驻链路里用：
    sess = LiveSession(models, on_event=print)
    while True:
        sess.accept(pcm_float32_16k)     # 麦克风来一块喂一块，任意长度
    sess.close()                         # 结束时把最后一句冲出来

事件（dict，type 字段区分）:
    speech_start  {t}                         检测到有人开始说话（t：音频时间，秒）
    speaking      {t, dur}                    正在说话，每 0.5s 一次（dur：这句已说了多久）
    speech_end    {t}                         VAD 判定这句说完了
    chunk         {id, start, end, dur, reason, endpoint_wait_ms}
                  切出一个 chunk。reason：silence（静音达到阈值）/ max_len（超长被切）
                  endpoint_wait_ms：语音结束到 VAD 判停之间等的静音（音频时间）
    partial       {chunk_id, model, text, start, end, decode_ms, lag_ms}
                  临时结果（partial_interval > 0 才有）：这句还没说完，把已经说了的部分（start → end）重新识别一遍。
                  chunk_id 是这句切出来以后会用的编号，同一句的 partial 一定排在它的 text 前面，text 到了就替换掉。
                  lag_ms：从这段音频到手到出临时结果用了多久
    text          {chunk_id, model, text, decode_ms, rtf, latency_ms, shown_ms, ...}
                  某个模型转写完一个 chunk（正式结果）。
                  latency_ms = endpoint_wait_ms + decode_ms：该模型【单独部署】时，从说完到出字的延迟
                  shown_ms：本次实际出字用时（两个模型排队跑，后跑的会多等前一个的解码；开了临时结果时，
                  还可能要等手上那次临时识别做完）
    error         {message}

两个模型拿到的是完全相同的 chunk，各自独立转写，便于公平对比。
"""
from __future__ import annotations

import json
import threading
import time
import wave
from collections import deque
from pathlib import Path

import numpy as np

import asr_engine

SR = asr_engine.SAMPLE_RATE
WIN = 512  # silero VAD 要求按 512 采样（32ms）精确喂入
ROOT = Path(__file__).resolve().parent


class LiveSession:
    def __init__(self, models: dict, on_event, min_silence: float = 0.5, max_chunk: float = 15.0,
                 pad: float = 0.2, lock: threading.Lock | None = None, log_dir: Path | None = None,
                 save_audio: bool = False, partial_interval: float = 0.0):
        """models: {key: AsrModel}，按顺序依次转写；lock：与其他推理共用的锁（保证计时不被抢 CPU）。

        log_dir：chunk / partial / text 事件追加到 <log_dir>/<时间>.jsonl。
        save_audio：同时把收到的原始音频存为同名 .wav（16kHz 单声道 Int16，第 0 秒 = 会话开始，
        与事件里的 start / end 对齐，可以拿来换参数重跑）。需要同时给 log_dir。
        partial_interval：说话过程中每隔多少秒（音频时间）出一次临时结果；0 = 不出。
        每次都把这句从头重新识别（离线模型没法接着上次往下算），一句话越长每次越慢，总计算量随句长平方增长。
        正式结果优先：转写线程有正式结果要做时，临时结果直接作废；排队的临时结果只留最新的一份。
        """
        self.models = models
        self.on_event = on_event
        self.min_silence = min_silence
        self.max_chunk = max_chunk
        self.pad = pad
        self.partial_interval = max(0.0, float(partial_interval))
        self.lock = lock or threading.Lock()
        self.vad = asr_engine.new_vad(min_silence=min_silence, max_speech=max_chunk,
                                      buffer_sec=max(60.0, max_chunk * 3))
        if self.vad is None:
            raise RuntimeError("没有 VAD 模型：python download_models.py")
        # 环形缓冲：保留最近 keep_sec 秒音频，buf_start 是 buf[0] 的绝对采样位置
        self.keep = int(SR * max(60.0, max_chunk * 3))
        self.buf = np.zeros(0, np.float32)
        self.buf_start = 0
        self.received = 0          # 已收到的总采样数（= 音频时钟）
        self._rest = np.zeros(0, np.float32)
        self._peak = 0.0           # VAD 自动增益：最近几秒的峰值（半衰期约 5s）
        self.gain = 1.0
        self._speaking = False
        self._speech_from = 0
        self._last_tick = 0
        self._last_partial = 0
        self.n_chunks = 0
        # 转写线程的待办：正式结果按顺序全部做；临时结果只有一个槽位，新的覆盖旧的，没有正式结果要做时才做
        self._cv = threading.Condition()
        self._finals: deque = deque()
        self._partial = None
        self._closing = False
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()
        self._emit_lock = threading.Lock()
        self._log = None
        self._wav = None
        self.audio_path: Path | None = None
        if log_dir is not None:
            log_dir.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%d-%H%M%S")
            stem, n = log_dir / stamp, 1
            while stem.with_suffix(".jsonl").exists() or stem.with_suffix(".wav").exists():  # 同一秒内开了两次
                n += 1
                stem = log_dir / f"{stamp}-{n}"
            self._log = open(stem.with_suffix(".jsonl"), "a", encoding="utf-8")
            if save_audio:
                self.audio_path = stem.with_suffix(".wav")
                self._wav = wave.open(str(self.audio_path), "wb")  # wave 每次写入都会补写头，中途崩溃文件也可读
                self._wav.setnchannels(1)
                self._wav.setsampwidth(2)
                self._wav.setframerate(SR)

    # ------------------------------------------------------------ 输入
    def accept(self, pcm: np.ndarray) -> None:
        """喂入 16kHz 单声道 float32 音频，长度任意。"""
        pcm = np.asarray(pcm, dtype=np.float32)
        if self._wav is not None and len(pcm):
            # 存送给 ASR 的原始音频（不含下面 VAD 的自动增益）；服务端收的是 Int16 / 32768，这里能无损还原
            self._wav.writeframes(np.clip(np.round(pcm * 32768.0), -32768, 32767).astype("<i2").tobytes())
        self.buf = np.concatenate([self.buf, pcm])
        if len(self.buf) > self.keep:
            cut = len(self.buf) - self.keep
            self.buf = self.buf[cut:]
            self.buf_start += cut
        # 自动增益只作用于 VAD 输入：轻声/远场时 silero 的人声概率贴近阈值，会把整句判丢
        # （实测 -40dBFS 的一句话在 Int16 输入下被丢掉，放大 1.3 倍即恢复；纯噪声放大 10 倍也不误触发）
        if len(pcm):
            self._peak = max(float(np.abs(pcm).max()), self._peak * 0.5 ** (len(pcm) / SR / 5.0))
            self.gain = float(np.clip(0.7 / max(self._peak, 0.07), 1.0, 10.0))
        x = np.concatenate([self._rest, np.clip(pcm * self.gain, -1.0, 1.0)])
        n = len(x) // WIN * WIN
        for i in range(0, n, WIN):
            self.vad.accept_waveform(x[i:i + WIN])
            self.received += WIN
            self._poll()
        self._rest = x[n:]

    def close(self, timeout: float = 60.0) -> None:
        """结束：把正在说的最后一句冲出来，等所有转写完成。"""
        self.vad.flush()
        self._poll(final=True)
        with self._cv:
            self._closing = True
            self._partial = None   # 最后一句的正式结果马上就出，临时结果不用再做
            self._cv.notify()
        self._worker.join(timeout)
        if self._log:
            self._log.close()
        if self._wav is not None:
            self._wav.close()
            self._wav = None

    # ------------------------------------------------------------ VAD 状态 → chunk
    def _poll(self, final: bool = False) -> None:
        speaking = self.vad.is_speech_detected()
        t = self.received / SR
        if speaking and not self._speaking:
            self._speech_from = self.received
            self._last_tick = self._last_partial = self.received
            self._emit({"type": "speech_start", "t": round(t, 2)})
        elif speaking and self.received - self._last_tick >= SR // 2:
            self._last_tick = self.received
            self._emit({"type": "speaking", "t": round(t, 2),
                        "dur": round((self.received - self._speech_from) / SR, 1)})
        if self._speaking and not speaking:
            self._emit({"type": "speech_end", "t": round(t, 2)})
        self._speaking = speaking

        while not self.vad.empty():
            seg = self.vad.front
            start, n = int(seg.start), len(seg.samples)
            self.vad.pop()
            end = start + n
            lo = max(start - int(self.pad * SR), self.buf_start)
            hi = min(end + int(self.pad * SR), self.buf_start + len(self.buf))
            audio = self.buf[lo - self.buf_start:hi - self.buf_start].copy()
            if len(audio) < SR // 5:
                continue
            dur = n / SR
            self.n_chunks += 1
            ch = {"type": "chunk", "id": self.n_chunks, "start": round(start / SR, 2), "end": round(end / SR, 2),
                  "dur": round(dur, 2), "reason": "max_len" if dur >= self.max_chunk - 0.05 else "silence",
                  # 语音结束到 VAD 判停之间等了多久（音频时钟；flush 时为 0）
                  "endpoint_wait_ms": 0.0 if final else round(max(0, self.received - end) / SR * 1000, 1)}
            self._emit(ch)
            with self._cv:
                self._finals.append((ch, audio, time.perf_counter()))
                self._partial = None   # 槽位里还没开始的临时结果就是刚切出的这句，作废
                self._cv.notify()
            self._last_partial = self.received  # 超长被切、人还在说时，下一句的临时结果从现在重新计时

        if (not final and speaking and self.partial_interval > 0
                and self.received - self._last_partial >= self.partial_interval * SR):
            self._last_partial = self.received
            self._queue_partial()

    def _segment_start(self) -> int:
        """VAD 正在攒的这一段从哪个采样开始（和切出来的 chunk 的 start 一致）。"""
        cs = getattr(self.vad, "current_segment", None)
        if cs is not None and cs.start >= 0:
            return int(cs.start)
        # 老版本 sherpa-onnx 没有 current_segment：VAD 确认有人说话时，把段起点定在
        # 当前位置 − 2 个窗口 − min_speech_duration（asr_engine.new_vad 里是 0.25s）
        return self._speech_from - 2 * WIN - SR // 4

    def _queue_partial(self) -> None:
        """把正在说的这句（VAD 当前段的起点 → 已收到的最新音频）放进临时结果槽位。"""
        start = self._segment_start()
        lo = max(start - int(self.pad * SR), self.buf_start)   # 和正式结果一样前面补 pad
        hi = self.buf_start + len(self.buf)
        if hi - lo < SR // 2:
            return
        audio = self.buf[lo - self.buf_start:].copy()
        with self._cv:
            self._partial = (self.n_chunks + 1, audio, start, hi, time.perf_counter())
            self._cv.notify()

    # ------------------------------------------------------------ 转写线程
    def _run(self) -> None:
        while True:
            with self._cv:
                while not (self._finals or self._partial is not None or self._closing):
                    self._cv.wait()
                if self._finals:
                    job, is_final = self._finals.popleft(), True
                elif self._partial is not None:
                    job, is_final, self._partial = self._partial, False, None
                else:
                    return   # 已 close，正式结果都做完了
            if is_final:
                self._transcribe(*job)
            else:
                self._transcribe_partial(*job)

    def _transcribe_partial(self, cid: int, audio: np.ndarray, start: int, end: int, t_job: float) -> None:
        for key, m in self.models.items():
            # 这句已经切出来了，或者有正式结果在排队：让正式结果先做，这次临时结果作废
            if cid <= self.n_chunks or self._finals:
                return
            try:
                with self.lock:
                    t0 = time.perf_counter()
                    res = m._offline_stream(m.recognizer, SR, audio)
                    dec = (time.perf_counter() - t0) * 1000
            except Exception:  # noqa: BLE001 —— 临时结果出错不报：正式结果还会再转写一遍
                continue
            self._emit({"type": "partial", "chunk_id": cid, "model": key, "text": res["text"],
                        "start": round(start / SR, 2), "end": round(end / SR, 2), "decode_ms": round(dec, 1),
                        "lag_ms": round((time.perf_counter() - t_job) * 1000, 1)})

    def _transcribe(self, ch: dict, audio: np.ndarray, t_chunk: float) -> None:
        for key, m in self.models.items():
            try:
                with self.lock:
                    t0 = time.perf_counter()
                    res = m._offline_stream(m.recognizer, SR, audio)
                    dec = (time.perf_counter() - t0) * 1000
                ev = {"type": "text", "chunk_id": ch["id"], "model": key, "text": res["text"],
                      "start": ch["start"], "end": ch["end"],
                      "decode_ms": round(dec, 1), "rtf": round(dec / 1000 / max(len(audio) / SR, 1e-3), 4),
                      "latency_ms": round(ch["endpoint_wait_ms"] + dec, 1),
                      "shown_ms": round(ch["endpoint_wait_ms"] + (time.perf_counter() - t_chunk) * 1000, 1)}
                for k in ("lang", "emotion", "event"):
                    if res.get(k):
                        ev[k] = res[k]
            except Exception as e:  # noqa: BLE001 —— 单个模型出错不影响另一个
                ev = {"type": "error", "chunk_id": ch["id"], "model": key, "message": f"{type(e).__name__}: {e}"}
            self._emit(ev)

    def _emit(self, ev: dict) -> None:
        # chunk 事件来自调用 accept 的线程，partial / text 事件来自转写线程，串行化
        with self._emit_lock:
            if self._log and ev["type"] in ("chunk", "partial", "text"):
                self._log.write(json.dumps(ev, ensure_ascii=False) + "\n")
                self._log.flush()
            try:
                self.on_event(ev)
            except Exception:  # noqa: BLE001 —— 回调出错（比如网页断开）不能拖垮会话
                pass
