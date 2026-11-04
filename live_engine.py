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
    text          {chunk_id, model, text, decode_ms, rtf, latency_ms, shown_ms, ...}
                  某个模型转写完一个 chunk。
                  latency_ms = endpoint_wait_ms + decode_ms：该模型【单独部署】时，从说完到出字的延迟
                  shown_ms：本次实际出字用时（两个模型排队跑，后跑的会多等前一个的解码）
    error         {message}

两个模型拿到的是完全相同的 chunk，各自独立转写，便于公平对比。
"""
from __future__ import annotations

import json
import queue
import threading
import time
from pathlib import Path

import numpy as np

import asr_engine

SR = asr_engine.SAMPLE_RATE
WIN = 512  # silero VAD 要求按 512 采样（32ms）精确喂入
ROOT = Path(__file__).resolve().parent


class LiveSession:
    def __init__(self, models: dict, on_event, min_silence: float = 0.5, max_chunk: float = 15.0,
                 pad: float = 0.2, lock: threading.Lock | None = None, log_dir: Path | None = None):
        """models: {key: AsrModel}，按顺序依次转写；lock：与其他推理共用的锁（保证计时不被抢 CPU）。"""
        self.models = models
        self.on_event = on_event
        self.min_silence = min_silence
        self.max_chunk = max_chunk
        self.pad = pad
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
        self.n_chunks = 0
        self._q: queue.Queue = queue.Queue()
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()
        self._emit_lock = threading.Lock()
        self._log = None
        if log_dir is not None:
            log_dir.mkdir(parents=True, exist_ok=True)
            self._log = open(log_dir / time.strftime("%Y%m%d-%H%M%S.jsonl"), "a", encoding="utf-8")

    # ------------------------------------------------------------ 输入
    def accept(self, pcm: np.ndarray) -> None:
        """喂入 16kHz 单声道 float32 音频，长度任意。"""
        pcm = np.asarray(pcm, dtype=np.float32)
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
        self._q.put(None)
        self._worker.join(timeout)
        if self._log:
            self._log.close()

    # ------------------------------------------------------------ VAD 状态 → chunk
    def _poll(self, final: bool = False) -> None:
        speaking = self.vad.is_speech_detected()
        t = self.received / SR
        if speaking and not self._speaking:
            self._speech_from = self.received
            self._last_tick = self.received
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
            self._q.put((ch, audio, time.perf_counter()))

    # ------------------------------------------------------------ 转写线程
    def _run(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                return
            ch, audio, t_chunk = item
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
        # chunk 事件来自调用 accept 的线程，text 事件来自转写线程，串行化
        with self._emit_lock:
            if self._log and ev["type"] in ("chunk", "text"):
                self._log.write(json.dumps(ev, ensure_ascii=False) + "\n")
                self._log.flush()
            try:
                self.on_event(ev)
            except Exception:  # noqa: BLE001 —— 回调出错（比如网页断开）不能拖垮会话
                pass
