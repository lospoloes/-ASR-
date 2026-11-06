"""持续监听：流式 VAD 切句 → 同一句分别交给每个 ASR 模型转写 → 事件回调。

和网页无关，可以直接搬到端侧常驻链路里用：
    sess = LiveSession(models, on_event=print, partial_interval=0.4)
    while True:
        sess.accept(pcm_float32_16k)     # 麦克风来一块喂一块，任意长度
    sess.close()                         # 结束时把最后一句冲出来

实时出字分三层（partial_interval > 0 时）：
    蹦字      说话过程中每隔 partial_interval 秒，把【当前这一段】从段起点重新识别一遍（partial，kind=live）
    段锁定    VAD 在句内停顿（≥ piece_gap 秒）或段长到 piece_max 秒时切开一段，整段识别一次，之后不再变（kind=lock）
    正式结果  VAD 判定这句说完（静音 ≥ min_silence）或句长到 max_chunk，整句重新识别一次，替换这句的全部临时文字（text）
离线模型没法接着上次往下算，每次都得从头识别；只重识别当前这一段（不是整句），每次识别的音频就不超过 piece_max 秒。

事件（dict，type 字段区分；时间都是会话音频时间，单位秒）:
    speech_start  {t, start, prob}        有人开始说话。t：VAD 确认的时刻；start：往回推的开口时刻；prob：这一帧的人声概率
    speaking      {t, dur, prob}          正在说话，每 0.5s 一次（dur：这句已说了多久；prob：最近 0.5s 的平均人声概率）
    speech_end    {t}                     VAD 判定说完了（静音够长，或会话结束）
    cut           {t, chunk_id, piece, at, reason}
                  句内切段：第 piece 段到 at 为止，下一段从 at 开始。reason：pause（停顿）/ max_len（段太长被切）。
                  chunk_id 是这句切出来以后会用的编号
    chunk         {id, start, end, dur, reason, endpoint_wait_ms}
                  切出一句。reason：silence（静音够长）/ max_len（超长被切，人还在说）/ flush（会话结束时还没说完）
                  endpoint_wait_ms：语音结束到 VAD 判停之间等的静音（音频时间）
    partial       {chunk_id, model, kind, rev, piece, piece_start, start, end, text, locked, live, audio, decode_ms, lag_ms}
                  临时结果（partial_interval > 0 才有）。text = locked + live：locked 是这句已锁定各段的文字，
                  live 是当前这段的最新识别（kind=lock 时为空）；start → end 是这句已经识别到的范围；
                  audio：这次实际送去识别的音频有几秒。
                  同一句同一模型 rev 递增，新的整体替换旧的；同一句的 partial 一定排在它的 text 前面，text 到了全部替换掉。
                  lag_ms：从这段音频到手到出临时结果用了多久
    text          {chunk_id, model, text, decode_ms, rtf, latency_ms, shown_ms, ...}
                  某个模型转写完一句（正式结果）。
                  latency_ms = endpoint_wait_ms + decode_ms：该模型【单独部署】时，从说完到出字的延迟
                  shown_ms：本次实际出字用时（两个模型排队跑，后跑的会多等前一个的解码；开了实时出字时，
                  还可能要等手上那次临时识别做完）
                  owner：配置了声纹注册时才有——true 主人 / false 陌生人 / null 判不了（<3 秒或声纹失败）；
                  转写内容不受影响
    error         {message}

两个模型拿到的是完全相同的音频，各自独立转写，便于公平对比。
"""
from __future__ import annotations

import json
import threading
import time
import wave
from collections import deque
from pathlib import Path

import numpy as np

import vad_stream
import speaker_owner

SR = vad_stream.SR
WIN = vad_stream.WIN  # silero VAD 按 512 采样（32ms）一帧
ROOT = Path(__file__).resolve().parent


def join_text(parts) -> str:
    """把几段文字接起来：前后都是英文字母 / 数字时补一个空格，中文直接接。"""
    out = ""
    for p in parts:
        if not p:
            continue
        if out and out[-1].isascii() and out[-1].isalnum() and p[0].isascii() and p[0].isalnum():
            out += " "
        out += p
    return out


class LiveSession:
    def __init__(self, models: dict, on_event, min_silence: float = 0.5, max_chunk: float = 15.0,
                 pad: float = 0.2, lock: threading.Lock | None = None, log_dir: Path | None = None,
                 save_audio: bool = False, partial_interval: float = 0.0, piece_gap: float = 0.2,
                 piece_max: float = 8.0, min_speech: float = 0.15, neg_threshold: float = 0.5,
                 owner_profile: "speaker_owner.OwnerProfile | None" = None,
                 owner_model=None, owner_threshold: float = speaker_owner.DIALOG_THRESHOLD,
                 owner_diarizer=None):
        """models: {key: AsrModel}，按顺序依次转写；lock：与其他推理共用的锁（保证计时不被抢 CPU）。

        min_silence：静音多久算一句说完；max_chunk：一句最长多少秒，到了就硬切（切在最近 2 秒里人声概率最低的地方）。
        log_dir：cut / chunk / partial / text 事件追加到 <log_dir>/<时间>.jsonl。
        save_audio：同时把收到的原始音频存为同名 .wav（16kHz 单声道 Int16，第 0 秒 = 会话开始，
        与事件里的时间对齐，可以拿来换参数重跑）。需要同时给 log_dir。
        partial_interval：说话过程中每隔多少秒（音频时间）把当前这一段重新识别一次；0 = 不出临时结果。
        piece_gap / piece_max：句内停顿多久切段 / 一段最长多少秒。只决定临时结果怎么分段，不影响正式结果。
        min_speech / neg_threshold：人声概率 > 0.5 持续多久算开口 / 开口后概率降到多少以下开始计静音。
        默认 0.15s / 0.5 是 tools/eval_vad.py 在 94 分钟真人对话上定的（sherpa 默认 0.25s / 0.35：
        开口晚约 0.1 秒、短句漏得多；有底噪的房间里判停晚、相邻两句容易连成一句）。
        转写线程的优先级：正式结果 > 段锁定 > 蹦字；蹦字只留最新的一份，过期就作废。
        """
        self.models = models
        self.on_event = on_event
        self.min_silence = min_silence
        self.max_chunk = max_chunk
        self.pad = pad
        self.partial_interval = max(0.0, float(partial_interval))
        self.piece_gap = piece_gap
        self.piece_max = piece_max
        self.lock = lock or threading.Lock()
        self.min_speech = min_speech
        self.neg_threshold = neg_threshold
        # 声纹主人识别：给了注册向量就必须给声纹模型；都没给 = 不判（text 事件无 owner 字段）
        self.owner_profile = owner_profile
        self.owner_model = owner_model
        self.owner_threshold = owner_threshold
        # 句内分角色段累积：Diarizer 段攒秒数，攒够 MIN_SEC 判一次后清零
        self.owner_acc: list = []
        if owner_profile is not None and owner_model is None:
            raise ValueError("owner_profile 需要同时给 owner_model（speaker_owner.load_model 的返回值）")
        if owner_profile is not None and owner_diarizer is None:
            raise ValueError("owner_profile 需要同时给 owner_diarizer（asr_engine.Diarizer 实例，句内分角色段用）")
        self.owner_diarizer = owner_diarizer
        self.vad = vad_stream.SileroStream(min_silence=min_silence, sentence_max=max_chunk,
                                           piece_gap=piece_gap, piece_max=piece_max,
                                           min_speech=min_speech, neg_threshold=neg_threshold)
        # 环形缓冲：保留最近 keep 个采样，buf_start 是 buf[0] 的绝对采样位置
        self.keep = int(SR * max(60.0, max_chunk * 3))
        self.buf = np.zeros(0, np.float32)
        self.buf_start = 0
        self.received = 0          # VAD 已处理的采样数（= 音频时钟）
        self._probs: deque = deque(maxlen=SR // 2 // WIN)   # 最近 0.5s 每帧的人声概率
        self._speech_from = 0
        self._last_tick = 0
        self._last_partial = 0
        # 正在说的这句（还没切出来）：只在调用 accept 的线程里改
        self._sent_start = 0       # 这句的起点
        self._sent_lo = 0          # 这句送去识别从哪里开始（开口处往前补 pad）
        self._piece = 0            # 当前是第几段
        self._piece_start = 0
        self._piece_lo = 0         # 当前这段送去识别从哪里开始
        self.n_chunks = 0
        # 转写线程的待办：正式结果、段锁定按顺序全部做；蹦字只有一个槽位，新的覆盖旧的，前两样都没有时才做
        self._cv = threading.Condition()
        self._finals: deque = deque()
        self._locks: deque = deque()
        self._partial = None
        self._closing = False
        # (chunk_id, 模型) → 这句在这个模型上的临时文字 {"locked": {段: 文字}, "live": {段: 文字}, "rev": n}；只有转写线程用
        self._drafts: dict = {}
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
            # 存送给 ASR 的原始音频（不含 VAD 的自动增益）；服务端收的是 Int16 / 32768，这里能无损还原
            self._wav.writeframes(np.clip(np.round(pcm * 32768.0), -32768, 32767).astype("<i2").tobytes())
        self.buf = np.concatenate([self.buf, pcm])
        if len(self.buf) > self.keep:
            cut = len(self.buf) - self.keep
            self.buf = self.buf[cut:]
            self.buf_start += cut
        for fr in self.vad.accept(pcm):    # 自动增益在 VAD 里，只作用于 VAD 的输入
            self.received = fr.end
            self._probs.append(fr.prob)
            for e in fr.events:
                self._on_vad(e)
            self._tick()

    def close(self, timeout: float = 60.0) -> None:
        """结束：把正在说的最后一句冲出来，等所有转写完成。"""
        for e in self.vad.flush():
            self._on_vad(e)
        with self._cv:
            self._closing = True
            self._partial = None   # 最后一句的正式结果马上就出，临时结果不用再做
            self._locks.clear()
            self._cv.notify()
        self._worker.join(timeout)
        if self._log:
            self._log.close()
        if self._wav is not None:
            self._wav.close()
            self._wav = None

    # ------------------------------------------------------------ VAD 事件 → 句 / 段
    def _on_vad(self, e: vad_stream.Event) -> None:
        t = round(self.received / SR, 2)
        if e.kind == "start":
            if e.reason == "speech":
                self._speech_from = self._last_tick = self.received
                self._emit({"type": "speech_start", "t": t, "start": round(e.at / SR, 2),
                            "prob": round(self._probs[-1], 3)})
                lo = e.at - int(self.pad * SR)
            else:
                lo = e.at   # 上一句超长被切开、人还在说：切点两边都不补 pad，免得切点附近的字被两句各识别一次
            self._sent_start = e.at
            self._sent_lo = lo
            self._new_piece(1, e.at, lo)
        elif e.kind == "cut":
            cid = self.n_chunks + 1
            self._emit({"type": "cut", "t": t, "chunk_id": cid, "piece": e.piece, "at": round(e.at / SR, 2),
                        "reason": e.reason})
            if self.partial_interval > 0:
                job = (cid, e.piece, self._sent_start, self._piece_start, e.hi, self._slice(self._piece_lo, e.hi),
                       time.perf_counter(), tuple(self.models))
                with self._cv:
                    self._locks.append(job)
                    self._partial = None   # 槽位里那份蹦字识别的就是刚切下来这段，锁定结果马上就来，作废
                    self._cv.notify()
            self._new_piece(e.piece + 1, e.at, e.at)
        else:   # end
            if e.reason != "max_len":
                self._emit({"type": "speech_end", "t": t})
            self._emit_chunk(e)

    def _new_piece(self, n: int, start: int, lo: int) -> None:
        self._piece, self._piece_start, self._piece_lo = n, start, lo
        self._last_partial = start   # 从段起点开始计时：第一段在 VAD 确认开口时已经过去约 0.3 秒

    def _slice(self, lo: int, hi: int) -> np.ndarray:
        lo = max(lo, self.buf_start)
        hi = min(hi, self.buf_start + len(self.buf))
        return self.buf[lo - self.buf_start:max(lo, hi) - self.buf_start].copy()

    def _emit_chunk(self, e: vad_stream.Event) -> None:
        start, end = e.start, e.at
        hi = end if e.reason == "max_len" else end + int(self.pad * SR)   # 超长硬切的切点后面是下一句，不补 pad
        audio = self._slice(self._sent_lo, hi)
        if len(audio) < SR // 5:
            return
        self.n_chunks += 1
        ch = {"type": "chunk", "id": self.n_chunks, "start": round(start / SR, 2), "end": round(end / SR, 2),
              "dur": round((end - start) / SR, 2), "reason": e.reason,
              # 语音结束到 VAD 判停之间等了多久（音频时钟；会话结束时为 0）
              "endpoint_wait_ms": 0.0 if e.reason == "flush" else round(max(0, self.received - end) / SR * 1000, 1)}
        self._emit(ch)
        with self._cv:
            self._finals.append((ch, audio, time.perf_counter()))
            self._partial = None   # 槽位里还没开始的蹦字就是刚切出的这句，作废
            self._cv.notify()

    def _tick(self) -> None:
        seg = self.vad.seg
        if not seg.in_speech:
            return
        if self.received - self._last_tick >= SR // 2:
            self._last_tick = self.received
            self._emit({"type": "speaking", "t": round(self.received / SR, 2),
                        "dur": round((self.received - self._speech_from) / SR, 1),
                        "prob": round(float(np.mean(self._probs)), 3)})
        if (self.partial_interval > 0 and seg.voiced_from != -1
                and self.received - seg.voiced_from >= SR // 5              # 这段已经有 0.2 秒人声
                and self.received - self._last_partial >= self.partial_interval * SR):
            self._queue_partial()

    def _queue_partial(self) -> None:
        """把当前这一段（段起点 → 已收到的最新音频）放进蹦字槽位。"""
        lo = max(self._piece_lo, self.buf_start)
        hi = self.buf_start + len(self.buf)
        if hi - lo < SR // 2:
            return
        self._last_partial = self.received
        job = (self.n_chunks + 1, self._piece, self._sent_start, self._piece_start, hi,
               self.buf[lo - self.buf_start:].copy(), time.perf_counter(), tuple(self.models))
        with self._cv:
            self._partial = job
            self._cv.notify()

    # ------------------------------------------------------------ 转写线程
    def _run(self) -> None:
        while True:
            with self._cv:
                while not (self._finals or self._locks or self._partial is not None or self._closing):
                    self._cv.wait()
                if self._finals:
                    kind, job = "final", self._finals.popleft()
                elif self._locks:
                    kind, job = "lock", self._locks.popleft()
                elif self._partial is not None:
                    kind, job, self._partial = "live", self._partial, None
                else:
                    return   # 已 close，正式结果都做完了
            if kind == "final":
                self._transcribe(*job)
            else:
                self._transcribe_piece(kind, *job)

    def _transcribe_piece(self, kind: str, cid: int, piece: int, sent_start: int, piece_start: int, hi: int,
                          audio: np.ndarray, t_job: float, keys: tuple) -> None:
        for i, key in enumerate(keys):
            if cid <= self.n_chunks:
                return   # 这句已经切出来了：正式结果会整句替换，临时的不用做了
            if self._finals or (kind == "live" and (self._locks or piece < self._piece)):
                # 有更要紧的活：蹦字直接作废（下一份很快就来）；段锁定把还没做的模型放回队头，先去做正式结果
                if kind == "lock":
                    with self._cv:
                        self._locks.appendleft((cid, piece, sent_start, piece_start, hi, audio, t_job, keys[i:]))
                return
            m = self.models[key]
            try:
                with self.lock:
                    t0 = time.perf_counter()
                    res = m._offline_stream(m.recognizer, SR, audio)
                    dec = (time.perf_counter() - t0) * 1000
                text = res["text"]
            except Exception:  # noqa: BLE001 —— 临时结果出错不报：正式结果还会再转写一遍
                if kind == "live":
                    continue
                text, dec = None, 0.0
            d = self._drafts.setdefault((cid, key), {"locked": {}, "live": {}, "rev": 0})
            if kind == "lock":
                # 锁定识别出错时，用这段最后一次蹦字顶上
                d["locked"][piece] = text if text is not None else d["live"].get(piece, "")
                live, upto = "", piece
            else:
                d["live"][piece] = live = text
                upto = piece - 1
            locked = join_text(d["locked"].get(p, d["live"].get(p, "")) for p in range(1, upto + 1))
            d["rev"] += 1
            self._emit({"type": "partial", "chunk_id": cid, "model": key, "kind": kind, "rev": d["rev"],
                        "piece": piece, "piece_start": round(piece_start / SR, 2), "start": round(sent_start / SR, 2),
                        "end": round(hi / SR, 2), "text": join_text([locked, live]), "locked": locked, "live": live,
                        "audio": round(len(audio) / SR, 2), "decode_ms": round(dec, 1),
                        "lag_ms": round((time.perf_counter() - t_job) * 1000, 1)})

    def _transcribe(self, ch: dict, audio: np.ndarray, t_chunk: float) -> None:
        for key, m in self.models.items():
            self._drafts.pop((ch["id"], key), None)
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
                # 声纹主人识别：只加 owner 字段，转写内容不动。判定失败保持 null，不影响转写事件。
                # 句内分角色段累积：Diarizer 把整句切说话人段，挑与主人声纹最像的段（主段）；
                # 主段向量按秒数加权攒进 owner_acc（跨句累积），攒够 MIN_SEC 才判。判过的秒数清零重新攒。
                if self.owner_profile is not None:
                    ev["owner"] = None
                    try:
                        with self.lock:
                            segs, _ = self.owner_diarizer.run(audio, num_speakers=-1)
                        cands = []
                        for s in segs:
                            dur = s["end"] - s["start"]
                            if dur < 0.5:
                                continue   # 段太短不可靠
                            seg_audio = audio[int(s["start"] * SR):int(s["end"] * SR)]
                            cands.append((speaker_owner.embed(seg_audio, self.owner_model), dur))
                        if cands:
                            emb, sec = max(cands, key=lambda c: speaker_owner.score(self.owner_profile, c[0]))
                            self.owner_acc.append((emb, sec))
                            acc = sum(s for _, s in self.owner_acc)
                            if acc >= speaker_owner.MIN_SEC:
                                mean = np.mean([e * s for e, s in self.owner_acc], axis=0)
                                mean /= np.linalg.norm(mean) + 1e-9
                                verdict = speaker_owner.decide(self.owner_profile, mean,
                                                               acc, self.owner_threshold)
                                ev["owner"] = None if verdict == "unknown" else verdict == "owner"
                                self.owner_acc.clear()
                    except Exception:  # noqa: BLE001
                        pass
                for k in ("lang", "emotion", "event"):
                    if res.get(k):
                        ev[k] = res[k]
            except Exception as e:  # noqa: BLE001 —— 单个模型出错不影响另一个
                ev = {"type": "error", "chunk_id": ch["id"], "model": key, "message": f"{type(e).__name__}: {e}"}
            self._emit(ev)

    def _emit(self, ev: dict) -> None:
        # speech / cut / chunk 事件来自调用 accept 的线程，partial / text 事件来自转写线程，串行化
        with self._emit_lock:
            if self._log and ev["type"] in ("cut", "chunk", "partial", "text"):
                self._log.write(json.dumps(ev, ensure_ascii=False) + "\n")
                self._log.flush()
            try:
                self.on_event(ev)
            except Exception:  # noqa: BLE001 —— 回调出错（比如网页断开）不能拖垮会话
                pass
