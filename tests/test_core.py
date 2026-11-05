"""核心逻辑单测（不需要模型）：python -m unittest discover -s tests"""
import functools
import io
import sys
import tempfile
import time
import unittest
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import asr_engine  # noqa: E402
import registry  # noqa: E402


def make_wav(x, sr=16000, ch=1):
    buf = io.BytesIO()
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2")
    if ch == 2:
        pcm = np.repeat(pcm, 2)
    with wave.open(buf, "wb") as w:
        w.setnchannels(ch)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


class TestText(unittest.TestCase):
    def test_numbers(self):
        cases = {"早上9点": "早上九点", "123000步": "十二万三千步", "305个": "三百零五个", "110": "一百一十",
                 "2026年": "二零二六年", "13800138000": "一三八零零一三八零零零", "3.5公里": "三点五公里",
                 "30%": "百分之三十", "10": "十", "1005": "一千零五", "0": "零"}
        for a, b in cases.items():
            self.assertEqual(asr_engine.arabic_to_zh(a), b, a)

    def test_cer(self):
        self.assertEqual(asr_engine.cer("开放时间早上九点至下午五点", "开放时间早上9点至下午5点。"), 0.0)
        self.assertAlmostEqual(asr_engine.cer("今天天气", "今天天汽"), 0.25)
        self.assertIsNone(asr_engine.cer("，。", "abc"))
        self.assertEqual(asr_engine.cer("Hello World", "hello，world"), 0.0)
        self.assertGreater(asr_engine.cer("早上九点", "早上9点", norm_numbers=False), 0)


class TestAudio(unittest.TestCase):
    def test_read_resample(self):
        x = np.sin(np.arange(44100) / 44100 * 2 * np.pi * 440).astype(np.float32) * 0.5
        y, sr = asr_engine.read_wav(make_wav(x, 44100, ch=2))
        self.assertEqual(sr, 44100)
        self.assertEqual(len(y), 44100)
        z = asr_engine.resample(y, sr, 16000)
        self.assertEqual(len(z), 16000)
        self.assertLess(abs(np.abs(z).max() - 0.5), 0.01)


class TestVad(unittest.TestCase):
    def test_two_segments(self):
        if not registry.resolve_files(registry.get_spec("silero-vad")):
            self.skipTest("未下载 silero-vad（python download_models.py --models silero-vad）")
        # 用仓库里的 TTS 测试音频：语音 + 1s 静音 + 语音，应切出 2 段
        d = Path(__file__).resolve().parents[1] / "benchmark" / "testset"
        a, _ = asr_engine.read_wav((d / "s01_cmd.wav").read_bytes())
        b, _ = asr_engine.read_wav((d / "s13_weather.wav").read_bytes())
        x = np.concatenate([a, np.zeros(16000, np.float32), b])
        segs = asr_engine.vad_segments(x)
        self.assertEqual(len(segs), 2, segs)
        self.assertLess(segs[0][1], len(a) / 16000 + 0.3)
        self.assertGreater(segs[1][0], len(a) / 16000 + 0.5)


class TestSemantic(unittest.TestCase):
    def test_completeness(self):
        import semantic_seg as ss
        cases = {"我想。": "incomplete", "如果明天下雨的话。": "incomplete", "因为孩子发烧了。": "incomplete",
                 "我今天去超市，": "incomplete", "那个": "incomplete", "你知道为什么天空是蓝色的吗": "complete",
                 "如果明天下雨的话，就提醒我带伞。": "complete", "晚安": "complete", "今天天气不错": "uncertain"}
        for t, want in cases.items():
            self.assertEqual(ss.completeness(t), want, t)
        self.assertTrue(ss.continues("然后买了水果"))
        self.assertTrue(ss.continues("嗯，所以呢"))
        self.assertFalse(ss.continues("再见"))
        self.assertFalse(ss.continues("明天几点出发"))

    def test_segment_merges_by_meaning(self):
        import semantic_seg as ss
        # 4 个片段：「我想。」| 0.5s |「明天去公园。」| 0.5s |「今天好累啊」| 0.4s |「然后想睡觉」
        frags = [(0.0, 1.0), (1.5, 3.0), (3.5, 4.5), (4.9, 6.0)]
        texts = iter(["我想。", "明天去公园。", "今天好累啊。", "然后想睡觉。"])
        utts, _ = ss.segment(np.zeros(16000 * 7, np.float32), frags, lambda x: next(texts))
        self.assertEqual([u.frag_ids for u in utts], [[0, 1], [2, 3]])
        self.assertEqual(utts[1].rejoined, 1)
        # 纯声学 0.25s 阈值会切成 4 段
        self.assertEqual(len(ss.fixed_gap_merge(frags, 0.25)), 4)


TESTSET = Path(__file__).resolve().parents[1] / "benchmark" / "testset"


def _sentences(*names, gap=1.0):
    """把测试句用 gap 秒静音连起来（最后一句后面也补静音，让它正常判停）。"""
    parts = []
    for n in names:
        x, _ = asr_engine.read_wav((TESTSET / f"{n}.wav").read_bytes())
        parts += [x, np.zeros(int(16000 * gap), np.float32)]
    return np.concatenate(parts)


@functools.lru_cache(maxsize=1)
def _live_models():
    """持续监听测试用的真模型（只加载一次）；没下载返回 None。"""
    downloaded = asr_engine.find_downloaded()
    if not asr_engine.vad_model_path() or "paraformer-zh-small" not in downloaded:
        return None
    models = {k: asr_engine.AsrModel(registry.get_spec(k), downloaded[k])
              for k in ("sensevoice", "paraformer-zh-small") if k in downloaded}
    for m in models.values():
        m.load()
    return models


def _feed(sess, x, pace=0.0):
    """模拟 0.1s 一块的流式输入；pace：每块之后睡多久（0 = 按最快速度喂）。"""
    for i in range(0, len(x), 1600):
        sess.accept(x[i:i + 1600])
        if pace:
            time.sleep(pace)
    sess.close()


class _SlowFake:
    """假模型：每次解码固定睡 delay 秒，返回音频时长。用来稳定地制造“这句切出来时临时结果还在算”的竞争。"""
    recognizer = None

    def __init__(self, delay):
        self.delay = delay

    def _offline_stream(self, rec, sr, audio):
        time.sleep(self.delay)
        return {"text": f"{len(audio) / sr:.2f}s"}


class _Recorder:
    """假模型：立刻返回，记下每次拿到的音频长度（采样数）。"""
    recognizer = None

    def __init__(self):
        self.lengths = []

    def _offline_stream(self, rec, sr, audio):
        self.lengths.append(len(audio))
        return {"text": str(len(audio))}


def _trimmed(name):
    """测试句去掉首尾静音（TTS 音频前后各有约 0.3 秒静音），用来精确控制句间停顿。"""
    x, _ = asr_engine.read_wav((TESTSET / f"{name}.wav").read_bytes())
    idx = np.flatnonzero(np.abs(x) > 0.02)
    return x[idx[0]:idx[-1] + 1]


def _joined(*names, gap):
    """去掉首尾静音的测试句用 gap 秒静音连起来，开头补 0.5 秒、结尾补 1 秒静音。"""
    z = lambda s: np.zeros(int(16000 * s), np.float32)  # noqa: E731
    parts = [z(0.5)]
    for i, n in enumerate(names):
        parts += [_trimmed(n)] + ([z(gap)] if i < len(names) - 1 else [])
    return np.concatenate(parts + [z(1.0)])


def _finals(events):
    return [(e["chunk_id"], e["model"], e["text"]) for e in events if e["type"] == "text"]


class TestLive(unittest.TestCase):
    def _assert_partials_before_finals(self, events):
        first_text = {}
        for i, e in enumerate(events):
            if e["type"] == "text":
                first_text.setdefault(e["chunk_id"], i)
        for i, e in enumerate(events):
            if e["type"] == "partial":
                self.assertLess(i, first_text.get(e["chunk_id"], len(events)), e)

    def test_two_chunks_both_models(self):
        import live_engine
        models = _live_models()
        if models is None:
            self.skipTest("需要 silero-vad 和 paraformer-zh-small（python download_models.py）")
        a, _ = asr_engine.read_wav((TESTSET / "s01_cmd.wav").read_bytes())
        b, _ = asr_engine.read_wav((TESTSET / "s13_weather.wav").read_bytes())
        x = np.concatenate([a, np.zeros(16000, np.float32), b])
        events = []
        _feed(live_engine.LiveSession(models, events.append, min_silence=0.5), x)
        chunks = [e for e in events if e["type"] == "chunk"]
        self.assertEqual(len(chunks), 2, chunks)
        for k in models:
            texts = [e for e in events if e["type"] == "text" and e["model"] == k]
            self.assertEqual(len(texts), 2, k)
            self.assertTrue(all(t["text"] for t in texts), k)
            self.assertTrue(all(t["latency_ms"] >= t["decode_ms"] for t in texts))
        self.assertIn("北京", "".join(e["text"] for e in events if e["type"] == "text"))
        self.assertFalse([e for e in events if e["type"] == "partial"])   # 默认不出临时结果
        starts = [e for e in events if e["type"] == "speech_start"]
        self.assertEqual(len(starts), 2)
        for e, c in zip(starts, chunks):
            self.assertEqual(e["start"], c["start"])        # 开口时刻 = 这句切出来后的起点
            self.assertLess(e["start"], e["t"])             # VAD 确认开口要晚一点
            self.assertGreater(e["prob"], 0.5)

    def test_partial_results(self):
        """开实时出字：说到一半就出字、越识别越长；同一句的临时结果都排在正式结果前面；正式结果和不开时一模一样。"""
        import live_engine
        models = _live_models()
        if models is None:
            self.skipTest("需要 silero-vad 和 paraformer-zh-small（python download_models.py）")
        x = _sentences("s07_long", "s10_story")
        base, events = [], []
        _feed(live_engine.LiveSession(models, base.append, min_silence=0.5), x)
        # 临时结果按音频时间触发：按最快速度喂，转写线程还没空，新的一份就把旧的覆盖了。这里按 5 倍速喂
        _feed(live_engine.LiveSession(models, events.append, min_silence=0.5, partial_interval=0.4), x, pace=0.02)
        self.assertEqual(_finals(events), _finals(base))
        chunks = [e for e in events if e["type"] == "chunk"]
        self.assertEqual([c["id"] for c in chunks], [1, 2])
        partials = [e for e in events if e["type"] == "partial"]
        for k in models:
            mine = [p for p in partials if p["model"] == k and p["chunk_id"] == 1]
            self.assertGreaterEqual(len(mine), 4, k)          # 6.5s 的长句，每 0.4 秒一次（5 倍速喂入，会被覆盖掉一部分）
            self.assertEqual([p["end"] for p in mine], sorted(p["end"] for p in mine))
            self.assertTrue(all(p["start"] == chunks[0]["start"] for p in mine))   # 起点就是这句切出来后的起点
            self.assertIn("今天天气", asr_engine.normalize_text(mine[-1]["text"], True), k)
            for c in chunks:   # 同一句同一模型 rev 严格递增
                revs = [p["rev"] for p in partials if p["model"] == k and p["chunk_id"] == c["id"]]
                self.assertEqual(revs, sorted(set(revs)))
        for p in partials:
            self.assertIn(p["kind"], ("live", "lock"))
            self.assertEqual(p["text"], live_engine.join_text([p["locked"], p["live"]]))
        self.assertLessEqual({p["chunk_id"] for p in partials}, {c["id"] for c in chunks})
        self._assert_partials_before_finals(events)

    def test_long_sentence_pieces(self):
        """一长句里有几处短停顿：按停顿切段、每段锁定一次且按顺序出；蹦字只识别当前这一段；正式结果不受影响。"""
        import live_engine
        models = _live_models()
        if models is None:
            self.skipTest("需要 silero-vad 和 paraformer-zh-small（python download_models.py）")
        x = _joined("s07_long", "s10_story", "s13_weather", gap=0.35)   # 约 13 秒，句内两处 0.35 秒停顿
        base, events = [], []
        _feed(live_engine.LiveSession(models, base.append, max_chunk=20), x)
        _feed(live_engine.LiveSession(models, events.append, max_chunk=20, partial_interval=0.4), x, pace=0.02)
        self.assertEqual(_finals(events), _finals(base))
        chunks = [e for e in events if e["type"] == "chunk"]
        self.assertEqual(len(chunks), 1, chunks)            # 停顿不到 0.5 秒，还是一句
        cuts = [e for e in events if e["type"] == "cut" and e["at"] < chunks[0]["end"]]
        self.assertEqual([c["reason"] for c in cuts], ["pause", "pause"], cuts)
        partials = [e for e in events if e["type"] == "partial"]
        self.assertGreaterEqual(max(p["piece"] for p in partials), 3)
        for k in models:
            locks = [p for p in partials if p["model"] == k and p["kind"] == "lock"]
            self.assertGreaterEqual(len(locks), 2, k)
            self.assertEqual([p["piece"] for p in locks], sorted(p["piece"] for p in locks))
            for a, b in zip(locks, locks[1:]):
                self.assertTrue(b["locked"].startswith(a["locked"]), (a["locked"], b["locked"]))   # 锁定的字不再变
            lives = [p for p in partials if p["model"] == k and p["kind"] == "live"]
            self.assertTrue(all(p["end"] - p["piece_start"] <= 8.2 for p in lives))   # 只识别当前这一段
            later = [p for p in lives if p["piece"] > 1]
            self.assertTrue(later and all(p["locked"] for p in later), k)   # 后面的段出字时，前面已锁定的字跟着带上
            for p in later:   # 后面的段只识别自己这段音频，不会把句首又识别一遍
                self.assertNotIn("今天天气", asr_engine.normalize_text(p["live"], True), p)
        self._assert_partials_before_finals(events)

    def test_max_chunk_is_hard(self):
        """一直不停地说：句长到 max_chunk 就硬切（sherpa 的上限是软的）；硬切点两边不补 pad，送去识别的音频不重叠。"""
        import live_engine
        if not asr_engine.vad_model_path():
            self.skipTest("需要 silero-vad（python download_models.py）")
        x = _joined("s07_long", "s10_story", "s13_weather", "s01_cmd", gap=0.0)
        rec, events = _Recorder(), []
        _feed(live_engine.LiveSession({"r": rec}, events.append, max_chunk=6.0), x)
        chunks = [e for e in events if e["type"] == "chunk"]
        self.assertGreaterEqual(len(chunks), 3, chunks)
        self.assertTrue(all(c["dur"] <= 6.0 for c in chunks), chunks)
        self.assertIn("max_len", [c["reason"] for c in chunks])
        self.assertEqual(len(rec.lengths), len(chunks))
        for i, (c, n) in enumerate(zip(chunks, rec.lengths)):
            if c["reason"] == "max_len":
                self.assertEqual(c["end"], chunks[i + 1]["start"])    # 切点就是下一句的起点
            front = 0.0 if i and chunks[i - 1]["reason"] == "max_len" else 0.2
            back = 0.0 if c["reason"] == "max_len" else 0.2
            self.assertAlmostEqual(n / 16000, c["dur"] + front + back, delta=0.011)

    def test_partials_never_drop_or_reorder_finals(self):
        """转写线程忙不过来时：正式结果一个不丢、按顺序出；过期的临时结果作废，不会排到同一句的正式结果后面。"""
        import live_engine
        if not asr_engine.vad_model_path():
            self.skipTest("需要 silero-vad（python download_models.py）")
        # 句间留 6s 静音：否则下一句的临时结果很快就把槽位里过期的那份覆盖掉，测不出“过期的临时结果排到正式结果后面”
        x = _sentences("s07_long", "s10_story", "s13_weather", gap=6.0)
        events = []
        # 两个假模型各 0.15s/次，一轮临时结果 0.3s；10 倍速喂入时每 0.05s 就来一份新的 → 大部分被覆盖或作废
        sess = live_engine.LiveSession({"a": _SlowFake(0.15), "b": _SlowFake(0.15)}, events.append,
                                       min_silence=0.5, partial_interval=0.5)
        _feed(sess, x, pace=0.01)
        self.assertFalse(sess._worker.is_alive())
        chunks = [e["id"] for e in events if e["type"] == "chunk"]
        self.assertEqual(chunks, [1, 2, 3])
        for k in ("a", "b"):
            self.assertEqual([e["chunk_id"] for e in events if e["type"] == "text" and e["model"] == k], chunks)
        partials = [e for e in events if e["type"] == "partial"]
        self.assertTrue(partials)
        self.assertLessEqual({p["chunk_id"] for p in partials}, set(chunks))
        for c in chunks:
            for k in ("a", "b"):
                pieces = [p["piece"] for p in partials if p["chunk_id"] == c and p["model"] == k and p["kind"] == "lock"]
                self.assertEqual(pieces, sorted(pieces))
        self._assert_partials_before_finals(events)

    def test_save_audio_lossless_and_no_overwrite(self):
        import live_engine
        if not asr_engine.vad_model_path():
            self.skipTest("需要 silero-vad（python download_models.py）")
        rng = np.random.default_rng(0)
        pcm = [rng.integers(-3000, 3000, n, dtype=np.int16) for n in (20000, 7777)]  # 两次会话，长度不是 512/1600 的整数倍
        with tempfile.TemporaryDirectory() as tmp:
            sessions = []
            for v in pcm:   # 两个会话几乎同时开始（大概率落在同一秒），不能互相覆盖
                s = live_engine.LiveSession({}, lambda e: None, log_dir=Path(tmp), save_audio=True)
                x = v.astype(np.float32) / 32768.0  # 和 server.py 收到 Int16 后的换算一致
                for i in range(0, len(x), 1600):
                    s.accept(x[i:i + 1600])
                sessions.append(s)
            for s in sessions:
                s.close()
            self.assertNotEqual(sessions[0].audio_path, sessions[1].audio_path)
            for s, v in zip(sessions, pcm):
                self.assertTrue(s.audio_path.with_suffix(".jsonl").exists())
                with wave.open(str(s.audio_path), "rb") as w:
                    self.assertEqual((w.getnchannels(), w.getsampwidth(), w.getframerate()), (1, 2, 16000))
                    self.assertEqual(w.readframes(w.getnframes()), v.astype("<i2").tobytes())
            quiet = live_engine.LiveSession({}, lambda e: None, log_dir=Path(tmp))  # 默认不存录音
            quiet.close()
            self.assertIsNone(quiet.audio_path)
            self.assertEqual(len(list(Path(tmp).glob("*.wav"))), 2)


class TestSegmenter(unittest.TestCase):
    """vad_stream.Segmenter 是纯 Python：直接喂人为构造的人声概率，不需要模型。"""

    @staticmethod
    def _probs(*spans):
        """依次若干段 (秒, 概率)，每帧 512 采样（32ms）。"""
        return np.concatenate([np.full(round(s * 16000 / 512), p, np.float32) for s, p in spans])

    @staticmethod
    def _inner(P, **kw):
        """切句，并只留落在句子内部的段切点（句末静音里的那一刀不算）。"""
        import vad_stream
        sents, cuts = vad_stream.replay(P, **kw)
        return sents, [c for c in cuts if any(s < c.at < e for s, e, _ in sents)]

    def test_pause_cuts_piece_not_sentence(self):
        sents, cuts = self._inner(self._probs((0.5, 0.02), (2.0, 0.95), (0.3, 0.05), (2.0, 0.95), (1.0, 0.02)),
                                  piece_gap=0.2)
        self.assertEqual(len(sents), 1)                     # 0.3 秒停顿不到 min_silence（0.5 秒），还是一句
        self.assertEqual([(c.reason, c.piece) for c in cuts], [("pause", 1)])
        self.assertTrue(2.5 * 16000 < cuts[0].at < 2.8 * 16000, cuts)   # 切点落在停顿里
        self.assertGreaterEqual(cuts[0].hi, cuts[0].at)
        # 停顿不够 piece_gap、停顿前这段不够 piece_min：都不切
        self.assertFalse(self._inner(self._probs((0.5, 0.02), (2.0, 0.95), (0.15, 0.05), (2.0, 0.95), (1.0, 0.02)),
                                     piece_gap=0.2)[1])
        self.assertFalse(self._inner(self._probs((0.5, 0.02), (0.6, 0.95), (0.3, 0.05), (2.0, 0.95), (1.0, 0.02)),
                                     piece_gap=0.2)[1])

    def test_neg_threshold_ends_sooner_in_noise(self):
        import vad_stream
        # 说完话以后概率停在 0.42（有底噪的房间里常见）：sherpa 的规则下 >0.35 都算还在说，降到 0.35 以下才开始计静音
        P = self._probs((0.5, 0.02), (2.0, 0.95), (1.5, 0.42), (1.0, 0.02))
        (s0, e0, r0), = vad_stream.replay(P)[0]
        (s1, e1, r1), = vad_stream.replay(P, neg_threshold=0.5)[0]
        self.assertEqual((s0, r0, r1), (s1, "silence", "silence"))
        self.assertAlmostEqual(e0 / 16000, 4.0, delta=0.05)    # 等概率掉到 0.35 以下
        self.assertAlmostEqual(e1 / 16000, 2.5, delta=0.05)    # 0.42 就算静音：判停早 1.5 秒
        self.assertEqual(vad_stream.Segmenter(neg_threshold=0.9).neg, 0.5)   # 不会高过 threshold

    def test_piece_max_cuts_at_lowest_prob(self):
        P = self._probs((0.5, 0.02), (20.0, 0.95), (1.0, 0.02))
        dips = [round((0.5 + t) * 16000 / 512) for t in (6.9, 7.3, 14.1)]
        P[dips[0]], P[dips[1]], P[dips[2]] = 0.6, 0.4, 0.45
        sents, cuts = self._inner(P, piece_max=8.0)
        self.assertEqual(len(sents), 1)
        self.assertEqual([c.reason for c in cuts], ["max_len", "max_len"])
        self.assertEqual([c.at for c in cuts], [dips[1] * 512 + 256, dips[2] * 512 + 256])   # 切在最低那一帧的中点
        bounds = [sents[0][0]] + [c.at for c in cuts] + [sents[0][1]]
        self.assertTrue(all(b - a <= 8 * 16000 for a, b in zip(bounds, bounds[1:])), bounds)

    def test_sentence_max_is_hard(self):
        import vad_stream
        sents, _ = vad_stream.replay(self._probs((0.5, 0.02), (25.0, 0.95), (1.0, 0.02)), sentence_max=10.0)
        self.assertEqual([r for *_, r in sents], ["max_len", "max_len", "silence"])
        self.assertTrue(all(e - s <= 10 * 16000 for s, e, _ in sents), sents)
        for (_, e, _), (s, _, _) in zip(sents, sents[1:]):
            self.assertEqual(e, s)                          # 硬切：上一句的终点就是下一句的起点

    def test_flush_mid_speech(self):
        import vad_stream
        seg = vad_stream.Segmenter()
        for p in self._probs((0.5, 0.02), (2.0, 0.95)):
            seg.step(float(p))
        self.assertTrue(seg.in_speech)
        (e,) = seg.flush()
        self.assertEqual((e.kind, e.reason, e.at), ("end", "flush", seg.cur))
        self.assertFalse(seg.in_speech)


class TestVadStream(unittest.TestCase):
    def setUp(self):
        if not asr_engine.vad_model_path():
            self.skipTest("需要 silero-vad（python download_models.py）")

    @staticmethod
    def _compare(x, agc, min_silence=0.5):
        """自研 SileroStream 和 sherpa 的 VoiceActivityDetector 吃完全一样的（放大后的）帧，逐帧对照。"""
        import vad_stream
        ref = asr_engine.new_vad(min_silence=min_silence, max_speech=600, buffer_sec=900)
        s = vad_stream.SileroStream(agc=agc, min_silence=min_silence)
        mine, theirs, flips, speaking, rest = [], [], 0, False, np.zeros(0, np.float32)
        for i in range(0, len(x), 1600):
            blk = x[i:i + 1600]
            frames = s.accept(blk)
            g = np.concatenate([rest, np.clip(blk * s.gain, -1.0, 1.0)])
            rest = g[len(frames) * 512:]
            for k, fr in enumerate(frames):
                ref.accept_waveform(g[k * 512:(k + 1) * 512])   # sherpa 必须一帧一帧喂：一次喂多帧会改变判定
                for e in fr.events:
                    speaking = e.kind == "start"
                    if e.kind == "end":
                        mine.append((e.start, e.at))
                flips += speaking != ref.is_speech_detected()
                while not ref.empty():
                    theirs.append((ref.front.start, ref.front.start + len(ref.front.samples)))
                    ref.pop()
        ref.flush()
        mine += [(e.start, e.at) for e in s.flush()]
        while not ref.empty():
            theirs.append((ref.front.start, ref.front.start + len(ref.front.samples)))
            ref.pop()
        return mine, theirs, flips

    def test_equivalent_to_sherpa(self):
        """同一个 silero_vad.onnx、同样的参数：切出的段一个采样点都不差，每一帧"是否在说话"都一致。"""
        xs = [asr_engine.read_wav((TESTSET / f"{n}.wav").read_bytes())[0]
              for n in ("s01_cmd", "s07_long", "s13_weather", "s10_story")]
        for gap, scale, agc, min_silence in ((0.8, 1.0, False, 0.5), (0.3, 1.0, False, 0.5),
                                             (0.6, 0.05, True, 0.5), (0.4, 1.0, False, 0.3)):
            x = np.concatenate([np.concatenate([v * scale, np.zeros(int(16000 * gap), np.float32)]) for v in xs])
            mine, theirs, flips = self._compare(x, agc, min_silence)
            self.assertGreaterEqual(len(theirs), 2)
            self.assertEqual(mine, theirs, (gap, scale, agc, min_silence))
            self.assertEqual(flips, 0, (gap, scale, agc, min_silence))
        ramc = sorted((TESTSET.parents[1] / "test_audio" / "ramc").glob("*.wav"))   # 本机有真人对话就顺便比一下
        if ramc:
            x, _ = asr_engine.read_wav(ramc[0].read_bytes())
            mine, theirs, flips = self._compare(x, True)
            self.assertEqual((mine, flips), (theirs, 0))

    def test_short_pause_cuts_piece(self):
        """两句之间留 0.35 秒静音：只切段、不切句，切点落在这段静音里；留 0.8 秒就切成两句。"""
        import vad_stream
        gap_at = 0.5 + len(_trimmed("s01_cmd")) / 16000
        for gap, n_sent in ((0.35, 1), (0.8, 2)):
            x = _joined("s01_cmd", "s13_weather", gap=gap)
            s = vad_stream.SileroStream(piece_gap=0.2)
            evs = [e for i in range(0, len(x), 1600) for f in s.accept(x[i:i + 1600]) for e in f.events]
            evs += s.flush()
            ends = [e for e in evs if e.kind == "end"]
            self.assertEqual(len(ends), n_sent, ends)
            inner = [e for e in evs if e.kind == "cut" and e.at < ends[0].at]
            if n_sent == 1:
                self.assertEqual(len(inner), 1, evs)
                self.assertTrue(gap_at * 16000 <= inner[0].at <= (gap_at + gap) * 16000, inner)
            else:
                self.assertFalse(inner)


class TestRegistry(unittest.TestCase):
    def test_specs(self):
        keys = [s["key"] for s in registry.all_specs()]
        self.assertEqual(len(keys), len(set(keys)))
        defaults = [s for s in registry.ASR_MODELS if s["default"]]
        self.assertEqual([s["key"] for s in defaults], ["sensevoice", "paraformer-zh-small"])
        for s in registry.ASR_MODELS:
            for k in ("name", "loader", "files", "doc_size_mb", "size_source", "public_cer", "license", "link"):
                self.assertIn(k, s, f"{s['key']} 缺 {k}")

    def test_resolve_prefers_top_level_and_int8(self):
        spec = {"key": "fake-transducer", "files": {
            "encoder": ["encoder*.int8.onnx", "encoder*.onnx"],
            "decoder": ["decoder-epoch-99-avg-1.onnx", "decoder.onnx"],
            "joiner": ["joiner*.int8.onnx", "joiner*.onnx"],
            "tokens": ["tokens.txt"]}}
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / spec["key"]
            (root / "64").mkdir(parents=True)
            for n in ["encoder-epoch-99-avg-1.int8.onnx", "encoder-epoch-99-avg-1.onnx",
                      "decoder-epoch-99-avg-1.int8.onnx", "decoder-epoch-99-avg-1.onnx",
                      "joiner-epoch-99-avg-1.int8.onnx", "tokens.txt"]:
                (root / n).write_bytes(b"x")
                (root / "64" / n).write_bytes(b"x")
            f = registry.resolve_files(spec, Path(d))
            self.assertEqual(f["encoder"].name, "encoder-epoch-99-avg-1.int8.onnx")
            self.assertEqual(f["decoder"].name, "decoder-epoch-99-avg-1.onnx")  # fp32 decoder
            self.assertEqual(f["encoder"].parent, root)
            (root / "tokens.txt").unlink()
            self.assertEqual(registry.resolve_files(spec, Path(d))["tokens"].parent.name, "64")


if __name__ == "__main__":
    unittest.main()
