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


class _NoCurrentSegment:
    """包一层 VAD，假装是没有 current_segment 的老版本 sherpa-onnx。"""

    def __init__(self, vad):
        self._vad = vad

    def __getattr__(self, name):
        if name == "current_segment":
            raise AttributeError(name)
        return getattr(self._vad, name)


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

    def test_partial_results(self):
        """开临时结果：长句说到一半就出字、越识别越长；同一句的临时结果都排在正式结果前面；正式结果和不开时一模一样。"""
        import live_engine
        models = _live_models()
        if models is None:
            self.skipTest("需要 silero-vad 和 paraformer-zh-small（python download_models.py）")
        x = _sentences("s07_long", "s10_story")
        base, events = [], []
        _feed(live_engine.LiveSession(models, base.append, min_silence=0.5), x)
        # 临时结果按音频时间触发：按最快速度喂，转写线程还没空，新的一份就把旧的覆盖了。这里按 5 倍速喂
        _feed(live_engine.LiveSession(models, events.append, min_silence=0.5, partial_interval=1.0), x, pace=0.02)
        finals = lambda evs: [(e["chunk_id"], e["model"], e["text"]) for e in evs if e["type"] == "text"]  # noqa: E731
        self.assertEqual(finals(events), finals(base))
        chunks = [e for e in events if e["type"] == "chunk"]
        self.assertEqual([c["id"] for c in chunks], [1, 2])
        partials = [e for e in events if e["type"] == "partial"]
        for k in models:
            mine = [p for p in partials if p["model"] == k and p["chunk_id"] == 1]
            self.assertGreaterEqual(len(mine), 2, k)          # 6.5s 的长句，每 1 秒一次
            self.assertEqual([p["end"] for p in mine], sorted(p["end"] for p in mine))
            self.assertTrue(all(p["start"] == chunks[0]["start"] for p in mine))   # 起点就是这句切出来后的起点
            self.assertIn("今天天气", asr_engine.normalize_text(mine[-1]["text"], True), k)
        self.assertLessEqual({p["chunk_id"] for p in partials}, {c["id"] for c in chunks})
        self._assert_partials_before_finals(events)

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
        self._assert_partials_before_finals(events)

    def test_partial_start_fallback_matches_vad(self):
        """老版本 sherpa-onnx 没有 current_segment 时，推算出的段起点要和 VAD 自己记的一致。"""
        import live_engine
        if not asr_engine.vad_model_path():
            self.skipTest("需要 silero-vad（python download_models.py）")
        x = _sentences("s07_long")
        sess = live_engine.LiveSession({}, lambda e: None)
        for i in range(0, 3 * 16000, 1600):     # 喂到这句说了一半
            sess.accept(x[i:i + 1600])
        self.assertTrue(sess.vad.is_speech_detected())
        real = sess._segment_start()
        sess.vad = _NoCurrentSegment(sess.vad)
        self.assertEqual(sess._segment_start(), real)
        sess.close()

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
