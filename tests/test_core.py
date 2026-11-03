"""核心逻辑单测（不需要模型）：python -m unittest discover -s tests"""
import io
import sys
import tempfile
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
