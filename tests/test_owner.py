"""主人识别单测（不需要声纹模型，向量用合成数据）：python -m unittest discover -s tests"""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import asr_engine  # noqa: E402
import speaker_owner  # noqa: E402
from speaker_owner import MIN_SEC, OwnerProfile, decide, score  # noqa: E402


def rand_unit(seed: int, dim: int = 64) -> "np.ndarray":
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype("float32")
    return v / np.linalg.norm(v)


class TestOwner(unittest.TestCase):
    def test_enroll_then_recognize(self):
        """注册两句后第三句判主人。"""
        v1, v2, v3 = rand_unit(1), rand_unit(1) + 0.01, rand_unit(1) + 0.02
        profile = OwnerProfile([v1, v2])
        self.assertEqual(decide(profile, v3, duration_sec=5.0, threshold=0.41), "owner")

    def test_stranger(self):
        """陌生人句判陌生人。"""
        profile = OwnerProfile([rand_unit(1), rand_unit(2)])
        stranger = rand_unit(999)   # 独立随机向量，与主人向量接近正交
        self.assertEqual(decide(profile, stranger, duration_sec=5.0, threshold=0.41), "stranger")

    def test_short_unknown(self):
        """1 秒句判 unknown（不看分数）。"""
        profile = OwnerProfile([rand_unit(1), rand_unit(2)])
        self.assertEqual(decide(profile, rand_unit(1), duration_sec=MIN_SEC - 0.1), "unknown")
        self.assertEqual(decide(profile, rand_unit(999), duration_sec=1.0), "unknown")

    def test_shuffled_not_owner(self):
        """注册向量打乱后不许判 owner。"""
        rng = np.random.default_rng(0)
        v = rand_unit(1)
        perm = rng.permutation(len(v))
        shuffled = v[perm]
        profile = OwnerProfile([shuffled])
        self.assertNotEqual(decide(profile, v, duration_sec=5.0, threshold=0.41), "owner")

    def test_score_range(self):
        """cos 相似度在 [-1, 1]。"""
        profile = OwnerProfile([rand_unit(1)])
        for other in (rand_unit(1), rand_unit(999)):
            self.assertLessEqual(score(profile, other), 1.0 + 1e-6)
            self.assertGreaterEqual(score(profile, other), -1.0 - 1e-6)

    def test_save_load(self):
        """save/load 往返，向量一致。"""
        profile = OwnerProfile([rand_unit(1), rand_unit(2)])
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "owner_profile.json"
            profile.save(p)
            loaded = OwnerProfile.load(p)
        self.assertEqual(len(loaded), 2)
        for a, b in zip(profile.vectors, loaded.vectors):
            self.assertTrue(np.allclose(a, b))

    def test_empty_profile(self):
        """空注册 → unknown。"""
        self.assertEqual(decide(OwnerProfile([]), rand_unit(1), duration_sec=5.0), "unknown")
        self.assertEqual(decide(None, rand_unit(1), duration_sec=5.0), "unknown")


class _FakeEmbModel:
    """假声纹模型：embed 时返回注册向量的固定线性组合，让相似度可控。"""

    def __init__(self, vec: "np.ndarray"):
        self.vec = vec

    def embed(self, x):
        return self.vec


class TestLiveOwner(unittest.TestCase):
    """live_engine 接入：text 事件的 owner 字段（不需要真声纹模型，喂真语音+假声纹模型）。"""

    def _sess_events(self, profile_vec, emb_ret, diarizer="one_seg", **kw):
        """profile_vec：注册向量；emb_ret：embed 返回的假向量（speaker_owner.embed 被 mock）。
        diarizer：'one_seg' 整句一段 / 'two_seg' 切两段各半（测主段挑选）；None = 不给（普通模式）。"""
        import unittest.mock as mock
        import live_engine
        events = []
        profile = OwnerProfile([profile_vec]) if profile_vec is not None else None
        model = object() if profile is not None else None   # 占位：embed 已被 mock，不会用到真模型
        dia = None
        if diarizer == "one_seg":
            dia = _FakeDiarizer([(0.0, 12.3, 0)])
        elif diarizer == "two_seg":
            dia = _FakeDiarizer([(0.0, 6.0, 0), (6.2, 12.3, 1)])
        ctx = mock.patch.object(speaker_owner, "embed", lambda x, m: emb_ret) if profile is not None else None
        sess = live_engine.LiveSession({"r": _Recorder()}, events.append, owner_profile=profile,
                                       owner_model=model, owner_diarizer=dia, piece_max=30, **kw)
        # s07_long(6.46s) 截掉尾部 0.3s 低幅后两遍拼接 → VAD 一段 12.3s（≥ MIN_SEC，判 owner/stranger）
        root = Path(__file__).resolve().parents[1]
        b, _ = asr_engine.read_wav((root / "benchmark/testset/s07_long.wav").read_bytes())
        x = np.concatenate([b[:-4800], b])
        try:
            if ctx:
                ctx.start()
            for i in range(0, len(x), 1600):
                sess.accept(x[i:i + 1600])
            sess.close()
            sess._worker.join(timeout=10)
        finally:
            if ctx:
                ctx.stop()
        return [e for e in events if e["type"] == "text"]

    def test_owner_field_present(self):
        """给了注册+模型：text 事件有 owner 字段且为 bool（假向量=注册向量 → owner=True）。"""
        v = rand_unit(1)
        texts = self._sess_events(v, v)
        self.assertTrue(texts)
        self.assertTrue(all(isinstance(t["owner"], bool) for t in texts), texts)
        self.assertTrue(all(t["owner"] for t in texts), texts)

    def test_owner_field_stranger(self):
        """陌生人向量（正交）→ owner=False。"""
        texts = self._sess_events(rand_unit(1), rand_unit(999))
        self.assertTrue(texts)
        self.assertTrue(all(t["owner"] is False for t in texts), texts)

    def test_no_profile_no_field(self):
        """没给注册：text 事件没有 owner 字段（普通模式不改动事件）。"""
        texts = self._sess_events(None, None)
        self.assertTrue(texts)
        self.assertFalse(any("owner" in t for t in texts), texts)

    def test_profile_without_model_raises(self):
        """给了注册不给模型：构造时报错（防止静默不判）。"""
        import live_engine
        with self.assertRaises(ValueError):
            live_engine.LiveSession({"r": _Recorder()}, lambda e: None,
                                    owner_profile=OwnerProfile([rand_unit(1)]))

    def test_profile_without_diarizer_raises(self):
        """给了注册不给分角色器：构造时报错（句内分角色段累积是判定路径）。"""
        import live_engine
        with self.assertRaises(ValueError):
            live_engine.LiveSession({"r": _Recorder()}, lambda e: None,
                                    owner_profile=OwnerProfile([rand_unit(1)]), owner_model=object())

    def test_two_seg_picks_main_speaker(self):
        """句内切两段（陌生人+主人各半）：embed 收到的是主段音频，不是整句。"""
        import unittest.mock as mock
        import speaker_owner as so
        v = rand_unit(1)
        seen = []
        real = so.embed
        with mock.patch.object(so, "embed", lambda x, m: (seen.append(len(x)), v)[1]):
            texts = self._sess_events(v, v, diarizer="two_seg")
        self.assertTrue(texts)
        self.assertTrue(all(t["owner"] for t in texts), texts)
        # 12.3s 句切两段各 6s：embed 收到的应是段长（约 6s×16000），不是整句 12.3s×16000
        self.assertTrue(all(n < 16000 * 10 for n in seen), seen)


class _FakeDiarizer:
    """假分角色器：返回固定段（start, end, speaker），不跑真模型。"""

    def __init__(self, segs):
        self.segs = segs

    def run(self, audio, num_speakers=-1, threshold=0.5):
        return [{"start": s, "end": e, "speaker": sp} for s, e, sp in self.segs], 0.0


class _Recorder:
    """live 测试用假 ASR：立刻返回。"""

    recognizer = None

    def _offline_stream(self, rec, sr, audio):
        return {"text": "x"}


if __name__ == "__main__":
    unittest.main()
