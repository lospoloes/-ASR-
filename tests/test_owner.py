"""主人识别单测（不需要声纹模型，向量用合成数据）：python -m unittest discover -s tests"""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
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


if __name__ == "__main__":
    unittest.main()
