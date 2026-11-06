"""主人识别：注册向量加载 + 余弦相似度评分 + 三态判定（owner / stranger / unknown）。

向量来自注册工具（tools/enroll_owner.py），存 data/owner_profile.json（生物信息，不提交）。
判定规则：
    duration < MIN_SEC（3 秒）       → unknown（不看分数）
    score >= THRESHOLD              → owner
    否则                            → stranger
阈值由 tools/eval_owner.py 的 FAR/FRR 曲线定（起点 0.41 = ERes2Net 的 EER 点）。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

MIN_SEC = 3.0          # 不足 3 秒不判（段太短，声纹不可靠）
# 双阈值（owner_eval.md 曲线，开发集定的，验证集未调参）：
#   对话档 0.45：FAR 1.4% / FRR 0%——主人少被拒，日常显示用
#   特权档 0.55：FAR 0.3% / FRR 12.1%——敏感操作用，宁可拒
DIALOG_THRESHOLD = 0.45
PRIVILEGED_THRESHOLD = 0.55
DEFAULT_THRESHOLD = DIALOG_THRESHOLD
DEFAULT_PROFILE = Path(__file__).resolve().parent / "data" / "owner_profile.json"


class OwnerProfile:
    """注册向量集：list[np.ndarray]，全部 L2 归一化。"""

    def __init__(self, vectors: list[np.ndarray]):
        self.vectors = [np.asarray(v, dtype=np.float32) for v in vectors]

    @classmethod
    def load(cls, path: Path | str = DEFAULT_PROFILE) -> "OwnerProfile":
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls([np.asarray(v, dtype=np.float32) for v in data["vectors"]])

    def save(self, path: Path | str = DEFAULT_PROFILE) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"vectors": [v.tolist() for v in self.vectors]}, ensure_ascii=False),
                        encoding="utf-8")

    def __len__(self) -> int:
        return len(self.vectors)


def score(profile: OwnerProfile, emb: "np.ndarray") -> float:
    """余弦相似度均值（Apple 式 mean-of-scores）。向量都已 L2 归一化，点积即余弦。"""
    if not profile.vectors:
        return 0.0
    e = np.asarray(emb, dtype=np.float32)
    e = e / (np.linalg.norm(e) + 1e-9)
    return float(np.mean([float(np.dot(e, v)) for v in profile.vectors]))


def decide(profile: OwnerProfile | None, emb, duration_sec: float,
           threshold: float = DIALOG_THRESHOLD) -> str:
    """三态判定：unknown（<3 秒或无注册）/ owner / stranger。

    threshold 传档位值：日常显示用 DIALOG_THRESHOLD（0.45），敏感操作用 PRIVILEGED_THRESHOLD（0.55）。
    """
    if profile is None or not profile.vectors or duration_sec < MIN_SEC:
        return "unknown"
    return "owner" if score(profile, emb) >= threshold else "stranger"


def load_model(model_path: Path | str):
    """加载声纹模型（ERes2Net），进程内缓存：同一模型只建一次。"""
    global _MODEL
    if _MODEL is None or str(_MODEL.model_path) != str(model_path):
        import sherpa_onnx as so

        cfg = so.SpeakerEmbeddingExtractorConfig(model=str(model_path), num_threads=2)
        if not cfg.validate():
            raise RuntimeError(f"声纹模型配置无效: {model_path}")
        ex = so.SpeakerEmbeddingExtractor(cfg)

        class _M:
            pass
        _MODEL = _M()
        _MODEL.model_path = str(model_path)
        _MODEL.ex = ex
    return _MODEL


_MODEL = None


def embed(x: "np.ndarray", model) -> "np.ndarray":
    """一条音频 → L2 归一化声纹向量（VAD 切过的正式句本来就是纯人声，这里不再过 VAD）。"""
    s = model.ex.create_stream()
    s.accept_waveform(16000, x)
    s.input_finished()
    e = np.asarray(model.ex.compute(s), dtype=np.float32)
    return e / (np.linalg.norm(e) + 1e-9)
