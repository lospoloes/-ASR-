"""生成标准文本初稿：多模型字级对齐，把各模型不一致的地方标成【】。

为什么不能直接用某一个模型的输出当初稿：拿哪个模型的输出当骨架，
CER 就会偏向那个模型（它的错字会被当成标准答案）。所以骨架用 SenseVoice
（VAD 模式，标点最完整），再把 Paraformer / Dolphin / multi-zh-hans 的输出
对齐上来，凡是大家不一致的位置标【】并列候选，交给人边听音频边定稿。

用法:
    # 音频不提交仓库，先下载（272s，sherpa-onnx 官方 release）
    curl -L -o test_audio/lei-jun-test.wav https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/lei-jun-test.wav
    python tools/make_ref_draft.py                       # 默认 test_audio/lei-jun-test.wav
    python tools/make_ref_draft.py --audio xxx.wav --out benchmark/long/ref_draft.txt

输出:
    benchmark/long/ref_draft.txt   带【候选1｜候选2】的初稿 + 每处差异的上下文
"""
from __future__ import annotations

import argparse
import difflib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import asr_engine  # noqa: E402
import registry  # noqa: E402

# 骨架 + 参与对齐的候选模型。顺序有意义：第一个是骨架。
SKELETON = "sensevoice"
OTHERS = ["paraformer-zh-small"]  # 其余候选已从注册表移除


def strip_punct(s: str) -> str:
    """去标点空格，并把阿拉伯数字转成汉字读法（三千五百 / 一九八七年），便于字级对齐。

    数字归一化是确定性规则，不偏向任何模型；初稿最终也要求数字写汉字读法。
    """
    s = asr_engine.arabic_to_zh(s)
    return "".join(ch for ch in s if not ch.isspace() and ch not in "，。、？！：；「」“”‘’（）()《》,.!?:;")


def transcribe_all(audio: Path, models: list[str], threads: int) -> dict[str, str]:
    downloaded = asr_engine.find_downloaded()
    x, sr = asr_engine.read_wav(audio.read_bytes())
    x = asr_engine.resample(x, sr)
    segs, vad_ms = asr_engine.compute_vad(x)
    print(f"音频 {len(x) / 16000:.1f}s，VAD {vad_ms:.0f} ms / {len(segs)} 段", flush=True)

    out = {}
    for key in models:
        if key not in downloaded:
            print(f"  跳过 {key}（未下载）", flush=True)
            continue
        spec = registry.get_spec(key)
        m = asr_engine.AsrModel(spec, downloaded[key], threads)
        m.load()
        r = m.transcribe(x, segs=segs)
        out[key] = r["text"]
        print(f"  {key:30} {r['elapsed_ms'] / 1000:6.1f}s  {len(r['text'])} 字", flush=True)
    return out


def build_draft(skel: str, others: dict[str, str]) -> tuple[str, list[dict]]:
    """把 others 逐个对齐到骨架，收集分歧点。

    三类分歧分开标记，避免把「某模型多几个字」误标成「某个字写法不同」：
      替换（等长）-> 【A｜B】          某个位置到底该写哪个字
      替换（不等长）-> 【ABCD｜XY】     一个片段整体该写什么
      插入（其他模型多字）-> 【+XY】    骨架这里可能漏了字
      删除（其他模型少字）-> 【AB｜∅】  骨架这里可能多写了字

    返回 (带标记的初稿, 分歧列表)。
    """
    skel_np = strip_punct(skel)
    n = len(skel_np)
    votes = [{"base": skel_np[i], "alts": set()} for i in range(n)]
    regions: dict[tuple[int, int], dict] = {}   # (i1, i2) -> {"kind", "skel", "alts": set()}
    inserts: dict[int, set] = {}                # i1 -> {其他模型在此处多出的片段}

    for name, text in others.items():
        hyp = strip_punct(text)
        sm = difflib.SequenceMatcher(None, skel_np, hyp, autojunk=False)
        for op, i1, i2, j1, j2 in sm.get_opcodes():
            if op == "equal":
                continue
            ref_seg, hyp_seg = skel_np[i1:i2], hyp[j1:j2]
            if op == "replace" and (i2 - i1) == (j2 - j1):
                # 等长替换：逐字对应，各自记候选
                for k, (rc, hc) in enumerate(zip(ref_seg, hyp_seg)):
                    if rc != hc:
                        votes[i1 + k]["alts"].add(hc)
            elif op == "insert":
                # 其他模型在这里多出字（骨架可能漏字）——挂在缝隙上，不动骨架字
                inserts.setdefault(i1, set()).add(hyp_seg)
            else:
                # 不等长替换 / 删除：整段作为一个区域
                r = regions.setdefault((i1, i2), {"kind": "replace" if op == "replace" else "delete",
                                                  "skel": ref_seg, "alts": set()})
                r["alts"].add(hyp_seg)

    # 生成带标记的初稿
    pieces, diffs = [], []
    region_at: dict[int, tuple[int, int]] = {}
    for (i1, i2) in regions:
        region_at[i1] = (i1, i2)

    i = 0
    while i < n:
        if i in inserts:
            for seg in sorted(inserts[i]):
                pieces.append("【+" + seg + "】")
                diffs.append({"kind": "插入", "pos": i, "skel": "∅", "alts": seg,
                              "ctx": skel_np[max(0, i - 8):i] + "[+*]" + skel_np[i:i + 8]})
        if i in region_at and region_at[i][1] > i:
            i1, i2 = region_at[i]
            r = regions[(i1, i2)]
            alts = sorted(a for a in r["alts"] if a) or ["∅"]   # 空串 = 该模型此处没有字
            pieces.append("【" + r["skel"] + "｜" + "｜".join(alts) + "】")
            diffs.append({"kind": "替换" if r["kind"] == "replace" else "删除", "pos": i1,
                          "skel": r["skel"], "alts": "、".join(alts),
                          "ctx": skel_np[max(0, i1 - 8):i1] + "[" + r["skel"] + "]" + skel_np[i2:i2 + 8]})
            i = i2
            continue
        v = votes[i]
        if v["alts"]:
            pieces.append("【" + "｜".join([v["base"]] + sorted(v["alts"])) + "】")
            diffs.append({"kind": "替换", "pos": i, "skel": v["base"], "alts": "、".join(sorted(v["alts"])),
                          "ctx": skel_np[max(0, i - 8):i] + "[" + v["base"] + "]" + skel_np[i + 1:i + 9]})
        else:
            pieces.append(v["base"])
        i += 1
    # 结尾之后还有插入的情况
    if n in inserts:
        for seg in sorted(inserts[n]):
            pieces.append("【+" + seg + "】")
            diffs.append({"kind": "插入", "pos": n, "skel": "∅", "alts": seg,
                          "ctx": skel_np[max(0, n - 8):n] + "[+*]"})
    return "".join(pieces), diffs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", default=str(ROOT / "test_audio" / "lei-jun-test.wav"))
    ap.add_argument("--out", default=str(ROOT / "benchmark" / "long" / "ref_draft.txt"))
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()

    audio = Path(args.audio)
    if not audio.exists():
        print(f"找不到音频: {audio}\n先下载：\n"
              "  curl -L -o test_audio/lei-jun-test.wav \\\n"
              "    https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/lei-jun-test.wav")
        return 1

    texts = transcribe_all(audio, [SKELETON] + OTHERS, args.threads)
    if SKELETON not in texts:
        print(f"缺少骨架模型 {SKELETON}，先 python download_models.py")
        return 1

    others = {k: v for k, v in texts.items() if k != SKELETON}
    draft, diffs = build_draft(texts[SKELETON], others)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 长音频标准文本初稿（自动生成，需人工听音校对）",
        "#",
        f"# 音频: {audio.name}  时长 {round(len(asr_engine.read_wav(audio.read_bytes())[0]) / 16000, 1)}s",
        f"# 骨架: {SKELETON}（VAD 模式）；对齐参考: {', '.join(others) or '无'}",
        f"# 不一致处: {len(diffs)} 处，用【候选1｜候选2】标出，∅ 表示该模型此处没有字",
        "# 校对规则: 按实际读音写；口误/重复照实写；数字写汉字读法（三十六年前、一九八七年）；标点随意",
        "# 定稿后存成 benchmark/long/refs.tsv，格式:  lei-jun-test.wav<TAB>全文",
        "",
        "## 初稿",
        "",
        draft,
        "",
        f"## 不一致处清单（{len(diffs)} 处）",
        "",
        "| # | 类型 | 位置 | 骨架 | 其他模型写法 | 上下文 |",
        "|---|---|---|---|---|---|",
    ]
    for k, d in enumerate(diffs, 1):
        lines.append(f"| {k} | {d['kind']} | {d['pos']} | {d['skel']} | {d['alts']} | {d['ctx']} |")

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n初稿已写入 {out}")
    print(f"需要人工判断 {len(diffs)} 处【】")
    return 0


if __name__ == "__main__":
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
    sys.exit(main())
