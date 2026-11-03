"""基于语义的切段：VAD 细切 + 文本完整度判断 + 自适应静音等待。

为什么需要：
    纯声学 VAD 只看停顿。人说话“想一下”的停顿（“我想……明天去公园”）会被切成两句，
    机器人会在用户没说完时就抢话；而把静音阈值调大又会让每句话都多等很久才响应。

做法（和实时系统的决策过程一致，可直接搬到端侧常驻链路）：
    1. VAD 用短静音（0.25s）把音频细切成片段 fragment。
    2. 每个片段结束时，识别该片段文字，结合“当前这句已累计的文字”判断完整度：
         complete   —— 以句末标点/语气词结尾（。？！吗呢吧……）→ 静音 SHORT_WAIT 就断句
         incomplete —— 以连接词/填充词/逗号结尾（然后、就是、那个、如果…的话、，）→ 要等 LONG_WAIT
         uncertain  —— 其他                                                     → 等 MID_WAIT
    3. 当前片段后面的静音 ≥ 对应等待时长 → 断句；否则与下一片段合并。
       句首规则：如果下一片段以承接词开头（然后/所以/但是/就…）且停顿 < LONG_WAIT，
       把它并回上一句（实时链路里相当于“刚才断早了”，至少保证缓存里的句子完整）。
       累计时长超过 MAX_UTT 时强制断句（防止长段独白无限增长）。
    4. 合并后的整句再整体识别一次，作为最终文本（上下文更完整）。

输出的整句时间段列表可以直接作为 segs 传给 AsrModel.transcribe()，与 VAD 段列表格式相同。
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------- 完整度规则
# 句末标点（来自 SenseVoice 或标点模型）
_END_PUNCT = "。？！?!…"
# 句末语气词：出现在句尾基本说明一句话说完了
_END_PARTICLES = ("吗", "呢", "吧", "啦", "呀", "嘛", "哦", "喔", "哈", "咯", "了吧", "好吗", "行吗", "可以吗",
                  "对吧", "是吧", "好不好", "行不行", "对不对", "谢谢", "晚安", "再见")
# 句尾出现这些词，说明话没说完（连接词、填充词、悬空的介词/动词）
_INCOMPLETE_TAILS = (
    "然后", "就是", "就是说", "那个", "这个", "那么", "因为", "所以", "所以说", "但是", "可是", "不过", "而且",
    "还有", "以及", "或者", "还是", "如果", "要是", "假如", "的话", "比如", "比如说", "然后呢", "我想", "我觉得",
    "我要", "我想要", "帮我", "给我", "请", "你帮我", "把", "给", "跟", "和", "与", "被", "让", "等", "等等",
    "嗯", "呃", "额", "啊那", "就", "也", "又", "再", "先", "去", "在", "从", "对于", "关于", "其实", "首先",
    "接着", "最后", "另外", "一个", "一下", "怎么", "为什么", "什么时候", "的时候", "之后", "以后", "之前", "以前",
)
# 条件/因果的前半句：出现了引导词但没有后半句
_OPEN_CLAUSE = [(re.compile(r"(如果|要是|假如|万一)"), re.compile(r"(就|那|则|的话.+)")),
                (re.compile(r"因为"), re.compile(r"(所以|才|就)")),
                (re.compile(r"虽然"), re.compile(r"(但|可是|不过|还是)"))]

# 片段以这些词开头，说明是在接上一句（“我今天去超市……然后买了水果”）
# 只收强承接词；「因为」「就是」「还有」「也」「又」在口语里常用来开启新的一句，不算
_CONTINUE_HEADS = ("然后", "所以", "但是", "可是", "不过", "而且", "并且", "以及", "或者", "因此", "那就",
                   "接着", "之后", "顺便", "结果", "就", "再")
_HEAD_STRIP_RE = re.compile(r"^[\s\u3000，,。.、！!？?嗯呃额啊]+")

_STRIP_RE = re.compile(r"[\s　，,、；;：:\"'“”‘’（）()\[\]【】<>《》]+$")


def completeness(text: str) -> str:
    """判断累计文本的完整度：'complete' | 'incomplete' | 'uncertain'。"""
    t = text.strip()
    if not t:
        return "incomplete"
    # 逗号/顿号结尾：明确未完
    if t[-1] in "，,、；;：:":
        return "incomplete"
    core = _STRIP_RE.sub("", t).rstrip(_END_PUNCT)
    if not core:
        return "uncertain"
    # 连接词/填充词结尾优先级最高：SenseVoice 会给每个片段补“。”，不能只看标点
    for w in sorted(_INCOMPLETE_TAILS, key=len, reverse=True):
        if core.endswith(w):
            return "incomplete"
    # 条件/因果句只有前半句
    for opener, closer in _OPEN_CLAUSE:
        m = opener.search(core)
        if m and not closer.search(core[m.end():]):
            return "incomplete"
    if t[-1] in _END_PUNCT:
        return "complete"
    if core.endswith(_END_PARTICLES):
        return "complete"
    return "uncertain"


def continues(text: str) -> bool:
    """片段是否以承接词开头（是上一句的延续）。"""
    t = _HEAD_STRIP_RE.sub("", text)
    if t.startswith(("再见", "就是", "就这样", "就好", "就行")):
        return False
    return bool(t) and t.startswith(_CONTINUE_HEADS)


# ---------------------------------------------------------------- 切段
@dataclass
class SemanticParams:
    short_wait: float = 0.30   # 完整句：静音这么久就断句
    mid_wait: float = 0.60     # 不确定
    long_wait: float = 1.20    # 未完句：要等更久（用户可能在想）
    max_utt: float = 20.0      # 单句最长时长
    pad: float = 0.20          # 识别片段时前后补的音频


@dataclass
class Utterance:
    start: float
    end: float
    frag_ids: list = field(default_factory=list)
    frag_texts: list = field(default_factory=list)
    decision: str = ""         # 断句时的完整度判断
    wait: float = 0.0          # 断句前实际等待的静音时长（= 响应延迟中的“断句等待”部分）
    reason: str = ""           # complete / uncertain / incomplete / max_len / gap / end
    rejoined: int = 0          # 因句首承接词被并回的次数（实时链路里相当于“断早了，补救”）


def _clip(samples, sr, a, b, prev_end, next_start, pad):
    lo = a - pad if prev_end is None else max(a - pad, (prev_end + a) / 2)
    hi = b + pad if next_start is None else min(b + pad, (b + next_start) / 2)
    return samples[max(0, int(lo * sr)):min(len(samples), int(hi * sr))]


def segment(samples: np.ndarray, frags: list[tuple[float, float]], transcribe_fn, sr: int = 16000,
            params: SemanticParams | None = None, punct_fn=None) -> tuple[list[Utterance], dict]:
    """把 VAD 细片段按语义合并成整句。

    transcribe_fn(audio) -> str：识别一个片段（决策用）。
    punct_fn(text) -> str：可选，给累计文本加标点（Paraformer 没有标点时用）。
    返回 (整句列表, 统计信息)。
    """
    p = params or SemanticParams()
    t0 = time.perf_counter()
    utts: list[Utterance] = []
    cur: Utterance | None = None
    asr_ms = punct_ms = 0.0
    for i, (a, b) in enumerate(frags):
        prev_end = frags[i - 1][1] if i > 0 else None
        next_start = frags[i + 1][0] if i + 1 < len(frags) else None
        c0 = time.perf_counter()
        text = transcribe_fn(_clip(samples, sr, a, b, prev_end, next_start, p.pad))
        asr_ms += (time.perf_counter() - c0) * 1000
        # 句首规则：新片段以承接词开头、且离上一句不远 → 并回上一句
        if (cur is None and utts and continues(text) and a - utts[-1].end < p.long_wait
                and b - utts[-1].start <= p.max_utt):
            cur = utts.pop()
            cur.rejoined += 1
        if cur is None:
            cur = Utterance(start=a, end=b)
        cur.end = b
        cur.frag_ids.append(i)
        cur.frag_texts.append(text)

        acc = "".join(cur.frag_texts)
        if punct_fn and acc and acc[-1] not in _END_PUNCT + "，,":
            c0 = time.perf_counter()
            acc = punct_fn(acc)
            punct_ms += (time.perf_counter() - c0) * 1000
        state = completeness(acc)
        need = {"complete": p.short_wait, "uncertain": p.mid_wait, "incomplete": p.long_wait}[state]
        gap = (next_start - b) if next_start is not None else float("inf")

        if next_start is None:
            cur.decision, cur.reason, cur.wait = state, "end", need
        elif cur.end - cur.start + (frags[i + 1][1] - next_start) > p.max_utt:
            cur.decision, cur.reason, cur.wait = state, "max_len", min(gap, need)
        elif gap >= need:
            cur.decision, cur.reason, cur.wait = state, state, need
        else:
            continue  # 静音不够长，和下一片段合并
        utts.append(cur)
        cur = None
    stats = {"n_frags": len(frags), "n_utts": len(utts), "asr_ms": round(asr_ms, 1),
             "punct_ms": round(punct_ms, 1), "total_ms": round((time.perf_counter() - t0) * 1000, 1)}
    return utts, stats


def fixed_gap_merge(frags: list[tuple[float, float]], gap: float, max_utt: float = 20.0) -> list[Utterance]:
    """对照组：只按固定静音阈值合并（纯声学）。gap=0 时等价于直接用 VAD 片段。"""
    utts: list[Utterance] = []
    for i, (a, b) in enumerate(frags):
        if utts and a - utts[-1].end < gap and b - utts[-1].start <= max_utt:
            utts[-1].end = b
            utts[-1].frag_ids.append(i)
        else:
            if utts:
                utts[-1].wait = gap
            utts.append(Utterance(start=a, end=b, frag_ids=[i], reason="gap"))
    if utts:
        utts[-1].wait = gap
    return utts


def spans(utts: list[Utterance]) -> list[tuple[float, float]]:
    """整句时间段，可直接作为 segs 传给 AsrModel.transcribe()。"""
    return [(u.start, u.end) for u in utts]
