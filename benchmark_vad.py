"""VAD 切段 vs 整段的对比测评（长音频）。

用法:
    # 先下载测试音频（272s，sherpa-onnx 官方 release）
    curl -L -o test_audio/lei-jun-test.wav https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/lei-jun-test.wav
    python benchmark_vad.py test_audio/lei-jun-test.wav

结果同时打印并写入 benchmark/vad_results.json。详见 docs/long-audio-report.md。
"""
import sys, time, json
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
from asr_engine import read_wav, AsrModel, find_downloaded
import registry

AUDIO = sys.argv[1] if len(sys.argv) > 1 else 'test_audio/lei-jun-test.wav'
samples, sr = read_wav(open(AUDIO, 'rb').read())
dur = len(samples)/sr
files = find_downloaded()
rows = []
for spec in registry.ASR_MODELS:
    k = spec['key']
    if k not in files: continue
    m = AsrModel(spec, files[k], 2)
    m.load()
    # VAD 模式
    t0=time.perf_counter()
    out = m._offline(samples, sr, use_vad=True) if not spec['streaming'] else m._online(samples, sr, use_vad=True)
    el=time.perf_counter()-t0
    # 非 VAD 对照
    t1=time.perf_counter()
    out2 = m._offline(samples, sr, use_vad=False) if not spec['streaming'] else m._online(samples, sr, use_vad=False)
    el2=time.perf_counter()-t1
    print(f"### {k} streaming={spec['streaming']}", flush=True)
    print(f"  VAD   : {el:6.1f}s RTF {el/dur:.4f} first={out.get('first_partial_at')} segs={out.get('n_segments')}", flush=True)
    print(f"    text: {out['text'][:110]}", flush=True)
    print(f"  noVAD : {el2:6.1f}s RTF {el2/dur:.4f} first={out2.get('first_partial_at')}", flush=True)
    print(f"    text: {out2['text'][:110]}", flush=True)
    rows.append(dict(key=k, streaming=spec['streaming'], vad=el, vad_rtf=round(el/dur,4),
        no_vad=el2, no_vad_rtf=round(el2/dur,4),
        vad_text=out['text'], no_vad_text=out2['text'],
        vad_first=out.get('first_partial_at'), no_vad_first=out2.get('first_partial_at'),
        segs=out.get('n_segments')))
json.dump(rows, open('benchmark/vad_results.json','w'), ensure_ascii=False, indent=1)
