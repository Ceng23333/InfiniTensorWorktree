#!/usr/bin/env python3
import json,pathlib,statistics
R=pathlib.Path(__file__).resolve().parents[1]
q=json.loads((R/'queue.json').read_text());lines=['# Measured iterations','',f"Current vLLM reference: {q['reference']['decode_tok_s']:.3f} tok/s; {q['reference']['prefill_ttft_s']:.3f} s TTFT. Prefix caching off; prompt lengths match (30 and 10,718).",'', '| Candidate | Decode tok/s | Prefill TTFT (s) | Phase / verdict |','| --- | ---: | ---: | --- |']
for c in q['candidates']:
 p=R/'results'/c['name']/'state.json'
 if not p.exists():continue
 d=json.loads(p.read_text());t=d.get('decode_tok_s');f=d.get('prefill_ttft_s')
 lines.append(f"| {c['name']} | {t:.3f} | {f:.3f} | {d['phase']}; {'PASS' if d['ok'] else d.get('error','not accepted')} |" if t is not None and f is not None else f"| {c['name']} | — | — | {d['phase']}; {d.get('error','pending')} |")
lines+=['','Throughput is completion tokens / entire request elapsed time, excluding the first warmup request. Early EOS/refusal means token counts differ; such candidates are not accepted for fixed-length decode comparisons. See raw host JSON and smoke outputs. Device graph candidates generated incorrect text and were rejected.','', 'Kernel-only delta-rule validation: 2,048-token BF16 100.986 ms → 20.253 ms (4.99×), tested outputs identical to the original. Splitting value columns alone regressed long-sequence timing and was rejected; warp reductions added only about 9% in the microbenchmark and were not integrated.','', 'The optimization that improved clean host prefill measured 28.765 s → 8.235 s (71.4% less). This does not establish full-model accuracy; the current vLLM reference itself fails the simple arithmetic smoke probe.']
(R/'RESULTS.md').write_text('\n'.join(lines)+'\n')
