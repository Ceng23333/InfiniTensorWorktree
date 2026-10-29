import json,pathlib
R=pathlib.Path(__file__).resolve().parents[1]
for rel in ['results/06_corrected_integrated_eager/traced/partial_decode_analysis.json','results/vllm_current/traced/analysis.json','results/07_prefill_only/analysis.json']:
 p=R/rel
 if not p.exists():continue
 d=json.loads(p.read_text());print(rel,'valid',d['valid'],'partial',d.get('partial_capture'))
 for r in d['results']:
  print(r['workload'],'elapsed',round(r['elapsed_s'],3),'skew',r['rank_skew'])
  g=r['per_gpu'].get('0');
  if g:
   print('GPU0 total_ms',round(g['kernel_ms'],3),'AR_ms',round(g['ar_ms'],3),'AR_count',g['ar_count'])
   for k in g['top_kernels'][:8]:print(round(k['total_ms'],3),k['count'],round(k['p50_us'],3),k['name'][:125])
