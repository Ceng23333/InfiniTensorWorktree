"""Analyze saved clean E2E measurements and complete GPU windows.

No serving configuration is changed. GPU union accounting avoids summing ranks
or double-counting overlapping kernels. Collective alignment requires equal
counts and consistent per-rank ordering. Timings remain tracer-conditioned.
"""
import json,pathlib,statistics,collections,orjson
ROOT=pathlib.Path(__file__).resolve().parents[1]

def load(p):return json.loads((ROOT/p).read_text())
def median(xs):return statistics.median(xs) if xs else None
def union(intervals):
 out=[]
 for a,b in sorted(intervals):
  if out and a<=out[-1][1]:out[-1]=(out[-1][0],max(b,out[-1][1]))
  else:out.append((a,b))
 return out

def duration(intervals):return sum(b-a for a,b in union(intervals))/1e6

def categories(name):
 if 'AllReduce' in name:return 'all_reduce'
 if 'AllGather' in name:return 'all_gather'
 if 'gemv' in name.lower() or 'gemm' in name.lower():return 'matrix_multiply'
 if 'norm' in name.lower():return 'normalization'
 if 'gated_delta' in name.lower():return 'delta_rule'
 if 'conv1d' in name.lower():return 'convolution'
 if 'flash_' in name.lower() or 'attention' in name.lower():return 'attention'
 return 'other'

def window(events,req,tag):
 lo,hi=req['start_ns'],req['end_ns'];by_rank=collections.defaultdict(list)
 for e in events:
  if e.get('pid') in {2,258,514,770} and e.get('ph')=='X' and e.get('dur',-1)>=0 and lo<=e.get('ts',0)<=hi:
   r=e.get('args',{}).get('device_id',(e['pid']-2)//256);by_rank[r].append(e)
 result={'name':tag,'workload':req['workload'],'elapsed_ms':req['elapsed_s']*1000,'completion_tokens':req['completion_tokens'],'traced':True,'per_rank':{}}
 ars={r:sorted((e for e in es if 'AllReduce' in e['name']),key=lambda e:e['ts']) for r,es in by_rank.items()}
 aligned=set(ars)=={0,1,2,3} and len({len(v) for v in ars.values()})==1 and len(ars[0])>0
 for r,es in by_rank.items():
  ints=lambda subset:[(max(lo,e['ts']),min(hi,e['ts']+e['dur'])) for e in subset]
  ar=[e for e in es if 'AllReduce' in e['name']];compute=[e for e in es if 'AllReduce' not in e['name']]
  all_ms=duration(ints(es));ar_ms=duration(ints(ar));other_ms=duration(ints(compute))
  cats=collections.defaultdict(lambda:{'calls':0,'sum_ms':0})
  for e in es:
   c=cats[categories(e['name'])];c['calls']+=1;c['sum_ms']+=e['dur']/1e6
  result['per_rank'][r]={'kernel_calls':len(es),'gpu_active_union_ms':all_ms,'gpu_no_kernel_ms':(hi-lo)/1e6-all_ms,'all_reduce_union_ms':ar_ms,'non_all_reduce_union_ms':other_ms,'ar_other_overlap_ms':ar_ms+other_ms-all_ms,'all_reduce_calls':len(ar),'categories':dict(cats)}
 result['equal_collective_counts']=aligned
 if aligned:
  rounds=[]
  for group in zip(*(ars[r] for r in range(4))):
   last=max(e['ts'] for e in group);end=max(e['ts']+e['dur'] for e in group)
   rounds.append({'arrival_skew_us':(last-min(e['ts'] for e in group))/1e3,'post_last_arrival_us':(end-last)/1e3,'residency_ms':sum(e['dur'] for e in group)/1e6,'pre_last_arrival_ms':sum(max(0,min(last,e['ts']+e['dur'])-e['ts']) for e in group)/1e6,'first_start':min(e['ts'] for e in group),'last_finish':end})
  result['collectives']={'rounds':len(rounds),'arrival_skew_p50_us':median([x['arrival_skew_us'] for x in rounds]),'post_last_arrival_p50_us':median([x['post_last_arrival_us'] for x in rounds]),'sum_residency_ms':sum(x['residency_ms'] for x in rounds),'sum_pre_last_arrival_ms':sum(x['pre_last_arrival_ms'] for x in rounds),'sum_post_last_arrival_ms':sum(x['post_last_arrival_us'] for x in rounds)/1000}
  # InfiniLM Qwen3.5: 64 layers, two reductions per layer. Only report
  # phase grouping when the count exactly agrees with the measured tokens.
  if tag.startswith('infinilm') and len(rounds)==128*req['completion_tokens']:
   steps=[]
   for i in range(req['completion_tokens']):
    block=rounds[i*128:(i+1)*128];start=block[0]['first_start'];end=block[-1]['last_finish']
    steps.append({'phase':'prefill' if i==0 else 'decode','first_ar_start_ns':start,'last_ar_finish_ns':end,'first_to_last_ar_ms':(end-start)/1e6,'post_last_arrival_ms':sum(x['post_last_arrival_us'] for x in block)/1000})
   gaps=[(b['first_ar_start_ns']-a['last_ar_finish_ns'])/1e6 for a,b in zip(steps,steps[1:])]
   result['step_windows']={'steps':steps,'decode_first_to_last_ar_p50_ms':median([s['first_to_last_ar_ms'] for s in steps[1:]]),'between_steps_p50_ms':median(gaps),'note':'AR boundaries omit embeddings, LM head, sampling, input transfer and server work; gaps are not pure CPU overhead.'}
 return result

def main():
 clean={}
 for name,path in [('baseline','results/baseline_host.json'),('register_eager','results/02_register_state_eager/clean/host.json'),('piecewise_graph','results/03_register_state_graph/clean/host.json'),('decode_graph','results/05_corrected_register_decode_graph/clean/host.json'),('corrected_eager','results/06_corrected_integrated_eager/clean/host.json'),('vllm','results/vllm_current/clean/host.json')]:
  d=load(path);rows=d if isinstance(d,list) else d['results'];ds=[r for r in rows if r['workload']=='decode_micro' and not r.get('warmup',False)]
  clean[name]={'source':path,'samples':len(ds),'completion_tokens':[r['completion_tokens'] for r in ds],'median_elapsed_s':median([r['elapsed_s'] for r in ds]),'median_tok_s':median([r['tok_s'] for r in ds]),'fixed_64':all(r['completion_tokens']==64 for r in ds)}
 for v in clean.values():v['slowdown_vs_vllm']=clean['vllm']['median_tok_s']/v['median_tok_s'] if v['fixed_64'] else None
 # Parse and discard each large raw file separately.
 p=ROOT/'results/06_corrected_integrated_eager/traced/trace_attempt1/infinilm-12.json'
 raw=orjson.loads(p.read_bytes());h=load('results/06_corrected_integrated_eager/traced/host.json')
 il=window(raw['traceEvents'],h['results'][1],'infinilm_completed_decode');del raw
 print('InfiniLM request analysis complete',flush=True)
 p=ROOT/'results/vllm_current/traced/gpu_trace.json';raw=orjson.loads(p.read_bytes());h=load('results/vllm_current/traced/host.json')
 vl=window(raw['traceEvents'],h['results'][0],'vllm_eager_warmup');del raw
 out={'clean_e2e':clean,'infinilm_traced_request':il,'vllm_traced_warmup':vl,'limitations':['Clean tok/s is whole-request output throughput, not steady-state decode TPOT.','vLLM trace is eager with JIT warmup; clean vLLM uses default graphs.','InfiniLM completed traced request produced 14 tokens; vLLM produced 64. Do not compare their unnormalized totals.','vLLM later trace windows are incomplete; neither backend has valid traced long-prefill parity.','Model-output correctness diverges; throughput alone does not qualify a release.']}
 (ROOT/'analysis/OUTPUT_THROUGHPUT.json').write_text(json.dumps(out,indent=2));print(json.dumps({'clean':clean,'infinilm_rank0':il['per_rank'][0],'collectives':il['collectives'],'decode_span_ms':il.get('step_windows',{}).get('decode_first_to_last_ar_p50_ms')}),flush=True)
if __name__=='__main__':main()
