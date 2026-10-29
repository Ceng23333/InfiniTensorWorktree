#!/usr/bin/env python3
import subprocess,time,urllib.request,json,pathlib,sys,collections
from bench import request
from profile.workload_chat import _pad_prompt
from profile.parse_mctracer import _load_chrome_json
from analyze import analyze
ROOT=pathlib.Path(__file__).resolve().parents[1];out=ROOT/'results/vllm_current/traced';out.mkdir(parents=True,exist_ok=False)
# Both backend kernel captures use eager execution. Clean vLLM retains its
# default graphs, and clean/traced metrics are deliberately kept separate.
subprocess.run(['docker','exec','-d','-e','VLLM_ENFORCE_EAGER=1','vllm-perf-loop','bash','-c','cd /perf/results/vllm_current/traced; exec /opt/maca/bin/mcTracer --mctx --odname trace --name vllm bash /perf/tools/vllm_launch.sh > server.log 2>&1'],check=True)
try:
 for _ in range(600):
  try:urllib.request.urlopen('http://127.0.0.1:28000/v1/models',timeout=1);break
  except Exception:time.sleep(1)
 else:raise TimeoutError('vllm tracer startup')
 rows=[]
 for i in range(2):
  r=request('http://127.0.0.1:28000','Output the word pad exactly 64 times, separated by spaces. No other text.',64);r.update(workload='decode_micro',warmup=i==0);rows.append(r)
 r=request('http://127.0.0.1:28000',_pad_prompt(48000),64,True);r.update(workload='prefill_12k',warmup=False);rows.append(r)
 h={'ok':True,'execution':'eager','results':rows};(out/'host.json').write_text(json.dumps(h,indent=2))
 subprocess.run(['docker','exec','vllm-perf-loop','pkill','-TERM','-f','^python -m vllm.entrypoints.openai.api_server'],check=False)
 for _ in range(60):
  if subprocess.run(['docker','exec','vllm-perf-loop','pgrep','-x','mcTracer'],stdout=subprocess.DEVNULL).returncode:break
  time.sleep(1)
 events=[];sources=[];truncated=False
 for path in sorted((out/'trace').glob('*.json')):
  raw=_load_chrome_json(path);es=raw.get('traceEvents',[]);gpu=[e for e in es if e.get('pid') in {2,258,514,770} and e.get('ph')=='X']
  if not gpu:continue
  ranks=collections.Counter()
  for e in es:
   if e.get('pid')==5 and e.get('name')=='mcclAllReduce':
    for item in e.get('args',{}).get('payload',[]):
     if item.get('desc')=='rank id':ranks[item['value']]+=1
  if len(ranks)!=1:raise RuntimeError(f'cannot establish unambiguous TP rank for {path}: {ranks}')
  rank=next(iter(ranks));sources.append({'path':str(path),'rank':rank,'gpu_events':len(gpu),'truncated':bool(raw.get('truncated'))});truncated|=bool(raw.get('truncated'))
  for e in gpu:e['pid']=2+256*rank;e.setdefault('args',{})['device_id']=rank
  events.extend(gpu)
 merged=out/'gpu_trace.json';merged.write_text(json.dumps({'traceEvents':events,'truncated':truncated}))
 a=analyze(merged,h);a['source_shards']=sources;a['execution']='eager';(out/'analysis.json').write_text(json.dumps(a,indent=2));print(json.dumps({'valid':a['valid'],'shards':sources}),flush=True)
finally:
 subprocess.run(['docker','restart','--time','5','vllm-perf-loop'],check=True,stdout=subprocess.DEVNULL)
