#!/usr/bin/env python3
"""One resumable build/clean-bench/traced-bench/analyze cycle on the isolated tree.

Called by loop.py for each source revision + runtime configuration. Never mutates
or promotes the production pin. A failed gate is recorded, not a successful run.
"""
import argparse,subprocess,json,pathlib,time,urllib.request,os,signal,statistics,hashlib,re
from bench import request
from smoke import smoke
from profile.workload_chat import _pad_prompt
from analyze import analyze
ROOT=pathlib.Path(__file__).resolve().parents[1]
PIN='/home/metax/qy/workspace/pilotdeck/InfiniTensorWorktree-v20261023-qy-minicpm5-2b-full-piecewise'
CTR='infinilm-perf-loop';BASE='http://127.0.0.1:18000'
def cmd(args,**kw):return subprocess.run(args,check=True,**kw)
def stop():
 # Stop only the experiment container. A restart reaps startup processes that
 # ignore SIGTERM, so the next run cannot compete with a surviving model.
 cmd(['docker','restart','--time','5',CTR],stdout=subprocess.DEVNULL)
def start(env,out,traced,attempt=1):
 argv=['docker','exec','-d']
 for k,v in env.items():argv+=['-e',f'{"PERF_KERNEL_PRELOAD" if k == "LD_PRELOAD" else k}={v}']
 log='/perf/'+str(out.relative_to(ROOT))+'/server.log'
 trace='/perf/'+str(out.relative_to(ROOT))
 # All interpolated values are quoted, including paths supplied by the caller.
 import shlex
 target=['bash','/perf/tools/launch.sh']
 trace_dir=f'trace_attempt{attempt}'
 if traced:target=['/opt/maca/bin/mcTracer','--mctx','--odname',trace_dir,'--name','infinilm']+target
 shell=f'cd {shlex.quote(trace)}; exec {shlex.join(target)} >{shlex.quote(log)} 2>&1'
 cmd(argv+[CTR,'bash','-c',shell])
 for i in range(600):
  try:
   urllib.request.urlopen(BASE+'/v1/models',timeout=1);return trace_dir
  except Exception:
   if i%10==0:
    ps=subprocess.run(['docker','exec',CTR,'pgrep','-f','^/opt/conda/bin/python python/infinilm/server/inference_server.py'],capture_output=True,text=True)
    if i>10 and ps.returncode:
     if traced and attempt<3:
      stop()
      (out/'server.log').rename(out/f'startup_attempt{attempt}.log')
      return start(env,out,traced,attempt+1)
     raise RuntimeError('server exited during startup')
   time.sleep(1)
 raise TimeoutError('readiness timeout')
def bench(out,repeats):
 rows=[]
 for i in range(repeats+1):
  r=request(BASE,'Output the word pad exactly 64 times, separated by spaces. No other text.',64);r.update(workload='decode_micro',warmup=i==0);rows.append(r)
  (out/'host.json').write_text(json.dumps({'ok':False,'phase':'in_progress','results':rows},indent=2))
 r=request(BASE,_pad_prompt(48000),64,True);r.update(workload='prefill_12k',warmup=False);rows.append(r)
 d={'ok':True,'results':rows};(out/'host.json').write_text(json.dumps(d,indent=2));return d
def cycle(name,env,trace=True,rebuild=True,adopt_clean=False):
 out=ROOT/'results'/name
 out.mkdir(parents=True,exist_ok=False)
 state={'name':name,'env':env,'phase':'rebuild','started_ns':time.time_ns(),'ok':False}
 state['source_sha']=subprocess.check_output(['git','-C',str(ROOT/'worktree'),'rev-parse','HEAD'],text=True).strip()
 (out/'source.diff').write_bytes(subprocess.check_output(['git','-C',str(ROOT/'worktree'),'diff','HEAD']))
 state['source_diff_sha256']=hashlib.sha256((out/'source.diff').read_bytes()).hexdigest()
 def save(phase):state['phase']=phase;(out/'state.json').write_text(json.dumps(state,indent=2))
 save('rebuild')
 try:
  if not adopt_clean:stop()
  if rebuild and not adopt_clean:
   with (out/'build.log').open('w') as log:cmd(['docker','exec',CTR,'bash','/perf/tools/rebuild.sh'],stdout=log,stderr=subprocess.STDOUT)
  binaries={}
  for relative in ['install/lib/libinfinicore_runtime.so','install/lib/libinfinilm_metax_extra.so','install/lib/libinfiniccl.so','install/lib/libinfiniops.so','InfiniLM/python/infinilm/lib/_infinilm.cpython-310-x86_64-linux-gnu.so']:
   binary=ROOT/'worktree'/relative
   if binary.exists():binaries[relative]=hashlib.sha256(binary.read_bytes()).hexdigest()
  (out/'binary_sha256.json').write_text(json.dumps(binaries,indent=2))
  for mode in ['clean']+(['traced'] if trace else []):
   sub=out/mode;sub.mkdir();save(mode)
   trace_dir=None
   if not (mode=='clean' and adopt_clean):trace_dir=start(env,sub,mode=='traced')
   h=bench(sub,3 if mode=='clean' else 1)
   log_text=(sub/'server.log').read_text(errors='replace') if (sub/'server.log').exists() else ''
   if re.search(r'Illegal memory|Memory violation|Traceback|exception during forward', log_text, re.I):raise RuntimeError('server log error gate failed')
   if mode=='clean':
    dec=[r for r in h['results'] if r['workload']=='decode_micro' and not r['warmup']]
    state['decode_tok_s']=statistics.median(r['tok_s'] for r in dec);state['prefill_ttft_s']=h['results'][-1]['ttft_s']
    # This fixed prompt must actually generate pad; nonempty nonsense is not a pass.
    sm=smoke(BASE);(sub/'smoke.json').write_text(json.dumps(sm,indent=2))
    state['correctness_ok']=sm['ok']
    state['fixed_decode_length_ok']=all(r['completion_tokens']==64 for r in dec)
   # Ask the traced application to shut down, allowing mcTracer to flush JSON.
   subprocess.run(['docker','exec',CTR,'pkill','-TERM','-f','^/opt/conda/bin/python python/infinilm/server/inference_server.py'],check=False)
   for _ in range(30):
    alive=subprocess.run(['docker','exec',CTR,'pgrep','-x','mcTracer'],stdout=subprocess.DEVNULL).returncode==0
    if not alive:break
    time.sleep(1)
   if mode=='traced':
    traces=list((sub/trace_dir).glob('*.json'))
    if len(traces)!=1:raise RuntimeError(f'expected one fresh trace, found {len(traces)}')
    a=analyze(traces[0],h);(sub/'analysis.json').write_text(json.dumps(a,indent=2));state['trace_valid']=a['valid']
    if not a['valid']:raise RuntimeError('trace incomplete or request GPU events missing')
   stop()
  state['ok']=state['correctness_ok'] and state['fixed_decode_length_ok'] and (state.get('trace_valid',False) if trace else True);save('complete')
 except Exception as e:state['error']=str(e);save('failed');stop()
 return state
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('name');p.add_argument('--env',type=json.loads,default={});p.add_argument('--no-trace',action='store_true');a=p.parse_args();print(json.dumps(cycle(a.name,a.env,not a.no_trace),indent=2))
