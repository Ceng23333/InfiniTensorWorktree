#!/usr/bin/env python3
"""Capture prefill in its own fresh process, with no preceding decode trace."""
import pathlib,sys,json,subprocess,time
from run_candidate import ROOT,start,stop,BASE
from bench import request
from profile.workload_chat import _pad_prompt
from analyze import analyze
out=ROOT/'results/08_prefill_kernel_only';out.mkdir(parents=True,exist_ok=False)
state={'phase':'startup','ok':False,'execution':'eager','capture_workload':'prefill_12k_only'}
try:
 stop();trace=start({'ENABLE_GRAPH':'0','INFINI_CUDAGRAPH_POLICY':'eager','MACA_ETC_PATH':'/perf/tools/tracer_config'},out,True);state['phase']='request'
 (out/'state.json').write_text(json.dumps(state,indent=2))
 r=request(BASE,_pad_prompt(48000),64,True);r.update(workload='prefill_12k',warmup=False);h={'ok':True,'results':[r]};(out/'host.json').write_text(json.dumps(h,indent=2))
 subprocess.run(['docker','exec','infinilm-perf-loop','pkill','-TERM','-f','^/opt/conda/bin/python python/infinilm/server/inference_server.py'],check=False)
 for _ in range(30):
  if subprocess.run(['docker','exec','infinilm-perf-loop','pgrep','-x','mcTracer'],stdout=subprocess.DEVNULL).returncode:break
  time.sleep(1)
 files=list((out/trace).glob('*.json'))
 if len(files)!=1:raise RuntimeError('missing or ambiguous trace')
 a=analyze(files[0],h);(out/'analysis.json').write_text(json.dumps(a,indent=2));state['ok']=a['valid'];state['phase']='complete'
except Exception as e:state.update(phase='failed',error=str(e))
finally:
 (out/'state.json').write_text(json.dumps(state,indent=2));stop();print(json.dumps(state),flush=True)
