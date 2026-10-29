#!/usr/bin/env python3
"""Consume reviewed local candidates until both vLLM proximity gates pass.

Each candidate may apply a source patch, then rebuild, benchmark clean and under
mcTracer, and analyze. Failed patches are reverted with git apply -R (never reset).
The queue can be extended while running; --watch waits for new concrete candidates.
"""
import argparse,json,pathlib,time,subprocess,fcntl,signal
from run_candidate import ROOT,cycle,cmd,stop

def main():
 p=argparse.ArgumentParser();p.add_argument('--queue',type=pathlib.Path,default=ROOT/'queue.json');p.add_argument('--watch',action='store_true');p.add_argument('--restore-production',action='store_true');p.add_argument('--decode-ratio',type=float,default=.8);p.add_argument('--ttft-ratio',type=float,default=1.25);a=p.parse_args()
 lock=(ROOT/'loop.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 ledger=ROOT/'results/ledger.json';history=json.loads(ledger.read_text()) if ledger.exists() else []
 completed={x['name'] for x in history};active=subprocess.check_output(['docker','inspect','pilotdeck-infinilm-prod','--format','{{.State.Running}}'],text=True).strip()=='true' or a.restore_production
 def interrupted(*_):raise KeyboardInterrupt()
 signal.signal(signal.SIGTERM,interrupted)
 try:
  if active:cmd(['docker','stop','--time','20','pilotdeck-infinilm-prod'])
  while True:
   queue=json.loads(a.queue.read_text());pending=[x for x in queue['candidates'] if x['name'] not in completed]
   if not pending:
    if not a.watch:break
    if active:
     cmd(['docker','start','pilotdeck-infinilm-prod'],stdout=subprocess.DEVNULL)
    time.sleep(10);continue
   if active:cmd(['docker','stop','--time','20','pilotdeck-infinilm-prod'],stdout=subprocess.DEVNULL)
   c=pending[0];patch=ROOT/c['patch'] if c.get('patch') else None
   applied=False
   if patch:
    check=subprocess.run(['git','-C',str(ROOT/'worktree'),'apply','--check',str(patch)],capture_output=True)
    if check.returncode==0:
     cmd(['git','-C',str(ROOT/'worktree'),'apply',str(patch)]);applied=True
    else:
     cmd(['git','-C',str(ROOT/'worktree'),'apply','-R','--check',str(patch)])
   result=cycle(c['name'],c['env'])
   ref=queue['reference'];result['reference']=ref
   result['decode_ratio']=result.get('decode_tok_s',0)/ref['decode_tok_s']
   result['ttft_ratio']=result['prefill_ttft_s']/ref['prefill_ttft_s'] if result.get('prefill_ttft_s') is not None else None
   result['target_met']=result['ok'] and result['decode_ratio']>=a.decode_ratio and result['ttft_ratio']<=a.ttft_ratio
   history.append(result);ledger.write_text(json.dumps(history,indent=2));completed.add(c['name']);print(json.dumps(result),flush=True)
   # Keep source changes only when their own correctness + capture gates pass.
   if applied and not result['ok']:cmd(['git','-C',str(ROOT/'worktree'),'apply','-R',str(patch)])
   if result['target_met']:break
 finally:
  stop()
  if active:cmd(['docker','start','pilotdeck-infinilm-prod'])
if __name__=='__main__':main()
