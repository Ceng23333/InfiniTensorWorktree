#!/usr/bin/env python3
"""Validated HTTP benchmark with epoch-ns request windows for mcTracer."""
import argparse,json,time,urllib.request,pathlib,statistics

def request(base, prompt, max_tokens, stream=False):
 payload={'model':'Qwen3.8-27B','temperature':0,'max_tokens':max_tokens,'stream':stream,'messages':[{'role':'user','content':prompt}], 'chat_template_kwargs':{'enable_thinking':False}}
 if stream:payload['stream_options']={'include_usage':True}
 req=urllib.request.Request(base+'/v1/chat/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
 start_ns=time.time_ns();t0=time.perf_counter();first=None;text='';usage={};done=False
 with urllib.request.urlopen(req,timeout=120) as resp:
  if stream:
   for raw in resp:
    line=raw.decode().strip()
    if not line.startswith('data:'):continue
    body=line[5:].strip()
    if body=='[DONE]':done=True;break
    d=json.loads(body)
    if d.get('error'):raise RuntimeError(str(d['error']))
    if d.get('usage'):usage=d['usage']
    for c in d.get('choices',[]):
     piece=c.get('delta',{}).get('content') or ''
     if piece and first is None:first=time.perf_counter()-t0
     text+=piece
  else:
   d=json.load(resp)
   if d.get('error'):raise RuntimeError(str(d['error']))
   usage=d.get('usage') or {};text=d['choices'][0]['message'].get('content') or '';done=True
 elapsed=time.perf_counter()-t0
 if not done or not text or usage.get('completion_tokens',0)<=0 or usage.get('prompt_tokens',0)<=0 or (stream and first is None):
  raise RuntimeError(f'incomplete response: done={done}, text={text!r}, usage={usage}, ttft={first}')
 return {'start_ns':start_ns,'end_ns':time.time_ns(),'elapsed_s':elapsed,'ttft_s':first,'tok_s':usage['completion_tokens']/elapsed,**usage,'text':text}

def main():
 p=argparse.ArgumentParser();p.add_argument('--base',default='http://127.0.0.1:8000');p.add_argument('--output',required=True);p.add_argument('--repeats',type=int,default=3);p.add_argument('--prefill',action='store_true');a=p.parse_args()
 results=[];out=pathlib.Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
 def run(name,prompt,n,stream=False):
  r=request(a.base,prompt,n,stream);r['workload']=name;results.append(r);out.write_text(json.dumps({'ok':False,'phase':'in_progress','results':results},indent=2));print(json.dumps(r),flush=True)
 for _ in range(a.repeats):run('decode_micro','Output the word pad exactly 64 times, separated by spaces. No other text.',64)
 if a.prefill:
  from profile.workload_chat import _pad_prompt
  run('prefill_12k',_pad_prompt(48000),64,True)
 out.write_text(json.dumps({'ok':True,'results':results},indent=2))
if __name__=='__main__':main()
