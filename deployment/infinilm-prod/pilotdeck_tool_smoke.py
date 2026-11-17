"""Live Pilotdeck smoke: execute tools, inspect their arguments, verify disk output."""
import argparse,asyncio,inspect,json,pathlib,shlex,subprocess,time,uuid
import requests,websockets

async def run(out):
 out.mkdir(parents=True,exist_ok=False)
 host_root=pathlib.Path('/home/metax/qy')
 project='/workspace/'+str(out.resolve().relative_to(host_root))
 http=requests.Session();http.trust_env=False
 result={'passed':False,'project':project,'cases':[]}
 def save(): (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 http.post('http://127.0.0.1:3001/api/projects/create-workspace',json={'workspaceType':'existing','path':project},timeout=15).raise_for_status()
 cases=[('bash','bash-proof.txt','PILOTDECK_BASH_OK_20261029\n',
         'You must invoke the Bash tool; do not print or simulate the command. Execute exactly this command: '+"printf 'PILOTDECK_BASH_OK_20261029\\n' > bash-proof.txt; cat bash-proof.txt"+'. Then reply DONE. Do not use any other tools.'),
        ('write_read','write-proof.txt','工具调用验证\nPILOTDECK_WRITE_READ_OK_20261029',
         'You must invoke the Write and Read tools; do not simulate their results. Use Write to create write-proof.txt with exactly these two lines, one newline between them and no trailing newline:\n工具调用验证\nPILOTDECK_WRITE_READ_OK_20261029\nThen use Read to read that file. After reading reply DONE and stop. Use only Write and Read.')]
 for name,filename,expected,prompt in cases:
  row={'name':name,'passed':False};events=[];started=time.monotonic()
  try:
   kwargs={'max_size':32*1024*1024}
   if 'proxy' in inspect.signature(websockets.connect).parameters:kwargs['proxy']=None
   async with websockets.connect('ws://127.0.0.1:3001/ws',**kwargs) as ws:
    await ws.send(json.dumps({'type':'pilotdeck-command','command':prompt,'options':{'projectPath':project,'cwd':project,'workspaceCwd':project,'runMode':'agent','permissionMode':'bypassPermissions','modelSelection':{'mode':'model','provider':'openai','model':'Qwen3.8-27B'},'userVisibleInput':prompt,'runId':str(uuid.uuid4()),'sessionSummary':'InfiniLM tool smoke '+name}}))
    deadline=time.monotonic()+300
    while True:
     e=json.loads(await asyncio.wait_for(ws.recv(),max(.1,deadline-time.monotonic())));events.append(e)
     (out/(name+'-events.json')).write_text(json.dumps(events,ensure_ascii=False,indent=2))
     if e.get('kind')=='error' or e.get('kind')=='complete':break
   calls=[e for e in events if e.get('kind')=='tool_use']
   responses=[e for e in events if e.get('kind')=='tool_result']
   errors=[e for e in responses if e.get('isError')]
   complete=next((e for e in reversed(events) if e.get('kind')=='complete'),{})
   file=out/filename;contents=file.read_text() if file.exists() else None
   names={e.get('toolName','').lower() for e in calls}
   required={'bash'} if name=='bash' else {'write','read'}
   # Some Pilotdeck releases expose canonical tool names.
   if 'write_file' in names:names.add('write')
   if 'read_file' in names:names.add('read')
   arguments_ok=bool(calls) and all(bool(e.get('toolInput')) for e in calls)
   completed_ids={e.get('toolId') for e in responses if not e.get('isError') and e.get('content') != 'Full tool result is available.'}
   tools_completed=bool(calls) and all(e.get('toolId') in completed_ids for e in calls)
   proof_visible=any(('PILOTDECK_BASH_OK_20261029' if name=='bash' else 'PILOTDECK_WRITE_READ_OK_20261029') in str(e.get('content','')) for e in responses)
   row.update(seconds=time.monotonic()-started,session_id=next((e['sessionId'] for e in events if e.get('sessionId')),None),tool_calls=calls,tool_results=responses,tool_errors=errors,file_content=contents,expected_content=expected,required_tools_present=required<=names,arguments_nonempty=arguments_ok,tools_completed=tools_completed,proof_visible_in_tool_result=proof_visible,complete=complete)
   row['passed']=bool(complete) and complete.get('success',True) and complete.get('exitCode',0)==0 and not errors and required<=names and arguments_ok and tools_completed and proof_visible and contents==expected
  except Exception as e:row.update(error=repr(e),seconds=time.monotonic()-started)
  result['cases'].append(row);save();print(name,'PASS' if row['passed'] else 'FAIL',flush=True)
  if not row['passed']:break
 r=http.get('http://127.0.0.1:8000/health',timeout=5)
 result['backend_healthy']=r.status_code==200
 result['passed']=len(result['cases'])==len(cases) and all(c['passed'] for c in result['cases']) and result['backend_healthy'];save()
 return result['passed']
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('output',type=pathlib.Path);a=p.parse_args()
 raise SystemExit(0 if asyncio.run(run(a.output)) else 1)
