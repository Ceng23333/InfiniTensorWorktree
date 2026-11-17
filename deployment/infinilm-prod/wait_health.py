import json,time,urllib.request,subprocess,pathlib
out=pathlib.Path(__file__).resolve().parent
for i in range(600):
 try:
  with urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=2) as r:
   if r.status==200:
    with urllib.request.urlopen('http://127.0.0.1:8000/v1/models',timeout=5) as m: models=json.load(m)
    data={'healthy':True,'models':models,'container':json.loads(subprocess.check_output(['docker','inspect','pilotdeck-infinilm-prod']))[0]}
    (out/'deployment.json').write_text(json.dumps(data,indent=2)+'\n')
    print('Backend healthy:',json.dumps(models),flush=True);break
 except Exception: pass
 time.sleep(1)
else: raise SystemExit('Health timeout; inspect production logs')
