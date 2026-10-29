import pathlib,json,collections,statistics,orjson
ROOT=pathlib.Path(__file__).resolve().parents[1]
d=json.loads((ROOT/'analysis/OUTPUT_THROUGHPUT.json').read_text());steps=d['infinilm_traced_request']['step_windows']['steps'];gaps=[(a['last_ar_finish_ns'],b['first_ar_start_ns']) for a,b in zip(steps,steps[1:])]
es=orjson.loads((ROOT/'results/06_corrected_integrated_eager/traced/trace_attempt1/infinilm-12.json').read_bytes())['traceEvents'];groups=collections.defaultdict(list);dma=[];gpu=[]
for e in es:
 if e.get('ph')!='X' or e.get('dur',-1)<0:continue
 if not any(a<=e.get('ts',0)<b for a,b in gaps):continue
 if e.get('pid')==1:groups[(e['name'],e.get('tid'))].append(e)
 elif e.get('pid') in {3,259,515,771}:dma.append(e)
 elif e.get('pid') in {2,258,514,770}:gpu.append(e)
rows=[]
for (name,tid),vals in groups.items():
 ds=sorted(e['dur']/1e6 for e in vals);rows.append({'api':name,'tid':tid,'count':len(ds),'sum_ms':sum(ds),'median_ms':statistics.median(ds),'max_ms':max(ds)})
rows.sort(key=lambda r:r['sum_ms'],reverse=True)
byname=collections.defaultdict(list)
for e in gpu:byname[e['name']].append(e['dur']/1e6)
out={'gap_count':len(gaps),'gap_sum_ms':sum(b-a for a,b in gaps)/1e6,'api_by_thread':rows,'gpu_kernels_in_gaps':[{'name':n,'count':len(v),'sum_ms':sum(v),'p50_ms':statistics.median(v)} for n,v in sorted(byname.items(),key=lambda x:sum(x[1]),reverse=True)],'dma_events':[{'pid':e['pid'],'name':e['name'],'bytes':e.get('args',{}).get('bytes'),'duration_us':e['dur']/1000} for e in dma],'note':'API durations can nest and overlap across threads; sums are diagnostic, not additive latency attribution.'}
(ROOT/'analysis/STEP_GAPS.json').write_text(json.dumps(out,indent=2));print(json.dumps({k:v[:15] if isinstance(v,list) else v for k,v in out.items()},indent=2))
