#!/usr/bin/env python3
"""Request-window trace analysis; never equate summed device time with E2E."""
import argparse,json,pathlib,statistics,collections
from profile.parse_mctracer import _load_chrome_json
GPUS={2,258,514,770,1026,1282,1538,1794}
def analyze(trace,host):
 raw=_load_chrome_json(pathlib.Path(trace));es=raw['traceEvents'];results=[]
 for req in host['results']:
  lo,hi=req['start_ns'],req['end_ns'];groups=collections.defaultdict(list);ranks=collections.defaultdict(list)
  for e in es:
   if e.get('pid') not in GPUS or e.get('ph')!='X' or e.get('dur',-1)<0 or not lo<=e.get('ts',0)<=hi:continue
   rank=e.get('args',{}).get('device_id',(e['pid']-2)//256)
   groups[(rank,e['name'])].append(e)
   if 'AllReduce'in e['name']:ranks[rank].append(e)
  per_gpu={}
  for rank in sorted({k[0] for k in groups}):
   top=[]
   for (r,name),events in groups.items():
    if r!=rank:continue
    ds=sorted(e['dur']/1e3 for e in events)
    top.append({'name':name,'count':len(ds),'total_ms':sum(ds)/1000,'p50_us':statistics.median(ds),'p95_us':ds[int(.95*(len(ds)-1))]})
   top.sort(key=lambda x:x['total_ms'],reverse=True)
   ar=[x for x in top if 'AllReduce'in x['name']]
   per_gpu[rank]={'kernel_ms':sum(x['total_ms'] for x in top),'ar_ms':sum(x['total_ms'] for x in ar),'ar_count':sum(x['count'] for x in ar),'top_kernels':top[:20]}
  skew=None
  if len(ranks)==4 and len({len(v) for v in ranks.values()})==1:
   lists=[sorted(v,key=lambda e:e['ts']) for v in ranks.values()];arrival=[];tail=[];wait=[]
   for group in zip(*lists):
    last=max(e['ts'] for e in group);end=max(e['ts']+e['dur'] for e in group)
    arrival.append((last-min(e['ts'] for e in group))/1e3);tail.append((end-last)/1e3)
    wait.append(sum(max(0,min(e['ts']+e['dur'],last)-e['ts']) for e in group)/1e6)
   skew={'matched_collectives':len(arrival),'arrival_skew_p50_us':statistics.median(arrival),'arrival_skew_p95_us':sorted(arrival)[int(.95*(len(arrival)-1))],'last_arrival_to_last_finish_p50_us':statistics.median(tail),'sum_pre_last_arrival_ms':sum(wait),'note':'ordinal alignment assumes identical collective order; GPU residency includes rank waiting'}
  ar_counts=[g['ar_count'] for g in per_gpu.values()]
  coverage_ok=set(per_gpu)=={0,1,2,3} and bool(ar_counts) and min(ar_counts)>0 and len(set(ar_counts))==1
  results.append({'workload':req['workload'],'elapsed_s':req['elapsed_s'],'per_gpu':per_gpu,'rank_skew':skew,'coverage_ok':coverage_ok})
 return {'trace':str(trace),'timestamp_unit':'ns','truncated':bool(raw.get('truncated')),'results':results,'valid':not raw.get('truncated') and bool(results) and all(r['coverage_ok'] for r in results)}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('trace');p.add_argument('host');p.add_argument('-o',required=True);a=p.parse_args();r=analyze(a.trace,json.loads(pathlib.Path(a.host).read_text()));pathlib.Path(a.o).write_text(json.dumps(r,indent=2));print(json.dumps({k:v for k,v in r.items() if k!='results'}))
