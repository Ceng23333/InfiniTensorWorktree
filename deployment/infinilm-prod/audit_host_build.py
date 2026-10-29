"""Audit effective optimization, expected object coverage, and library hashes."""
import argparse,pathlib,re,json,hashlib
ROOT=pathlib.Path(__file__).resolve().parents[1]

def audit(root=ROOT, baseline=None):
 records=[];selected={}
 for p in sorted((root/'worktree/InfiniLM/build/.deps').rglob('*.o.d')):
  values=p.read_text(errors='replace').rsplit('values =',1)[-1]
  flags=re.findall(r'"([^"\n]+)"',values)
  if not flags or not any('/g++' in f or '/gcc' in f for f in flags):continue
  opts=[f for f in flags if re.fullmatch(r'-O(?:[0-3sgz]|fast)?',f)]
  row={'path':str(p.relative_to(root)),'compiler':flags[0],'optimization_flags':opts,'effective_optimization':opts[-1] if opts else '-O0 (compiler default)','target':p.relative_to(root/'worktree/InfiniLM/build/.deps').parts[0],'ndebug':'-DNDEBUG' in flags,'fast_math':'-ffast-math' in flags or '-Ofast' in opts}
  records.append(row)
  if p.name in ['random_sample_metax.cc.o.d','rank_worker.cpp.o.d','graph.cc.o.d']:selected[row['path']]=values
 summary={}
 for target in sorted({r['target'] for r in records}):
  rows=[r for r in records if r['target']==target];summary[target]={'objects':len(rows),'without_optimization_flag':sum(not r['optimization_flags'] for r in rows),'not_optimized':sum(r['effective_optimization'] not in ['-O2','-O3'] for r in rows)}
 missing=[]
 if baseline:
  old=json.loads(pathlib.Path(baseline).read_text());missing=sorted({r['path'] for r in old['objects']}-{r['path'] for r in records})
 good=bool(records) and {'_infinilm','_infinicore','infinicore_runtime'}<=set(summary) and not missing and all(r['effective_optimization'] in ['-O2','-O3'] and not r['fast_math'] for r in records)
 return {'optimized':good,'summary':summary,'missing_baseline_objects':missing,'representative_compile_values':selected,'objects':records}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=pathlib.Path,default=ROOT/'analysis/HOST_BUILD_AUDIT.json');p.add_argument('--baseline',type=pathlib.Path);p.add_argument('--require-optimized',action='store_true');a=p.parse_args();r=audit(baseline=a.baseline);a.output.write_text(json.dumps(r,indent=2));print(json.dumps({k:v for k,v in r.items() if k not in ['objects','representative_compile_values']},indent=2))
 if a.require_optimized and not r['optimized']:raise SystemExit(1)
