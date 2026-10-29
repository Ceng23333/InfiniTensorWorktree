import pathlib,re,json,hashlib,collections
ROOT=pathlib.Path(__file__).resolve().parents[1]
records=[]
for p in sorted((ROOT/'worktree/InfiniLM/build/.deps').rglob('*.o.d')):
 text=p.read_text(errors='replace');values=text.rsplit('values =',1)[-1]
 flags=re.findall(r'"([^"\n]+)"',values)
 if not flags or not any('/g++' in f or '/gcc' in f for f in flags):continue
 records.append({'path':str(p.relative_to(ROOT)),'compiler':flags[0],'optimization_flags':[f for f in flags if re.fullmatch(r'-O(?:[0-3sgz]|fast)',f)],'target':p.relative_to(ROOT/'worktree/InfiniLM/build/.deps').parts[0]})
summary={}
for target in sorted({r['target'] for r in records}):
 rows=[r for r in records if r['target']==target];summary[target]={'objects':len(rows),'without_optimization_flag':sum(not r['optimization_flags'] for r in rows)}
selected={}
for p in (ROOT/'worktree/InfiniLM/build/.deps').rglob('*.o.d'):
 if p.name in ['random_sample_metax.cc.o.d','rank_worker.cpp.o.d','graph.cc.o.d']:
  selected[str(p.relative_to(ROOT))]=p.read_text(errors='replace').rsplit('values =',1)[-1]
old=json.loads((ROOT/'results/06_corrected_integrated_eager/binary_sha256.json').read_text())
match={p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()==sha for p,sha in old.items()}
out={'summary':summary,'representative_compile_values':selected,'binary_matches_measured_iteration_06':match,'objects':records}
(ROOT/'analysis/HOST_BUILD_AUDIT.json').write_text(json.dumps(out,indent=2));print(json.dumps({'summary':summary,'binary_matches':match},indent=2))
