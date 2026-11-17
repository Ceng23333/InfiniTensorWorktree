"""Run existing benchmark against a loaded isolated server, then smoke checks."""
import argparse,json,pathlib,sys
from run_candidate import bench
from smoke import smoke
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('output',type=pathlib.Path);p.add_argument('--base',default='http://127.0.0.1:18000');a=p.parse_args()
 import run_candidate
 run_candidate.BASE=a.base;a.output.mkdir(parents=True,exist_ok=True);bench(a.output,3);r=smoke(a.base);(a.output/'smoke.json').write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)
