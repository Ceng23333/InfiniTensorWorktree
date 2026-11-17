import ast,pathlib,types,sys,unittest
import torch
from unittest.mock import patch
p=pathlib.Path(__file__).resolve().parents[1]/'worktree/InfiniLM/python/infinilm/processors/qwen3_5_processor.py'
tree=ast.parse(p.read_text());method=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_append_qwen35_position_ids')
ns={};exec(compile(ast.Module(body=[method],type_ignores=[]),str(p),'exec'),ns)
class Processor:
 _append_qwen35_position_ids=ns['_append_qwen35_position_ids']
 def _update_request_mrope_delta(self,r):pass
 def _mrope_delta(self,r):return 7
 def _build_mrope_position_ids(self,r,start,length):return torch.arange(start,start+length).expand(3,-1)
def req(n,offset=0,cached=0):return types.SimpleNamespace(num_local_cached_tokens=cached,chunk_prefill_offset=offset,get_input_tokens=lambda:list(range(n)),get_prefill_tokens=lambda:list(range(n)),get_total_length=lambda:n)
class Positions(unittest.TestCase):
 def run_input(self,rows,requests,prefill):
  out={}
  with patch.dict(sys.modules,{'infinicore':types.SimpleNamespace(from_torch=lambda t:t)}):Processor()._append_qwen35_position_ids(out,types.SimpleNamespace(rows=rows,scheduled_requests=requests,is_prefill=prefill))
  return out['position_ids']
 def test_continuation_chunk(self):
  r=req(10718,2048);p=self.run_input([types.SimpleNamespace(request=r,is_prefill_row=True,num_scheduled_tokens=2048)],[r],True)
  self.assertEqual(tuple(p.shape),(3,2048));self.assertEqual(p[0,0],2048);self.assertEqual(p[0,-1],4095)
 def test_mixed_rows(self):
  a,b=req(33),req(10718,10240);p=self.run_input([types.SimpleNamespace(request=a,is_prefill_row=False,num_scheduled_tokens=1),types.SimpleNamespace(request=b,is_prefill_row=True,num_scheduled_tokens=2048)],[a,b],True)
  self.assertEqual(tuple(p.shape),(3,479));self.assertEqual(p[0,0],39);self.assertEqual(p[0,1],10240);self.assertEqual(p[0,-1],10717)
 def test_legacy(self):
  r=req(20,cached=5);p=self.run_input([], [r],True);self.assertEqual(tuple(p.shape),(3,15));self.assertEqual(p[0,0],5)
class StateRows(unittest.TestCase):
 def test_initial_continuation_and_decode(self):
  method=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='build_model_inputs')
  method.decorator_list=[]
  for arg in method.args.args:arg.annotation=None
  class Output:pass
  ns={'SchedulerOutput':Output,'StaticSchedulerOutput':type('StaticOutput',(),{})}
  exec(compile(ast.Module(body=[method],type_ignores=[]),str(p),'exec'),ns)
  processor=types.SimpleNamespace(_build_model_input_from_batch_scheduler_output=lambda *a:{},_append_qwen35_mm_inputs=lambda *a:None,_append_qwen35_position_ids=lambda *a:None)
  requests=[req(10718),req(10718,2048),req(33)]
  for i,r in enumerate(requests):r.mamba_cache_index=i+1
  out=Output();out.scheduled_requests=requests;out.is_prefill=True;out.rows=[types.SimpleNamespace(is_prefill_row=v) for v in [True,True,False]]
  with patch.dict(sys.modules,{'infinicore':types.SimpleNamespace(from_list=lambda x,**k:x,int32='int32')}):
   d=ns['build_model_inputs'](processor,out)
  self.assertEqual(d['mamba_init_state_indices'],[0,2,3]);self.assertEqual(d['mamba_final_state_indices'],[1,2,3])
if __name__=='__main__':unittest.main()
