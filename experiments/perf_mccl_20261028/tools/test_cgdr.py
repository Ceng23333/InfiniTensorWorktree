#!/usr/bin/env python3
import ctypes as C,json,re,pathlib,time
import torch
# Read the actual launch ABI, rather than duplicating field offsets.
root=pathlib.Path('/perf');src=root/'worktree/InfiniLM/csrc/infinicore/src/ops/chunk_gated_delta_rule'
h=(src/'chunk_gated_delta_rule_metax_launch.h').read_text();h=re.sub(r'/\*.*?\*/','',h,flags=re.S);h=h.split('typedef struct InfiniLMChunkGatedLaunchArgs {')[1].split('}')[0]
fields=[]
for decl in h.split(';'):
 decl=decl.strip()
 if not decl:continue
 if decl.startswith('int '):typ=C.c_int;names=decl[4:]
 elif decl.startswith('ptrdiff_t '):typ=C.c_ssize_t;names=decl[10:]
 else:typ=C.c_void_p;names=decl.split('*')[1]
 fields.extend((n.strip(),typ) for n in names.split(','))
class Args(C.Structure):_fields_=fields
libs=[C.CDLL(str(root/'tools'/f'libcgdr_{name}.so')) for name in ['baseline','register']]
for lib in libs:lib.infinilm_chunk_gated_delta_rule_metax_launch.argtypes=[C.POINTER(Args)]
torch.set_num_threads(1);torch.manual_seed(20261028)
records=[]
for D,T,dtype,indexed in [(64,1,torch.float32,False),(128,31,torch.float32,False),(128,31,torch.bfloat16,True),(128,257,torch.bfloat16,False),(128,2048,torch.bfloat16,False)]:
 Hk,Hv=2,4;B=2 if indexed else 1;pool=4 if indexed else 1
 q=torch.randn(1,T,Hk,D,dtype=dtype).cuda().contiguous();k=torch.randn_like(q);v=torch.randn(1,T,Hv,D,dtype=dtype).cuda().contiguous();g=-torch.rand(1,T,Hv,device='cuda');beta=torch.rand_like(g)
 initial=(torch.randn(pool,Hv,D,D)*.1).to(dtype).cuda().contiguous();cu=torch.tensor([0,T//2,T],device='cuda',dtype=torch.int32) if indexed else None
 ri=torch.tensor([0,1],device='cuda',dtype=torch.int64) if indexed else None;wi=torch.tensor([2,3],device='cuda',dtype=torch.int64) if indexed else None
 # Independent FP32 recurrent reference; initial state layout is [value,key].
 qc,kc,vc,gc,bc=[x.float().cpu() for x in [q,k,v,g,beta]];ic=initial.float().cpu();expected=torch.empty_like(vc);final=ic.clone();bounds=[0,T//2,T] if indexed else [0,T]
 for b in range(B):
  state=ic[b].transpose(-1,-2).contiguous()
  for t in range(bounds[b],bounds[b+1]):
   qt=qc[0,t].repeat_interleave(Hv//Hk,0);kt=kc[0,t].repeat_interleave(Hv//Hk,0)
   qt=qt*torch.rsqrt((qt*qt).sum(-1,keepdim=True)+1e-6)/(D**.5);kt=kt*torch.rsqrt((kt*kt).sum(-1,keepdim=True)+1e-6)
   decay=gc[0,t].exp()[:,None,None];state=state*decay
   delta=(vc[0,t]-(kt[:,:,None]*state).sum(1))*bc[0,t,:,None]
   state=state+kt[:,:,None]*delta[:,None,:];expected[0,t]=(qt[:,:,None]*state).sum(1)
  final[b+2 if indexed else b]=state.transpose(-1,-2)
 outputs=[];times=[]
 for lib in libs:
  a=Args();a.dtype=2 if dtype==torch.bfloat16 else 0;a.gate_dtype=0;a.B=B;a.T=T;a.Hk=Hk;a.Hv=Hv;a.Dk=D;a.Dv=D;a.chunk_size=64;a.pool_size=pool;a.value_heads_per_key_head=Hv//Hk;a.use_qk_l2norm=1;a.has_cu_seqlens=int(indexed);a.indexed_state_pool=int(indexed);a.initial_state_indices_i64=1;a.final_state_indices_i64=1;a.has_final_state=int(not indexed)
  inp=initial.clone().contiguous();out=torch.empty_like(v).contiguous();fin=torch.empty_like(initial).contiguous();ws=torch.empty(B*Hv*D*D,device='cuda');a.stream=torch.cuda.current_stream().cuda_stream
  tensors={'workspace':ws,'out':out,'initial_state':inp,'final_state':fin,'q':q,'k':k,'v':v,'g':g,'beta':beta,'cu_seqlens':cu,'initial_state_indices':ri,'final_state_indices':wi}
  for name,x in tensors.items():setattr(a,name,x.data_ptr() if x is not None else None)
  for name,x in [('out',out),('initial',inp),('final',fin),('q',q),('k',k),('v',v),('g',g),('beta',beta)]:
   for j,stride in enumerate(x.stride()[:4 if name in ['initial','final'] else 3]):setattr(a,f'{name}_s{j}',stride)
   if name in ['initial','final']:a.__setattr__(f'{name}_s3',x.stride(3))
  assert lib.infinilm_chunk_gated_delta_rule_metax_launch(C.byref(a))==0;torch.cuda.synchronize()
  print("checking",lib._name,"D",D,"T",T,flush=True)
  got=out.float().cpu();state=(inp if indexed else fin).float().cpu()
  torch.testing.assert_close(got,expected.to(dtype).float(),atol=.004 if dtype==torch.bfloat16 else 2e-5,rtol=.025 if dtype==torch.bfloat16 else 2e-4)
  sel=slice(2,4) if indexed else slice(None)
  torch.testing.assert_close(state[sel],final[sel].to(dtype).float(),atol=.008 if dtype==torch.bfloat16 else 3e-5,rtol=.025 if dtype==torch.bfloat16 else 2e-4)
  outputs.append(got)
  start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);start.record()
  for _ in range(10):assert lib.infinilm_chunk_gated_delta_rule_metax_launch(C.byref(a))==0
  end.record();end.synchronize();times.append(start.elapsed_time(end)*1000/10)
 row={'D':D,'T':T,'dtype':str(dtype),'indexed':indexed,'baseline_us':times[0],'register_us':times[1],'speedup':times[0]/times[1],'max_diff':(outputs[0]-outputs[1]).abs().max().item(),'reference_pass':True};records.append(row);print(json.dumps(row),flush=True)
(root/'results/cgdr_test.json').write_text(json.dumps(records,indent=2))
