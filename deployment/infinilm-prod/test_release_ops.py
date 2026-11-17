import sys,pathlib
sys.path=[p for p in sys.path if pathlib.Path(p or '.').resolve()!=pathlib.Path(__file__).resolve().parent]
import numpy as np
import infinicore as ic
from infinicore import ops
ic.set_device(ic.device('metax',0))
a_np=np.arange(32,dtype=np.float32).reshape(4,8)/16
b_np=np.arange(48,dtype=np.float32).reshape(8,6)/32
a=ic.from_numpy(a_np,device='metax:0');b=ic.from_numpy(b_np,device='metax:0')
np.testing.assert_allclose(ops.matmul(a,b).to_numpy(),a_np@b_np,rtol=1e-5,atol=1e-5)
np.testing.assert_allclose(ops.add(a,a).to_numpy(),a_np*2,rtol=1e-6)
weight=ic.from_numpy(np.ones(8,dtype=np.float32),device='metax:0')
np.testing.assert_allclose(ops.rms_norm(a,weight,1e-5).to_numpy(),a_np/np.sqrt(np.mean(a_np*a_np,axis=-1,keepdims=True)+1e-5),rtol=1e-5,atol=1e-5)
np.testing.assert_array_equal(ops.cat([a,a],dim=0).to_numpy(),np.concatenate([a_np,a_np],axis=0))
print('PASS: optimized MetaX matmul, add, RMSNorm, concat and host/device transfers')
