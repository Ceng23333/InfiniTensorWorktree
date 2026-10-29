import ast,pathlib,unittest
import torch
p=pathlib.Path(__file__).resolve().parents[1]/'worktree/InfiniLM/python/infinilm/modeling_utils.py';tree=ast.parse(p.read_text());method=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_remap_qwen3_5')
ns={'torch':torch,'drop_keys':lambda d,keys:{k:v for k,v in d.items() if not any(k.startswith(x) for x in keys)}};exec(compile(ast.Module(body=[method],type_ignores=[]),str(p),'exec'),ns)
class NormWeights(unittest.TestCase):
 def test_language_and_legacy_final_norm(self):
  keys=['model.norm.weight','model.language_model.norm.weight','model.language_model.layers.0.input_layernorm.weight','model.language_model.layers.0.linear_attn.norm.weight','model.visual.merger.norm.weight']
  source={k:torch.tensor([-.25,.5]) for k in keys};out=ns['_remap_qwen3_5'](source,{'text_config':{'linear_key_head_dim':128,'linear_num_key_heads':16}})
  for k in keys[:3]:torch.testing.assert_close(out[k],torch.tensor([.75,1.5]))
  for k in keys[3:]:torch.testing.assert_close(out[k],torch.tensor([-.25,.5]))
if __name__=='__main__':unittest.main()
