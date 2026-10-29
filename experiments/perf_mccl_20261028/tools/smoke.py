import re
from bench import request

def smoke(base):
 rows=[]
 for prompt,expected in [('What is 2 + 2? Reply with only the number.','4'),('What is the capital of France? Reply with only its name.','Paris')]:
  r=request(base,prompt,16);r.update(workload='correctness_smoke',expected=expected)
  r['pass']=r['text'].strip().strip('.').strip()==expected;rows.append(r)
 return {'ok':all(r['pass'] for r in rows),'results':rows}
