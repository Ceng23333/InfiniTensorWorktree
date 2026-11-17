import json,pathlib,tempfile,unittest,sys
from profile.parse_mctracer import parse_mctracer
from analyze import analyze
class ProfileTests(unittest.TestCase):
 def test_units_and_invalid_durations(self):
  for ts,dur,expected_ns in [(1793182661937181855,7424,True),(1000,7.424,False)]:
   with tempfile.TemporaryDirectory() as d:
    p=pathlib.Path(d)/'trace.json';p.write_text(json.dumps({'traceEvents':[{'pid':2,'ph':'X','ts':ts,'dur':dur,'name':'AllReduce'},{'pid':2,'ts':ts,'dur':-1,'name':'bad'}]}))
    s=parse_mctracer(p);self.assertAlmostEqual(s.allreduce_ms,.007424);self.assertEqual(s.event_counts['timestamp_unit_ns'],expected_ns)
 def test_incomplete_trace_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   p=pathlib.Path(d)/'trace.json';p.write_text('{"traceEvents":[{"pid":2,"ph":"X","ts":100,"dur":10,"name":"AllReduce"},')
   s=analyze(p,{'results':[{'start_ns':90,'end_ns':150,'workload':'test','elapsed_s':1}]});self.assertFalse(s['valid']);self.assertTrue(s['truncated'])
 def test_request_window_and_rank_wait(self):
  es=[{'pid':2+256*r,'ph':'X','ts':100+r*10,'dur':50-r*10,'name':'AllReduce'} for r in range(4)]
  es.append({'pid':2,'ph':'X','ts':1,'dur':999,'name':'load'})
  with tempfile.TemporaryDirectory() as d:
   p=pathlib.Path(d)/'trace.json';p.write_text(json.dumps({'traceEvents':es}));s=analyze(p,{'results':[{'start_ns':90,'end_ns':200,'workload':'test','elapsed_s':1}]})
   self.assertTrue(s['valid']);r=s['results'][0];self.assertAlmostEqual(r['rank_skew']['arrival_skew_p50_us'],.03);self.assertEqual(r['per_gpu'][0]['ar_count'],1)
if __name__=='__main__':unittest.main()
