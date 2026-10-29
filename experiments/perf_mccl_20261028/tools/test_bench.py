import io,json,unittest
from unittest.mock import patch
from bench import request
class BenchTests(unittest.TestCase):
 def run_stream(self,data):
  with patch('urllib.request.urlopen',return_value=io.BytesIO(data.encode())):
   return request('http://unused','test',8,True)
 def test_empty_stream_rejected(self):
  with self.assertRaises(RuntimeError):self.run_stream('data: [DONE]\n')
 def test_error_event_rejected(self):
  with self.assertRaisesRegex(RuntimeError,'worker failed'):self.run_stream('data: {"error":"worker failed"}\n')
 def test_partial_stream_rejected(self):
  with self.assertRaises(RuntimeError):self.run_stream('data: {"choices":[{"delta":{"content":"pad"}}],"usage":{"prompt_tokens":10,"completion_tokens":1}}\n')
 def test_complete_stream(self):
  r=self.run_stream('data: {"choices":[{"delta":{"content":"pad"}}],"usage":{"prompt_tokens":10,"completion_tokens":1}}\ndata: [DONE]\n')
  self.assertEqual(r['completion_tokens'],1);self.assertEqual(r['text'],'pad');self.assertIsNotNone(r['ttft_s'])
if __name__=='__main__':unittest.main()
