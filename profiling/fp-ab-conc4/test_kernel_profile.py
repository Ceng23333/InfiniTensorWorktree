import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from bench_client import request_one
import bench_client
from summarize_kernel_micro import summarize

ROOT = Path(__file__).resolve().parent


class KernelProfileTests(unittest.TestCase):
    def test_prompt_limit_applies_to_both_batches(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prompts = root / 'prompts.json'
            output = root / 'result.json'
            prompts.write_text(json.dumps(dict(target_prompt_tokens=512,
                prompts=[dict(id=str(i), text='test') for i in range(8)])))
            result = dict(makespan_s=1, output_chunks=1, output_tokens_per_s=1, ttft_ms=dict(p50=1))
            argv = ['bench_client', '--base-url', 'http://localhost', '--prompts', str(prompts),
                    '--output', str(output), '--mode', 'full_and_piecewise', '--max-prompts', '1']
            with patch.object(sys, 'argv', argv), patch.object(bench_client, 'run_batch', return_value=result) as run, patch('builtins.print'):
                bench_client.main()
            self.assertEqual(run.call_count, 2)
            self.assertTrue(all(len(call.args[1]) == 1 for call in run.call_args_list))
            self.assertEqual(json.loads(output.read_text())['workload']['requests'], 1)

    def test_rope_kv_matrix_rejects_bitwise_mismatch(self):
        rows = [dict(operation=op, dtype=dtype, tokens=n, padded=pad, interleaved=inter,
                     correctness='passed', hashes=['same'])
                for dtype in ('bf16', 'fp16', 'fp32') for n in (1, 4, 512)
                for pad in (False, True) for op in ('rope_qk', 'kv_cache')
                for inter in ((False, True) if op == 'rope_qk' else (False,))]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for variant in ('installed', 'candidate_off', 'candidate_on'):
                (root / f'{variant}.json').write_text(json.dumps(dict(rows=rows)))
            command = [sys.executable, str(ROOT / 'compare_rope_kv_checks.py'), str(root)]
            self.assertEqual(subprocess.run(command, capture_output=True).returncode, 0)
            rows[0]['hashes'] = ['different']
            (root / 'candidate_on.json').write_text(json.dumps(dict(rows=rows)))
            self.assertEqual(subprocess.run(command, capture_output=True).returncode, 1)

    def test_micro_timings_use_device_intervals(self):
        events = [dict(kind=56, start=t, elapsed=2000, kernel_name='norm',
                       grid_size=[4, 1, 1], block_size=[1024, 1, 1]) for t in (5, 12, 15, 20)]
        checks = dict(rows=[dict(engine='infinilm', tokens=4, operation='add_rmsnorm',
                                samples=[dict(start_ns=10, end_ns=20)])])
        rows = summarize(dict(activities=events), checks)
        self.assertEqual(rows[0]['device_us_mean'], 4)
        self.assertEqual(rows[0]['kernels'][0]['count'], 2)

    def test_optional_generated_text(self):
        lines = [b'data: {"choices":[{"delta":{"content":"hello"}}]}\n',
                 b'data: {"choices":[{"delta":{"content":" world"},"finish_reason":"length"}]}\n']
        for include in (False, True):
            with patch('urllib.request.urlopen') as response:
                response.return_value.__enter__.return_value = lines
                result = request_one('http://localhost', dict(id='test', text='prompt'), 2,
                                     include_output_text=include)
                self.assertEqual(result['output_chunks'], 2)
                self.assertEqual('output_text' in result, include)
                if include:
                    self.assertEqual(result['output_text'], 'hello world')

    def test_windows_and_launch_shapes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            events = [dict(kind=56, start=t, elapsed=1000, kernel_name='AddKernel',
                           grid_size=[g, 1, 1], block_size=[1024, 1, 1])
                      for t, g in [(999999999, 1), (1000000000, 4), (1500000000, 512), (2000000000, 1)]]
            (root / 'trace.json').write_text('{"activities": [\n' + ',\n'.join(
                json.dumps(event, separators=(',', ':')) for event in events) + '\n]}\n')
            (root / 'run.json').write_text(json.dumps(dict(measured_wall=dict(start_epoch=1, end_epoch=2))))
            subprocess.run([sys.executable, str(ROOT / 'kernel_shape_profile.py'),
                '--trace', str(root / 'trace.json'), '--runs', str(root / 'run.json'),
                '--output', str(root / 'out.json')], check=True, capture_output=True)
            result = json.loads((root / 'out.json').read_text())
            measured = [r for r in result['kernels'] if r['phase'] == 'measured']
            self.assertEqual(len(measured), 2)
            self.assertAlmostEqual(result['kind_sum_ms']['measured']['56'], .002)
            self.assertEqual(sum(r['count'] for r in result['kernels']), 4)

    def validate_fixture(self, count, log):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'vllm-checks.json').write_text(json.dumps(dict(rows=[dict(correctness='passed')] * 6)))
            (root / 'vllm-profiler.log').write_text(log)
            with sqlite3.connect(root / 'vllm.metric') as conn:
                conn.execute('create table kprof_metric (value real)')
                conn.executemany('insert into kprof_metric values (?)', [(1,)] * count)
            result = subprocess.run([sys.executable, str(ROOT / 'validate_kernel_profile.py'),
                str(root), '--mode', 'kprof'], capture_output=True)
            return result.returncode

    def test_empty_counters_are_not_success(self):
        self.assertEqual(self.validate_fixture(0, 'Capture completed'), 1)

    def test_child_failure_is_not_success(self):
        self.assertEqual(self.validate_fixture(3, 'ERROR: Capture failed: -1'), 1)

    def test_populated_clean_capture(self):
        self.assertEqual(self.validate_fixture(3, 'Pass 1 of 2\nPass 2 of 2\nCapture completed'), 0)

    def test_partial_passes_are_not_success(self):
        self.assertEqual(self.validate_fixture(3, 'Pass 1 of 9\nCapture completed'), 1)


if __name__ == '__main__':
    unittest.main()
