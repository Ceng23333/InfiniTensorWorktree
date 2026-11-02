"""Full Layer B EM run with unchanged legacy prompts, adapter and scoring."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--benchmark-root', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--arm', required=True)
    ap.add_argument('--limit', type=int, default=59)
    ap.add_argument('--diagnostic-output', type=int, default=0)
    args = ap.parse_args()
    sys.path.insert(0, str(args.benchmark_root))
    from common.config import load_config, dataset_path, validate
    from transformers import AutoTokenizer
    from workloads.longbench_task_benchmark import build_work_items, score_wave

    cfg_path = args.benchmark_root / 'configs/minicpm5_2b_layerb_qy.json'
    cfg = load_config(cfg_path)
    validate(cfg)
    dataset = dataset_path(cfg)
    rows = [json.loads(line) for line in dataset.read_text().splitlines() if line.strip()][:args.limit]
    if len(rows) != args.limit:
        raise ValueError('requested sample count is unavailable')
    model = cfg['models'][0]
    tokenizer = AutoTokenizer.from_pretrained(model['model_path'], trust_remote_code=True)
    items = build_work_items(cfg, rows, tokenizer, 1, model, 4)
    if args.diagnostic_output:
        for item in items:
            item['case']['output'] = args.diagnostic_output
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    for name in ('cases', 'assets', 'logs', 'infinilm'):
        (out / name).mkdir(exist_ok=True)
    metadata = dict(arm=args.arm, expected_samples=len(rows), expected_waves=len(items),
                    config=json.loads(cfg_path.read_text()), dataset_sha256=hashlib.sha256(dataset.read_bytes()).hexdigest(),
                    scorer_sha256=hashlib.sha256((args.benchmark_root / 'workloads/longbench_task_benchmark.py').read_bytes()).hexdigest(),
                    diagnostic_output_tokens=args.diagnostic_output,
                    started_epoch=time.time(), waves=[], complete=False)
    write(out / 'summary.json', metadata)
    for item in items:
        index, case = item['index'], item['case']
        case_path = out / 'cases' / f'wave{index:03d}.json'
        write(case_path, case)
        write(out / 'assets' / f'wave{index:03d}.json', item['asset'])
        command = [sys.executable, str(Path(__file__).with_name('em_adapter_evidence.py')),
                   str(args.benchmark_root / 'infinilm_benchmark.py'), '--config', str(cfg_path),
                   '--run-dir', str(out), '--case-json', str(case_path), '--prompt-asset', '-']
        started = time.time()
        print(f"{args.arm}: wave {index+1}/{len(items)} starting", flush=True)
        with (out / 'logs' / f'wave{index:03d}.stdout.log').open('w') as stdout, (out / 'logs' / f'wave{index:03d}.stderr.log').open('w') as stderr:
            process = subprocess.run(command, input=json.dumps(item['asset']), text=True,
                                     stdout=stdout, stderr=stderr, timeout=cfg['measurement']['timeout_seconds'] + 300)
        result_path = out / 'infinilm' / f"{case['case_id']}.json"
        result = json.loads(result_path.read_text()) if result_path.exists() else dict(status='missing', requests=[])
        quality = score_wave(item, result)
        valid = process.returncode == 0 and result.get('status') == 'success' and not result.get('teardown_status', '').startswith('failed')
        wave = dict(index=index, valid=valid, exit_code=process.returncode, elapsed_s=time.time()-started,
                    quality=quality, input_hashes=[r['token_ids_sha256'] for r in item['asset']['requests']])
        metadata['waves'].append(wave)
        metadata['matched'] = sum(w['quality']['matched'] for w in metadata['waves'])
        metadata['scored_samples'] = sum(w['quality']['total'] for w in metadata['waves'])
        write(out / 'summary.json', metadata)
        print(f"{args.arm}: wave {index+1} valid={valid}, EM={metadata['matched']}/{metadata['scored_samples']}, seconds={wave['elapsed_s']:.1f}", flush=True)
        if not valid:
            raise RuntimeError(f'invalid wave {index}; see retained logs')
    metadata['complete'] = True
    metadata['completed_epoch'] = time.time()
    write(out / 'summary.json', metadata)


if __name__ == '__main__':
    main()
