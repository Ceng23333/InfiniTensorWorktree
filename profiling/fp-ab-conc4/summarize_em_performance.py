"""Validate a complete EM arm and aggregate inference-only request metrics."""
import argparse
import json
import math
from pathlib import Path
import statistics

from compare_em_gate import samples


def load(path):
    return json.loads(path.read_text())


def distribution(values):
    values = sorted(values)
    if not values or any(not math.isfinite(x) or x < 0 for x in values):
        raise ValueError('invalid timing samples')

    def percentile(p):
        position = (len(values) - 1) * p
        low = math.floor(position)
        high = math.ceil(position)
        return values[low] + (values[high] - values[low]) * (position - low)

    return dict(mean=statistics.mean(values), p50=percentile(.5), p95=percentile(.95))


def summarize(root, expected=59):
    summary = load(root / 'summary.json')
    scored = samples(summary, expected)
    requests, waves = [], []
    fingerprints = None
    for wave in summary['waves']:
        stem = f"wave{wave['index']:03d}"
        case = load(root / 'cases' / f'{stem}.json')
        asset = load(root / 'assets' / f'{stem}.json')
        result = load(root / 'infinilm' / f"{case['case_id']}.json")
        if result['status'] != 'success' or result.get('errors') or result.get('teardown_status', '').startswith('failed'):
            raise ValueError(f'invalid result: {stem}')
        if result['effective_config']['input_token_ids_sha256'] != wave['input_hashes']:
            raise ValueError(f'input fingerprint mismatch: {stem}')
        if not result['runtime_settings']['graph']:
            raise ValueError(f'graph disabled: {stem}')
        if case['output'] != 512 or case['ignore_eos'] or case['batch_size'] > 4:
            raise ValueError(f'unexpected workload: {stem}')
        current = result['requests']
        if len(current) != len(asset['requests']) or len(current) != wave['quality']['total']:
            raise ValueError(f'missing request: {stem}')
        for request, source in zip(current, asset['requests']):
            if request['request_id'] != source['request_id']:
                raise ValueError(f'reordered result: {stem}')
            if not 0 < request['output_tokens_actual'] <= case['output']:
                raise ValueError(f'invalid output length: {stem}')
            if request['output_tokens_actual'] != len(request['output_token_ids']):
                raise ValueError(f'inconsistent output tokens: {stem}')
        loaded = load(root / 'maps' / f'{stem}.json')
        if not loaded or not all(any(name in path for path in loaded) for name in
                                 ('_infinilm.', '_infinicore.', 'libinfinicore_runtime.so', 'libinfiniops.so')):
            raise ValueError(f'missing library evidence: {stem}')
        if fingerprints is not None and loaded != fingerprints:
            raise ValueError(f'libraries changed between waves: {stem}')
        fingerprints = loaded
        tokens = sum(r['output_tokens_actual'] for r in current)
        rate = result['metrics']['output_tokens_per_second']
        if not math.isfinite(rate) or rate <= 0:
            raise ValueError(f'invalid throughput: {stem}')
        duration = tokens / rate
        requests.extend(current)
        waves.append(dict(index=wave['index'], samples=len(current), matched=wave['quality']['matched'],
                          output_tokens=tokens, measured_duration_s=duration, output_tok_s=rate,
                          ttft_ms=result['metrics']['ttft_ms'],
                          mean_request_latency_s=result['metrics']['end_to_end_latency_seconds']))
    total_output = sum(w['output_tokens'] for w in waves)
    total_time = sum(w['measured_duration_s'] for w in waves)
    matched = sum(row['exact_match'] for row in scored.values())
    if len(requests) != expected or summary['matched'] != matched or summary['scored_samples'] != expected:
        raise ValueError('inconsistent summary totals')
    return dict(complete=True, samples=expected, matched=matched, em=matched / expected,
                unscorable=sum(r['prediction'] is None for r in scored.values()),
                prompt_truncated=sum(r['prompt_truncated'] for r in scored.values()),
                output_budget_reached=sum(r['output_tokens_actual'] == 512 for r in requests),
                total_output_tokens=total_output, measured_duration_s=total_time,
                aggregate_output_tok_s=total_output / total_time,
                ttft_ms=distribution([r['ttft_ms'] for r in requests]),
                request_latency_s=distribution([r['end_to_end_latency_seconds'] for r in requests]),
                run_wall_s=summary['completed_epoch'] - summary['started_epoch'],
                loaded_libraries=fingerprints, waves=waves,
                notes=['Inference timing excludes model load, graph capture and warmup.',
                       'Aggregate throughput = total output tokens / sum of measured wave makespans.',
                       'Percentiles use linear interpolation across requests.',
                       'EM uses the unchanged legacy last-isolated-A/B/C/D scorer.',
                       'A full single-arm result is not a paired numerical-equivalence test.'])


def compare_historical(root, report, reference):
    current = samples(load(root / 'summary.json'), report['samples'])
    previous = load(reference)
    comparisons = {}
    backend_names = set.intersection(*(set(case['backends']) for case in previous['results']['cases']))
    for backend in sorted(backend_names):
        rows = [case['backends'][backend] for case in previous['results']['cases']]
        if any(row['status'] != 'success' for row in rows):
            raise ValueError(f'invalid historical backend: {backend}')
        old = {}
        for row in rows:
            for sample in row['quality']['samples']:
                if sample['sample_id'] in old:
                    raise ValueError('duplicate historical sample')
                old[sample['sample_id']] = sample
        if old.keys() != current.keys():
            raise ValueError('historical sample IDs differ')
        for key, sample in current.items():
            for field in ('answer', 'prompt_tokens', 'original_prompt_tokens', 'prompt_truncated'):
                if sample[field] != old[key][field]:
                    raise ValueError(f'historical {field} differs for {key}')
        lost = [key for key in old if old[key]['exact_match'] and not current[key]['exact_match']]
        gained = [key for key in old if current[key]['exact_match'] and not old[key]['exact_match']]
        tokens = sum(row['output_tokens_actual'] for row in rows)
        duration = sum(row['output_tokens_actual'] / row['metrics']['output_tokens_per_second'] for row in rows)
        old_rate = tokens / duration
        comparisons[backend] = dict(
            matched=sum(row['exact_match'] for row in old.values()),
            prediction_matches=sum(row['prediction'] == current[key]['prediction'] for key, row in old.items()),
            lost_correct=lost, gained_correct=gained, output_tokens=tokens,
            measured_duration_s=duration, aggregate_output_tok_s=old_rate,
            current_throughput_ratio=report['aggregate_output_tok_s'] / old_rate,
            mean_ttft_ms=sum(row['metrics']['ttft_ms'] * row['quality']['total'] for row in rows) / len(old),
            mean_request_latency_s=sum(row['metrics']['end_to_end_latency_seconds'] * row['quality']['total'] for row in rows) / len(old))
    return dict(source=str(reference), backends=comparisons,
                caveat='Historical context only: sample IDs, answers and prompt lengths match, but this summary lacks per-input hashes; runtime configurations and dates differ. Throughput uses rounded historical metrics and output counts can differ.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('arm_directory', type=Path)
    parser.add_argument('--expected', type=int, default=59)
    parser.add_argument('--reference', type=Path)
    args = parser.parse_args()
    report = summarize(args.arm_directory, args.expected)
    if args.reference:
        report['historical_comparison'] = compare_historical(args.arm_directory, report, args.reference)
    destination = args.arm_directory.parent / 'em-performance.json'
    destination.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('waves', 'loaded_libraries')}, indent=2))


if __name__ == '__main__':
    main()
