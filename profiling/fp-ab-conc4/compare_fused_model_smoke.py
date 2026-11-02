#!/usr/bin/env python3
"""Summarize short model A/Bs without treating latency as a quality gate."""
import argparse
import json
from pathlib import Path
import statistics


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('directory', type=Path)
    args = ap.parse_args()
    runs = {}
    result = {}
    for arm in ('baseline', 'fused'):
        runs[arm] = {p.stem.removeprefix(arm + '-'): json.loads(p.read_text())['measured']
                     for p in sorted(args.directory.glob(arm + '-run*.json'))}
        if not runs[arm]:
            raise ValueError(f'no measured runs for {arm}')
        texts = {}
        for run in runs[arm].values():
            for item in run['items']:
                texts.setdefault(item['id'], set()).add(item['output_text'])
        result[arm] = dict(runs=len(runs[arm]),
                          median_makespan_s=statistics.median(x['makespan_s'] for x in runs[arm].values()),
                          median_p50_ttft_ms=statistics.median(x['ttft_ms']['p50'] for x in runs[arm].values()),
                          distinct_outputs_per_prompt={k: len(v) for k, v in texts.items()})
    if runs['baseline'].keys() != runs['fused'].keys():
        raise ValueError('run sets differ')
    pairs = []
    for key in runs['baseline']:
        baseline = {item['id']: item for item in runs['baseline'][key]['items']}
        fused = {item['id']: item for item in runs['fused'][key]['items']}
        if baseline.keys() != fused.keys():
            raise ValueError('prompt sets differ')
        for prompt in baseline:
            pairs.append(dict(run=key, prompt=prompt,
                              exact_text_match=baseline[prompt]['output_text'] == fused[prompt]['output_text']))
    result['paired_text_matches'] = sum(pair['exact_text_match'] for pair in pairs)
    result['paired_requests'] = len(pairs)
    result['pairs'] = pairs
    result['makespan_speedup'] = result['baseline']['median_makespan_s'] / result['fused']['median_makespan_s']
    result['caveat'] = 'Short smoke run, not a quality benchmark. Scheduling and baseline variability can affect output comparisons.'
    (args.directory / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'pairs'}, indent=2))


if __name__ == '__main__':
    main()
