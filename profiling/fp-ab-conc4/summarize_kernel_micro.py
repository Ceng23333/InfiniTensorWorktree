#!/usr/bin/env python3
"""Associate device activity with synchronized microbenchmark sample intervals."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics


def summarize(trace, checks):
    events = [event for event in trace['activities'] if event.get('kind') == 56]
    rows = []
    for case in checks['rows']:
        samples = []
        kernels = defaultdict(list)
        for sample in case['samples']:
            matched = [event for event in events if sample['start_ns'] <= event['start'] < sample['end_ns']]
            samples.append(sum(event['elapsed'] for event in matched) / 1000)
            for event in matched:
                key = (event['kernel_name'], tuple(event['grid_size']), tuple(event['block_size']))
                kernels[key].append(event['elapsed'] / 1000)
        if not samples or not all(samples):
            raise ValueError(f'missing device activity for {case}')
        rows.append(dict(engine=case['engine'], tokens=case['tokens'], operation=case['operation'],
                         norm_mode='fused' if case['engine'] == 'vllm' else case.get('norm_mode', 'separate'),
                         device_us_mean=statistics.mean(samples), device_us_median=statistics.median(samples),
                         kernels=[dict(name=k[0], grid=k[1], block=k[2], count=len(v), mean_us=statistics.mean(v))
                                  for k, v in kernels.items()]))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('directory', type=Path)
    args = ap.parse_args()
    rows = []
    for path in sorted(args.directory.glob('*-checks.json')):
        engine = path.name.removesuffix('-checks.json')
        rows.extend(summarize(json.loads((args.directory / f'{engine}.json').read_text()), json.loads(path.read_text())))
    (args.directory / 'device-timings.json').write_text(json.dumps(rows, indent=2) + '\n')
    for row in rows:
        print(f"{row['engine']:8} {row['norm_mode']:8} n={row['tokens']:3} {row['operation']:12} "
              f"mean={row['device_us_mean']:.3f} us median={row['device_us_median']:.3f} us")


if __name__ == '__main__':
    main()
