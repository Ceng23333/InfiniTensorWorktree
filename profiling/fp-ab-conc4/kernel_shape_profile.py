#!/usr/bin/env python3
"""Stream native DLPTI exports into measured-window, launch-shape statistics."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics


def summarize(values):
    values = sorted(values)
    return dict(count=len(values), sum_ms=sum(values) / 1e6,
                mean_us=statistics.mean(values) / 1000,
                p50_us=statistics.median(values) / 1000,
                p95_us=values[min(len(values)-1, int(len(values)*.95))] / 1000,
                max_us=values[-1] / 1000)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--trace', type=Path, required=True)
    ap.add_argument('--runs', type=Path, nargs='+', required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    windows = []
    for path in args.runs:
        wall = json.loads(path.read_text())['measured_wall']
        windows.append((int(wall['start_epoch'] * 1e9), int(wall['end_epoch'] * 1e9)))
    groups = defaultdict(list)
    totals = defaultdict(lambda: defaultdict(float))
    inside = False
    with args.trace.open() as stream:
        for line in stream:
            if not inside:
                inside = '"activities"' in line and '[' in line
                continue
            text = line.strip().rstrip(',')
            if text.startswith(']'):
                break
            # The native exporter writes one object per line. Skip API traffic.
            if not text.startswith(('{"kind":56,', '{"kind":58,', '{"kind":50,')):
                continue
            obj = json.loads(text)
            start = obj.get('start', 0)
            phase = 'measured' if any(a <= start < b for a, b in windows) else 'outside_measured'
            duration = obj.get('elapsed', 0)
            totals[phase][str(obj['kind'])] += duration / 1e6
            if obj['kind'] != 56:
                continue
            key = (phase, obj['kernel_name'], tuple(obj.get('grid_size', [])),
                   tuple(obj.get('block_size', [])), obj.get('cu_mask'))
            groups[key].append(duration)
    rows = [dict(phase=k[0], kernel=k[1], grid=k[2], block=k[3], cu_mask=k[4],
                 **summarize(v)) for k, v in groups.items()]
    rows.sort(key=lambda row: row['sum_ms'], reverse=True)
    result = dict(trace=str(args.trace), measured_windows_ns=windows,
                  kind_sum_ms=totals, kernels=rows,
                  caveat='Summed device activity is not critical-path wall time; grid is not a token-shape label.')
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    for phase in ('measured', 'outside_measured'):
        print(phase, dict(totals[phase]))
        for row in [r for r in rows if r['phase'] == phase][:15]:
            print(f"{row['sum_ms']:10.2f} ms n={row['count']:7} mean={row['mean_us']:8.2f} "
                  f"p50={row['p50_us']:8.2f} grid={row['grid']} block={row['block']} {row['kernel'][:100]}")


if __name__ == '__main__':
    main()
