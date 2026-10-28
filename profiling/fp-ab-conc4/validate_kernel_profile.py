#!/usr/bin/env python3
"""Reject profiler false-success exits and empty hardware-counter exports."""
import argparse
import json
from pathlib import Path
import re
import sqlite3


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('directory', type=Path)
    ap.add_argument('--mode', choices=['capture', 'kprof'], required=True)
    args = ap.parse_args()
    status = {}
    for path in args.directory.glob('*-checks.json'):
        engine = path.name.removesuffix('-checks.json')
        log_path = args.directory / f'{engine}-profiler.log'
        log = (log_path if log_path.exists() else args.directory / 'runner.log').read_text()
        failures = [text for text in ('Capture failed', 'corrupted', 'Traceback') if text in log]
        checks = json.loads(path.read_text())['rows']
        if len(checks) != 6 or any(row['correctness'] != 'passed' for row in checks):
            failures.append('missing/failed operator checks')
        metrics = None
        if args.mode == 'kprof':
            passes = [(int(a), int(b)) for a, b in re.findall(r'Pass (\d+) of (\d+)', log)]
            if not passes or len({b for _, b in passes}) != 1 or {a for a, _ in passes} != set(range(1, passes[0][1] + 1)):
                failures.append('incomplete profiler passes')
            db = args.directory / f'{engine}.metric'
            with sqlite3.connect(f'file:{db}?mode=ro', uri=True) as conn:
                metrics = conn.execute('select count(*) from kprof_metric').fetchone()[0]
            if not metrics:
                failures.append('empty counter export')
        status[engine] = dict(valid=not failures, failures=failures, metric_rows=metrics)
    if not status:
        status['run'] = dict(valid=False, failures=['no checks artifacts'])
    (args.directory / 'validation.json').write_text(json.dumps(status, indent=2) + '\n')
    print(json.dumps(status, indent=2))
    raise SystemExit(0 if all(row['valid'] for row in status.values()) else 1)


if __name__ == '__main__':
    main()
