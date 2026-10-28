#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def key(row):
    return tuple(row[field] for field in ('operation', 'dtype', 'tokens', 'padded', 'interleaved'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('directory', type=Path)
    args = ap.parse_args()
    rows = {}
    for variant in ('installed', 'candidate_off', 'candidate_on'):
        data = json.loads((args.directory / f'{variant}.json').read_text())['rows']
        if len(data) != 54 or len({key(row) for row in data}) != 54:
            raise ValueError(f'incomplete case matrix: {variant}')
        if any(row['correctness'] != 'passed' for row in data):
            raise ValueError(f'failed case: {variant}')
        rows[variant] = {key(row): row['hashes'] for row in data}
    mismatches = {variant: [list(k) for k, v in rows['installed'].items() if rows[variant].get(k) != v]
                  for variant in ('candidate_off', 'candidate_on')}
    result = dict(cases_per_variant=54, bitwise_mismatches=mismatches,
                  passed=not any(mismatches.values()))
    (args.directory / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['passed'] else 1)


if __name__ == '__main__':
    main()
