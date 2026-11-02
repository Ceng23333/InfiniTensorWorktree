"""Reject incomplete/mismatched EM runs and report paired answer changes."""
import argparse
import json
from pathlib import Path


def samples(summary, expected):
    if summary.get('diagnostic_output_tokens', 0):
        raise ValueError('short-output diagnostics are not an EM gate')
    if not summary['complete'] or summary['expected_samples'] != expected:
        raise ValueError('incomplete EM run or wrong sample count')
    if len(summary['waves']) != summary['expected_waves']:
        raise ValueError('missing waves')
    result = {}
    for wave in summary['waves']:
        if not wave['valid'] or wave['exit_code'] != 0:
            raise ValueError('invalid producer wave')
        rows = wave['quality']['samples']
        if len(rows) != len(wave['input_hashes']):
            raise ValueError('missing input hashes')
        for row, fingerprint in zip(rows, wave['input_hashes']):
            if row['sample_id'] in result:
                raise ValueError('duplicate sample')
            if row['exact_match'] != (row['prediction'] == row['answer']):
                raise ValueError('inconsistent EM score')
            result[row['sample_id']] = dict(row, input_hash=fingerprint)
    if len(result) != expected:
        raise ValueError('missing samples')
    return result


def compare(baseline, candidate, expected=59, floor=16):
    for field in ('dataset_sha256', 'scorer_sha256', 'config'):
        if baseline[field] != candidate[field]:
            raise ValueError(f'unmatched {field}')
    left, right = samples(baseline, expected), samples(candidate, expected)
    if left.keys() != right.keys():
        raise ValueError('unmatched sample IDs')
    regressions, improvements, changed = [], [], []
    for key in left:
        a, b = left[key], right[key]
        if a['input_hash'] != b['input_hash'] or a['answer'] != b['answer']:
            raise ValueError(f'unmatched input/answer: {key}')
        if a['prediction'] != b['prediction']:
            changed.append(key)
        if a['exact_match'] and not b['exact_match']:
            regressions.append(key)
        if b['exact_match'] and not a['exact_match']:
            improvements.append(key)
    baseline_em = sum(row['exact_match'] for row in left.values())
    candidate_em = sum(row['exact_match'] for row in right.values())
    return dict(samples=expected, baseline_matched=baseline_em, candidate_matched=candidate_em,
                prediction_matches=expected-len(changed), changed_predictions=changed,
                lost_correct=regressions, gained_correct=improvements, quality_floor=floor,
                passed=baseline_em >= floor and candidate_em >= baseline_em,
                caveat='Finite-set aggregate EM gate, not numerical equivalence or a statistical non-inferiority claim.')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('directory', type=Path)
    ap.add_argument('--expected', type=int, default=59)
    ap.add_argument('--floor', type=int, default=16)
    args = ap.parse_args()
    baseline = json.loads((args.directory / 'baseline/summary.json').read_text())
    candidate = json.loads((args.directory / 'candidate/summary.json').read_text())
    result = compare(baseline, candidate, args.expected, args.floor)
    (args.directory / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['passed'] else 1)


if __name__ == '__main__':
    main()
