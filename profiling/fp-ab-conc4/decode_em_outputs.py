"""Retain decoded completions beside the unchanged benchmark scores."""
import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('arm_directory', type=Path)
    args = parser.parse_args()
    root = args.arm_directory
    summary = json.loads((root / 'summary.json').read_text())
    tokenizer = None
    rows = []
    for wave in summary['waves']:
        case = json.loads((root / 'cases' / f"wave{wave['index']:03d}.json").read_text())
        result = json.loads((root / 'infinilm' / f"{case['case_id']}.json").read_text())
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(result['effective_config']['tokenizer_path'], trust_remote_code=True)
        for score, request in zip(wave['quality']['samples'], result['requests'], strict=True):
            rows.append(dict(score, wave=wave['index'], request_id=request['request_id'],
                             output_tokens=request['output_tokens_actual'],
                             output_budget_reached=request['output_tokens_actual'] == case['output'],
                             text=tokenizer.decode(request['output_token_ids'], skip_special_tokens=True)))
    if len(rows) != summary['scored_samples']:
        raise ValueError('missing decoded outputs')
    (root / 'decoded-predictions.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
    print(json.dumps([row for row in rows if row['prediction'] is None], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
