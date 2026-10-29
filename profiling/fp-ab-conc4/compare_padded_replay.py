"""Verify fixed-schedule sampled IDs and device-only replay; report logit drift."""
import argparse
import json
from pathlib import Path
import re
import torch


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('directory', type=Path)
    args = ap.parse_args()
    root = args.directory
    eager = json.loads((root / 'eager.json').read_text())['steps']
    graph = json.loads((root / 'graph.json').read_text())['steps']
    left = torch.load(root / 'eager-logits.pt', map_location='cpu', weights_only=True)
    right = torch.load(root / 'graph-logits.pt', map_location='cpu', weights_only=True)
    log = (root / 'graph.log').read_text().split('PROBE_READY graph', 1)[1]
    if len(eager) != 34 or len(graph) != 34:
        raise ValueError('incomplete fixed-schedule fixture')
    rows = []
    live_tokens = 0
    for ref, got in zip(eager, graph):
        for field in ('name', 'tokens', 'past', 'total', 'requests', 'flags'):
            if ref[field] != got[field]:
                raise ValueError(f'unmatched schedule field {field}')
        name = ref['name']
        if ref['sampled'] != got['sampled']:
            raise AssertionError(f'sampled token mismatch: {name}')
        section = log.split(f'PROBE_STEP_BEGIN {name}\n', 1)[1].split('PROBE_STEP_END', 1)[0]
        profiles = re.findall(r'hostop_ms=([0-9.]+) op_list_ms=([0-9.]+) used_device=(\d)', section)
        if not profiles or any(float(h) != 0 or float(e) != 0 or d != '1' for h, e, d in profiles):
            raise AssertionError(f'non-device replay: {name}')
        if ref['tokens'] != ref['requests'] and 'full_prefill_replay:' not in section:
            raise AssertionError(f'prefill missed device replay: {name}')
        a, b = left[name].float(), right[name].float()
        if a.shape != b.shape or not torch.isfinite(b).all():
            raise AssertionError(f'invalid logits: {name}')
        diff = a - b
        live_tokens += sum(ref['flags'])
        rows.append(dict(name=name, sampled_match=True, device_only=True,
                         max_abs_logit_diff=diff.abs().max().item(),
                         relative_rms_logit_diff=(diff.square().mean().sqrt()/a.square().mean().sqrt().clamp_min(1e-12)).item()))
    result = dict(passed=True, steps=len(rows), matching_live_tokens=live_tokens, rows=rows,
                  note='Logit differences are reported, not asserted bitwise equal. All consumed greedy tokens must match.')
    (root / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, indent=2))


if __name__ == '__main__':
    main()
