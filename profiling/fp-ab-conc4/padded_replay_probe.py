"""Fixed-schedule teacher-forced replay: growing KV, padding and mixed rows."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import ctypes
import json
from pathlib import Path
import sys
import time

import infinicore as ic
import torch


def cpu_tensor(values, dtype):
    return ic.from_torch(torch.tensor(values, dtype=dtype))


def read_logits(tensor):
    if tensor.dtype != ic.bfloat16:
        raise ValueError(f'unexpected logit dtype {tensor.dtype}')
    host = tensor.contiguous().to('cpu')
    out = torch.empty(tuple(tensor.shape), dtype=torch.bfloat16)
    ctypes.memmove(out.data_ptr(), host.data_ptr(), out.numel() * out.element_size())
    return out


def run(args):
    sys.path.insert(0, '/bench')
    from common.config import load_config, dataset_path
    from infinilm_benchmark import create_engine
    from workloads.longbench_task_benchmark import encode_row
    from transformers import AutoTokenizer

    cfg = load_config('/bench/configs/minicpm5_2b_layerb_qy.json')
    cfg['backend_overrides']['infinilm']['enable_graph'] = args.mode == 'graph'
    model = cfg['models'][0]
    tok = AutoTokenizer.from_pretrained(model['model_path'], trust_remote_code=True)
    rows = [json.loads(line) for line in dataset_path(cfg).read_text().splitlines() if line.strip()][:4]
    inputs = [encode_row(cfg, row, tok, 8160)['token_ids'] for row in rows]
    case = dict(model_path=model['model_path'], precision='bf16', tp=1, batch_size=4,
                prompt=8160, output=32, ignore_eos=False)
    holder, settings = create_engine(case, cfg)
    engine = holder.engine.model_runner.model_engine
    plan = [('first', [(0, 512, False)]), ('mid', [(0, 512, False)]),
            ('final_padded', [(0, 13, True)]), ('decode', [(0, 1, True)]),
            ('new_short', [(1, 7, True)]), ('decode_after_padding', [(0, 1, True)]),
            ('mixed_start', [(0, 1, True), (2, 127, False)]),
            ('mixed_mid', [(0, 1, True), (2, 385, False)]),
            ('mixed_final', [(0, 1, True), (2, 17, True)])]
    plan += [(f'long_mid_{i}', [(3, 512, False)]) for i in range(15)]
    plan += [('long_final', [(3, 33, True)]),
             ('packed_exact', [(i, 128, True) for i in range(4)]),
             ('packed_ragged', [(0, 1, True), (1, 3, True), (2, 63, False), (3, 257, True)])]
    plan += [(f'decode_b{n}', [(i, 1, True) for i in range(n)]) for n in (2, 3, 4)]
    plan += [('one_pad_token', [(1, 511, False)]), ('short_after_511', [(1, 65, True)]),
             ('decode_reordered', [(i, 1, True) for i in (3, 1, 0, 2)]),
             ('shrink_to_long', [(3, 2, True)])]
    past = [0] * 4
    results, logits = [], {}
    print('PROBE_READY', args.mode, flush=True)
    try:
        for name, requests in plan:
            ids, positions, slots, totals, offsets, cu, tables, flags = [], [], [], [], [0], [0], [], []
            lengths = []
            width = max((past[r] + n + 255) // 256 for r, n, _ in requests)
            for rid, n, final in requests:
                p, total = past[rid], past[rid] + n
                lengths.append(p)
                ids.extend(inputs[rid][p:total])
                positions.extend(range(p, total))
                slots.extend(((pos // 256) * 4 + rid) * 256 + pos % 256 for pos in range(p, total))
                totals.append(total)
                offsets.append(offsets[-1] + n)
                cu.append(cu[-1] + total)
                tables.append([block * 4 + rid for block in range(width)])
                flags.append(final)
                past[rid] = total
            if len(ids) != offsets[-1]:
                raise ValueError('fixture exceeds source tokens')
            print('PROBE_STEP_BEGIN', name, flush=True)
            started = time.perf_counter()
            output = engine.forward_raw(
                cpu_tensor([ids], torch.int64),
                position_ids=cpu_tensor(positions, torch.int64),
                past_kv_lengths=cpu_tensor(lengths, torch.int32),
                total_kv_lengths=cpu_tensor(totals, torch.int32),
                input_offsets=cpu_tensor(offsets, torch.int32),
                cu_seqlens=cpu_tensor(cu, torch.int32),
                block_tables=cpu_tensor(tables, torch.int32),
                slot_mapping=cpu_tensor(slots, torch.int64),
                is_final_prefill_chunk=flags, sample_all_positions=False,
                temperature=0, top_k=1, top_p=1)
            sampled = output['output_ids'].to_numpy().tolist()
            for index, final in enumerate(flags):
                if not final and sampled[index] != -1:
                    raise AssertionError('continuing chunk was sampled')
            logits[name] = read_logits(output['logits'])
            results.append(dict(name=name, tokens=len(ids), past=lengths, total=totals,
                                requests=len(requests), flags=flags, sampled=sampled,
                                elapsed_s=time.perf_counter()-started))
            print('PROBE_STEP_END', json.dumps(results[-1]), flush=True)
        args.output.mkdir(parents=True, exist_ok=True)
        torch.save(logits, args.output / f'{args.mode}-logits.pt')
        (args.output / f'{args.mode}.json').write_text(json.dumps(dict(mode=args.mode, settings=settings, steps=results), indent=2) + '\n')
    finally:
        holder.engine.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode', choices=['eager', 'graph'], required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    with ThreadPoolExecutor(max_workers=1) as worker:
        try:
            worker.submit(run, args).result()
        finally:
            worker.submit(ic.sync_device).result()
            worker.submit(ic.set_device, 'cpu').result()


if __name__ == '__main__':
    main()
