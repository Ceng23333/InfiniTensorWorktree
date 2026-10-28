#!/usr/bin/env python3
"""Bounded BF16 Add/RMSNorm/activation probe using installed production operators."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import time

import torch


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--engine', choices=['infinilm', 'vllm'], required=True)
    ap.add_argument('--tokens', type=int, nargs='+', default=[1, 4, 512])
    ap.add_argument('--iterations', type=int, default=5)
    ap.add_argument('--norm-mode', choices=['separate', 'fused'], default='separate')
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if args.iterations < 1 or any(n < 1 for n in args.tokens):
        ap.error('iterations and tokens must be positive')
    torch.manual_seed(20261028)
    if args.engine == 'infinilm':
        import infinicore as ic
        ic.set_device(ic.device('qy', 0))

        def wrap(t):
            return ic.from_blob(t.data_ptr(), list(t.shape), dtype=ic.bfloat16,
                                device=ic.device('qy', 0))

        def sync():
            ic.sync_device()
            torch.cuda.synchronize()
    else:
        from vllm import _custom_ops  # noqa: F401 - registers the production operators
        sync = torch.cuda.synchronize

    rows = []
    for n in args.tokens:
        x = torch.randn(n, 2048, device='cuda', dtype=torch.bfloat16)
        residual = torch.randn_like(x)
        weight = torch.ones(2048, device='cuda', dtype=torch.bfloat16)
        summed = torch.empty_like(x)
        normed = torch.empty_like(x)
        gate = torch.randn(n, 12288, device='cuda', dtype=torch.bfloat16)
        activated = torch.empty(n, 6144, device='cuda', dtype=torch.bfloat16)
        xf, rf = x.float(), residual.float()
        add_ref = (xf + rf).bfloat16()
        norm_input = add_ref.float() if args.engine == 'infinilm' else xf + rf
        norm_ref = (norm_input * torch.rsqrt(norm_input.square().mean(-1, keepdim=True) + 1e-6)).bfloat16()
        act_ref = (torch.nn.functional.silu(gate[:, :6144].float()) * gate[:, 6144:].float()).bfloat16()
        sync()
        if args.engine == 'infinilm':
            ix, ir, iw, isum, ino, ig, ia = map(wrap, (x, residual, weight, summed, normed, gate, activated))

            def add_norm():
                if args.norm_mode == 'fused':
                    ic.ops.add_rms_norm(ino, isum, iw, 1e-6, out=ino, residual_out=isum)
                else:
                    ic.ops.add(ix, ir, out=isum)
                    ic.ops.rms_norm(isum, iw, 1e-6, out=ino)

            def activation():
                ic.ops.silu_and_mul(ig, out=ia)

            operations = [('add_rmsnorm', add_norm, normed, norm_ref),
                          ('silu', activation, activated, act_ref)]
        else:
            # The fused operator mutates both inputs. Reset outside the timed interval.
            def add_norm():
                torch.ops._C.fused_add_rms_norm(normed, summed, weight, 1e-6)

            def activation():
                torch.ops._C.silu_and_mul(activated, gate)

            operations = [('add_rmsnorm', add_norm, normed, norm_ref),
                          ('silu', activation, activated, act_ref)]
        for name, op, out, ref in operations:
            def reset():
                if name == 'add_rmsnorm' and (args.engine == 'vllm' or args.norm_mode == 'fused'):
                    normed.copy_(x)
                    summed.copy_(residual)
                sync()

            reset()
            op()
            sync()
            torch.testing.assert_close(out, ref, rtol=.03, atol=.03)
            if name == 'add_rmsnorm':
                torch.testing.assert_close(summed, add_ref, rtol=0, atol=0)
            samples = []
            for _ in range(args.iterations):
                reset()
                start = time.time_ns()
                op()
                sync()
                end = time.time_ns()
                samples.append(dict(start_ns=start, end_ns=end))
            row = dict(engine=args.engine, tokens=n, operation=name,
                       norm_mode='fused' if args.engine == 'vllm' else args.norm_mode, correctness='passed',
                       samples=samples)
            rows.append(row)
            print(json.dumps(row), flush=True)
    args.output.write_text(json.dumps(dict(rows=rows,
        caveat='Synchronized host intervals include dispatch and profiler overhead; use device activities for kernel timing.'), indent=2) + '\n')


def release_device():
    ic = sys.modules.get('infinicore')
    if ic is not None:
        ic.sync_device()
        ic.set_device('cpu')


if __name__ == '__main__':
    # Match RankWorker: finish device/TLS cleanup while the vendor runtime is alive.
    with ThreadPoolExecutor(max_workers=1) as worker:
        try:
            worker.submit(main).result()
        finally:
            worker.submit(release_device).result()
