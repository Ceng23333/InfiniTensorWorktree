#!/usr/bin/env python3
"""Owned-memory RoPE/KV reference, padding, replay and timing probes."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import ctypes
import hashlib
import json
from pathlib import Path
import time

import infinicore as ic
import torch


def owned(tensor):
    return ic.from_torch(tensor).to('qy:0')


def read(tensor, dtype):
    host = tensor.contiguous().to('cpu')
    out = torch.empty(tuple(tensor.shape), dtype=dtype)
    ctypes.memmove(out.data_ptr(), host.data_ptr(), out.numel() * out.element_size())
    return out


def digest(tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def rope_case(tokens, dtype, padded, interleaved):
    positions = ((torch.arange(tokens) * 13) % 1024).long()
    angles = torch.randn(1024, 64)
    sine, cosine = angles.sin().to(dtype), angles.cos().to(dtype)
    ipos, isin, icos = map(owned, (positions, sine, cosine))
    parts = []
    for heads in (16, 2):
        stride = 256 if padded else 128
        width = 8704 if padded else heads * 128
        source = torch.randn(tokens, width, dtype=dtype)
        expected = source.clone()
        view = expected.as_strided((tokens, heads, 128), (width, stride, 1))
        original = view.clone()
        first = original[..., ::2] if interleaved else original[..., :64]
        second = original[..., 1::2] if interleaved else original[..., 64:]
        s, c = sine[positions, None, :], cosine[positions, None, :]
        v0, v1 = first * c - second * s, first * s + second * c
        if interleaved:
            view[..., ::2], view[..., 1::2] = v0, v1
        else:
            view[..., :64], view[..., 64:] = v0, v1
        store = owned(source)
        tensor = store.as_strided((tokens, heads, 128), (width, stride, 1))
        touched = torch.zeros(source.shape, dtype=torch.bool)
        touched.as_strided((tokens, heads, 128), (width, stride, 1)).fill_(True)
        parts.append((source, expected, store, tensor, touched))
    algo = ic.ops.RoPEAlgo.GPT_J if interleaved else ic.ops.RoPEAlgo.GPT_NEOX

    def reset():
        for source, _, store, _, _ in parts:
            store.copy_(owned(source))
        ic.sync_device()

    def run():
        for _, _, _, tensor, _ in parts:
            ic.ops.rope(tensor, ipos, isin, icos, algo, out=tensor)

    def check():
        result = []
        for source, expected, store, _, touched in parts:
            actual = read(store, dtype)
            tolerance = {torch.bfloat16: .035, torch.float16: .004, torch.float32: 2e-5}[dtype]
            torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
            untouched = ~touched
            torch.testing.assert_close(actual[untouched], source[untouched], rtol=0, atol=0)
            result.append(digest(actual))
        return result
    return reset, run, check


def kv_case(tokens, dtype, padded, skipped_slots=True):
    heads, dim, block_size = 2, 128, 256
    width = 136 if padded else dim
    key, value = [torch.randn(tokens, heads, width, dtype=dtype) for _ in range(2)]
    slots = torch.arange(tokens - 1, -1, -1).long()
    if skipped_slots:
        slots[::7] = -1
    blank = torch.full((4, block_size, heads, width), -7, dtype=dtype)
    expected = [blank.clone(), blank.clone()]
    for cache, src in zip(expected, (key, value)):
        valid = slots >= 0
        cache[slots[valid] // block_size, slots[valid] % block_size, :, :dim] = src[valid, :, :dim]
    ik, iv = (owned(x).narrow(2, 0, dim) for x in (key, value))
    stores = [owned(blank), owned(blank)]
    ck, cv = (x.narrow(3, 0, dim) for x in stores)
    islots = owned(slots)

    def reset():
        for store in stores:
            store.copy_(owned(blank))
        ic.sync_device()

    def run():
        ic.ops.paged_caching(ck, cv, ik, iv, islots)

    def check():
        results = []
        for store, ref in zip(stores, expected):
            actual = read(store, dtype)
            torch.testing.assert_close(actual, ref, rtol=0, atol=0)
            results.append(digest(actual))
        return results
    return reset, run, check


def probe(args):
    ic.set_device('qy:0')
    rows = []
    dtypes = (torch.bfloat16, torch.float16, torch.float32) if args.check_all else (torch.bfloat16,)
    for dtype in dtypes:
        for tokens in (1, 4, 512):
            for padded in ((False, True) if args.check_all else (True,)):
                for operation in ('rope_qk', 'kv_cache'):
                    for interleaved in ((False, True) if args.check_all and operation == 'rope_qk' else (False,)):
                        torch.manual_seed(20261028)
                        factory = rope_case if operation == 'rope_qk' else kv_case
                        inputs = (tokens, dtype, padded, interleaved) if operation == 'rope_qk' else (tokens, dtype, padded, args.check_all)
                        reset, run, check = factory(*inputs)
                        reset()
                        run()
                        ic.sync_device()
                        eager_hashes = check()
                        graph = None
                        if args.check_all:
                            reset()
                            ic.start_graph_recording()
                            try:
                                run()
                                graph = ic.stop_graph_recording()
                            except BaseException:
                                ic.cancel_graph_recording()
                                raise
                        samples = []
                        for _ in range(args.iterations):
                            reset()
                            start = time.time_ns()
                            graph.run() if graph is not None else run()
                            ic.sync_device()
                            samples.append(dict(start_ns=start, end_ns=time.time_ns()))
                            if args.check_all:
                                if check() != eager_hashes:
                                    raise AssertionError('graph replay differs from eager')
                        rows.append(dict(engine='infinilm', tokens=tokens, dtype=str(dtype), padded=padded,
                                         interleaved=interleaved, operation=operation, correctness='passed',
                                         skipped_slots=args.check_all if operation == 'kv_cache' else False,
                                         hashes=eager_hashes, samples=samples))
                        print(f'{operation} {dtype} tokens={tokens} padded={padded} interleaved={interleaved}: passed', flush=True)
    args.output.write_text(json.dumps(dict(rows=rows), indent=2) + '\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--engine', choices=['infinilm'], default='infinilm')
    ap.add_argument('--norm-mode', default='separate')
    ap.add_argument('--iterations', type=int, default=3)
    ap.add_argument('--check-all', action='store_true')
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if args.iterations < 1:
        ap.error('iterations must be positive')
    with ThreadPoolExecutor(max_workers=1) as worker:
        try:
            worker.submit(probe, args).result()
        finally:
            worker.submit(ic.cancel_graph_recording).result()
            worker.submit(ic.sync_device).result()
            worker.submit(ic.set_device, 'cpu').result()


if __name__ == '__main__':
    main()
