"""Control for unrelated GEMM changes after rebuilding the provider."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import ctypes
import hashlib
import json
from pathlib import Path
import time
import infinicore as ic
import torch


def run(args):
    ic.set_device('qy:0')
    rows = []
    torch.manual_seed(42)
    for batch, n in ((1, 1), (1, 4), (1, 512), (2, 4)):
        leading = () if batch == 1 else (batch,)
        ha = torch.randn(*leading, n, 2048, dtype=torch.bfloat16)
        hb = torch.randn(*leading, 2048, 2048, dtype=torch.bfloat16)
        a, b = (ic.from_torch(x).to('qy:0') for x in (ha, hb))
        out = ic.empty((*leading, n, 2048), dtype=ic.bfloat16, device='qy:0')
        for _ in range(2):
            ic.ops.matmul(a, b, out=out)
            ic.sync_device()
        samples = []
        for _ in range(5):
            start = time.time_ns()
            ic.ops.matmul(a, b, out=out)
            ic.sync_device()
            samples.append(dict(start_ns=start, end_ns=time.time_ns()))
        host = out.to('cpu')
        result = torch.empty((*leading, n, 2048), dtype=torch.bfloat16)
        ctypes.memmove(result.data_ptr(), host.data_ptr(), result.numel() * result.element_size())
        torch.testing.assert_close(result.float(), ha.float() @ hb.float(), rtol=.03, atol=.5)
        fingerprint = hashlib.sha256(result.view(torch.uint8).numpy().tobytes()).hexdigest()
        rows.append(dict(engine='infinilm', operation='gemm', tokens=n, batch=batch,
                         correctness='passed', sha256=fingerprint, samples=samples))
        print(batch, n, [(s['end_ns']-s['start_ns'])/1000 for s in samples], flush=True)
    args.output.write_text(json.dumps(dict(rows=rows), indent=2) + '\n')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    with ThreadPoolExecutor(max_workers=1) as worker:
        try:
            worker.submit(run, args).result()
        finally:
            worker.submit(ic.sync_device).result()
            worker.submit(ic.set_device, 'cpu').result()
