# Chunked prefill and padded graph replay repair

Date: 2026-10-28, Asia/Shanghai.
Worktree branch: `fix/qy-chunked-padded-replay-20261028` (umbrella and InfiniLM).
Published tag `v2026.10.28-qy-kernel-em-pending` is unchanged.

Follow-up: the requested full 59-sample run completed at 22:14:37 on October 28.
It scored 14/59, below the historical graph result's 16/59, despite clean
execution and passing targeted replay tests. See `FULL_59_REPLAY_BENCH.md`.
The repair is not full-dataset EM-qualified.

## Implementation

- FULL attention capture uses configured capacities rather than representative
  warmup row lengths: query 512 and KV 8448 in the validated configuration.
- Initial, continuing, final, and mixed chunks replay the FULL graph. Missing
  captures and capacity overflow throw; they do not select eager fallback.
- Query and KV cumulative tails stay independent. Inactive cache slots are -1,
  token/position padding is cleared, and only compact live logits are returned.
- Chunk-final flags pass from ModelRunner through the Python engine and native
  input bridge. Continuing chunks return -1 IDs; mixed batches sample only
  final/decode rows. Default empty flags retain existing all-final behavior.
- FULL capture rejects graphs containing host segments. QY MiniCPM5-2B B1-B4
  decode uses the existing persistent provider with its fused-QKV layout checks.
- Redundant mid-chunk FULL captures are removed. Optional fused norm, RoPE, KV,
  and draft SiLU promotion is not part of this repair.

## Validation

Candidate artifacts: `results/padded_replay_20261028_175543/`.
All 34 fixed-schedule steps passed; all 31 consumed greedy tokens matched the
eager reference. The reference is a targeted oracle, not a runtime fallback.
Coverage includes exact/padded chunks, mixed mid/final/decode rows, context
growth to 8102, B1-B4 decode, reordered requests, a 511-token chunk, and batch
shrinkage after longer batches. Every runtime graph profile reported
`used_device=1`, `hostop_ms=0`, and `op_list_ms=0`. Startup warmup is excluded.

Logits are not bitwise equal: maximum absolute difference 0.515625 and maximum
relative RMS difference 0.03445164. Sampled-token agreement is not a general
numerical-equivalence or full-dataset quality guarantee.

Fifteen CPU tests passed: ten matching `test_*chunk*.py` and five FULL replay
guard tests. Shell syntax and InfiniLM diff-whitespace checks passed.

The real four-sample, up-to-512-output-token graph-only check in
`results/em_gate_20261028_172811/baseline/` matched the retained 1/4 EM reference
(D,D,A,A), with 2168 device-only runtime graph executions. The optional
all-kernel arm scored 0/4 and remains unqualified. See `EM_FINDINGS.md`.
The fallback-based full EM run was stopped and was not restarted.

Validated scope: BF16 MiniCPM5-2B, QY KS38 GPU 1, TP1, batch <=4, token bucket
512, configured context capacity 8448 (runtime exercised through 8102).
Other models, devices, tensor/pipeline parallel configurations and capacities
are not qualified by these tests. Probe timing includes diagnostics, cold
operations, and logit copies; it is not an end-to-end performance benchmark.

## Installation

After the candidate passed and inference containers exited, the matching
engine, core Python binding and runtime were installed into the worktree's
Python library directories. The runtime was also installed in the shared
`InfiniCore-v20261009-post-refactor-prhead-install/lib` directory. Files were
staged first and individually renamed into place. Existing binaries and
before/after SHA256 manifests are retained in
`results/replay_install_20261028_175840/`.

The installed provider was not replaced; its SHA256 remains
`09b541a50de2345b7c6ef3627e28d42a29cc620679dc5cf4dc96842ce72dac8b`.
Engine SHA256: `2eab7a446b818ef8feb3686b83699cf381910391ca432ffaeaa5b72bd0f42f72`.
Core binding SHA256: `ed4d1627558db44203a25ae6db1fb53fd38343392d3dd86c45d6dd808240861d`.
Runtime SHA256: `a5143b260e84857fe4e88222d5d54bc4ba6a3a420df6189bff077de2a1d7a86e`.

An installed-library graph check, without candidate library bind mounts, is
recorded in `results/padded_replay_installed_20261028_175840/`. It also passed
all 34 steps and all 31 live tokens, with device-only runtime graphs. Its
comparison reuses `eager.json` and `eager-logits.pt` from the candidate run
above, rather than rerunning the reference. Source changes remain uncommitted.

## Reproduction

From the umbrella worktree:

```bash
bash profiling/fp-ab-conc4/run_padded_replay_probe.sh
USE_INSTALLED=1 bash profiling/fp-ab-conc4/run_padded_replay_probe.sh
```

These run a bounded reference/graph comparison. `MODES=graph` runs only replay
and needs retained reference artifacts for a subsequent comparison. Do not
enable optional kernel flags when reproducing the graph-only quality result.
