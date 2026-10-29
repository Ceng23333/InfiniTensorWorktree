# EM correctness gate

Date: 2026-10-28, Asia/Shanghai. Status: graph replay repaired, but the full
59-sample rerun is below historical EM; optional kernel promotion held.

The completed repaired-graph run `results/em_gate_20261028_201439/` scored
**14/59 (23.73%)**, versus the retained October 26 graph result's 16/59.
All 15 waves exited cleanly. Measured output throughput was 8.585 tok/s
(historical graph: 7.877 tok/s); mean TTFT was 66.003 s and mean request
latency 164.632 s. This is not an EM qualification pass. See
`FULL_59_REPLAY_BENCH.md` for configuration, answer changes, timing definitions,
and the limitations of historical comparison and the unchanged legacy scorer.

The initial fresh 59-sample run was stopped after its first completed wave:
the installed-kernel baseline with FULL prefill scored 0/4 and generated
gibberish for all four requests. It did not complete the 59-sample gate, and the
optimized candidate was not run. Evidence is retained in
`results/em_gate_20261028_135303/baseline/`.

The original FULL capture used eight representative requests in a 512-token
bucket, giving captured attention max query/KV lengths of 64. Runtime spans
such as 512/512 or 512/7680 cannot safely replay that fixed launch plan merely
by updating device length tensors. Inspection also found that padded KV
cumulative lengths reused the last query offset, padded slot mappings could
write slot zero, and padded output rows could overwrite live logits.

A historical diagnostic engine added a conservative captured-length guard (eager fallback
outside the capture bounds), independent query/KV tail padding, invalid FULL
padding slots, and output copies restricted to live requests. The first short
long-context control is `results/em_gate_20261028_142507/`; it is a 32-output-token
diagnostic, not an EM gate. Its log confirms the guard rejects runtime 512/512
against captured 64/64.

That workaround has been superseded. The follow-up engine captures the full
configured query/KV capacities, replays initial/mid/final/mixed chunks, and
rejects unsupported FULL capacities instead of running eager. Chunk-final
flags now survive the Python/native bridge and suppress non-final sampling.
Padded replay uses independent query/KV tails, invalid inactive cache slots,
and compact live logits. FULL capture requires device executables throughout;
QY MiniCPM5-2B decode B1-B4 also uses device-only attention capture.

The graph-only four-sample long-context check in
`results/em_gate_20261028_172811/baseline/` scored 1/4, with predictions
D,D,A,A, matching the retained corrected reference. Its runtime trace contains
2168 device graph executions and 124 FULL prefill replays (116 mid), with no
host/op-list executions. This is subset evidence, not a 59-sample EM pass.
The optional all-kernel arm in `results/em_gate_20261028_171313/candidate/`
scored 0/4 (A,A,A,A); it remains unqualified and is not enabled by this repair.
No individual fused kernel has been isolated as the cause of that difference.

Until a complete quality run passes, earlier FULL-model latency comparisons
must not be treated as EM-qualified results. Isolated kernel timing and bitwise
operator checks remain separate evidence. See `PADDED_REPLAY_FIX.md` for final
replay validation and installation status. The new SiLU preparation remains
disabled and unvalidated. The rejected fallback-based 59-sample run was stopped.
