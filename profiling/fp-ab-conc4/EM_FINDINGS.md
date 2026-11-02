# EM correctness gate

Date: 2026-10-28, Asia/Shanghai. Status: investigating; kernel promotion held.

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

A candidate engine adds a conservative captured-length guard (eager fallback
outside the capture bounds), independent query/KV tail padding, invalid FULL
padding slots, and output copies restricted to live requests. The first short
long-context control is `results/em_gate_20261028_142507/`; it is a 32-output-token
diagnostic, not an EM gate. Its log confirms the guard rejects runtime 512/512
against captured 64/64.

Until a complete quality run passes, earlier FULL-model latency comparisons
must not be treated as EM-qualified results. Isolated kernel timing and bitwise
operator checks remain separate evidence. No installed binaries have been
replaced. The new SiLU preparation remains disabled and unvalidated.
