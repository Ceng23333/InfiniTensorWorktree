Fresh vLLM clean reference: TP4 BF16 Qwen3.8-27B, max length 32768,
prefix caching disabled, temperature 0, thinking explicitly disabled.
The shared decode prompt has 30 prompt tokens on both engines; the long prompt
has 10,718 tokens on both. Excluding warmup, median decode is 54.6006 tok/s;
long-prompt TTFT is 2.05893 s.

Accuracy smoke is not a valid pass for this reference: vLLM answered the arithmetic
probe with `2` and the capital probe with a corrective sentence ending in Paris.
The raw responses are retained. These probes expose a reference/model reliability
issue; they must not be silently marked passing or used to claim model accuracy.
The repeat-pad response is coherent (with an opening code fence); InfiniLM's earlier
gibberish/refusal outputs are a separate observable divergence.
