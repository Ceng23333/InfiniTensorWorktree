# QY MoE bf16/warpnk decode — 16a3 TP2 surpass

Tag: `v2026.11.09-qy-moe-bf16-warpnk`.
Builds on `v2026.11.09-qy-moe-decode-full` (WALL=0.558 checkpoint) with vLLM-matched
`use_moe_cu` size-1 MoE metadata under decode FULL.

## Included state

- Decode FULL one-segment (`device_segments=1`); tiles 64/64/32; DL align capacity.
- When `M*top_k/E <= 16`, skip `moe_align_` and pass size-1 sorted/expert/num_tokens
  metadata → `_invoke_fused_moe_bf16` + `warpnk` (prefill stays take/store).
- Unprofiled wall `20261109_112353_912929fa`: p512 **21.07** vs vLLM **20.88**
  (`WALL_DECODE_RATIO=1.008937`); p2048 **20.10** vs **18.30**. **Surpass.**
- InfiniCore SHA unchanged.

## Lock and publication

Umbrella worktree annotated tag fixes the source snapshot.
Nested InfiniLM shares this annotated tag name.
