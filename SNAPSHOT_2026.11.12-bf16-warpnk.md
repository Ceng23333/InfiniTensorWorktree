# QY MoE bf16/warpnk decode — republish surpass pin

Tag: `v2026.11.12-qy-moe-bf16-warpnk`.
Republish of `v2026.11.09-qy-moe-bf16-warpnk` (same IC/IL/Ops/RT/CCL SHAs) under a
new annotated tag for the fresh InfiniTensor stacked-PR campaign.

## Included state

- Decode FULL one-segment (`device_segments=1`); tiles 64/64/32; DL align capacity.
- When `M*top_k/E <= 16`, skip `moe_align_` and pass size-1 sorted/expert/num_tokens
  metadata → `_invoke_fused_moe_bf16` + `warpnk` (prefill stays take/store).
- Unprofiled wall `20261109_112353_912929fa`: p512 **21.07** vs vLLM **20.88**
  (`WALL_DECODE_RATIO=1.008937`); p2048 **20.10** vs **18.30**. **Surpass.**
- InfiniCore SHA unchanged from the 11.09 surpass pin.

## Lock and publication

Umbrella worktree annotated tag fixes the source snapshot.
Nested InfiniLM shares this annotated tag name.
Prior tag `v2026.11.09-qy-moe-bf16-warpnk` remains as history.
