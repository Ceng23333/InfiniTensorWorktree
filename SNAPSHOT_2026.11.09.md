# QY MoE decode-FULL WALL=0.558 checkpoint

Tag: `v2026.11.09-qy-moe-decode-full`.
Pre-surpass InfiniLM pin after decode FULL nosplit + MoE tile/align Denglin capacity work.

## Included state

- Decode FULL one-segment (`device_segments=1`); no PiecewiseDecode.
- MoE tiles `64/64/32`, align=64; vLLM Denglin align capacity in fused MoE runner.
- Harness checkpoint run `20261109_100933_35f7f8f8`: p512≈11.65, p2048≈11.41,
  `WALL_DECODE_RATIO=0.558` vs frozen vLLM 20.88 (still below; **not surpass**).
- InfiniCore SHA unchanged from `v2026.11.06-qy-full-pd`.

## Not included / not claimed

- Surpass of vLLM decode throughput.
- bf16/warpnk MoE kernel family dispatch (open gap → iter4).

## Lock and publication

Umbrella worktree annotated tag fixes the source snapshot.
Submodule URLs remain the `Ceng23333` forks. Nested InfiniLM shares this
annotated tag name.
