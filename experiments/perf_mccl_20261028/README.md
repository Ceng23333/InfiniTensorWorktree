# Qwen3.8 MetaX performance worktree

Experimental source snapshot and profiling evidence; **not a release promotion**.
The stack pins the InfiniLM commit containing the tested MetaX deployment additions
and subsequent performance/metadata changes. InfiniCore remains pinned at the
recorded revision. Its InfiniCCL and InfiniRT sources match their upstream pins.
Four local InfiniOps compatibility changes are retained as an explicit overlay.

After recursively initializing submodules, run `bash scripts/apply_perf_mccl_overlay.sh`
from the stack root before rebuilding. The overlay is idempotent and includes the
MetaX FlashAttention ABI and CMake compatibility changes used by the measured pin.

Start with [output-throughput analysis](analysis/OUTPUT_THROUGHPUT_GAP.md), then the
[experiment report](REPORT.md) and [iteration results](RESULTS.md). The major newly
confirmed issue is that 300 recorded C++ objects have no optimization flag despite
the release label. That has **not** been fixed or benchmarked end to end in this snapshot.

`tools/` contains the local build/profile/analyze loop, numerical checks, and trace
validation. `results/` contains compact evidence; model weights, build artifacts,
container images, and multi-gigabyte raw traces are intentionally external. Saved
absolute paths and the initial local snapshot SHA identify the original experiment;
they are not portable paths or new upstream revisions.

The measured environment is `/home/metax/qy/workspace/pilotdeck/perf_mccl_loop`.
Tools expect a plain versioned source snapshot at `worktree/`, mounted inside the
experiment container at the legacy pin path, and this experiment directory mounted
at `/perf`. The scripts are environment-specific and require the recorded SDK,
container images, model, and installed dependency prefix. A clean bootstrap on a
new host was not validated. Build the standalone baseline kernel with
`tools/build_reference_kernel.sh` before the comparative GPU kernel test.

The runtime build scripts and the real five-kernel MetaX DSO rebuild are included.
The publishing branch preserves the measured implementation; release optimization
and further graph changes must be separate measured candidates. No speedup is
claimed for the unmeasured release-build correction.
