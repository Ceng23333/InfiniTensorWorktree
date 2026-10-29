# InfiniLM production snapshot — 2026-10-29

This release records the source and serving configuration of
`pilotdeck-infinilm:promoted-tool-fix-20261029`, image
`sha256:d92545d02ac8270090f5fee09e276b768504cf7f5c8b7fc983376b336ac8f70e`.
The InfiniLM submodule includes optimized release rules, the InfiniOps metadata
query bridge, tokenizer vocabulary caching, SiLU graph recording, graph allocation
retention, and corrected prefill KV slot mapping. The opt-in graph checkpoints are
present in the validated source but disabled by default.

## Serving configuration

Qwen3.8-27B, BF16, MetaX TP4, batch limit 1, paged KV 512 x 256,
2048-token chunked prefill, maximum output 4096. Native segmented decode graphs
are enabled (`ENABLE_GRAPH=1`, `INFINI_CUDAGRAPH_POLICY=legacy`,
`INFINI_FA_INGRAPH=decode`), with 17 device segments and 16 attention host breaks
per rank. Native prefill capture, prefix caching and thinking are disabled.
Normal EOS stopping is enabled. Hybrid prefix caching is not implemented here.

## Source checkout and build

Clone this tag with recursive submodules. The top-level MANIFEST pins InfiniCore,
InfiniLM and the existing InfiniOps compatibility overlay. Apply the overlay to
the pinned InfiniOps source if rebuilding providers:

```bash
git -C InfiniCore/submodules/InfiniOps apply --check ../../../experiments/perf_mccl_20261028/infiniops-compatibility.patch
git -C InfiniCore/submodules/InfiniOps apply ../../../experiments/perf_mccl_20261028/infiniops-compatibility.patch
```

The overlay path above is relative to InfiniOps. This source tag does not contain
model weights, compiled libraries or the local MetaX dependency Docker image.
The original dependency image was `pilotdeck-infinilm:with-deps` (MACA 3.8.2.6,
Python 3.10, Torch 2.10). Historical stack build helpers in `scripts/` require
that SDK/toolchain; inspect their local-path defaults before rebuilding elsewhere.
Do not build into a running production install.

`rebuild.sh` explicitly configures release mode and rebuilds all three host
artifacts; it assumes the provider headers/libraries and xmake ABI configuration
are already prepared. Set HOST_ONLY=1 INSTALL_HOST=0 HOST_STAGE=/path/to/stage
for isolated staging. The retained provider hashes are in
`production-runtime-manifest.json`; a fresh rebuild is not asserted bit-identical.
`Dockerfile.runtime` packages a prepared install and Python libraries. The manifest
records the original deployed artifact identities, not identities of a new build.

## Verification and operations

`validation.json` records passing Bash execution, Write/Read with file-content
verification, four prefill slot tests and the original Chinese outline regression.
The latter created a 6296-byte outline with one Write call and no errors.
`runtime_verification.json` records loaded host/provider hashes.

`start.sh` creates the production container using the versioned image and a
read-only model mount; it refuses to replace a container with a different image.
`wait_health.py` verifies readiness. The model mount path is site-specific.
Production backend and Pilotdeck use `unless-stopped`; vLLM containers have
restart disabled. These are recorded policies, not changes made by checking out
this tag. Production still has known long-tool-stream buffering/idle-timeout and
long-context performance limitations; the validation is not a broad quality claim.

Live tool smoke: run `pilotdeck_tool_smoke.py OUTPUT_DIRECTORY` on the Pilotdeck
host, using a fresh directory beneath its mounted `/home/metax/qy` workspace.
It requires successful tool calls and actual file/read-back evidence, not merely
a text answer. Raw user sessions and profiling traces are excluded from this tag.
