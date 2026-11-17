Run from `/home/metax/qy/workspace/pilotdeck/perf_mccl_loop`.

- `loop.py`: checked patch queue, rebuild, clean benchmark, traced benchmark, analysis, gates, restoration. Add a uniquely named candidate to `queue.json` and run `python3 tools/loop.py --restore-production`. It resumes the ledger; completed names are not rerun. `--watch` waits for newly added candidates, with production restored while idle.
- `rebuild.sh`: runs in the isolated container; rebuilds InfiniCore runtime / InfiniLM and all five MetaX extra kernels. It installs the real provider DSO. Standalone preload candidates are diagnostic history, not the deployment path.
- `bench.py`: complete-response checks and epoch-ns request boundaries. No invented fallback tokens. `smoke.py` adds explicit answer probes; they currently fail even on the vLLM arithmetic reference.
- `analyze.py`: request windows, per-device kernels and all-reduce rank-arrival decomposition. Requires all four GPUs and matched collective counts; does not translate summed residency to wall-clock savings.
- `tracer_config/mcTracer.json`: optional kernel/DMA-only mcTracer config (set `MACA_ETC_PATH=/perf/tools/tracer_config` in the traced process). Standalone capture tested; see the report for inference outcome.
- `vllm_capture.py`: fresh eager kernel capture, per-process rank attribution from MCCL markers, shard validation. Default graph-enabled vLLM host metrics are separate.
- `write_summary.py`: regenerates `RESULTS.md` from preserved iteration states.

Source snapshot: `worktree/`; original production pin remains untouched. Test: `python3 tools/test_profile.py`, `python3 tools/test_bench.py`, `python3 tools/profile/testdata/test_parse_golden.py`. GPU/reference tests: `docker exec infinilm-perf-loop python /perf/tools/test_cgdr.py`; processor tests: `test_positions.py`, `test_norm.py` in the same container.

The runner's proximity goal defaults to 80% of reference throughput and ≤1.25× reference TTFT, and also requires response smoke, fixed decode length, and valid trace coverage. **These acceptance gates have not passed.** Failed experiments remain reviewable; no production promotion occurs.
