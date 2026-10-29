#!/usr/bin/env python3
"""Offline golden check for parse_mctracer (no mcTracer required)."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))

from profile.compare import compare_summaries
from profile.parse_mctracer import parse_mctracer
from profile.schema import HostStats, ProfileSummary


GOLDEN = Path(__file__).with_name("golden_trace.json")


class ParseGoldenTest(unittest.TestCase):
	def test_parse_pid_buckets(self) -> None:
		s = parse_mctracer(GOLDEN, backend="synthetic")
		# dur is us -> ms: API 2000us=2ms, GPU 50000+10000+20000=80ms, DMA 4ms, TX 1.5ms
		self.assertAlmostEqual(s.api_ms, 2.0, places=6)
		self.assertAlmostEqual(s.gpu_kernel_ms, 80.0, places=6)
		self.assertAlmostEqual(s.dma_ms, 4.0, places=6)
		self.assertAlmostEqual(s.tx_ms, 1.5, places=6)
		self.assertAlmostEqual(s.fa_splitkv_ms, 10.0, places=6)
		self.assertEqual(s.event_counts["gpu"], 3)
		self.assertEqual(s.top_kernels[0].name, "flash_fwd_kernel")
		self.assertAlmostEqual(s.top_kernels[0].total_ms, 50.0, places=6)

	def test_compare_floor(self) -> None:
		il = ProfileSummary(
			backend="infinilm",
			host=HostStats(tok_s=40.0, elapsed_s=1.0, tokens=40, workload="chat"),
		)
		vl = ProfileSummary(
			backend="vllm",
			host=HostStats(tok_s=100.0, elapsed_s=1.0, tokens=100, workload="chat"),
		)
		bad = compare_summaries(il, vl, tok_s_floor=0.5)
		self.assertFalse(bad["ok"])
		self.assertEqual(bad["status"], "PROFILE_REGRESS")

		il2 = ProfileSummary(
			backend="infinilm",
			host=HostStats(tok_s=60.0, elapsed_s=1.0, tokens=60, workload="chat"),
		)
		good = compare_summaries(il2, vl, tok_s_floor=0.5)
		self.assertTrue(good["ok"])
		self.assertAlmostEqual(good["tok_s_ratio"], 0.6, places=6)

	def test_roundtrip_dict(self) -> None:
		s = parse_mctracer(GOLDEN, backend="synthetic", host=HostStats(tok_s=12.5))
		blob = json.dumps(s.to_dict())
		s2 = ProfileSummary.from_dict(json.loads(blob))
		self.assertEqual(s2.backend, "synthetic")
		self.assertAlmostEqual(s2.host.tok_s or 0.0, 12.5, places=6)


if __name__ == "__main__":
	unittest.main()
