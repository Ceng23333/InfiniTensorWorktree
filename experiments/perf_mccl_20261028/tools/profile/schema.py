"""Profile summary schema for mcTracer Chrome-trace parses."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class KernelStat:
	name: str
	count: int
	total_ms: float
	max_ms: float

	def to_dict(self) -> Dict[str, Any]:
		return asdict(self)

	@staticmethod
	def from_dict(d: Dict[str, Any]) -> "KernelStat":
		return KernelStat(
			name=str(d["name"]),
			count=int(d["count"]),
			total_ms=float(d["total_ms"]),
			max_ms=float(d["max_ms"]),
		)


@dataclass
class HostStats:
	elapsed_s: Optional[float] = None
	tokens: Optional[int] = None
	tok_s: Optional[float] = None
	workload: Optional[str] = None
	ttft_s: Optional[float] = None
	prompt_tokens: Optional[int] = None

	def to_dict(self) -> Dict[str, Any]:
		return {k: v for k, v in asdict(self).items() if v is not None}

	@staticmethod
	def from_dict(d: Optional[Dict[str, Any]]) -> "HostStats":
		if not d:
			return HostStats()
		return HostStats(
			elapsed_s=_opt_float(d.get("elapsed_s")),
			tokens=_opt_int(d.get("tokens")),
			tok_s=_opt_float(d.get("tok_s")),
			workload=d.get("workload"),
			ttft_s=_opt_float(d.get("ttft_s")),
			prompt_tokens=_opt_int(d.get("prompt_tokens")),
		)


@dataclass
class ProfileSummary:
	backend: str
	gpu_kernel_ms: float = 0.0
	api_ms: float = 0.0
	dma_ms: float = 0.0
	tx_ms: float = 0.0
	fa_splitkv_ms: float = 0.0
	allreduce_ms: float = 0.0
	htod_ms: float = 0.0
	# Post-load proxy: totals after 95% of HTOD events by count (wrap includes weight-load).
	allreduce_ms_post_load: float = 0.0
	htod_ms_post_load: float = 0.0
	gpu_kernel_ms_post_load: float = 0.0
	top_kernels: List[KernelStat] = field(default_factory=list)
	host: HostStats = field(default_factory=HostStats)
	event_counts: Dict[str, int] = field(default_factory=dict)
	source: Optional[str] = None

	def to_dict(self) -> Dict[str, Any]:
		return {
			"backend": self.backend,
			"gpu_kernel_ms": self.gpu_kernel_ms,
			"api_ms": self.api_ms,
			"dma_ms": self.dma_ms,
			"tx_ms": self.tx_ms,
			"fa_splitkv_ms": self.fa_splitkv_ms,
			"allreduce_ms": self.allreduce_ms,
			"htod_ms": self.htod_ms,
			"allreduce_ms_post_load": self.allreduce_ms_post_load,
			"htod_ms_post_load": self.htod_ms_post_load,
			"gpu_kernel_ms_post_load": self.gpu_kernel_ms_post_load,
			"top_kernels": [k.to_dict() for k in self.top_kernels],
			"host": self.host.to_dict(),
			"event_counts": dict(self.event_counts),
			"source": self.source,
		}

	@staticmethod
	def from_dict(d: Dict[str, Any]) -> "ProfileSummary":
		return ProfileSummary(
			backend=str(d.get("backend") or "unknown"),
			gpu_kernel_ms=float(d.get("gpu_kernel_ms") or 0.0),
			api_ms=float(d.get("api_ms") or 0.0),
			dma_ms=float(d.get("dma_ms") or 0.0),
			tx_ms=float(d.get("tx_ms") or 0.0),
			fa_splitkv_ms=float(d.get("fa_splitkv_ms") or 0.0),
			allreduce_ms=float(d.get("allreduce_ms") or 0.0),
			htod_ms=float(d.get("htod_ms") or 0.0),
			allreduce_ms_post_load=float(d.get("allreduce_ms_post_load") or 0.0),
			htod_ms_post_load=float(d.get("htod_ms_post_load") or 0.0),
			gpu_kernel_ms_post_load=float(d.get("gpu_kernel_ms_post_load") or 0.0),
			top_kernels=[KernelStat.from_dict(x) for x in (d.get("top_kernels") or [])],
			host=HostStats.from_dict(d.get("host")),
			event_counts={str(k): int(v) for k, v in (d.get("event_counts") or {}).items()},
			source=d.get("source"),
		)


def _opt_float(v: Any) -> Optional[float]:
	if v is None or v == "":
		return None
	return float(v)


def _opt_int(v: Any) -> Optional[int]:
	if v is None or v == "":
		return None
	return int(v)
