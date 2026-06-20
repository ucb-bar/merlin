"""LAYER 1 — EMPIRICAL: the measured table, behind a ``measure(config, workload, target)`` seam.

v1 INGESTS already-measured numbers from the cached harness JSONs (does NOT run the board). The
``Measurement`` interface is the seam where a future ``--run`` would call
``k1_e2e_xnnpack.run_cfg`` / ``k1_cross_framework.measure_*`` to take new board measurements; the
``run=True`` path is a stub that fails loud so nothing silently fabricates a board run.

Honest: a (config, workload) cell missing from every cached source is reported as
``status='not_measured'`` (never invented).

Spec-name -> cached-JSON-key mapping (the RVV/K1 particularity, kept here so the spec stays
target-agnostic):
  baseline             -> "baseline"
  ours_wholemodel      -> "ours_wholemodel"
  ours_wholemodel_vf   -> "ours_wholemodel_vf"
  ours_v3 / ours_tiled -> "ours_v3" / "ours_tiled"   (any ours_* maps to its like-named key)
  xnnpack              -> "xnnpack_kernels"
  openblas             -> "openblas_kernels"
For isolated gemm shapes the source is ``cross_framework_matrix_k1.jsonl`` whose ``source`` field is
  baseline -> "ours_baseline", xnnpack -> "xnnpack", openblas -> "openblas",
  ours_*   -> "ours-intrinsic" (best ours intrinsic kernel) else the like-named source.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .spec import Config, Workload


# ---- repo-relative source locations (deterministic, recorded into the manifest) ----
def _repo_root() -> Path:
    # merlin/python/merlin/compare/empirical.py -> repo root is parents[4]
    return Path(__file__).resolve().parents[4]


# Whole-model JSONs, in PREFERENCE order per workload. A vf-bearing file is preferred when a config
# needs ours_wholemodel_vf (only k1_vf_*.json carries it); otherwise the 4-way file is canonical.
_MODEL_SOURCES = {
    "openvla": ["output/rvv_bench/k1_vf_openvla.json", "output/rvv_bench/k1_4way_openvla.json"],
    "rdt2": ["output/rvv_bench/k1_4way_rdt2.json"],
    "bitvla": ["output/rvv_bench/k1_4way_bitvla.json"],
}
_GEMM_SOURCE = "output/kernels/ceiling/cross_framework_matrix_k1.jsonl"

# spec config name -> whole-model JSON key
_MODEL_KEY = {
    "baseline": "baseline",
    "xnnpack": "xnnpack_kernels",
    "openblas": "openblas_kernels",
}
# spec config name -> cross-framework JSONL `source`
_GEMM_SOURCE_KEY = {
    "baseline": "ours_baseline",
    "xnnpack": "xnnpack",
    "openblas": "openblas",
}


@dataclass
class Measurement:
    config: str
    workload: str
    target: str
    metric: str
    status: str                     # "measured" | "not_measured"
    value: float | None = None      # min wall (ns) or instret, per metric
    spread_pct: float | None = None
    cos: float | None = None
    source: str | None = None       # the file the number came from (provenance)
    detail: dict[str, Any] = field(default_factory=dict)


def _model_key_for(cfg: Config) -> str:
    if cfg.name in _MODEL_KEY:
        return _MODEL_KEY[cfg.name]
    # ours_* configs map to their like-named JSON key.
    return cfg.name


def _gemm_source_for(cfg: Config) -> str:
    if cfg.name in _GEMM_SOURCE_KEY:
        return _GEMM_SOURCE_KEY[cfg.name]
    if cfg.name.startswith("ours"):
        return "ours-intrinsic"
    return cfg.name


def _ingest_model(cfg: Config, wl: Workload, target: str, metric: str,
                  root: Path) -> Measurement:
    key = _model_key_for(cfg)
    for rel in _MODEL_SOURCES.get(wl.name, []):
        p = root / rel
        if not p.is_file():
            continue
        data = json.loads(p.read_text())
        node = data.get(key)
        if not isinstance(node, dict):
            continue
        if node.get("skipped") or node.get("min_wall_ns") is None:
            continue
        spread = node.get("spread") or {}
        return Measurement(
            config=cfg.name, workload=wl.name, target=target, metric=metric,
            status="measured",
            value=float(node["min_wall_ns"]),
            spread_pct=spread.get("range_pct"),
            cos=node.get("fp32_cos"),
            source=rel,
            detail={"run_id": node.get("run_id"), "tag": node.get("tag"),
                    "compiler_features": node.get("compiler_features")},
        )
    return Measurement(config=cfg.name, workload=wl.name, target=target, metric=metric,
                       status="not_measured")


def _ingest_gemm(cfg: Config, wl: Workload, target: str, metric: str,
                 root: Path) -> Measurement:
    src = _gemm_source_for(cfg)
    m, n, k = wl.mnk
    p = root / _GEMM_SOURCE
    if p.is_file():
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            if o.get("source") == src and (o.get("M"), o.get("N"), o.get("K")) == (m, n, k):
                wall = o.get("wall_ns_est")
                return Measurement(
                    config=cfg.name, workload=wl.name, target=target, metric=metric,
                    status="measured" if wall is not None else "not_measured",
                    value=float(wall) if wall is not None else None,
                    spread_pct=None,
                    cos=None,
                    source=_GEMM_SOURCE,
                    detail={"ticks": o.get("ticks"), "kernel_file": o.get("kernel_file"),
                            "kernel_status": o.get("status"), "mnk": [m, n, k]},
                )
    return Measurement(config=cfg.name, workload=wl.name, target=target, metric=metric,
                       status="not_measured")


def measure(cfg: Config, wl: Workload, target: str, metric: str = "wall", *,
            run: bool = False, root: Path | None = None) -> Measurement:
    """The measurement seam. v1 ingests cached numbers; ``run=True`` is the (unimplemented in v1)
    hook where a board run would be launched. Do NOT touch the board in v1."""
    root = root or _repo_root()
    if run:
        # SEAM: this is where k1_e2e_xnnpack.run_cfg / k1_cross_framework.measure_* would be called
        # to take fresh board measurements. Left as a loud stub so v1 never silently runs the board.
        raise NotImplementedError(
            "merlin-compare v1 ingests cached measurements only; --run (live board measurement) "
            "is a declared seam and is not implemented. Remove --run to ingest.")
    if wl.kind == "gemm":
        return _ingest_gemm(cfg, wl, target, metric, root)
    return _ingest_model(cfg, wl, target, metric, root)


def measure_all(spec, *, run: bool = False, root: Path | None = None) -> dict:
    """Return {(config_name, workload_name): Measurement} for the full spec grid."""
    out: dict[tuple[str, str], Measurement] = {}
    for cfg in spec.configs:
        for wl in spec.workloads:
            out[(cfg.name, wl.name)] = measure(cfg, wl, spec.target, spec.metric,
                                               run=run, root=root)
    return out
