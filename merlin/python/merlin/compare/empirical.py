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
from merlin.common.paths import repo_root
from typing import Any

from .spec import Config, Workload


# ---- repo-relative source locations (deterministic, recorded into the manifest) ----
def _repo_root() -> Path:
    return repo_root()


# Whole-model JSONs, in PREFERENCE order per workload. A vf-bearing file is preferred when a config
# needs ours_wholemodel_vf (only k1_vf_*.json carries it); otherwise the 4-way file is canonical.
_MODEL_SOURCES = {
    "openvla": ["out/artifacts/kernel-mining/rvv/bench/k1_vf_openvla.json", "out/artifacts/kernel-mining/rvv/bench/k1_4way_openvla.json"],
    "rdt2": ["out/artifacts/kernel-mining/rvv/bench/k1_4way_rdt2.json"],
    "bitvla": ["out/artifacts/kernel-mining/rvv/bench/k1_4way_bitvla.json"],
}
_GEMM_SOURCE = "out/artifacts/ceiling/cross_framework_matrix_k1.jsonl"

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


# workload name -> its recapture bundle (the model2MLIR capture the four-way driver builds from).
def _model_dir(name: str, root: Path) -> Path | None:
    exact = {"bitvla": "bitvla_fp32_consistent", "openvla": "openvla_fp32_consistent",
             "rdt2": "rdt2_fp32_consistent", "tiny_llama": "tiny_llama_bf16"}
    rec = root / "out/artifacts/recaptures"
    if name in exact and (rec / exact[name]).is_dir():
        return rec / exact[name]
    hits = sorted(rec.glob(f"{name}*")) if rec.is_dir() else []
    return hits[0] if hits else None


def _four_way_out(name: str, root: Path) -> Path:
    """The k1_4way JSON path _ingest_model reads first (see _MODEL_SOURCES)."""
    return root / "out/artifacts/kernel-mining/rvv/bench" / f"k1_4way_{name}.json"


def prime_board_cache(spec, root: Path, *, n: int = 3) -> dict[str, str]:
    """LIVE seam: run the four-way board driver ONCE per model workload (all arms in one pass) and
    write the k1_4way_<model>.json cache that ``_ingest_model`` reads. Board-gated + board_lock via
    the driver's own run_on_k1. Returns {workload: status}. gemm-shape workloads are primed by the
    per-op driver (a declared follow-on; not yet wired here)."""
    import sys as _sys

    from merlin.rvvgen import k1 as _k1
    if not _k1.available():
        raise RuntimeError("merlin-compare --run: K1 board unavailable (set MERLIN_K1_HOST / "
                           "MERLIN_K1_SSH_KEY / MERLIN_K1_TOOLCHAIN in .env). Remove --run to ingest cache.")
    _sys.path.insert(0, str(root / "build_tools" / "scripts"))
    import k1_e2e_xnnpack as _e2e   # the refactored driver exposing run_workload()

    status: dict[str, str] = {}
    models = [wl for wl in spec.workloads if wl.kind == "model"]
    for wl in models:
        md = _model_dir(wl.name, root)
        if md is None:
            status[wl.name] = "no_recapture"
            continue
        out = _four_way_out(wl.name, root)
        _e2e.run_workload(md, n=n, out=str(out))   # writes the schema _ingest_model reads
        status[wl.name] = f"measured -> {out}"
    return status


# external framework (executorch / tvm / buddy / exo / ggml) e2e results live in the independent
# cross-framework BaselineResult tree, NOT the four-way JSON. Ingest them via the aggregate collector
# so an `executorch` config is a first-class ARM in the SAME matrix as the inside-Merlin arms.
_BASELINE_TREE = "out/artifacts/measurements/k1_spacemit"


def _ingest_external(cfg: Config, wl: Workload, target: str, metric: str, root: Path) -> Measurement:
    from ..baselines import aggregate as _agg
    tree = root / _BASELINE_TREE
    if not tree.is_dir():
        return Measurement(config=cfg.name, workload=wl.name, target=target, metric=metric,
                           status="not_measured")
    rows = _agg.dedupe_latest(_agg.collect_dir(tree))
    mine = [r for r in rows if r.framework == cfg.name and r.model == wl.name and r.e2e_wall_ns]
    # not_run_is_not_pass: only a PASSING result counts as a real number. A fail (e.g. ET openvla's
    # degenerate 16 ms) is NOT reported as measured — it's a not_measured cell with the reason kept.
    passing = [r for r in mine if r.status() == "pass"]
    passing.sort(key=lambda r: (r.variant == "fp32"), reverse=True)   # prefer fp32 (matches four-way)
    r = passing[0] if passing else None
    if r is None:
        fail = next((x for x in mine), None)
        return Measurement(config=cfg.name, workload=wl.name, target=target, metric=metric,
                           status="not_measured",
                           detail={"reason": f"no passing {cfg.name} result"
                                   + (f" (latest {fail.variant}={fail.status()})" if fail else "")})
    return Measurement(config=cfg.name, workload=wl.name, target=target, metric=metric,
                       status="measured", value=float(r.e2e_wall_ns),
                       cos=getattr(r, "cos", None), source=f"{_BASELINE_TREE} (baselines.aggregate)",
                       detail={"framework": r.framework, "variant": r.variant, "status": r.status()})


def measure(cfg: Config, wl: Workload, target: str, metric: str = "wall", *,
            run: bool = False, root: Path | None = None) -> Measurement:
    """The measurement seam. Ingests cached numbers; when ``run=True`` the board cache is refreshed
    first by :func:`prime_board_cache` (called once in :func:`measure_all`), so this per-cell path
    always ingests the freshly-written JSON. External-framework arms (executorch/tvm/...) ingest from
    the independent cross-framework BaselineResult tree so they sit in the SAME matrix as ours/xnn."""
    root = root or _repo_root()
    if cfg.kind == "external":
        return _ingest_external(cfg, wl, target, metric, root)
    if wl.kind == "gemm":
        return _ingest_gemm(cfg, wl, target, metric, root)
    return _ingest_model(cfg, wl, target, metric, root)


def measure_all(spec, *, run: bool = False, root: Path | None = None) -> dict:
    """Return {(config_name, workload_name): Measurement} for the full spec grid. When ``run=True``,
    refresh the board cache once (all model workloads, all arms) BEFORE ingesting."""
    root = root or _repo_root()
    if run:
        prime_board_cache(spec, root)
    out: dict[tuple[str, str], Measurement] = {}
    for cfg in spec.configs:
        for wl in spec.workloads:
            out[(cfg.name, wl.name)] = measure(cfg, wl, spec.target, spec.metric,
                                               run=run, root=root)
    return out
