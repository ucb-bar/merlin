"""Measured dispatch coupling — the first measured runtime leg.

Most of the framework is structural; this is one place a previously-*estimated* quantity becomes
*measured*. The command_batching / autonomous_K_loop axes hinge on "dispatches per replan", which
we had proxied by the matmul count. Here we run a real captured model through the host reference
executor (:func:`merlin.runtime.dispatch_runtime.run_model`) and count the **actual** dispatches
issued per forward — all ops, not just matmuls.

The measurements live in ``merlin/benchmarks/dse_guidance/measured_dispatch.yaml`` (recorded with
``measure()``; cos=1.0 vs the torch golden, so the runs are faithful). The clean measured datum is
``n_kernels`` (dispatch count). Wall time is host-interpreter timing on the Python executor — not
the deployable runtime — so it grounds dispatch *counts*, not absolute latency. No speedup is
claimed; the benefit of batching still needs the per-dispatch host cost on the real runtime.
"""
from __future__ import annotations

import csv
import io
from dataclasses import dataclass

from merlin.common import paths
from merlin.common.yaml import load_yaml


@dataclass
class DispatchMeasurement:
    model: str
    n_kernels: int                 # MEASURED total dispatches per forward
    n_unique: int
    matmul_estimate: int           # structural proxy used before (matmul count)
    cos: float
    exec_wall_s: float | None = None
    # Per-dispatch host-cost breakdown (P1-b), host reference executor (machine-dependent ms).
    compute_call_ms: float | None = None
    host_overhead_ms: float | None = None
    overhead_frac: float | None = None    # dispatch/alloc overhead / total host time (stable)

    @property
    def undercount_ratio(self) -> float | None:
        """How much the matmul-count proxy under-counts the real dispatch granularity."""
        return (self.n_kernels / self.matmul_estimate) if self.matmul_estimate else None

    @property
    def per_dispatch_host_ms(self) -> float | None:
        """Total host time per dispatch (compute call + its share of overhead)."""
        if self.compute_call_ms is None or self.host_overhead_ms is None or not self.n_kernels:
            return None
        return (self.compute_call_ms + self.host_overhead_ms) / self.n_kernels


def load_measured(path=None) -> list[DispatchMeasurement]:
    p = path or (paths.bench_dir() / "dse_guidance" / "measured_dispatch.yaml")
    doc = load_yaml(p)
    def _f(r, k):
        return float(r[k]) if r.get(k) is not None else None
    return [DispatchMeasurement(
        model=r["model"], n_kernels=int(r["n_kernels"]), n_unique=int(r.get("n_unique", 0)),
        matmul_estimate=int(r.get("matmul_estimate", 0)), cos=float(r.get("cos", 0.0)),
        exec_wall_s=_f(r, "exec_wall_s"), compute_call_ms=_f(r, "compute_call_ms"),
        host_overhead_ms=_f(r, "host_overhead_ms"), overhead_frac=_f(r, "overhead_frac"))
        for r in doc.get("points", [])]


def measure_host_breakdown(model_dir, workdir=None, cache_dir=None) -> DispatchMeasurement:
    """Measure the per-forward host-time split (compute calls vs dispatch/alloc overhead).

    Uses a timing tap over the executor (no runtime modification): each op's wall time is the
    delta between consecutive tap calls. ``func.call`` ops are the compiled-kernel dispatches;
    everything else (output allocation, view/glue) is host dispatch overhead. ``overhead_frac``
    is the stable signal; absolute ms are host-interpreter, machine-dependent.
    """
    import tempfile
    import time
    from pathlib import Path
    from merlin.runtime import dispatch_runtime as DR
    md = Path(model_dir)
    wd = Path(workdir or tempfile.mkdtemp())
    cache = Path(cache_dir or tempfile.mkdtemp())
    DR.run_model(md, wd, cache_dir=cache)                 # warm the compile cache
    events: list = []
    t0 = time.perf_counter()
    res = DR.run_model(md, wd, cache_dir=cache,
                       tap=lambda op, val: events.append((op.name, time.perf_counter())))
    compute = overhead = 0.0
    ndisp = 0
    prev = t0
    for name, t in events:
        dt = t - prev
        prev = t
        if name == "func.call":
            compute += dt
            ndisp += 1
        else:
            overhead += dt
    total = compute + overhead
    matmuls = (md / "model.mlir").read_text().count("linalg.matmul")
    return DispatchMeasurement(
        model=md.name, n_kernels=int(res["n_kernels"]),
        n_unique=int(res.get("n_unique_kernels") or 0), matmul_estimate=matmuls,
        cos=float(res.get("cos", 0.0)), exec_wall_s=round(total, 3),
        compute_call_ms=round(compute * 1e3, 1), host_overhead_ms=round(overhead * 1e3, 1),
        overhead_frac=round(overhead / total, 3) if total else None)


def measure(model_dir, workdir=None, cache_dir=None) -> DispatchMeasurement:
    """Run a captured model through the host reference executor and measure its dispatch count.

    Reproduces a row of ``measured_dispatch.yaml``. Needs the capture (model.mlir + weights +
    inputs + golden) and a host C compiler. Raises if the runtime is unavailable.
    """
    import tempfile
    import time
    from pathlib import Path
    from merlin.runtime import dispatch_runtime as DR
    md = Path(model_dir)
    wd = Path(workdir or tempfile.mkdtemp())
    cache = Path(cache_dir or tempfile.mkdtemp())
    DR.run_model(md, wd, cache_dir=cache)                 # warm the compile cache
    t0 = time.perf_counter()
    res = DR.run_model(md, wd, cache_dir=cache)
    wall = time.perf_counter() - t0
    matmuls = (md / "model.mlir").read_text().count("linalg.matmul")
    return DispatchMeasurement(
        model=md.name, n_kernels=int(res["n_kernels"]),
        n_unique=int(res.get("n_unique_kernels") or 0), matmul_estimate=matmuls,
        cos=float(res.get("cos", 0.0)), exec_wall_s=round(wall, 3))


def calibration_rows(measured: list[DispatchMeasurement] | None = None) -> list[dict]:
    """Predicted (matmul-count proxy) vs measured dispatch count — a real calibration anchor."""
    rows = []
    for m in (measured or load_measured()):
        err = ((m.matmul_estimate - m.n_kernels) / m.n_kernels * 100.0) if m.n_kernels else None
        rows.append({
            "model": m.model, "quantity": "dispatches_per_forward",
            "predicted_matmul_proxy": m.matmul_estimate, "measured": m.n_kernels,
            "error_pct": None if err is None else round(err, 1),
            "undercount_ratio": None if m.undercount_ratio is None else round(m.undercount_ratio, 1),
            "evidence_type": "measured",
        })
    return rows


def report_md(measured: list[DispatchMeasurement] | None = None) -> str:
    ms = measured or load_measured()
    L = ["# Measured dispatch coupling (host reference executor)\n"]
    L.append("> The dispatch *count* per forward is measured by running the real captured model "
             "through the dispatch runtime (cos=1.0 vs the torch golden). It grounds the "
             "`dispatches per replan` input to `command_batching` / `autonomous_K_loop`, which was "
             "previously estimated from the matmul count. No speedup is claimed — the per-dispatch "
             "host cost on the deployable runtime is still required to quantify a benefit.\n")
    L.append("| model | matmul estimate | measured dispatches | undercount | unique | cos |")
    L.append("|-------|-----------------|---------------------|-----------|--------|-----|")
    for m in ms:
        uc = "n/a" if m.undercount_ratio is None else f"{m.undercount_ratio:.0f}x"
        L.append(f"| {m.model} | {m.matmul_estimate} | {m.n_kernels} | {uc} | {m.n_unique} | "
                 f"{m.cos:.4f} |")
    L.append("")
    if ms:
        avg = sum(m.undercount_ratio for m in ms if m.undercount_ratio) / len(ms)
        L.append(f"**Finding:** the matmul-count proxy under-counts real dispatch granularity by "
                 f"~{avg:.0f}x (real dispatches include every elementwise/norm/view/glue kernel). "
                 "So the command-batching / autonomous-loop opportunity is *larger* than the "
                 "matmul-only estimate implied — now grounded in a measured dispatch count.\n")
    # Per-dispatch host-cost breakdown (P1-b)
    bd = [m for m in ms if m.overhead_frac is not None]
    if bd:
        L.append("## Per-dispatch host cost (host reference executor)\n")
        L.append("| model | dispatches | compute-call ms | dispatch/alloc overhead ms | overhead frac |")
        L.append("|-------|-----------|-----------------|----------------------------|---------------|")
        for m in bd:
            L.append(f"| {m.model} | {m.n_kernels} | {m.compute_call_ms} | {m.host_overhead_ms} | "
                     f"{m.overhead_frac:.2f} |")
        avgf = sum(m.overhead_frac for m in bd) / len(bd)
        L.append("")
        L.append(f"**Finding:** ~{avgf*100:.0f}% of host time per forward is dispatch/allocation "
                 "overhead, NOT compute-kernel calls — the forward is **host-dispatch-bound** on "
                 "this executor, which is exactly the regime `command_batching` / "
                 "`autonomous_K_loop` target. Absolute ms are host-interpreter (Python reference "
                 "executor), machine-dependent — the stable, deployable-relevant signals are the "
                 "dispatch *count* and the overhead *fraction*, not the absolute latency.\n")
    return "\n".join(L)


def to_csv(rows: list[dict]) -> str:
    cols = ["model", "quantity", "predicted_matmul_proxy", "measured", "error_pct",
            "undercount_ratio", "evidence_type"]
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols)
    w.writeheader()
    for r in rows:
        w.writerow({k: ("" if r.get(k) is None else r[k]) for k in cols})
    return buf.getvalue()
