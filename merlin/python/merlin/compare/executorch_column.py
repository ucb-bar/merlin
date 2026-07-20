"""Read the TRUE ExecuTorch(+XNNPACK) whole-model e2e result for one comparison cell — board-free.

The autonomous beam experiment compares the beam-discovered fork and the manual four-way against the
XNNPACK / OpenBLAS GEMM microkernels **routed inside OUR OWN runtime** (the ``xnnpack_kernels`` arm of
``k1_4way_<model>.json``). That is deliberately NOT a standalone external system: XNNPACK-in-our-runtime
shares our loader, our arena planner, and our op dispatch — only the GEMM microkernel is XNNPACK's.

ExecuTorch (+ the XNNPACK delegate) is the TRUE external whole-model system: its OWN runtime executes
the entire model, delegating what its partitioner can claim to XNNPACK's RVV microkernels and running
everything else on portable/scalar kernels. This module reads the latest **passing** ExecuTorch
``BaselineResult`` for a ``(model, dtype)`` cell via :mod:`merlin.baselines.aggregate` — it does NOT run
the board — and returns a structured, honestly-labeled column for the report.

Fail-closed (``not_run_is_not_pass``): a fail / not_run / absent ExecuTorch result yields
``status='not_measured'`` with a concrete reason — never a fabricated number and never another cell's
number (e.g. an int8 cell never borrows the fp32 wall).
"""
from __future__ import annotations

from pathlib import Path

from merlin.baselines import aggregate as _agg
from merlin.baselines import bundle as _bundle
from merlin.common.paths import repo_root

# The cross-framework BaselineResult tree (same location compare.empirical._ingest_external reads).
_BASELINE_TREE = "out/artifacts/measurements/k1_spacemit"

# beam dtype token -> BaselineResult.variant. Kept explicit so a new dtype fails loud, not silent.
_DTYPE_TO_VARIANT = {"fp32": "fp32", "int8": "int8", "fp16": "fp16"}

# --- comparability labels: the in-runtime XNNPACK GEMM arm vs the true external system ----------
XNNPACK_KERNELS_LABEL = (
    "xnnpack_kernels = XNNPACK GEMM microkernel ROUTED INSIDE OUR runtime (four-way arm; shares our "
    "loader/arena/dispatch — only the GEMM is XNNPACK's, NOT a standalone external system)")
EXECUTORCH_LABEL = (
    "executorch = ExecuTorch + XNNPACK delegate — the TRUE external whole-model system (its OWN "
    "runtime executes the entire model; portable/scalar kernels run everything the partitioner can't "
    "delegate)")


def _collect(root: Path) -> list:
    """Latest-per-cell BaselineResults under the measurements tree (empty if the tree is absent)."""
    tree = root / _BASELINE_TREE
    if not tree.is_dir():
        return []
    return _agg.dedupe_latest(_agg.collect_dir(tree))


def gate_basis(model: str) -> str:
    """Per-model honesty label for what a passing cos actually MEANS on this model.

    Random-init models ship no reproducible weights, so their gate is LOWERING-EXACTNESS (framework
    vs eager-torch on this seeded instantiation), not a semantic match against trained weights."""
    if _bundle.golden_unreproducible(model):
        return ("lowering-exactness (random-init model: cos measures framework-vs-eager-torch on THIS "
                "seeded instantiation, NOT a semantic match against trained weights)")
    return "semantic (gated against the model's captured trained-weight golden)"


def _not_measured_reason(model: str, variant: str, rows: list) -> str:
    """Honest reason a cell has no ExecuTorch number: an executed-but-failed run, a not_run, or absence."""
    if rows:
        r = rows[0]  # dedupe_latest yields exactly one row per (framework, model, variant)
        gap = (r.gap_reason or "").strip()
        detail = f": {gap}" if gap else ""
        if r.status() == "fail" and r.cos is not None:
            detail += f" (cos={r.cos})"
        return f"latest executorch {variant} = {r.status()}{detail}"
    if model in _bundle.K1_RAM_INFEASIBLE:
        return (f"no executorch {variant} result; {model} is K1 RAM-infeasible whole-model (7B-class "
                "VLA exceeds the 3.8 GB board) -> honest not_run, never a false fit")
    return f"no executorch {variant} result under {_BASELINE_TREE} (board run not yet performed)"


def executorch_cell(model: str, dtype: str, *, root: Path | None = None) -> dict:
    """The ExecuTorch column for a ``(model, dtype)`` cell — structured + honestly labeled.

    Returns a dict always carrying ``executorch_status`` (``measured`` | ``not_measured``),
    ``executorch_wall_ns`` (a real number ONLY when a passing ExecuTorch run of the SAME variant
    exists, else ``None``), the ``variant`` selected, the per-model ``gate_basis`` honesty label, the
    comparability ``label``, and — when not measured — a concrete ``reason``.
    """
    root = root or repo_root()
    variant = _DTYPE_TO_VARIANT.get(dtype, dtype)
    rows = [r for r in _collect(root)
            if r.framework == "executorch" and r.model == model and r.variant == variant]
    passing = [r for r in rows if r.status() == "pass" and r.e2e_wall_ns]
    basis = gate_basis(model)
    if passing:
        r = passing[0]
        return {
            "executorch_status": "measured",
            "executorch_wall_ns": float(r.e2e_wall_ns),
            "variant": r.variant,
            "cos": r.cos,
            "rel": r.rel,
            "rvv_coverage_overall": r.rvv_coverage_overall,
            "gate_basis": basis,
            "label": EXECUTORCH_LABEL,
            "source": f"{_BASELINE_TREE} (baselines.aggregate)",
        }
    return {
        "executorch_status": "not_measured",
        "executorch_wall_ns": None,
        "variant": variant,
        "gate_basis": basis,
        "label": EXECUTORCH_LABEL,
        "reason": _not_measured_reason(model, variant, rows),
    }
