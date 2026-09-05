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

import json

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

# --- per-dtype comparability: same storage dtype is NECESSARY but not SUFFICIENT for a fair compare.
# Each cell carries this so the report states, per dtype, exactly what is matched and what is NOT.
_DTYPE_COMPARABILITY = {
    "fp32": ("fp32-vs-fp32: storage AND accumulate both f32 on both sides — a direct wall + cos "
             "compare is like-for-like. External arms available: xnnpack_kernels (in-runtime GEMM "
             "swap) AND executorch (whole-system)."),
    "int8": ("int8-vs-int8 (W8A8): storage matched; quantization is applied in model2MLIR (NOT "
             "Merlin) and BOTH sides consume the same int8 export + integer golden (gate tier T1, "
             "golden_w8a8.npy). BUT there is NO in-runtime XNNPACK-kernel-swap arm for int8 (the "
             "four-way is f32-only — k1.py rejects int8 for xnnpack/openblas backends), so the only "
             "external int8 reference is ExecuTorch(+XNNPACK qd8) as a WHOLE SYSTEM, not a clean "
             "single-kernel head-to-head."),
    "fp16": ("fp16-vs-fp16: storage matched (f16) but ACCUMULATE-ASYMMETRIC — XNNPACK f16 GEMM "
             "accumulates in f16, ours emits vfwmacc.vf (f32 accumulate). Same nominal dtype, "
             "DIFFERENT compute precision: a raw cos/speed compare is caveated, not a like-for-like "
             "numeric match. (No fp16 external result measured yet either.)"),
}


def dtype_comparability(dtype: str) -> str:
    """What a same-dtype comparison for this dtype actually means — matched axes AND the caveats.
    Same storage dtype is necessary but not sufficient (fp16 accumulate asymmetry; int8 has no
    in-runtime kernel-swap arm). Unknown dtype fails loud rather than implying a clean match."""
    return _DTYPE_COMPARABILITY.get(
        dtype, f"UNKNOWN dtype {dtype!r}: no comparability contract — do NOT assume a like-for-like match")


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


def _result_caveats(model: str) -> str:
    """Why a RECORDED ExecuTorch fail/number for this model may not mean what it looks like. A low cos
    on a RAM-infeasible or random-init model is an artifact of the run conditions, NOT evidence of an
    ExecuTorch kernel defect -- surfacing this stops a false 'ET is numerically wrong' read."""
    notes = []
    if model in _bundle.K1_RAM_INFEASIBLE:
        notes.append("model is K1 RAM-infeasible whole-model (7B-class VLA vs the 3.8GB board), so any "
                     "whole-model wall/cos is from a RAM-constrained or truncated run, not trustworthy")
    if _bundle.golden_unreproducible(model):
        notes.append("random-init model: a semantic cos vs the captured golden is meaningless (the gate "
                     "should be lowering-exactness); a low cos here is a golden-provenance artifact, "
                     "NOT an ET numerical defect")
    return (" [CAVEAT: " + "; ".join(notes) + "]") if notes else ""


def _not_measured_reason(model: str, variant: str, rows: list) -> str:
    """Honest reason a cell has no ExecuTorch number: an executed-but-failed run, a not_run, or absence."""
    if rows:
        r = rows[0]  # dedupe_latest yields exactly one row per (framework, model, variant)
        gap = (r.gap_reason or "").strip()
        detail = f": {gap}" if gap else ""
        if r.status() == "fail" and r.cos is not None:
            # Name the bar it missed and BY WHICH TERM. "fail (cos=0.994)" reads as a broken
            # reference; "cos passed, rel 0.106 against its own 0.05 int8 bar" is a different fact
            # and a reader has to be able to tell them apart, because the bias here runs one way:
            # a reference refused on accuracy produces no row, so a cell we would have LOST is
            # simply absent. That is not a neutral outcome and it must not read as one.
            terms = []
            if r.cos_threshold is not None and r.cos is not None:
                terms.append(f"cos {r.cos:.6g} {'>=' if r.cos >= r.cos_threshold else '<'} "
                             f"{r.cos_threshold:g}")
            if r.rel_threshold is not None and r.rel is not None:
                terms.append(f"rel {r.rel:.6g} {'<=' if r.rel <= r.rel_threshold else '>'} "
                             f"{r.rel_threshold:g}")
            if terms:
                detail += " (" + ", ".join(terms) + ")"
            return (f"latest executorch {variant} = fail against ITS OWN declared bar{detail}"
                    f"{_result_caveats(model)}. No ratio is published for this cell — note that "
                    "this suppresses a LOSS as readily as a win, so read the absence as missing "
                    "evidence, not as parity.")
        return f"latest executorch {variant} = {r.status()}{detail}{_result_caveats(model)}"
    if model in _bundle.K1_RAM_INFEASIBLE:
        return (f"no executorch {variant} result; {model} is K1 RAM-infeasible whole-model (7B-class "
                "VLA exceeds the 3.8 GB board) -> honest not_run, never a false fit")
    return f"no executorch {variant} result under {_BASELINE_TREE} (board run not yet performed)"


def bundle_mismatch_reason(ours_bundle_id: str, ref_bundle_id: str) -> str | None:
    """Why these two measurements are not comparable, or None if they are.

    (model, dtype) is NOT a sufficient key. ``bundle.resolve()`` prefers ``<model>_<variant>_full``
    (the real/native architecture) over the older TRUNCATED ``_consistent`` bundle when both exist, so
    two runs of the "same" cell can be two different models. That is not hypothetical: a beam wall
    recorded on ``rdt2_int8_consistent`` was divided by an ExecuTorch reference exported at native
    depth, and the ratio was quoted as a headline. The direction happened to survive (ours ran the
    SMALLER model and was still slower, so the true gap is worse), but the number was not citable, and
    nothing in the code could have said so.

    UNKNOWN is refused as firmly as a mismatch. An empty ``bundle_id`` means the producer did not
    record which bundle it measured -- treating that as "presumably the same" is precisely how the bad
    ratio got published, and it is the failure mode this repo keeps hitting: a check that could not run
    reporting success.
    """
    if not ours_bundle_id or not ref_bundle_id:
        missing = [n for n, v in (("ours", ours_bundle_id), ("reference", ref_bundle_id)) if not v]
        return (f"bundle identity UNKNOWN for {' and '.join(missing)}: cannot verify the two "
                "measurements are on the same capture, and (model, dtype) does not determine it "
                "(_full is a different model from _consistent, not a smaller capture of it). "
                "Re-measure with bundle_id recorded on both sides.")
    if ours_bundle_id != ref_bundle_id:
        eq = layout_equivalence(ours_bundle_id, ref_bundle_id)
        if eq is not None:
            return None          # declared, and the caller must record `eq` in the artifact
        return (f"bundle MISMATCH: ours measured on {ours_bundle_id!r}, the reference on "
                f"{ref_bundle_id!r}. These are different models, not different runs of one -- a ratio "
                "between them is not a speedup. Re-measure both on one bundle.")
    return None


def layout_equivalence(ours_bundle_id: str, ref_bundle_id: str) -> dict | None:
    """Is ``ours`` a LAYOUT-ONLY derivative of ``ref``? The record, or None.

    Two bundles differing only in how the same weights are STORED are the same model, and comparing
    across them is legitimate ONLY because both sides do their layout work once, outside the window
    being timed: XNNPACK prepacks into its blocked layout at delegate init (measured here as a 26 ms
    ``load_ns``, 13x a warm inference) and we pre-apply the transposed layout at build time. Timing
    ours WITH the transposes while theirs are hoisted out of the measured region compares the
    accounting, not the compilers -- which is the same error as timing a warm loop against a
    cold-amortized average.
    
    DERIVED FROM THE ARTIFACT, never a name rule: the derived bundle ships
    ``bundle.rewrites.json`` naming its ``source_bundle`` and the per-argument soundness argument
    (each hoisted transpose is its argument's SOLE consumer, and the stored bytes are asserted equal
    to ``stored.T``). A bundle that carries no such record, or whose record names a different
    source, is NOT equivalent and the mismatch stands. The byte count is unchanged by the rewrite,
    so offsets and the arg table are untouched.
    """
    from merlin.common.artifacts import recaptures_dir

    rec = recaptures_dir() / str(ours_bundle_id) / "bundle.rewrites.json"
    if not rec.is_file():
        return None
    try:
        rewrites = (json.loads(rec.read_text(encoding="utf-8")) or {}).get("rewrites") or []
    except (OSError, ValueError):
        return None              # unreadable provenance is not a licence to compare
    for r in rewrites:
        if not isinstance(r, dict):
            continue
        if str(r.get("source_bundle") or "") != str(ref_bundle_id):
            continue
        if str(r.get("name") or "") not in _LAYOUT_ONLY_REWRITES:
            continue
        return {"kind": "layout_only", "rewrite": r.get("name"),
                "source_bundle": r.get("source_bundle"),
                "soundness": r.get("soundness"),
                "note": ("comparable ONLY because both sides do their weight layout once outside "
                         "the timed window -- ExecuTorch at delegate init, ours at build time. "
                         "Record both one-time costs beside any ratio taken across this.")}
    return None


#: Rewrites that change only how weights are STORED, never what is computed. Named explicitly: a
#: rewrite that changes the graph (layer truncation, a fused subgraph, a different quantization) is
#: a different model and must keep failing the bundle check.
_LAYOUT_ONLY_REWRITES: frozenset = frozenset({"hoist_weight_transposes"})


#: The int8 recipes, and which of them are the same ARITHMETIC. `weight_only` is the odd one out and
#: the trap: its dequant const-folds into an fp32 const weight that XNNPACK partitions as a normal
#: fp32 GEMM, so a cell labelled int8 measures fp32 compute with int8 STORAGE and never reaches an
#: int8 ukernel. `pt2e_qd8` is the one that mirrors merlin's own datapath (dynamic per-row activation
#: quant against per-channel weights).
QUANT_RECIPE_LABELS = {
    "weight_only": "weight-only int8 storage; compute is a normal fp32 GEMM (no int8 ukernel)",
    "pt2e_qs8": "PT2E static per-tensor activation quant (XNNPACK qs8)",
    "pt2e_qd8": "PT2E per-channel weights + dynamic per-row activation quant (XNNPACK qd8)",
    "merlin_int8_w8a8": "merlin passes_quant_int: per-channel weight scales + dynamic per-row "
                        "activation quant to i8, symmetric (the qd8 arithmetic)",
}

#: Recipe names that denote THE SAME ARITHMETIC and may therefore be compared.
#:
#: Declared here, once, with its justification -- never inferred at a call site. A caller that
#: derives "ours" from what the REFERENCE happened to run compares the reference against itself,
#: and the guard can never fire; that is the same inert-guard shape as a bundle check whose
#: ``ours_bundle_id`` defaults to None, and it shipped here first.
#:
#: merlin's ``llvmlower/passes_quant_int`` dynamically quantizes each activation to i8, symmetric,
#: PER OUTPUT ROW, against per-channel weight scales -- which is what XNNPACK calls qd8.
#:
#: Residual the equivalence deliberately does NOT hide: merlin additionally runs I-BERT integer
#: softmax/GELU where ExecuTorch keeps those fp32, so the two are matched on the GEMM and merlin is
#: MORE integer elsewhere. State that beside any ratio taken across this equivalence.
QUANT_RECIPE_EQUIVALENT: tuple[frozenset[str], ...] = (
    frozenset({"merlin_int8_w8a8", "pt2e_qd8"}),
)


#: How an accuracy score identifies WHAT IT WAS SCORED AGAINST. A cos/rel is meaningless without
#: it: merlin's gate reports the W8A8 tier's score (int8 output vs a HOST INT8 reference -- "is our
#: int8 arithmetic right"), while ExecuTorch's int8 path forces ``compute_golden`` and scores against
#: an fp32 reference RECOMPUTED from the model it loaded ("how far is int8 from fp32", which includes
#: the quantization error itself). Those answer different questions and cannot be ordered. Ranking
#: them anyway produced a false "ExecuTorch is more accurate than us" inside this very workstream,
#: an hour after the same error class was fixed on the recipe axis.
ACCURACY_REFERENCE_LABELS = {
    "capture_golden_fp32": "the capture bundle's golden.npy (fp32)",
    "capture_golden_w8a8": "the capture bundle's golden_w8a8.npy (host int8 reference)",
    "recomputed_fp32": "an fp32 reference recomputed from the model the runner loaded",
}


def accuracy_reference_mismatch_reason(ours_ref: str, ref_ref: str) -> str | None:
    """Why these two accuracy scores may not be compared, or None if they may.

    Third guard of the same fail-closed shape as :func:`bundle_mismatch_reason` and
    :func:`quant_recipe_mismatch_reason`, on the reference axis. UNKNOWN is refused as firmly as a
    mismatch -- a score whose reference was not recorded cannot be shown to match.
    """
    if not ours_ref or not ref_ref:
        missing = [n for n, v in (("ours", ours_ref), ("reference", ref_ref)) if not v]
        return (f"accuracy reference UNKNOWN for {' and '.join(missing)}: a cos/rel does not say "
                "what it was scored against, and int8-vs-host-int8 and int8-vs-fp32 are different "
                "questions (the latter includes quantization error). Record the reference on both "
                "sides and re-score.")
    if ours_ref != ref_ref:
        return (f"accuracy reference MISMATCH: ours scored against {ours_ref!r} "
                f"({ACCURACY_REFERENCE_LABELS.get(ours_ref, 'unknown reference')}), the reference "
                f"against {ref_ref!r} ({ACCURACY_REFERENCE_LABELS.get(ref_ref, 'unknown reference')}). "
                "These measure different things and cannot be ordered -- neither side is 'more "
                "accurate' than the other on these numbers.")
    return None


def quant_recipe_mismatch_reason(ours_recipe: str, ref_recipe: str) -> str | None:
    """Why these two int8 measurements are not comparable ARITHMETIC, or None if they are.

    Same shape and same rationale as :func:`bundle_mismatch_reason`, on the recipe axis. dtype is not
    a sufficient key either: ``variant="int8"`` has meant three different computations in this repo,
    and the default one is not int8 compute at all. Every ours-vs-ExecuTorch int8 ratio taken before
    this field existed was against an unlabelled recipe, so UNKNOWN is refused as firmly as a
    mismatch -- an unrecorded recipe cannot be shown to match, which is exactly why it cannot be
    cited either.
    """
    if not ours_recipe or not ref_recipe:
        missing = [n for n, v in (("ours", ours_recipe), ("reference", ref_recipe)) if not v]
        return (f"quantization recipe UNKNOWN for {' and '.join(missing)}: dtype does not determine "
                "it (weight_only / pt2e_qs8 / pt2e_qd8 are three different computations, and "
                "weight_only is not int8 compute at all -- its dequant const-folds to an fp32 GEMM). "
                "Re-measure with quant_recipe recorded on both sides.")
    same = ours_recipe == ref_recipe or any(
        {ours_recipe, ref_recipe} <= fam for fam in QUANT_RECIPE_EQUIVALENT)
    if not same:
        return (f"quantization recipe MISMATCH: ours ran {ours_recipe!r} "
                f"({QUANT_RECIPE_LABELS.get(ours_recipe, 'unknown recipe')}), the reference ran "
                f"{ref_recipe!r} ({QUANT_RECIPE_LABELS.get(ref_recipe, 'unknown recipe')}). A ratio "
                "between two different quantization schemes measures the schemes, not the compilers.")
    return None


def executorch_cell(model: str, dtype: str, *, root: Path | None = None,
                    ours_bundle_id: str | None = None) -> dict:
    """The ExecuTorch column for a ``(model, dtype)`` cell — structured + honestly labeled.

    Returns a dict always carrying ``executorch_status`` (``measured`` | ``not_measured``),
    ``executorch_wall_ns`` (a real number ONLY when a passing ExecuTorch run of the SAME variant
    exists, else ``None``), the ``variant`` selected, the per-model ``gate_basis`` honesty label, the
    comparability ``label``, and — when not measured — a concrete ``reason``.

    ``ours_bundle_id`` is the capture bundle OUR side was measured on. Pass it whenever this cell will
    be divided by an ours-number: the reference's own ``bundle_id`` is compared against it and a
    mismatch (or an unrecorded identity on either side) yields ``not_measured`` with a concrete reason
    INSTEAD OF A NUMBER. Omitting it keeps the historical behavior for callers that only display the
    ExecuTorch column and never form a ratio; the cell still reports ``ref_bundle_id`` so a reader can
    check by eye.
    """
    root = root or repo_root()
    variant = _DTYPE_TO_VARIANT.get(dtype, dtype)
    rows = [r for r in _collect(root)
            if r.framework == "executorch" and r.model == model and r.variant == variant]
    passing = [r for r in rows if r.status() == "pass" and r.e2e_wall_ns]
    basis = gate_basis(model)
    if passing:
        r = passing[0]
        ref_bundle = getattr(r, "bundle_id", "") or ""
        mismatch = (bundle_mismatch_reason(ours_bundle_id, ref_bundle)
                    if ours_bundle_id is not None else None)
        if mismatch is not None:
            return {
                "executorch_status": "not_measured",
                "executorch_wall_ns": None,
                "variant": variant,
                "ref_bundle_id": ref_bundle,
                "ours_bundle_id": ours_bundle_id,
                "gate_basis": basis,
                "label": EXECUTORCH_LABEL,
                "dtype_comparability": dtype_comparability(dtype),
                "reason": mismatch,
            }
        return {
            "executorch_status": "measured",
            "executorch_wall_ns": float(r.e2e_wall_ns),
            "variant": r.variant,
            "ref_bundle_id": ref_bundle,
            "ours_bundle_id": ours_bundle_id,
            "cos": r.cos,
            "rel": r.rel,
            "rvv_coverage_overall": r.rvv_coverage_overall,
            "gate_basis": basis,
            "label": EXECUTORCH_LABEL,
            "dtype_comparability": dtype_comparability(dtype),
            "source": f"{_BASELINE_TREE} (baselines.aggregate)",
        }
    return {
        "executorch_status": "not_measured",
        "executorch_wall_ns": None,
        "variant": variant,
        "ref_bundle_id": "",
        "ours_bundle_id": ours_bundle_id,
        "gate_basis": basis,
        "label": EXECUTORCH_LABEL,
        "dtype_comparability": dtype_comparability(dtype),
        "reason": _not_measured_reason(model, variant, rows),
        # The bar the reference was held to, recorded so a reader can check it against the bar OUR
        # arm was held to. The two gates have different shapes (ours adds argmax and a per-element
        # term and carries no aggregate rel bound at the fp32 tier), so "both passed" never means
        # "both cleared the same test" and the row must not imply it.
        "ref_accuracy_bar": ({"cos_threshold": rows[0].cos_threshold,
                              "rel_threshold": rows[0].rel_threshold,
                              "cos": rows[0].cos, "rel": rows[0].rel} if rows else None),
    }
