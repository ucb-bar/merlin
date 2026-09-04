"""Board-free tests for the TRUE ExecuTorch whole-model COLUMN of the autonomous beam experiment.

Exercises merlin.compare.executorch_column (ingest + honest labeling) and the dtype-appropriate
reference logic in build_tools/scripts/run_autonomous_beam_experiment.py — the int8-reference-bug fix.
Named with "experiment"/"compare" so the pre-commit -k filter selects them. No board, no ET venv:
fixtures are hand-written BaselineResult JSONs (pass / not_run / int8 variant-mismatch).
"""
from __future__ import annotations

import importlib.util

from merlin.baselines.contract import BaselineResult
from merlin.common.paths import repo_root
from merlin.compare.executorch_column import (EXECUTORCH_LABEL, XNNPACK_KERNELS_LABEL,
                                              dtype_comparability, executorch_cell, gate_basis)


def _write_result(root, model, variant, *, built, ran, cos=None, rel=None, wall_ns=None,
                  cos_threshold=0.9999, rel_threshold=2e-3, gap_reason="", ts="20260101T000000Z"):
    """Write one baseline_result.json into the measurements tree the ingester reads."""
    res = BaselineResult(framework="executorch", model=model, variant=variant,
                         substrate="k1_spacemit", built=built, ran=ran, cos=cos, rel=rel,
                         e2e_wall_ns=wall_ns, cos_threshold=cos_threshold,
                         rel_threshold=rel_threshold, gap_reason=gap_reason, timestamp=ts)
    d = root / "out/artifacts/measurements/k1_spacemit" / model / f"{model}_{variant}_{ts}"
    d.mkdir(parents=True, exist_ok=True)
    res.write(d)
    return res


def _load_experiment_module():
    """Import the (non-package) experiment driver by path so we can test its reference logic."""
    p = repo_root() / "build_tools" / "scripts" / "run_autonomous_beam_experiment.py"
    spec = importlib.util.spec_from_file_location("_autobeam_under_test", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- executorch_cell ingest/labeling
def test_experiment_executorch_column_passing_cell_is_measured(tmp_path):
    # bitvla is random-init -> gate_basis must say lowering-exactness, not semantic.
    _write_result(tmp_path, "bitvla", "fp32", built=True, ran=True, cos=0.99999999,
                  rel=1.6e-6, wall_ns=146_000_000)
    cell = executorch_cell("bitvla", "fp32", root=tmp_path)
    assert cell["executorch_status"] == "measured"
    assert cell["executorch_wall_ns"] == 146_000_000.0
    assert cell["variant"] == "fp32"
    assert "lowering-exactness" in cell["gate_basis"]
    assert cell["label"] == EXECUTORCH_LABEL


def test_experiment_executorch_column_semantic_gate_label(tmp_path):
    # rdt2's captured golden IS reproducible -> semantic gate label (not lowering-exactness).
    _write_result(tmp_path, "rdt2", "fp32", built=True, ran=True, cos=0.99999,
                  rel=1e-4, wall_ns=855_000_000)
    cell = executorch_cell("rdt2", "fp32", root=tmp_path)
    assert cell["executorch_status"] == "measured"
    assert cell["gate_basis"].startswith("semantic")
    assert "lowering-exactness" not in cell["gate_basis"]


def test_experiment_executorch_column_not_run_is_not_measured(tmp_path):
    _write_result(tmp_path, "rdt2", "fp32", built=True, ran=False,
                  gap_reason="K1 board unavailable (MERLIN_K1_HOST unset)")
    cell = executorch_cell("rdt2", "fp32", root=tmp_path)
    assert cell["executorch_status"] == "not_measured"
    assert cell["executorch_wall_ns"] is None
    assert "not_run" in cell["reason"] and "board unavailable" in cell["reason"]


def test_experiment_executorch_column_fail_never_reported_as_number(tmp_path):
    # An executed-but-failed run (missed tolerance) is a gap, NEVER a wall number.
    _write_result(tmp_path, "rdt2", "fp32", built=True, ran=True, cos=0.5, rel=0.9,
                  wall_ns=999_000_000, gap_reason="")
    cell = executorch_cell("rdt2", "fp32", root=tmp_path)
    assert cell["executorch_status"] == "not_measured"
    assert cell["executorch_wall_ns"] is None
    assert "fail" in cell["reason"] and "cos=0.5" in cell["reason"]


def test_experiment_executorch_column_int8_never_borrows_fp32_number(tmp_path):
    # A passing fp32 exists but int8 was NOT run -> the int8 cell must be not_measured, never the
    # fp32 wall (variant-mismatch honesty).
    _write_result(tmp_path, "rdt2", "fp32", built=True, ran=True, cos=0.99999, rel=1e-4,
                  wall_ns=855_000_000)
    cell = executorch_cell("rdt2", "int8", root=tmp_path)
    assert cell["executorch_status"] == "not_measured"
    assert cell["executorch_wall_ns"] is None
    assert cell["variant"] == "int8"
    assert "int8" in cell["reason"]


def test_experiment_executorch_column_absent_tree_is_not_measured(tmp_path):
    cell = executorch_cell("bitvla", "fp32", root=tmp_path)
    assert cell["executorch_status"] == "not_measured"
    assert cell["executorch_wall_ns"] is None


def test_experiment_executorch_column_ram_infeasible_reason(tmp_path):
    # openvla has no ET result AND is RAM-infeasible -> reason must say so honestly.
    cell = executorch_cell("openvla", "fp32", root=tmp_path)
    assert cell["executorch_status"] == "not_measured"
    assert "RAM-infeasible" in cell["reason"]


def test_experiment_gate_basis_matches_random_init_set():
    assert "lowering-exactness" in gate_basis("bitvla")   # random-init
    assert gate_basis("rdt2").startswith("semantic")      # reproducible golden


# ---------------------------------------------------------------- int8 reference-bug fix
def test_experiment_reference_int8_uses_executorch_never_fp32_xnnpack(tmp_path):
    mod = _load_experiment_module()
    # A passing ExecuTorch int8 result exists -> it is the int8 reference (int8-vs-int8).
    _write_result(tmp_path, "rdt2", "int8", built=True, ran=True, cos=0.995, rel=1e-2,
                  wall_ns=700_000_000, cos_threshold=0.99, rel_threshold=5e-2)
    ref = mod._reference("rdt2", "int8", tmp_path)
    assert ref["kind"] == "executorch_external"
    assert ref["wall_ns"] == 700_000_000.0
    assert ref["dtype"] == "int8"
    assert ref["label"] == EXECUTORCH_LABEL


def test_experiment_reference_int8_without_et_is_none_never_fp32(tmp_path):
    mod = _load_experiment_module()
    # Only a passing fp32 ET result exists -> the int8 reference must be None, NEVER the fp32 wall.
    _write_result(tmp_path, "rdt2", "fp32", built=True, ran=True, cos=0.99999, rel=1e-4,
                  wall_ns=855_000_000)
    ref = mod._reference("rdt2", "int8", tmp_path)
    assert ref["kind"] is None
    assert ref["wall_ns"] is None
    assert "NEVER" in ref["note"] and "fp32 XNNPACK" in ref["note"]


def test_experiment_reference_fp32_is_xnnpack_in_runtime_label(tmp_path):
    mod = _load_experiment_module()
    ref = mod._reference("rdt2", "fp32", tmp_path)   # wall_ns may be None if no local four-way cache
    assert ref["kind"] == "xnnpack_kernels_in_runtime"
    assert ref["label"] == XNNPACK_KERNELS_LABEL
    assert ref["dtype"] == "fp32"


def test_experiment_comparability_labels_are_distinct():
    # The two arms MUST NOT read as the same thing.
    assert "OUR runtime" in XNNPACK_KERNELS_LABEL
    assert "TRUE external" in EXECUTORCH_LABEL
    assert XNNPACK_KERNELS_LABEL != EXECUTORCH_LABEL


def test_dtype_comparability_states_the_per_dtype_caveats():
    """Same storage dtype is necessary but NOT sufficient — the contract must SAY what differs."""
    fp32, int8, fp16 = (dtype_comparability(d) for d in ("fp32", "int8", "fp16"))
    # fp32 is the only like-for-like row (storage + accumulate both f32).
    assert "like-for-like" in fp32 and "f32" in fp32
    # int8: no in-runtime kernel-swap arm; external ref is ExecuTorch whole-system.
    assert "NO in-runtime" in int8 and "ExecuTorch" in int8
    # fp16: same storage, DIFFERENT accumulate — must be flagged, never presented as a clean match.
    assert "ACCUMULATE-ASYMMETRIC" in fp16 and "vfwmacc" in fp16
    assert "not a like-for-like" in fp16 and "caveated" in fp16   # explicitly negated, never clean
    # an unknown dtype fails loud rather than implying a match.
    assert "UNKNOWN" in dtype_comparability("fp8")
    # every measured/not-measured cell carries the field.
    for st in (executorch_cell("bitvla", "int8"), executorch_cell("bitvla", "fp16")):
        assert "dtype_comparability" in st


# ---------------------------------------------------------------------------------------
# Bundle identity. (model, dtype) does NOT determine the comparand: `bundle.resolve()`
# prefers `<model>_<variant>_full` (real/native architecture) over the older TRUNCATED
# `_consistent` bundle when both exist, so two runs of the "same" cell can be two
# different models. A beam wall recorded on rdt2_int8_consistent was divided by an
# ExecuTorch reference exported at native depth and the ratio was quoted as a headline.
# The direction survived (ours ran the SMALLER model and was still slower, so the true
# gap is worse) but the number was not citable -- and nothing in the code could say so.
# ---------------------------------------------------------------------------------------

def test_a_bundle_mismatch_is_refused_rather_than_ratioed():
    from merlin.compare.executorch_column import bundle_mismatch_reason

    assert bundle_mismatch_reason("rdt2_int8_full", "rdt2_int8_full") is None
    why = bundle_mismatch_reason("rdt2_int8_consistent", "rdt2_int8_full")
    assert why and "MISMATCH" in why
    assert "rdt2_int8_consistent" in why and "rdt2_int8_full" in why
    assert "not a speedup" in why          # says what the consequence IS


def test_an_unrecorded_bundle_identity_is_refused_not_assumed_to_match():
    """The recurring failure in this repo is a check that could not run reporting success. An empty
    bundle_id means the producer did not record what it measured; that is UNKNOWN, not a match."""
    from merlin.compare.executorch_column import bundle_mismatch_reason

    for ours, ref in (("", "rdt2_int8_full"), ("rdt2_int8_full", ""), ("", "")):
        why = bundle_mismatch_reason(ours, ref)
        assert why and "UNKNOWN" in why, (ours, ref, why)


def test_the_cell_yields_not_measured_with_a_reason_instead_of_a_number():
    """The plan's gate: a mismatch must produce status='not_measured' and a concrete reason, never a
    wall a caller could divide by."""
    cell = executorch_cell("bitvla", "int8", ours_bundle_id="bitvla_int8_consistent")
    assert cell["executorch_status"] == "not_measured"
    assert cell["executorch_wall_ns"] is None
    assert cell["reason"]
    assert "bundle" in cell["reason"].lower()
    # the identities are surfaced either way, so a reader can check by eye
    assert "ref_bundle_id" in cell and "ours_bundle_id" in cell


def test_omitting_ours_bundle_id_keeps_the_display_only_behaviour():
    """Callers that only DISPLAY the ExecuTorch column and never form a ratio must not be broken by the
    guard -- but the cell still reports the reference's bundle identity."""
    cell = executorch_cell("bitvla", "int8")
    assert cell["ours_bundle_id"] is None
    assert "ref_bundle_id" in cell


def test_the_measurement_record_carries_the_bundle_it_was_taken_on():
    """The guard is only as good as the field it reads: BaselineResult must carry bundle_id, and it
    must survive a JSON round-trip (old records without it load as UNKNOWN, not as a match)."""
    import json
    from pathlib import Path
    from tempfile import mkdtemp

    from merlin.baselines.contract import BaselineResult

    r = BaselineResult(framework="executorch", model="rdt2", variant="int8",
                       built=True, ran=True, cos=1.0, e2e_wall_ns=123,
                       bundle_id="rdt2_int8_full")
    d = Path(mkdtemp()) / "baseline_result.json"
    d.write_text(json.dumps({k: v for k, v in r.__dict__.items()
                             if k not in ("regions", "scalar_fallbacks")}
                            | {"regions": [], "scalar_fallbacks": []}))
    assert BaselineResult.load(d).bundle_id == "rdt2_int8_full"
    # a record predating the field loads with an EMPTY id, which the guard treats as UNKNOWN
    raw = json.loads(d.read_text()); raw.pop("bundle_id")
    d.write_text(json.dumps(raw))
    assert BaselineResult.load(d).bundle_id == ""


def test_an_unrecorded_quant_recipe_is_refused_like_an_unrecorded_bundle():
    """dtype does not determine the arithmetic.

    `variant="int8"` has meant three different computations here, and the DEFAULT one --
    `weight_only` -- is not int8 compute at all: its dequant const-folds into an fp32 const weight
    that XNNPACK partitions as a normal fp32 GEMM, so the cell measures fp32 math with int8 storage
    and never reaches an int8 ukernel. Every cached int8 ratio in this repo was taken against that,
    unlabelled. So an unrecorded recipe must be refused exactly as firmly as an unrecorded bundle.
    """
    from merlin.compare.executorch_column import quant_recipe_mismatch_reason as why

    assert why("", "pt2e_qd8"), "unknown on our side must refuse"
    assert why("pt2e_qd8", ""), "unknown on the reference side must refuse"
    assert "UNKNOWN" in why("", "")
    assert why("pt2e_qd8", "pt2e_qd8") is None, "same recipe is comparable"


def test_a_recipe_mismatch_names_both_computations():
    """A refusal has to be actionable: say what each side actually ran, not just that they differ."""
    from merlin.compare.executorch_column import quant_recipe_mismatch_reason as why

    r = why("pt2e_qd8", "weight_only")
    assert r is not None
    assert "pt2e_qd8" in r and "weight_only" in r
    assert "fp32 GEMM" in r, "the reader must learn that weight_only is not int8 compute"


def test_the_result_contract_carries_recipe_protocol_and_conditions():
    """These three are what make a cell comparable at all; defaults must be UNKNOWN, never a guess."""
    from merlin.baselines.contract import BaselineResult

    r = BaselineResult(framework="executorch", model="m", variant="int8")
    assert r.quant_recipe == ""
    assert r.num_executions is None
    assert r.board_conditions is None
    assert "quant_recipe" in r.to_dict()


def test_our_recipe_is_never_derived_from_what_the_reference_ran():
    """The guard must compare OUR recipe, not a copy of the comparand's.

    The fair-compare harness once passed ``"pt2e_qd8" if arm is qd8 else "merlin_int8_w8a8"`` as
    ours, so on the qd8 arm the check compared the reference against itself and could not fail on
    any input -- while the weight-only arm, fed our real name, refused correctly. An inert guard is
    worse than none: it prints a verdict that reads as if it had been checked. Pin the constant.
    """
    src = (repo_root() / "build_tools" / "scripts" / "k1_int8_fair_compare.py").read_text()
    assert 'OURS_QUANT_RECIPE = "merlin_int8_w8a8"' in src
    assert "quant_recipe_mismatch_reason(OURS_QUANT_RECIPE, ref_recipe)" in src
    assert 'recipe_requested"] == "pt2e_qd8"' not in src, (
        "our recipe is being selected from the reference arm again")


def test_the_qd8_equivalence_is_declared_and_everything_else_still_refuses():
    """merlin_int8_w8a8 IS the qd8 arithmetic -- by declaration, with the residual stated."""
    from merlin.compare.executorch_column import (QUANT_RECIPE_EQUIVALENT,
                                                  quant_recipe_mismatch_reason)
    assert frozenset({"merlin_int8_w8a8", "pt2e_qd8"}) in QUANT_RECIPE_EQUIVALENT
    # comparable, in both directions
    assert quant_recipe_mismatch_reason("merlin_int8_w8a8", "pt2e_qd8") is None
    assert quant_recipe_mismatch_reason("pt2e_qd8", "merlin_int8_w8a8") is None
    # the equivalence is NOT a blanket pass: the other two recipes are still different computations
    for other in ("weight_only", "pt2e_qs8"):
        why = quant_recipe_mismatch_reason("merlin_int8_w8a8", other)
        assert why and "MISMATCH" in why, other
        # and our own recipe must be NAMED in the refusal, never rendered as 'unknown recipe'
        assert "unknown recipe" not in why, why
        assert "merlin_int8_w8a8" in why


def test_an_unknown_recipe_is_still_refused_even_against_an_equivalent_one():
    """The equivalence must not resurrect the UNKNOWN case: empty is refused, as before."""
    from merlin.compare.executorch_column import quant_recipe_mismatch_reason
    assert "UNKNOWN" in (quant_recipe_mismatch_reason("", "pt2e_qd8") or "")
    assert "UNKNOWN" in (quant_recipe_mismatch_reason("merlin_int8_w8a8", "") or "")


def test_accuracy_scored_against_different_references_is_refused_not_ranked():
    """int8-vs-host-int8 and int8-vs-fp32 answer different questions.

    merlin's gate reports the W8A8 tier's score (does our int8 arithmetic match a correct int8
    computation); ExecuTorch's int8 path forces compute_golden and scores int8 against fp32, which
    includes the quantization error itself. Ranking 0.0148 against 0.0076 across that boundary
    produced a false 'ExecuTorch is more accurate than us' in this workstream. Refuse instead.
    """
    from merlin.compare.executorch_column import accuracy_reference_mismatch_reason as why
    m = why("capture_golden_w8a8", "recomputed_fp32")
    assert m and "MISMATCH" in m
    assert "cannot be ordered" in m
    # the same reference on both sides IS comparable
    assert why("capture_golden_fp32", "capture_golden_fp32") is None
    # and UNKNOWN is refused as firmly as a mismatch, like the other two guards
    for pair in (("", "recomputed_fp32"), ("capture_golden_fp32", "")):
        assert "UNKNOWN" in (why(*pair) or "")


def test_the_harness_keeps_every_tier_score_not_the_collapsed_pair():
    """_gate collapses out['rel'] to the W8A8 tier when present; fp32_rel must survive to the
    artifact, since it is the only number comparable with an fp32-scored reference."""
    src = (repo_root() / "build_tools" / "scripts" / "k1_int8_fair_compare.py").read_text()
    assert "accuracy_reference_by_tier" in src
    assert 'OURS_ACCURACY_REFERENCE = "capture_golden_fp32"' in src
    assert "accuracy_reference_mismatch_reason(OURS_ACCURACY_REFERENCE, ref_acc)" in src
    # load_ns must be carried: it decides whether our offline weight-transpose hoisting mirrors
    # XNNPACK's delegate-init prepacking or is an advantage we granted ourselves.
    assert '"load_ns": getattr(r, "load_ns", None)' in src
    assert '"executorch_load_ns": load' in src
