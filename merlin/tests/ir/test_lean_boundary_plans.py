"""Authored preprocessing denominators refuse omitted or misattributed boundary occurrences."""

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from merlin.llvmlower import passes_xdsl
from merlin.verify.lean import check, handoff, plans, stages

SOURCE = "func.func @forward(%a: i32) -> i32 { return %a : i32 }"


def _selection(monkeypatch, selected):
    monkeypatch.setenv("MERLIN_PASSES", "layer-norm-chunked-sums" if selected else "-layer-norm-chunked-sums")
    return plans.preprocessing()


def _options():
    return handoff.Options("forward", 64, Path("unused-test-compiler"))


def _stub_checks(monkeypatch):
    """Exercise gate orchestration only; actual Lean chain coverage lives in test_lean_handoff."""
    calls = []

    def stage(path, **kwargs):
        calls.append(path.name)
        return {"verdict": "verified"}

    monkeypatch.setattr(check, "check_typed_stage", stage)
    for name in ("check_support", "check_definedness"):
        monkeypatch.setattr(check, name, lambda *a, **kw: {"verdict": "verified"})
    monkeypatch.setattr(
        check, "compose_chain", lambda *a, **kw: {"verdict": "verified", "domain": "test-only", "assumptions": []}
    )
    return calls


def _synthetic(plan, changes=None):
    def preprocess(source):
        occurrences = list(plan.boundaries)
        if changes:
            occurrences = changes(occurrences)
        for slot in occurrences:
            stages.record_boundary(slot.name, slot.pass_entry, source, source)
        return source, {}

    return preprocess


@pytest.mark.parametrize("selected", [False, True])
def test_actual_preprocessing_matches_authored_plan_without_changing_bytes(tmp_path, monkeypatch, selected):
    expected = _selection(monkeypatch, selected)
    plain = passes_xdsl.preprocess_text(SOURCE)
    calls = _stub_checks(monkeypatch)
    result = handoff.run(SOURCE, tmp_path / "upstream.mlir", _options(), passes_xdsl.preprocess_text, plan=expected)
    assert (result.text, result.stats) == plain
    assert len(calls) == (5 if selected else 4)
    report = json.loads(result.report.read_text())
    assert report["coverage_granularity"] == "individual-preprocessing-boundaries"
    assert report["expected_boundary_count"] == len(expected.boundaries)
    assert report["plan_sha256"] == expected.sha256
    assert Path(report["plan"]).read_bytes() == expected.canonical_bytes()
    plans.replay(expected, Path(report["plan"]), result.report.parent / "stages")
    assert [b.name for b in expected.boundaries] == [
        "xdsl-parse",
        "xdsl-pruned",
        "xdsl-quant-lowered",
        *(["xdsl-layer-norm-chunked-sums"] if selected else []),
        "xdsl-c-interface",
    ]


@pytest.mark.parametrize("fault", ["omitted-unchanged", "duplicate", "reordered", "unexpected", "wrong-pass"])
def test_roster_fault_refuses_before_any_semantic_checker(tmp_path, monkeypatch, fault):
    expected = _selection(monkeypatch, False)
    monkeypatch.setattr(check, "check_typed_stage", lambda *a, **kw: pytest.fail("bad roster reached checker"))

    def change(slots):
        match fault:
            case "omitted-unchanged":
                return slots[:1] + slots[2:]
            case "duplicate":
                return slots[:2] + slots[1:]
            case "reordered":
                return [slots[1], slots[0], *slots[2:]]
            case "unexpected":
                return [*slots, plans.Boundary("unexpected", "unreviewed.pass")]
            case "wrong-pass":
                return [slots[0], plans.Boundary(slots[1].name, "unreviewed.pass"), *slots[2:]]
        raise AssertionError(fault)

    with pytest.raises(handoff.VerificationError):
        handoff.run(SOURCE, tmp_path / "upstream.mlir", _options(), _synthetic(expected, change), plan=expected)
    report = json.loads(next(tmp_path.glob("lean-shared-passes/*/verification.json")).read_text())
    assert report["verdict"] == "abstracted"
    assert report["plan_sha256"] == expected.sha256


@pytest.mark.parametrize("initial", [False, True])
def test_changed_optional_selection_cannot_change_the_precomputed_denominator(tmp_path, monkeypatch, initial):
    expected = _selection(monkeypatch, initial)
    _selection(monkeypatch, not initial)
    monkeypatch.setattr(check, "check_typed_stage", lambda *a, **kw: pytest.fail("changed plan reached checker"))
    with pytest.raises(handoff.VerificationError, match="expected .* boundary occurrences"):
        handoff.run(SOURCE, tmp_path / "upstream.mlir", _options(), passes_xdsl.preprocess_text, plan=expected)


@pytest.mark.parametrize("fault", ["plan", "pass", "extra", "incomplete", "unpublished"])
def test_independent_roster_replay_rejects_changed_evidence(tmp_path, monkeypatch, fault):
    expected = _selection(monkeypatch, False)
    _stub_checks(monkeypatch)
    result = handoff.run(SOURCE, tmp_path / "upstream.mlir", _options(), _synthetic(expected), plan=expected)
    manifest = result.report.parent / "expected-plan.json"
    root = result.report.parent / "stages"
    match fault:
        case "plan":
            manifest.write_bytes(manifest.read_bytes() + b"\n")
        case "pass":
            meta = stages.completed(root)[1] / "meta.json"
            payload = json.loads(meta.read_text())
            payload["pass"] = "unreviewed.pass"
            meta.write_text(json.dumps(payload))
        case "extra":
            monkeypatch.setenv(stages.ENV, str(root))
            stages.record_boundary("extra", "unreviewed.pass", SOURCE, SOURCE)
        case "incomplete":
            (root / ".reserved" / "000004").mkdir()
        case "unpublished":
            (root / "0004_unpublished").mkdir()
    with pytest.raises(plans.PlanMismatch):
        plans.replay(expected, manifest, root)


@pytest.mark.parametrize("fault", ["plan", "roster"])
def test_gate_rechecks_roster_after_semantic_replay(tmp_path, monkeypatch, fault):
    expected = _selection(monkeypatch, False)
    _stub_checks(monkeypatch)
    compose = check.compose_chain

    def race(published, *args, **kwargs):
        result = compose(published, *args, **kwargs)
        if fault == "plan":
            manifest = published[0].parent.parent / "expected-plan.json"
            manifest.write_bytes(manifest.read_bytes() + b"\n")
        else:
            meta = published[0] / "meta.json"
            payload = json.loads(meta.read_text())
            payload["pass"] = "unreviewed.pass"
            meta.write_text(json.dumps(payload))
        return result

    monkeypatch.setattr(check, "compose_chain", race)
    with pytest.raises(handoff.VerificationError):
        handoff.run(SOURCE, tmp_path / "upstream.mlir", _options(), _synthetic(expected), plan=expected)


def test_unplanned_generic_capture_reports_only_recorded_boundary_coverage(tmp_path, monkeypatch):
    _stub_checks(monkeypatch)
    plan = plans.Plan("synthetic test", (plans.Boundary("test", "test.pass"),))
    result = handoff.run(SOURCE, tmp_path / "upstream.mlir", _options(), _synthetic(plan))
    report = json.loads(result.report.read_text())
    assert report["coverage_granularity"] == "recorded-boundary-only"
    assert "roster completeness not established" in report["reason"]
    assert "plan_sha256" not in report and "plan" not in report


def test_plan_is_immutable_and_hash_binds_order_and_pass_identity():
    a, b = plans.Boundary("a", "a.pass"), plans.Boundary("b", "b.pass")
    plan = plans.Plan("test", (a, b))
    with pytest.raises(FrozenInstanceError):
        plan.scope = "changed"
    with pytest.raises(FrozenInstanceError):
        a.pass_entry = "changed"
    with pytest.raises(ValueError, match="immutable"):
        plans.Plan("test", [a, b])
    assert len({plan.sha256, plans.Plan("test", (b, a)).sha256, plans.Plan("test", (a, a)).sha256}) == 3


def test_production_lowering_supplies_plan_before_preprocessing(tmp_path, monkeypatch):
    from merlin.llvmlower import lower

    _selection(monkeypatch, False)
    calls = []

    def stopped(source, output, options, preprocess, *, plan):
        calls.append(plan)
        assert plan == plans.preprocessing()
        raise handoff.VerificationError("stopped before preprocessing")

    monkeypatch.setattr(handoff, "run", stopped)
    monkeypatch.setattr(lower, "preprocess_text", lambda *a, **kw: pytest.fail("preprocessing ran"))
    monkeypatch.setattr(lower, "lower_to_llvm_ir", lambda *a, **kw: pytest.fail("downstream ran"))
    with pytest.raises(handoff.VerificationError, match="stopped before preprocessing"):
        lower.lower_model(SOURCE, tmp_path, targets=(), index_bits=64, shared_pass_verification=_options())
    assert len(calls) == 1


def test_textual_route_refuses_before_unplanned_transformations(tmp_path, monkeypatch):
    from merlin.llvmlower import lower

    monkeypatch.setattr(passes_xdsl, "preprocess_text_textual", lambda *a, **kw: pytest.fail("textual transform ran"))
    with pytest.raises(handoff.VerificationError, match="planned structural preprocessing"):
        lower.lower_model(
            SOURCE, tmp_path, targets=(), textual=True, index_bits=64, shared_pass_verification=_options()
        )


@pytest.mark.parametrize("index_bits", [32, 64])
@pytest.mark.parametrize("selected", [False, True])
def test_planned_preprocessing_replays_real_defined_chain(tmp_path, monkeypatch, lean_tools, index_bits, selected):
    expected = _selection(monkeypatch, selected)
    options = handoff.Options("forward", index_bits, lean_tools.m2m_python)
    result = handoff.run(SOURCE, tmp_path / "upstream.mlir", options, passes_xdsl.preprocess_text, plan=expected)
    report = json.loads(result.report.read_text())
    chain = json.loads(Path(report["chain"]).read_text())
    assert report["verdict"] == chain["verdict"] == "verified"
    assert chain["source_definedness"]["verdict"] == "verified"
    assert chain["replayed"] and chain["chained"]
    assert report["expected_boundary_count"] == len(report["boundaries"]) == (5 if selected else 4)
    plans.replay(expected, Path(report["plan"]), result.report.parent / "stages")


@pytest.mark.parametrize("duplicate", [False, True])
def test_selected_optional_occurrence_requires_one_authored_anchor(monkeypatch, duplicate):
    _selection(monkeypatch, True)
    anchor = next(slot for slot in passes_xdsl._PREPROCESS if slot[0] == "quant_ext_lowered")
    slots = tuple(slot for slot in passes_xdsl._PREPROCESS if slot[0] != "quant_ext_lowered")
    monkeypatch.setattr(passes_xdsl, "_PREPROCESS", (*slots, anchor, anchor) if duplicate else slots)
    with pytest.raises(ValueError, match="exactly one quantization"):
        plans.preprocessing()


@pytest.mark.parametrize("selected", [False, True])
@pytest.mark.parametrize("fault", ["delete", "swap", "duplicate", "statistic", "stage", "transform"])
def test_production_table_drift_cannot_author_its_own_coherent_denominator(tmp_path, monkeypatch, selected, fault):
    from merlin.llvmlower import lower

    expected = _selection(monkeypatch, selected)
    rows = list(passes_xdsl._PREPROCESS)
    statistic, stage, transform = rows[0]
    if fault == "delete":
        rows = rows[1:]
    elif fault == "swap":
        rows[:2] = reversed(rows[:2])
    elif fault == "duplicate":
        rows.insert(1, rows[0])
    elif fault == "statistic":
        rows[0] = ("different_statistic", stage, transform)
    elif fault == "stage":
        rows[0] = (statistic, "xdsl-renamed", transform)
    else:
        # A real, callable, harmless-on-this-input pass makes the records fully
        # coherent; replacing it with a nonexistent function would test a crash.
        rows[0] = (statistic, stage, "add_c_interface")
    monkeypatch.setattr(passes_xdsl, "_PREPROCESS", tuple(rows))

    root, output = tmp_path / "stages", tmp_path / "upstream.mlir"
    with stages.capture(root):
        text, _ = passes_xdsl.preprocess_text(SOURCE)
        output.write_text(text)
        stages.record_handoff(output)
    published = stages.completed(root)
    # Simulate coherently regenerated records AND expected manifest, the former
    # self-derived denominator. They agree perfectly but cannot authorize entry.
    recorded = [json.loads((path / "meta.json").read_text()) for path in published]
    coherent = plans.Plan(
        expected.scope,
        tuple(plans.Boundary(row["boundary"], row["pass"]) for row in recorded),
        expected.granularity,
    )
    manifest = tmp_path / "coherent-plan.json"
    manifest.write_bytes(coherent.canonical_bytes())
    plans.replay(coherent, manifest, root)
    with pytest.raises(plans.PlanMismatch, match="independently authored ordinary roster"):
        plans.preprocessing()

    monkeypatch.setattr(lower, "preprocess_text", lambda *a, **k: pytest.fail("changed roster reached passes"))
    monkeypatch.setattr(lower, "lower_to_llvm_ir", lambda *a, **k: pytest.fail("changed roster reached native"))
    monkeypatch.setattr(check, "check_typed_stage", lambda *a, **k: pytest.fail("changed roster reached checker"))
    with pytest.raises(plans.PlanMismatch, match="independently authored ordinary roster"):
        lower.lower_model(SOURCE, tmp_path / "lower", targets=(), index_bits=64, shared_pass_verification=_options())


@pytest.mark.parametrize("selected", [False, True])
def test_authored_roster_retains_existing_plan_bytes(monkeypatch, selected):
    expected = _selection(monkeypatch, selected)
    legacy = [plans.Boundary("xdsl-parse", "merlin.llvmlower.passes_xdsl.parse_mlir_text")]
    for statistic, stage, transform in passes_xdsl._PREPROCESS:
        legacy.append(plans.Boundary(stage, f"merlin.llvmlower.passes_xdsl.{transform}"))
        if selected and statistic == "quant_ext_lowered":
            legacy.append(
                plans.Boundary(
                    "xdsl-layer-norm-chunked-sums",
                    "merlin.llvmlower.normalization_reassociation.chunk_layer_norm_sums",
                )
            )
    assert (
        expected.canonical_bytes() == plans.Plan(expected.scope, tuple(legacy), expected.granularity).canonical_bytes()
    )
