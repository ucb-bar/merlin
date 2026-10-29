"""Unverified preparation captures compose only with exact original-source strict evidence."""

import json
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest
from test_lean_boundary_plans import SOURCE, _stub_checks

from merlin.runtime.backends import zephyr_model
from merlin.verify.lean import check, handoff, plans, preparation, stages


@pytest.fixture(autouse=True)
def _ordinary_selection(monkeypatch):
    monkeypatch.delenv("MERLIN_PASSES", raising=False)
    monkeypatch.delenv("MERLIN_VEC_RANK", raising=False)
    from merlin.llvmlower.optional_passes import REGISTRY

    for entry in REGISTRY:
        if entry.switch:
            monkeypatch.delenv(entry.switch, raising=False)


def _options(compiler=Path("unused"), bits=64):
    return handoff.Options("forward", bits, compiler)


def _capture(tmp_path, options=None, source=SOURCE):
    path = tmp_path / "model.mlir"
    path.write_text(source)
    return preparation.capture(path, tmp_path / "prepare", options or _options())


def test_actual_preparation_capture_preserves_bytes_resources_and_tuple_api(tmp_path):
    path = tmp_path / "model.mlir"
    path.write_text(SOURCE)
    plain_work = tmp_path / "plain"
    plain_work.mkdir()
    plain, features = zephyr_model.prepare_for_lowering(path, plain_work)
    captured = preparation.capture(path, tmp_path / "captured", _options())
    assert path.read_text() == SOURCE
    assert captured.target == plain.read_bytes() == captured.output.read_bytes()
    assert captured.features == features == frozenset()
    assert set(p.name for p in plain_work.iterdir()) == {p.name for p in captured.output.parent.iterdir()} - {
        "lean-preparation"
    }
    published, _ = preparation.validate(captured, _options(), captured.target)
    assert len(published) == 13
    assert json.loads((captured.root.parent / "capture.json").read_text())["status"] == "captured_unverified"
    with pytest.raises(FrozenInstanceError):
        captured.target = b"changed"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"int8_compute": True},
        {"quant_passes": []},
        {"features": frozenset({"unreviewed-feature"})},
        {"matrix": object()},
        {"device": object()},
        {"outline_int8": True},
        {"prepared_model_transform": lambda *a: None},
    ],
)
def test_unsupported_selected_configuration_refuses_before_preparation(tmp_path, monkeypatch, kwargs):
    monkeypatch.setattr(zephyr_model, "prepare_for_lowering", lambda *a, **kw: pytest.fail("preparation entered"))
    with pytest.raises((ValueError, KeyError)):
        preparation.capture(tmp_path / "absent.mlir", tmp_path / "work", _options(), **kwargs)
    assert not (tmp_path / "work").exists()


@pytest.mark.parametrize(
    "name,value",
    [
        ("MERLIN_VEC_RANK", "1"),
        ("MERLIN_PASSES", "int-softmax-table"),
        ("MERLIN_PASSES", "int-layernorm"),
        ("MERLIN_PASSES", "layer-norm-chunked-sums"),
        ("MERLIN_LAYER_NORM_CHUNKED_SUMS", "1"),
    ],
)
def test_effective_environment_selection_refuses_before_preparation(tmp_path, monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    monkeypatch.setattr(zephyr_model, "prepare_for_lowering", lambda *a, **kw: pytest.fail("preparation entered"))
    with pytest.raises(ValueError):
        preparation.capture(tmp_path / "absent.mlir", tmp_path / "work", _options())
    assert not (tmp_path / "work").exists()


def test_two_phase_gate_checks_original_source_and_one_concatenated_chain(tmp_path, monkeypatch):
    captured = _capture(tmp_path)
    calls = _stub_checks(monkeypatch)
    support_sources = []
    chain_inputs = []
    monkeypatch.setattr(
        check,
        "check_support",
        lambda source, *a, **kw: support_sources.append(source.read_bytes()) or {"verdict": "verified"},
    )
    compose = check.compose_chain

    def chain(published, *a, **kw):
        chain_inputs.append(list(published))
        return compose(published, *a, **kw)

    monkeypatch.setattr(check, "compose_chain", chain)
    from merlin.llvmlower.passes_xdsl import preprocess_text

    result = handoff.run(
        captured.target.decode(),
        tmp_path / "upstream.mlir",
        _options(),
        preprocess_text,
        plan=plans.preprocessing(),
        prefix=captured,
    )
    assert support_sources == [captured.source]
    assert len(calls) == len(chain_inputs[0]) == 17
    assert chain_inputs[0][0] == stages.completed(captured.root)[0]
    report = json.loads(result.report.read_text())
    assert report["source_sha256"] != report["preprocessing_input_sha256"]
    assert report["preparation_plan_sha256"] == captured.plan.sha256
    assert report["coverage_granularity"] == "planned-preparation-hooks-and-composite-tail-plus-preprocessing"


@pytest.mark.parametrize("fault", ["source", "output", "plan", "metadata", "extra", "incomplete", "domain"])
def test_stale_preparation_refuses_before_preprocessing_or_checker(tmp_path, monkeypatch, fault):
    captured = _capture(tmp_path)
    options = _options()
    match fault:
        case "source":
            captured.source_path.write_text("changed")
        case "output":
            captured.output.write_text("changed")
        case "plan":
            captured.manifest.write_bytes(captured.manifest.read_bytes() + b"\n")
        case "metadata":
            meta = stages.completed(captured.root)[1] / "meta.json"
            payload = json.loads(meta.read_text())
            payload["pass"] = "unreviewed.pass"
            meta.write_text(json.dumps(payload))
        case "extra":
            monkeypatch.setenv(stages.ENV, str(captured.root))
            stages.record_boundary("extra", "unreviewed.pass", SOURCE, SOURCE)
        case "incomplete":
            (captured.root / ".reserved" / "000013").mkdir()
        case "domain":
            options = _options(bits=32)
    monkeypatch.setattr(check, "check_typed_stage", lambda *a, **kw: pytest.fail("stale capture reached checker"))
    with pytest.raises(handoff.VerificationError):
        handoff.run(
            captured.target.decode(),
            tmp_path / "upstream.mlir",
            options,
            lambda *a: pytest.fail("stale capture reached preprocessing"),
            plan=plans.preprocessing(),
            prefix=captured,
        )


@pytest.mark.parametrize("fault", ["original", "endpoint", "manifest", "stage"])
def test_preparation_inventory_rechecked_after_chain_replay(tmp_path, monkeypatch, fault):
    captured = _capture(tmp_path)
    _stub_checks(monkeypatch)

    def race(*a, **kw):
        match fault:
            case "original":
                captured.source_path.write_text("changed")
            case "endpoint":
                captured.output.write_text("changed")
            case "manifest":
                captured.manifest.write_bytes(captured.manifest.read_bytes() + b"\n")
            case "stage":
                path = stages.completed(captured.root)[1] / "meta.json"
                path.write_bytes(path.read_bytes() + b"\n")
        return {"verdict": "verified", "domain": "test-only", "assumptions": []}

    monkeypatch.setattr(check, "compose_chain", race)
    from merlin.llvmlower.passes_xdsl import preprocess_text

    with pytest.raises(handoff.VerificationError):
        handoff.run(
            captured.target.decode(),
            tmp_path / "upstream.mlir",
            _options(),
            preprocess_text,
            plan=plans.preprocessing(),
            prefix=captured,
        )


@pytest.mark.parametrize("bits", [32, 64])
def test_real_preparation_to_preprocessing_strict_original_source_chain(tmp_path, lean_tools, bits):
    options = _options(lean_tools.m2m_python, bits)
    captured = _capture(tmp_path, options)
    from merlin.llvmlower.passes_xdsl import preprocess_text

    result = handoff.run(
        captured.target.decode(),
        tmp_path / "upstream.mlir",
        options,
        preprocess_text,
        plan=plans.preprocessing(),
        prefix=captured,
    )
    report = json.loads(result.report.read_text())
    chain = json.loads(Path(report["chain"]).read_text())
    assert report["verdict"] == chain["verdict"] == "verified"
    assert chain["source_definedness"]["verdict"] == "verified"
    assert chain["replayed"] and chain["chained"]
    assert len(report["boundaries"]) == 17


@pytest.mark.parametrize("fault", ["omitted-unchanged", "duplicate", "wrong-pass"])
def test_actual_preparation_hook_roster_cannot_hide_unrecorded_occurrences(tmp_path, monkeypatch, fault):
    record = stages.record_boundary

    def changed(name, pass_entry, before, after, **kwargs):
        if name == "prepare-torchao-affine":
            assert before == after  # Omitting an unchanged slot must still fail completeness.
            if fault == "omitted-unchanged":
                return None
            if fault == "duplicate":
                record(name, pass_entry, before, after, **kwargs)
            if fault == "wrong-pass":
                pass_entry = "unreviewed.pass"
        return record(name, pass_entry, before, after, **kwargs)

    monkeypatch.setattr(stages, "record_boundary", changed)
    with pytest.raises(plans.PlanMismatch):
        _capture(tmp_path)


def test_explicit_preparation_prefix_requires_enabled_lowering_gate(tmp_path):
    from merlin.llvmlower.lower import lower_model

    captured = _capture(tmp_path)
    with pytest.raises(ValueError, match="requires explicit shared-pass verification"):
        lower_model(captured.target.decode(), tmp_path / "lower", targets=(), shared_pass_preparation=captured)


def test_lowering_forwards_explicit_captured_prefix_and_consumes_exact_verified_text(tmp_path, monkeypatch):
    from merlin.llvmlower import ciface_inline, lower

    captured = _capture(tmp_path)
    _stub_checks(monkeypatch)
    consumed = []

    def backend(text, **kwargs):
        consumed.append(text)
        kwargs["lowering_selection"].update(index_bits=64, effective_pipeline="test-only backend")
        return "define void @forward() { ret void }"

    monkeypatch.setattr(lower, "lower_to_llvm_ir", backend)
    monkeypatch.setattr(ciface_inline, "inline_wide_ciface_implementations", lambda text: (text, {}))
    result = lower.lower_model_file(
        captured.output,
        tmp_path / "lower",
        targets=(),
        index_bits=64,
        shared_pass_verification=_options(),
        shared_pass_preparation=captured,
    )
    assert consumed == [(result.workdir / "model.upstream.mlir").read_text()]
    report = json.loads(Path(result.stats["shared_pass_verification"]).read_text())
    assert len(report["boundaries"]) == 17


def test_mandatory_boolean_frontend_repair_is_not_automatically_verified(tmp_path, lean_tools):
    source = """func.func @forward(%b: i1) -> f32 {
      %r = arith.sitofp %b : i1 to f32
      return %r : f32
    }"""
    options = _options(lean_tools.m2m_python)
    captured = _capture(tmp_path, options, source)
    repair = next(p for p in stages.completed(captured.root) if p.name.endswith("prepare-fix-bool-sitofp"))
    assert json.loads((repair / "meta.json").read_text())["changed"]
    assert '"arith.sitofp"' in (repair / "before.mlir").read_text()
    assert '"arith.uitofp"' in (repair / "after.mlir").read_text()
    from merlin.llvmlower.passes_xdsl import preprocess_text

    with pytest.raises(handoff.VerificationError):
        handoff.run(
            captured.target.decode(),
            tmp_path / "upstream.mlir",
            options,
            preprocess_text,
            plan=plans.preprocessing(),
            prefix=captured,
        )
    report = json.loads(next(tmp_path.glob("lean-shared-passes/*/verification.json")).read_text())
    assert report["verdict"] != "verified"


@pytest.mark.parametrize("mutation", ["omitted", "reordered"])
def test_reconstructed_capture_cannot_supply_its_own_denominator(tmp_path, mutation):
    captured = _capture(tmp_path)
    original = stages.completed(captured.root)
    assert (original[1] / "before.mlir").read_bytes() == (original[1] / "after.mlir").read_bytes()
    assert (original[2] / "before.mlir").read_bytes() == (original[2] / "after.mlir").read_bytes()
    indices = [0, *range(2, len(original))] if mutation == "omitted" else [0, 2, 1, *range(3, len(original))]
    expected = plans.Plan(
        captured.plan.scope, tuple(captured.plan.boundaries[n] for n in indices), captured.plan.granularity
    )
    root = tmp_path / "reconstructed" / "stages"
    with stages.capture(root) as recording:
        for index in indices:
            stage = original[index]
            meta = json.loads((stage / "meta.json").read_text())
            stages.record_boundary(
                meta["boundary"], meta["pass"], (stage / "before.mlir").read_text(), (stage / "after.mlir").read_text()
            )
        stages.record_handoff(captured.output)
    (root.parent / "source.mlir").write_bytes(captured.source)
    (root.parent / "expected-plan.json").write_bytes(expected.canonical_bytes())
    _, fingerprints = handoff._inventory(recording)
    forged = replace(
        captured, root=root, run=recording.run, plan=expected, fingerprints=tuple(sorted(fingerprints.items()))
    )
    # All attacker-supplied metadata, bytes, reservations and source/target endpoints agree.
    # Replaying against the supplied denominator alone therefore succeeds.
    plans.replay(forged.plan, forged.manifest, forged.root)
    with pytest.raises(ValueError, match="independently authored preparation roster"):
        preparation.validate(forged, _options(), forged.target)


def test_reconstructed_capture_cannot_claim_unchecked_concrete_features(tmp_path):
    captured = _capture(tmp_path)
    forged = replace(captured, features=frozenset({"unchecked-feature"}))
    with pytest.raises(ValueError, match="concrete preparation features"):
        preparation.validate(forged, _options(), forged.target)


@pytest.mark.parametrize("name", ["MERLIN_VEC_RANK", "MERLIN_PASSES"])
def test_effective_selection_drift_does_not_publish_a_completed_capture(tmp_path, monkeypatch, name):
    original = zephyr_model.prepare_for_lowering

    def drift(*args, **kwargs):
        result = original(*args, **kwargs)
        monkeypatch.setenv(name, "1" if name == "MERLIN_VEC_RANK" else "int-softmax-table")
        return result

    monkeypatch.setattr(zephyr_model, "prepare_for_lowering", drift)
    with pytest.raises(ValueError, match="effective policy selections"):
        _capture(tmp_path)
    assert not list((tmp_path / "prepare").glob("lean-preparation/*/capture.json"))
