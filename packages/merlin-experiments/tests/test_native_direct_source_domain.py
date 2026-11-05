"""Original host transport with neutral source/receipt fixtures; no target runs."""

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase2 import _archive_source_domain as DOMAIN
from merlin_experiments.phase2 import _native_direct_source_domain as DIRECT
from merlin_experiments.phase2 import public_native_exploration as N
from merlin_experiments.phase2 import public_source_diagnostic as POLICY
from merlin_experiments.phase2.contracts import StageGateError, sha256_file
from merlin_experiments.phase2.ordinary_checked_source import record
from test_archive_source_domain import catalog
from test_archive_source_domain import source as text_source
from test_checked_source_model_diagnostic import selected as selected
from test_ordinary_native_archive_screen import ordinary_inputs as ordinary_inputs
from test_ordinary_native_archive_screen import source as source
from test_public_native_binding_correspondence import inputs as inputs
from test_public_native_binding_correspondence import setup

from merlin.targetgen import checked_source_model_diagnostic as S


def pointer(passes, facts, target="fixture"):
    return S.select_checked_source_diagnostic(passes, target=target, rtl_facts=facts,
        board_config="fixture", native_execution_protocol=S.POINTER_PROTOCOL)


def receipt_fixture(tmp_path, selected):
    passes, _, _, facts, _ = selected
    checked = pointer(passes, facts)
    # A separately reconstructed child selector has the same pinned record.
    child = pointer(passes, facts)
    assert child is not checked and record(child, passes) == record(checked, passes)
    policy = SimpleNamespace(_checked_source_diagnostic=checked, _compiler_pass_selection=passes)
    capdir, workspace = tmp_path / "capsule", tmp_path / "execution"
    capdir.mkdir()
    workspace.mkdir()
    raw = text_source().encode()
    (capdir / "capsule.interface.mlir").write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    cb = {"compiler_pointer_abi": {"version": 1, "arguments": ["lhs", "rhs", "out"]}, "tensors": {
        name: {"shape": shape, "dtype": dtype, "role": role} for name, shape, dtype, role in (
            ("lhs", [2, 3], "i8", "input"), ("rhs", [3, 4], "i8", "input"),
            ("out", [2, 4], "i32", "output"))}}
    compiled = {"status": "observed", "elf_sha256": "a" * 64, "candidate_diagnostics": {
        "source_emission_join": {"status": "verified", "source_sha256": digest},
        "artifacts": {"source": {"sha256": digest}}}}
    proof = {"interface_sha256": digest, "operations": [{"ordinal": 2, "kind": "multiply_accumulate",
        "initial_interval": [0, 0], "reduction_extent": 3, "result_bits": 32}]}
    qualification = {"schema": "merlin.phase0.connected_integer_qualification.v1", "status": "qualified",
        "method": "checked_native_integer_v1", "source_proof": proof, "source_rule": {"fixture": True},
        "realization": {"status": "verified_source_execution_correspondence", "source_sha256": digest,
            "source_rule": {"fixture": True}}}
    original = {"functional_status": "PASS", "compiled": compiled, "checked_source_diagnostic": record(child, passes),
        "original_grade": {"checked_source_status": "complete_source_L0_L1_L2_diagnostic_pass",
            "complete_source_qualification": qualification,
            "tiers": {tier: {"status": "pass"} for tier in ("L0", "L1", "L2")}}}
    member = SimpleNamespace(source_dir=capdir, source_sha256="b" * 64,
        descriptor={"operation": {"op": "model"}, "interface_mlir": "capsule.interface.mlir"})
    return policy, member, workspace, original, cb, catalog(DOMAIN.typed_profile(raw.decode()))


def publish(workspace, original, cb):
    path = workspace / "command_buffer.json"
    path.write_text(json.dumps(cb))
    original["compiled"]["candidate_diagnostics"]["artifacts"]["command_buffer"] = {"sha256": sha256_file(path)}
    path = workspace / "ordinary_observation.json"
    path.write_text(json.dumps(original))
    value = {key: item for key, item in original.items() if key not in {"original_grade", "checked_source_diagnostic"}}
    return {**value, "evidence_sha256": sha256_file(path)}


def test_direct_source_uses_original_child_receipt_with_equal_record_not_same_object(tmp_path, selected):
    policy, member, workspace, original, cb, evidence = receipt_fixture(tmp_path, selected)
    value = publish(workspace, original, cb)
    before = (workspace / "ordinary_observation.json").read_bytes()
    result = DIRECT.current(policy, member, workspace, value, evidence)
    assert result["status"] == "source_domain_supported"
    assert result["training_source_groups"] == ["training-group"]
    assert result["source_domain_protocol"] == DIRECT.PROTOCOL
    assert result["numerical_receipt_sha256"] == value["evidence_sha256"]
    assert DOMAIN.current(member, value["compiled"], workspace, evidence)["status"] == "UNKNOWN"
    assert (workspace / "ordinary_observation.json").read_bytes() == before


@pytest.mark.parametrize("changed", ["source", "command_buffer", "receipt", "selector"])
def test_direct_source_rechecks_current_inputs_after_structural_binding(tmp_path, selected, monkeypatch, changed):
    policy, member, workspace, original, cb, evidence = receipt_fixture(tmp_path, selected)
    value = publish(workspace, original, cb)
    train_support = DOMAIN.train_support

    def mutate(catalog, profile):
        result = train_support(catalog, profile)
        paths = {"source": member.source_dir / "capsule.interface.mlir",
                 "command_buffer": workspace / "command_buffer.json",
                 "receipt": workspace / "ordinary_observation.json"}
        if changed == "selector":
            object.__setattr__(policy._checked_source_diagnostic, "_record_json", "{}")
        else:
            with paths[changed].open("ab") as stream:
                stream.write(b"\n")
        return result

    monkeypatch.setattr(DOMAIN, "train_support", mutate)
    assert DIRECT.current(policy, member, workspace, value, evidence)["status"] == "UNKNOWN"


@pytest.mark.parametrize("changed", ["command_buffer", "receipt"])
def test_reopened_payload_bytes_must_match_the_actual_original_pin(tmp_path, selected, monkeypatch, changed):
    policy, member, workspace, original, cb, evidence = receipt_fixture(tmp_path, selected)
    value = publish(workspace, original, cb)
    target = workspace / ("command_buffer.json" if changed == "command_buffer" else "ordinary_observation.json")
    read_bytes = Path.read_bytes

    def changed_read(path):
        raw = read_bytes(path)
        if path == target:
            item = json.loads(raw)
            if changed == "receipt":
                item["original_grade"]["injected"] = "not part of the pinned original receipt"
            else:
                item["injected_commands"] = "not part of the pinned original command buffer"
            return json.dumps(item).encode()
        return raw

    monkeypatch.setattr(Path, "read_bytes", changed_read)
    if changed == "receipt":
        with pytest.raises(StageGateError, match="changed during reopening"):
            DIRECT.current(policy, member, workspace, value, evidence)
    else:
        result = DIRECT.current(policy, member, workspace, value, evidence)
        assert result["status"] == "UNKNOWN" and "changed during reopening" in result["reason"]


@pytest.mark.parametrize("fault", ["source", "proof", "abi", "layout", "shape", "nonzero", "unseen", "capacity",
    "qualification", "realization", "source_rule", "join"])
def test_direct_source_never_expands_semantics_geometry_or_source_correspondence(tmp_path, selected, fault):
    policy, member, workspace, original, cb, evidence = receipt_fixture(tmp_path, selected)
    qualification = original["original_grade"]["complete_source_qualification"]
    if fault == "source":
        (member.source_dir / "capsule.interface.mlir").write_text(text_source() + "\n")
    elif fault == "proof":
        qualification["source_proof"]["operations"][0]["reduction_extent"] = 4
    elif fault == "abi":
        cb["compiler_pointer_abi"]["arguments"] = ["rhs", "lhs", "out"]
    elif fault == "layout":
        cb["tensors"]["rhs"]["layout"] = "transposed"
    elif fault == "shape":
        cb["tensors"]["out"]["shape"] = [2, 5]
    elif fault == "nonzero":
        (member.source_dir / "capsule.interface.mlir").write_text(text_source().replace("constant 0", "constant 1"))
    elif fault == "unseen":
        evidence["sources"]["source"]["semantic_profile"]["n"] = 8
    elif fault == "capacity":
        member.descriptor["performance"] = {"capacity": True}
    elif fault == "qualification":
        qualification["status"] = "saved_pass"
    elif fault == "realization":
        qualification["realization"]["source_sha256"] = "c" * 64
    elif fault == "source_rule":
        qualification["source_rule"] = {"foreign": True}
    else:
        original["compiled"]["candidate_diagnostics"]["source_emission_join"]["status"] = "UNKNOWN"
    value = publish(workspace, original, cb)
    assert DIRECT.current(policy, member, workspace, value, evidence)["status"] == "UNKNOWN"


@pytest.mark.parametrize("fault", ["selector", "record", "L2", "functional", "receipt_hash"])
def test_saved_selector_or_weaker_original_numerical_receipt_is_not_authority(tmp_path, selected, fault):
    policy, member, workspace, original, cb, evidence = receipt_fixture(tmp_path, selected)
    if fault == "selector":
        policy._checked_source_diagnostic = copy.copy(policy._checked_source_diagnostic)
    elif fault == "record":
        original["checked_source_diagnostic"] = {"status": "PASS"}
    elif fault == "L2":
        original["original_grade"]["tiers"]["L2"]["status"] = "pending"
    elif fault == "functional":
        original["functional_status"] = "FAIL"
    value = publish(workspace, original, cb)
    if fault == "receipt_hash":
        value["evidence_sha256"] = "0" * 64
    with pytest.raises((ValueError, StageGateError)):
        DIRECT.current(policy, member, workspace, value, evidence)


def test_explicit_direct_domain_requires_live_pointer_policy_and_original_bridge(source, inputs):
    args, selected_native, bridge = setup(source, inputs)
    with pytest.raises(StageGateError, match="original compiler/native bridge"):
        N.prepare_public_native_exploration(selected_native, source_domain_protocol=DIRECT.PROTOCOL)
    owner = N.prepare_public_native_exploration(selected_native, binding_correspondence=bridge,
        source_domain_protocol=DIRECT.PROTOCOL)
    with pytest.raises(StageGateError, match="explicit pointer-source"):
        POLICY.prepare_public_source_diagnostic(**args, native_exploration=owner)
    args["checked_source_diagnostic"] = pointer(args["compiler_pass_selection"],
        args["checked_source_diagnostic"].record()["rtl_facts"]["path"], args["target_experiment"].target)
    policy = POLICY.prepare_public_source_diagnostic(**args, native_exploration=owner)
    assert owner.record()["schema"] == "merlin.public_native_exploration.v5"
    assert policy.binding()["formal_admission"] is False and policy._ordinary_screen is None
    object.__setattr__(owner, "source_domain_protocol", None)
    with pytest.raises(StageGateError, match="original issued owners"):
        owner.verify()


@pytest.mark.parametrize("protocol", [True, {}, "automatic", ""])
def test_unknown_direct_protocol_refuses_before_owner_issuance(source, protocol):
    from test_public_native_exploration import selection

    with pytest.raises(StageGateError, match="exact explicit pointer"):
        N.prepare_public_native_exploration(selection(source), source_domain_protocol=protocol)


def test_legacy_owner_does_not_load_the_optional_direct_core_checker(source, monkeypatch):
    from test_public_native_exploration import selection

    monkeypatch.setattr(DIRECT, "source_modules", lambda: pytest.fail("legacy core needs no direct checker"))
    owner = N.prepare_public_native_exploration(selection(source))
    assert owner.record()["schema"] == N.SCHEMA
    assert "source_domain_protocol" not in owner.record()


@pytest.mark.parametrize("fault", ["function", "core_binding", "helper_source", "numerical_join",
                                  "direct_products", "original_products"])
def test_direct_owner_pins_current_helper_code_and_source(source, inputs, monkeypatch, fault):
    _, selected_native, bridge = setup(source, inputs)
    owner = N.prepare_public_native_exploration(selected_native, binding_correspondence=bridge,
        source_domain_protocol=DIRECT.PROTOCOL)
    if fault == "function":
        monkeypatch.setattr(DIRECT, "current", lambda *_a: {"status": "source_domain_supported"})
    elif fault == "core_binding":
        monkeypatch.setattr(DIRECT.source_modules()[0], "bind_direct_source_contraction", lambda *_a: {})
    elif fault == "numerical_join":
        monkeypatch.setattr(N.NATIVE, "original_numerical_observation", lambda *_a, **_kw: {})
    elif fault == "direct_products":
        monkeypatch.setattr(DIRECT, "emitted_source", lambda *_a, **_kw: {})
    elif fault == "original_products":
        monkeypatch.setattr(DIRECT.source_modules()[-1], "reopen_native_products", lambda *_a, **_kw: {})
    else:
        original = N.F.pin

        def changed(path, **kwargs):
            pin = original(path, **kwargs)
            return {**pin, "sha256": "0" * 64} if str(path) == DIRECT.__file__ else pin

        monkeypatch.setattr(N.F, "pin", changed)
    with pytest.raises(StageGateError, match="issued owners|evidence changed"):
        owner.verify()


def test_direct_opt_in_preserves_fresh_counts_and_unchanged_raw_activity_gate(tmp_path, selected, monkeypatch):
    """Neutral observer checks the execution order; it grants no native domain."""
    from merlin_experiments.phase2 import _public_native_observation as OBS

    policy, member, workspace, original, cb, evidence = receipt_fixture(tmp_path, selected)
    elf = workspace / "original.elf"
    elf.write_bytes(b"neutral executable placeholder; no target run")
    original["compiled"]["elf_sha256"] = sha256_file(elf)
    value = publish(workspace, original, cb)
    calls = []
    owner = SimpleNamespace(source_domain_protocol=DIRECT.PROTOCOL, catalog_json=json.dumps(evidence),
        count_feedback_protocol=None, exploratory_estimate_protocol=None,
        reviewed_json=json.dumps({"raw_activity_ranges": {"actual125": "unchanged"}}),
        selection=SimpleNamespace(runtime=object()), record=lambda: {"neutral": True})
    report_path = workspace / "raw-counts.json"
    report_path.write_text(json.dumps({"raw": "outside original range"}))

    def observe(actual, **kwargs):
        assert actual == elf and kwargs["numerical_pass"] is True
        calls.append("fresh original count observation")
        return {"status": "observed", "report": {"path": str(report_path), "sha256": sha256_file(report_path)}}

    def raw_gate(report, ranges):
        assert report == {"raw": "outside original range"} and ranges == {"actual125": "unchanged"}
        calls.append("original raw activity gate")
        return "target.class.store_act1 outside original TRAIN range"

    monkeypatch.setattr(OBS.COUNTS, "observe", observe)
    monkeypatch.setattr(OBS.REVIEW, "domain_reason", raw_gate)
    monkeypatch.setattr(OBS.NATIVE, "compare", lambda *_a, **_kw: pytest.fail("outside-domain counts cannot predict"))
    result = OBS.observe(owner, policy, member, workspace, value, sandbox=None, remaining_s=10)
    assert calls == ["fresh original count observation", "original raw activity gate"]
    assert result["status"] == "UNKNOWN" and result["measured_cycles"] is None
    assert result["promotion_authority"] == "NONE" and result["prediction"] is None
