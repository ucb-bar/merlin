"""Original selected runtime consumes pinned observation data without raw-facts substitution."""

import copy
import hashlib
import json
import os
import runpy
from types import SimpleNamespace

import pytest
import yaml

from merlin.common.paths import repo_root
from merlin.runtime.backends import base, chipyard_rocc_bound, rocc_selection
from merlin.targetgen import trace_check
from merlin.targetgen.rocc import selected_operand_protocol as P
from merlin.targetgen.rocc import semantics


def fixture(tmp_path):
    observer = runpy.run_path(str(repo_root() / "merlin/tests/targetgen/test_selected_operand_protocol.py"))
    original, facts, _, sidecar, source, isa = observer["selected"](tmp_path)
    runtime = runpy.run_path(str(repo_root() / "merlin/tests/runtime/test_bound_chipyard_rocc_backend.py"))
    contract, native_facts = runtime["selection"].__wrapped__(tmp_path)
    contract["name"] = "array"
    contract["rocc_operand_roles"]["instructions"] = [{"funct": 4, "class": "INPUT", "operands": {}}]
    native_facts["inputs"].update(facts["inputs"])
    native_facts["inputs"]["target"] = "array"
    native_facts["facts"]["interfaces"] = facts["facts"]["interfaces"]
    fact_bytes = json.dumps(native_facts).encode()
    sidecar["original_facts_sha256"] = hashlib.sha256(fact_bytes).hexdigest()
    fact_file, sidecar_file = tmp_path / "facts.json", tmp_path / "operand-protocol.json"
    fact_file.write_bytes(fact_bytes)
    sidecar_file.write_text(json.dumps(sidecar))

    def pin(path):
        return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    selection = {"original_facts": pin(fact_file), "sidecar": pin(sidecar_file)}
    return contract, native_facts, selection, fact_bytes, source, isa, original


def test_factory_selection_reaches_original_structural_checker_and_preserves_unknowns(tmp_path, monkeypatch):
    contract, facts, selection, _, _, _, _ = fixture(tmp_path)
    before = copy.deepcopy((contract, facts))
    selected = rocc_selection.select_operand_protocol(contract, selection)
    from merlin.targetgen import target_registry
    from merlin.targetgen.rtl import facts as rtl_facts

    monkeypatch.setattr(target_registry, "resolve", lambda target: SimpleNamespace(load_contract=lambda: selected))
    monkeypatch.setattr(rtl_facts, "load_facts", lambda *a, **k: facts)
    monkeypatch.setattr(base, "_ensure_discovered", lambda: pytest.fail("selected data must not discover providers"))
    bound = base.get_backend("array")
    assert type(bound) is chipyard_rocc_bound.BoundRoCCBackend
    assert bound.rocc_semantics._facts == facts
    assert bound.rocc_semantics.isa_constants("array")["FUNCT_CLASS"] == {4: "INPUT"}
    trace = {"abi": {"custom_opcode": "0x2b", "funct3": "0x7"}, "instructions": [
        {"class": "INPUT", "funct": 4, "rs1": {"kind": "argbase", "raw": None},
         "rs2": {"kind": "const", "raw": 17 << 20}, "decoded": {}}]}
    view, _, raw_classes, activated = trace_check._bound_trace_view(trace, "array")
    assert activated and raw_classes == ["INPUT"]
    assert view["instructions"][0]["class"] == "MVIN"
    assert view["instructions"][0]["decoded"] == {"rows": 17}
    assert trace["instructions"][0]["decoded"] == {} and (contract, facts) == before
    checked = trace_check.check(trace, {"modes": {"resident_reuse": True}}, target="array")
    assert checked["trace_requirement_status"] == "unavailable_fields"
    assert "resident_reuse" in checked["unavailable_requirements"]


def test_legacy_contract_has_no_ambient_operand_activation(tmp_path):
    contract, facts, _, _, _, _, _ = fixture(tmp_path)
    assert semantics.bind(target="array", contract=contract, facts=facts).operand_observer is None


@pytest.mark.parametrize("mutation", ["sidecar", "facts", "source", "missing", "extra", "already_selected"])
def test_bad_original_selection_cannot_publish_runtime_observer(tmp_path, mutation):
    contract, facts, selection, _, source, _, _ = fixture(tmp_path)
    selected = rocc_selection.select_operand_protocol(contract, selection)
    if mutation in {"sidecar", "facts", "source"}:
        name = "facts.json" if mutation == "facts" else "operand-protocol.json"
        path = source if mutation == "source" else tmp_path / name
        before = path.stat()
        data = path.read_bytes()
        changed = data.replace(b"4", b"5", 1) if b"4" in data else bytes([data[0] ^ 1]) + data[1:]
        path.write_bytes(changed)
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    elif mutation == "missing":
        (tmp_path / "operand-protocol.json").unlink()
    elif mutation == "extra":
        selected["rocc_operand_selection"]["unselected"] = True
    else:
        selected["rocc_operand_protocol"] = {"version": 1, "encoding": {"semantic_class": {4: "MVIN"}},
                                              "roles": {"classes": {}}}
    with pytest.raises((ValueError, OSError)):
        semantics.bind(target="array", contract=selected, facts=facts)


def test_selection_file_mutation_at_final_boundary_refuses_before_publication(tmp_path, monkeypatch):
    contract, facts, selection, _, _, _, _ = fixture(tmp_path)
    selected = rocc_selection.select_operand_protocol(contract, selection)
    bind = P.bind_original_operand_selection

    def changed(*a, **k):
        observed = bind(*a, **k)
        path = tmp_path / "operand-protocol.json"
        before = path.stat()
        path.write_bytes(path.read_bytes().replace(b"4", b"5", 1))
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
        return observed

    monkeypatch.setattr(P, "bind_original_operand_selection", changed)
    with pytest.raises(ValueError, match="selected byte pin"):
        semantics.bind(target="array", contract=selected, facts=facts)


def test_release_contract_records_selection_without_changing_original_bytes(tmp_path):
    from merlin_experiments.corpus.preparation import _stage_capability_contract

    contract, _, selection, _, _, _, _ = fixture(tmp_path)
    source = tmp_path / "original.yaml"
    source.write_text(yaml.safe_dump(contract))
    original = source.read_bytes()
    digest = hashlib.sha256(original).hexdigest()
    destination = tmp_path / "selected.yaml"
    record = _stage_capability_contract(source, digest, destination, None, operand_selection=selection)
    staged = yaml.safe_load(destination.read_bytes())
    assert staged.pop("rocc_operand_selection") == selection
    assert staged == contract and source.read_bytes() == original
    assert record["sha256"] == digest
    assert record["operand_protocol_selection"]["selected_contract_sha256"] == hashlib.sha256(
        destination.read_bytes()).hexdigest()


@pytest.mark.parametrize("mutation", [None, "formatting", "numeric", "facts", "raw_pin", "raw_duplicate",
                                     "field_type", "missing_host_source", "extra"])
def test_reviewed_handoff_requires_original_facts_complete_host_closure_and_exact_document(
    tmp_path, monkeypatch, mutation,
):
    from merlin_experiments.phase0 import coverage_commitment
    from merlin_experiments.phase1 import corpus_inputs

    root = tmp_path / "release/experiment"
    root.mkdir(parents=True)
    contract, facts, selection, raw, _, _, _ = fixture(tmp_path)
    (root / "rtl_facts").mkdir()
    (root / "contracts").mkdir()
    (root / "rtl_facts/facts.json").write_bytes(raw)
    (root / "contracts/operand-protocol.json").write_bytes((tmp_path / "operand-protocol.json").read_bytes())
    for role, relative in (("original_facts", "rtl_facts/facts.json"), ("sidecar", "contracts/operand-protocol.json")):
        selection[role]["path"] = str(root / relative)
    selected = rocc_selection.select_operand_protocol(contract, selection)
    host = [{"path": path} for path in P.selected_inputs(selection)]
    if mutation == "numeric":
        selected["numeric_policy"] = {"approximate": True}
    elif mutation == "facts":
        raw = raw.replace(b"selected_config", b"changed_config")
    elif mutation == "formatting":
        raw = (json.dumps(json.loads(raw), sort_keys=True, indent=2) + "\n").encode()
    elif mutation == "raw_duplicate":
        raw = b'{"inputs":null,' + raw[1:]
    elif mutation == "field_type":
        changed = json.loads(raw)
        changed["facts"]["interfaces"][0]["legal_funct"] = [4.0]
        raw = json.dumps(changed).encode()
    elif mutation == "missing_host_source":
        host.pop()
    elif mutation == "extra":
        selected["rocc_operand_selection"]["extra"] = 1
    contract_file = root / "contracts/target_contract.yaml"
    contract_file.write_text(yaml.safe_dump(selected))
    authored = root / "input_bundles/el4/input_bundle_manifest.yaml"
    authored.parent.mkdir(parents=True)
    authored.write_text("bundle_id: el4\n")
    te = SimpleNamespace(path=root / "target_experiment.yaml", capsule_corpus=root / "corpus/isa",
                         declared_contract_path=lambda: contract_file)
    monkeypatch.setattr(coverage_commitment, "read_inputs", lambda _: {
        "capability_contract": contract, "raw_facts_utf8": raw.decode(),
        "evidence": {"raw_facts_sha256": "0" * 64 if mutation == "raw_pin"
                     else hashlib.sha256(raw).hexdigest()},
    })
    bundle = {"bundle_id": "el4", "allowed": [{"mode": "ro", "path": str(contract_file)}], "host_inputs": host}
    if mutation in (None, "formatting"):
        corpus_inputs.require_reviewed_bundle(te, authored, bundle)
    else:
        with pytest.raises(ValueError):
            corpus_inputs.require_reviewed_bundle(te, authored, bundle)
