"""Host reconstruction binds original semantic bytes and refuses invented admission."""

from __future__ import annotations

import copy
import hashlib
import json

import pytest
import yaml

from merlin.verify.lean_isa_semantics import reconstruct_isa_semantics, select_isa_semantics


def _inputs(tmp_path, *, admitted=False):
    facts = {
        "facts": {
            "target": "semantic_probe",
            "word_bits": 12,
            "data_bits": 8,
            "fields": [{"offset": 0, "width": 4, "value": 1}, {"offset": 4, "width": 8}],
        }
    }
    raw = json.dumps(facts, sort_keys=True).encode()
    facts_path = tmp_path / "facts.json"
    facts_path.write_bytes(raw)

    def ref(*parts):
        return {"fact": list(parts)}

    software = {
        "schema": "merlin.software_spec.v1",
        "status": "reviewed",
        "target": "semantic_probe",
        "isa": {
            "schema": "merlin.isa_spec.v1",
            "namespace": "probe",
            "facts_sha256": hashlib.sha256(raw).hexdigest(),
            "word_bits": ref("facts", "word_bits"),
            "resources": [
                {"name": name, "kind": "state", "unit": "state_value"} for name in ("configuration", "observer")
            ],
            "instructions": [
                {
                    "name": "configure",
                    "operands": [
                        {
                            "name": "value",
                            "type": {"kind": "integer", "bits": ref("facts", "data_bits"), "signed": False},
                            "unit": "value",
                        }
                    ],
                    "results": [],
                    "immediates": [],
                    "encoding": [
                        {
                            "offset": ref("facts", "fields", index, "offset"),
                            "width": ref("facts", "fields", index, "width"),
                            **(
                                {"fixed": ref("facts", "fields", index, "value")}
                                if index == 0
                                else {"argument": "value"}
                            ),
                        }
                        for index in range(2)
                    ],
                    "effects": {"reads": ["observer"], "writes": ["configuration"]},
                }
            ],
        },
    }
    if admitted:
        software["formal_semantics"] = {
            "schema": "merlin.formal_semantics_admission.v1",
            "admitted_instructions": ["probe.configure"],
            "atomic_instructions": ["probe.configure"],
            "disjoint_resources": [["probe:configuration", "probe:observer"]],
            "premises": {},
        }
    spec_path = tmp_path / "software.yaml"
    spec_path.write_text(yaml.safe_dump(software))
    return spec_path, facts_path, software


def test_empty_original_selection_uses_only_builtin_host_defined_behavior():
    selected = select_isa_semantics()
    assert selected.registry == {} and selected.models == ()
    assert selected.record()["selection"] == "builtins_only"
    assert reconstruct_isa_semantics(selected.record()) == selected
    contract = selected.effect_contract(index_bits=64)
    assert contract.operations == () and contract.disjoint_resources == ()
    assert contract.standard_memory_sequential and contract.standard_allocations_successful
    assert not contract.dynamic_memref_accesses_defined
    assert contract.index_bits == 64


def test_registry_comes_from_original_raw_facts_without_automatic_isa_atomicity(tmp_path):
    software, facts, _ = _inputs(tmp_path)
    selected = select_isa_semantics(software_spec_path=software, facts_path=facts, target="semantic_probe")
    assert selected.record()["selection"] == "software_isa"
    assert selected.registry["probe.configure"]["operand_types"] == [{"kind": "integer", "bits": 8, "signed": False}]
    assert selected.registry["probe.configure"]["effects"] == {"reads": ["observer"], "writes": ["configuration"]}
    assert selected.effect_contract(index_bits=64).operations == ()
    assert reconstruct_isa_semantics(selected.record()).record() == selected.record()
    assert selected.record()["rtl_facts"]["sha256"] == hashlib.sha256(facts.read_bytes()).hexdigest()


def test_only_reviewed_original_admission_supplies_isa_state_and_disjointness(tmp_path):
    software, facts, _ = _inputs(tmp_path, admitted=True)
    selected = select_isa_semantics(software_spec_path=software, facts_path=facts, target="semantic_probe")
    contract = selected.effect_contract(index_bits=32)
    assert len(contract.operations) == 1
    assert contract.operations[0].name == "probe.configure"
    assert contract.operations[0].writes == ("probe:configuration",)
    assert contract.operations[0].sequential_atomic_complete
    assert contract.disjoint_resources == (("probe:configuration", "probe:observer"),)
    assert contract.index_bits == 32


def test_snapshot_callback_uses_frozen_bytes_after_original_source_changes(tmp_path):
    software, facts, _ = _inputs(tmp_path)
    record = select_isa_semantics(software_spec_path=software, facts_path=facts).record()
    frozen = tmp_path / "snapshot"
    frozen.mkdir()
    selected = {}
    for source in (software, facts):
        target = frozen / source.name
        target.write_bytes(source.read_bytes())
        selected[source] = target
    software.write_text("schema: changed\n")
    assert reconstruct_isa_semantics(record, resolve_path=selected.__getitem__).record() == record
    with pytest.raises(ValueError):
        reconstruct_isa_semantics(record)


@pytest.mark.parametrize(
    "fault", ["software_bytes", "facts_bytes", "target", "registry", "model", "atomic", "resource", "premise"]
)
def test_semantic_mutations_cannot_change_original_admission(tmp_path, fault):
    software, facts, document = _inputs(tmp_path, admitted=True)
    record = select_isa_semantics(software_spec_path=software, facts_path=facts).record()
    if fault == "software_bytes":
        software.write_text(software.read_text() + "\n")
    elif fault == "facts_bytes":
        facts.write_text(facts.read_text() + "\n")
    elif fault in {"target", "registry", "model"}:
        record = copy.deepcopy(record)
        record[{"target": "target", "registry": "registry_sha256", "model": "model_sha256"}[fault]] = (
            "different" if fault == "target" else "f" * 64
        )
    else:
        if fault == "atomic":
            document["formal_semantics"]["atomic_instructions"] = []
        elif fault == "resource":
            document["formal_semantics"]["disjoint_resources"] = [["probe:configuration", "missing"]]
        else:
            document["formal_semantics"]["premises"] = {"arbitrary_async_safe": True}
        software.write_text(yaml.safe_dump(document))
    with pytest.raises(ValueError):
        reconstruct_isa_semantics(record)


def test_no_target_from_filename_and_no_symlink_or_missing_facts_fallback(tmp_path):
    software, facts, _ = _inputs(tmp_path)
    with pytest.raises(ValueError):
        select_isa_semantics(software_spec_path=software, facts_path=facts, target="filename_guess")
    with pytest.raises(ValueError):
        select_isa_semantics(software_spec_path=software)
    alias = tmp_path / "software_alias.yaml"
    alias.symlink_to(software)
    with pytest.raises(ValueError):
        select_isa_semantics(software_spec_path=alias, facts_path=facts)
