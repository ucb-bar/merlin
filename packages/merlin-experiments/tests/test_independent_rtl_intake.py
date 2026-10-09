"""Actual local RTL replay, source exclusion and nontransferable intake authority."""

import dataclasses
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase0.rtl_intake import (
    IndependentHardwareIntake,
    RtlIntakeRefusal,
    _exclusion_prefix,
    _outside,
    bind_component_hardware,
    issue_independent_hardware_intake,
)

from merlin.targetgen.rtl import source_selection


@pytest.fixture
def selected(tmp_path):
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("local C compiler unavailable for the actual deterministic producer fixture")
    # The fixture producer implements only one closed test module. These tests
    # exercise actual invocation/replay and census integrity, not CIRCT or a
    # target's semantics, instruction encodings, simulator or bitstream.
    source = tmp_path / "producer.c"
    source.write_text(
        "#include <stdio.h>\n#include <string.h>\n"
        "int main(int argc, char **argv) {\n"
        "  for (int i = 1; i + 1 < argc; ++i) {\n"
        '    if (!strcmp(argv[i], "-o")) {\n'
        '      FILE *f = fopen(argv[i+1], "w"); if (!f) return 2;\n'
        '      fputs("module {\\n  hw.module @Unit() {\\n  }\\n}\\n", f);\n'
        "      return fclose(f) != 0;\n"
        "    }\n"
        "  }\n  return 1;\n}\n"
    )
    tool = tmp_path / "fixture-producer"
    subprocess.run([compiler, str(source), "-o", str(tool)], check=True, capture_output=True)
    firrtl = tmp_path / "selected.fir"
    firrtl.write_text(
        "FIRRTL version 3.2.0\n"
        "circuit Unit :\n"
        "  module Unit : @[generators/test_unit/src/Unit.scala 1:1]\n"
        "    input clock : Clock\n"
        "    smem bank : UInt<8>[4] @[generators/test_unit/src/Unit.scala 3:1]\n"
    )
    bundle = source_selection.produce_selection(
        target="test_unit",
        firrtl=firrtl,
        generator="test_unit",
        config="TestConfiguration",
        core_root="Unit",
        firtool=tool,
        output=tmp_path / "original",
    )
    descriptor = tmp_path / "descriptor.yaml"
    descriptor.write_text("target: test_unit\n")
    private = tmp_path / "private-implementation"
    private.mkdir()
    return {
        "target": "test_unit",
        "descriptor": descriptor,
        "source_bundle": bundle,
        "forbidden_roots": (private,),
        "output": tmp_path / "fresh-intake",
    }


def test_actual_replay_derives_source_facts_and_explicit_unknowns(selected):
    intake = issue_independent_hardware_intake(**selected)
    intake.verify()
    assert intake.fact("memories.0.bytes") == 4
    assert intake.fact("memories.0.elem_bits") == 8
    assert intake.fact("census.unit_root") == "Unit"
    assert len(intake.sha256) == 64
    intake.verify_public_facts(selected["output"] / "facts.json")
    assert set(intake.public_facts()["unknowns"]) >= {
        "historical_source_to_firrtl_origin",
        "instruction_encoding_and_numerical_semantics",
        "bitstream_correspondence",
        "performance",
    }
    assert any(pin.role == "firtool" for pin in intake.source_pins)
    assert (selected["output"] / "production/firtool.log").is_file()
    with pytest.raises(RtlIntakeRefusal, match="not independently derived"):
        intake.fact("funct_decode_table.custom_opcode")


def test_actual_intake_keeps_absent_exclusion_prefixes(selected):
    excluded = selected["output"].parent / "future-private" / "answers"
    assert not excluded.exists()
    selected["forbidden_roots"] += (excluded,)
    intake = issue_independent_hardware_intake(**selected)
    intake.verify()
    assert not excluded.exists()  # admission must not materialize the tree
    assert intake.fact("memories.0.bytes") == 4
    with pytest.raises(RtlIntakeRefusal, match="protected implementation"):
        _outside(excluded / "later-source.fir", (_exclusion_prefix(excluded),))
    with pytest.raises(RtlIntakeRefusal, match="protected implementation"):
        issue_independent_hardware_intake(**{**selected, "output": excluded / "forbidden-output"})
    assert not excluded.exists()


def test_exclusion_prefix_never_opens_contents_or_follows_aliases(tmp_path, monkeypatch):
    protected = tmp_path / "protected"
    protected.mkdir()
    source = protected / "secret.fir"
    source.write_text("unread protected sentinel")

    def forbidden_read(*args, **kwargs):
        raise AssertionError("exclusion-prefix admission opened protected bytes")

    monkeypatch.setattr(Path, "read_bytes", forbidden_read)
    monkeypatch.setattr(Path, "read_text", forbidden_read)
    assert _exclusion_prefix(protected) == protected
    absent = protected / "absent-child"
    assert _exclusion_prefix(absent) == absent
    with pytest.raises(RtlIntakeRefusal, match="ordinary directory or absent"):
        _exclusion_prefix(source)
    alias = tmp_path / "alias"
    alias.symlink_to(protected, target_is_directory=True)
    with pytest.raises(RtlIntakeRefusal, match="indirect protected exclusion"):
        _exclusion_prefix(alias / "still-absent")
    broken_alias = tmp_path / "broken-alias"
    broken_alias.symlink_to(tmp_path / "uncreated-directory", target_is_directory=True)
    with pytest.raises(RtlIntakeRefusal, match="indirect protected exclusion"):
        _exclusion_prefix(broken_alias)
    with pytest.raises(RtlIntakeRefusal, match="indirect protected exclusion"):
        _exclusion_prefix(protected / ".." / "another")


def test_constructor_or_saved_receipt_does_not_mint_authority(selected):
    intake = issue_independent_hardware_intake(**selected)
    copy = IndependentHardwareIntake(intake.target, intake.source_pins, intake.facts_json, intake.receipt_json)
    with pytest.raises(RtlIntakeRefusal, match="live independently issued"):
        copy.verify()
    altered = dataclasses.replace(intake, target="different_target")
    with pytest.raises(RtlIntakeRefusal, match="live independently issued"):
        altered.verify()
    saved = json.loads((selected["output"] / "intake.json").read_bytes())
    assert saved["authority"].startswith("live protected issuer only")


def test_changed_source_or_issued_object_refuses(selected):
    intake = issue_independent_hardware_intake(**selected)
    descriptor = selected["descriptor"]
    descriptor.write_text(descriptor.read_text() + "revision: changed\n")
    with pytest.raises(RtlIntakeRefusal, match="source changed"):
        intake.verify()
    object.__setattr__(intake, "facts_json", b"{}")
    with pytest.raises(RtlIntakeRefusal, match="live independently issued"):
        intake.verify()


def test_cached_hw_with_rewritten_hashes_does_not_survive_actual_replay(selected):
    path = selected["source_bundle"]
    document = json.loads(path.read_bytes())
    core = Path(document["sources"]["core_hw"]["path"])
    core.write_text(core.read_text().replace("@Unit()", "@Unit(in %extra : i8)"))
    document["sources"]["core_hw"]["sha256"] = hashlib.sha256(core.read_bytes()).hexdigest()
    document["production"]["core_hw_sha256"] = document["sources"]["core_hw"]["sha256"]
    path.write_text(json.dumps(document))
    with pytest.raises(RtlIntakeRefusal, match="HW bytes do not reproduce"):
        issue_independent_hardware_intake(**selected)


def test_protected_source_is_refused_before_reading_its_bytes(selected, monkeypatch):
    private = selected["forbidden_roots"][0] / "private.fir"
    private.write_text("private answer sentinel")
    path = selected["source_bundle"]
    document = json.loads(path.read_bytes())
    document["sources"]["firrtl"] = {"path": str(private), "sha256": "f" * 64}
    path.write_text(json.dumps(document))
    original = Path.read_bytes

    def guarded(member):
        if member == private:
            raise AssertionError("a protected source was read before exclusion")
        return original(member)

    monkeypatch.setattr(Path, "read_bytes", guarded)
    with pytest.raises(RtlIntakeRefusal, match="protected implementation"):
        issue_independent_hardware_intake(**selected)


def test_intake_refuses_instruction_transcription_and_indirect_sources(selected):
    path = selected["source_bundle"]
    document = json.loads(path.read_bytes())
    document["sources"]["instruction_table"] = document["sources"]["core_hw"]
    path.write_text(json.dumps(document))
    with pytest.raises(RtlIntakeRefusal, match="only FIRRTL"):
        issue_independent_hardware_intake(**selected)
    document["sources"].pop("instruction_table")
    original = Path(document["sources"]["firrtl"]["path"])
    alias = original.with_name("indirect.fir")
    alias.symlink_to(original)
    document["sources"]["firrtl"]["path"] = str(alias)
    path.write_text(json.dumps(document))
    with pytest.raises(RtlIntakeRefusal, match="indirect"):
        issue_independent_hardware_intake(**selected)


def test_receipt_commands_are_never_run_and_public_fact_bytes_are_complete(selected):
    path = selected["source_bundle"]
    document = json.loads(path.read_bytes())
    sentinel = selected["output"].parent / "must-not-run"
    document["production"]["command"] = ["touch", str(sentinel)]
    path.write_text(json.dumps(document))
    intake = issue_independent_hardware_intake(**selected)
    assert not sentinel.exists()
    public = selected["output"].parent / "public-facts.json"
    altered = intake.public_facts()
    altered["injected_schedule"] = {"selected": True}
    public.write_text(json.dumps(altered))
    with pytest.raises(RtlIntakeRefusal, match="complete structural projection"):
        intake.verify_public_facts(public)


def test_component_generation_binding_cannot_launder_an_old_table_or_support(selected):
    from merlin_experiments.phase0.evidence import EvidenceSource

    intake = issue_independent_hardware_intake(**selected)
    evidence = SimpleNamespace(target=intake.target, raw_facts=intake.facts_json, source_snapshots=())
    assert bind_component_hardware(evidence, intake) == {"hardware_intake_sha256": intake.sha256}
    evidence.raw_facts = b'{"facts":{"arrays":[{"rows":42}]}}'
    with pytest.raises(RtlIntakeRefusal, match="complete independently derived facts"):
        bind_component_hardware(evidence, intake)
    evidence.raw_facts = intake.facts_json
    for role in ("support-source", "instruction-semantics", "isa-source"):
        evidence.source_snapshots = (EvidenceSource(selected["descriptor"], role, b"private compiler table"),)
        with pytest.raises(RtlIntakeRefusal, match="legacy target support or ISA transcription"):
            bind_component_hardware(evidence, intake)
    evidence.source_snapshots = (EvidenceSource(selected["descriptor"], "rtl-source:other_path", b"invented RTL"),)
    with pytest.raises(RtlIntakeRefusal, match="outside the independent intake"):
        bind_component_hardware(evidence, intake)


def test_fresh_evidence_selection_never_calls_legacy_support_or_readout_hooks(selected, monkeypatch):
    from merlin_experiments.phase0.evidence import select_evidence

    from merlin.runtime.backends import base
    from merlin.targetgen import readout_facet, target_registry

    intake = issue_independent_hardware_intake(**selected)

    def forbidden(*args, **kwargs):
        pytest.fail("fresh evidence attempted a legacy support or readout hook")

    monkeypatch.setattr(target_registry, "resolve", forbidden)
    monkeypatch.setattr(readout_facet, "capture_inputs", forbidden)
    monkeypatch.setattr(base, "execution_capability_facts", forbidden)
    evidence = select_evidence(
        intake.target,
        descriptor=selected["descriptor"],
        facts_path=selected["output"] / "facts.json",
        hardware_intake=intake,
    )
    assert bind_component_hardware(evidence, intake) == {"hardware_intake_sha256": intake.sha256}
    assert all(row["satisfied"] is None for row in evidence.performance_facts["execution_capabilities"].values())


def test_fresh_evidence_refuses_header_transcriptions_and_legacy_semantics(selected):
    from merlin_experiments.phase0.evidence import select_evidence

    intake = issue_independent_hardware_intake(**selected)
    common = {"target": intake.target, "facts_path": selected["output"] / "facts.json", "hardware_intake": intake}
    with pytest.raises(RtlIntakeRefusal, match="ISA header transcription"):
        select_evidence(**common, descriptor={"isa_headers": ["old-derived-table.py"]})
    with pytest.raises(RtlIntakeRefusal, match="legacy instruction semantics"):
        select_evidence(**common, capability_contract_path={"name": intake.target, "instruction_semantics": "old.yaml"})
