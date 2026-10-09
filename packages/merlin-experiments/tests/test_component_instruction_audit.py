"""Real selected public tools and sources prove whole-ELF policy enforcement.

The optional integration input explicitly selects clean public source checkouts,
native tool binaries, protected symbols and harmless assembly control sources.
It does not select a handwritten compiler or issue physical runtime authority.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import os
import shutil
from pathlib import Path

import pytest
from merlin_experiments.phase0.accessor_intake import issue_independent_accessor_intake
from merlin_experiments.phase0.command_intake import issue_independent_command_intake
from merlin_experiments.phase0.rtl_intake import issue_independent_hardware_intake
from merlin_experiments.phase0.source_predicate_intake import (
    SourcePredicateSelection,
    issue_independent_source_predicate_intake,
)
from merlin_experiments.phase2 import component_runtime_support as runtime_support
from merlin_experiments.phase2.component_instruction_audit import (
    IndependentLinkedInstructionCheck,
    InstructionDecoderSelection,
)
from merlin_experiments.phase2.component_instruction_policy import issue_independent_instruction_policy
from merlin_experiments.phase2.component_runtime_qualification import RuntimeControlRefusal
from merlin_experiments.phase2.contracts import StageGateError, mapping_file

from merlin.common import invocation_record
from merlin.targetgen.contract.build_service import file_digest


def test_actual_public_native_decoder_rejects_linked_policy_word_before_execution(tmp_path):
    selected = os.environ.get("MERLIN_TEST_INSTRUCTION_SELECTION")
    if not selected:
        pytest.skip("requires explicit independent public source and native tool selections")
    config = json.loads(Path(selected).read_text())
    protected = tmp_path / "protected"
    protected.mkdir()
    exclusions = (protected,)
    hardware = issue_independent_hardware_intake(
        **config["hardware"], forbidden_roots=exclusions, output=tmp_path / "hardware"
    )
    command = issue_independent_command_intake(
        hardware=hardware,
        **{**config["command"], "function_span": tuple(config["command"]["function_span"])},
        forbidden_roots=exclusions,
        output=tmp_path / "command",
    )
    routing = issue_independent_source_predicate_intake(
        command_intake=command,
        public_checkout=config["command"]["checkout"],
        commit=config["command"]["commit"],
        selections=tuple(
            SourcePredicateSelection(Path(row["source"]), row["binding"], row["operand"]) for row in config["routing"]
        ),
        forbidden_roots=exclusions,
        output_root=tmp_path / "routing",
    )
    policy = issue_independent_instruction_policy(
        command_intake=command,
        routing_intake=routing,
        policy_file=config["policy_file"],
        forbidden_roots=exclusions,
        output=tmp_path / "policy.json",
    )
    accessor = issue_independent_accessor_intake(
        **config["accessor"], forbidden_roots=exclusions, output_root=tmp_path / "accessor"
    )
    check = IndependentLinkedInstructionCheck(policy, accessor, InstructionDecoderSelection(**config["decoder"]))
    service = check.admission_service()
    facts = accessor.public_facts()
    field = facts["fields"][check.selection.selector_member]
    word = facts["source_constants"][check.selection.match_constant] | (policy.selectors[0][1] << field["offset"])
    observed = accessor.decode_words((word,), tmp_path / "injection-proof")["rows"][0]
    assert observed[check.selection.selector_member] == policy.selectors[0][1]
    template = Path(config["injection_template"]).read_text()
    assert template.count("__PROHIBITED_BYTES__") == 1
    sources = {
        "positive": Path(config["positive_source"]).read_text(),
        "negative": template.replace(
            "__PROHIBITED_BYTES__",
            ".byte " + ",".join(str(byte) for byte in word.to_bytes(observed["length"], config["byte_order"])),
        ),
    }
    results = {}
    for direction, text in sources.items():
        source, elf = (tmp_path / (direction + suffix) for suffix in (".S", ".elf"))
        source.write_text(text)
        invocation_record.run(
            [config["compiler"], *config["compile_flags"], str(source), "-o", str(elf)],
            directory=tmp_path,
            stage="independent_instruction_control_elf",
            inputs=(source,),
            outputs=(elf,),
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        result = service.evaluate(elf=elf, target=hardware.target, evidence_root=tmp_path / (direction + "-audit"))
        assert result["status"] == ("accepted" if direction == "positive" else "refused")
        assert result["elf_sha256"] == file_digest(elf)
        results[direction] = result
    assert not results["positive"]["prohibited_hits"]
    hits = results["negative"]["prohibited_hits"]
    assert hits and all(row["source_symbol"] == policy.selectors[0][0] for row in hits)
    assert all(row["word"] == word and row["length"] == observed["length"] for row in hits)
    if "additional_elf" in config:
        additional = service.evaluate(
            elf=Path(config["additional_elf"]),
            target=hardware.target,
            evidence_root=tmp_path / "additional-artifact-audit",
        )
        assert additional["status"] == "accepted" and not additional["prohibited_hits"]
    records = [invocation_record.verify(path) for path in tmp_path.rglob("invocation.json")]
    assert any(row["stage"] == "native_accessor_decode" and row["kind"] == "subprocess" for row in records)
    assert sum(row["stage"] == "whole_linked_instruction_policy" for row in records) == 2 + ("additional_elf" in config)
    assert not any(row["stage"] == "execution" for row in records)
    assert "physical_cpu_equivalence" in results["positive"]["unknowns"]
    if "diagnostic_service_owner" in config:
        _exercise_fixed_runtime_instruction_control(check, hardware, tmp_path, config)
    # Any subsequent artifact mutation invalidates the static admission.
    elf = tmp_path / "positive.elf"
    original = elf.read_bytes()
    try:
        elf.write_bytes(original + b"changed")
        with pytest.raises(ValueError, match="ELF changed after admission"):
            service.revalidate(elf=elf, result=results["positive"], target=hardware.target)
    finally:
        elf.write_bytes(original)
    service.revalidate(elf=elf, result=results["positive"], target=hardware.target)


def _exercise_fixed_runtime_instruction_control(check, hardware, root, config):
    owner = Path(config["diagnostic_service_owner"])
    module_spec = importlib.util.spec_from_file_location("independent_runtime_control_test_owner", owner)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    build, execution = module.prepare_services(hardware.target)
    contract = root / "private-contract"
    (contract / "schemas").mkdir(parents=True)
    from merlin.common.paths import data_path

    for name in ("manifest.schema.json", "capsule.schema.json", "command_buffer.schema.json"):
        shutil.copyfile(data_path("contract") / "schemas" / name, contract / "schemas" / name)
    files = {Path(config["hardware"]["descriptor"]), *contract.rglob("*.json")}
    files.update(Path(path) for path, _ in (*build.source_pins, *execution.source_pins))
    files.update(path for path, _ in check.source_pins)
    files.update(
        Path(inspect.getsourcefile(value))
        for value in (
            runtime_support.PreparedIndependentRuntimeContext,
            runtime_support.controls.parse_primitive,
            runtime_support.execute_component,
            runtime_support.PrivateRuntimeControlExecutor,
            runtime_support.prepare_source_control,
            runtime_support.stage_products.collect,
        )
    )
    files.add(Path(runtime_support.instruction_control.__file__))
    context = runtime_support.PreparedIndependentRuntimeContext(
        hardware,
        Path(config["hardware"]["descriptor"]),
        contract,
        build,
        execution,
        tuple((path, file_digest(path)) for path in sorted(files)),
        instruction_check=check,
    )
    context.verify()
    positive = context.prepare_control("instruction_audit.positive", root / "runtime-positive")
    score = context.services.grade(**positive.grade_arguments)
    assert score["integrity_status"] == "clean" and score["per_capsule"][0]["numeric"] == "pass"
    with pytest.raises(StageGateError, match="remain UNKNOWN"):
        context.services.stage_verifier(**positive.witness_arguments)
    negative = context.prepare_control("instruction_audit.negative", root / "runtime-negative")
    with pytest.raises(RuntimeControlRefusal) as refused:
        context.services.grade(**negative.grade_arguments)
    assert refused.value.case_id == negative.case_id and refused.value.mechanism == "instruction_audit"
    assert all(file_digest(path) == digest for path, digest in refused.value.evidence_files)
    actual = mapping_file(negative.grade_arguments["runs_root"] / "private_runtime_control" / "result.json")
    assert actual["native"]["execution"] == "not_attempted" and "numeric_report" not in actual
    records = [invocation_record.verify(path) for path in negative.evidence_root.rglob("invocation.json")]
    assert set(negative.required_invocation_stages) <= {row["stage"] for row in records}
    assert any(row["stage"] == "elf" and row["kind"] == "subprocess" for row in records)
    assert not any(row["stage"] == "execution" for row in records)
    context.verify_control(negative)
    mutation = mapping_file(negative.evidence_root / "instruction_mutation.json")
    source = Path(mutation["source"])
    original = source.read_bytes()
    try:
        source.write_bytes(original + b"\n")
        with pytest.raises(StageGateError, match="source, native proof or policy changed"):
            context.verify_control(negative)
    finally:
        source.write_bytes(original)
    context.verify_control(negative)
