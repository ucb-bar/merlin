"""Large, stripped artifacts and a distinct data-defined ISA through the real broker."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest
from merlin_experiments.phase1.feedback import reference_analysis as analysis
from merlin_experiments.phase1.feedback import reference_experiments as reference
from test_reference_analysis_tool import _request, _world

from merlin.compile.analysis.compute_endpoints import roles_from_table
from merlin.compile.analysis.disassembly import RawInsn, tokenize_text
from merlin.compile.analysis.schedule_structure import analyze_normalized_commands
from merlin.kernels.instruction_comparison import compare_instruction_summaries, instruction_summary
from merlin.kernels.machine_evidence import machine_projection
from merlin.targetgen.rocc.normalized_operands import BoundOperandObserver


def observer(*, offset=20, opcode=43, name="FETCH_PANEL"):
    facts = dict(
        interfaces=[
            dict(
                name="register_bundle_layouts",
                bundles={
                    "Packet": dict(
                        width=64, fields={"extent": dict(offset=offset, width=5), "slot": dict(offset=0, width=8)}
                    )
                },
            )
        ]
    )
    protocol = dict(
        version=1,
        encoding=dict(semantic_class={4: name}),
        roles=dict(
            classes={
                name: dict(
                    fields=[
                        dict(name="height", operand="rs2", bundle="Packet", field="extent"),
                        dict(name="slot", operand="rs1", bundle="Packet", field="slot"),
                    ]
                )
            }
        ),
    )
    return BoundOperandObserver(
        target="array_fixture",
        protocol=protocol,
        facts=facts,
        raw_isa=dict(CUSTOM_OPCODE=opcode, FUNCT3=3, FUNCT_CLASS={4: "RAW_NAME"}),
        word_bits=64,
    )


def test_machine_fields_follow_selected_isa_not_command_names_or_literal_opcode():
    # LUI x6, 17<<20; selected custom4 has funct4, rs2=x6.
    rows = [
        RawInsn(0, "lui", [], "01100337"),
        RawInsn(4, ".insn", [], f"{4 << 25 | 6 << 20 | 3 << 12 | 43:08x}"),
        RawInsn(8, "ret", [], "00008067"),
    ]
    model = analyze_normalized_commands([], source_kind="static_disassembly")
    decoder = dict(
        semantics=observer(),
        target="array_fixture",
        structure=dict(instruction_roles={"FETCH_PANEL": ["operand_load"]}),
    )
    cca, result = machine_projection(rows, model=model, evidence_id="test", decoder=decoder)
    assert result["instruction_classes"] == {"FETCH_PANEL": 1}
    assert result["field_profiles"]["FETCH_PANEL.height"] == [dict(value=17, sites=1)]
    assert result["role_sites"] == {"operand_load": 1}
    assert cca.compute.contraction_form is None
    moved = {**decoder, "semantics": observer(offset=21, name="INPUT_RECTANGLE")}
    _, result = machine_projection(rows, model=model, evidence_id="test", decoder=moved)
    assert result["field_profiles"]["INPUT_RECTANGLE.height"] == [dict(value=8, sites=1)]
    _, result = machine_projection(
        rows, model=model, evidence_id="test", decoder={**decoder, "semantics": observer(opcode=11)}
    )
    assert not result["instruction_classes"]


def test_machine_repetition_and_boundaries_are_witnessed_without_claiming_reuse_legality():
    custom = f"{4 << 25 | 6 << 20 | 3 << 12 | 43:08x}"
    rows = [
        RawInsn(0, "lui", [], "01100337"),
        RawInsn(4, ".insn", [], custom),
        RawInsn(8, ".insn", [], custom),
        RawInsn(12, "fence", [], "0ff0000f"),
        RawInsn(16, ".insn", [], custom),
        RawInsn(20, "ret", [], "00008067"),
    ]
    model = analyze_normalized_commands([], source_kind="static_disassembly")
    decoder = dict(semantics=observer(name="ARBITRARY_NAME"), target="array_fixture",
                   structure=dict(instruction_roles={"ARBITRARY_NAME": ["operand_load"]}))
    _, report = machine_projection(rows, model=model, evidence_id="repetition", decoder=decoder)
    mechanisms = report["mechanism_observations"]
    assert mechanisms["scope"] == "listed_instruction_sites"
    assert mechanisms["ordering"] == "lexical_listing_not_dynamic_issue_trace"
    repeated = mechanisms["repeated_operand_groups"][0]
    assert repeated["sites"] == 3 and repeated["fully_recovered"] is True
    assert repeated["first_witness"]["pc"] == 4 and repeated["last_witness"]["pc"] == 16
    assert [phase["command_sites"] for phase in mechanisms["fence_delimited_groups"]] == [2, 1]
    assert mechanisms["fence_delimited_groups"][0]["end_boundary"]["pc"] == 12
    assert {field["field"] for field in mechanisms["invariant_fields"]} == {"height", "slot"}
    assert mechanisms["candidates"][0]["mechanism"] == "repeated_transfer_operand_fields"
    assert mechanisms["candidates"][0]["optimization_applicability"] == "unknown"
    assert mechanisms["physical_vs_useful_reduction"]["status"] == "unknown"
    # Exact same fields/names acquire no transfer interpretation without selected roles.
    _, unbound = machine_projection(rows, model=model, evidence_id="unbound",
                                   decoder={**decoder, "structure": {}})
    unbound = unbound["mechanism_observations"]
    assert not unbound["candidates"] and unbound["unmapped_instruction_role_sites"] == 3


def test_instruction_comparison_preserves_operand_immediates_and_registers():
    def view(word):
        rows = tokenize_text(f"100: {word} addi a0,a0,16\n104: 00008067 ret\n")
        return dict(instruction_view=instruction_summary(rows))

    same = compare_instruction_summaries(view("01050513"), view("01050513"))
    changed = compare_instruction_summaries(view("01050513"), view("02050513"))
    renamed = compare_instruction_summaries(view("01050513"), view("01058593"))
    assert same["same_words"] and same["same_ordered_structure"]
    assert not changed["same_ordered_structure"] and not renamed["same_ordered_structure"]
    assert changed["semantic_alignment"] == changed["execution_multiplicity"] == "unknown"


def test_endpoint_roles_bind_only_selected_isa_names_and_keep_missing_declarations():
    document = dict(
        endpoints=dict(
            array=dict(
                target="array_fixture",
                encoding=dict(source="rtl_facts"),
                roles=dict(operand_load=["NEW_FETCH"], accumulate=["OLD_COMPUTE"]),
            )
        )
    )
    table = dict(names={"4": "NEW_FETCH", "9": "NEW_COMPUTE"})
    roles, missing = roles_from_table(document, target="array_fixture", table=table)
    assert roles == {"NEW_FETCH": ["operand_load"]}
    assert missing == {"array": {"accumulate": ["OLD_COMPUTE"]}}
    assert roles_from_table(document, target="another_array", table=table) == ({}, {})


def test_recovered_operand_witnesses_page_with_instructions_and_digest_the_complete_stream():
    rows = [RawInsn(i * 4, "nop", [], "00000013") for i in range(12)]
    rows += [
        RawInsn(48, "lui", [], "01100337"),
        RawInsn(52, ".insn", [], f"{4 << 25 | 6 << 20 | 3 << 12 | 43:08x}"),
        RawInsn(56, "ret", [], "00008067"),
    ]
    decoder = dict(semantics=observer(), target="array_fixture")
    model = analyze_normalized_commands([], source_kind="static_disassembly")
    _, first = machine_projection(rows, model=model, evidence_id="page", decoder=decoder)
    _, second = machine_projection(rows, model=model, evidence_id="page", decoder=decoder, instruction_offset=13)
    assert not first["command_witnesses"]
    assert first["command_page"]["next_instruction_offset"] == 13
    assert second["command_witnesses"][0]["fields"]["height"] == 17
    assert first["operand_sequence_sha256"] == second["operand_sequence_sha256"]


@pytest.mark.parametrize(
    "format,reason", [("source_mlir", "analysis_source_parse_failed"), ("emitted_mlir", "analysis_emitted_invalid")]
)
def test_malformed_ir_is_a_bounded_adapter_gap(tmp_path, format, reason):
    fixture = _world(tmp_path)
    (tmp_path / "selfcheck_out").mkdir()
    (tmp_path / "selfcheck_out/broken.mlir").write_text("module { invalid syntax ! }")
    with reference.invocation(fixture["path"], fixture["sha"]):
        result = analysis.handle(_request("candidate", "selfcheck_out/broken.mlir", format=format), workspace=tmp_path)
    assert result["status"] == "unknown" and result["reason"] == reason


def test_emitted_view_keeps_real_counted_work_separate_from_source_binding(tmp_path):
    fixture = _world(tmp_path)
    (tmp_path / "selfcheck_out").mkdir()
    (tmp_path / "selfcheck_out/llvm.mlir").write_text("builtin.module { llvm.func @kernel_entry() { llvm.return } }")
    with reference.invocation(fixture["path"], fixture["sha"]):
        result = analysis.handle(
            _request("candidate", "selfcheck_out/llvm.mlir", format="emitted_mlir"), workspace=tmp_path
        )
    assert result["status"] == "analyzed", result
    assert result["report"]["counted_work"]["status"] == "observed_counted_cfg"
    assert result["report"]["source_correspondence"] == "unknown"


def test_emitted_symbol_selects_actual_function_and_refuses_missing_symbols(tmp_path, monkeypatch):
    fixture = _world(tmp_path)
    (tmp_path / "selfcheck_out").mkdir()
    (tmp_path / "selfcheck_out/functions.mlir").write_text("""module {
      llvm.func @kernel_entry() { llvm.return }
      llvm.func @kernel_complete() {
        llvm.br ^done
      ^done:
        "llvm.inline_asm"() <{asm_string = "fence", constraints = "", has_side_effects}> : () -> ()
        llvm.return
      }
    }""")
    monkeypatch.setattr(
        analysis, "_operand_decoder", lambda *args, **kwargs: dict(semantics=observer(), target="array_fixture")
    )
    with reference.invocation(fixture["path"], fixture["sha"]):
        for symbol, count in (("kernel_entry", 0), ("kernel_complete", 1)):
            result = analysis.handle(
                _request("candidate", "selfcheck_out/functions.mlir", format="emitted_mlir", symbol=symbol),
                workspace=tmp_path,
            )
            assert result["status"] == "analyzed", result
            report = result["report"]
            assert report["functions"] == [symbol] and report["scope"] == "selected_function"
            assert report["operand_summary"]["commands"] == count
        missing = analysis.handle(
            _request("candidate", "selfcheck_out/functions.mlir", format="emitted_mlir", symbol="missing"),
            workspace=tmp_path,
        )
        assert missing["status"] == "unknown" and missing["reason"] == "analysis_symbol_not_unique"


def test_equal_count_variants_expose_panel_storage_and_issue_order_in_real_emitted_ir(tmp_path, monkeypatch):
    fixture = _world(tmp_path)
    (tmp_path / "selfcheck_out").mkdir()
    monkeypatch.setattr(
        analysis, "_operand_decoder", lambda *args, **kwargs: dict(semantics=observer(), target="array_fixture")
    )

    def emitted(values):
        body = []
        for index, (slot, height) in enumerate(values):
            body += [
                f"%s{index} = llvm.mlir.constant({slot} : i64) : i64",
                f"%h{index} = llvm.mlir.constant({height << 20} : i64) : i64",
                f'"llvm.inline_asm"(%s{index}, %h{index}) <{{asm_string = '
                '".insn r 0x2b, 3, 4, x0, $0, $1", constraints = "r,r", has_side_effects}> '
                ": (i64, i64) -> ()",
            ]
        return "module { llvm.func @kernel_entry() { " + "\n".join(body) + "\nllvm.return } }"

    # The last field change falls beyond both the four displayed values and
    # the sixteen retained profile values, while the complete ISA stream fits.
    original = [(16 + index, index) for index in range(20)]
    variants = dict(
        base=original,
        panel=[*original[:-1], (original[-1][0], 30)],
        storage=[*original[:-1], (90, original[-1][1])],
        reordered=list(reversed(original)),
    )
    for name, values in variants.items():
        (tmp_path / f"selfcheck_out/{name}.mlir").write_text(emitted(values))
    with reference.invocation(fixture["path"], fixture["sha"]):
        reports = {}
        for name in variants:
            response = analysis.handle(
                _request(
                    "compare",
                    "selfcheck_out/base.mlir",
                    format="emitted_mlir",
                    candidate=dict(path=f"selfcheck_out/{name}.mlir", format="emitted_mlir"),
                ),
                workspace=tmp_path,
            )
            assert response["status"] == "analyzed", response
            reports[name] = response["report"]
    assert reports["base"]["operand_comparison"]["same_recovered_operands"] is True
    assert (reports["panel"]["reference"]["operand_summary"]["field_profiles"] ==
            reports["panel"]["candidate"]["operand_summary"]["field_profiles"])
    assert any(
        row["field"] == "FETCH_PANEL.height"
        for row in reports["panel"]["operand_comparison"]["operand_field_differences"]
    )
    assert any(
        row["field"] == "FETCH_PANEL.slot"
        for row in reports["storage"]["operand_comparison"]["operand_field_differences"]
    )
    reordered = reports["reordered"]["operand_comparison"]
    assert reordered["same_recovered_operand_population"] is True
    assert reordered["order_change"] == "reordered_same_recovered_operands"
    assert not reordered["operand_field_differences"]
    assert all(report["semantic_compatibility"] == report["performance"] == "unknown" for report in reports.values())


def test_oversized_comparison_retains_complete_evidence_and_published_tasks(tmp_path):
    from merlin_experiments.phase1.feedback.result_details import load_row

    fixture = _world(tmp_path)
    (tmp_path / "selfcheck_out").mkdir()
    functions = ["llvm.func @f" + str(i) + "x" * 200 + "() { llvm.return }" for i in range(400)]
    (tmp_path / "selfcheck_out/large.mlir").write_text("module {" + "\n".join(functions) + "}")
    with reference.invocation(fixture["path"], fixture["sha"]):
        response = analysis.handle(
            _request(
                "compare",
                "selfcheck_out/large.mlir",
                format="emitted_mlir",
                candidate=dict(path="selfcheck_out/large.mlir", format="emitted_mlir"),
            ),
            workspace=tmp_path,
            compiler_sha256=fixture["compiler"],
        )
        assert response["status"] == "analyzed", response
        assert len(json.dumps(response).encode()) < 64 << 10
        assert response["report_in_detail"] is True and response["detail"]["path"].startswith("qa/cases/")
        full = load_row(response, workspace=tmp_path)
        assert len(full["report"]["reference"]["functions"]) == 400
        assert full["report"]["experiment"]["tasks"]
        assert full["report_identity"] == response["report_identity"]
        assert full["qualification_issued"] is False and full["all_pass"] is None


def _elf(tmp_path):
    from merlin.compile.analysis.disassembly import objdump_bin

    clang = shutil.which("clang")
    if not clang or Path(objdump_bin()).name == "objdump":
        pytest.skip("RISC-V compiler and disassembler required")
    source = tmp_path / "program.S"
    source.write_text("""
.section .text,"ax",@progbits
.globl _start
_start:
  .insn 4, 0x0200307b
  ret
  .insn 4, 0x0200307b
  .insn 4, 0x0200307b
  ret
""")
    output = tmp_path / "reference-kernels/program.elf"
    subprocess.run(
        [clang, "--target=riscv64-unknown-elf", "-march=rv64gc", "-c", str(source), "-o", str(output)],
        check=True,
        capture_output=True,
        timeout=30,
    )
    strip = shutil.which("llvm-strip")
    if not strip:
        pytest.skip("LLVM strip required")
    subprocess.run([strip, "--strip-all", str(output)], check=True, capture_output=True, timeout=30)
    # Model weights make the actual file large without bloating the test.
    with output.open("ab") as stream:
        stream.truncate((256 << 20) + 4096)
    return output


def test_large_stripped_elf_inventory_region_selection_and_broker_comparison(tmp_path, monkeypatch):
    fixture = _world(tmp_path)
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    elf = _elf(tmp_path)
    (tmp_path / "selfcheck_out").mkdir()
    # Hard links retain the exact >256 MiB bytes; neither analysis copies weights.
    (tmp_path / "selfcheck_out/candidate.elf").hardlink_to(elf)
    with reference.invocation(fixture["path"], fixture["sha"]):
        inventory = analysis.handle(
            _request("inventory", "reference-kernels/program.elf"), workspace=tmp_path, facts=fixture["facts"]
        )
        assert inventory["status"] == "analyzed", inventory
        report = inventory["report"]
        assert report["coverage"]["accelerator_sites"] == report["coverage"]["selected_region_sites"] == 3
        assert report["total"] == 2
        selected = report["regions"][1]["stable_id"]
        pair = analysis.handle(
            _request(
                "compare",
                "reference-kernels/program.elf",
                region=selected,
                candidate=dict(path="selfcheck_out/candidate.elf", region=selected),
            ),
            workspace=tmp_path,
            facts=fixture["facts"],
        )
        assert pair["status"] == "analyzed", pair
        assert pair["report"]["instruction_comparison"]["same_words"]
        assert pair["report"]["semantic_compatibility"] == "unknown"
        invalid = analysis.handle(
            _request("lift", "reference-kernels/program.elf", region="wrong"),
            workspace=tmp_path,
            facts=fixture["facts"],
        )
        assert invalid["reason"] == "analysis_region_not_unique"
    assert elf.stat().st_size > 256 << 20


def test_open_elf_detects_truncation_without_mapping_mutable_bytes(tmp_path):
    from merlin.compile.analysis.artifact_reader import PinnedELF, open_ordinary

    (tmp_path / "reference-kernels").mkdir()
    elf = _elf(tmp_path)
    artifact = PinnedELF(open_ordinary(elf), "reference-kernels/program.elf")
    try:
        with elf.open("r+b") as stream:
            stream.truncate(0)
        with pytest.raises(ValueError, match="analysis_input_changed"):
            artifact.check()
        with pytest.raises(ValueError, match="analysis_input_changed"):
            artifact.read_at(0, 64)
    finally:
        artifact.close(check=False)


def test_v2_representation_bundle_reopens_each_pin_and_preserves_unknown_join(tmp_path):
    fixture = _world(tmp_path)
    (tmp_path / "selfcheck_out").mkdir()
    source = tmp_path / "selfcheck_out/original.mlir"
    source.write_text("""module {
      func.func @f(%a: tensor<2xi32>, %b: tensor<2xi32>) -> tensor<2xi32> {
        %c = arith.addi %a, %b : tensor<2xi32>
        func.return %c : tensor<2xi32>
      }
    }""")
    binary = tmp_path / "selfcheck_out/code.dis"
    binary.write_text("100: 0200307b .insn 4,0x0200307b\n104: 00008067 ret\n")

    def pin(path, format):
        return dict(
            path=path.relative_to(tmp_path).as_posix(),
            format=format,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )

    document = dict(
        schema="merlin.reference_artifact_views.v2",
        views=dict(source=pin(source, "source_mlir"), binary=pin(binary, "disassembly")),
    )
    manifest = tmp_path / "selfcheck_out/bundle.json"
    manifest.write_text(json.dumps(document))
    with reference.invocation(fixture["path"], fixture["sha"]):
        first = analysis.handle(
            _request("candidate", "selfcheck_out/bundle.json"), workspace=tmp_path, facts=fixture["facts"]
        )
        assert first["status"] == "analyzed", first
        assert first["report"]["views"]["source"]["semantic_region"]["operations"]
        assert first["report"]["emitted_correspondence"] == "unknown"
        source.write_text(source.read_text() + "\n")
        stale = analysis.handle(
            _request("candidate", "selfcheck_out/bundle.json"), workspace=tmp_path, facts=fixture["facts"]
        )
        assert stale["reason"] == "analysis_artifact_pin_mismatch"
