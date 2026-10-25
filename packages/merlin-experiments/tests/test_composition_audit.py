"""Composition credit needs source identities and the mechanism in that region."""

import copy
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from merlin.targetgen import composition_audit as CA
from merlin.targetgen import frozen_placement as FP
from merlin.targetgen import native_instruction_audit as NA
from merlin.targetgen import trace_check as TC
from merlin.targetgen.rocc import decode
from merlin.targetgen.rtl import facts

_SOURCE_SHA = hashlib.sha256(b"source").hexdigest()


def _bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _row(kind, **decoded):
    if kind == "FENCE":
        decoded = {"pred": 15, "succ": 15, "fm": 0, **decoded}
    elif kind == "MVOUT":
        decoded = {"acc_addr": 32, "rows": 3, "cols": 3, "readout": "i8", "is_acc_addr": True, **decoded}
    elif kind == "PRELOAD":
        decoded = {"is_acc_addr": True, **decoded}
    elif kind in {"MVIN", "MVIN2", "MVIN3"}:
        decoded = {"spad_addr": 32, "rows": 3, "cols": 3, **decoded}
    return {"class": kind, "decoded": decoded, "kernel": "fused", "region_id": "chain"}


def _compute(count=1, **decoded):
    return [
        row
        for _ in range(count)
        for row in (
            _row("PRELOAD", c_addr=32, c_rows=3, c_cols=3),
            _row("COMPUTE_PRELOADED", a_spad=8, **decoded),
        )
    ]


def _operand(index):
    return {"kind": "argbase", "arg_index": index, "offset": 0}


@pytest.fixture(autouse=True)
def selected_hardware(monkeypatch):
    # Deliberately moved select codes/state IDs/dimensions expose baked values.
    document = {
        "facts": {
            "timing_structure": {
                "dataflow": {
                    "status": "derived",
                    "modes": {"9": "output_stationary", "4": "weight_stationary"},
                }
            }
        }
    }
    monkeypatch.setattr(facts, "load_facts", lambda _: copy.deepcopy(document))
    semantics = SimpleNamespace(
        load_states=SimpleNamespace(ports=(("MVIN", 6), ("MVIN2", 3))),
        isa_constants=lambda _: {
            "DIM": 3,
            "OPERAND_FIELDS": {
                "PRELOAD": {"fields": [{"as": "flag", "mask": 512}]},
            },
        },
    )
    monkeypatch.setattr(decode, "_semantics", lambda _: semantics)
    monkeypatch.setattr(TC, "_bound_trace_view", lambda *_: None)
    return document, semantics


def _case(rows, modes, parameters=None):
    expected = {
        "composition": {
            "producer": "contraction",
            "source_op_indices": [0, 1, 2],
            "producer_source_op_indices": [0],
            "stages": {"readout": [1, 2]},
        },
        "stage_scoped_modes": {"readout": modes},
        "mode_parameters": parameters or {},
    }
    evidence = {
        "source_sha256": _SOURCE_SHA,
        "routed_source_sha256": _SOURCE_SHA,
        "frozen_placement": {
            "program_sha256": _SOURCE_SHA,
            "source_placements": {0: "accelerator", 1: "accelerator", 2: "accelerator"},
        },
        "routed": [{"symbol": "fused", "region_id": "chain", "source_op_indices": [0, 1, 2]}],
        "host_intermediate_consumers": [],
        "native_instruction_audit": {
            "status": "pass",
            "pass_": True,
            "outside_kernel_instructions": 0,
            "kernels": [{"symbol": "fused", "native_instructions": len(rows), "bound_roots": 1, "observed_entries": 1}],
        },
    }
    trace = {"instructions": [_row("FENCE"), *rows, _row("FENCE")], "composition_evidence": evidence}
    return trace, expected


def _readout_case():
    rows = [
        _row("CONFIG_ST", acc_scale_bits=_bits(0.375), acc_scale=0.375, relu=True),
        *_compute(),
        _row("MVOUT", dram=_operand(1)),
    ]
    return _case(
        rows,
        {"readout_multipliers": True, "relu": True, "acc_scale": True},
        {"readout_multipliers": {"acc_scale_bits": [_bits(0.375)]}},
    )


def _check(case):
    return TC.check(*case, target="synthetic", alternatives={})


def test_fused_scaled_relu_and_exact_multiplier_bits():
    case = _readout_case()
    assert _check(case)["status"] == "pass"
    case[1]["mode_parameters"]["readout_multipliers"]["acc_scale_bits"] = [_bits(0.5)]
    assert _check(case)["status"] == "fail"


@pytest.mark.parametrize(
    "changes",
    [
        {"acc_addr": 48},
        {"acc_addr": 34, "rows": 2},
        {"cols": 4},
        {"readout": "i32"},
    ],
)
def test_readout_must_consume_contraction_rows_in_narrow_mode(changes):
    trace, expected = _readout_case()
    trace["instructions"][-2]["decoded"].update(changes)
    assert _check((trace, expected))["status"] == "fail"


def test_missing_readout_lineage_is_unknown():
    trace, expected = _readout_case()
    del trace["instructions"][-2]["decoded"]["acc_addr"]
    result = _check((trace, expected))
    assert result["status"] == "fail"
    assert result["composition_audit"]["evidence_status"] == "UNKNOWN"


@pytest.mark.parametrize("kind", ["PRELOAD", "MVOUT"])
def test_scratchpad_address_cannot_supply_accumulator_lineage(kind):
    trace, expected = _readout_case()
    next(row for row in trace["instructions"] if row["class"] == kind)["decoded"]["is_acc_addr"] = False
    assert _check((trace, expected))["status"] == "fail"


def test_readout_lineage_also_applies_to_standalone_trace_modes():
    trace, expected = _readout_case()
    expected.pop("composition")
    expected["modes"] = expected.pop("stage_scoped_modes")["readout"]
    trace["instructions"][-2]["decoded"]["acc_addr"] = 48
    assert _check((trace, expected))["status"] == "fail"


@pytest.mark.parametrize("privilege", [0, 1, 3])
def test_retirement_parser_requires_privilege_and_excludes_disassembly(privilege):
    assert NA._trace_word(f"core 0: {privilege} 0x8000 (0x0a00002b)".encode()) == (0x8000, 0x0A00002B)
    assert NA._trace_word(b"core 0: 0x8000 (0x0a00002b) custom") is None


@pytest.mark.parametrize(
    "line",
    [
        b"core 0: 3 bad-pc (0x0a00002b)",
        b"core 0: 3 0x8000 bad-word",
        b"core bad: 3 0x8000 (0x0a00002b)",
        b"core 0: 3 0x8000",
    ],
)
def test_malformed_committed_records_fail_closed(line):
    with pytest.raises(ValueError, match="UNKNOWN"):
        NA._trace_word(line)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda t: t["composition_evidence"]["frozen_placement"]["source_placements"].update({2: "host"}),
        lambda t: t["composition_evidence"].update(host_intermediate_consumers=[2]),
        lambda t: t["composition_evidence"].pop("host_intermediate_consumers"),
        lambda t: t["composition_evidence"]["routed"][0].update(symbol="unrelated"),
        lambda t: t["composition_evidence"]["routed"].append(
            {"symbol": "fused", "region_id": "other", "source_op_indices": [2]}
        ),
        lambda t: t["composition_evidence"]["routed"][0].update(source_op_indices=[0, 1]),
        lambda t: t["composition_evidence"]["native_instruction_audit"].update(outside_kernel_instructions=1),
        lambda t: t["composition_evidence"]["native_instruction_audit"]["kernels"][0].update(observed_entries=0),
        lambda t: t["instructions"][1].update(kernel="decoy"),
        lambda t: t["instructions"][1].update(region_id="decoy"),
    ],
)
def test_host_split_wrong_kernel_and_count_matching_cannot_credit_fusion(mutate):
    trace, expected = _readout_case()
    mutate(trace)
    assert _check((trace, expected))["status"] == "fail"


def test_overwritten_setup_and_separate_activation_cannot_match_counts():
    trace, expected = _readout_case()
    trace["instructions"].insert(2, _row("CONFIG_ST", acc_scale_bits=_bits(0.375), acc_scale=0.375, relu=False))
    assert _check((trace, expected))["status"] == "fail"


@pytest.mark.parametrize("field", ["acc_scale_bits", "acc_scale", "relu"])
def test_missing_fused_readout_fields_are_unknown(field):
    trace, expected = _readout_case()
    del trace["instructions"][1]["decoded"][field]
    result = _check((trace, expected))
    assert result["status"] == "fail"
    assert result["composition_audit"]["evidence_status"] == "UNKNOWN"


def test_scaled_accumulate_load_uses_the_active_config_for_its_derived_state():
    rows = [
        _row("CONFIG_LD", state_id=6, scale_bits=_bits(0.625)),
        _row("CONFIG_LD", state_id=3, scale_bits=_bits(0.25)),
        _row("MVIN", is_acc_addr=True, accumulate=True),
        *_compute(),
        _row("MVOUT"),
    ]
    case = _case(rows, {"scaled_accumulate_load": True}, {"scaled_accumulate_load": {"scale_bits": [_bits(0.625)]}})
    assert _check(case)["status"] == "pass"
    case[0]["instructions"][3]["decoded"]["accumulate"] = False
    assert _check(case)["status"] == "fail"


def test_scaled_load_cannot_borrow_scale_from_another_state():
    rows = [
        _row("CONFIG_LD", state_id=3, scale_bits=_bits(0.625)),
        _row("MVIN", is_acc_addr=True, accumulate=True),
        *_compute(),
        _row("MVOUT"),
    ]
    result = _check(
        _case(rows, {"scaled_accumulate_load": True}, {"scaled_accumulate_load": {"scale_bits": [_bits(0.625)]}})
    )
    assert result["status"] == "fail"
    assert result["composition_audit"]["evidence_status"] == "UNKNOWN"


def test_scaled_operand_sum_does_not_require_a_contraction():
    rows = [
        _row("CONFIG_LD", state_id=6, scale_bits=_bits(0.625)),
        _row("MVIN", is_acc_addr=True, accumulate=True),
        _row("MVOUT"),
    ]
    case = _case(rows, {"scaled_accumulate_load": True}, {"scaled_accumulate_load": {"scale_bits": [_bits(0.625)]}})
    case[1]["composition"]["producer"] = "operand_sum"
    assert _check(case)["status"] == "pass"
    case[0]["instructions"][2]["decoded"]["accumulate"] = False
    assert _check(case)["status"] == "fail"


def test_pool_window_is_effective_at_store_and_missing_fields_fail_closed():
    fields = dict(pool_stride=2, pool_size=3, pool_out_dim=19, porows=7, pocols=8, orows=13, ocols=15, upad=1, lpad=2)
    case = _case(
        [_row("CONFIG_ST", **fields), *_compute(), _row("MVOUT")],
        {"pooled_store": True},
        {"pooled_store": {"fields": fields}},
    )
    assert _check(case)["status"] == "pass"
    case[0]["instructions"][1]["decoded"].pop("pool_size")
    assert _check(case)["status"] == "fail"


def test_standalone_pooling_chain_can_have_a_movement_producer():
    fields = dict(pool_stride=2, pool_size=3, pool_out_dim=19, porows=7, pocols=8, orows=13, ocols=15, upad=1, lpad=2)
    case = _case(
        [_row("CONFIG_ST", **fields), _row("MVIN"), _row("MVOUT")],
        {"pooled_store": True},
        {"pooled_store": {"fields": fields}},
    )
    case[1]["composition"]["producer"] = "movement"
    assert _check(case)["status"] == "pass"
    case[1]["composition"]["producer"] = "contraction"
    assert _check(case)["status"] == "fail"


def test_output_stationary_is_derived_and_applies_to_every_compute(selected_hardware):
    case = _case([_row("CONFIG_EX", dataflow=9), *_compute(), _row("MVOUT")], {"output_stationary": True})
    assert _check(case)["status"] == "pass"
    selected_hardware[0]["facts"]["timing_structure"]["dataflow"]["modes"] = {"4": "output_stationary"}
    assert _check(case)["status"] == "fail"
    selected_hardware[0]["facts"]["timing_structure"]["dataflow"].pop("modes")
    assert _check(case)["composition_audit"]["evidence_status"] == "UNKNOWN"


def test_i32_exit():
    case = _case([*_compute(), _row("MVOUT", readout="i32")], {"i32_exit": True})
    assert _check(case)["status"] == "pass"
    case[0]["instructions"][-2]["decoded"]["readout"] = "i8"
    assert _check(case)["status"] == "fail"


def test_extra_k_tiles_are_required_per_output_not_as_a_global_count():
    case = _case(
        [*_compute(3), _row("MVOUT", acc_addr=32), *_compute(3), _row("MVOUT", acc_addr=32)],
        {"k_extension": True},
        {"k_extension": {"source_k": 6, "extension_k": 3}},
    )
    assert _check(case)["status"] == "pass"
    # Same total count, but a producer output lacks its coefficient-digit work.
    rows = [*_compute(2), _row("MVOUT", acc_addr=32), *_compute(4), _row("MVOUT", acc_addr=32)]
    assert _check(_case(rows, case[1]["stage_scoped_modes"]["readout"], case[1]["mode_parameters"]))["status"] == "fail"


def test_k_tile_counts_cannot_borrow_work_for_another_accumulator():
    rows = [*_compute(3), _row("MVOUT", acc_addr=48)]
    case = _case(rows, {"k_extension": True}, {"k_extension": {"source_k": 6, "extension_k": 3}})
    assert _check(case)["status"] == "fail"


def test_horizontal_outputs_share_one_resident_input_load_set():
    load = _row("MVIN", spad_addr=8, rows=3, cols=3, dram=_operand(0))
    rows = [load, *_compute(), _row("MVOUT", dram=_operand(1)), *_compute(), _row("MVOUT", dram=_operand(2))]
    case = _case(rows, {"shared_input": True}, {"shared_input": {"input": _operand(0)}})
    assert _check(case)["status"] == "pass"
    case[0]["instructions"].insert(5, copy.deepcopy(load))
    assert _check(case)["status"] == "fail"


def test_partially_overwritten_resident_input_cannot_credit_horizontal_fusion():
    rows = [
        _row("MVIN", spad_addr=8, rows=3, cols=3, dram=_operand(0)),
        *_compute(),
        _row("MVOUT", dram=_operand(1)),
        _row("MVIN", spad_addr=9, rows=1, cols=3, dram=_operand(3)),
        *_compute(),
        _row("MVOUT", dram=_operand(2)),
    ]
    case = _case(rows, {"shared_input": True}, {"shared_input": {"input": _operand(0)}})
    assert _check(case)["status"] == "fail"


def test_column_block_requires_reissue_even_for_identical_neighbor_scales():
    config = _row("CONFIG_ST", acc_scale_bits=_bits(0.375))
    rows = [config, *_compute(), _row("MVOUT"), copy.deepcopy(config), *_compute(), _row("MVOUT")]
    case = _case(
        rows,
        {"column_block": True},
        {"column_block": {"acc_scale_bits": [_bits(0.375)] * 2, "stores_per_block": [1, 1]}},
    )
    assert _check(case)["status"] == "pass"
    del case[0]["instructions"][5]
    assert _check(case)["status"] == "fail"


def test_new_modes_also_check_standalone_traces_and_missing_composition_evidence():
    trace, expected = _readout_case()
    assert _check(({"instructions": trace["instructions"]}, expected))["status"] == "fail"
    expected.pop("composition")
    expected["modes"] = expected.pop("stage_scoped_modes")["readout"]
    assert _check((trace, expected))["status"] == "pass"
    trace["instructions"][1]["decoded"].pop("acc_scale_bits")
    assert _check((trace, expected))["status"] == "fail"


def test_column_block_can_keep_its_configuration_across_multiple_row_stores():
    config = _row("CONFIG_ST", acc_scale_bits=_bits(0.375))
    rows = [
        config,
        *_compute(),
        _row("MVOUT"),
        *_compute(),
        _row("MVOUT"),
        copy.deepcopy(config),
        *_compute(),
        _row("MVOUT"),
        *_compute(),
        _row("MVOUT"),
    ]
    case = _case(
        rows,
        {"column_block": True},
        {
            "column_block": {"acc_scale_bits": [_bits(0.375)] * 2, "stores_per_block": [2, 2]},
        },
    )
    assert _check(case)["status"] == "pass"
    del case[0]["instructions"][8]
    assert _check(case)["status"] == "fail"


def test_frozen_source_identity_inventory_is_retained_and_duplicate_claims_refuse(tmp_path):
    program = tmp_path / "interface.mlir"
    program.write_text("module {}")
    digest = hashlib.sha256(program.read_bytes()).hexdigest()
    row = {
        "placement": "accelerator",
        "mlir_operation": "linalg.generic",
        "count": 3,
        "status": "admitted",
        "source_op_indices": [0, 1, 2],
    }
    declaration = {
        "interface_mlir": program.name,
        "software_screen": {
            "status": "admitted",
            "program_sha256": digest,
            "decisions": [row],
            "placements": {"accelerator": 1},
        },
    }
    assert FP.read(declaration, tmp_path)["source_placements"] == dict.fromkeys(range(3), "accelerator")
    row["source_op_indices"] = [0, 1, 1]
    with pytest.raises(FP.FrozenPlacementRefusal):
        FP.read(declaration, tmp_path)


@pytest.fixture
def native_files(tmp_path):
    compiler = shutil.which("cc")
    if compiler is None or shutil.which("readelf") is None:
        pytest.skip("ELF fixture requires cc and readelf")
    assembly, elf, trace = (tmp_path / name for name in ("kernel.s", "kernel.o", "retired.log"))
    assembly.write_text(".text\n.globl fused\n.type fused,@function\nfused:\n.long 0\n.size fused, .-fused\n")
    subprocess.run([compiler, "-c", str(assembly), "-o", str(elf)], check=True)
    return elf, trace


@pytest.mark.parametrize("opcode", [43, 91])
def test_in_repo_counter_derives_opcode_and_counts_outside_instructions(native_files, opcode):
    elf, trace = native_files
    document = {
        "facts": {"interfaces": [{"name": "funct_decode_table", "custom_opcode": opcode, "names": {"5": "WORK"}}]}
    }
    word = (5 << 25) | opcode
    trace.write_text(f"core 0: 3 0x0 (0x{word:08x})\n")
    arguments = dict(elf=elf, trace=trace, routed=[{"symbol": "fused"}], facts=document)
    result = NA.audit_case_instructions(**arguments)
    assert result["status"] == "pass" and result["outside_kernel_instructions"] == 0
    assert result["kernels"][0]["functs"] == {"5": 1}
    trace.write_text(trace.read_text() + f"core 0: 3 0x100 (0x{word:08x})\n")
    result = NA.audit_case_instructions(**arguments)
    assert result["status"] == "fail" and result["outside_kernel_instructions"] == 1
    del document["facts"]["interfaces"][0]["custom_opcode"]
    assert NA.audit_case_instructions(**arguments)["status"] == "UNKNOWN"


def test_counter_refuses_unknown_commands_absent_entries_and_malformed_traces(native_files):
    elf, trace = native_files
    arguments = dict(
        elf=elf,
        trace=trace,
        routed=[{"symbol": "fused"}],
        facts={
            "interfaces": [{"name": "funct_decode_table", "custom_opcode": 43, "names": {"5": "WORK"}}],
        },
    )
    trace.write_text("core 0: 3 0x0 (0x0000002b)\n")
    assert NA.audit_case_instructions(**arguments)["status"] == "fail"
    trace.write_text("")
    assert NA.audit_case_instructions(**arguments)["status"] == "fail"
    trace.write_text("core 0: 3 bad-pc (0x0000002b)\n")
    assert NA.audit_case_instructions(**arguments)["status"] == "UNKNOWN"


def test_native_composition_reopens_pins_and_never_accepts_a_cached_pass(tmp_path):
    from merlin.targetgen.native_model_execution import NativeModelExecutionError

    with pytest.raises(NativeModelExecutionError):
        NA.audit_native_composition(
            {"composition_audit": {"status": "pass"}}, policy={"composition_expected": {}}, facts={}, target="synthetic"
        )


@pytest.fixture
def composition_execution_inputs(tmp_path, monkeypatch, native_files):
    import yaml

    from merlin.targetgen.native_model_execution import _digest

    elf, retired = native_files
    trace, expected = _readout_case()
    codes = {"CONFIG_ST": 7, "PRELOAD": 5, "COMPUTE_PRELOADED": 9, "MVOUT": 3}
    decoded_fields = {}
    words = []
    for row in trace["instructions"]:
        if row["class"] == "FENCE":
            words.append(15)  # Standard RISC-V fence, not a device opcode.
            continue
        code = codes[row["class"]]
        decoded_fields[code] = (row["class"], copy.deepcopy(row["decoded"]))
        word = (code << 25) | 43
        row.update(
            pc=len(words) * 4,
            word=word,
            funct=code,
            rs1={"kind": "const", "raw": 0},
            rs2={"kind": "const", "raw": 0},
        )
        # The original decoder must ignore these volunteered field claims.
        row["decoded"] = {"relu": False}
        words.append(word)
    assembly = tmp_path / "replay.s"
    assembly.write_text(
        ".text\n.globl fused\n.type fused,@function\nfused:\n"
        + "".join(f".long {word}\n" for word in words)
        + ".size fused, .-fused\n"
    )
    subprocess.run([shutil.which("cc"), "-c", str(assembly), "-o", str(elf)], check=True)
    retired.write_text("".join(f"core 0: 3 0x{index * 4:x} (0x{word:08x})\n" for index, word in enumerate(words)))
    trace["retirement_trace_sha256"] = hashlib.sha256(retired.read_bytes()).hexdigest()
    trace.pop("composition_evidence")
    monkeypatch.setattr(
        decode,
        "_semantics",
        lambda _: SimpleNamespace(
            isa_constants=lambda _: {
                "CUSTOM_OPCODE": 43,
                "OPERAND_FIELDS": {
                    "PRELOAD": {"fields": [{"as": "flag", "mask": 512}]},
                },
            },
            decode_instruction=lambda code, rs1, rs2, isa: decoded_fields[code],
        ),
    )
    source = tmp_path / "interface.mlir"
    source.write_text("""module {
      func.func @forward(%a: i32, %b: i32) -> i32 {
        %0 = arith.addi %a, %b : i32
        %1 = arith.addi %0, %b : i32
        %2 = arith.addi %1, %b : i32
        func.return %2 : i32
      }
    }""")
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    declaration = {
        "interface_mlir": source.name,
        "expected": expected,
        "software_screen": {
            "status": "admitted",
            "program_sha256": source_sha,
            "placements": {"accelerator": 1},
            "decisions": [
                {
                    "status": "admitted",
                    "placement": "accelerator",
                    "mlir_operation": "linalg.generic",
                    "count": 3,
                    "source_op_indices": [0, 1, 2],
                }
            ],
        },
    }
    capsule = tmp_path / "capsule.yaml"
    capsule.write_text(yaml.safe_dump(declaration))
    bindings = [{"symbol": "fused", "region_id": "chain", "source_op_indices": [0, 1, 2]}]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"source_sha256": source_sha, "bindings": bindings}))
    documents = {
        "region_offload": {
            "routed": bindings,
            "manifest": str(manifest),
            "artifact_sha256": {"manifest": hashlib.sha256(manifest.read_bytes()).hexdigest()},
        },
        "host_chain_audit": {"status": "verified", "source_sha256": source_sha, "intermediate_consumers": []},
        "composition_trace": trace,
    }
    native = {"elf": _digest(elf), "retirement_trace": _digest(retired)}
    for role, document in documents.items():
        path = tmp_path / f"{role}.json"
        path.write_text(json.dumps(document))
        native[role] = _digest(path)
    policy = {"capsule_declaration": _digest(capsule), "composition_expected": expected}
    facts_document = {
        "interfaces": [
            {
                "name": "funct_decode_table",
                "custom_opcode": 43,
                "names": {str(code): name for name, code in codes.items()},
            }
        ]
    }
    return native, policy, facts_document, trace


def test_native_composition_replays_pinned_source_manifest_and_executed_sequence(
    composition_execution_inputs, tmp_path
):
    from merlin.targetgen.native_model_execution import _digest

    native, policy, facts_document, trace = composition_execution_inputs
    result = NA.audit_native_composition(native, policy=policy, facts=facts_document, target="synthetic")
    assert result["trace_check"]["status"] == "pass"
    # A byte-pinned but reordered decode cannot reuse an otherwise clean counter.
    trace["instructions"][1]["pc"] = 8
    path = tmp_path / "composition_trace.json"
    path.write_text(json.dumps(trace))
    native["composition_trace"] = _digest(path)
    with pytest.raises(ValueError, match="PC/word/kernel sequence"):
        NA.audit_native_composition(native, policy=policy, facts=facts_document, target="synthetic")


def test_native_composition_rejects_repinned_operand_claims(composition_execution_inputs, tmp_path):
    from merlin.targetgen.native_model_execution import _digest

    native, policy, document, trace = composition_execution_inputs
    trace["instructions"][1]["rs1"]["raw"] = 123
    path = Path(native["composition_trace"]["path"])
    path.write_text(json.dumps(trace))
    native["composition_trace"] = _digest(path)
    with pytest.raises(ValueError, match="committed register replay"):
        NA.audit_native_composition(native, policy=policy, facts=document, target="synthetic")


def test_executed_operands_replay_padded_register_names_and_hart_state(composition_execution_inputs, tmp_path):
    native, _, _, _ = composition_execution_inputs
    log = tmp_path / "registers.log"
    word = (7 << 25) | (5 << 15) | (6 << 20) | 43
    log.write_text(f"core 0: 3 0x1000 (0x00000013) x5 0x123 x 6 0x456\ncore 0: 3 0x4 (0x{word:08x})\n")
    arguments = dict(
        elf=Path(native["elf"]["path"]),
        trace=log,
        routed=[{"symbol": "fused", "region_id": "chain"}],
        target="synthetic",
    )
    decoded = NA.decode_retirement_trace(**arguments)["instructions"][0]
    assert decoded["rs1"]["raw"] == 0x123 and decoded["rs2"]["raw"] == 0x456
    log.write_text(log.read_text().replace("core 0: 3 0x4", "core 1: 3 0x4"))
    with pytest.raises(ValueError, match="unobserved integer register"):
        NA.decode_retirement_trace(**arguments)


def test_owned_composition_executes_then_produces_and_replays_instruments(
    composition_execution_inputs, tmp_path, monkeypatch
):
    from merlin.targetgen import native_model_execution as native_model
    from merlin.targetgen import owned_host_model_execution as owned
    from merlin.targetgen.native_model_execution import _digest

    native, policy, facts_document, _ = composition_execution_inputs
    policy = {**policy, "required_tiers": ["L0", "L1", "L2"]}
    fact_path = tmp_path / "facts.json"
    fact_path.write_text(json.dumps(facts_document))
    selected = {"path": str(fact_path)}
    elf = native["elf"]
    # Start from the ordinary build record, without volunteered audit instruments.
    record = {
        "runtime_inputs": str(tmp_path),
        "build": {
            "output": {
                "elf": elf["path"],
                "elf_sha256": elf["sha256"],
                "device_artifacts": {"kernels": {"wrapper": {"kernel_symbol": "fused"}}},
            }
        },
        "source_device_join": {
            "assigned": [{"symbol": "wrapper", "group": 9, "source_region_id": "chain"}],
            "joined_groups": {9: 4},
        },
    }
    (tmp_path / "model.mlir").write_bytes((tmp_path / "interface.mlir").read_bytes())
    monkeypatch.setattr(native_model, "_frozen_model_policy", lambda *a, **k: policy)
    monkeypatch.setattr(owned, "_selected_facts", lambda *a, **k: selected)
    monkeypatch.setattr(owned, "_dispatch_proof", lambda *a, **k: {"linked_elf": elf})
    executed = []

    def static(*args, **kwargs):
        assert not executed
        return {"L0": {"status": "pass"}, "L1": {"status": "pass"}}

    monkeypatch.setattr(owned, "_static_tiers", static)
    monkeypatch.setattr(
        NA,
        "form_exact_groups",
        lambda module, *a: (
            [
                SimpleNamespace(
                    index=4,
                    members=list(next(op for op in module.walk() if op.name == "func.func").body.block.ops)[:-1],
                )
            ],
            None,
        ),
    )
    log = Path(native["retirement_trace"]["path"]).read_bytes()
    runner = tmp_path / "functional.py"
    runner.write_text(
        f"import sys\nsys.stderr.buffer.write({log!r})\nsys.stdout.buffer.write(b'complete numerical output')\n"
    )

    def invocation(selected_elf):
        executed.append(True)
        return [sys.executable, str(runner), "-l", "--log-commits", str(selected_elf)], dict(os.environ)

    backend = SimpleNamespace(functional_trace_invocation=invocation)
    monkeypatch.setattr(owned, "_functional_engine_for_facts", lambda *a: (backend, {"digest": "engine"}, lambda: None))
    monkeypatch.setattr(owned, "_numeric_full_output", lambda *a: ({"status": "pass"}, {}))
    result = owned._execute_required_tiers(
        {"__dir__": str(tmp_path)}, record, target="synthetic", facts=fact_path, timeout=10
    )
    assert executed == [True]
    assert result["status"] == "verified", result
    assert Path(result["tiers"]["L2"]["console"]["path"]).read_bytes() == b"complete numerical output"
    for role in ("elf", "region_offload", "host_chain_audit", "composition_trace", "retirement_trace"):
        assert native_model._pinned_native_file(result, role).is_file()
    assert result["tiers"]["L1"]["composition_audit"]["trace_check"]["status"] == "pass"
    # Finishing must reopen these products and the actual source/group/kernel join.
    fresh_record = {**record, **result}
    assert (
        owned._audit_owned_composition(fresh_record, policy=policy, facts=facts_document, target="synthetic")[
            "trace_check"
        ]["status"]
        == "pass"
    )
    sidecar_path = Path(result["region_offload"]["path"])
    sidecar_bytes = sidecar_path.read_bytes()
    sidecar = json.loads(sidecar_bytes)
    sidecar["routed"][0]["source_op_indices"] = [0, 1]
    sidecar_path.write_text(json.dumps(sidecar))
    fresh_record["region_offload"] = _digest(sidecar_path)
    with pytest.raises(ValueError, match="original source groups"):
        owned._audit_owned_composition(fresh_record, policy=policy, facts=facts_document, target="synthetic")
    sidecar_path.write_bytes(sidecar_bytes)
    fresh_record["region_offload"] = result["region_offload"]
    Path(result["retirement_trace"]["path"]).write_bytes(log + b"changed\n")
    with pytest.raises(native_model.NativeModelExecutionError):
        owned._audit_owned_composition(fresh_record, policy=policy, facts=facts_document, target="synthetic")


def test_original_source_def_use_finds_host_arithmetic_even_with_matching_chain_placement():
    source = """module {
      func.func @forward(%a: i32, %b: i32) -> i32 {
        %0 = arith.addi %a, %b : i32
        %1 = arith.addi %0, %b : i32
        %2 = arith.addi %1, %b : i32
        %3 = arith.addi %0, %b : i32
        func.return %2 : i32
      }
    }"""
    trace, expected = _readout_case()
    frozen = trace["composition_evidence"]["frozen_placement"]
    frozen["source_placements"][3] = "host"
    assert FP.host_chain_consumers(source, frozen, expected["composition"]) == [3]
    final_result_use = source.replace("%3 = arith.addi %0, %b", "%3 = arith.addi %2, %b")
    assert FP.host_chain_consumers(final_result_use, frozen, expected["composition"]) == []
    del frozen["source_placements"][3]
    with pytest.raises(FP.FrozenPlacementRefusal, match="UNKNOWN"):
        FP.host_chain_consumers(source, frozen, expected["composition"])


def test_capsule_schema_documents_and_accepts_composition_requirements():
    from jsonschema import Draft202012Validator

    from merlin.common.paths import repo_root

    schema = json.loads((repo_root() / "merlin/contract/schemas/capsule.schema.json").read_text())
    expected_schema = schema["properties"]["expected"]
    expected = _readout_case()[1]
    expected["instruction_classes"] = []
    Draft202012Validator(expected_schema).validate(expected)
    for mode in CA.MODES:
        assert mode in expected_schema["properties"]["modes"]["description"]


def test_shared_siblings_audit_independent_narrow_and_full_width_exits():
    case = _readout_case()
    case[0]["instructions"].insert(-1, _row("MVOUT", dram=_operand(2), readout="i32"))
    case[1]["stage_scoped_modes"]["readout"]["i32_exit"] = True
    case[1]["mode_parameters"]["readout_multipliers"]["outputs"] = [_operand(1)]
    case[1]["mode_parameters"]["i32_exit"] = dict(outputs=[_operand(2)])
    assert _check(case)["status"] == "pass"
    case[0]["instructions"][-2]["decoded"]["readout"] = "i8"
    assert _check(case)["status"] == "fail"


def test_output_scoped_audit_cannot_ignore_an_absent_sibling():
    case = _readout_case()
    case[1]["mode_parameters"]["readout_multipliers"]["outputs"] = [_operand(2)]
    assert _check(case)["status"] == "fail"


@pytest.mark.parametrize("mutation", [None, "unrelated_rows", "late_seed", "wrong_source", "overwritten", "scratchpad"])
def test_accumulator_seed_is_source_bound_and_precedes_its_contraction(mutation):
    seed = _row("MVIN", dram=_operand(3), is_acc_addr=True, accumulate=False)
    rows = [seed, *_compute(), _row("MVOUT")]
    rows[1]["decoded"]["accumulate"] = True
    if mutation == "unrelated_rows":
        seed["decoded"]["spad_addr"] = 64
    elif mutation == "late_seed":
        rows.insert(-1, rows.pop(0))
    elif mutation == "wrong_source":
        seed["decoded"]["dram"] = _operand(4)
    elif mutation == "overwritten":
        rows.insert(1, _row("MVIN", dram=_operand(4), is_acc_addr=True, accumulate=False))
    elif mutation == "scratchpad":
        seed["decoded"]["is_acc_addr"] = False
    case = _case(rows, {"accumulator_preload": True}, {"accumulator_preload": {"input": _operand(3)}})
    result = _check(case)
    assert result["status"] == ("pass" if mutation is None else "fail"), result


def test_accumulator_seed_without_source_binding_is_unknown():
    case = _case([_row("MVIN", is_acc_addr=True), *_compute(), _row("MVOUT")], {"accumulator_preload": True})
    assert _check(case)["composition_audit"]["evidence_status"] == "UNKNOWN"


@pytest.mark.parametrize("mutation", [None, "no_fence", "fence_before_load", "wrong_window", "unrelated_rows"])
def test_scratchpad_pooling_needs_a_fence_between_source_load_and_store(mutation):
    fields = dict(pool_stride=2, pool_size=2, pool_out_dim=2, porows=2, pocols=2, orows=4, ocols=4, upad=0, lpad=0)
    rows = [
        _row("CONFIG_ST", **fields),
        _row("MVIN", is_acc_addr=False),
        _row("FENCE"),
        _row("MVOUT", is_acc_addr=False),
    ]
    if mutation == "no_fence":
        rows.pop(2)
    elif mutation == "fence_before_load":
        rows[1], rows[2] = rows[2], rows[1]
    elif mutation == "wrong_window":
        rows[0]["decoded"]["pool_size"] = 3
    elif mutation == "unrelated_rows":
        rows[1]["decoded"]["spad_addr"] = 64
    case = _case(
        rows, {"pooled_store": True, "scratchpad_source_pooling_fenced": True}, {"pooled_store": {"fields": fields}}
    )
    case[1]["composition"]["producer"] = "movement"
    result = _check(case)
    assert result["status"] == ("pass" if mutation is None else "fail"), result


@pytest.mark.parametrize("mutation", [None, "missing_entry_register", "overlapping_buffers", "unbound_address"])
def test_retired_shared_input_uses_observed_runtime_argument_bases(
    native_files, monkeypatch, selected_hardware, mutation
):
    selected_hardware[0]["facts"]["datapaths"] = [dict(name="input", dtype="i8")]
    observer = selected_hardware[1]
    elf, log = native_files
    rows = [
        _row("CONFIG_LD", state_id=6, stride=3),
        _row("CONFIG_ST", out_stride_bytes=3, pool_stride=0),
        _row("MVIN", dram={"kind": "const", "raw": 4096}, spad_addr=8, is_acc_addr=False),
        *_compute(),
        _row("MVOUT", dram={"kind": "const", "raw": 8192}),
        *_compute(),
        _row("MVOUT", dram={"kind": "const", "raw": 12288}),
    ]
    codes = {index + 1: (row["class"], row["decoded"]) for index, row in enumerate(rows)}
    monkeypatch.setattr(NA, "_function_ranges", lambda *a: [(0, 4 * len(rows), "fused")])
    monkeypatch.setattr(
        decode,
        "_semantics",
        lambda _: SimpleNamespace(
            isa_constants=lambda _: {
                "CUSTOM_OPCODE": 43,
                "OPERAND_FIELDS": {"PRELOAD": {"fields": [{"as": "flag", "mask": 512}]}},
            },
            decode_instruction=lambda code, *a: copy.deepcopy(codes[code]),
            load_states=observer.load_states,
        ),
    )
    arguments = [dict(arg_index=i, register=10 + i, byte_extent=64) for i in range(3)]
    writes = "x10 0x1000 x11 0x2000 x12 0x3000"
    if mutation == "missing_entry_register":
        writes = "x10 0x1000 x11 0x2000"
    elif mutation == "overlapping_buffers":
        writes = "x10 0x1000 x11 0x1020 x12 0x3000"
    elif mutation == "unbound_address":
        next(decoded for kind, decoded in codes.values() if kind == "MVOUT")["dram"]["raw"] = 16384
    log.write_text(
        f"core 0: 3 0x10000 (0x00000013) {writes}\n"
        + "".join(f"core 0: 3 0x{index * 4:x} (0x{code << 25 | 43:08x})\n" for index, code in enumerate(codes))
    )
    try:
        retired = NA.decode_retirement_trace(
            elf=elf,
            trace=log,
            routed=[dict(symbol="fused", region_id="chain")],
            target="synthetic",
            runtime_abi={"fused": arguments},
        )
        case = _case(retired["instructions"], {"shared_input": True}, {"shared_input": {"input": _operand(0)}})
        result = _check(case)
        assert result["status"] == ("pass" if mutation is None else "fail"), result
    except ValueError as error:
        assert mutation is not None and "UNKNOWN" in str(error)


def test_symbolic_runtime_binding_requires_the_runs_abi_record(composition_execution_inputs):
    native, policy, document, _ = composition_execution_inputs
    with pytest.raises(ValueError, match="UNKNOWN.*runtime.*ABI"):
        NA._composition_runtime_abi(native, {"inputs": []}, [dict(symbol="fused")])


@pytest.mark.parametrize("mutation", [None, "missing_host_abi", "scalar_argument", "wrong_shape", "changed_interface"])
def test_runtime_abi_is_rederived_from_the_pinned_build_interface(native_files, tmp_path, mutation):
    from merlin.common.paths import repo_root
    from merlin.targetgen.native_model_execution import _digest

    elf, _ = native_files
    # Standard ELF e_machine identifies the host ISA independently of device facts.
    blob = bytearray(elf.read_bytes())
    blob[18:20] = (243).to_bytes(2, "little")
    elf.write_bytes(blob)
    source = tmp_path / "source.mlir"
    source.write_text("""module {
      func.func @forward(%a: tensor<3x3xi8>, %w: tensor<3x3xi8>) -> tensor<3x3xi32> {
        %e = tensor.empty() : tensor<3x3xi32>
        func.return %e : tensor<3x3xi32>
      }
    }""")
    declaration = dict(
        interface_mlir=source.name,
        inputs=[
            dict(name="A", role="input", shape=[3, 3], dtype="i8"),
            dict(name="W", role="weight", shape=[3, 3], dtype="i8"),
        ],
    )
    template = repo_root() / "merlin/contract/capsules/_model_layers/G_matmul_m3136k64n64_raw/capsule.interface.mlir"
    interface = tmp_path / "interface.mlir"
    interface.write_text(template.read_text().replace("3136", "3").replace("64", "3"))
    llvm = tmp_path / "kernel.ll"
    llvm.write_text("define void @entry(ptr %w, ptr %a, ptr %o) { ret void }\n")
    if mutation == "scalar_argument":
        llvm.write_text(llvm.read_text().replace("ptr %w", "i64 %w"))
    compilation, rename = tmp_path / "compile.json", tmp_path / "rename.json"
    compilation.write_text(
        json.dumps(
            dict(
                status="completed",
                returncode=0,
                argv=[] if mutation == "missing_host_abi" else ["clang", "--target=riscv64-unknown-elf"],
            )
        )
    )
    rename.write_text(
        json.dumps(dict(status="completed", returncode=0, argv=["objcopy", "--redefine-sym=entry=fused"]))
    )
    native = dict(
        elf=_digest(elf),
        capsule_directory=str(tmp_path),
        build=dict(
            output=dict(
                device_artifacts=dict(
                    kernels=dict(
                        wrapper=dict(
                            kernel_symbol="fused",
                            artifacts=dict(logical_interface=_digest(interface), llvm_ir=_digest(llvm)),
                            custody=dict(compilation=_digest(compilation), rename=_digest(rename)),
                        )
                    )
                )
            )
        ),
    )
    if mutation == "wrong_shape":
        declaration["inputs"][0]["shape"] = [4, 3]
    elif mutation == "changed_interface":
        interface.write_text(interface.read_text() + " ")
    if mutation is None:
        abi = NA._composition_runtime_abi(native, declaration, [dict(symbol="fused")])
        assert abi == {
            "fused": [
                dict(arg_index=1, register=10, byte_extent=9),
                dict(arg_index=0, register=11, byte_extent=9),
                dict(arg_index=2, register=12, byte_extent=36),
            ]
        }
    else:
        with pytest.raises((ValueError, RuntimeError)):
            NA._composition_runtime_abi(native, declaration, [dict(symbol="fused")])


def test_seed_address_flags_come_from_the_selected_operand_observer(selected_hardware, monkeypatch):
    observer = selected_hardware[1]
    monkeypatch.setattr(decode, "_semantics", lambda _: SimpleNamespace(operand_observer=observer))
    rows = [
        _row("MVIN", dram=_operand(3), is_acc_addr=True, accumulate=False),
        _row("PRELOAD", c_addr=32, c_rows=3, c_cols=3, accumulate=True),
        _row("COMPUTE_PRELOADED", a_spad=8),
        _row("MVOUT"),
    ]
    case = _case(rows, {"accumulator_preload": True}, {"accumulator_preload": {"input": _operand(3)}})
    assert _check(case)["status"] == "pass"


@pytest.mark.parametrize("store_start", [32, 35])
def test_partial_seed_overwrite_preserves_only_untouched_rows(store_start):
    rows = [
        _row("MVIN", dram=_operand(3), rows=6, is_acc_addr=True, accumulate=False),
        _row("MVIN", dram=_operand(4), spad_addr=35, rows=3, is_acc_addr=True, accumulate=False),
        _row("PRELOAD", c_addr=store_start, c_rows=3, c_cols=3, accumulate=True),
        _row("COMPUTE_PRELOADED", a_spad=8),
        _row("MVOUT", acc_addr=store_start),
    ]
    case = _case(rows, {"accumulator_preload": True}, {"accumulator_preload": {"input": _operand(3)}})
    assert _check(case)["status"] == ("pass" if store_start == 32 else "fail")


@pytest.mark.parametrize("word", [0x0000000F, 0x0330000F, 0x0F00000F, 0x00F0000F, 0x8FF0000F, 0x0FF0000F])
def test_pooling_credit_requires_retired_ordering_fence(native_files, monkeypatch, word):
    elf, log = native_files
    rows = [_row("MVIN", is_acc_addr=False), _row("MVOUT", is_acc_addr=False)]
    monkeypatch.setattr(NA, "_function_ranges", lambda *a: [(0, 12, "fused")])
    monkeypatch.setattr(
        decode,
        "_semantics",
        lambda _: SimpleNamespace(
            isa_constants=lambda _: {"CUSTOM_OPCODE": 43},
            decode_instruction=lambda code, *a: (rows[code - 1]["class"], rows[code - 1]["decoded"]),
        ),
    )
    log.write_text(f"core 0: 3 0x0 (0x0200002b)\ncore 0: 3 0x4 (0x{word:08x})\ncore 0: 3 0x8 (0x0400002b)\n")
    trace = NA.decode_retirement_trace(
        elf=elf, trace=log, routed=[dict(symbol="fused", region_id="chain")], target="synthetic"
    )
    case = _case(trace["instructions"], {"scratchpad_source_pooling_fenced": True})
    case[1]["composition"]["producer"] = "movement"
    result = _check(case)
    assert result["status"] == ("pass" if word == 0x0FF0000F else "fail"), result


@pytest.mark.parametrize(
    "offset,stride,width,passes",
    [(63, 3, 1, False), (55, 3, 1, True), (55, 4, 1, False), (55, 0, 1, True), (50, 4, 2, True), (51, 4, 2, False)],
)
def test_runtime_binding_covers_complete_strided_dma(selected_hardware, offset, stride, width, passes):
    selected_hardware[0]["facts"]["datapaths"] = [dict(name="input", dtype=f"i{width * 8}")]
    rows = [
        _row("CONFIG_LD", state_id=6, stride=stride),
        _row("CONFIG_ST", out_stride_bytes=3),
        _row("MVIN", dram=dict(kind="const", raw=4096 + offset), spad_addr=8, is_acc_addr=False),
        *_compute(),
        _row("MVOUT", dram=_operand(1)),
        *_compute(),
        _row("MVOUT", dram=_operand(2)),
    ]
    rows[2]["runtime_buffer_bindings"] = [dict(arg_index=0, base=4096, byte_extent=64)]
    result = _check(_case(rows, {"shared_input": True}, {"shared_input": {"input": _operand(0)}}))
    assert result["status"] == ("pass" if passes else "fail"), result


@pytest.mark.parametrize("missing", ["rows", "cols", "stride", "width", "load_state"])
def test_unresolved_runtime_dma_footprint_is_unknown(selected_hardware, missing):
    selected_hardware[0]["facts"]["datapaths"] = [dict(name="input", dtype="i8")]
    rows = [
        _row("CONFIG_LD", state_id=6, stride=3),
        _row("MVIN", dram=dict(kind="const", raw=4096), is_acc_addr=False),
        *_compute(),
        _row("MVOUT"),
    ]
    rows[1]["runtime_buffer_bindings"] = [dict(arg_index=0, base=4096, byte_extent=64)]
    if missing in {"rows", "cols"}:
        del rows[1]["decoded"][missing]
    elif missing == "stride":
        del rows[0]["decoded"]["stride"]
    elif missing == "width":
        selected_hardware[0]["facts"]["datapaths"] = []
    else:
        rows[0]["decoded"]["state_id"] = 3
    result = _check(_case(rows, {"i32_exit": True}))
    assert result["status"] == "fail"
    assert result["composition_audit"]["evidence_status"] == "UNKNOWN"


@pytest.mark.parametrize(
    "readout,offset,passes", [("i8", 55, True), ("i8", 56, False), ("i32", 28, True), ("i32", 29, False)]
)
def test_runtime_store_footprint_uses_decoded_readout_width(selected_hardware, readout, offset, passes):
    selected_hardware[0]["facts"]["datapaths"] = [dict(name="input", dtype="i8")]
    rows = [
        _row("CONFIG_ST", out_stride_bytes=12 if readout == "i32" else 3, pool_stride=0),
        *_compute(),
        _row("MVOUT", dram=dict(kind="const", raw=8192 + offset), readout=readout),
    ]
    rows[-1]["runtime_buffer_bindings"] = [dict(arg_index=1, base=8192, byte_extent=64)]
    case = _case(
        rows,
        {"i32_exit": True} if readout == "i32" else {"readout_multipliers": True},
        {"readout_multipliers": {"acc_scale_bits": [0]}},
    )
    rows[0]["decoded"].update(acc_scale_bits=0)
    # _case keeps the row objects, so the active store configuration is updated too.
    result = _check(case)
    assert result["status"] == ("pass" if passes else "fail"), result


def test_pooling_completion_with_missing_fence_semantics_is_unknown():
    fence = _row("FENCE")
    del fence["decoded"]["pred"]
    case = _case(
        [_row("MVIN", is_acc_addr=False), fence, _row("MVOUT", is_acc_addr=False)],
        {"scratchpad_source_pooling_fenced": True},
    )
    case[1]["composition"]["producer"] = "movement"
    result = _check(case)
    assert result["status"] == "fail"
    assert result["composition_audit"]["evidence_status"] == "UNKNOWN"
