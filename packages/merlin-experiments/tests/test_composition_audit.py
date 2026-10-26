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
                    "support": "both",
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
    assert NA._trace_word(b"core 0: >>>> _start") is None


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


def test_original_load_mapping_fallback_is_rederived(selected_hardware, monkeypatch):
    from merlin.targetgen import composition_facts, target_registry

    observer = selected_hardware[1]
    del observer.load_states
    mapping = SimpleNamespace(ports=(("MVIN", 5), ("MVIN2", 8)), max_element_bytes=4)
    calls = []
    monkeypatch.setattr(target_registry, "load_contract", lambda _: {"selected": True})

    def derive(**kwargs):
        calls.append(kwargs)
        return mapping

    monkeypatch.setattr(composition_facts, "load_state_mapping", derive)
    assert CA._selected_hardware("synthetic", {"scaled_accumulate_load"})["load_states"] == dict(mapping.ports)
    assert len(calls) == 1 and calls[0]["contract"] == {"selected": True}


@pytest.mark.parametrize("mutation", [None, "missing", "overlap", "stride", "geometry"])
def test_pooled_bands_cover_source_windows_exactly(mutation):
    geometry = dict(image=[5, 5, 4], window=2, stride=2)
    config = dict(
        pool_size=2,
        pool_stride=2,
        pool_out_dim=2,
        pocols=2,
        ocols=5,
        orows=2,
        porows=1,
        upad=0,
        lpad=0,
        out_stride_bytes=4,
    )
    commits = [
        (
            {"dram": dict(kind="argbase", arg_index=1, offset=row * 8 + column), "rows": 1, "cols": 2},
            dict(config),
            i,
            [],
        )
        for i, (row, column) in enumerate([(0, 0), (0, 2), (1, 0), (1, 2)])
    ]
    if mutation == "missing":
        commits.pop()
    elif mutation == "overlap":
        commits.append(commits[0])
    elif mutation == "stride":
        commits[0][1]["out_stride_bytes"] = 3
    elif mutation == "geometry":
        commits[0][1]["orows"] = 3
    if mutation:
        with pytest.raises(ValueError):
            CA._pool(commits, dict(geometry=geometry))
    else:
        CA._pool(commits, dict(geometry=geometry))


def _column_partition_case():
    rows = []
    for offset, scale in ((0, 0.25), (3, 0.5)):
        rows.extend(
            [
                _row("CONFIG_ST", acc_scale_bits=_bits(scale), out_stride_bytes=6, pool_stride=0),
                _row("MVOUT", dram={**_operand(2), "offset": offset}, rows=2, cols=3),
            ]
        )
    parameters = dict(
        partitions=[dict(output=_operand(2), shape=[2, 6], column_scale_bits=[_bits(0.25)] * 3 + [_bits(0.5)] * 3)]
    )
    return rows, parameters


def test_readout_binds_multiplier_to_columns_and_covers_every_output():
    rows, parameters = _column_partition_case()
    CA._readout(CA._events(rows, {})[0], parameters)
    # Equal multiplier multisets cannot hide a swap of column partitions.
    rows[0]["decoded"]["acc_scale_bits"], rows[2]["decoded"]["acc_scale_bits"] = (
        rows[2]["decoded"]["acc_scale_bits"],
        rows[0]["decoded"]["acc_scale_bits"],
    )
    with pytest.raises(ValueError, match="multiplier differs"):
        CA._readout(CA._events(rows, {})[0], parameters)


@pytest.mark.parametrize("fault", ["missing", "overlap", "allocation", "width", "stride", "wide"])
def test_partition_readout_refuses_missing_overlapping_or_unbound_stores(fault):
    rows, parameters = _column_partition_case()
    if fault == "missing":
        rows.pop()
    elif fault == "overlap":
        rows.append(copy.deepcopy(rows[1]))
    elif fault == "allocation":
        rows[1]["decoded"]["dram"]["arg_index"] = 3
    elif fault == "width":
        rows[1]["decoded"]["cols"] = 4
    elif fault == "stride":
        rows[0]["decoded"]["out_stride_bytes"] = 5
    else:
        rows[1]["decoded"]["readout"] = "i32"
    with pytest.raises(ValueError):
        CA._readout(CA._events(rows, {})[0], parameters)


def _bound_load_case():
    operands = [
        dict(input=_operand(i), shape=[2, 6], scale_bits=_bits(scale), accumulate=i == 1)
        for i, scale in enumerate((0.5, 0.25))
    ]
    loads = [
        (
            {
                "dram": {**_operand(i), "offset": offset},
                "is_acc_addr": True,
                "accumulate": i == 1,
                "rows": 2,
                "cols": 3,
            },
            {"scale_bits": _bits(scale), "stride": 6, "shrink": 1},
        )
        for i, scale in enumerate((0.5, 0.25))
        for offset in (0, 3)
    ]
    return loads, operands


def test_scaled_load_binds_both_overwrite_and_accumulate_operands_across_tiles():
    loads, operands = _bound_load_case()
    CA._scaled_load(loads, dict(operands=operands))


@pytest.mark.parametrize("fault", ["missing", "duplicate", "scale", "swap", "accumulate", "shrink", "stride"])
def test_bound_scaled_load_refuses_changed_operand_configuration_or_coverage(fault):
    loads, operands = _bound_load_case()
    if fault == "missing":
        loads.pop()
    elif fault == "duplicate":
        loads.append(copy.deepcopy(loads[0]))
    elif fault == "scale":
        loads[0][1]["scale_bits"] = _bits(1.0)
    elif fault == "swap":
        for load, config in loads:
            config["scale_bits"] = _bits(0.25 if load["dram"]["arg_index"] == 0 else 0.5)
    elif fault == "accumulate":
        loads[0][0]["accumulate"] = True
    elif fault == "shrink":
        del loads[0][1]["shrink"]
    else:
        loads[0][1]["stride"] = 7
    with pytest.raises(ValueError):
        CA._scaled_load(loads, dict(operands=operands))


@pytest.mark.parametrize("shrink,expected", [(0, 18), (1, 9)])
def test_accumulator_dma_footprint_uses_active_shrink(shrink, expected):
    hardware = dict(max_load_bytes=4, input_bytes=1)
    assert (
        CA._dma_footprint("MVIN", dict(is_acc_addr=True, rows=3, cols=3), dict(stride=3, shrink=shrink), hardware)
        == expected
    )


def test_static_dma_objects_come_from_defined_same_elf_symbols(monkeypatch):
    monkeypatch.setattr(
        NA.subprocess,
        "check_output",
        lambda *_a, **_k: (
            """
    1: 0000000000001000 16 OBJECT LOCAL DEFAULT 2 weights
    2: 0000000000002000 0 OBJECT LOCAL DEFAULT 2 empty
    3: 0000000000000000 16 OBJECT GLOBAL DEFAULT UND external
    4: 0000000000003000 16 FUNC GLOBAL DEFAULT 1 kernel
    """
        ),
    )
    assert NA.elf_object_buffers("same.elf") == [dict(symbol="weights", base=4096, byte_extent=16)]


@pytest.mark.parametrize("address,extent,passes", [(4096, 9, True), (4096, 8, False), (4097, 9, False)])
def test_runtime_dma_static_binding_requires_full_object_extent(selected_hardware, address, extent, passes):
    selected_hardware[0]["facts"]["datapaths"] = [dict(name="input", dtype="i8")]
    rows = [
        _row("CONFIG_LD", state_id=6, stride=3),
        _row("MVIN", dram=dict(kind="const", raw=address), is_acc_addr=False),
    ]
    rows[1]["runtime_buffer_bindings"] = [dict(arg_index=0, base=8192, byte_extent=9)]
    rows[1]["static_buffer_bindings"] = [dict(symbol="weights", base=4096, byte_extent=extent)]
    if passes:
        projected = CA._logical_runtime_operands(rows, target="moved")
        assert projected[1]["decoded"]["dram"] == dict(kind="global", symbol="weights", offset=0)
    else:
        with pytest.raises(ValueError, match="footprint"):
            CA._logical_runtime_operands(rows, target="moved")


@pytest.mark.parametrize("static_extent,passes", [(9, True), (10, False)])
def test_direct_elf_input_alias_requires_the_exact_argument_extent(selected_hardware, static_extent, passes):
    selected_hardware[0]["facts"]["datapaths"] = [dict(name="input", dtype="i8")]
    rows = [_row("CONFIG_LD", state_id=6, stride=3), _row("MVIN", dram=dict(kind="const", raw=4096), is_acc_addr=False)]
    rows[1]["runtime_buffer_bindings"] = [dict(arg_index=0, base=4096, byte_extent=9)]
    rows[1]["static_buffer_bindings"] = [dict(symbol="input", base=4096, byte_extent=static_extent)]
    if passes:
        projected = CA._logical_runtime_operands(rows, target="moved")
        assert projected[1]["decoded"]["dram"] == _operand(0)
    else:
        with pytest.raises(ValueError, match="unique logical buffer"):
            CA._logical_runtime_operands(rows, target="moved")


def test_output_partitions_cannot_bypass_explicit_config_reissue_mode():
    rows, parameters = _column_partition_case()
    with pytest.raises(ValueError, match="per-block CONFIG_ST re-issue"):
        CA._readout(CA._events(rows, {})[0], parameters, per_block=True)


def test_k_extension_binds_blocked_work_to_each_output_accumulator():
    work = [(None, dict(c_addr=address)) for address in (32, 64) for _ in range(3)]
    commits = [(dict(acc_addr=32), None, None, work), (dict(acc_addr=64), None, None, [])]
    parameters, hardware = dict(source_k=6, extension_k=3), dict(tile_k=3, address_flags=0)
    CA._k_extension(commits, parameters, hardware)
    # The same aggregate count cannot conceal unequal per-output work.
    work[0] = (None, dict(c_addr=64))
    with pytest.raises(ValueError, match="extra K tiles"):
        CA._k_extension(commits, parameters, hardware)


def test_k_extension_cannot_reuse_or_leave_unstored_compute_work():
    work = [(None, dict(c_addr=32))] * 3
    parameters, hardware = dict(source_k=6, extension_k=3), dict(tile_k=3, address_flags=0)
    for commits in (
        [(dict(acc_addr=32), None, None, work), (dict(acc_addr=32), None, None, [])],
        [(dict(acc_addr=32), None, None, work + [(None, dict(c_addr=64))])],
    ):
        with pytest.raises(ValueError):
            CA._k_extension(commits, parameters, hardware)


def test_independent_sibling_stage_parameters_are_enforced():
    trace, expected = _readout_case()
    original = copy.deepcopy(expected["mode_parameters"])
    expected["mode_parameters"] = {"readout_multipliers": {"acc_scale_bits": [0]}}
    expected["stage_scoped_mode_parameters"] = {"readout": original}
    assert _check((trace, expected))["status"] == "pass"
    expected["stage_scoped_mode_parameters"]["readout"] = expected["mode_parameters"]
    assert _check((trace, expected))["status"] == "fail"


def test_mean_emission_uses_declared_mac_duties_not_a_row_name():
    instructions = [_row("COMPUTE_PRELOADED")]
    binding = dict(op="aten.mean.dim", numeric=dict(window=4))
    rows = {
        "renamed_mac": dict(families=["window_mean"], arithmetic="integer_multiply_accumulate"),
        "sum": dict(families=["window_mean"], arithmetic="integer_sum"),
    }
    assert NA._emission_declaration(binding, instructions, rows) == "renamed_mac"
    for changed in ({}, dict(rows, ambiguous=rows["renamed_mac"])):
        with pytest.raises(ValueError, match="declared MAC realization"):
            NA._emission_declaration(binding, instructions, changed)
    with pytest.raises(ValueError, match="observed MAC realization"):
        NA._emission_declaration(binding, [], rows)
    binding["software_declaration"] = "explicit"
    assert NA._emission_declaration(binding, instructions, rows) == "explicit"


def test_mean_emission_enforces_duties_and_does_not_erase_unknown_bindings():
    binding = dict(op="aten.mean.dim", numeric=dict(window=4), symbol="fused")
    rows = [_row("CONFIG_EX", dataflow=4), *_compute(), _row("MVOUT", readout="i32")]
    spec = dict(
        operations=[
            dict(
                id="arbitrary_mac",
                families=["window_mean"],
                arithmetic="integer_multiply_accumulate",
                emission_obligations=dict(
                    stationary=dict(status="instruction_audit", check="weight_stationary", constraints=dict(eq="WS"))
                ),
            )
        ]
    )
    report = NA.audit_declared_emission(dict(instructions=rows), [binding], spec, target="synthetic")
    assert report["fused"]["obligations"]["stationary"]["observation_status"] == "pass"
    rows[0]["decoded"]["dataflow"] = 9
    with pytest.raises(ValueError, match="emission obligation"):
        NA.audit_declared_emission(dict(instructions=rows), [binding], spec, target="synthetic")
    with pytest.raises(ValueError, match="lacks a software declaration"):
        NA.audit_declared_emission(dict(instructions=rows), [dict(binding, op="unknown")], spec, target="synthetic")


def test_k_extension_refuses_ordinary_source_work_even_when_numeric_readout_matches():
    # One source tile per accumulator output cannot establish an extra digit tile.
    work = [(None, dict(c_addr=address)) for address in (32, 64)]
    commits = [(dict(acc_addr=32), None, None, work), (dict(acc_addr=64), None, None, [])]
    with pytest.raises(ValueError, match="extra K tiles"):
        CA._k_extension(commits, dict(source_k=3, extension_k=3), dict(tile_k=3, address_flags=0))


def test_unclean_native_audit_retains_the_underlying_reason():
    with pytest.raises(ValueError, match="missing declaration"):
        NA.require_observed_kernels(dict(status="UNKNOWN", issues=["missing declaration"]), ["fused"])


@pytest.mark.parametrize("support", [None, "fixed_select", "unknown", "weight_stationary"])
def test_output_stationary_emission_requires_selected_support(selected_hardware, support):
    selected_hardware[0]["facts"]["timing_structure"]["dataflow"]["support"] = support
    report = _check(_case([_row("CONFIG_EX", dataflow=9), *_compute(), _row("MVOUT")], {"output_stationary": True}))
    assert report["status"] == "fail"
    assert report["composition_audit"]["evidence_status"] == "UNKNOWN"


@pytest.mark.parametrize("mutation", [None, "drop", "extra", "wrong_base", "unresolved_extent"])
def test_pooled_k_extension_authenticates_every_accumulator_tile(mutation):
    # One pooled MVOUT drains five rows: two full tiles and one partial tile.
    work = [(None, dict(c_addr=address)) for address in (32, 34, 36) for _ in range(3)]
    store, config = dict(acc_addr=32), dict(pool_stride=2, orows=1, ocols=5)
    if mutation == "drop":
        work.pop()
    elif mutation == "extra":
        work.append((None, dict(c_addr=38)))
    elif mutation == "wrong_base":
        store["acc_addr"] = 34
    elif mutation == "unresolved_extent":
        config.pop("ocols")
    commits = [(store, config, None, work)]
    parameters, hardware = dict(source_k=4, extension_k=2), dict(tile_k=2, address_flags=0)
    if mutation is None:
        CA._k_extension(commits, parameters, hardware)
    else:
        with pytest.raises(ValueError):
            CA._k_extension(commits, parameters, hardware)


def _scaled_load_node():
    readout = dict(exit="requant", stage_indices=[], stages=[])
    return dict(form="operand_sum", realization="scaled_loads", readout=readout, seed=False, residual=None, scheme="s")


def test_scaled_load_without_a_mesh_product_needs_no_dataflow_select(selected_hardware):
    from merlin_experiments.phase0.fusion_cases import chain_for_node

    from merlin.targetgen.functional_truth_compositions import _audit_modes, _family_refusals

    chain = chain_for_node(_scaled_load_node(), "not_applicable")
    case = dict(
        producer="operand_sum",
        exit="requant",
        readout_stages=[],
        dataflow="not_applicable",
        operand_multipliers=[1.0, 1.0],
        negates_load_minimum=False,
    )
    modes = _audit_modes(chain, case, dict(readout=[]))
    assert "dataflows" not in chain and "output_stationary" not in modes
    scheme = dict(operand_sum_rounding="single_rounding")
    assert list(_family_refusals(chain, scheme, case, {}, {})) == []
    # No dataflow select is derivable here; a scaled-load kernel must not ask for one.
    selected_hardware[0]["facts"]["timing_structure"]["dataflow"].pop("modes")
    rows = [
        _row("CONFIG_LD", state_id=6, scale_bits=_bits(0.625)),
        _row("MVIN", is_acc_addr=True, accumulate=True),
        _row("MVOUT"),
    ]
    trace, expected = _case(
        rows,
        dict.fromkeys(modes, True),
        {"scaled_accumulate_load": {"scale_bits": [_bits(0.625)]}},
    )
    expected["composition"]["producer"] = "operand_sum"
    assert _check((trace, expected))["status"] == "pass"


def test_contraction_still_requires_a_derived_dataflow_select(selected_hardware):
    from merlin_experiments.phase0.fusion_cases import chain_for_node
    from merlin_experiments.phase0.fusion_execution import _trace_modes

    from merlin.targetgen.functional_truth_compositions import _audit_modes, _family_refusals

    node = dict(_scaled_load_node(), form="contraction", realization="matmul")
    chain = chain_for_node(node, "OS")
    case = dict(producer="contraction", exit="requant", readout_stages=[], dataflow="OS")
    assert "output_stationary" in _audit_modes(chain, case, dict(readout=[]))
    selected_hardware[0]["facts"]["timing_structure"]["dataflow"].pop("modes")
    report = _check(_case([_row("CONFIG_EX", dataflow=9), *_compute(), _row("MVOUT")], {"output_stationary": True}))
    assert report["status"] == "fail"
    assert report["composition_audit"]["evidence_status"] == "UNKNOWN"
    # A contraction cannot shed its dataflow obligation by claiming none applies.
    unselected = dict(case, dataflow="not_applicable")
    assert "a mesh-product producer requires a selected dataflow" in _family_refusals(
        chain_for_node(node, "not_applicable"), {}, unselected, {}, {}
    )
    candidate = dict(semantic=dict(structure="single", nodes=[node], dataflow="not_applicable"))
    with pytest.raises(ValueError, match="mesh-product semantics"):
        _trace_modes(None, candidate, dict(readout=[]))
    scaled = dict(semantic=dict(structure="single", nodes=[_scaled_load_node()], dataflow="OS"))
    with pytest.raises(ValueError, match="mesh-product semantics"):
        _trace_modes(None, scaled, dict(readout=[]))


@pytest.mark.parametrize(
    "case",
    [
        dict(producer="contraction"),
        dict(producer="contraction", reduction_kind="sum", window_mean=True),
        # A window mean is a ones-operand MAC whatever producer label its chain carries.
        dict(producer="movement", window_mean=True),
        dict(producer="movement", reduction_kind="sum"),
    ],
)
def test_every_mesh_product_producer_refuses_not_applicable(case):
    from merlin_experiments.phase0.fusion_cases import chain_for_node

    from merlin.targetgen.functional_truth_compositions import _family_refusals

    node = dict(_scaled_load_node(), form="window_movement", realization="mean")
    case = dict(case, exit="requant", readout_stages=[], dataflow="not_applicable")
    refusals = list(_family_refusals(chain_for_node(node, "not_applicable"), {}, case, {}, {}))
    assert "a mesh-product producer requires a selected dataflow" in refusals


def test_mesh_free_and_unknown_producers_under_not_applicable():
    from merlin.targetgen.functional_truth_compositions import _family_refusals

    chain = dict(producer="movement", constraints=[])
    pool = dict(producer="movement", exit="requant", readout_stages=[], dataflow="not_applicable")
    assert "a mesh-product producer requires a selected dataflow" not in _family_refusals(chain, {}, pool, {}, {})
    unknown = dict(pool, producer="elementwise")
    assert "producer has no declared mesh-product semantics" in _family_refusals(chain, {}, unknown, {}, {})


# Runtime-input readout partitions.


def _runtime_case(scale=0.25, relu=True):
    from merlin.targetgen import runtime_scale_trace as RT

    bits = _bits(scale)
    config = dict(acc_scale_bits=bits, acc_scale=scale, acc_act=int(relu), relu=relu, out_stride_bytes=3, pool_stride=0)
    raw = dict(acc_scale_bits=_bits(1.0), acc_scale=1.0, acc_act=0, relu=False, out_stride_bytes=12, pool_stride=0)
    rows = [
        {**_row("CONFIG_ST", **raw), "funct": 0, "pc": 0x200, "rs2": dict(kind="const", raw=_bits(1.0) << 32 | 12)},
        _row("MVOUT", dram=_operand(4), rows=2, cols=3, readout="i32"),
        {**_row("CONFIG_ST", **config), "funct": 0, "pc": 0x208, "rs2": dict(kind="const", raw=bits << 32 | 3)},
        _row("MVOUT", dram=_operand(5), rows=2, cols=3),
    ]
    for ordinal, row in enumerate(rows):
        row.update(command_ordinal=ordinal, funct=row.get("funct", 3), pc=row.get("pc", 0x300 + 4 * ordinal))
    load = dict(kernel="fused", call=1, pc=0x204, word=0x4288, arg_index=3, address=0xC0000400, offset=0)
    use = dict(kernel="fused", call=1, command_ordinal=2, pc=0x208, word=0x00A6307B, rs2_register="x10")
    use.update(rs2=bits << 32 | 3, arg_index=3, scale_bits=bits, static_bits=3, load_pc=0x204, chain=[])
    report = dict(
        schema=RT.SCHEMA,
        status="pass",
        elf_sha256="e" * 64,
        retirement_trace_sha256="t" * 64,
        slot=dict(offset=32, width=32),
        loads=[dict(load, loaded_bits=bits, register="x10")],
        config_uses=[dict(use, chain_sha256="c" * 64)],
        evidence={},
    )
    report["evidence"]["fused:3"] = RT.scale_load_evidence_id(report, "fused", 3)
    report["sha256"] = RT._sha({k: v for k, v in report.items() if k != "sha256"})
    partition = dict(
        output=_operand(5),
        shape=[2, 3],
        scale_input=dict(kind="argbase", arg_index=3, offset=0, dtype="f32", shape=[1]),
        expected_input_bits=bits,
        scale_load_evidence_id=report["evidence"]["fused:3"],
    )
    return rows, partition, report


def _runtime_readout(rows, partition, report, **kwargs):
    commits = CA._events(rows, {})[0]
    CA._readout(commits, dict(partitions=[partition]), rows=rows, scale_loads=report, **kwargs)


@pytest.mark.parametrize("scale", [0.25, 0.0, 1.0, -0.5])
def test_runtime_partition_checks_retired_bits_and_load_provenance(scale):
    rows, partition, report = _runtime_case(scale, relu=scale >= 0)
    # A runtime scale of 0 or 1 is still the source operand; no non-unit test applies.
    _runtime_readout(rows[2:], partition, report, relu=scale >= 0, scaled=True)


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ("stale_bits", "differ from the frozen runtime input"),
        ("wrong_register", "no retired load provenance"),
        ("wrong_pc", "no retired load provenance"),
        ("wrong_input", "not this scale input's observed chain"),
        ("missing_load", "UNKNOWN: runtime readout has no retired scale-load evidence"),
        ("tampered", "report digest differs from its rows"),
        ("exclusive", "mutually exclusive"),
        ("nonfinite", "UNKNOWN: runtime readout has no frozen finite"),
        ("missing_store", "do not cover every source output"),
        ("overlap", "overlap source output"),
        ("activation", "lacks simultaneous ReLU"),
    ],
)
def test_runtime_partition_mutations_fail(mutation, error):
    rows, partition, report = _runtime_case()
    rows = rows[2:]
    if mutation == "stale_bits":
        # The effective CONFIG_ST still carries the previous call's scale.
        rows[0]["decoded"]["acc_scale_bits"] = _bits(0.5)
    elif mutation == "wrong_register":
        rows[0]["command_ordinal"] = 7
    elif mutation == "wrong_pc":
        rows[0]["pc"] = 0x20C
    elif mutation == "wrong_input":
        partition["scale_input"]["arg_index"] = 2
    elif mutation == "missing_load":
        report = None
    elif mutation == "tampered":
        report["config_uses"][0]["rs2"] ^= 1
    elif mutation == "exclusive":
        partition["column_scale_bits"] = [partition["expected_input_bits"]] * 3
    elif mutation == "nonfinite":
        partition["expected_input_bits"] = 0x7F800000
    elif mutation == "missing_store":
        rows[1]["decoded"]["rows"] = 1
    elif mutation == "overlap":
        rows.append(copy.deepcopy(rows[1]))
    else:
        rows[0]["decoded"].update(relu=False, acc_act=0)
    with pytest.raises(ValueError, match=error):
        _runtime_readout(rows, partition, report, relu=True)


def test_raw_sibling_of_runtime_readout_requires_identity_config():
    rows, _, _ = _runtime_case()
    parameters = {"i32_exit": dict(outputs=[_operand(4)], identity_config=True)}
    CA.check_modes(rows, {"i32_exit": True}, parameters, target="synthetic")
    for field, value in (("acc_scale_bits", _bits(0.25)), ("acc_act", 1), ("pool_stride", 2)):
        changed = copy.deepcopy(rows)
        changed[0]["decoded"][field] = value
        with pytest.raises(ValueError, match="identity scale and no activation"):
            CA.check_modes(changed, {"i32_exit": True}, parameters, target="synthetic")
    # Without a runtime sibling the historical i32_exit check is unchanged.
    changed[0]["decoded"]["acc_scale_bits"] = _bits(0.25)
    CA.check_modes(changed, {"i32_exit": True}, {"i32_exit": dict(outputs=[_operand(4)])}, target="synthetic")
