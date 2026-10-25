"""Composition credit needs source identities and the mechanism in that region."""

import copy
import hashlib
import json
import shutil
import struct
import subprocess
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
    if kind == "MVOUT":
        decoded = {"acc_addr": 32, "rows": 3, "cols": 3, "readout": "i8", **decoded}
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


def test_native_composition_replays_pinned_source_manifest_and_executed_sequence(tmp_path, monkeypatch, native_files):
    import yaml

    from merlin.targetgen.native_model_execution import _digest

    elf, retired = native_files
    trace, expected = _readout_case()
    codes = {"CONFIG_ST": 7, "PRELOAD": 5, "COMPUTE_PRELOADED": 9, "MVOUT": 3}
    decoded_fields = {}
    words = []
    for row in trace["instructions"]:
        if row["class"] == "FENCE":
            continue
        code = codes[row["class"]]
        decoded_fields[code] = (row["class"], copy.deepcopy(row["decoded"]))
        word = (code << 25) | 43
        row.update(
            pc=len(words) * 4,
            word=word,
            funct=code,
            rs1={"kind": "const", "raw": code},
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
    result = NA.audit_native_composition(native, policy=policy, facts=facts_document, target="synthetic")
    assert result["trace_check"]["status"] == "pass"
    # A byte-pinned but reordered decode cannot reuse an otherwise clean counter.
    trace["instructions"][1]["pc"] = 4
    path = tmp_path / "composition_trace.json"
    path.write_text(json.dumps(trace))
    native["composition_trace"] = _digest(path)
    with pytest.raises(ValueError, match="PC/word/kernel sequence"):
        NA.audit_native_composition(native, policy=policy, facts=facts_document, target="synthetic")


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
