"""Immutable command parameters never excuse host tensor arithmetic or payload."""

from types import SimpleNamespace

import pytest
from xdsl.context import Context
from xdsl.dialects import builtin, llvm
from xdsl.parser import Parser

from merlin.targetgen.rocc import decode
from merlin.targetgen.rocc.normalized_operands import BoundOperandObserver
from merlin.verify import host_compute_audit as HA


def _observer(monkeypatch):
    facts = {
        "facts": {
            "interfaces": [
                {
                    "name": "register_bundle_layouts",
                    "bundles": {
                        "Selector": {"fields": {"kind": {"offset": 0, "width": 2}}},
                        "Parameter": {"fields": {"scale": {"offset": 32, "width": 32}}},
                    },
                }
            ]
        }
    }
    raw = {"CUSTOM_OPCODE": 123, "FUNCT3": 3, "FUNCT_CLASS": {0: "CONFIG", 4: "COMPUTE"}}
    protocol = {
        "version": 1,
        "encoding": {"semantic_class": raw["FUNCT_CLASS"], "config_subtype": {2: "CONFIG_PARAMETER"}},
        "roles": {
            "config": {"class": "CONFIG", "selector": {"operand": "rs1", "bundle": "Selector", "field": "kind"}},
            "classes": {
                "CONFIG_PARAMETER": {
                    "fields": [{"name": "scale", "operand": "rs2", "bundle": "Parameter", "field": "scale"}]
                }
            },
        },
    }
    observer = BoundOperandObserver(target="fixture", protocol=protocol, facts=facts, raw_isa=raw, word_bits=64)
    monkeypatch.setattr(decode, "_semantics", lambda target: SimpleNamespace(operand_observer=observer))
    return observer


_SOURCE = """module {
  llvm.mlir.global internal constant @parameters(dense<[1056964608, 1040187392, 1061158912, 1056964608]>
    : tensor<4xi32>) {addr_space = 0 : i32, alignment = 64 : i64} : !llvm.array<4 x i32>
  llvm.func @entry(%input: !llvm.ptr, %output: !llvm.ptr) {
    %zero = llvm.mlir.constant(0 : i64) : i64
    %one = llvm.mlir.constant(1 : i64) : i64
    %count = llvm.mlir.constant(4 : i64) : i64
    %shift = llvm.mlir.constant(32 : i64) : i64
    %selector = llvm.mlir.constant(2 : i64) : i64
    %stride = llvm.mlir.constant(33 : i64) : i64
    %table = llvm.mlir.addressof @parameters : !llvm.ptr
    llvm.br ^head(%zero : i64)
  ^head(%i: i64):
    %condition = llvm.icmp "ult" %i, %count : i64
    llvm.cond_br %condition, ^body, ^exit
  ^body:
    %pointer = llvm.getelementptr %table[%i] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %bits = llvm.load %pointer : !llvm.ptr -> i32
    %wide = llvm.zext %bits : i32 to i64
    %shifted = llvm.shl %wide, %shift : i64
    %packed = llvm.or %shifted, %stride : i64
    "llvm.inline_asm"(%selector, %packed) <{asm_string = ".insn r 0x7b, 3, 0, x0, $0, $1",
      constraints = "r,r,~{memory}", has_side_effects}> : (i64, i64) -> ()
    %next = llvm.add %i, %one : i64
    llvm.br ^head(%next : i64)
  ^exit:
    llvm.return
  }
}"""


def _audit(source=_SOURCE, *, target="fixture"):
    module = _module(source)
    return HA.audit(module, [HA.GroupSite(0, "unit", "entry", 4)], target=target)["groups"][0]


def _module(source):
    ctx = Context()
    ctx.load_dialect(builtin.Builtin)
    ctx.load_dialect(llvm.LLVM)
    module = Parser(ctx, source).parse_module()
    module.verify()
    return module


def test_selected_immutable_parameters_exclude_only_the_proved_loads_and_packing(monkeypatch):
    _observer(monkeypatch)
    result = _audit()
    assert result["verdict"] == HA.CLEAN, result
    assert result["host_raw_payload_bytes"] == 16 and result["host_payload_bytes"] == 0
    proof = result["command_parameters"]
    assert proof["status"] == "proven", proof
    assert (proof["payload_bytes"], proof["arithmetic"], proof["conversions"]) == (16, 8, 4)
    assert proof["globals"] == ["parameters"] and proof["configuration_commands"] == 4
    assert proof["configuration_fields"] == ["CONFIG_PARAMETER.scale"]
    assert proof["selected_config_selector"] == {"operand": "rs1", "offset": 0, "width": 2}
    assert result["host_total_arithmetic"] > result["host_arithmetic"]


@pytest.mark.parametrize(
    "old,new",
    [
        ("internal constant", "internal"),
        (".insn r 0x7b, 3, 0, x0, $0, $1", ".insn r 0x7b, 3, 4, x0, $0, $1"),
        (".insn r 0x7b, 3, 0, x0, $0, $1", ".insn r 0x7b, 3, 0, x0, $1, $0"),
        (".insn r 0x7b, 3, 0, x0, $0, $1", ".insn r 0x7b, 3, 0, x0, $0, $1; add a0,a0,a1"),
        ("%selector = llvm.mlir.constant(2", "%selector = llvm.mlir.constant(1"),
        ("%count = llvm.mlir.constant(4", "%count = llvm.mlir.constant(5"),
        ("llvm.or %shifted, %stride", "llvm.add %shifted, %stride"),
        ("%wide = llvm.zext %bits", "%changed = llvm.add %bits, %bits : i32\n    %wide = llvm.zext %changed"),
        ("%wide = llvm.zext %bits", "llvm.store %bits, %output : i32, !llvm.ptr\n    %wide = llvm.zext %bits"),
        ("llvm.br ^head(%zero", "%escape = llvm.ptrtoint %table : !llvm.ptr to i64\n    llvm.br ^head(%zero"),
        ("%pointer = llvm.getelementptr %table", "%pointer = llvm.getelementptr %input"),
        ("%shift = llvm.mlir.constant(32", "%shift = llvm.mlir.constant(64"),
        ("llvm.load %pointer :", "llvm.load %pointer atomic monotonic :"),
        ("llvm.shl %wide, %shift", "llvm.shl %wide, %wide"),
    ],
)
def test_tensor_work_or_unproved_configuration_provenance_is_not_exempt(monkeypatch, old, new):
    _observer(monkeypatch)
    result = _audit(_SOURCE.replace(old, new))
    assert result["verdict"] == HA.HOST_COMPUTE, result
    assert result["command_parameters"]["status"] == "unproven"
    assert result["host_payload_bytes"] == result["host_raw_payload_bytes"]


def test_missing_or_wrong_selected_observer_preserves_original_audit(monkeypatch):
    _observer(monkeypatch)
    assert _audit(target="other")["verdict"] == HA.HOST_COMPUTE
    monkeypatch.setattr(decode, "_semantics", lambda target: SimpleNamespace())
    assert _audit()["verdict"] == HA.HOST_COMPUTE
    assert _audit(target=None)["verdict"] == HA.HOST_COMPUTE


def test_additional_use_in_another_function_refuses_the_constant_exemption(monkeypatch):
    _observer(monkeypatch)
    source = (
        _SOURCE[:-1]
        + """llvm.func @other() {
      %table = llvm.mlir.addressof @parameters : !llvm.ptr
      llvm.return
    }}"""
    )
    assert _audit(source)["verdict"] == HA.HOST_COMPUTE


def test_other_tensor_payload_remains_charged_beside_proven_command_parameters(monkeypatch):
    _observer(monkeypatch)
    source = _SOURCE.replace(
        "%next = llvm.add",
        """%payload = llvm.load %input : !llvm.ptr -> i32
    %computed = llvm.add %payload, %payload : i32
    llvm.store %computed, %output : i32, !llvm.ptr
    %next = llvm.add""",
    )
    result = _audit(source)
    assert result["command_parameters"]["status"] == "proven"
    assert result["verdict"] == HA.HOST_COMPUTE
    assert result["host_raw_payload_bytes"] == 48 and result["host_payload_bytes"] == 32


def test_reserved_bits_are_not_selected_configuration_fields(monkeypatch):
    observer = _observer(monkeypatch)
    observer._isa["OPERAND_FIELDS"]["CONFIG_PARAMETER"]["fields"][0]["width"] = 16
    result = _audit()
    assert result["verdict"] == HA.HOST_COMPUTE
    assert "no selected source operand field" in result["command_parameters"]["detail"]


def test_proof_bound_exhaustion_does_not_turn_unknown_into_zero_payload(monkeypatch):
    _observer(monkeypatch)
    from merlin.perf.host_cfg_activity import analyze_host_cfg_activity
    from merlin.verify.host_command_parameters import prove_command_parameters

    ctx = Context()
    ctx.load_dialect(builtin.Builtin)
    ctx.load_dialect(llvm.LLVM)
    module = Parser(ctx, _SOURCE).parse_module()
    function = HA._functions(module)["entry"]
    activity = analyze_host_cfg_activity(function)
    activity["blocks"][0]["execution_count"] = 1000001
    assert prove_command_parameters(function, activity, target="fixture")["status"] == "unproven"


def test_loaded_bits_may_not_change_the_selected_configuration_subtype(monkeypatch):
    _observer(monkeypatch)
    source = _SOURCE.replace("(%selector, %packed)", "(%wide, %packed)")
    result = _audit(source)
    assert result["verdict"] == HA.HOST_COMPUTE
    assert result["command_parameters"]["status"] == "unproven"


def test_global_value_used_for_cpu_comparison_is_not_exempt(monkeypatch):
    _observer(monkeypatch)
    source = _SOURCE.replace("%next = llvm.add", '%compare = llvm.icmp "eq" %bits, %bits : i32\n    %next = llvm.add')
    assert _audit(source)["verdict"] == HA.HOST_COMPUTE


def test_unproved_shift_flags_preserve_original_charges(monkeypatch):
    _observer(monkeypatch)
    from xdsl.dialects.builtin import IntegerAttr

    module = _module(_SOURCE)
    function = HA._functions(module)["entry"]
    shift = next(op for op in function.walk() if op.name == "llvm.shl")
    shift.properties["isExact"] = IntegerAttr(1, 64)
    result = HA.audit(module, [HA.GroupSite(0, "unit", "entry", 4)], target="fixture")["groups"][0]
    assert result["verdict"] == HA.HOST_COMPUTE
    assert result["command_parameters"]["status"] == "unproven"


def test_census_retains_raw_totals_and_the_exact_parameter_proof(monkeypatch):
    _observer(monkeypatch)
    from merlin.targetgen.offload_census import host_compute_row

    buffer = {"tensors": {"out": {"shape": [4], "dtype": "i8"}}, "kernel_abi": {"outputs": ["out"]}}
    row = host_compute_row("fixture", buffer, _SOURCE, "offloaded", target="fixture")
    assert row["verdict"] == HA.CLEAN
    assert row["host_raw_payload_bytes"] == 16 and row["host_payload_bytes"] == 0
    assert row["command_parameters"]["configuration_commands"] == 4
