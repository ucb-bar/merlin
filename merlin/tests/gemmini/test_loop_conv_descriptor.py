"""Native descriptor packing against the actual header; no accelerator execution."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from merlin.common.paths import repo_root


ROOT = repo_root()
SPEC = importlib.util.spec_from_file_location(
    "native_loop_conv", ROOT / "merlin/targets/gemmini/backend/gemmini_loop_conv.py")
CONV = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONV)
INCLUDE = ROOT / "merlin/experiments/capsule_bench/targets/gemmini/contracts/harness_curated/gemmini-rocc-tests/include"
FACTS = ROOT / "merlin/targets/gemmini/contracts/rtl_facts/facts.json"


@pytest.fixture(scope="module")
def contract():
    facts = json.loads(FACTS.read_text())
    table = next(row for row in facts["facts"]["interfaces"] if row.get("name") == "funct_decode_table")
    hw = Path(table["hw_source"])
    if not hw.is_file():
        pytest.skip("pinned RTL bytes unavailable; never substitute a fabricated contract")
    return CONV.NativeConvContract(header=INCLUDE / "gemmini.h", params=INCLUDE / "gemmini_params.h",
                                   facts=FACTS, core_hw=hw)


def sample():
    return {
        "opcode": "CONV2D", "operands": {"ifm": "X", "weight": "W", "dst": "Y"},
        "attributes": {"kernel": [2, 2, 1, 1], "stride": [1, 1], "padding": [0, 0, 0, 0],
                       "dilation": [1, 1], "layout": "nhwc", "output_dtype": "i8", "epilogue": []},
    }, {"X": {"shape": [1, 3, 3, 1], "dtype": "i8"},
        "W": {"shape": [4, 1], "dtype": "i8"}, "Y": {"shape": [4, 1], "dtype": "i8"}}


def emit(contract, command=None, tensors=None):
    command0, tensors0 = sample()
    return CONV.emit_native_conv(command or command0, tensors or tensors0, contract=contract,
                                 pointers={"ifm": "x", "weight": "w", "dst": "y"},
                                 row_strides={"ifm": 1, "weight": 1, "dst": 1})


def test_actual_header_c_macro_matches_every_emitted_register(contract, tmp_path):
    """Independent C expression evaluation of the actual macro, not our AST evaluator."""
    cc = shutil.which("cc")
    if cc is None:
        pytest.skip("host C compiler unavailable")
    receipt = emit(contract)
    macro = contract.header.macro("gemmini_loop_conv_ws")
    numbers = {"x": 101, "w": 203, "y": 307}
    call = receipt["c_source"].splitlines()[-1]
    defines = "\n".join(f"#define {row['name'] if row['name'].startswith('k_') else 'k_' + row['name']} {row['funct']}"
                        for row in receipt["instructions"])
    source = ("#include <stdint.h>\n#include <stdio.h>\n#define XCUSTOM_ACC 0\n" + defines
              + '\n#define ROCC_INSTRUCTION_RS1_RS2(op,a,b,f) printf("%d %llu %llu\\n", (int)(f), (unsigned long long)(a), (unsigned long long)(b));\n'
              + "#define gemmini_loop_conv_ws(" + ",".join(macro.params) + ") " + macro.body + "\n"
              + "int main(void) { uint64_t x=101,w=203,y=307; " + call + " return 0; }\n")
    path = tmp_path / "packing.c"
    path.write_text(source)
    subprocess.run([cc, "-std=c11", str(path), "-o", str(tmp_path / "packing")], check=True, capture_output=True, timeout=10)
    actual = subprocess.run([str(tmp_path / "packing")], check=True, text=True, capture_output=True, timeout=5)
    expected = [(row["funct"], numbers.get(row["rs1"], row["rs1"]), numbers.get(row["rs2"], row["rs2"]))
                for row in receipt["instructions"]]
    assert [tuple(map(int, line.split())) for line in actual.stdout.splitlines()] == expected
    assert receipt["readout_proof"]["full_width_bit"] == 0
    assert receipt["numerical_runtime_qualification"] == "UNPROVEN"
    assert receipt["cycles"] == "UNKNOWN"


@pytest.mark.parametrize("attribute,value", [("epilogue", ["requant"]), ("epilogue", ["bias_add", "relu"]),
                                            ("stride", [2, 2]), ("dilation", [2, 2]),
                                            ("padding", [1, 1, 1, 1]), ("layout", "nchw"),
                                            ("narrowing", "modular")])
def test_unsupported_semantics_refuse_instead_of_coercing(contract, attribute, value):
    command, tensors = sample()
    command["attributes"][attribute] = value
    with pytest.raises(CONV.UnsupportedNativeConv):
        emit(contract, command, tensors)


def test_full_width_and_float_are_not_recast(contract):
    for dtype in ("i32", "f32"):
        command, tensors = sample()
        command["attributes"]["output_dtype"] = dtype
        tensors["Y"]["dtype"] = dtype
        with pytest.raises(CONV.UnsupportedNativeConv, match="full-width"):
            emit(contract, command, tensors)


def test_native_relu_and_dense_geometry(contract):
    command, tensors = sample()
    command["attributes"]["epilogue"] = ["relu"]
    receipt = emit(contract, command, tensors)
    assert receipt["parameters"]["activation"] == CONV._integer(contract.header, "RELU")
    assert receipt["capacity"]["input_rows"] == 9
    assert receipt["capacity"]["weight_rows"] == 4
    assert receipt["capacity"]["accumulator_rows"] == 4


def test_capacity_and_resident_handle_fail_closed(contract):
    command, tensors = sample()
    command["operands"]["weight"] = "unresolved_resident"
    with pytest.raises(CONV.UnsupportedNativeConv, match="unresolved resident"):
        emit(contract, command, tensors)
    command, tensors = sample()
    side = contract.capacity["max_acc_rows"] + 1
    tensors["X"]["shape"] = [1, side, side, 1]
    tensors["Y"]["shape"] = [(side - 1) ** 2, 1]
    with pytest.raises(CONV.UnsupportedNativeConv, match="capacity"):
        emit(contract, command, tensors)


def test_stale_hardware_refused_before_descriptor(tmp_path):
    bad = tmp_path / "changed.hw.mlir"
    bad.write_text("hw.module @wrong() {}")
    with pytest.raises(CONV.UnsupportedNativeConv, match="stale core hardware"):
        CONV.NativeConvContract(header=INCLUDE / "gemmini.h", params=INCLUDE / "gemmini_params.h",
                                facts=FACTS, core_hw=bad)


def test_changed_header_opcode_cannot_pass_hardware_table(contract):
    altered = deepcopy(contract)
    from dataclasses import replace
    macro = altered.header.macro("k_LOOP_CONV_WS")
    altered.header = replace(altered.header, macros=tuple(
        replace(item, body=str(macro.int_value + 1)) if item.name == macro.name else item
        for item in altered.header.macros))
    with pytest.raises(CONV.UnsupportedNativeConv, match="opcode mismatch"):
        emit(altered)


def test_field_overflow_is_not_silently_packed():
    with pytest.raises(CONV.UnsupportedNativeConv, match="field overflow"):
        CONV._expression("(a << 4) | b", {"a": 1, "b": 16})


def test_changed_header_field_cannot_pass_pinned_decoder(contract):
    from dataclasses import replace
    altered = deepcopy(contract)
    macro = altered.header.macro("gemmini_loop_conv_ws")
    wrong = macro.body.replace("(out_channels) << 48", "(out_channels) << 47")
    assert wrong != macro.body
    altered.header = replace(altered.header, macros=tuple(
        replace(item, body=wrong) if item.name == macro.name else item
        for item in altered.header.macros))
    with pytest.raises(CONV.UnsupportedNativeConv, match="pinned RTL descriptor mismatch"):
        emit(altered)


def test_real_entrypoint_native_selection_and_exact_default_fallback(contract):
    from merlin.runtime.backends.base import get_backend
    from merlin.targetgen.eval.gemmini_conformance import build
    codegen = get_backend("gemmini").gemmini_codegen_mlir
    command, tensors = sample()
    cb = {"abi_version": "0.1", "target": "gemmini", "commands": [command], "tensors": tensors, "outputs": ["Y"]}
    selection = {}
    text, arguments = codegen.emit_kernel_mlir(cb, native_conv_contract=contract,
                                               native_conv_selection_receipt=selection)
    assert selection["selection"] == "selected_explicit_opt_in"
    assert arguments == ["W", "X", "Y"]
    assert text.count(".insn r ") == len(selection["instructions"]) + len(selection["entry_instructions"])
    assert "im2col" not in text and selection["default_enabled"] is False
    proof = selection["descriptor_to_rtl_field_qualification"]
    assert proof["status"] == "verified_exact_descriptor_register_updates"
    assert proof["required_physical_address_bits"]
    default_cb = build("C0")
    before = codegen.emit_kernel_mlir(default_cb)
    refusal = {}
    after = codegen.emit_kernel_mlir(default_cb, native_conv_contract=contract,
                                   native_conv_selection_receipt=refusal)
    assert before == after
    assert refusal["selection"] == "refused_native_fallback_unchanged"
