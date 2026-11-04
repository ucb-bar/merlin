"""Neutral storage/call controls; no target or compiler qualification."""

import base64
import copy
import shutil
import struct
import subprocess

import pytest

from merlin.common.paths import runtime_dir
from merlin.runtime import harness_render as H
from merlin.runtime.direct_kernel_counter import DirectKernelCounterPlan
from merlin.runtime.direct_kernel_harness import DirectKernelAbi, render_direct_kernel
from merlin.runtime.direct_kernel_invocation import DirectKernelInvocationPlan
from merlin.runtime.out_b64 import OutB64Decoder
from merlin.targetgen.contract.compile_only import CompileOnlySourceAbi, CompileOnlyTensor
from merlin.targetgen.contract.readback_policy import COHERENT_DUMP_V1, FULL_VALUES_B64, ReadbackPolicy


@pytest.fixture
def contract():
    return {
        "harness_abi": {
            "version": 2,
            "kind": "logical_pointer",
            "entry_symbol": "control_entry",
            "fence_symbol": "control_complete",
            "tensor_alignment": 8,
            "byte_order": "little",
            "main_convention": "void",
            "readback_transport": FULL_VALUES_B64,
            "prelude_symbol": "control_prelude",
        }
    }


@pytest.fixture
def program():
    return {
        "tensors": {
            "X": {"shape": [2, 3], "dtype": "i64", "role": "input"},
            "Y": {"shape": [2, 3], "dtype": "i64", "role": "output"},
        }
    }, {"X": [[-(1 << 63), (1 << 53) + 1, -7], [0, (1 << 63) - 1, 13]]}


def test_ordinary_semantic_roster_uses_exact_logical_extent_and_input_values(contract, program):
    cb, inputs = program
    before = copy.deepcopy((cb, inputs, contract))
    source = H.render_harness(cb, target="synthetic", inputs=inputs, contract=contract)
    assert "static unsigned char tensor_0[48]" in source
    assert "static unsigned char tensor_1[48]" in source
    assert "control_entry(tensor_0, tensor_1);" in source
    assert (
        source.index("control_prelude();") < source.index("merlin_poison_index") < source.index("control_entry(tensor_")
    )
    assert source.index("control_complete();") < source.index("OUT_B64_BEGIN v1 Y 2 3 8 s")
    assert (cb, inputs, contract) == before
    raw = struct.pack("<6q", *(v for row in inputs["X"] for v in row))
    assert ",".join(f"0x{byte:02x}" for byte in raw) in source


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "legacy",
        "bool_version",
        "wrong_kind",
        "headers",
        "externs",
        "no_fence",
        "readback",
        "alignment",
        "prelude_overlap",
    ],
)
def test_incomplete_or_compiler_bearing_contract_refuses_without_rendering(contract, monkeypatch, mutation):
    if mutation == "missing":
        contract.clear()
    elif mutation == "legacy":
        contract["harness_abi"]["version"] = 1
    elif mutation == "bool_version":
        contract["harness_abi"]["version"] = True
    elif mutation == "wrong_kind":
        contract["harness_abi"]["kind"] = "packed_pointer"
    elif mutation == "headers":
        contract["harness_abi"]["includes"] = ["compute.h"]
    elif mutation == "externs":
        contract["harness_abi"]["extern_decls"] = ["void hidden_compute(void);"]
    elif mutation == "no_fence":
        contract["harness_abi"].pop("fence_symbol")
    elif mutation == "readback":
        contract["harness_abi"]["readback_transport"] = "sampled"
    elif mutation == "alignment":
        contract["harness_abi"]["tensor_alignment"] = 3
    else:
        contract["harness_abi"]["prelude_symbol"] = "control_entry"
    monkeypatch.setattr(H, "render_direct_kernel", lambda *a, **k: pytest.fail("renderer reached"))
    with pytest.raises(ValueError):
        H.validate_contract(contract)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_input",
        "extra_input",
        "short",
        "ragged",
        "integer_float",
        "integer_bool",
        "overflow",
        "padding",
        "strides",
        "offset",
        "dynamic",
        "role",
        "derive",
        "packed",
        "precomputed",
        "conflicting_preload",
    ],
)
def test_incomplete_or_transformed_storage_refuses(contract, program, mutation):
    cb, inputs = copy.deepcopy(program)
    if mutation == "missing_input":
        inputs.clear()
    elif mutation == "extra_input":
        inputs["extra"] = [1]
    elif mutation == "short":
        inputs["X"].pop()
    elif mutation == "ragged":
        inputs["X"][1].pop()
    elif mutation == "integer_float":
        inputs["X"][0][0] = 1.0
    elif mutation == "integer_bool":
        inputs["X"][0][0] = True
    elif mutation == "overflow":
        inputs["X"][0][0] = 1 << 63
    elif mutation == "padding":
        cb["tensors"]["X"]["storage_shape"] = [2, 4]
    elif mutation == "strides":
        cb["tensors"]["X"]["strides"] = [4, 1]
    elif mutation == "offset":
        cb["tensors"]["X"]["offset"] = 1
    elif mutation == "dynamic":
        cb["tensors"]["X"]["shape"][0] = -1
    elif mutation == "role":
        cb["tensors"]["X"]["role"] = "intermediate"
    elif mutation == "derive":
        cb["params"] = {"im2col_recipes": [{"source": "X", "target": "X"}]}
    elif mutation == "packed":
        cb["tensors"]["X"]["dtype"] = "i4"
    elif mutation == "precomputed":
        cb["tensors"]["Y"]["data"] = [[1, 2, 3], [4, 5, 6]]
    else:
        cb["tensors"]["X"]["preload_b64"] = base64.b64encode(bytes(48)).decode()
    with pytest.raises(ValueError):
        H.render_harness(cb, target="synthetic", inputs=inputs, contract=contract)


def _explicit(cb):
    cb["kernel_abi"] = {
        "kind": "whole_program",
        "args": [
            {"tensor": "Y", "access": "write"},
            {"tensor": "X", "access": "read"},
        ],
        "outputs": ["Y"],
    }


def test_explicit_pointer_order_is_preserved_and_source_declaration_is_optional(contract, program):
    cb, inputs = copy.deepcopy(program)
    _explicit(cb)
    source = H.render_harness(cb, target="synthetic", inputs=inputs, contract=contract)
    assert "tensor_0[48]" in source and "tensor_1[48]" in source
    assert "((volatile unsigned char*)tensor_0)[merlin_poison_index]=165" in source
    original = CompileOnlySourceAbi((CompileOnlyTensor("X", (2, 3), "i64"),), (CompileOnlyTensor("Y", (2, 3), "i64"),))
    assert source == H.render_harness(cb, target="synthetic", inputs=inputs, contract=contract, original_abi=original)
    wrong = CompileOnlySourceAbi(original.inputs, (CompileOnlyTensor("Y", (3, 2), "i64"),))
    with pytest.raises(ValueError):
        H.render_harness(cb, target="synthetic", inputs=inputs, contract=contract, original_abi=wrong)


@pytest.mark.parametrize(
    "mutation", ["missing_pointer", "duplicate_pointer", "missing_output", "extra_output", "global_order"]
)
def test_explicit_order_and_complete_outputs_cannot_be_substituted(contract, program, mutation):
    cb, inputs = copy.deepcopy(program)
    _explicit(cb)
    if mutation == "missing_pointer":
        cb["kernel_abi"]["args"].pop()
    elif mutation == "duplicate_pointer":
        cb["kernel_abi"]["args"].append(dict(cb["kernel_abi"]["args"][0]))
    elif mutation == "missing_output":
        cb["kernel_abi"]["outputs"].clear()
    elif mutation == "extra_output":
        cb["kernel_abi"]["outputs"].append("X")
    else:
        cb["params"] = {"global_program_plan": {"entry_bindings": ["Y"]}}
    with pytest.raises(ValueError):
        H.render_harness(cb, target="synthetic", inputs=inputs, contract=contract)


@pytest.mark.parametrize(
    "dtype,bits",
    [
        ("f16", bytes.fromhex("00800100")),
        ("bf16", bytes.fromhex("0080a17f")),
        ("f32", struct.pack("<2I", 0x80000000, 0x7FA54321)),
        ("f64", struct.pack("<2Q", 0x8000000000000000, 0x7FF0000000000001)),
    ],
)
def test_exact_floating_bits_are_transported_without_numeric_conversion(contract, dtype, bits):
    cb = {
        "tensors": {
            "X": {"shape": [2], "dtype": dtype, "role": "input"},
            "Y": {"shape": [2], "dtype": dtype, "role": "output"},
        }
    }
    source = H.render_harness(cb, target="synthetic", inputs={"X": bits}, contract=contract)
    assert ",".join(f"0x{value:02x}" for value in bits) in source


@pytest.mark.parametrize("value", [0.1, float("inf"), float("nan"), 1, True])
def test_float_values_cannot_silently_round_or_lose_bits(contract, value):
    cb = {
        "tensors": {
            "X": {"shape": [1], "dtype": "f32", "role": "input"},
            "Y": {"shape": [1], "dtype": "f32", "role": "output"},
        }
    }
    with pytest.raises(ValueError):
        H.render_harness(cb, target="synthetic", inputs={"X": [value]}, contract=contract)


def test_complete_budget_refuses_before_input_bytes_or_values_are_encoded(contract, program, monkeypatch):
    cb, inputs = program
    monkeypatch.setattr(H, "_payload", lambda *a: pytest.fail("payload allocation reached"))
    with pytest.raises(ValueError, match="storage budget"):
        H.render_harness(cb, target="synthetic", inputs=inputs, contract=contract, max_storage_bytes=95)


def test_existing_flat_typed_input_projection_and_source_data_are_checked(contract, program):
    cb, inputs = copy.deepcopy(program)
    flat = [value for row in inputs["X"] for value in row]
    typed = {"X": {"shape": [2, 3], "dtype": "i64", "values": flat}}
    cb["tensors"]["X"]["data"] = inputs["X"]
    expected = H.render_harness(cb, target="synthetic", inputs=inputs, contract=contract)
    assert expected == H.render_harness(cb, target="synthetic", inputs=typed, contract=contract)
    assert expected == H.render_harness(cb, target="synthetic", inputs={"X": flat}, contract=contract)
    typed["X"]["values"][5] += 1
    with pytest.raises(ValueError, match="source values"):
        H.render_harness(cb, target="synthetic", inputs=typed, contract=contract)


def test_exact_signed_zero_numeric_projection_does_not_change_the_wire_bits(contract):
    cb = {
        "tensors": {
            "X": {"shape": [2], "dtype": "f32", "role": "input"},
            "Y": {"shape": [2], "dtype": "f32", "role": "output"},
        }
    }
    values = H.render_harness(cb, target="synthetic", inputs={"X": [-0.0, 1.25]}, contract=contract)
    bits = H.render_harness(cb, target="synthetic", inputs={"X": struct.pack("<2f", -0.0, 1.25)}, contract=contract)
    assert values == bits


def test_existing_repeats_counters_and_prelude_keep_poison_outside_entry_brackets(contract, program):
    cb, inputs = copy.deepcopy(program)
    cb["kernel_abi"] = {
        "kind": "whole_program",
        "args": [{"tensor": "X", "access": "read"}, {"tensor": "Y", "access": "write"}],
        "outputs": ["Y"],
    }
    original = CompileOnlySourceAbi((CompileOnlyTensor("X", (2, 3), "i64"),), (CompileOnlyTensor("Y", (2, 3), "i64"),))
    plan = DirectKernelInvocationPlan(original, 3, "history", "completed")
    counter = DirectKernelCounterPlan("read_counter", "read_state", "samples", 2, 1024)
    contract["harness_abi"]["readback_transport"] = COHERENT_DUMP_V1
    source = H.render_harness(
        cb, target="synthetic", inputs=inputs, contract=contract, invocation_plan=plan, counter_plan=counter
    )
    start = source.index("for(uint64_t invocation=")
    poison = source.index("merlin_poison_index", start)
    bracket = source.index("counter_state_before=", start)
    call = source.index("control_entry(tensor_", start)
    completion = source.index("control_complete();", start)
    assert start < poison < bracket < call < completion < source.index("counter_end=", start)
    assert source.count("control_prelude();") == 1
    assert source.index("control_prelude();") < source.index("counter_index=0")
    assert "history_0[144]" in source and "history_0[invocation*UINT64_C(48)+byte]" in source
    with pytest.raises(ValueError, match="storage budget"):
        H.render_harness(
            cb,
            target="synthetic",
            inputs=inputs,
            contract=contract,
            invocation_plan=plan,
            counter_plan=counter,
            max_storage_bytes=400,
        )


def test_direct_renderer_legacy_defaults_and_explicit_poison_validation(program):
    cb, inputs = copy.deepcopy(program)
    _explicit(cb)
    raw = struct.pack("<6q", *(v for row in inputs["X"] for v in row))
    cb["tensors"]["X"]["preload_b64"] = base64.b64encode(raw).decode()
    abi = DirectKernelAbi("control_entry", None, 8, "little", "void")
    kwargs = {"inputs": {"X": {}}, "readback_policy": ReadbackPolicy(FULL_VALUES_B64), "abi": abi}
    source = render_direct_kernel(cb, **kwargs)
    assert source == render_direct_kernel(cb, **kwargs, output_poison=None, prelude_symbol=None)
    assert "merlin_poison_index" not in source
    for bad in (True, -1, 256, "165"):
        with pytest.raises(ValueError, match="poison"):
            render_direct_kernel(cb, **kwargs, output_poison=bad)


@pytest.mark.parametrize("byte_order", ["little", "big"])
def test_owned_host_call_returns_every_tail_word_and_poisoned_missing_slot(tmp_path, contract, program, byte_order):
    compiler = shutil.which("cc", path="/usr/bin:/bin")
    if compiler is None:
        pytest.skip("requires an owned harmless host C control")
    cb, inputs = program
    contract["harness_abi"]["byte_order"] = byte_order
    harness = tmp_path / "harness.c"
    harness.write_text(H.render_harness(cb, target="synthetic", inputs=inputs, contract=contract))
    (tmp_path / "htif.h").write_text("void console_init(void); void htif_puts(const char*); void htif_exit(int);\n")
    (tmp_path / "out_b64.h").write_bytes((runtime_dir() / "baremetal/out_b64.h").read_bytes())
    control = tmp_path / "control.c"
    control.write_text(
        "#include <stdio.h>\n#include <stdlib.h>\n#include <string.h>\n"
        "static int prelude,completed;void console_init(void){}\n"
        "void control_prelude(void){prelude++;}\n"
        "void control_entry(void*x,void*y){if(prelude!=1)exit(5);memcpy(y,x,40);}\n"
        "void control_complete(void){completed++;}\n"
        "void htif_puts(const char*s){if(completed!=1)exit(6);fputs(s,stdout);}\n"
        "void htif_exit(int code){exit(code);}\nint harness_main(void);int main(void){return harness_main();}\n"
    )
    obj, elf = tmp_path / "harness.o", tmp_path / "control.elf"
    env = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"}
    commands = [
        [compiler, "-std=c11", "-O2", "-Dmain=harness_main", "-c", str(harness), "-o", str(obj)],
        [compiler, "-std=c11", "-O2", str(control), str(obj), "-o", str(elf)],
        [str(elf)],
    ]
    for argv in commands:
        result = subprocess.run(argv, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=10, check=True)
    outputs, decoder = {}, OutB64Decoder()
    for line in result.stdout.splitlines():
        decoder.consume(line.split(), outputs)
    assert result.stdout.splitlines()[-1] == "DONE"
    poison = int.from_bytes(b"\xa5" * 8, "little", signed=True)
    assert outputs == {"Y": [inputs["X"][0], [*inputs["X"][1][:2], poison]]}


def test_owned_repeated_host_calls_poison_again_and_keep_every_history_byte(tmp_path, contract, program):
    compiler = shutil.which("cc", path="/usr/bin:/bin")
    if compiler is None:
        pytest.skip("requires an owned harmless host C control")
    cb, inputs = copy.deepcopy(program)
    cb["kernel_abi"] = {
        "kind": "whole_program",
        "args": [{"tensor": "X", "access": "read"}, {"tensor": "Y", "access": "write"}],
        "outputs": ["Y"],
    }
    source_abi = CompileOnlySourceAbi(
        (CompileOnlyTensor("X", (2, 3), "i64"),), (CompileOnlyTensor("Y", (2, 3), "i64"),)
    )
    plan = DirectKernelInvocationPlan(source_abi, 2, "history", "completed_count")
    contract["harness_abi"]["readback_transport"] = COHERENT_DUMP_V1
    harness = tmp_path / "harness.c"
    harness.write_text(H.render_harness(cb, target="synthetic", inputs=inputs, contract=contract, invocation_plan=plan))
    (tmp_path / "htif.h").write_text("void console_init(void);void htif_puts(const char*);void htif_exit(int);\n")
    control = tmp_path / "control.c"
    control.write_text(
        "#include <stdio.h>\n#include <stdlib.h>\n#include <string.h>\n"
        "extern unsigned char history_0[96];extern volatile unsigned char completed_count[8];\n"
        "static int prelude,calls,completed;void console_init(void){}\n"
        "void control_prelude(void){prelude++;}\n"
        "void control_entry(void*x,void*y){if(prelude!=1)exit(5);calls++;memcpy(y,x,calls==1?48:40);}\n"
        "void control_complete(void){completed++;}\n"
        "void htif_puts(const char*s){if(completed!=2)exit(6);fputs(s,stderr);}\n"
        "void htif_exit(int code){if(code)exit(code);fwrite(history_0,1,96,stdout);"
        "for(unsigned i=0;i<8;i++)fputc(completed_count[i],stdout);exit(0);}\n"
        "int harness_main(void);int main(void){return harness_main();}\n"
    )
    obj, elf = tmp_path / "harness.o", tmp_path / "control.elf"
    env = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"}
    for argv in [
        [compiler, "-std=c11", "-O2", "-Dmain=harness_main", "-c", str(harness), "-o", str(obj)],
        [compiler, "-std=c11", "-O2", str(control), str(obj), "-o", str(elf)],
        [str(elf)],
    ]:
        result = subprocess.run(argv, cwd=tmp_path, env=env, capture_output=True, timeout=10, check=True)
    assert result.stderr == b"DONE\n" and len(result.stdout) == 104
    observed = plan.decode(
        {"history_0": result.stdout[:96], "completed_count": result.stdout[96:]},
        cb=cb,
        entry_symbol="control_entry",
        completion_symbol="control_complete",
        byte_order="little",
    )
    raw = struct.pack("<6q", *(value for row in inputs["X"] for value in row))
    assert observed.observed_count == 2
    assert observed.output_bytes == (("Y", (raw, raw[:40] + b"\xa5" * 8)),)


def _select_scatter_chain():
    """A raw-interface select_scatter: S reshaped to [1, 4], then inserted at row 1 of D."""
    return {
        "tensors": {
            "D": {"shape": [2, 4], "dtype": "i8", "role": "input"},
            "S": {"shape": [4], "dtype": "i8", "role": "input"},
            "S1": {"shape": [1, 4], "dtype": "i8", "role": "intermediate"},
            "Y0": {"shape": [2, 4], "dtype": "i8", "role": "output"},
        },
        "commands": [
            {"opcode": "MOVEMENT", "operands": {"src": "S", "dst": "S1"}, "attributes": {"form": "reshape"}},
            {"opcode": "FENCE", "operands": {}, "attributes": {}},
            {"opcode": "INSERT_SLICE", "operands": {"dest": "D", "src": "S1", "dst": "Y0"},
             "attributes": {"offsets": [1, 0], "strides": [1, 1], "output_dtype": "i8"}},
        ],
        "params": {},
        "outputs": ["Y0"],
    }, {"D": [[1, 2, 3, 4], [5, 6, 7, 8]], "S": [-1, -2, -3, -4]}


def _gradient_chain():
    """A raw-interface convolution backward: three results, four produced intermediates."""
    i8, i32 = "i8", "i32"
    return {
        "tensors": {
            "Xcol": {"shape": [16, 36], "dtype": i8, "role": "input"},
            "dY": {"shape": [16, 16], "dtype": i8, "role": "input"},
            "W": {"shape": [36, 16], "dtype": i8, "role": "weight"},
            "xt": {"shape": [36, 16], "dtype": i8, "role": "intermediate"},
            "grad_weight": {"shape": [36, 16], "dtype": i32, "role": "output"},
            "dyt": {"shape": [16, 16], "dtype": i8, "role": "intermediate"},
            "gic": {"shape": [36, 16], "dtype": i8, "role": "intermediate"},
            "gic3": {"shape": [1, 36, 16], "dtype": i8, "role": "intermediate"},
            "grad_input": {"shape": [1, 4, 4, 4], "dtype": i32, "role": "output"},
            "grad_bias": {"shape": [1, 16], "dtype": i32, "role": "output"},
        },
        "commands": [
            {"opcode": "MOVEMENT", "operands": {"src": "Xcol", "dst": "xt"}, "attributes": {"form": "permute"}},
            {"opcode": "RES_PACK", "operands": {"src": "dY", "dst": "dY_res"}, "attributes": {}},
            {"opcode": "MATMUL_RESIDENT", "operands": {"lhs": "xt", "rhs": "dY_res", "dst": "acc_w"}},
            {"opcode": "COMMIT", "operands": {"src": "acc_w", "dst": "grad_weight"}, "attributes": {"output_dtype": i32}},
            {"opcode": "MOVEMENT", "operands": {"src": "dY", "dst": "dyt"}, "attributes": {"form": "permute"}},
            {"opcode": "RES_PACK", "operands": {"src": "dyt", "dst": "dyt_res"}, "attributes": {}},
            {"opcode": "MATMUL_RESIDENT", "operands": {"lhs": "W", "rhs": "dyt_res", "dst": "acc_i"}},
            {"opcode": "COMMIT", "operands": {"src": "acc_i", "dst": "gic"}, "attributes": {"output_dtype": i8}},
            {"opcode": "MOVEMENT", "operands": {"src": "gic", "dst": "gic3"}, "attributes": {"form": "reshape"}},
            {"opcode": "WINDOW_SUM", "operands": {"src": "gic3", "dst": "grad_input"}, "attributes": {}},
            {"opcode": "WINDOW_SUM", "operands": {"src": "dY", "dst": "grad_bias"}, "attributes": {}},
        ],
        "params": {},
        "outputs": ["grad_weight", "grad_input", "grad_bias"],
    }


def test_produced_intermediate_is_compiler_owned_under_the_contract_default_roster(contract):
    from merlin.targetgen.contract.harness_render import kernel_arg_order

    cb, inputs = _select_scatter_chain()
    before = copy.deepcopy(cb)
    normalized, slots = H._roster(cb)
    # logical_kernel_abi.default: inputs in declaration order, then results; an intermediate is never a pointer.
    assert list(slots) == kernel_arg_order(cb) == ["D", "S", "Y0"]
    assert "S1" not in normalized["tensors"] and H.logical_output_names(cb) == ("Y0",)
    source = H.render_harness(cb, target="synthetic", inputs=inputs, contract=contract)
    assert "control_entry(tensor_0, tensor_1, tensor_2);" in source
    assert "tensor_3" not in source  # no harness storage for the kernel's own intermediate
    assert cb == before


def test_multi_result_gradient_chain_keeps_the_contract_result_order(contract):
    from merlin.targetgen.contract.harness_render import kernel_arg_order

    cb = _gradient_chain()
    _, slots = H._roster(cb)
    expected = ["Xcol", "dY", "W", "grad_weight", "grad_input", "grad_bias"]
    assert list(slots) == kernel_arg_order(cb) == expected
    assert [slots[name] for name in expected] == ["read"] * 3 + ["write"] * 3
    inputs = {name: [[0] * cb["tensors"][name]["shape"][1]] * cb["tensors"][name]["shape"][0] for name in expected[:3]}
    source = H.render_harness(cb, target="synthetic", inputs=inputs, contract=contract)
    assert "control_entry(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5);" in source
    assert "tensor_6" not in source


def test_role_inference_agrees_with_the_contract_on_an_ordinary_roster(program):
    from merlin.targetgen.contract.harness_render import kernel_arg_order

    cb, _ = program
    assert list(H._roster(cb)[1]) == kernel_arg_order(cb) == ["X", "Y"]


@pytest.mark.parametrize(
    "mutation", ["unproduced", "declared_result", "no_declared_outputs", "source_bytes", "scratch_grant", "order"]
)
def test_compiler_owned_intermediate_refuses_outside_the_exact_default_rule(contract, mutation):
    cb, inputs = _select_scatter_chain()
    kwargs = {}
    if mutation == "unproduced":
        cb["commands"] = [c for c in cb["commands"] if c["operands"].get("dst") != "S1"]
    elif mutation == "declared_result":
        cb["outputs"] = ["S1", "Y0"]
    elif mutation == "no_declared_outputs":
        cb.pop("outputs")  # the reference publishes the MOVEMENT dst S1: the two orders disagree
    elif mutation == "source_bytes":
        cb["tensors"]["S1"]["data"] = [[0, 0, 0, 0]]
    elif mutation == "scratch_grant":
        kwargs["source_owned_mutables"] = ("S1",)
    else:
        cb["tensors"] = {name: cb["tensors"][name] for name in ("Y0", "S1", "D", "S")}
        cb["outputs"] = ["Y0"]
        cb["commands"].append({"opcode": "COMMIT", "operands": {"src": "acc", "dst": "D"}})
    with pytest.raises(ValueError):
        H.render_harness(cb, target="synthetic", inputs=inputs, contract=contract, **kwargs)
