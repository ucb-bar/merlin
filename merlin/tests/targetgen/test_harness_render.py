"""The solution-neutral runner-owned harness (logical kernel ABI v2), host side and, when the bare-metal
toolchain is present, end to end on spike against a naive reference kernel and its mutants."""

from __future__ import annotations

import importlib.util

import pytest

from merlin.common.paths import repo_root
from merlin.runtime.reference import reference_outputs
from merlin.targetgen.contract import harness_render as hr
from merlin.targetgen.contract.reference_kernel import MUTATIONS, render_reference_kernel

HOOKS = hr.HostHooks("probe_kernel", "cycle_window_probe", "rdcycle %0", "fence")


@pytest.fixture(scope="module")
def abi():
    return hr.logical_abi()


def render(cb, abi, **kw):
    return hr.render_with(cb, abi=abi, hooks=HOOKS, **kw)


def call_args(text):
    line = next(x.strip() for x in text.splitlines() if x.strip().startswith("probe_kernel("))
    return [a.strip().rpartition(")")[2][2:] for a in line[len("probe_kernel(") : -2].split(",")]


def resident(bias=False, pool=False):
    tensors = {
        "W1": {"shape": [16, 20], "dtype": "i8", "role": "weight"},
        "A": {"shape": [16, 16], "dtype": "i8", "role": "input"},
        "W0": {"shape": [16, 12], "dtype": "i8", "role": "weight"},
    }
    attrs = {"epilogue": [], "output_dtype": "i32"}
    if bias:
        tensors["B"] = {"shape": [20], "dtype": "i32", "role": "bias"}
        attrs = {"epilogue": ["bias_add", "relu"], "output_dtype": "i8", "bias": "B"}
    if pool:
        attrs = {
            "epilogue": ["maxpool"],
            "output_dtype": "i32",
            "pool_in_dims": [4, 4],
            "pool_size": [2, 2],
            "pool_stride": [2, 2],
            "pool_padding": [0, 0, 0, 0],
        }
    return {
        "abi_version": "0.1",
        "tensors": tensors,
        "commands": [
            {"opcode": "RES_PACK", "operands": {"src": "W0", "dst": "R0"}},
            {"opcode": "RES_PACK", "operands": {"src": "W1", "dst": "R1"}},
            {"opcode": "MATMUL_RESIDENT", "operands": {"lhs": "A", "rhs": "R1", "dst": "a1"}},
            {"opcode": "COMMIT", "operands": {"src": "a1", "dst": "Y1"}, "attributes": attrs},
            {"opcode": "MATMUL_RESIDENT", "operands": {"lhs": "A", "rhs": "R0", "dst": "a0"}},
            {
                "opcode": "COMMIT",
                "operands": {"src": "a0", "dst": "Y0"},
                "attributes": {"epilogue": [], "output_dtype": "i32"},
            },
        ],
    }


def test_contract_declares_the_logical_abi(abi):
    assert abi.version == 2
    assert abi.default_order == ("logical_inputs_in_declaration_order", "logical_outputs_in_result_order")
    assert abi.alignment_bytes & (abi.alignment_bytes - 1) == 0


def test_pointer_order_is_declaration_then_results_not_emitter_order(abi):
    text = render(resident(), abi)
    # Inputs in declaration order (W1 before A before W0), then results in program order.
    assert call_args(text) == ["W1", "A", "W0", "Y1", "Y0"]


def test_buffers_are_dense_logical_tensors_with_standard_headers_only(abi):
    text = render(resident(), abi)
    assert "static const int8_t T_W1[320]" in text  # 16x20: the logical extent
    assert "static int32_t T_Y0[192]" in text  # 16x12 result
    assert "T_Y0[i * 12 + j]" in text and 'printf("OUT Y0 16 12")' in text
    includes = [x for x in text.splitlines() if x.startswith("#include")]
    assert includes == ["#include <stdint.h>", "#include <stdio.h>"]
    assert 'fence" ::: "memory");' in text and "rdcycle %0" in text


def test_outputs_are_poisoned_before_every_invocation(abi, monkeypatch):
    monkeypatch.setenv("MERLIN_CACHE_STATE", "warm")
    text = render(resident(), abi)
    assert text.count(f"__builtin_memset(T_Y0, {abi.poison_byte}, sizeof(T_Y0));") == 2
    assert text.count("probe_kernel(") == 3  # declaration, warm call, measured call
    measured = text.split("uint64_t c0 = merlin_read_cycles();", 1)[1].split("uint64_t c1", 1)[0]
    assert "memset" not in measured


def test_result_shapes_come_from_declarations_or_merlin_derivation(abi):
    pooled = render(resident(pool=True), abi)
    assert "static int32_t T_Y1[80]" in pooled  # 16 rows pooled 4x4 -> 2x2 = 4 rows x 20
    biased = render(resident(bias=True), abi)
    assert call_args(biased) == ["W1", "A", "W0", "B", "Y1", "Y0"]
    assert "static int8_t T_Y1[320]" in biased  # the declared i8 readout container


def test_runner_derived_matrices_are_never_pointers(abi):
    cb = {
        "abi_version": "0.1",
        "tensors": {
            "IFM": {"shape": [1, 6, 6, 4], "dtype": "i8", "role": "input"},
            "W": {"shape": [36, 8], "dtype": "i8", "role": "weight"},
            "IFM_cols": {"shape": [16, 36], "dtype": "i8", "role": "input"},
        },
        "params": {"im2col_recipes": [{"source": "IFM", "target": "IFM_cols", "kh": 3, "kw": 3, "ci": 4}]},
        "commands": [
            {"opcode": "RES_PACK", "operands": {"src": "W", "dst": "R"}},
            {"opcode": "MATMUL_RESIDENT", "operands": {"lhs": "IFM_cols", "rhs": "R", "dst": "acc"}},
            {"opcode": "COMMIT", "operands": {"src": "acc", "dst": "Y"}, "attributes": {"output_dtype": "i32"}},
        ],
    }
    assert call_args(render(cb, abi)) == ["IFM", "W", "Y"]


def test_host_lane_results_without_a_producer_are_outputs(abi):
    cb = {
        "abi_version": "0.1",
        "commands": [],
        "tensors": {
            "x": {"shape": [3, 5], "dtype": "bf16", "role": "input"},
            "y": {"shape": [3, 5], "dtype": "f32", "role": "output"},
        },
    }
    text = render(cb, abi)
    assert call_args(text) == ["x", "y"]
    assert "static const uint16_t T_x[15]" in text and "static uint32_t T_y[15]" in text


def test_large_inputs_link_as_aligned_blobs(abi):
    cb = resident()
    cb["tensors"]["A"]["shape"] = [64, 16]
    cb["tensors"]["W1"]["shape"] = [16, 80]
    blobs = {}
    text = render(cb, abi, blobs=blobs)
    assert set(blobs) == {"T_W1", "T_A"} and "extern const int8_t T_W1[1280];" in text
    assert blobs["T_A"]["align"] == abi.alignment_bytes and len(blobs["T_A"]["bytes"]) == 1024


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"readback_policy": "coherent_dump_v1"}, "whole-program"),
        ({"warm_profile": object(), "source_owned_mutables": ("x",)}, "whole-program"),
        ({"prepack_authorizations": {}}, "whole-program"),
    ],
)
def test_whole_program_modes_are_refused_elsewhere(abi, kwargs, message):
    from merlin.targetgen.contract.readback_policy import ReadbackPolicy

    if isinstance(kwargs.get("readback_policy"), str):
        kwargs = {"readback_policy": ReadbackPolicy(kwargs["readback_policy"])}
    with pytest.raises(hr.HarnessRenderError, match=message):
        render(resident(), abi, **kwargs)


@pytest.mark.parametrize("transport, frame", [("out_bin_v1", "OUT_BIN_BEGIN v1"), ("out_b64_v1", "OUT_B64_BEGIN v1")])
def test_a_console_transport_frames_the_default_abis_logical_outputs(abi, transport, frame):
    """A packed console frame needs no whole-program boundary: it reads back the logical interface's
    own output buffers, and the roster check derives the same closed output set."""
    from merlin.targetgen.contract.readback_policy import ReadbackPolicy, _console_output_roster

    cb = resident()
    text = render(cb, abi, readback_policy=ReadbackPolicy(transport))
    outputs = [b for b in hr.logical_interface(cb, abi) if b.kind == "output"]
    assert outputs and all(f"{frame} {b.name} " in text for b in outputs)
    assert "printf(\"OUT " not in text, "no per-element text frame beside the packed one"
    names, tensors = _console_output_roster(cb)
    assert names == [b.name for b in outputs] and all(tensors[b.name]["shape"] == list(b.shape) for b in outputs)


def test_whole_program_outputs_must_be_write_only(abi):
    cb = {
        "abi_version": "0.1",
        "commands": [],
        "tensors": {
            "x": {"shape": [4], "dtype": "i32", "role": "input"},
            "y": {"shape": [4], "dtype": "i32", "role": "output"},
        },
        "kernel_abi": {
            "kind": "whole_program",
            "outputs": ["y"],
            "args": [{"tensor": "x", "access": "read"}, {"tensor": "y", "access": "readwrite"}],
        },
    }
    with pytest.raises(hr.HarnessRenderError, match="write-only"):
        render(cb, abi)
    cb["kernel_abi"]["args"][1]["access"] = "write"
    assert call_args(render(cb, abi)) == ["x", "y"]


def test_explicit_storage_encoding_is_honoured(abi):
    from merlin.perf.storage_encoding import GroupedAxesStorage

    enc = {
        "A": GroupedAxesStorage((2, 3), "i8", ((0,), (1,)), (2, 3), (5, 1), 10),
        "Y": GroupedAxesStorage((2, 3), "i16", ((1,), (0,)), (3, 2), (4, 1), 12),
    }
    cb = {
        "abi_version": "0.1",
        "commands": [],
        "tensors": {
            n: {"shape": list(e.physical_shape), "dtype": e.dtype, "role": "output" if n == "Y" else "input"}
            for n, e in enc.items()
        },
        "params": {"storage_encodings": {n: e.to_dict() for n, e in enc.items()}},
        "kernel_abi": {
            "kind": "whole_program",
            "outputs": ["Y"],
            "args": [{"tensor": "A", "access": "read"}, {"tensor": "Y", "access": "write"}],
        },
    }
    text = render(cb, abi, inputs={"A": [[-3, -2, -1], [0, 1, 2]]})
    assert "T_A[10]" in text and "{-3,-2,-1,0,0,0,1,2,0,0}" in text and "T_Y[12]" in text


def test_a_requested_counter_bracket_the_target_does_not_declare_is_refused(abi, monkeypatch):
    monkeypatch.setenv("MERLIN_HW_COUNTERS", "1")
    with pytest.raises(hr.HarnessRenderError, match="counter instrumentation unavailable"):
        render(resident(), abi)


def test_a_counter_bracket_wraps_only_the_measured_window(abi, monkeypatch):
    from merlin.targetgen.contract import counter_bracket as CB

    monkeypatch.setenv("MERLIN_HW_COUNTERS", "1")
    seen = {}

    def fake(target, spec, *, unit=None):
        seen.update(target=target, spec=spec, unit=unit)
        return {"helper": "/*helper*/\n", "prologue": ["  /*pro*/"], "epilogue": ["  /*epi*/"]}

    monkeypatch.setattr(CB, "bracket_for_target", fake)
    hooks = hr.HostHooks("probe_kernel", None, "rdcycle %0", "fence", target="probe", counter_bracket={"x": 1})
    text = hr.render_with(resident(), abi=abi, hooks=hooks)
    lines = [line.strip() for line in text.splitlines()]
    assert "#include" not in text.replace("#include <stdint.h>", "").replace("#include <stdio.h>", "")
    assert lines.index("/*pro*/") < lines.index("uint64_t c0 = merlin_read_cycles();")
    assert lines.index("/*epi*/") > lines.index("uint64_t c1 = merlin_read_cycles();")
    assert seen == {"target": "probe", "spec": {"x": 1}, "unit": None}
    monkeypatch.delenv("MERLIN_HW_COUNTERS")
    assert "/*pro*/" not in hr.render_with(resident(), abi=abi, hooks=hooks)


def test_missing_host_block_is_refused():
    with pytest.raises(hr.HarnessRenderError, match="logical_harness"):
        hr.hooks_from_contract({"harness_abi": {"entry_symbol": "k"}}, target="probe")


def test_published_results_match_the_reference_roster(abi):
    for cb in (resident(), resident(bias=True), resident(pool=True)):
        assert hr.logical_outputs(cb, abi.published_by) == list(reference_outputs(cb))


def test_reference_kernel_mutations_are_well_formed():
    for mutation in (None, *(m for m in MUTATIONS if m != "swapped_inputs")):
        source = render_reference_kernel(resident(bias=True), symbol="probe_kernel", mutation=mutation)
        assert "void probe_kernel(void *arg0, void *arg1, void *arg2, void *arg3, void *arg4, void *arg5)" in source


# --------------------------------------------------------------------------------------------------
# end to end on spike (skipped without the bare-metal toolchain)
# --------------------------------------------------------------------------------------------------
def _gate():
    path = repo_root() / "build_tools" / "scripts" / "harness_reference_gate.py"
    spec = importlib.util.spec_from_file_location("_harness_reference_gate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def spike_env(tmp_path_factory):
    from merlin.runtime.backends import spike

    if not (spike.gcc_path().is_file() and spike.spike_path().is_file()):
        pytest.skip("bare-metal RISC-V toolchain / spike unavailable")
    gate = _gate()
    work = tmp_path_factory.mktemp("spike_harness")
    return gate, work, gate._runtime_objects(work)


@pytest.mark.parametrize("variant", ["plain", "bias_relu_i8", "pooled"])
def test_reference_kernel_reproduces_reference_and_mutants_fail(spike_env, abi, variant):
    gate, work, runtime = spike_env
    cb = {"plain": resident(), "bias_relu_i8": resident(bias=True), "pooled": resident(pool=True)}[variant]
    expected = reference_outputs(cb)
    gcc = gate._toolchain()[0]
    harness_c = work / f"{variant}_harness.c"
    harness_c.write_text(render(cb, abi), encoding="utf-8")
    harness_o = work / f"{variant}_harness.o"
    gate._compile(gcc, harness_c, harness_o, "-Dmain=merlin_harness_main")
    for mutation in (None, "off_by_one", "uninitialized_output", "transposed", "wrong_scale"):
        kernel_c = work / f"{variant}_{mutation}.c"
        kernel_c.write_text(render_reference_kernel(cb, symbol=HOOKS.entry_symbol, mutation=mutation), encoding="utf-8")
        kernel_o = work / f"{variant}_{mutation}.o"
        gate._compile(gcc, kernel_c, kernel_o)
        ok, why = gate._compare(
            gate._link_and_run(work, harness_o, kernel_o, runtime, f"{variant}_{mutation}", 300), expected
        )
        if mutation is None:
            assert ok, why
        elif not gate._vacuous(mutation, cb, expected):
            assert not ok, f"{mutation} was not caught"
