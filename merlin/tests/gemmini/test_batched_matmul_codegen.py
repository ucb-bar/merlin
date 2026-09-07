"""Direct Gemmini lowering for independently varying rank-N integer contractions."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from merlin.runtime.backends import base as _backends
from merlin.targetgen.contract.interface_emit import parse_interface_mlir
from merlin.targetgen.rocc.decode import decode_text


gem = _backends.get_backend("gemmini")
gm = gem.gemmini_codegen_mlir
CodegenError = gem.gemmini_codegen.CodegenError


def _cb(*, batch_shape=(2,), m=3, k=5, n=4) -> dict:
    batch = 1
    for extent in batch_shape:
        batch *= extent
    # Deliberately W/A/Y, not semantic operand order: this proves the original declaration-order ABI.
    return {
        "abi_version": "0.1", "target": "gemmini", "version": "0.1", "params": {},
        "tensors": {
            "W": {"role": "weight", "shape": [*batch_shape, k, n], "dtype": "i8"},
            "A": {"role": "input", "shape": [*batch_shape, m, k], "dtype": "i8"},
            "Y": {"role": "output", "shape": [*batch_shape, m, n], "dtype": "i32"},
        },
        "commands": [{
            "opcode": "BATCHED_MATMUL",
            "operands": {"a": "A", "w": "W", "dst": "Y"},
            "attributes": {"batch": batch, "output_dtype": "i32"},
        }],
        "outputs": ["Y"],
    }


def _fake_isa(*, scratchpad_rows=32, accumulator_rows=16):
    return SimpleNamespace(
        DIM=16, SCRATCHPAD_ROWS=scratchpad_rows, ACCUMULATOR_ROWS=accumulator_rows,
        ACC_ELEM_DTYPE="i32", ACC_ELEM_BITS=32,
    )


def _funct_count(text: str, funct: int) -> int:
    token = f", {funct}, x0"
    return sum(token in line for line in text.splitlines() if ".insn r" in line)


def _initializer(source: str, name: str) -> list[int]:
    line, = [line for line in source.splitlines()
             if line.startswith("static const") and f"T_{name}[" in line]
    body = line.rsplit("{", 1)[1].split("}", 1)[0]
    return [int(value) for value in body.split(",")]


def test_rank3_slice_offsets_are_exact_compact_padded_buffer_offsets():
    plan = gm._batched_matmul_plan(_cb())
    assert plan.arguments == ("W", "A", "Y")
    assert (plan.mp, plan.kp, plan.np) == (16, 16, 16)
    assert plan.slice(0).batch_index == (0,)
    second = plan.slice(1)
    assert second.batch_index == (1,)
    assert (second.a_byte_offset, second.w_byte_offset, second.dst_byte_offset) == (48, 80, 192)


def test_rank4_slice_order_and_offsets_preserve_each_declared_batch_axis():
    plan = gm._batched_matmul_plan(_cb(batch_shape=(2, 3)))
    assert plan.geometry.output_shape == (2, 3, 3, 4)
    assert plan.geometry.batch_count == 6
    assert plan.slice(3).batch_index == (1, 0)
    assert plan.slice(5).batch_index == (1, 2)
    assert (plan.slice(5).a_byte_offset,
            plan.slice(5).w_byte_offset,
            plan.slice(5).dst_byte_offset) == (240, 400, 960)


def test_emitter_moves_each_varying_rhs_and_reads_each_private_slice_output():
    cb = _cb(batch_shape=(2, 3))
    text, arguments = gm.emit_kernel_mlir(cb)
    isa = gm._isa()
    assert arguments == ["W", "A", "Y"]
    assert text.count("llvm.func @gemmini_kernel") == 1
    # Tiny one-tile slices: one W + one A move-in and one readout for every flattened batch slice.
    assert _funct_count(text, isa.K_MVIN) == 12
    assert _funct_count(text, isa.K_MVOUT) == 6
    assert _funct_count(text, isa.K_COMPUTE_PRELOADED) == 6
    assert "RES_PACK" not in text and "block_diagonal" not in text


def test_decoded_rank3_trace_has_exact_tail_offsets_and_resets_each_slice_accumulator():
    text, arguments = gm.emit_kernel_mlir(_cb(k=17))
    instructions = decode_text(text, target="gemmini")["instructions"]
    w_arg, a_arg, y_arg = (arguments.index(name) for name in ("W", "A", "Y"))
    mvins = [item for item in instructions if item["class"] == "MVIN"]
    w_mvins = [item for item in mvins if item["decoded"]["dram"]["arg_index"] == w_arg]
    a_mvins = [item for item in mvins if item["decoded"]["dram"]["arg_index"] == a_arg]
    assert [(item["decoded"]["dram"]["offset"], item["decoded"]["rows"],
             item["decoded"]["cols"]) for item in w_mvins] == [
        (0, 16, 4), (256, 1, 4), (272, 16, 4), (528, 1, 4)]
    assert [(item["decoded"]["dram"]["offset"], item["decoded"]["rows"],
             item["decoded"]["cols"]) for item in a_mvins] == [
        (0, 3, 16), (16, 3, 1), (96, 3, 16), (112, 3, 1)]
    preloads = [item for item in instructions if item["class"] == "PRELOAD"]
    assert [item["decoded"]["accumulate"] for item in preloads] == [False, True, False, True]
    mvouts = [item for item in instructions if item["class"] == "MVOUT"]
    assert [(item["decoded"]["dram"]["arg_index"],
             item["decoded"]["dram"]["offset"], item["decoded"]["rows"],
             item["decoded"]["cols"]) for item in mvouts] == [
        (y_arg, 0, 3, 4), (y_arg, 192, 3, 4)]


def test_rank4_harness_keeps_varying_rhs_storage_and_one_warm_measured_boundary(monkeypatch):
    cb = _cb(batch_shape=(2, 3))
    a_values = list(range(6 * 3 * 5))
    # A distinct first value per logical RHS slice makes accidental shared-W packing observable.
    w_values = [-60 + 20 * batch + row * 4 + col
                for batch in range(6) for row in range(5) for col in range(4)]
    monkeypatch.setenv("MERLIN_CACHE_STATE", "warm")
    source = gem.render_harness(
        cb, target="gemmini", inputs={"A": a_values, "W": w_values})

    assert "gemmini_kernel((void*)T_W, (void*)T_A, (void*)T_Y);" in source
    assert source.count("  gemmini_kernel(") == 2  # one unmeasured warm-up, one measured invocation
    assert source.count("uint64_t c0 = read_cycles();") == 1
    assert "static const elem_t T_W[512]" in source   # 30 logical rows -> 32, row pitch 16
    assert "static const elem_t T_A[512]" in source   # 18 logical rows -> 32, row pitch 16
    assert "static int32_t T_Y[512]" in source
    assert 'printf("OUT_ND Y 4 2 3 3 4")' in source
    assert 'printf("OUT Y ' not in source
    packed_w = _initializer(source, "W")
    assert [packed_w[batch * 5 * 16] for batch in range(6)] == [-60, -40, -20, 0, 20, 40]


def test_rank_n_output_protocol_reconstructs_declared_shape_exactly():
    values = list(range(48))
    console = ("OUT_ND Y 4 2 2 3 4 " + " ".join(str(value) for value in values)
               + "\nMETRIC cycles 77\nDONE\n")
    outputs, metrics = gem.parse_output(console)
    assert metrics == {"cycles": 77}
    assert outputs["Y"] == [
        [[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]],
         [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]],
        [[[24, 25, 26, 27], [28, 29, 30, 31], [32, 33, 34, 35]],
         [[36, 37, 38, 39], [40, 41, 42, 43], [44, 45, 46, 47]]],
    ]


def test_capacity_uses_maximum_one_slice_footprint_not_total_batch_work():
    # K=N=33 has nine RHS tiles (144 rows).  At 160 rows the slice-resident RHS plus one A tile
    # fits, but its three-tile A panel does not; the six-slice rank-4 case gets the same decision.
    rank3 = gm._batched_matmul_plan(
        _cb(batch_shape=(2,), m=17, k=33, n=33),
        isa=_fake_isa(scratchpad_rows=160))
    rank4 = gm._batched_matmul_plan(
        _cb(batch_shape=(2, 3), m=17, k=33, n=33),
        isa=_fake_isa(scratchpad_rows=160))
    assert rank3.schedule == rank4.schedule == "slice_weight_resident"
    assert gm._batched_matmul_plan(
        _cb(batch_shape=(2, 3), m=17, k=33, n=33),
        isa=_fake_isa(scratchpad_rows=32)).schedule == "two_tile_streaming"
    assert gm._batched_matmul_plan(
        _cb(batch_shape=(2, 3)), isa=_fake_isa(scratchpad_rows=32)
    ).schedule == "slice_weight_and_a_panel_resident"


@pytest.mark.parametrize("scratchpad_rows,schedule,mvin_count,max_local", [
    (112, "slice_weight_resident", 18, 96),
    (32, "two_tile_streaming", 24, 16),
])
def test_capacity_selected_nondefault_schedules_emit_their_exact_traffic(
        scratchpad_rows, schedule, mvin_count, max_local):
    """Exercise the two capacity fallbacks, not merely the schedule-name preflight.

    K=33/N=17 produces six RHS tiles.  The resident fallback loads each exactly once; the
    two-tile fallback reloads W beside A for each of the two M tiles.  Both retain one measured
    kernel boundary and keep every operand move inside the capacity used to select the schedule.
    """
    cb = _cb(batch_shape=(1,), m=17, k=33, n=17)
    plan = gm._batched_matmul_plan(
        cb, isa=_fake_isa(scratchpad_rows=scratchpad_rows))
    assert plan.schedule == schedule
    text, arguments = gm._emit_batched_matmul_mlir(cb, plan)
    instructions = decode_text(text, target="gemmini")["instructions"]
    mvins = [item for item in instructions if item["class"] == "MVIN"]
    assert len(mvins) == mvin_count
    assert max(item["decoded"]["spad_addr"] for item in mvins) == max_local
    assert all(item["decoded"]["spad_addr"] < scratchpad_rows for item in mvins)
    assert sum(item["class"] == "MVOUT" for item in instructions) == 4
    assert sum(item["class"] == "COMPUTE_PRELOADED" for item in instructions) == 12
    assert sum(item["class"] == "CONFIG_EX" for item in instructions) == 1
    assert sum(item["class"] == "CONFIG_ST" for item in instructions) == 1
    assert sum(item["class"] == "FENCE" for item in instructions) == 2
    w_arg = arguments.index("W")
    w_offsets = [item["decoded"]["dram"]["offset"] for item in mvins
                 if item["decoded"]["dram"]["arg_index"] == w_arg]
    one_rhs = [0, 16, 512, 528, 1024, 1040]
    streamed_rhs = [0, 512, 1024, 16, 528, 1040]
    assert w_offsets == (one_rhs if schedule == "slice_weight_resident"
                         else streamed_rhs + streamed_rhs)


@pytest.mark.parametrize("scratchpad_rows,accumulator_rows,fragment", [
    (31, 16, "two concurrent 16-row operand tiles"),
    (32, 15, "one 16-row accumulator tile"),
])
def test_capacity_preflight_fails_closed(scratchpad_rows, accumulator_rows, fragment):
    with pytest.raises(CodegenError, match=fragment):
        gm._batched_matmul_plan(
            _cb(), isa=_fake_isa(
                scratchpad_rows=scratchpad_rows, accumulator_rows=accumulator_rows))


@pytest.mark.parametrize("mutation,fragment", [
    (lambda cb: cb["tensors"]["W"].update(shape=[1, 5, 4]), "no broadcasting"),
    (lambda cb: cb["tensors"]["Y"].update(shape=[2, 3, 5]), "destination shape"),
    (lambda cb: cb["tensors"]["A"].update(dtype="f32"), "requires two i8"),
    (lambda cb: cb["tensors"]["Y"].update(dtype="i8"), "i32 destination"),
    (lambda cb: cb["tensors"]["A"].update(role="output"), "leaf input or weight"),
    (lambda cb: cb["tensors"].update(
        B={"role": "bias", "shape": [4], "dtype": "i32"}), "exactly its a/w/dst"),
    (lambda cb: cb["tensors"].update(
        S={"role": "scale", "shape": [1], "dtype": "i8"}), "exactly its a/w/dst"),
    (lambda cb: cb.update(outputs=["A"]), "outputs must be exactly"),
    (lambda cb: cb["commands"][0]["attributes"].update(epilogue=["relu"]),
     "does not implement an epilogue"),
    (lambda cb: cb["commands"][0]["attributes"].update(batch=3), "disagrees"),
    (lambda cb: cb["params"].update(im2col_recipes=[{"target": "A"}]),
     "consumes its original a/w slices"),
    (lambda cb: cb["commands"].append({"opcode": "EVICT", "operands": {"handle": "x"}}),
     "exactly one isolated"),
])
def test_unsupported_semantics_fail_closed(mutation, fragment):
    cb = deepcopy(_cb())
    mutation(cb)
    with pytest.raises(CodegenError, match=fragment):
        gm.emit_kernel_mlir(cb)


@pytest.mark.parametrize("line,fragment", [
    ("OUT_ND Y 0 1", "rank must be positive"),
    ("OUT_ND Y 3 2 3", "dimension list is truncated"),
    ("OUT_ND Y 3 2 3 4 1 2", "expected 24 values"),
])
def test_malformed_rank_n_output_fails_closed(line, fragment):
    with pytest.raises(gem.GemminiError, match=fragment):
        gem.parse_output(line + "\nDONE\n")


@pytest.mark.parametrize("relative", [
    "contract/capsules/layers/GB0_batched_matmul_i8/capsule.interface.mlir",
    "contract/capsules/model_slices/SY_rank_contraction_batched/capsule.interface.mlir",
])
def test_real_rank3_capsule_parses_to_a_declared_output_and_reaches_target_emission(relative):
    from merlin.common.paths import merlin_dir

    interface = merlin_dir() / relative
    cb = parse_interface_mlir(interface.read_text(encoding="utf-8"))
    assert cb["tensors"]["Y0"] == {
        "shape": [2, 16, 16], "dtype": "i32", "role": "output"}
    plan = gm._batched_matmul_plan(cb)
    assert plan.geometry.output_shape == (2, 16, 16)
    text, arguments = gm.emit_kernel_mlir(cb)
    assert arguments == ["A0", "W", "Y0"]
    assert text.count("llvm.func @gemmini_kernel") == 1
    harness = gem.render_harness(cb, target="gemmini")
    assert 'printf("OUT_ND Y0 3 2 16 16")' in harness


def test_registered_compile_path_routes_batched_matmul_through_mlir_harness_link(
        monkeypatch, tmp_path):
    from merlin.targetgen.contract import compile as contract_compile

    calls = []

    def fake_emit(candidate):
        assert candidate is cb
        calls.append("emit")
        return "module { llvm.func @gemmini_kernel() { llvm.return } }\n", ["W", "A", "Y"]

    expected = tmp_path / "routed.elf"

    def fake_compile(candidate, lowered, workdir, *, target):
        calls.append((candidate, lowered, Path(workdir), target))
        return expected

    cb = _cb()
    monkeypatch.setattr(gm, "emit_kernel_mlir", fake_emit)
    monkeypatch.setattr(contract_compile, "compile_lowered_to_elf", fake_compile)
    assert gem.compile_command_buffer(cb, tmp_path) == expected
    assert calls[0] == "emit"
    assert calls[1] == (cb, "module { llvm.func @gemmini_kernel() { llvm.return } }\n",
                        tmp_path, "gemmini")
    assert not (tmp_path / "main.c").exists()
