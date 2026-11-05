"""Original coefficient identity and actual counted store-index mutations."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
from test_compact_dense import encode, project
from test_selected_dense_contracts import inputs, schedule

from merlin.compile.scheduling.block_schedule import Compute, Load, Preload, Store
from merlin.compile.scheduling.compact_verification import verify_compact_dense
from merlin.compile.scheduling.dense_contracts import DenseTiledIntegerReadout, TiledIntegerStore
from merlin.sched.ir import Const, evaluate, instances


def tiled_inputs(tmp_path, *, shape=(5, 14, 13), tile=(2, 3, 4), scales=None):
    scales = scales or [0.5] * 4 + [1.5] * 4 + [0.9900199770927429] * 4 + [2.0]

    def declare(facts, software, dense):
        facts["output_bits"] = 8
        facts["codes"].append(5)
        narrow = deepcopy(software["isa"]["instructions"][-1])
        narrow["name"] = "store_tiled_i8"
        narrow["encoding"][0]["fixed"]["fact"][-1] = 5
        narrow["operands"].append(
            {
                "name": "readout_tile",
                "unit": "integer",
                "type": {"kind": "integer", "signed": True, "bits": {"fact": ["facts", "arg_bits"]}},
            }
        )
        narrow["encoding"][-1:] = [
            {
                "offset": {"fact": ["facts", "offsets", 5]},
                "width": {"fact": ["facts", "arg_bits"]},
                "argument": "readout_tile",
            },
            {
                "offset": {"fact": ["facts", "padding_offsets", 6]},
                "width": {"fact": ["facts", "padding_widths", 6]},
                "fixed": {"fact": ["facts", "origin"]},
            },
        ]
        software["isa"]["instructions"].append(narrow)
        dense["output_readout"] = {
            "schema": "merlin.dense_tiled_integer_readout.v1",
            "column_scales": scales,
            "tile_columns": tile[2],
            "relu": False,
            "output": {"dtype": "i8", "bits": {"fact": ["facts", "output_bits"]}},
            "instruction": "store_tiled_i8",
        }

    selected, problem, storage, adapter = inputs(tmp_path, tile=tile, shape=shape, mutate=declare)
    readout = selected.dense.output_readout

    def encode_tiled(op):
        if type(op) is TiledIntegerStore:
            base = encode(Store(op.dram_row, op.dram_col, op.rows, op.cols, op.accumulator_row))
            return replace(base, instr=readout.instruction_name, args=(*base.args, ("readout_tile", op.readout_tile)))
        return encode(op)

    def project_tiled(call):
        if call.instr == readout.instruction_name:
            op = project(replace(call, instr="store", args=tuple((k, v) for k, v in call.args if k != "readout_tile")))
            return TiledIntegerStore(
                op.dram_row,
                op.dram_col,
                op.rows,
                op.cols,
                op.accumulator_row,
                readout.identity,
                call.arg("readout_tile"),
            )
        return project(call)

    return selected, problem, storage, replace(adapter, encode=encode_tiled, project=project_tiled)


def test_complete_source_bits_and_exact_rounding_are_preserved():
    scales = [0.5] * 3 + [1.5] * 3 + [0.9900199770927429] * 2
    readout = DenseTiledIntegerReadout.from_column_scales(scales, tile_columns=3)
    assert len(readout.table_words) == 3 and readout.columns == 8
    for col, scale in enumerate(scales):
        for acc in [-(1 << 31), -257, -255, -3, -1, 0, 1, 3, 85, 127, 255, (1 << 31) - 1]:
            expected = int(np.clip(np.rint(np.float32(np.float32(acc) * np.float32(scale))), -128, 127))
            assert readout.evaluate(acc, col) == expected
    changed = DenseTiledIntegerReadout.from_column_scales([0.5] * 3 + [1.5] * 5, tile_columns=3)
    assert changed.identity != readout.identity
    assert replace(readout, relu=True).evaluate(-3, 0) == 0


@pytest.mark.parametrize(
    "scales,tile",
    [([0.5, 1.0], 2), ([0], 1), ([float("nan")], 1), ([float("inf")], 1), ([-0.5], 1), ([], 1), ([True], 1), ([1], 0)],
)
def test_unmodeled_source_scales_are_refused(scales, tile):
    with pytest.raises(ValueError):
        DenseTiledIntegerReadout.from_column_scales(scales, tile_columns=tile)


def test_actual_source_index_output_width_and_full_coverage(tmp_path):
    model = tiled_inputs(tmp_path)
    readout = model[0].dense.output_readout
    result = schedule(model, output_readout=readout, readout_qualification_identity="neutral-indexed-readout-v1")
    assert result.kernel.args[-1].dtype == "i8"
    assert result.realization["requested_store_bytes"] == model[1].m * model[1].n
    assert result.realization["tiled_readout"]["parameter_words"] == 4
    for call, env in instances(result.kernel):
        if call.instr == readout.instruction_name:
            assert evaluate(call.arg("readout_tile"), env) * readout.tile_columns == evaluate(call.arg("col"), env)
    result.revalidate()

    # Interpret actual projected operations, then compare every result against
    # an independent full matrix product and binary32 nearest-even epilogue.
    _, p, s, adapter = model
    lhs = (np.arange(p.m * p.k) % 11 - 5).reshape(p.m, p.k)
    rhs = (np.arange(p.k * p.n) % 9 - 4).reshape(p.k, p.n)
    output = np.full((p.m, p.n), -1000)
    memories = {view.name: {} for view in (s.lhs, s.rhs, s.accumulator)}
    preload = None
    for call, env in instances(result.kernel):
        actual = replace(call, args=tuple((key, Const(evaluate(value, env))) for key, value in call.args))
        op = adapter.project(actual)
        args = {
            name: evaluate(getattr(op, name), {})
            for name in (
                ("row", "dram_row", "dram_col", "rows", "cols")
                if isinstance(op, Load)
                else ("input_row", "rows", "cols")
                if isinstance(op, Compute)
                else ("dram_row", "dram_col", "rows", "cols", "accumulator_row", "readout_tile")
                if isinstance(op, Store)
                else ()
            )
        }
        if isinstance(op, Load):
            view, source = (s.lhs, lhs) if op.role == "lhs" else (s.rhs, rhs)
            for i in range(args["rows"]):
                for j in range(args["cols"]):
                    memories[view.name][args["row"] + i, j] = int(source[args["dram_row"] + i, args["dram_col"] + j])
        elif isinstance(op, Preload):
            preload = op
        elif isinstance(op, Compute):
            for i in range(args["rows"]):
                for j in range(evaluate(preload.cols, {})):
                    slot = evaluate(preload.accumulator_row, {}) + i, j
                    value = memories[s.accumulator.name][slot] if evaluate(preload.accumulate, {}) else 0
                    for k in range(args["cols"]):
                        value += (
                            memories[s.lhs.name][args["input_row"] + i, k]
                            * memories[s.rhs.name][evaluate(preload.weight_row, {}) + k, j]
                        )
                    memories[s.accumulator.name][slot] = value
        else:
            for i in range(args["rows"]):
                for j in range(args["cols"]):
                    value = memories[s.accumulator.name][args["accumulator_row"] + i, j]
                    output[args["dram_row"] + i, args["dram_col"] + j] = readout.evaluate(
                        value, args["readout_tile"] * readout.tile_columns + j
                    )
    scales = np.array([np.frombuffer(word.to_bytes(4, "little"), dtype="<f4")[0] for word in readout.column_scale_bits])
    expected = np.clip(np.rint((lhs @ rhs).astype(np.float32) * scales), -128, 127)
    np.testing.assert_array_equal(output, expected)


@pytest.mark.parametrize("mutation", ["wrong_tile", "scalar_store", "wrong_identity"])
def test_actual_store_cannot_substitute_other_column_semantics(tmp_path, mutation):
    model = tiled_inputs(tmp_path)
    selected, problem, storage, adapter = model
    readout = selected.dense.output_readout
    result = schedule(model, output_readout=readout, readout_qualification_identity="neutral-indexed-readout-v1")

    def corrupt(call):
        op = adapter.project(call)
        if type(op) is not TiledIntegerStore:
            return op
        if mutation == "wrong_tile":
            return replace(op, readout_tile=Const(0))
        if mutation == "scalar_store":
            return Store(op.dram_row, op.dram_col, op.rows, op.cols, op.accumulator_row, op.readout)
        return replace(op, readout="other-original-source")

    with pytest.raises(ValueError, match="invariant|readout"):
        verify_compact_dense(
            result.kernel,
            problem,
            selected.dense._resource_limits(storage),
            storage,
            replace(adapter, project=corrupt),
            static_budget=80,
            timeout_ms=5000,
            output_readout=readout,
        )


def test_hundreds_of_source_tiles_share_one_compact_body(tmp_path):
    tile = (2, 3, 4)
    scales = [scale for i in range(352) for scale in [0.5 if i % 2 else 1.5] * tile[2]]
    model = tiled_inputs(tmp_path, shape=(3, 31, len(scales)), tile=tile, scales=scales)
    readout = model[0].dense.output_readout
    result = schedule(model, output_readout=readout, readout_qualification_identity="neutral-indexed-readout-v1")
    assert len(readout.table_words) == 352
    assert result.realization["static_instruction_calls"] < 80
    assert result.realization["output_tiles"] == 704
    assert result.realization["requested_store_bytes"] == 3 * len(scales)
    result.revalidate()


def test_selected_source_vector_cannot_change_after_check(tmp_path):
    model = tiled_inputs(tmp_path)
    readout = model[0].dense.output_readout
    result = schedule(model, output_readout=readout, readout_qualification_identity="neutral-indexed-readout-v1")
    changed = replace(readout, column_scale_bits=(readout.column_scale_bits[4],) * readout.columns)
    with pytest.raises(ValueError, match="readout|checked"):
        replace(result, output_readout=changed, checked_readout=changed.identity).revalidate()


def test_emitted_immutable_table_is_configuration_metadata_only(monkeypatch):
    from test_host_command_parameters import _observer
    from xdsl.dialects import llvm
    from xdsl.dialects.builtin import ModuleOp

    from merlin.llvmlower.host_builder import PTR, FnBuilder
    from merlin.llvmlower.integer_readout import load_tiled_readout_parameter, tiled_readout_parameter_table
    from merlin.verify import host_compute_audit as audit

    _observer(monkeypatch)
    readout = DenseTiledIntegerReadout.from_column_scales([0.5] * 4 + [1.5] * 4, tile_columns=4)
    table = tiled_readout_parameter_table(readout, symbol="source_scale_bits")
    builder = FnBuilder([PTR, PTR])
    address = builder.add(llvm.AddressOfOp("source_scale_bits", PTR)).results[0]

    def body(index):
        bits = load_tiled_readout_parameter(builder, address, index)
        packed = builder.or_i(builder.shl_i(bits, 32), builder.const(8))
        builder.add(
            llvm.InlineAsmOp(
                ".insn r 0x7b, 3, 0, x0, $0, $1", "r,r,~{memory}", [builder.const(2), packed], has_side_effects=True
            )
        )

    builder.for_loop(0, len(readout.table_words), 1, body, retain_loop=True)
    function = llvm.FuncOp("entry", llvm.LLVMFunctionType([PTR, PTR], llvm.LLVMVoidType()), body=builder.finish())
    module = ModuleOp([table, function])
    module.verify()
    row = audit.audit(module, [audit.GroupSite(0, "unit", "entry", 8)], target="fixture")["groups"][0]
    assert row["command_parameters"]["status"] == "proven", row
    assert row["host_raw_payload_bytes"] == 8 and row["host_payload_bytes"] == 0
