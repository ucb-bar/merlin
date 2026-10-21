"""Tensor algebra and adversarial realized-domain checks, no device grants."""

from dataclasses import replace

import pytest
from xdsl.dialects import arith, builtin, func, tensor
from xdsl.dialects.linalg.ops import GenericOp
from xdsl.utils.exceptions import VerifyException

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.transforms.certificates import ir_text
from merlin.transforms.dense_row_batching import DenseRowBatchingBudgets, plan_source_dense_row_batching
from merlin.transforms.dense_row_batching_check import verify_dense_row_batching
from merlin.transforms.pipeline_state import PipelineContext


def source(
    parts=(1, 1, 1), k=7, n=8, *, rank3=True, dtype="i8", accum="i32", update_state=False, uniform_epilogue=None
):
    total = sum(parts)
    lhs_type = f"tensor<{total}x1x{k}x{dtype}>" if rank3 else f"tensor<{total}x{k}x{dtype}>"
    rhs_type = f"tensor<1x{n}x{k}x{dtype}>" if rank3 else f"tensor<{k}x{n}x{dtype}>"
    result_types = [f"tensor<{m}x{n}x{accum}>" for m in parts]
    lines = [
        f"module {{ func.func @forward(%lhs:{lhs_type},%rhs:{rhs_type},%state:{lhs_type}) -> "
        f"({','.join([*result_types, *result_types, lhs_type, rhs_type, lhs_type])}) {{",
        f"%zero = arith.constant 0 : {accum}",
    ]
    start = 0

    def slice_text(result, operand, before, after, offsets, sizes):
        return (
            f'{result} = "tensor.extract_slice"({operand}) <{{static_offsets=array<i64:{",".join(map(str, offsets))}>,'
            f"static_sizes=array<i64:{','.join(map(str, sizes))}>,"
            f"static_strides=array<i64:{','.join('1' for _ in sizes)}>,"
            f"operandSegmentSizes=array<i32:1,0,0,0>}}> : ({before}) -> {after}"
        )

    for i, m in enumerate(parts):
        if rank3:
            lines += [
                slice_text(f"%s{i}", "%lhs", lhs_type, f"tensor<{m}x1x{k}x{dtype}>", (start, 0, 0), (m, 1, k)),
                f'%f{i} = "tensor.collapse_shape"(%s{i}) <{{reassociation=[[0:i64,1:i64,2:i64]]}}> : '
                f"(tensor<{m}x1x{k}x{dtype}>) -> tensor<{m * k}x{dtype}>",
                f'%a{i} = "tensor.expand_shape"(%f{i}) <{{reassociation=[[0:i64,1:i64]],'
                f"static_output_shape=array<i64:{m},{k}>}}> : "
                f"(tensor<{m * k}x{dtype}>) -> tensor<{m}x{k}x{dtype}>",
                slice_text(f"%r{i}", "%rhs", rhs_type, rhs_type, (0, 0, 0), (1, n, k)),
                f'%rf{i} = "tensor.collapse_shape"(%r{i}) <{{reassociation=[[0:i64,1:i64,2:i64]]}}> : '
                f"({rhs_type}) -> tensor<{n * k}x{dtype}>",
                f'%re{i} = "tensor.expand_shape"(%rf{i}) <{{reassociation=[[0:i64,1:i64]],'
                f"static_output_shape=array<i64:{n},{k}>}}> : "
                f"(tensor<{n * k}x{dtype}>) -> tensor<{n}x{k}x{dtype}>",
                f"%rt{i} = tensor.empty() : tensor<{k}x{n}x{dtype}>",
                f'%b{i} = "linalg.transpose"(%re{i},%rt{i}) <{{permutation=array<i64:1,0>}}> ({{',
                f"^bbt{i}(%ti:{dtype},%to:{dtype}): linalg.yield %ti : {dtype}",
                f"}}) : (tensor<{n}x{k}x{dtype}>,tensor<{k}x{n}x{dtype}>) -> tensor<{k}x{n}x{dtype}>",
            ]
            rhs = f"%b{i}"
        else:
            lines += [slice_text(f"%a{i}", "%lhs", lhs_type, f"tensor<{m}x{k}x{dtype}>", (start, 0), (m, k))]
            rhs = "%rhs"
        lines += [
            f'%z{i} = "tensor.splat"(%zero) : ({accum}) -> {result_types[i]}',
            f"%c{i} = linalg.generic {{indexing_maps=[affine_map<(m,n,k)->(m,k)>,"
            "affine_map<(m,n,k)->(k,n)>,affine_map<(m,n,k)->(m,n)>],"
            'iterator_types=["parallel","parallel","reduction"]} '
            f"ins(%a{i},{rhs}:tensor<{m}x{k}x{dtype}>,tensor<{k}x{n}x{dtype}>) outs(%z{i}:{result_types[i]}) {{",
            f"^bb{i}(%a:{dtype},%b:{dtype},%c:{accum}):",
        ]
        if dtype != accum:
            lines += [f"%ax = arith.extsi %a : {dtype} to {accum}", f"%bx = arith.extsi %b : {dtype} to {accum}"]
            left, right = "%ax", "%bx"
        else:
            left, right = "%a", "%b"
        lines += [
            f"%p = arith.muli {left},{right} : {accum}",
            f"%v = arith.addi %p,%c : {accum}",
            f"linalg.yield %v : {accum}",
            f"}} -> {result_types[i]}",
        ]
        if i == 0 and uniform_epilogue is not None:
            kind, literal = uniform_epilogue
            mapping = "affine_map<(m,n)->(m,n)>"
            lines += [
                f"%literal = arith.constant {literal} : {accum}",
                f"%uniform = tensor.splat %literal : {result_types[0]}",
                f"%epilogue_empty = tensor.empty() : {result_types[0]}",
                f"%epilogue = linalg.generic {{indexing_maps=[{mapping},{mapping},{mapping}],"
                f'iterator_types=["parallel","parallel"]}} '
                f"ins(%c0,%uniform:{result_types[0]},{result_types[0]}) outs(%epilogue_empty:{result_types[0]}) {{",
                f"^bbe(%x:{accum},%literal_arg:{accum},%unused:{accum}):",
                f"%changed = arith.{kind} %x,%literal_arg : {accum}",
                f"linalg.yield %changed : {accum}",
                f"}} -> {result_types[0]}",
            ]
        start += m
    outputs = [*[f"%c{i}" for i in range(len(parts))], *[f"%z{i}" for i in range(len(parts))], "%lhs", "%rhs", "%state"]
    if uniform_epilogue is not None:
        outputs[0] = "%epilogue"
    types = [*result_types, *result_types, lhs_type, rhs_type, lhs_type]
    if update_state:
        rank = 3 if rank3 else 2
        axes = ",".join(f"d{i}" for i in range(rank))
        mapping = f"affine_map<({axes})->({axes})>"
        iterators = ",".join('"parallel"' for _ in range(rank))
        lines += [
            f"%state_empty = tensor.empty() : {lhs_type}",
            f"%next_state = linalg.generic {{indexing_maps=[{mapping},{mapping},{mapping}],"
            f"iterator_types=[{iterators}]}} "
            f"ins(%lhs,%state:{lhs_type},{lhs_type}) outs(%state_empty:{lhs_type}) {{",
            f"^bbs(%x:{dtype},%s:{dtype},%unused:{dtype}):",
            f"%updated = arith.addi %x,%s : {dtype}",
            f"linalg.yield %updated : {dtype}",
            f"}} -> {lhs_type}",
        ]
        outputs[-1] = "%next_state"
    lines += [f"return {','.join(outputs)} : {','.join(types)}", "} }"]
    module = parse_mlir_text("\n".join(lines))
    module.verify()
    return module


def plan(module, **kwargs):
    return plan_source_dense_row_batching(
        module,
        PipelineContext(numerical_policy="typed-ordered-modular-source"),
        replace(DenseRowBatchingBudgets(), **kwargs),
    )


def batch(module):
    return next(op for op in module.walk() if type(op) is GenericOp and op.results[0].type.get_shape()[0] > 1)


@pytest.mark.parametrize("rank3", [False, True])
@pytest.mark.parametrize(
    "parts,k,n,dtype,accum",
    [
        ((1, 1, 1), 7, 8, "i8", "i32"),
        ((2, 1), 5, 9, "i16", "i32"),
        ((1, 2, 1), 3, 2, "i32", "i32"),
    ],
)
def test_complete_original_outputs_and_initializers_survive_exact_row_batching(rank3, parts, k, n, dtype, accum):
    module = source(parts, k, n, rank3=rank3, dtype=dtype, accum=accum)
    before = ir_text(module)
    selected = plan(module)
    assert selected is not None
    selected.revalidate()
    assert ir_text(module) == before
    assert len(selected.record()["batches"]) == 1
    assert selected.record()["batches"][0]["shape"] == [sum(parts), k, n]
    assert selected.candidate.body.block.first_op.function_type == module.body.block.first_op.function_type
    assert len(tuple(op for op in selected.candidate.walk() if type(op) is GenericOp)) == 1
    assert len(tuple(op for op in selected.candidate.walk() if type(op) is tensor.ConcatOp)) == 0
    assert len(selected.candidate.body.block.first_op.body.block.last_op.arguments) == 2 * len(parts) + 3
    # The original observed initializer values are still the same source ops.
    assert len(tuple(op for op in selected.candidate.walk() if type(op) is tensor.SplatOp)) == len(parts) + 1


def test_each_original_contraction_boundary_annotation_survives_on_its_actual_result():
    module = source()
    dots = [op for op in module.walk() if type(op) is GenericOp]
    for i, dot in enumerate(dots):
        dot.attributes["prov.region_id"] = builtin.StringAttr(f"original-{i}")
    selected = plan(module)
    actual = [use.operation for use in batch(selected.candidate).results[0].uses]
    assert {op.attributes["prov.region_id"].data for op in actual} == {f"original-{i}" for i in range(3)}
    actual[0].attributes["prov.region_id"] = builtin.StringAttr("changed")
    with pytest.raises(ValueError, match="changed"):
        verify_dense_row_batching(module, selected.candidate, index_bits=64)


def test_refused_intervening_scalar_metadata_leaves_no_analysis_clone_ssa_uses():
    module = source(update_state=True)
    function = module.body.block.first_op
    entry = function.body.block
    generics = [op for op in entry.ops if type(op) is GenericOp]
    update = generics[-1]
    empty = update.outputs[0].owner
    entry.detach_op(empty)
    entry.detach_op(update)
    entry.insert_ops_before((empty, update), generics[1])
    update.body.block.first_op.attributes["prov.dispatch_id"] = builtin.StringAttr("scalar-unproved")
    values = (*entry.args, *(result for op in module.walk() for result in op.results))
    before = tuple(frozenset((use.operation, use.index) for use in value.uses) for value in values)
    assert plan(module) is None
    assert before == tuple(frozenset((use.operation, use.index) for use in value.uses) for value in values)


def test_replanning_an_existing_batch_preserves_the_legal_original():
    selected = plan(source())
    before = ir_text(selected.candidate)
    assert plan(selected.candidate) is None
    assert ir_text(selected.candidate) == before


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_members": 2},
        {"max_source_elements": 2},
        {"max_constructed_operations": 2},
        {"max_candidates": 1},
        {"index_bits": 4},
    ],
)
def test_unknown_or_insufficient_original_budgets_preserve_legal_original(kwargs):
    module = source()
    original = ir_text(module)
    assert plan(module, **kwargs) is None
    assert ir_text(module) == original


@pytest.mark.parametrize(
    "damage", ["unknown_call", "unknown_hint", "unknown_scope", "overlap", "stride", "different_rhs"]
)
def test_unproved_original_source_keeps_original_without_mutation(damage):
    module = source()
    dots = [op for op in module.walk() if type(op) is GenericOp]
    if damage == "unknown_call":
        call = func.CallOp("observe", [], [])
        dots[1].parent.insert_op_before(call, dots[1])
        module.body.block.add_op(func.FuncOp.external("observe", [], []))
    elif damage == "unknown_hint":
        dots[1].attributes["prov.numeric_policy"] = builtin.StringAttr("unreviewed")
    elif damage == "unknown_scope":
        module.body.block.first_op.attributes["unknown_lifecycle"] = builtin.UnitAttr()
    elif damage in ("overlap", "stride"):
        slices = [
            op
            for op in module.walk()
            if type(op) is tensor.ExtractSliceOp and op.source is module.body.block.first_op.body.block.args[0]
        ]
        target = slices[1]
        target.properties["static_offsets"] = builtin.DenseArrayBase.from_list(
            builtin.i64, (0 if damage == "overlap" else 1, 0, 0)
        )
        if damage == "stride":
            target.properties["static_strides"] = builtin.DenseArrayBase.from_list(builtin.i64, (2, 1, 1))
    else:
        # Keep this a valid same-shape semantic mutation.
        module = source(k=7, n=7)
        dots = [op for op in module.walk() if type(op) is GenericOp]
        dots[1].inputs[1].owner.properties["permutation"] = builtin.DenseArrayBase.from_list(builtin.i64, (0, 1))
    module.verify()
    before = ir_text(module)
    assert plan(module) is None
    assert ir_text(module) == before


@pytest.mark.parametrize(
    "damage",
    [
        "slice_overlap",
        "wrong_root",
        "wrong_rhs",
        "wrong_arithmetic",
        "overflow",
        "observer",
        "changed_return",
        "changed_initializer",
        "dropped_output",
        "view_hint",
    ],
)
def test_independent_checker_reconstructs_actual_domains_and_untouched_opaque_wiring(damage):
    module = source(k=3, n=3, rank3=False)
    selected = plan(module)
    actual = selected.candidate.clone()
    combined = batch(actual)
    function = actual.body.block.first_op
    entry = function.body.block
    slices = [use.operation for use in combined.results[0].uses]
    if damage == "slice_overlap":
        slices[0].properties["static_offsets"] = builtin.DenseArrayBase.from_list(builtin.i64, (0, 0))
        slices[1].properties["static_offsets"] = builtin.DenseArrayBase.from_list(builtin.i64, (0, 0))
    elif damage == "wrong_root":
        combined.inputs[0].owner.operands[0].owner.operands = (entry.args[2],)
    elif damage == "wrong_rhs":
        combined.operands = (combined.inputs[0], entry.args[0], combined.outputs[0])
    elif damage == "wrong_arithmetic":
        multiply = next(op for op in combined.walk() if type(op) is arith.MuliOp)
        multiply.operands = (multiply.operands[0], multiply.operands[0])
    elif damage == "overflow":
        multiply = next(op for op in combined.walk() if type(op) is arith.MuliOp)
        multiply.properties["overflowFlags"] = arith.IntegerOverflowAttr((arith.IntegerOverflowFlag.NSW,))
    elif damage == "observer":
        entry.insert_op_before(func.CallOp("observe", [], []), combined)
        actual.body.block.add_op(func.FuncOp.external("observe", [], []))
    elif damage == "changed_return":
        returned = entry.last_op
        returned.operands = (*returned.operands[:-1], entry.args[0])
    elif damage == "changed_initializer":
        zero = next(op for op in actual.walk() if type(op) is arith.ConstantOp)
        zero.properties["value"] = builtin.IntegerAttr(1, builtin.i32)
    elif damage == "dropped_output":
        returned = entry.last_op
        returned.operands = (*returned.operands[:1], returned.operands[0], *returned.operands[2:])
    else:
        unchanged = next(op for op in actual.walk() if type(op) is tensor.ExtractSliceOp and op not in slices)
        unchanged.attributes["prov.hidden_effect"] = builtin.StringAttr("unknown")
    with pytest.raises((ValueError, VerifyException)):
        verify_dense_row_batching(module, actual, index_bits=64)


def test_original_and_selected_ssa_mutations_invalidate_issued_plan():
    module = source()
    selected = plan(module)
    constant = next(op for op in selected.candidate.walk() if type(op) is arith.ConstantOp)
    constant.properties["value"] = builtin.IntegerAttr(1, builtin.i32)
    with pytest.raises(ValueError, match="changed"):
        selected.revalidate()


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_sha256", "f" * 64),
        ("ir_level", "other-level"),
        ("context", PipelineContext(numerical_policy="changed-policy")),
    ],
)
def test_every_original_stamp_field_is_frozen(field, value):
    selected = plan(source())
    object.__setattr__(selected.stamp, field, value)
    with pytest.raises(ValueError, match="changed"):
        selected.revalidate()


@pytest.mark.parametrize("helper", ["recognize_signed_integer_matmul", "_trace_attributes"])
def test_actual_indirect_dense_reader_replacement_invalidates_plan(monkeypatch, helper):
    from merlin.transforms import dense_ir

    selected = plan(source())
    original = getattr(dense_ir, helper)
    monkeypatch.setattr(dense_ir, helper, lambda *args, **kwargs: original(*args, **kwargs))
    with pytest.raises(ValueError, match="changed"):
        selected.revalidate()


def test_original_budget_and_checker_reader_mutations_invalidate_issued_plan(monkeypatch):
    from merlin.transforms import dense_row_batching as owner

    selected = plan(source())
    object.__setattr__(selected.budgets, "max_members", 2)
    with pytest.raises(ValueError, match="changed"):
        selected.revalidate()
    selected = plan(source())
    original = owner._functions()
    monkeypatch.setattr(owner, "_functions", lambda: original)
    with pytest.raises(ValueError, match="changed"):
        selected.revalidate()


@pytest.mark.parametrize("scalarized,materialized", [(False, False), (True, False), (False, True)])
def test_ordinary_native_row_batching_preserves_nine_outputs_and_actual_carried_state(
    tmp_path, scalarized, materialized
):
    """One ordinary CPU pair; no device-retention or performance assertion."""
    import hashlib
    import json
    import os
    from pathlib import Path

    import numpy as np

    from merlin.common.paths import merlin_dir
    from merlin.llvmlower import abi, codegen, pipeline, stock_cli, toolchain
    from merlin.llvmlower import lower as lowering
    from merlin.llvmlower.abi import HostModel, PrivateHostImagePolicy
    from merlin.llvmlower.codegen import mlir_runtime_c
    from merlin.llvmlower.lower import lower_model_file
    from merlin.llvmlower.toolchain import clang, compiler_python, host_llc, llvm_install, mlir_translate
    from merlin.transforms.dense_row_batching import implementation_paths

    if materialized:
        from test_dense_row_materialization import computed_rows

        original = computed_rows(rank3=True, k=5, n=7, dtype="i16", update_state=True)
    else:
        original = source(
            k=5, n=7, dtype="i16", update_state=True, uniform_epilogue=("addi", 4) if scalarized else None
        )
    if scalarized:
        from merlin.llvmlower.splat_inputs import scalarize_splat_inputs

        if not materialized:
            assert scalarize_splat_inputs(original) == 1
    selected = plan(
        original, **({"allow_lhs_materialization": True, "max_materialized_bytes": 30} if materialized else {})
    )
    assert selected is not None

    def pin(path):
        return {"path": str(path), "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}

    controls = {
        key: os.environ.get(key)
        for key in (
            "MERLIN_COMPILER_DRIVER",
            "MERLIN_COMPILER_PYTHON",
            "MERLIN_MLIR_INSTALL",
            "MERLIN_MLIR_TRANSLATE",
            "MERLIN_CLANG",
            "MERLIN_LLVM_LLC",
            "CC",
        )
    }
    tools = [clang().resolve(), llvm_install() / "bin/mlir-opt", mlir_translate()]
    if controls["MERLIN_COMPILER_DRIVER"] != "stock-cli":
        tools.append(compiler_python().resolve())
    if host_llc() is not None:
        tools.append(host_llc().resolve())
    if controls["CC"]:
        from shutil import which

        tools.append(Path(which(controls["CC"]) or controls["CC"]).resolve())
    owners = [
        pin(path)
        for path in (
            *implementation_paths(),
            merlin_dir() / "tests/ir/test_dense_row_batching.py",
            merlin_dir() / "tests/ir/test_dense_row_materialization.py",
            Path(abi.__file__),
            Path(lowering.__file__),
            mlir_runtime_c(),
            *(Path(owner.__file__) for owner in (codegen, pipeline, stock_cli, toolchain)),
            *tools,
        )
    ]
    plan_record = selected.record()
    arms = {}
    for name, module in (("baseline", original), ("candidate", selected.candidate)):
        selected.revalidate()
        path = tmp_path / f"{name}.mlir"
        path.write_text(ir_text(module))
        compiled = lower_model_file(path, tmp_path / name, targets=("host",), vectorize=False, parallel=False)
        assert compiled.host_so is not None
        image = compiled.host_so
        assert image.with_suffix(".o").is_file()
        assert image.with_name("mlir_runtime_host.o").is_file()
        artifacts = [pin(path), *(pin(p) for p in sorted(compiled.workdir.rglob("*")) if p.is_file())]
        native = HostModel.load(str(image), image_policy=PrivateHostImagePolicy(image.parent.resolve()))
        assert native.image_sha256 == pin(image)["sha256"]
        state = np.full((3, 1, 5), np.iinfo(np.int16).max, dtype=np.int16)
        calls = []
        for epoch in range(3):
            lhs = np.asarray([((i * 197 + epoch * 271) % 65536) - 32768 for i in range(15)], dtype=np.int16).reshape(
                3, 1, 5
            )
            rhs = np.asarray([((i * 401 + epoch * 113) % 65536) - 32768 for i in range(35)], dtype=np.int16).reshape(
                1, 7, 5
            )
            lhs[0, 0, :], rhs[0, 0, :] = 32767, 32767
            if materialized and epoch == 1:
                lhs = rhs.reshape(-1)[:15].reshape(3, 1, 5)
            args = (lhs, rhs, state.copy())
            reference, overflows = [], 0
            for row in range(3):
                output = np.zeros((1, 7), dtype=np.int32)
                for col in range(7):
                    total = 0
                    for depth in range(5):
                        wide = total + int(lhs[row, 0, depth]) * int(rhs[0, col, depth])
                        overflows += not -(1 << 31) <= wide < (1 << 31)
                        total = (wide + (1 << 31)) % (1 << 32) - (1 << 31)
                    output[0, col] = total
                if scalarized and not materialized and row == 0:
                    for col in range(7):
                        output[0, col] = (int(output[0, col]) + 4 + (1 << 31)) % (1 << 32) - (1 << 31)
                reference.append(output)
            next_state = np.asarray(
                [(int(a) + int(b) + 32768) % 65536 - 32768 for a, b in zip(lhs.flat, state.flat, strict=True)],
                dtype=np.int16,
            ).reshape(state.shape)
            reference += [*[np.zeros((1, 7), dtype=np.int32) for _ in range(3)], lhs, rhs, next_state]
            assert overflows > 0
            in_buffers = [np.full(a.size + 32, 61, dtype=a.dtype) for a in args]
            arguments = [a[16:-16].reshape(v.shape) for a, v in zip(in_buffers, args, strict=True)]
            if materialized and epoch == 1:
                in_buffers[0] = in_buffers[1]
                arguments[0] = in_buffers[0][16:31].reshape(lhs.shape)
                assert arguments[0].ctypes.data == arguments[1].ctypes.data
            for actual, value in zip(arguments, args, strict=True):
                actual[:] = value
            before = [a.copy() for a in in_buffers]
            out_buffers = [np.full(a.size + 32, 89, dtype=a.dtype) for a in reference]
            outputs = [a[16:-16].reshape(v.shape) for a, v in zip(out_buffers, reference, strict=True)]
            native([(a.ctypes.data, a.shape) for a in (*arguments, *outputs)])
            assert all(a.tobytes() == b.tobytes() for a, b in zip(outputs, reference, strict=True))
            assert all(a.tobytes() == b.tobytes() for a, b in zip(in_buffers, before, strict=True))
            assert all(np.all(a[:16] == 89) and np.all(a[-16:] == 89) for a in out_buffers)
            actual_path = tmp_path / f"{name}-{epoch}.npz"
            np.savez(
                actual_path,
                **{f"output{i}": a for i, a in enumerate(outputs)},
                **{f"input{i}": a for i, a in enumerate(arguments)},
                **{f"input_guard{i}": a for i, a in enumerate(in_buffers)},
                **{f"output_guard{i}": a for i, a in enumerate(out_buffers)},
            )
            calls.append(
                {
                    **pin(actual_path),
                    "ordered_i32_overflow_events": overflows,
                    "readonly_input_alias": materialized and epoch == 1,
                }
            )
            state = outputs[-1].copy()
        arms[name] = {"calls": calls, "artifacts": artifacts, "loaded_image_sha256": native.image_sha256}
    selected.revalidate()
    assert controls == {key: os.environ.get(key) for key in controls}
    for row in owners + [p for arm in arms.values() for field in ("calls", "artifacts") for p in arm[field]]:
        assert pin(row["path"]) == {key: row[key] for key in ("path", "sha256")}
    (tmp_path / "pair.json").write_text(
        json.dumps(
            {
                "scope": "ordinary native CPU source algebra; no device retention/timing qualification",
                "plan": plan_record,
                "owners": owners,
                "original_compiler_environment": controls,
                "arms": arms,
            },
            indent=2,
        )
        + "\n"
    )


@pytest.mark.parametrize("kind,literal", [("addi", 4), ("maxsi", -3), ("minsi", 5)])
def test_actual_scalarized_literal_broadcast_span_preserves_all_outputs_and_carry(kind, literal):
    import numpy as np

    from merlin.llvmlower.splat_inputs import scalarize_splat_inputs
    from merlin.runtime.linalg_numpy import evaluate

    original = source(uniform_epilogue=(kind, literal), update_state=True)
    assert scalarize_splat_inputs(original) == 1
    before = ir_text(original)
    selected = plan(original)
    assert selected is not None
    state = np.full((3, 1, 7), 127, np.int8)
    for epoch in range(3):
        lhs = ((np.arange(21) * 3 + epoch * 5) % 17 - 8).astype(np.int8).reshape(3, 1, 7)
        rhs = ((np.arange(56) * 7 + epoch * 3) % 17 - 8).astype(np.int8).reshape(1, 8, 7)
        expected = []
        for row in range(3):
            result = np.asarray(
                [[sum(int(lhs[row, 0, k]) * int(rhs[0, col, k]) for k in range(7)) for col in range(8)]], np.int32
            )
            if row == 0:
                epilogue = {
                    "addi": lambda x: x + literal,
                    "maxsi": lambda x: max(x, literal),
                    "minsi": lambda x: min(x, literal),
                }[kind]
                result = np.asarray([[epilogue(int(value)) for value in result[0]]], np.int32)
            expected.append(result)
        next_state = ((lhs.astype(np.int16) + state + 128) % 256 - 128).astype(np.int8)
        expected += [*[np.zeros((1, 8), np.int32) for _ in range(3)], lhs, rhs, next_state]
        for module in (original, selected.candidate):
            actual = evaluate(module, [lhs, rhs, state.copy()])
            assert len(actual) == len(expected) == 9
            for got, want in zip(actual, expected, strict=True):
                assert got.dtype == want.dtype and got.shape == want.shape
                np.testing.assert_array_equal(got, want)
        state = next_state.copy()
    selected.revalidate()
    assert ir_text(original) == before


def scalarized_broadcast():
    from merlin.llvmlower.splat_inputs import scalarize_splat_inputs

    module = source(uniform_epilogue=("addi", 4))
    assert scalarize_splat_inputs(module) == 1
    operation = next(
        op
        for op in module.walk()
        if type(op) is GenericOp and len(op.inputs) == 2 and tuple(op.inputs[1].type.get_shape()) == ()
    )
    return module, operation


@pytest.mark.parametrize("damage", ["nonconstant", "map", "flags", "initializer", "opaque", "partial", "metadata"])
def test_scalarized_span_requires_literal_maps_totality_and_known_effects(damage):
    from xdsl.ir.affine import AffineDimExpr, AffineMap

    from merlin.transforms import dense_row_batching_check as check

    module, operation = scalarized_broadcast()
    assert check.safe_span((operation,)) and plan(module) is not None
    block = operation.body.block
    arithmetic = block.first_op
    splat = operation.inputs[1].owner
    if damage == "nonconstant":
        function = module.body.block.first_op
        argument = function.body.block.insert_arg(builtin.i32, len(function.body.block.args))
        function.properties["function_type"] = builtin.FunctionType.from_lists(
            [*function.function_type.inputs, builtin.i32], function.function_type.outputs
        )
        splat.operands = (argument,)
    elif damage == "map":
        maps = list(operation.indexing_maps)
        maps[0] = builtin.AffineMapAttr(AffineMap(2, 0, (AffineDimExpr(1), AffineDimExpr(0))))
        operation.properties["indexing_maps"] = builtin.ArrayAttr(maps)
    elif damage == "flags":
        arithmetic.properties["overflowFlags"] = arith.IntegerOverflowAttr((arith.IntegerOverflowFlag.NSW,))
    elif damage == "initializer":
        arithmetic.operands = (block.args[0], block.args[-1])
    elif damage in ("opaque", "partial"):
        if damage == "opaque":
            module.body.block.add_op(func.FuncOp.external("opaque", [builtin.i32, builtin.i32], [builtin.i32]))
            replacement = func.CallOp("opaque", [block.args[0], block.args[1]], [builtin.i32])
        else:
            splat.input.owner.properties["value"] = builtin.IntegerAttr(0, builtin.i32)
            replacement = arith.DivSIOp(block.args[0], block.args[1])
        block.insert_op_before(replacement, arithmetic)
        arithmetic.results[0].replace_all_uses_with(replacement.results[0])
        block.erase_op(arithmetic)
    else:
        splat.attributes["prov.unknown_effect"] = builtin.StringAttr("unknown")
    module.verify()
    assert not check.safe_span((operation,))
    assert plan(module) is None


def test_scalarized_span_refuses_float_literal_broadcast():
    from merlin.transforms import dense_row_batching_check as check

    module = parse_mlir_text("""module { func.func @forward(%a:tensor<2x2xf32>) -> tensor<2x2xf32> {
      %c = arith.constant 4.0 : f32
      %s = tensor.splat %c : tensor<f32>
      %e = tensor.empty() : tensor<2x2xf32>
      %r = linalg.generic {indexing_maps=[affine_map<(i,j)->(i,j)>,affine_map<(i,j)->()>,affine_map<(i,j)->(i,j)>],
        iterator_types=["parallel","parallel"]} ins(%a,%s:tensor<2x2xf32>,tensor<f32>) outs(%e:tensor<2x2xf32>) {
      ^bb(%x:f32,%literal_arg:f32,%unused:f32):
        %v = arith.addf %x,%literal_arg : f32
        linalg.yield %v : f32
      } -> tensor<2x2xf32>
      return %r : tensor<2x2xf32>
    }}""")
    module.verify()
    operation = next(o for o in module.walk() if type(o) is GenericOp)
    assert not check.safe_span((operation,))


def test_scalarized_span_requires_original_literal_dominance():
    from merlin.transforms import dense_row_batching_check as check

    module, operation = scalarized_broadcast()
    assert check.safe_span((operation,))
    splat = operation.inputs[1].owner
    constant = splat.input.owner
    parent = splat.parent
    parent.detach_op(constant)
    parent.insert_op_after(constant, splat)
    assert not check.safe_span((operation,))


def test_scalarized_span_reader_replacement_invalidates_issued_plan(monkeypatch):
    from merlin.transforms import dense_row_batching_check as check

    module, _ = scalarized_broadcast()
    selected = plan(module)
    assert selected is not None
    monkeypatch.setattr(check, "_constant_broadcast_source", lambda op: True)
    with pytest.raises(ValueError, match="changed"):
        selected.revalidate()
