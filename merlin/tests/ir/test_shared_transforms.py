"""Execute actual before/after IR and reject malformed transformation witnesses."""

from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from itertools import islice

import pytest
from xdsl.dialects import arith, builtin, func, memref, scf
from xdsl.interpreter import Interpreter, InterpreterFunctions, impl, register_impls
from xdsl.interpreters.arith import ArithFunctions
from xdsl.interpreters.func import FuncFunctions
from xdsl.interpreters.memref import MemRefFunctions, xtype_for_el_type
from xdsl.interpreters.scf import ScfFunctions
from xdsl.interpreters.shaped_array import ShapedArray
from xdsl.interpreters.utils.ptr import TypedPtr
from xdsl.ir import Block, Region

from merlin.transforms import PipelineOptions, apply_common_pipeline, iter_segments, plan_dependencies, plan_storage
from merlin.transforms.certificates import TransformationChange, ir_digest, ir_text, parse_ir
from merlin.transforms.normalization import signed_wrap
from merlin.transforms.planning import apply_order, reuse_storage, validate_order
from merlin.transforms.validation import validate_change


@register_impls
class _ViewAndBoundFunctions(InterpreterFunctions):
    @impl(memref.CastOp)
    def cast(self, interpreter, op, args):
        return args

    @impl(memref.DimOp)
    def dim(self, interpreter, op, args):
        source, axis = args
        return (source.shape[axis],)

    @impl(arith.MinUIOp)
    def minimum(self, interpreter, op, args):
        assert min(args) >= 0
        return (min(args),)


def _interpret(module, inputs=(), bits=64):
    interpreter = Interpreter(module, index_bitwidth=bits)
    for implementations in (
        ArithFunctions(),
        ScfFunctions(),
        MemRefFunctions(),
        FuncFunctions(),
        _ViewAndBoundFunctions(),
    ):
        interpreter.register_implementations(implementations)
    return interpreter.call_op("forward", inputs)


def _array(values, shape, dtype=builtin.i8):
    return ShapedArray(TypedPtr.new(values, xtype=xtype_for_el_type(dtype, 64)), list(shape))


def _chain(m, n, k):
    """Two outputs, shared producer, views, wrapping reduction and reusable temps."""
    types = [builtin.MemRefType(builtin.i8, s) for s in ((m, k), (k, n), (m, n), (m, n))]
    entry = Block(arg_types=types)
    bounds = {}
    for size in {0, 1, m, n, k}:
        bounds[size] = arith.ConstantOp.from_int_and_width(size, builtin.IndexType())
        entry.add_op(bounds[size])
    zero = arith.ConstantOp.from_int_and_width(0, builtin.i8)
    one = arith.ConstantOp.from_int_and_width(1, builtin.i8)
    entry.add_ops([zero, one])
    view = memref.CastOp.get(entry.args[0], types[0])
    dimension = memref.DimOp.build(operands=[view.dest, bounds[0]], result_types=[builtin.IndexType()])
    entry.add_ops([view, dimension])
    # These constants exercise wrap folding; the neutral expression below also
    # exercises nonconstant scalar simplification inside a carried reduction.
    left, right = (arith.ConstantOp.from_int_and_width(v, builtin.i8) for v in (120, 20))
    folded = arith.AddiOp(left, right)
    entry.add_ops([left, right, folded])

    def nest(shape, body_fn, parent=entry):
        def emit(dimension, indices, block):
            if dimension == len(shape):
                body_fn(block, indices)
                return
            body = Block(arg_types=[builtin.IndexType()])
            emit(dimension + 1, (*indices, body.args[0]), body)
            body.add_op(scf.YieldOp())
            block.add_op(scf.ForOp(bounds[0], bounds[shape[dimension]], bounds[1], [], body))

        emit(0, (), parent)

    temp_type = types[2]
    temp1 = memref.AllocOp([], [], temp_type)
    entry.add_op(temp1)
    nest((m, n), lambda block, indices: block.add_op(memref.StoreOp.get(zero, temp1.memref, indices)))

    def contraction(block, indices):
        row, col = indices
        reduction = Block(arg_types=[builtin.IndexType(), builtin.i8])
        rk, accumulator = reduction.args
        a = memref.LoadOp.get(view.dest, [row, rk])
        b = memref.LoadOp.get(entry.args[1], [rk, col])
        multiply = arith.MuliOp(a, b)
        add = arith.AddiOp(accumulator, multiply)
        neutral = arith.AddiOp(add, zero)
        reduction.add_ops([a, b, multiply, add, neutral, scf.YieldOp(neutral)])
        loop = scf.ForOp(bounds[0], bounds[k], bounds[1], [zero], reduction)
        block.add_ops([loop, memref.StoreOp.get(loop.results[0], temp1.memref, indices)])

    nest((m, n), contraction)

    def consumers(block, indices):
        load = memref.LoadOp.get(temp1.memref, indices)
        add = arith.AddiOp(load, one)
        offset = arith.AddiOp(load, folded)
        block.add_ops(
            [
                load,
                add,
                offset,
                memref.StoreOp.get(add, entry.args[2], indices),
                memref.StoreOp.get(offset, entry.args[3], indices),
            ]
        )

    nest((m, n), consumers)
    entry.add_op(memref.DeallocOp.get(temp1.memref))
    temp2 = memref.AllocOp([], [], temp_type)
    entry.add_op(temp2)
    nest((m, n), lambda block, indices: block.add_op(memref.StoreOp.get(zero, temp2.memref, indices)))

    def second_lifetime(block, indices):
        load = memref.LoadOp.get(entry.args[2], indices)
        block.add_ops([load, memref.StoreOp.get(load, temp2.memref, indices)])
        reload = memref.LoadOp.get(temp2.memref, indices)
        add = arith.AddiOp(reload, one)
        block.add_ops([reload, add, memref.StoreOp.get(add, entry.args[2], indices)])

    nest((m, n), second_lifetime)
    entry.add_ops([memref.DeallocOp.get(temp2.memref), func.ReturnOp(dimension.result)])
    module = builtin.ModuleOp([func.FuncOp("forward", (types, [builtin.IndexType()]), Region(entry))])
    module.verify()
    return module


def _reduction(lower, upper, step=1, width=8, initial=7):
    dtype = builtin.IntegerType(width)
    entry = Block()
    bounds = [arith.ConstantOp.from_int_and_width(v, builtin.IndexType()) for v in (lower, upper, step)]
    seed = arith.ConstantOp.from_int_and_width(initial, dtype, truncate_bits=True)
    factor = arith.ConstantOp.from_int_and_width(3, dtype)
    increment = arith.ConstantOp.from_int_and_width(1, dtype)
    entry.add_ops([*bounds, seed, factor, increment])
    body = Block(arg_types=[builtin.IndexType(), dtype])
    multiply = arith.MuliOp(body.args[1], factor)
    add = arith.AddiOp(multiply, increment)
    body.add_ops([multiply, add, scf.YieldOp(add.result)])
    loop = scf.ForOp(*bounds, [seed], body)
    entry.add_ops([loop, func.ReturnOp(loop.results[0])])
    module = builtin.ModuleOp([func.FuncOp("forward", ([], [dtype]), Region(entry))])
    module.verify()
    return module


@pytest.mark.parametrize("m,n,k,tiles", [(2, 3, 7, (3, 2, 4)), (3, 5, 1, (2, 4, 8)), (1, 2, 9, (7, 1, 3))])
def test_connected_program_executes_before_after_and_repeated_calls(m, n, k, tiles):
    source = _chain(m, n, k)
    original = ir_text(source)
    result = apply_common_pipeline(source, PipelineOptions(64, tiles), plan_only=True)
    assert ir_text(source) == original
    assert {c.kind for c in result.changes} == {
        "shape_view_normalization",
        "exact_integer_simplification",
        "buffer_reuse",
        "ordered_loop_segmentation",
    }
    assert sum(isinstance(op, memref.AllocOp) for op in source.walk()) == 2
    assert sum(isinstance(op, memref.AllocOp) for op in result.module.walk()) == 1
    for repeat in range(2):
        av = [signed_wrap(i * 23 + repeat * 11, 8) for i in range(m * k)]
        bv = [signed_wrap(i * 17 - repeat * 7, 8) for i in range(k * n)]
        expected0, expected1 = [], []
        for row in range(m):
            for col in range(n):
                dot = 0
                for rk in range(k):
                    dot = signed_wrap(dot + signed_wrap(av[row * k + rk] * bv[rk * n + col], 8), 8)
                expected0.append(signed_wrap(dot + 2, 8))
                expected1.append(signed_wrap(dot - 116, 8))
        for module in (source, result.module):
            inputs = [
                _array(av, (m, k)),
                _array(bv, (k, n)),
                _array([99] * (m * n), (m, n)),
                _array([-99] * (m * n), (m, n)),
            ]
            assert _interpret(module, tuple(inputs)) == (m,)
            assert inputs[2].data == expected0
            assert inputs[3].data == expected1
    for change in result.changes:
        validate_change(change)
        decoded = TransformationChange.from_dict(json.loads(json.dumps(change.to_dict())))
        validate_change(decoded)
        assert change.certificate.before_ir_sha256 == ir_digest(change.before_ir)
        assert change.certificate.after_ir_sha256 == ir_digest(change.after_ir)


@pytest.mark.parametrize(
    "lower,upper,step,tile", [(0, 0, 1, 3), (0, 1, 1, 8), (2, 19, 3, 4), (7, 33, 2, 3), (0, 15, 1, 2)]
)
@pytest.mark.parametrize("width", [8, 16, 32])
def test_ordered_reduction_preserves_each_accumulator_step(lower, upper, step, tile, width):
    source = _reduction(lower, upper, step, width)
    result = apply_common_pipeline(source, PipelineOptions(64, (tile,), reuse_buffers=False), plan_only=True)
    expected = 7
    for _ in range(lower, upper, step):
        expected = signed_wrap(expected * 3 + 1, width)
    assert _interpret(source) == _interpret(result.module) == (expected,)
    assert [
        i
        for segment in iter_segments(lower, upper, step, tile)
        for i in range(segment.lower, segment.upper, segment.step)
    ] == list(range(lower, upper, step))


def test_huge_bounds_generate_constant_size_ir_and_lazy_segments():
    small = apply_common_pipeline(_reduction(0, 33), PipelineOptions(64, (17,)), plan_only=True)
    large = apply_common_pipeline(_reduction(0, 10**12 + 3), PipelineOptions(64, (17,)), plan_only=True)
    assert len(tuple(small.module.walk())) == len(tuple(large.module.walk()))
    assert len(ir_text(large.module)) < len(ir_text(small.module)) + 100
    assert len(list(islice(iter_segments(0, 10**12, 1, 17), 3))) == 3
    parse_ir(ir_text(large.module)).verify()


@pytest.mark.parametrize("program", ["ordered_reduction", "connected_buffers"])
def test_large_program_builds_and_links_without_executing(tmp_path, program):
    from merlin.llvmlower import toolchain

    mlir_opt = toolchain.llvm_install() / "bin" / "mlir-opt"
    translate, clang = toolchain.mlir_translate(), toolchain.clang()
    if not all(path.is_file() for path in (mlir_opt, translate, clang)):
        pytest.skip("selected native LLVM toolchain is not installed")
    source = _reduction(0, 10**12 + 3) if program == "ordered_reduction" else _chain(100003, 257, 10007)
    result = apply_common_pipeline(source, PipelineOptions(64, (31, 13, 17)), plan_only=True)
    pipeline = (
        "builtin.module(expand-strided-metadata,convert-scf-to-cf,convert-arith-to-llvm,"
        "finalize-memref-to-llvm,convert-cf-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)"
    )
    lowered = subprocess.run(
        [str(mlir_opt), f"--pass-pipeline={pipeline}"],
        input=ir_text(result.module),
        text=True,
        capture_output=True,
        check=True,
        timeout=20,
    )
    translated = subprocess.run(
        [str(translate), "--mlir-to-llvmir"],
        input=lowered.stdout,
        text=True,
        capture_output=True,
        check=True,
        timeout=20,
    )
    llvm_file, obj, linked = tmp_path / "large.ll", tmp_path / "large.o", tmp_path / "large.so"
    llvm_file.write_text(translated.stdout)
    subprocess.run(
        [str(clang), "-O0", "-fPIC", "-c", str(llvm_file), "-o", str(obj)],
        capture_output=True,
        check=True,
        timeout=20,
    )
    subprocess.run([str(clang), "-shared", str(obj), "-o", str(linked)], capture_output=True, check=True, timeout=20)
    assert obj.stat().st_size < 100_000
    assert linked.stat().st_size < 100_000


def test_requires_host_verifier_and_leaves_source_untouched():
    source = _reduction(0, 11)
    original = ir_text(source)
    with pytest.raises(ValueError, match="host verifier"):
        apply_common_pipeline(source, PipelineOptions(64, (3,)))
    assert ir_text(source) == original
    observed = []
    with pytest.raises(ValueError, match="fresh host Lean completion"):
        apply_common_pipeline(source, PipelineOptions(64, (3,)), verifier=observed.append)
    apply_common_pipeline(source, PipelineOptions(64, (3,)), verifier=observed.append, plan_only=True)
    assert len(observed) == 2
    # This observation callback does not issue a formal proof receipt.
    assert not hasattr(observed[0], "accepted")


def test_overflow_flags_are_not_folded_and_index_overflow_refuses():
    source = parse_ir("""builtin.module {
      func.func @forward() -> i8 {
        %a = arith.constant 120 : i8
        %b = arith.constant 20 : i8
        %c = arith.addi %a, %b overflow<nsw> : i8
        return %c : i8
      }
    }""")
    result = apply_common_pipeline(source, PipelineOptions(64), plan_only=True)
    assert not result.changes
    addition = next(op for op in source.walk() if isinstance(op, arith.AddiOp))
    addition.properties["overflowFlags"] = arith.IntegerOverflowAttr("none")
    wrapping = apply_common_pipeline(source, PipelineOptions(64), plan_only=True)
    assert len(wrapping.changes) == 1
    assert _interpret(wrapping.module) == (-116,)
    with pytest.raises(ValueError, match="index arithmetic"):
        apply_common_pipeline(_reduction(0, 120), PipelineOptions(8, (16,)), plan_only=True)


def test_forged_loop_state_tail_or_body_is_rejected():
    result = apply_common_pipeline(_reduction(0, 11), PipelineOptions(64, (4,)), plan_only=True)
    change = result.changes[0]
    for fault in ("tail", "state", "body"):
        modified = parse_ir(change.after_ir)
        loops = [op for op in modified.walk() if isinstance(op, scf.ForOp)]
        outer, inner = loops
        if fault == "tail":
            inner.operands = [inner.lb, outer.ub, inner.step, *inner.iter_args]
        elif fault == "state":
            inner.operands = [inner.lb, inner.ub, inner.step, *outer.iter_args]
        else:
            multiply = next(op for op in inner.walk() if isinstance(op, arith.MuliOp))
            multiply.operands = [multiply.lhs, multiply.lhs]
        after = ir_text(modified)
        certificate = replace(change.certificate, after_ir_sha256=ir_digest(after))
        forged = replace(change, after_ir=after, certificate=certificate)
        with pytest.raises(ValueError):
            validate_change(forged)
    with pytest.raises(ValueError, match="identity"):
        validate_change(replace(change, after_ir=change.before_ir))
    with pytest.raises(ValueError, match="coverage"):
        validate_change(replace(change, certificate=replace(change.certificate, scope=())))
    with pytest.raises(ValueError, match="tile"):
        validate_change(replace(change, parameters={"index_bits": 64, "tile_iterations": (999,)}))


def test_storage_refuses_escaping_alias_and_missing_full_initialization():
    source = _chain(2, 3, 4)
    initial = plan_storage(source)
    assert len(initial.witnesses[0].intervals) == 2
    first_alloc = next(op for op in source.walk() if isinstance(op, memref.AllocOp))
    entry = first_alloc.parent
    assert entry is not None
    cast = memref.CastOp.get(first_alloc.memref, first_alloc.memref.type)
    entry.insert_op_after(cast, first_alloc)
    escaped = plan_storage(source)
    assert any("alias" in reason for reason in escaped.unknown)
    with pytest.raises(ValueError, match="current IR"):
        reuse_storage(source, initial)
    source = _chain(2, 3, 4)
    first_alloc = next(op for op in source.walk() if isinstance(op, memref.AllocOp))
    first_init = first_alloc.next_op
    assert isinstance(first_init, scf.ForOp)
    first_init.parent.erase_op(first_init)
    assert any("initialization" in reason for reason in plan_storage(source).unknown)


def test_overlapping_storage_lifetimes_get_distinct_slots():
    source = _chain(2, 3, 4)
    entry = next(op for op in source.walk() if isinstance(op, func.FuncOp)).body.block
    deallocations = [op for op in entry.ops if isinstance(op, memref.DeallocOp)]
    entry.detach_op(deallocations[0])
    entry.insert_op_before(deallocations[0], deallocations[1])
    plan = plan_storage(source)
    assert len({i.slot for w in plan.witnesses for i in w.intervals}) == 2
    assert not reuse_storage(source, plan)


def test_dependency_schedule_preserves_effects_and_unknown_operations():
    source = parse_ir("""builtin.module {
      func.func @forward() -> i8 {
        %a = arith.constant 2 : i8
        %b = arith.constant 3 : i8
        "foreign.configure"() : () -> ()
        %c = arith.addi %a, %b : i8
        "foreign.issue"() : () -> ()
        return %c : i8
      }
    }""")
    plan = plan_dependencies(source)
    block = next(w for w in plan.witnesses if len(w.operation_sha256) == 6)
    assert block.unknown_effect_indices == (2, 4)
    validate_order(block, [1, 0, 2, 3, 4, 5])
    with pytest.raises(ValueError, match="dependency"):
        validate_order(block, [0, 1, 4, 3, 2, 5])
    with pytest.raises(ValueError, match="exactly once"):
        validate_order(block, [0, 1, 2, 3, 4, 4])
    witness = apply_order(source, block.block_index, [1, 0, 2, 3, 4, 5])
    assert witness.order == (1, 0, 2, 3, 4, 5)
    source.verify()
