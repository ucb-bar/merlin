"""Exact complete epilogue chains reach real compact-loop C emission."""

import ctypes
import shutil
import subprocess
from dataclasses import replace

import numpy as np
import pytest
from test_dense_integer_readout import QUALIFICATION, readout_inputs
from test_dense_source import compile_source, contraction
from test_dense_source_catalog import implementation

from merlin.compile.scheduling.dense_source import DenseSourceUnavailable, read_source_dense_readout_chain
from merlin.frontends.linalg_mlir import parse_mlir_text


def source(shape, dtypes, *, bound=127, extra_observer=False, unknown=False, reordered=False):
    m, _, n = shape

    def stage(index, previous, in_bits, out_bits, body):
        return f"""
%e{index}=tensor.empty():tensor<{m}x{n}xi{out_bits}>
%v{index}=linalg.generic {{indexing_maps=[affine_map<(i,j)->(i,j)>,affine_map<(i,j)->(i,j)>],
 iterator_types=["parallel","parallel"]}} ins(%{previous}:tensor<{m}x{n}xi{in_bits}>)
 outs(%e{index}:tensor<{m}x{n}xi{out_bits}>) {{^bb0(%x:i{in_bits},%old:i{out_bits}):
 {body}
 linalg.yield %y:i{out_bits}
}}->tensor<{m}x{n}xi{out_bits}>
"""

    positive = "%z=arith.constant 0:i32\n %y=arith.maxsi %x,%z:i32"
    bounded = f"%h=arith.constant {bound}:i32\n %y=arith.minsi %x,%h:i32"
    narrowed = "%y=arith.trunci %x:i32 to i8"
    parts = [stage(0, "r", 32, 32, positive), stage(1, "v0", 32, 32, bounded), stage(2, "v1", 32, 8, narrowed)]
    if reordered:
        parts = [
            stage(0, "r", 32, 8, narrowed),
            stage(1, "v0", 8, 32, "%y=arith.extsi %x:i8 to i32"),
            stage(2, "v1", 32, 32, positive),
            stage(3, "v2", 32, 32, bounded),
            stage(4, "v3", 32, 8, narrowed),
        ]
    extra = f"%seen=tensor.cast %v0:tensor<{m}x{n}xi32> to tensor<?x?xi32>" if extra_observer else ""
    if unknown:
        parts[1] = "func.call @opaque():()->()\n" + parts[1]
    text = f"""module {{
    {"func.func private @opaque()" if unknown else ""}
    func.func @forward(%a:tensor<{m}x{shape[1]}x{dtypes[0]}>,%b:tensor<{shape[1]}x{n}x{dtypes[1]}>)
     ->tensor<{m}x{n}xi8> {{
     {contraction(shape, dtypes)}
     {"".join(parts)}
     {extra}
     func.return %v{len(parts) - 1}:tensor<{m}x{n}xi8>
    }} }}"""
    module = parse_mlir_text(text)
    module.verify()
    ops = tuple(op for op in module.walk() if op.name == "linalg.generic")
    return module, ops[0], ops[1:]


@pytest.mark.parametrize("tile,widths,shape", [((2, 3, 4), (1, 1, 4), (5, 14, 13)), ((3, 2, 5), (2, 2, 4), (7, 9, 16))])
def test_chain_reaches_actual_native_loop_kernel_and_preserves_all_outputs(tmp_path, tile, widths, shape):
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("independent native C compiler unavailable")
    model = readout_inputs(tmp_path, tile=tile, widths=widths, shape=shape)
    module, dense, stages = source(shape, model[0].dense.dtypes)
    result = compile_source(
        model, module, dense, output_operations=stages, readout_qualification_identity=QUALIFICATION
    )
    assert result.operations == (dense, *stages) and result.output_dtype == "i8"
    assert result.output_operation is stages[-1] and len(result.readout_operation_ordinals) == 3
    result.revalidate()
    code = implementation(
        result,
        model[1],
        model[2],
        model[0].dense.dtypes,
        "chain",
        readout_body="if(value<0)value=0;if(value>127)value=127;",
        output_dtype="i8",
    )
    cfile, library = tmp_path / "chain.c", tmp_path / f"chain_w{widths[0]}_m{tile[0]}.so"
    cfile.write_text(code)
    subprocess.run(
        [compiler, "-std=c11", "-O2", "-shared", "-fPIC", str(cfile), "-o", str(library)],
        check=True,
        capture_output=True,
        timeout=30,
    )
    native = ctypes.CDLL(str(library))
    native.chain.argtypes = [ctypes.c_void_p] * 3
    native.chain.restype = None
    m, k, n = shape
    rng = np.random.default_rng(405)
    dtype = np.int8 if widths[0] == 1 else np.int16
    info = np.iinfo(dtype)
    for _ in range(3):
        left = rng.integers(info.min, info.max, (m, k), dtype=dtype)
        right = rng.integers(info.min, info.max, (k, n), dtype=dtype)
        # Include saturation thresholds and signed-i32 modular overflow.
        left[0] = info.max
        right[:, 0] = info.max
        right[:, 1] = info.min
        raw = left.astype(np.int64) @ right.astype(np.int64)
        raw = ((raw + 2**31) % 2**32 - 2**31).astype(np.int64)
        expected = np.clip(raw, 0, 127).astype(np.int8)
        assert np.any(expected == 127) and np.any(expected == 0)
        destination = np.full(m * n + 32, -91, dtype=np.int8)
        result_view = destination[16:-16]
        native.chain(left.ctypes.data, right.ctypes.data, result_view.ctypes.data)
        np.testing.assert_array_equal(result_view.reshape(m, n), expected)
        assert np.all(destination[:16] == -91) and np.all(destination[-16:] == -91)
        assert native.chain_guards() == 1


@pytest.mark.parametrize("mutation", ["bound", "observer", "effects", "conversion"])
def test_changed_recipe_or_observed_intermediate_keeps_source_route(tmp_path, mutation):
    model = readout_inputs(tmp_path)
    module, dense, stages = source(
        (5, 14, 13),
        model[0].dense.dtypes,
        bound=126 if mutation == "bound" else 127,
        extra_observer=mutation == "observer",
        unknown=mutation == "effects",
        reordered=mutation == "conversion",
    )
    with pytest.raises(DenseSourceUnavailable):
        compile_source(model, module, dense, output_operations=stages, readout_qualification_identity=QUALIFICATION)
    ordinary = compile_source(model, module, dense)
    assert ordinary.output_dtype == "i32" and ordinary.operations == (dense,)


@pytest.mark.parametrize("mutation", ["omit", "reorder", "stale", "ordinal", "strip"])
def test_checked_chain_custody_cannot_be_changed_after_scheduling(tmp_path, mutation):
    model = readout_inputs(tmp_path)
    module, dense, stages = source((5, 14, 13), model[0].dense.dtypes)
    result = compile_source(
        model, module, dense, output_operations=stages, readout_qualification_identity=QUALIFICATION
    )
    if mutation == "omit":
        result = replace(result, readout_operations=stages[1:])
    elif mutation == "reorder":
        result = replace(result, readout_operations=tuple(reversed(stages)))
    elif mutation == "stale":
        from xdsl.dialects.builtin import IntegerAttr, i32

        constant = next(op for op in stages[1].walk() if op.name == "arith.constant")
        constant.properties["value"] = IntegerAttr(126, i32)
    elif mutation == "ordinal":
        result = replace(result, readout_operation_ordinals=(0, 1, 2))
    else:
        result = replace(result, readout_operations=(), readout_operation_ordinals=())
    with pytest.raises(ValueError, match="replan"):
        result.revalidate()


def test_aggregate_stage_operation_and_byte_budgets_are_enforced(tmp_path):
    model = readout_inputs(tmp_path)
    _, dense, stages = source((5, 14, 13), model[0].dense.dtypes)
    for limit in ({"max_stages": 2}, {"max_scalar_operations": 3}, {"max_scalar_ir_bytes": 100}):
        with pytest.raises(DenseSourceUnavailable):
            read_source_dense_readout_chain(
                dense, stages, selected_contracts=model[0], qualification_identity=QUALIFICATION, **limit
            )


def test_compiled_program_from_another_shape_cannot_replace_source_binding(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    first = readout_inputs(a, shape=(5, 14, 13))
    second = readout_inputs(b, shape=(7, 9, 16), tile=(3, 2, 5))
    ma, oa, ca = source((5, 14, 13), first[0].dense.dtypes)
    mb, ob, cb = source((7, 9, 16), second[0].dense.dtypes)
    ra = compile_source(first, ma, oa, output_operations=ca, readout_qualification_identity=QUALIFICATION)
    rb = compile_source(second, mb, ob, output_operations=cb, readout_qualification_identity=QUALIFICATION)
    with pytest.raises(ValueError, match="source geometry|selection"):
        replace(ra, compiled=rb.compiled, lowered_stamp=rb.lowered_stamp).revalidate()


def test_same_shape_different_source_cannot_rebind_an_original_readout_chain(tmp_path):
    from xdsl.dialects import arith, tensor
    from xdsl.dialects.builtin import i8
    from xdsl.dialects.linalg.ops import FillOp
    from xdsl.rewriter import InsertPoint, Rewriter

    model = readout_inputs(tmp_path)
    module, dense, stages = source((5, 14, 13), model[0].dense.dtypes)
    zero = arith.ConstantOp.from_int_and_width(0, i8)
    empty = tensor.EmptyOp([], dense.inputs[0].type)
    fill = FillOp([zero.result], [empty.tensor])
    other = dense.clone({dense.inputs[0]: fill.results[0]})
    Rewriter.insert_op([zero, empty, fill, other], InsertPoint.before(dense.parent_block().last_op))
    module.verify()
    result = compile_source(
        model, module, dense, output_operations=stages, readout_qualification_identity=QUALIFICATION
    )
    ordinal = next(i for i, op in enumerate(module.walk()) if op is other)
    with pytest.raises(ValueError, match="consumer|source|replan"):
        replace(result, operation=other, operation_ordinal=ordinal).revalidate()
