"""Shared host composition preserves products, live users and numerical tails."""

import hashlib
import subprocess

import numpy as np
import pytest

from merlin.compile.scheduling.dense_contracts import DenseIntegerReadout, DenseLogicalPointerABI
from merlin.llvmlower.abi import HostModel
from merlin.llvmlower.borrowed_buffer_calls import BorrowedBufferEffects
from merlin.llvmlower.integer_matmul_calls import lower_integer_matmul_program, prepare_integer_matmul_calls
from merlin.llvmlower.logical_borrowed_abi import (
    LogicalBorrowedABI,
)
from merlin.llvmlower.target_data_layout import _query
from merlin.llvmlower.toolchain import clang

EFFECTS = BorrowedBufferEffects("neutral synchronous native test product", True, True, True)
READOUT = DenseIntegerReadout(
    """module {func.func @forward(%x:i32)->i8 {
      %lo=arith.constant -128:i32
      %hi=arith.constant 127:i32
      %l=arith.maxsi %x,%lo:i32
      %h=arith.minsi %l,%hi:i32
      %r=arith.trunci %h:i32 to i8
      return %r:i8 }}""", "i8", "neutral_saturated_store", (),
)


def source(m=3, k=7, n=5, *, initial=0, extension="extsi", extra_return=False):
    result = f"tensor<{m}x{n}xi32>"
    out = f"tensor<{m}x{n}xi8>"
    returned = f"{result},{out}" if extra_return else out
    values = "%r,%y" if extra_return else "%y"
    return f"""module {{
func.func @forward(%a:tensor<{m}x{k}xi8>,%b:tensor<{k}x{n}xi8>) -> ({returned}) {{
 %z=arith.constant {initial}:i32
 %init=tensor.splat %z:{result}
 %r=linalg.generic {{indexing_maps=[affine_map<(m,n,k)->(m,k)>,
 affine_map<(m,n,k)->(k,n)>,affine_map<(m,n,k)->(m,n)>],
 iterator_types=["parallel","parallel","reduction"]}}
 ins(%a,%b:tensor<{m}x{k}xi8>,tensor<{k}x{n}xi8>) outs(%init:{result}) attrs = {{prov.region_id="dot"}} {{
 ^bb0(%x:i8,%w:i8,%acc:i32):
 %xx=arith.{extension} %x:i8 to i32
 %ww=arith.{extension} %w:i8 to i32
 %p=arith.muli %xx,%ww:i32
 %s=arith.addi %acc,%p:i32
 linalg.yield %s:i32
 }} -> {result}
 %empty=tensor.empty():{out}
 %y=linalg.generic {{indexing_maps=[affine_map<(m,n)->(m,n)>,affine_map<(m,n)->(m,n)>],
 iterator_types=["parallel","parallel"]}} ins(%r:{result}) outs(%empty:{out}) {{
 ^bb1(%v:i32,%unused:i8):
 %lo=arith.constant -128:i32
 %hi=arith.constant 127:i32
 %low=arith.maxsi %v,%lo:i32
 %high=arith.minsi %low,%hi:i32
 %narrow=arith.trunci %high:i32 to i8
 linalg.yield %narrow:i8
 }} -> {out}
 return {values}:{returned}
}}}}"""


def prepare(text, **kwargs):
    return prepare_integer_matmul_calls(text, select=lambda call: True, borrowed_effects=EFFECTS, **kwargs)


@pytest.mark.parametrize("changes", [{"initial": 1}, {"extension": "extui"}])
def test_unsupported_recurrence_retains_exact_source(changes):
    text = source(**changes)
    prepared = prepare(text)
    assert not prepared.calls and prepared.source == text


def test_selection_decline_and_changed_prepared_file(tmp_path):
    text = source()
    declined = prepare_integer_matmul_calls(text, select=lambda call: False, borrowed_effects=EFFECTS)
    assert declined.source == text and not declined.calls
    prepared = prepare(text)
    path = tmp_path / "prepared.mlir"
    path.write_text(prepared.source + "\n")
    with pytest.raises(ValueError, match="changed"):
        prepared.borrowed_calls(path)
    with pytest.raises(ValueError, match="Boolean"):
        prepare_integer_matmul_calls(text, select=lambda call: 1, borrowed_effects=EFFECTS)


def test_changed_arithmetic_and_unknown_effects_keep_original_source():
    for text in (
        source().replace("arith.addi %acc,%p", "arith.subi %acc,%p"),
        source().replace('prov.region_id="dot"', 'unknown.effect="opaque"'),
    ):
        assert prepare(text).source == text


def test_existing_symbols_are_not_captured():
    text = source().replace("module {", "module {func.func private @__merlin_integer_dot_0()\n", 1)
    prepared = prepare(text)
    assert prepared.calls[0].symbol == "__merlin_integer_dot_1"


def split_epilogue():
    """Separate broadcast constants and cmp/select stages as emitted by captures."""
    text = source().split(" %empty=tensor.empty()")[0]
    for name, previous, constant, predicate in (("lower", "%r", -128, "slt"), ("upper", "%lower", 127, "sgt")):
        text += f"""
 %c_{name}=arith.constant {constant}:i32
 %t_{name}=tensor.splat %c_{name}:tensor<3x5xi32>
 %e_{name}=tensor.empty():tensor<3x5xi32>
 %{name}=linalg.generic {{indexing_maps=[affine_map<(m,n)->(m,n)>,affine_map<(m,n)->(m,n)>,
 affine_map<(m,n)->(m,n)>],iterator_types=["parallel","parallel"]}}
 ins({previous},%t_{name}:tensor<3x5xi32>,tensor<3x5xi32>) outs(%e_{name}:tensor<3x5xi32>) {{
 ^bb0(%v:i32,%bound:i32,%unused:i32):
 %p=arith.cmpi {predicate},%v,%bound:i32
 %selected=arith.select %p,%bound,%v:i32
 linalg.yield %selected:i32
 }} -> tensor<3x5xi32>
"""
    return text + """
 %e=tensor.empty():tensor<3x5xi8>
 %y=linalg.generic {indexing_maps=[affine_map<(m,n)->(m,n)>,affine_map<(m,n)->(m,n)>],
 iterator_types=["parallel","parallel"]} ins(%upper:tensor<3x5xi32>) outs(%e:tensor<3x5xi8>) {
 ^bb0(%v:i32,%unused:i8):
 %t=arith.trunci %v:i32 to i8
 linalg.yield %t:i8
 } -> tensor<3x5xi8>
 return %y:tensor<3x5xi8>
}}"""


@pytest.mark.parametrize("text", [source(), split_epilogue()])
def test_exact_scalar_readout_is_proved_and_host_epilogue_removed(text):
    prepared = prepare(text, output_readout=READOUT)
    assert prepared.calls[0].output_dtype == "i8"
    assert prepared.calls[0].readout_identity == READOUT.identity
    assert len(prepared.readout_proofs) == 1 and prepared.readout_proofs[0].verified
    assert '"linalg.generic"' not in prepared.source
    assert '"arith.trunci"' not in prepared.source


@pytest.mark.parametrize("text", [
    source(extra_return=True),
    source().replace("constant -128", "constant -127"),
    source().replace("%low,%hi:i32", "%v,%hi:i32"),
    source().replace("arith.maxsi", "arith.maxui"),
    source().replace("%high:i32 to i8", "%v:i32 to i8"),
    split_epilogue().replace("cmpi slt", "cmpi ult"),
    source().replace(" %empty=", " func.call @observe() : () -> ()\n %empty=").replace(
        "module {", "module {func.func private @observe()\n", 1,
    ),
])
def test_other_observers_changed_numerics_and_intervening_effects_keep_raw_output(text):
    prepared = prepare(text, output_readout=READOUT)
    assert prepared.calls[0].output_dtype == "i32"
    assert not prepared.readout_proofs
    assert '"arith.trunci"' in prepared.source


def test_backend_may_decline_narrow_readout_and_unknown_attributes_are_retained():
    from xdsl.dialects import builtin

    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.transforms.certificates import ir_text

    prepared = prepare_integer_matmul_calls(
        source(), select=lambda call: call.output_dtype == "i32", borrowed_effects=EFFECTS, output_readout=READOUT,
    )
    assert prepared.calls[0].output_dtype == "i32" and not prepared.readout_proofs
    for name in ("tensor.splat", "arith.constant", "linalg.yield", "arith.trunci"):
        module = parse_mlir_text(source())
        op = [o for o in module.walk() if o.name == name][-1]
        op.attributes["unknown"] = builtin.UnitAttr()
        prepared = prepare(ir_text(module), output_readout=READOUT)
        assert not prepared.calls or prepared.calls[0].output_dtype == "i32"
        assert not prepared.readout_proofs


def test_a_fill_that_reads_its_initializer_cannot_supply_a_zero_seed():
    from xdsl.dialects import tensor
    from xdsl.dialects.linalg.ops import FillOp
    from xdsl.rewriter import Rewriter

    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.transforms.certificates import ir_text

    module = parse_mlir_text(source())
    splat = next(o for o in module.walk() if o.name == "tensor.splat")
    empty = tensor.EmptyOp((), splat.results[0].type)
    fill = FillOp(inputs=[splat.operands[0]], outputs=[empty.results[0]], res=[splat.results[0].type])
    fill.body.block.last_op.operands = (fill.body.block.args[1],)
    Rewriter.replace_op(splat, [empty, fill], fill.results)
    text = ir_text(module)
    assert prepare(text).source == text


@pytest.mark.parametrize("shape", [(1, 1, 1), (3, 7, 5), (17, 31, 19)])
@pytest.mark.parametrize("extra_return", [False, True])
@pytest.mark.parametrize("fuse", [False, True])
@pytest.mark.parametrize("inline_helpers", [False, True, None])
def test_complete_native_outputs_and_live_raw_user(
    tmp_path, upstream_host_tools, shape, extra_return, fuse, inline_helpers,
):
    from merlin.llvmlower.tensor_helper_inlining import FEATURE

    m, k, n = shape
    readout = READOUT if fuse else None
    prepared = prepare(source(m, k, n, extra_return=extra_return), output_readout=readout)
    path = tmp_path / "prepared.mlir"
    path.write_text(prepared.source)
    assert len(prepared.calls) == 1 and prepared.calls[0].shapes == ((m, k), (k, n), (m, n))
    # Execute a separate neutral implementation through the existing checked
    # descriptor bridge, not a replacement implementation of the host source.
    call = prepared.calls[0]
    capability = LogicalBorrowedABI(
        DenseLogicalPointerABI(
            "dense_row_major", (1, 1, 4), ("read_only", "read_only", "fully_written"),
            "logical_elements", "synchronous",
        ), EFFECTS, "neutral test raw signed product", 1 << 20,
        readout_format_qualification_identity="neutral synchronous saturation implementation" if fuse else None,
    )
    adapter = tmp_path / "adapter.c"
    adapter.write_text("#include <stdint.h>\n" + f"""
void {call.kernel_symbol}(void *aa,void *bb,void *cc) {{
 const int8_t *a=aa,*b=bb;{'int8_t' if call.output_dtype == 'i8' else 'int32_t'} *c=cc;
 for(int i=0;i<{m};i++)for(int j=0;j<{n};j++) {{
   int32_t sum=0;for(int p=0;p<{k};p++)sum+=(int32_t)a[i*{k}+p]*b[p*{n}+j];
   c[i*{n}+j]={'sum < -128 ? -128 : sum > 127 ? 127 : sum' if call.output_dtype == 'i8' else 'sum'};
 }}
}}
""")
    implementation = tmp_path / "implementation.ll"
    subprocess.run(
        [str(clang()), "-O2", "-S", "-emit-llvm", str(adapter), "-o", str(implementation)],
        check=True, capture_output=True, timeout=30,
    )
    pins = ((adapter, hashlib.sha256(adapter.read_bytes()).hexdigest()),)
    program = lower_integer_matmul_program(
        source(m, k, n, extra_return=extra_return), tmp_path / "composed",
        select=lambda call: True, emit_kernel=lambda call: (implementation.read_text(), pins),
        logical_abi=capability, data_layout=_query(clang(), ("-O2", "-fPIC")),
        output_readout=readout,
        host_features=None if inline_helpers is None else (FEATURE,) if inline_helpers else (),
    )
    assert program.preparation.calls[0].output_dtype == ("i8" if fuse and not extra_return else "i32")
    if (inline_helpers is None or inline_helpers) and fuse and not extra_return:
        assert "@llvm.memcpy" not in program.host.ll_path.read_text()
        assert "@malloc" not in program.host.ll_path.read_text()
    library = tmp_path / f"model_{m}_{k}_{n}_{extra_return}.so"
    subprocess.run(
        [str(clang()), "-O2", "-shared", "-fPIC", str(program.host.ll_path),
         *map(str, program.kernels), *map(str, program.bridges), "-o", str(library)],
        check=True, capture_output=True, timeout=30,
    )
    invoke = HostModel.load(str(library))
    left = ((np.arange(m * k, dtype=np.int32) * 53 + 128) % 256 - 128).astype(np.int8).reshape(m, k)
    right = ((np.arange(k * n, dtype=np.int32) * 71 + 125) % 256 - 128).astype(np.int8).reshape(k, n)
    original_left, original_right = left.copy(), right.copy()
    expected = left.astype(np.int32) @ right.astype(np.int32)
    for _ in range(2):
        guard = np.full(m * n + 32, 73, np.int8)
        output = guard[16:-16].reshape(m, n)
        raw = np.full((m, n), 0x5A5A5A5A, np.int32)
        arrays = [left, right, *([raw] if extra_return else []), output]
        invoke([(value.ctypes.data, value.shape) for value in arrays])
        np.testing.assert_array_equal(output, np.clip(expected, -128, 127).astype(np.int8))
        if extra_return:
            np.testing.assert_array_equal(raw, expected)
        np.testing.assert_array_equal(left, original_left)
        np.testing.assert_array_equal(right, original_right)
        assert np.all(guard[:16] == 73) and np.all(guard[-16:] == 73)
