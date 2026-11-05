"""Typed pad/gather realization, bounds safety and conservative source refusal."""

from __future__ import annotations

import pytest
from xdsl.dialects import builtin, func, scf, tensor
from xdsl.dialects.linalg.ops import GenericOp
from xdsl.ir import Block, Region
from xdsl.utils.exceptions import VerifyException

from merlin.common import mlir_query as mq
from merlin.llvmlower.im2col_padding import match_padding_gather, rewrite_padding_gathers
from merlin.transforms.certificates import ir_text


def source(*, height=3, width=4, top=1, left=1, bottom=1, right=1, sh=1, sw=1, dh=1, dw=1):
    kh, kw = 3, 3
    ph, pw = height + top + bottom, width + left + right
    oh, ow = (ph - (kh - 1) * dh - 1) // sh + 1, (pw - (kw - 1) * dw - 1) // sw + 1
    image, padded = f"tensor<1x1x{height}x{width}xi8>", f"tensor<1x1x{ph}x{pw}xi8>"
    output = f"tensor<1x{kh}x{kw}x1x{oh}x{ow}xi8>"
    return f"""module {{
func.func @forward(%image:{image})->{output} {{
%zero=arith.constant 0:i8
%pe=tensor.empty():{padded}
%fill=linalg.fill ins(%zero:i8) outs(%pe:{padded})->{padded}
%pad="tensor.insert_slice"(%image,%fill) <{{operandSegmentSizes=array<i32:1,1,0,0,0>,
static_offsets=array<i64:0,0,{top},{left}>,static_sizes=array<i64:1,1,{height},{width}>,
static_strides=array<i64:1,1,1,1>}}> :({image},{padded})->{padded}
%ge=tensor.empty():{output}
%g=linalg.generic {{indexing_maps=[affine_map<(c,kh,kw,n,oh,ow)->(n,c,oh*{sh}+kh*{dh},ow*{sw}+kw*{dw})>,
affine_map<(c,kh,kw,n,oh,ow)->(c,kh,kw,n,oh,ow)>],
iterator_types=["parallel","parallel","parallel","parallel","parallel","parallel"]}}
ins(%pad:{padded}) outs(%ge:{output}) {{^bb0(%value:i8,%old:i8):
linalg.yield %value:i8
}}->{output}
func.return %g:{output}
}}
}}"""


def realized(text=None):
    module = mq.parse(source() if text is None else text)
    module.verify()
    reports = rewrite_padding_gathers(module, index_bits=64)
    module.verify()
    return module, reports


@pytest.mark.parametrize("shape", [{}, {"height": 5, "width": 7, "sh": 2, "sw": 2, "dh": 2}])
def test_realization_has_only_guarded_unpadded_reads_and_retains_complete_output(shape):
    before = mq.parse(source(**shape))
    module, reports = realized(source(**shape))
    assert len(reports) == 1 and reports[0]["added_operations"] == 35
    assert not any(type(op) is tensor.InsertSliceOp for op in module.walk())
    writer = next(op for op in module.walk() if type(op) is GenericOp)
    assert not writer.inputs and len(writer.outputs) == 1
    reads = [op for op in module.walk() if type(op) is tensor.ExtractOp]
    assert len(reads) == 1
    branch = next(op for op in writer.walk() if type(op) is scf.IfOp)
    assert reads[0].parent_block() is branch.true_region.block
    assert not any(type(op) is tensor.ExtractOp for op in branch.false_region.walk())
    assert next(op for op in before.walk() if type(op) is GenericOp).results[0].type == writer.results[0].type
    # Surviving source image is captured directly, not a padded alias.
    assert reads[0].tensor is module.body.block.first_op.body.block.args[0]


@pytest.mark.parametrize(
    "change",
    [
        lambda s: s.replace("constant 0:i8", "constant 1:i8"),
        lambda s: s.replace("static_strides=array<i64:1,1,1,1>", "static_strides=array<i64:1,1,1,2>"),
        lambda s: s.replace("linalg.yield %value:i8", "linalg.yield %old:i8"),
        lambda s: s.replace("oh*1+kh*1", "oh*1+kh*1+1"),
        lambda s: s.replace("%ge=tensor.empty()", "func.call @opaque() : () -> ()\n%ge=tensor.empty()").replace(
            "func.func @forward", "func.func private @opaque()\nfunc.func @forward"
        ),
    ],
)
def test_unrecognized_padding_or_window_remains_byte_identical(change):
    module = mq.parse(change(source()))
    previous = ir_text(module)
    assert rewrite_padding_gathers(module, index_bits=64) == ()
    assert ir_text(module) == previous


def test_growth_and_index_bounds_refuse_before_construction():
    for budget in (0, 34):
        module = mq.parse(source())
        previous = ir_text(module)
        assert rewrite_padding_gathers(module, index_bits=64, max_added_operations=budget) == ()
        assert ir_text(module) == previous
    module = mq.parse(source())
    gather = next(op for op in module.walk() if type(op) is GenericOp)
    assert match_padding_gather(gather, index_bits=7) is None  # complete 108-byte gather exceeds signed bound
    for bits in (True, 0, 129):
        with pytest.raises(ValueError, match="index width"):
            match_padding_gather(gather, index_bits=bits)


def test_zero_extent_preserves_source_and_invalid_scalar_result_refuses_before_mutation():
    module = mq.parse(source(height=0))
    module.verify()
    previous = ir_text(module)
    assert rewrite_padding_gathers(module, index_bits=64) == ()
    assert ir_text(module) == previous
    module = mq.parse(source())
    gather = next(op for op in module.walk() if type(op) is GenericOp)
    gather.properties.pop("iterator_types")
    previous = ir_text(module)
    with pytest.raises(VerifyException, match="iterator_types"):
        rewrite_padding_gathers(module, index_bits=64)
    assert ir_text(module) == previous


def test_live_padded_observer_refuses_but_existing_downstream_call_and_fanout_survive():
    text = source().replace("func.func @forward", "func.func private @observe(tensor<1x1x5x6xi8>)\nfunc.func @forward")
    text = text.replace("func.return %g", "func.call @observe(%pad):(tensor<1x1x5x6xi8>)->()\nfunc.return %g")
    module = mq.parse(text)
    module.verify()
    previous = ir_text(module)
    assert rewrite_padding_gathers(module, index_bits=64) == ()
    assert ir_text(module) == previous

    result = "tensor<1x3x3x1x3x4xi8>"
    text = (
        source()
        .replace("func.func @forward", f"func.func private @consume({result})->{result}\nfunc.func @forward")
        .replace(f"->{result} {{", f"->({result},{result}) {{", 1)
    )
    text = text.replace(
        f"func.return %g:{result}",
        f"%used=func.call @consume(%g):({result})->{result}\nfunc.return %g,%used:{result},{result}",
    )
    module, report = realized(text)
    assert len(report) == 1
    writer = next(op for op in module.walk() if type(op) is GenericOp)
    assert {use.operation.name for use in writer.results[0].uses} == {"func.call", "func.return"}


@pytest.mark.parametrize("kind", ["unmodeled_region", "function_metadata", "module_metadata"])
def test_unknown_enclosing_execution_scope_keeps_exact_original(kind):
    module = mq.parse(source())
    function = module.body.block.first_op
    if kind == "unmodeled_region":
        block = function.body.block
        original_return = block.last_op
        value = original_return.arguments[0]
        block.erase_op(original_return)
        inner = Block()
        for op in tuple(block.ops):
            inner.add_op(block.detach_op(op))
        inner.add_op(scf.YieldOp(value))
        scope = scf.ExecuteRegionOp([value.type], Region(inner))
        block.add_ops([scope, func.ReturnOp(scope.results[0])])
    else:
        (function if kind == "function_metadata" else module).attributes["unknown.execution"] = builtin.UnitAttr()
    module.verify()
    previous = ir_text(module)
    assert rewrite_padding_gathers(module, index_bits=64) == ()
    assert ir_text(module) == previous


def test_explicit_choice_uses_independent_checks_and_default_retains_exact_source(monkeypatch):
    from merlin.llvmlower.automatic_gathers import prepare_automatic_gathers
    from merlin.llvmlower.optional_passes import Selection, applied

    monkeypatch.delenv("MERLIN_AUTOMATIC_HOST_GATHERS", raising=False)
    text = source()
    unchanged, disabled = prepare_automatic_gathers(text, serial=True)
    assert unchanged == text and disabled["rewrites"] == 0 and disabled["timing"] == "UNKNOWN"
    with applied(Selection.parse("automatic-host-gathers")):
        selected, report = prepare_automatic_gathers(text, serial=True)
        assert selected != text and report["rewrites"] == 1
        assert len(report["realized_regions"]) == 1
        assert [row["name"] for row in report["checks"]] == [
            "padding_gather_source",
            "padding_gather_rewrite",
            "padding_gather_conformance",
        ]
        assert report["checks"][1]["invalidated"] == ["source"] or report["checks"][1]["invalidated"] == ("source",)
        for controls in (
            dict(serial=False),
            dict(serial=True, index_bits=32),
            dict(serial=True, max_added_operations=34),
        ):
            result, declined = prepare_automatic_gathers(text, **controls)
            assert result == text and declined["rewrites"] == 0


def test_checker_cannot_change_the_original_source_after_observation(monkeypatch):
    from merlin.llvmlower import im2col_padding_check
    from merlin.llvmlower.automatic_gathers import prepare_automatic_gathers
    from merlin.llvmlower.optional_passes import Selection, applied

    verify = im2col_padding_check.verify_padding_gather_rewrite

    def changed(before, after, **kwargs):
        observed = verify(before, after, **kwargs)
        before.attributes["late_mutation"] = builtin.UnitAttr()
        return observed

    monkeypatch.setattr(im2col_padding_check, "verify_padding_gather_rewrite", changed)
    with applied(Selection.parse("automatic-host-gathers")), pytest.raises(ValueError, match="original source changed"):
        prepare_automatic_gathers(source(), serial=True)


@pytest.mark.parametrize("width", [8, 16, 32])
@pytest.mark.parametrize(
    "geometry",
    [dict(), dict(shape=(2, 2, 3, 5), padding=(1, 2, 2, 1), kernel=(2, 3), steps=(2, 2, 2, 1))],
)
def test_realizer_handles_independent_rectangular_multibatch_and_width_cases(width, geometry):
    from test_im2col_padding_check import pair

    from merlin.llvmlower.im2col_padding_check import verify_padding_gather_rewrite

    # The checker's sources are constructed independently, including calls on
    # both sides of the island and complete result fanout.
    original = mq.parse(pair(width=width, **geometry)[0])
    current = original.clone()
    assert len(rewrite_padding_gathers(current, index_bits=64)) == 1
    current.verify()
    observed = verify_padding_gather_rewrite(original, current, index_bits=64)
    assert len(observed) == 1
    assert observed[0]["eliminated_padding_payload_bytes"] > 0
