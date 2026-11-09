"""Closed source/dataflow proofs; target/actual TorchAO checks are independent."""

from pathlib import Path

import pytest

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.demand_pointwise_gather import DemandGatherEffects, rewrite_module, transform_file
from merlin.llvmlower.deployable_ir import checked_generic_text

EFFECTS = DemandGatherEffects(True, True, True)


def source(n=17, k=23):
    return f"""module {{
func.func @forward(%q:tensor<{n}x{k}xi8>,%s:tensor<{n}x1xf32>,%z:tensor<{n}x1xi8>,
 %ids:tensor<2x4xi64>)->tensor<2x4x{k}xf32> {{
 %e0=tensor.empty():tensor<{n}x{k}xf32>
 %qf=linalg.generic {{indexing_maps=[affine_map<(i,j)->(i,j)>,
 affine_map<(i,j)->(i,j)>],
 iterator_types=["parallel","parallel"]}}
 ins(%q:tensor<{n}x{k}xi8>) outs(%e0:tensor<{n}x{k}xf32>) {{^bb0(%x:i8,%old:f32):
 %v=arith.sitofp %x:i8 to f32
 linalg.yield %v:f32 }}->tensor<{n}x{k}xf32>
 %e1=tensor.empty():tensor<{n}x1xf32>
 %zf=linalg.generic {{indexing_maps=[affine_map<(i,j)->(i,j)>,
 affine_map<(i,j)->(i,j)>],
 iterator_types=["parallel","parallel"]}}
 ins(%z:tensor<{n}x1xi8>) outs(%e1:tensor<{n}x1xf32>) {{^bb0(%x:i8,%old:f32):
 %v=arith.sitofp %x:i8 to f32
 linalg.yield %v:f32 }}->tensor<{n}x1xf32>
 %e2=tensor.empty():tensor<{n}x{k}xf32>
 %sub=linalg.generic {{indexing_maps=[affine_map<(i,j)->(i,j)>,
 affine_map<(i,j)->(i,0)>,
 affine_map<(i,j)->(i,j)>],
 iterator_types=["parallel","parallel"]}}
 ins(%qf,%zf:tensor<{n}x{k}xf32>,tensor<{n}x1xf32>) outs(%e2:tensor<{n}x{k}xf32>) {{^bb0(%x:f32,%z0:f32,%old:f32):
 %v=arith.subf %x,%z0:f32
 linalg.yield %v:f32 }}->tensor<{n}x{k}xf32>
 %e3=tensor.empty():tensor<{n}x{k}xf32>
 %dq=linalg.generic {{indexing_maps=[affine_map<(i,j)->(i,j)>,
 affine_map<(i,j)->(i,0)>,
 affine_map<(i,j)->(i,j)>],
 iterator_types=["parallel","parallel"]}}
 ins(%sub,%s:tensor<{n}x{k}xf32>,tensor<{n}x1xf32>) outs(%e3:tensor<{n}x{k}xf32>) {{^bb0(%x:f32,%s0:f32,%old:f32):
 %v=arith.mulf %x,%s0:f32
 linalg.yield %v:f32 }}->tensor<{n}x{k}xf32>
 %e4=tensor.empty():tensor<2x4x{k}xf32>
 %r=linalg.generic {{indexing_maps=[affine_map<(b,t,c)->(b,t)>,
 affine_map<(b,t,c)->(b,t,c)>],
 iterator_types=["parallel","parallel","parallel"]}}
 ins(%ids:tensor<2x4xi64>) outs(%e4:tensor<2x4x{k}xf32>) {{^bb0(%id:i64,%old:f32):
 %row=arith.index_cast %id:i64 to index
 %col=linalg.index 2:index
 %v=tensor.extract %dq[%row,%col]:tensor<{n}x{k}xf32>
 linalg.yield %v:f32 }}->tensor<2x4x{k}xf32>
 return %r:tensor<2x4x{k}xf32>
}}}}"""


def test_none_performs_no_io_and_leaves_owned_module_uncloned(tmp_path):
    p = tmp_path / "missing"
    assert transform_file(p, tmp_path / "absent", None) is p
    m = parse_mlir_text(source())
    result, r = rewrite_module(m)
    assert result is m and r["selected"] is False


@pytest.mark.parametrize("flag", ["", " overflow<nsw>"])
def test_actual_stock_truncation_roundtrip_retains_unflagged_cast_only(tmp_path, flag):
    import hashlib
    import json

    text = source().removesuffix("}") + (
        "func.func @cast(%x:i32)->i8 { %y=arith.trunci %x" + flag
        + ":i32 to i8 return %y:i8 }\n}"
    )
    original = tmp_path / "original.mlir"
    original.write_text(text)
    output = tmp_path / "selected"
    if flag:
        with pytest.raises(ValueError, match="default-none"):
            transform_file(original, output, EFFECTS)
        assert not (output / "model.mlir").exists()
    else:
        result = transform_file(original, output, EFFECTS)
        module = parse_mlir_text(result.read_text())
        module.verify()
        assert sum(op.name == "arith.trunci" for op in module.walk()) == 1
        report = json.loads((output / "report.json").read_text())
        assert len(report["rewritten"]) == len(report["stock_integer_compatibility"]) == 1
        for path, digest in report["stock_integer_compatibility_input_sha256"].items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
    assert original.read_text() == text


def test_changed_compatibility_dependency_refuses_output(tmp_path, monkeypatch):
    from merlin.llvmlower import stock_arith_compat

    dependency = tmp_path / "bound_helper.py"
    dependency.write_text("# original binding\n")
    monkeypatch.setattr(stock_arith_compat, "__file__", str(dependency))
    actual = stock_arith_compat.unflagged_truncations

    def changed(module):
        result = actual(module)
        dependency.write_text("# changed binding\n")
        return result

    monkeypatch.setattr(stock_arith_compat, "unflagged_truncations", changed)
    original = tmp_path / "source.mlir"
    original.write_text(source())
    with pytest.raises(ValueError, match="source changed"):
        transform_file(original, tmp_path / "refused", EFFECTS)
    assert not (tmp_path / "refused/model.mlir").exists()


def test_original_arithmetic_attributes_and_indices_preserved():
    from xdsl.dialects.builtin import StringAttr

    m = parse_mlir_text(source())
    for i, op in enumerate(m.walk()):
        if op.name in {"arith.sitofp", "arith.subf", "arith.mulf"}:
            op.attributes["prov.fqn"] = StringAttr("source" + str(i))
    original = checked_generic_text(m)
    before = [
        (op.name, tuple(str(v.type) for v in op.operands), dict(op.properties), dict(op.attributes))
        for op in m.walk()
        if op.name in {"arith.sitofp", "arith.subf", "arith.mulf"}
    ]
    fused, r = rewrite_module(m, EFFECTS)
    assert checked_generic_text(m) == original and len(r["rewritten"]) == 1 and not r["refused"]
    assert before == [
        (op.name, tuple(str(v.type) for v in op.operands), dict(op.properties), dict(op.attributes))
        for op in fused.walk()
        if op.name in {"arith.sitofp", "arith.subf", "arith.mulf"}
    ]
    assert sum(op.name == "linalg.generic" for op in fused.walk()) == 1
    assert sum(op.name == "tensor.extract" for op in fused.walk()) == 3
    assert sum(op.name == "arith.index_cast" for op in fused.walk()) == 1
    assert r["specialized_input_values"] is False


@pytest.mark.parametrize(
    "bad", ["init_read", "contract", "library", "broadcast", "partial", "extra_user", "barrier", "scalar_capture"]
)
def test_unsupported_semantics_refuse_without_partial_rewrite(bad):
    from xdsl.dialects import arith, cf, tensor
    from xdsl.dialects.builtin import AffineMapAttr, ArrayAttr, IndexType, StringAttr, i1
    from xdsl.ir.affine import AffineConstantExpr, AffineMap

    m = parse_mlir_text(source())
    gs = [op for op in m.walk() if op.name == "linalg.generic"]
    if bad == "init_read":
        op = next(op for op in gs[-2].body.block.ops if op.name == "arith.mulf")
        op.operands = [gs[-2].body.block.args[-1], op.operands[1]]
    elif bad == "contract":
        gs[0].attributes["prov.unknown_contract"] = StringAttr("unproved")
    elif bad == "library":
        gs[0].properties["library_call"] = StringAttr("unproved")
    elif bad == "broadcast":
        maps = list(gs[-2].indexing_maps)
        maps[1] = AffineMapAttr(AffineMap(2, 0, (AffineConstantExpr(0), AffineConstantExpr(0))))
        gs[-2].indexing_maps = ArrayAttr(maps)
    elif bad == "partial":
        maps = list(gs[0].indexing_maps)
        maps[-1] = AffineMapAttr(AffineMap(2, 0, (AffineConstantExpr(0), AffineConstantExpr(0))))
        gs[0].indexing_maps = ArrayAttr(maps)
    elif bad == "extra_user":
        zero = arith.ConstantOp.from_int_and_width(0, IndexType())
        e = tensor.ExtractOp(gs[-2].results[0], [zero.result, zero.result], gs[-2].results[0].type.element_type)
        gs[-1].parent.insert_ops_before([zero, e], gs[-1])
    elif bad == "barrier":
        yes = arith.ConstantOp.from_int_and_width(1, i1)
        guard = cf.AssertOp(yes.result, "retained guard")
        gs[-1].parent.insert_ops_before([yes, guard], gs[-1])
    elif bad == "scalar_capture":
        op = next(op for op in gs[-2].body.block.ops if op.name == "arith.mulf")
        op.attributes["unknown.numeric_effect"] = StringAttr("unproved")
    original = checked_generic_text(m)
    result, r = rewrite_module(m, EFFECTS)
    assert not r["rewritten"] and r["refused"]
    assert checked_generic_text(result) == original


def test_effect_contract_and_expansion_budget_are_explicit():
    m = parse_mlir_text(source())
    with pytest.raises(ValueError, match="contract"):
        rewrite_module(m, DemandGatherEffects(True, False, True))
    result, r = rewrite_module(m, EFFECTS, max_scalar_ops=1)
    assert not r["rewritten"] and "budget" in r["refused"][0]["reason"]


def test_custom_form_unknown_attribute_is_preserved_and_refused(tmp_path):
    text = source().replace(
        'iterator_types=["parallel","parallel"]',
        'iterator_types=["parallel","parallel"],audit.unproved_effects="unknown"',
        1,
    )
    original = tmp_path / "custom.mlir"
    original.write_text(text)
    output = transform_file(original, tmp_path / "selected", EFFECTS)
    import json

    report = json.loads((output.parent / "report.json").read_text())
    assert not report["rewritten"] and report["refused"]
    assert "audit.unproved_effects" in (output.parent / "normalized.mlir").read_text()
    assert "audit.unproved_effects" in output.read_text()


def test_unknown_gather_or_seed_contract_is_not_dropped():
    from xdsl.dialects.builtin import StringAttr

    for which in ("gather", "seed"):
        module = parse_mlir_text(source())
        gs = [op for op in module.walk() if op.name == "linalg.generic"]
        op = gs[-1] if which == "gather" else gs[0].outputs[0].owner
        op.attributes["unknown.effect"] = StringAttr("unproved")
        original = checked_generic_text(module)
        after, report = rewrite_module(module, EFFECTS)
        assert not report["rewritten"] and report["refused"]
        assert checked_generic_text(after) == original


def test_repeated_dag_inputs_are_budgeted_as_actual_tree_expansion():
    from xdsl.dialects.builtin import ArrayAttr

    module = parse_mlir_text(source())
    mul = [op for op in module.walk() if op.name == "linalg.generic"][-2]
    mul.operands = [mul.inputs[0], mul.inputs[0], mul.outputs[0]]
    maps = list(mul.indexing_maps)
    maps[1] = maps[0]
    mul.indexing_maps = ArrayAttr(maps)
    module.verify()
    before = checked_generic_text(module)
    after, report = rewrite_module(module, EFFECTS, max_scalar_ops=10)
    assert not report["rewritten"] and "budget" in report["refused"][0]["reason"]
    assert checked_generic_text(after) == before
