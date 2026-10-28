"""Refold actual extract/reshape matrix patterns without changing live results."""

import pytest
from xdsl.dialects import builtin, tensor

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.transforms.certificates import ir_digest
from merlin.transforms.head_batching import plan_independent_head_batching


def source(h=3, m=2, k=7, n=5, *, planes=1, omit=None, barrier=False):
    args = []
    for role, shapes in (("a", (h, m, k)), ("b", (h, k, n))):
        for p in range(planes):
            args.append(f"%{role}{p}:tensor<{'x'.join(map(str,shapes))}xi8>")
    outputs, operations, types = [], [], []
    for ap in range(planes):
        for bp in range(planes):
            for head in range(h):
                if head == omit:
                    continue
                name = f"p{ap}_{bp}_{head}"
                for role, p, rows, cols in (("a", ap, m, k), ("b", bp, k, n)):
                    operations.append(f"""
 %{name}{role}s=tensor.extract_slice %{role}{p}[{head},0,0][1,{rows},{cols}][1,1,1]
 :tensor<{h}x{rows}x{cols}xi8> to tensor<1x{rows}x{cols}xi8>
 %{name}{role}f=tensor.collapse_shape %{name}{role}s [[0,1,2]]
 :tensor<1x{rows}x{cols}xi8> into tensor<{rows*cols}xi8>
 %{name}{role}=tensor.expand_shape %{name}{role}f [[0,1]] output_shape [{rows},{cols}]
 :tensor<{rows*cols}xi8> into tensor<{rows}x{cols}xi8>
""")
                operations.append(f"""
 %{name}z=arith.constant 0:i32
 %{name}init=tensor.splat %{name}z:tensor<{m}x{n}xi32>
 %{name}=linalg.generic {{indexing_maps=[affine_map<(m,n,k)->(m,k)>,
 affine_map<(m,n,k)->(k,n)>,affine_map<(m,n,k)->(m,n)>],
 iterator_types=["parallel","parallel","reduction"]}}
 ins(%{name}a,%{name}b:tensor<{m}x{k}xi8>,tensor<{k}x{n}xi8>)
 outs(%{name}init:tensor<{m}x{n}xi32>) {{
 ^bb0(%a:i8,%b:i8,%c:i32):
 %aa=arith.extsi %a:i8 to i32
 %bb=arith.extsi %b:i8 to i32
 %p=arith.muli %aa,%bb:i32
 %s=arith.addi %p,%c:i32
 linalg.yield %s:i32
 }} -> tensor<{m}x{n}xi32>
""")
                if barrier and head == 0:
                    operations.append("func.call @observe():()->()")
                outputs.append("%"+name)
                types.append(f"tensor<{m}x{n}xi32>")
    declaration = "func.func private @observe()" if barrier else ""
    return (f"module{{{declaration}\nfunc.func @forward({','.join(args)})->({','.join(types)}){{"
            + "\n".join(operations) + f"return {','.join(outputs)}:{','.join(types)}\n}}}}")


@pytest.mark.parametrize("shape", [(3, 2, 7, 5), (2, 1, 5, 3), (5, 3, 2, 7)])
@pytest.mark.parametrize("planes", [1, 2])
def test_complete_independent_heads_and_every_plane_pair(shape, planes):
    module = parse_mlir_text(source(*shape, planes=planes))
    original = ir_digest(module)
    plan = plan_independent_head_batching(module)
    record = plan.record()
    assert len(record["batches"]) == planes * planes
    assert record["original_products"] == shape[0] * planes * planes
    assert ir_digest(module) == original
    assert record["source_sha256"] != record["candidate_sha256"]
    # All original observer positions survive the inverse full-IR check.
    plan.revalidate()


@pytest.mark.parametrize("kwargs", [{"omit": 1}, {"barrier": True}])
def test_incomplete_or_effect_crossing_keeps_original(kwargs):
    module = parse_mlir_text(source(**kwargs))
    plan = plan_independent_head_batching(module)
    assert not plan.record()["batches"]
    assert ir_digest(plan.candidate) == ir_digest(module)


def test_changed_actual_result_coordinates_and_mutated_source_refuse():
    module = parse_mlir_text(source())
    plan = plan_independent_head_batching(module)
    view = next(o for o in plan.candidate.walk()
                if type(o) is tensor.ExtractSliceOp and o.source.type.element_type == builtin.i32)
    view.properties["static_offsets"] = builtin.DenseArrayBase.from_list(builtin.i64, [1, 0, 0])
    with pytest.raises(ValueError, match="changed"):
        plan.revalidate()


def test_unchanged_plan_reuses_inverse_but_reconstruction_checks_it_again(monkeypatch):
    from dataclasses import replace

    from merlin.transforms import head_batching

    checked = []
    verify = head_batching._verify

    def counted(*args):
        checked.append(None)
        return verify(*args)

    monkeypatch.setattr(head_batching, "_verify", counted)
    plan = plan_independent_head_batching(parse_mlir_text(source()))
    first = plan.record()
    for _ in range(3):
        assert plan.record() == first
    assert len(checked) == 1
    # Returned nested records cannot change future reports.
    first["batches"][0]["lhs"]["shape"] = (99,)
    assert plan.record()["batches"][0]["lhs"]["shape"] == (3, 2, 7)
    assert replace(plan).record() == plan.record()
    assert len(checked) == 2
    # A caller cannot rebuild an equal candidate while retaining stale SSA
    # witnesses from the old graph.
    with pytest.raises((ValueError, KeyError)):
        replace(plan, candidate=plan.candidate.clone())


def test_revalidation_rejects_source_changes_and_equal_ir_ssa_replacement():
    from xdsl.dialects import func

    plan = plan_independent_head_batching(parse_mlir_text(source()))
    plan.source.attributes["observer"] = builtin.StringAttr("new")
    with pytest.raises(ValueError, match="changed"):
        plan.revalidate()

    plan = plan_independent_head_batching(parse_mlir_text(source()))
    returned = next(op for op in plan.candidate.walk() if type(op) is func.ReturnOp)
    block = returned.parent
    block.insert_op_before(returned.clone(), returned)
    block.erase_op(returned)
    assert ir_digest(plan.candidate) == plan._candidate_digest
    with pytest.raises(ValueError, match="SSA or inverse witness changed"):
        plan.revalidate()
