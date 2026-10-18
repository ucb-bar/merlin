"""A multi-group rewrite shares shape observations without changing its statements."""

from merlin.common import mlir_query as mq
from merlin.llvmlower.device_offload import rewrite_groups_to_device
from merlin.targetgen import model_coverage
from merlin.xdsl_dialects.lowering import compute_groups as CG
from merlin.xdsl_dialects.lowering import group_command as GC


class Oracle:
    def ask(self, *, family, in_dtype, **kwargs):
        return (
            CG.Admission(True, units=("unit0",))
            if family == "contraction" and in_dtype == "int8"
            else CG.Admission(False, "undeclared", "not available")
        )

    def unit_for(self, *args):
        return "unit0"

    def absorbs(self, *args):
        return CG.Admission(False, "undeclared", "no readout")


def module():
    shapes = ((3, 7, 5), (11, 13, 17), (2, 19, 23))
    arguments = []
    body = []
    returns = []
    types = []
    for i, (m, k, n) in enumerate(shapes):
        arguments.extend((f"%a{i}: tensor<{m}x{k}xi8>", f"%b{i}: tensor<{k}x{n}xi8>"))
        output = f"tensor<{m}x{n}xi32>"
        body.append(f"""%zero{i} = arith.constant 0 : i32
          %empty{i} = tensor.empty() : {output}
          %init{i} = linalg.fill ins(%zero{i} : i32) outs(%empty{i} : {output}) -> {output}
          %result{i} = linalg.generic {{indexing_maps=[affine_map<(d0,d1,d2)->(d0,d2)>,affine_map<(d0,d1,d2)->(d2,d1)>,affine_map<(d0,d1,d2)->(d0,d1)>],iterator_types=["parallel","parallel","reduction"]}}
          ins(%a{i},%b{i}:tensor<{m}x{k}xi8>,tensor<{k}x{n}xi8>) outs(%init{i}:{output}) {{
          ^bb0(%lhs:i8,%rhs:i8,%acc:i32):
            %wide_a=arith.extsi %lhs:i8 to i32
            %wide_b=arith.extsi %rhs:i8 to i32
            %product=arith.muli %wide_a,%wide_b:i32
            %sum=arith.addi %product,%acc:i32
            linalg.yield %sum:i32
          }} -> {output}""")
        returns.append(f"%result{i}")
        types.append(output)
    source = (
        "module {func.func @forward("
        + ",".join(arguments)
        + ") -> ("
        + ",".join(types)
        + ") {"
        + "\n".join(body)
        + "\n return "
        + ",".join(returns)
        + ":"
        + ",".join(types)
        + "}}"
    )
    result = mq.parse(source)
    result.verify()
    return result


def test_source_statement_observation_is_shared_within_one_rewrite(monkeypatch):
    original = model_coverage._contraction_extents
    calls = []

    def observed(module):
        calls.append(module)
        return original(module)

    monkeypatch.setattr(model_coverage, "_contraction_extents", observed)
    source = module()
    result = rewrite_groups_to_device(source, "neutral", select=lambda _: True, oracle=Oracle(), weight_args={1, 3, 5})
    source.verify()
    assert result.moved == 3
    assert {(e["M"], e["K"], e["N"]) for e in result.entries.values()} == {(3, 7, 5), (11, 13, 17), (2, 19, 23)}
    assert len(calls) <= 2  # Formation and the rewrite each observe once, never per group.


def test_indexed_rewrite_is_identical_to_fresh_statement_observation(monkeypatch):
    indexed = module()
    result = rewrite_groups_to_device(indexed, "neutral", select=lambda _: True, oracle=Oracle(), weight_args={1, 3, 5})
    original = GC.program

    def uncached(group, **kwargs):
        kwargs["extents"] = None
        return original(group, **kwargs)

    monkeypatch.setattr(GC, "program", uncached)
    fresh = module()
    reference = rewrite_groups_to_device(
        fresh, "neutral", select=lambda _: True, oracle=Oracle(), weight_args={1, 3, 5}
    )
    assert result.to_sidecar() == reference.to_sidecar()
    assert str(indexed) == str(fresh)
