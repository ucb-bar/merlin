from types import SimpleNamespace

import pytest
from xdsl.dialects import func

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.perf.model_group_work import summarize_group_work
from merlin.xdsl_dialects._common import text

SOURCE = """module {func.func @forward(%a:tensor<2x3xf32>,%b:tensor<3x4xf32>,
  %c:tensor<2x4xf32>) -> tensor<2x4xf32> {
  %r=linalg.matmul ins(%a,%b:tensor<2x3xf32>,tensor<3x4xf32>)
    outs(%c:tensor<2x4xf32>) -> tensor<2x4xf32>
  %x=linalg.generic {indexing_maps=[affine_map<(m,n)->(m,n)>,
    affine_map<(m,n)->(m,n)>,affine_map<(m,n)->(m,n)>],
    iterator_types=["parallel","parallel"]}
    ins(%r,%c:tensor<2x4xf32>,tensor<2x4xf32>) outs(%c:tensor<2x4xf32>) {
      ^bb0(%u:f32,%v:f32,%w:f32):
        %sum=arith.addf %u,%v:f32
        linalg.yield %sum:f32
    } -> tensor<2x4xf32>
  return %x:tensor<2x4xf32>
}}"""


def fixture(source=SOURCE):
    module = parse_mlir_text(source)
    owner = next(op for op in module.body.block.ops if isinstance(op, func.FuncOp) and op.body.blocks)
    operations = tuple(owner.body.block.ops)[:-1]
    groups = tuple(SimpleNamespace(index=i, placement="descriptive", members=[op])
                   for i, op in enumerate(operations))
    return module, groups


def test_all_source_groups_include_non_mac_host_work_without_runtime_claims():
    module, groups = fixture()
    before = text(module, generic=True)
    report = summarize_group_work(module, groups)
    assert report["source_macs"] == 24
    assert report["groups"][0]["source_macs"] == 24
    assert report["groups"][1]["source_macs"] == 0
    assert report["groups"][0]["known_scalar_body_instances"]["arith.mulf"] == 24
    assert report["groups"][1]["known_scalar_body_instances"] == {"arith.addf": 8}
    assert all(row["elapsed_cycles"] is None and row["requested_transfer_bytes"] is None
               for row in report["groups"])
    assert not report["timing_ranking_available"]
    assert not report["placement_or_effect_admission"]
    assert text(module, generic=True) == before


def test_missing_group_retains_unassigned_mac_work_instead_of_zero():
    module, groups = fixture()
    report = summarize_group_work(module, groups[1:])
    assert report["source_macs"] is None
    assert report["known_source_macs_subtotal"] == 0
    assert report["unassigned_mac_operations"][0]["macs"] == 24


def test_opaque_call_is_unknown_not_a_free_group():
    module, groups = fixture("""module {func.func private @external()
      func.func @forward() {func.call @external():()->() return}}""")
    report = summarize_group_work(module, groups)
    assert report["source_macs"] is None
    assert report["groups"][0]["source_macs"] is None
    assert "multiplicity" in report["groups"][0]["unknown_mac_reasons"][0]


def test_large_parallel_source_domain_is_counted_without_expansion():
    module, groups = fixture(SOURCE.replace("2x3", "2000000x3").replace("2x4", "2000000x4"))
    report = summarize_group_work(module, groups)
    assert report["source_macs"] == 24000000
    assert report["groups"][1]["known_scalar_body_instances"]["arith.addf"] == 8000000


@pytest.mark.parametrize("mutation", ["duplicate", "foreign", "index"])
def test_ambiguous_group_custody_refuses(mutation):
    module, groups = fixture()
    if mutation == "duplicate":
        groups = (*groups, SimpleNamespace(index=2, placement="descriptive", members=groups[0].members))
    elif mutation == "foreign":
        groups = (*groups, SimpleNamespace(index=2, placement="descriptive", members=fixture()[1][0].members))
    else:
        groups[1].index = groups[0].index
    with pytest.raises(ValueError, match="current groups"):
        summarize_group_work(module, groups)
