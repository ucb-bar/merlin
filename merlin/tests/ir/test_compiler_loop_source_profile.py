"""Original source coverage classification never runs a producer or a proof."""

import pytest
from test_compiler_loop_context import source as sibling_source
from test_compiler_loop_generalization import source as word_source
from test_compiler_loop_selected_depth import source as depth_source
from test_compiler_nested_unroll_modes import nested_source
from test_compiler_unroll_modes import effect_source

from merlin.verify import compiler_loop_source_profile as profile
from merlin.verify.compiler_profile import SourceOutsideProfile
from merlin.verify.gate_errors import GateUnavailable


@pytest.fixture(autouse=True)
def no_source_probe_producer(monkeypatch):
    monkeypatch.setattr(
        "merlin.transforms.unrolling.unroll_ordered_loops", lambda *a, **k: pytest.fail("source probe unrolled")
    )
    monkeypatch.setattr("merlin.transforms.loops.segment_loops", lambda *a, **k: pytest.fail("source probe tiled"))


@pytest.mark.parametrize("trips,carry", [(0, 0), (0, 1), (1, 1), (2, 1), (5, 0), (5, 1), (6, 1)])
def test_word_source_modes_remain_in_supported_model(trips, carry):
    assert profile.admit_source(effect_source(trips, carries=carry), kind=profile.UNROLL) == profile.WORD_PROFILE


@pytest.mark.parametrize("depth,kind", [(2, profile.UNROLL), (3, profile.UNROLL), (2, profile.TILE), (3, profile.TILE)])
def test_actual_selected_depth_with_dynamic_ancestor_upper_is_admitted(depth, kind):
    assert profile.admit_source(depth_source(depth), kind=kind, depth=depth) == profile.WORD_PROFILE


@pytest.mark.parametrize(
    "source", [sibling_source(), nested_source(5), "builtin.module {" + word_source(4, 3, trips=8) + "}"]
)
def test_word_sibling_nested_and_factor7_carried_source(source):
    assert profile.admit_source(source, kind=profile.UNROLL, factor=7) == profile.WORD_PROFILE


TENSOR_SOURCE = """builtin.module {
func.func private @prefix(tensor<4xf32>)->tensor<4xf32>
func.func private @tick(index,tensor<4xf32>)
func.func @forward(%left:tensor<4xf32>,%right:tensor<4xf32>)->tensor<4xf32> {
%r=func.call @prefix(%left):(tensor<4xf32>)->tensor<4xf32>
%l=arith.constant 0:index
%u=arith.constant 5:index
%s=arith.constant 1:index
scf.for %i=%l to %u step %s {func.call @tick(%i,%right):(index,tensor<4xf32>)->() scf.yield}
func.return %r:tensor<4xf32>
}}
"""


@pytest.mark.parametrize("kind", [profile.UNROLL, profile.TILE])
def test_bounded_tensor_context_uses_generic_model_without_integer_payload_claim(kind):
    assert profile.admit_source(TENSOR_SOURCE, kind=kind) == profile.GENERIC_PROFILE


@pytest.mark.parametrize(
    "source,settings",
    [
        (TENSOR_SOURCE, {"kind": profile.TILE, "depth": 2}),
        (TENSOR_SOURCE, {"kind": profile.UNROLL, "factor": 4}),
        (effect_source(5), {"kind": profile.UNROLL, "depth": 10**12}),
        (effect_source(5), {"kind": profile.UNROLL, "factor": True}),
        (effect_source(5), {"kind": profile.UNROLL, "index_bits": 32}),
        (effect_source(5).replace("13 : index", str(2**63 - 1) + " : index"), {"kind": profile.UNROLL}),
    ],
)
def test_known_original_model_exclusions_are_explicit(source, settings):
    with pytest.raises(SourceOutsideProfile):
        profile.admit_source(source, **settings)


@pytest.mark.parametrize(
    "source",
    [
        effect_source(5).replace("%lb = arith.constant 3 : index", "%lb = arith.constant -3 : index"),
        depth_source(2).replace("%lb=arith.constant 0:index", '%lb="test.lower"():()->index'),
    ],
)
def test_negative_or_unestablished_ancestor_lower_is_outside_source_model(source):
    with pytest.raises(SourceOutsideProfile, match="nonnegative"):
        profile.admit_source(source, kind=profile.UNROLL, depth=2 if "test.lower" in source else 1)


def test_parser_tool_or_resource_failures_cannot_become_source_exclusions(monkeypatch):
    def unavailable(*args, **kwargs):
        raise GateUnavailable("actual parser resource unavailable")

    monkeypatch.setattr(profile, "_parse_bounded", unavailable)
    with pytest.raises(GateUnavailable, match="actual parser resource"):
        profile.admit_source(effect_source(5), kind=profile.UNROLL)
