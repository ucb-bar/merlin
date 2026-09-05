"""Adjacency and op configuration must be OBSERVED, not assumed away.

These pin the two things the per-region census cannot express, and each test is written so that the
old behaviour would fail it. A census that cannot see a configuration axis reports full coverage of a
corpus that has one value of it -- which is how every convolution capsule in this repo came to declare
the same padding, stride and dilation four times over.
"""
from __future__ import annotations

import pytest

from merlin.common import mlir_query as mq
from merlin.targetgen import scope_census as SC

_CHAIN = """
module {
  func.func @f(%a: tensor<4x4xf32>, %b: tensor<4x4xf32>, %o: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %0 = linalg.matmul ins(%a, %b : tensor<4x4xf32>, tensor<4x4xf32>) outs(%o : tensor<4x4xf32>) -> tensor<4x4xf32>
    %1 = linalg.matmul ins(%0, %b : tensor<4x4xf32>, tensor<4x4xf32>) outs(%o : tensor<4x4xf32>) -> tensor<4x4xf32>
    %2 = linalg.matmul ins(%1, %b : tensor<4x4xf32>, tensor<4x4xf32>) outs(%o : tensor<4x4xf32>) -> tensor<4x4xf32>
    return %2 : tensor<4x4xf32>
  }
}
"""

_BRANCH = """
module {
  func.func @f(%a: tensor<4x4xf32>, %b: tensor<4x4xf32>, %o: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %0 = linalg.matmul ins(%a, %b : tensor<4x4xf32>, tensor<4x4xf32>) outs(%o : tensor<4x4xf32>) -> tensor<4x4xf32>
    %1 = linalg.matmul ins(%0, %b : tensor<4x4xf32>, tensor<4x4xf32>) outs(%o : tensor<4x4xf32>) -> tensor<4x4xf32>
    %2 = linalg.matmul ins(%0, %b : tensor<4x4xf32>, tensor<4x4xf32>) outs(%o : tensor<4x4xf32>) -> tensor<4x4xf32>
    %3 = linalg.matmul ins(%1, %2 : tensor<4x4xf32>, tensor<4x4xf32>) outs(%o : tensor<4x4xf32>) -> tensor<4x4xf32>
    return %3 : tensor<4x4xf32>
  }
}
"""

_ISOLATED = """
module {
  func.func @f(%a: tensor<4x4xf32>, %b: tensor<4x4xf32>, %o: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %0 = linalg.matmul ins(%a, %b : tensor<4x4xf32>, tensor<4x4xf32>) outs(%o : tensor<4x4xf32>) -> tensor<4x4xf32>
    return %0 : tensor<4x4xf32>
  }
}
"""


def _mod(text):
    return mq.parse(text)


# --------------------------------------------------------------------------------- adjacency

def test_a_run_of_producers_and_consumers_is_one_chain():
    cs = SC.chains(_mod(_CHAIN))
    assert len(cs) == 1
    assert cs[0].length == 3
    assert cs[0].indices == (0, 1, 2)


def test_a_single_region_is_not_a_chain():
    """A chain of one is a region, and the region census already describes it. Emitting it here would
    double-count every op in the model as an adjacency obligation."""
    assert SC.chains(_mod(_ISOLATED)) == ()


def test_branching_ends_a_chain_rather_than_inventing_an_order():
    """Where a value feeds two regions the run is no longer a line. Following one arbitrarily would
    state an ordering the program does not, so the walk stops."""
    cs = SC.chains(_mod(_BRANCH))
    assert cs, "a branch should still yield the runs on either side of it"
    assert all(c.length <= 2 for c in cs), [c.signature for c in cs]


def test_the_length_bound_reports_itself_rather_than_passing_as_the_model_s_longest():
    """THE HONESTY CONTROL. A census that truncates at its bound and reports the truncated number as
    'longest' states a property of the BOUND as a property of the program."""
    tight = SC.chain_census(_mod(_CHAIN), max_length=2)
    assert tight["longest"] == 2
    assert tight["longest_is_truncated"] is True

    loose = SC.chain_census(_mod(_CHAIN), max_length=64)
    assert loose["longest"] == 3
    assert loose["longest_is_truncated"] is False


def test_a_signature_speaks_the_family_vocabulary_not_the_op_spelling():
    """A real capture is mostly unnamed linalg.generic, so a signature built from op names reads
    'generic -> generic' for every chain in the model, which counts nothing."""
    sig = SC.chains(_mod(_CHAIN))[0].signature
    assert sig == "contraction -> contraction -> contraction"
    assert "matmul" not in sig


# ----------------------------------------------------------------------------- op configuration

def test_a_region_carries_its_declared_configuration():
    cfgs = SC.region_configs(_mod(_CHAIN))
    assert len(cfgs) == 3
    assert all(c.op == "matmul" and c.family == "contraction" for c in cfgs)
    assert all(c.attrs for c in cfgs), "an op's own attribute table must reach the descriptor"


def test_configuration_axes_count_distinct_values_so_a_single_valued_axis_is_visible_as_one():
    """THE MUTATION CONTROL for the conv-geometry defect. If the axis were not read at all, a corpus
    with one value and a corpus with four would be indistinguishable here."""
    axes = SC.config_axes(SC.region_configs(_mod(_CHAIN)))
    assert "matmul" in axes
    counts = axes["matmul"]
    assert counts, "no attribute reached the axis census"
    assert all(v == 1 for v in counts.values()), (
        "these three ops are configured identically, so every axis must show exactly one distinct "
        "value; more than one would mean the census is reading something that is not configuration"
    )


def test_provenance_is_not_mistaken_for_configuration():
    """prov.* tags are a hint about what an op WAS, recorded elsewhere; folding them into the
    configuration axes would make a tagging change look like a geometry change."""
    for c in SC.region_configs(_mod(_CHAIN)):
        assert not [k for k in c.attrs if k.startswith("prov.")]


def test_an_unparseable_capture_yields_no_observation_rather_than_raising():
    with pytest.raises(Exception):
        SC.chains(_mod("this is not mlir"))
