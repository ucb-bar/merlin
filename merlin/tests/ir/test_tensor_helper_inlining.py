"""Reject unsupported pass boundaries while retaining ordinary buffer ownership."""
import pytest

from merlin.llvmlower.tensor_helper_inlining import edit_pipeline


@pytest.mark.parametrize("passes", [
    [],
    ["one-shot-bufferize"],
    ["buffer-results-to-out-params", "one-shot-bufferize"],
    ["one-shot-bufferize", "one-shot-bufferize", "buffer-results-to-out-params"],
])
def test_missing_or_ambiguous_bufferization_boundary_refuses(passes):
    original = list(passes)
    with pytest.raises(ValueError, match="ordered bufferization"):
        edit_pipeline(passes)
    assert passes == original


def test_feature_resolves_through_existing_lazy_module_registry():
    from merlin.llvmlower.impr_features import get
    from merlin.llvmlower.tensor_helper_inlining import FEATURE

    assert get(FEATURE).edit_pipeline is edit_pipeline
