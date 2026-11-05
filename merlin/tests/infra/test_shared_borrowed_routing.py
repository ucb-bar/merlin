"""Original source owners transport reviewed ABI effects, never infer them."""

from dataclasses import replace

import pytest
from test_shared_pointwise_routing import prepare

from merlin.llvmlower.borrowed_buffer_calls import BorrowedBufferEffects
from merlin.llvmlower.shared_source_routing import SharedSourceRouting
from merlin.llvmlower.source_model import source_borrowed_calls


def test_legacy_source_owner_retains_exact_explicit_effects_and_source(tmp_path):
    route, provider, _source, selected = prepare(tmp_path)
    roster = source_borrowed_calls(route.build, selected)
    assert roster is not None and len(roster.calls) == 1
    assert roster.calls[0].effects is provider.borrowed_effects
    roster.revalidate(selected.read_text())
    assert roster.calls[0].result_argument == 2
    assert len(roster.calls[0].argument_types) == 3
    with pytest.raises(ValueError, match="stale for the current source"):
        roster.revalidate(selected.read_text() + "\n")


def test_numerically_qualified_source_does_not_supply_missing_abi_effects(tmp_path):
    route, provider, _source, selected = prepare(tmp_path, change="missing_borrowed_effects")
    assert provider.qualifications == 1
    assert source_borrowed_calls(route.build, selected) is None


def test_borrowed_source_owner_rejects_other_path_and_mutated_original_effects(tmp_path):
    route, _provider, _source, selected = prepare(tmp_path)
    copied = tmp_path / "copied.mlir"
    copied.write_bytes(selected.read_bytes())
    with pytest.raises(ValueError, match="original prepared source"):
        source_borrowed_calls(route.build, copied)
    state = route._prepared
    writer = replace(
        state.writers[0],
        borrowed_effects=BorrowedBufferEffects("another-original-review", True, True, True),
    )
    route._prepared = replace(state, writers=(writer,))
    with pytest.raises(ValueError, match="original writer ABI effects changed"):
        source_borrowed_calls(route.build, selected)


def test_method_name_and_foreign_owner_cannot_supply_borrowed_effects(tmp_path):
    class Foreign:
        build = SharedSourceRouting.build

    assert source_borrowed_calls(Foreign().build, tmp_path / "missing.mlir") is None
