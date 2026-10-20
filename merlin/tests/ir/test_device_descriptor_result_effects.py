"""Original effects-owner custody; no synthetic premise qualifies device code."""

from dataclasses import replace
from pathlib import Path

import pytest
import test_device_descriptor_result_build as fixtures
from test_device_build_parallel import synthetic_build  # noqa: F401

from merlin.llvmlower import descriptor_result_identity as R
from merlin.llvmlower import device_build as B
from merlin.llvmlower.borrowed_buffer_calls import BorrowedBufferCalls
from merlin.llvmlower.compilation_recipe import CompilationRecipe
from merlin.runtime.device_descriptor_results import DeviceDescriptorResults

issued = fixtures.issued


def reissue(issued, tmp_path, factory):
    route, original, _, built, source = issued
    guard = DeviceDescriptorResults(replace(route, descriptor_effects=factory))
    directory = tmp_path / "qualified"
    directory.mkdir()
    recipe = CompilationRecipe(directory, producer=Path(__file__))
    guard.bind_sources(recipe)
    guard.bind_input(source, original._inputs[1].path, recipe)
    return guard, recipe, built


def test_checked_descriptor_alone_cannot_supply_missing_kernel_laws(tmp_path):
    route = B.DeviceRouting("neutral", tmp_path, "i8", "i32", descriptor_results=True)
    with pytest.raises(ValueError, match="qualified borrowed-buffer effects owner"):
        DeviceDescriptorResults(route)
    # The existing ordinary route remains available without the new opt-in.
    assert DeviceDescriptorResults(replace(route, descriptor_results=False)).options() == {}


def test_effects_factory_requires_original_source_bound_code(tmp_path):
    class NotSourceBound:
        def __call__(self, *_):
            pytest.fail("must refuse before calling")

    route = B.DeviceRouting(
        "neutral", tmp_path, "i8", "i32", descriptor_results=True, descriptor_effects=NotSourceBound()
    )
    with pytest.raises(ValueError, match="source-bound Python owner"):
        DeviceDescriptorResults(route)


@pytest.mark.parametrize(
    "mutation",
    [
        "not_owner",
        "source_path",
        "source_digest",
        "input_digest",
        "missing_object",
        "missing_shim",
        "call_roster",
        "argument_types",
        "result_argument",
        "no_retention",
        "no_release",
        "completion",
    ],
)
def test_original_owner_must_join_actual_source_objects_abi_and_all_effects(issued, tmp_path, mutation):
    def factory(source, built):
        owner = fixtures.neutral_effects_owner(source, built)
        if mutation == "not_owner":
            return owner.record()
        if mutation in ("source_path", "source_digest", "input_digest"):
            return replace(
                owner,
                **{
                    "source_path": {"source_path": source.with_name("other.mlir")},
                    "source_digest": {"source_sha256": "0" * 64},
                    "input_digest": {"input_sha256": "0" * 64},
                }[mutation],
            )
        if mutation.startswith("missing_"):
            omitted = built.shim_object.with_suffix(".c") if mutation == "missing_shim" else built.objects[-1]
            return replace(owner, pins=tuple(p for p in owner.pins if p[0] != omitted.resolve()))
        call = owner.calls[0]
        if mutation == "call_roster":
            return replace(owner, calls=(replace(call, borrowed_symbol="unrelated_borrow"),))
        if mutation == "argument_types":
            return replace(owner, calls=(replace(call, argument_types=("memref<1xi8>",) * 3),))
        if mutation == "result_argument":
            return replace(owner, calls=(replace(call, result_argument=0),))
        attribute = {
            "no_retention": "no_retained_aliases",
            "no_release": "no_releases",
            "completion": "complete_before_return",
        }[mutation]
        object.__setattr__(call.effects, attribute, False)
        return owner

    guard, recipe, built = reissue(issued, tmp_path, factory)
    with pytest.raises(ValueError, match="owner|borrowed ABI"):
        guard.bind_shim(built, recipe)


@pytest.mark.parametrize(
    "mutation", ["identity", "effects", "validator", "implementation", "effect_validation", "record"]
)
def test_later_effects_owner_changes_cannot_refresh_original_selection(issued, monkeypatch, mutation):
    _, guard, recipe, built, _ = issued
    guard.bind_shim(built, recipe)
    owner = guard._effects
    if mutation == "identity":
        object.__setattr__(owner.calls[0].effects, "qualification_identity", "different-qualification")
    elif mutation == "effects":
        object.__setattr__(owner.calls[0].effects, "complete_before_return", False)
    elif mutation == "validator":
        object.__setattr__(owner, "validate_original", lambda: None)
    elif mutation == "effect_validation":
        monkeypatch.setattr(R, "validate_effects", lambda *_: None)
    elif mutation == "record":
        (built.shim_object.parent / "descriptor-result-effects.json").write_text("changed record")
    else:
        monkeypatch.setattr(BorrowedBufferCalls, "revalidate", lambda *_: None)
    with pytest.raises(ValueError, match="changed"):
        guard.revalidate()


@pytest.mark.parametrize("mutation", ["object", "route"])
def test_factory_cannot_change_original_build_or_route_during_qualification(issued, tmp_path, mutation):
    original_route = issued[0]

    def factory(source, built):
        owner = fixtures.neutral_effects_owner(source, built)
        if mutation == "object":
            built.shim_object.write_bytes(b"changed in effects factory")
        else:
            original_route.numeric_policy["reduction"] = "changed in effects factory"
        return owner

    guard, recipe, built = reissue(issued, tmp_path, factory)
    with pytest.raises(ValueError, match="changed|identity"):
        guard.bind_shim(built, recipe)


@pytest.mark.parametrize(
    "attribute,value", [("qualification_identity", "changed inside validator"), ("complete_before_return", False)]
)
def test_original_validator_cannot_change_effects_and_refresh_its_own_record(issued, tmp_path, attribute, value):
    def factory(source, built):
        owner = fixtures.neutral_effects_owner(source, built)

        def validator():
            object.__setattr__(owner.calls[0].effects, attribute, value)

        return replace(owner, validate_original=validator)

    guard, recipe, built = reissue(issued, tmp_path, factory)
    with pytest.raises(ValueError, match="changed during qualification validation"):
        guard.bind_shim(built, recipe)
