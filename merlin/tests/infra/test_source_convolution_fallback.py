"""A partial panel-provider selection is discarded before any implementation callback."""

import gc
import json
import weakref
from dataclasses import replace
from math import prod

import pytest
from test_convolution_route_model import CASES, model_source
from test_source_dense_lowering import run, selection
from test_source_logical_abi import declaration

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower import convolution_routes, logical_borrowed_abi, source_dense
from merlin.llvmlower.borrowed_buffer_calls import BorrowedBufferEffects
from merlin.llvmlower.logical_borrowed_abi import LogicalBorrowedABI
from merlin.llvmlower.prepared_model_transform import apply_prepared_model_transform
from merlin.llvmlower.source_model import bind_source_lowerings, source_provider_builder
from merlin.transforms.certificates import ir_text


def configured(tmp_path, *, logical):
    geometry = CASES[0]
    n, channels, _, _ = geometry["shape"]
    k, columns = channels * prod(geometry["kernel"]), n * prod(geometry["output"])
    config, _, seen = selection(
        tmp_path,
        hardware=dict(
            tile=geometry["tile"],
            widths=(geometry["width"], geometry["width"], 4),
            shared=geometry["shared"],
            shape=(geometry["outputs"], k, columns),
            mutate=(lambda facts, software, dense: dense.update(logical_pointer_abi=declaration()))
            if logical
            else None,
        ),
        static_budget=1000,
        max_convolution_packing_bytes=geometry["budget"],
        borrowed_effects=BorrowedBufferEffects("neutral-reviewed-panel-reuse-v1", True, True, True),
    )
    if logical:
        config = replace(
            config,
            logical_abi=LogicalBorrowedABI(
                config.selected_contracts.dense.logical_pointer_abi,
                config.borrowed_effects,
                "neutral-reviewed-logical-format-v1",
                4096,
            ),
        )
    return config, seen, parse_mlir_text(model_source(geometry))


@pytest.mark.parametrize("logical", [False, True])
@pytest.mark.parametrize("unavailable", ["main", "tail", "all"])
@pytest.mark.parametrize("full_available", [False, True])
def test_partial_panel_coverage_replans_original_once_before_preparing_any_provider(
    tmp_path, monkeypatch, logical, unavailable, full_available
):
    config, seen, module = configured(tmp_path, logical=logical)
    original_discover = source_dense.discover_dense
    original_normalize = convolution_routes.normalize_convolution_source
    original_bridge = logical_borrowed_abi.prepare_binding
    discoveries, normalizations, bridges = [], [], []
    discarded_modules = []

    def discover(module, selection, validate):
        if not discoveries:
            discarded_modules.append(weakref.ref(module))
        else:
            gc.collect()
            assert discarded_modules[0]() is None, "the retry must release the discarded module and exception frame"
        regions, declined = original_discover(module, selection, validate)
        widths = tuple(region.compilation.operation.inputs[1].type.get_shape()[1] for region in regions)
        discoveries.append(widths)
        if len(discoveries) == 1:
            assert set(widths) == {1, 3}, "the private candidate must contain both real panel arms"
            refused = {"main": {3}, "tail": {1}, "all": {1, 3}}[unavailable]
            regions = tuple(
                region
                for region in regions
                if region.compilation.operation.inputs[1].type.get_shape()[1] not in refused
            )
        elif not full_available:
            regions = ()
        return regions, (*declined, "independent fixture provider availability")

    def normalize(*args):
        normalizations.append(True)
        return original_normalize(*args)

    def bridge(binding, *args, **kwargs):
        bridges.append(binding.compilation.operation.inputs[1].type.get_shape()[1])
        return original_bridge(binding, *args, **kwargs)

    monkeypatch.setattr(source_dense, "discover_dense", discover)
    monkeypatch.setattr(convolution_routes, "normalize_convolution_source", normalize)
    monkeypatch.setattr(logical_borrowed_abi, "prepare_binding", bridge)
    result, builder = run(tmp_path, config, module)
    assert discoveries == [(3, 1), (21,)]
    assert normalizations == [True], "repair must not repeat the rejected normalization"
    assert len(seen) == int(full_available)
    if full_available:
        assert source_provider_builder(builder) is builder
        assert tuple(seen[0].compilation.operation.inputs[1].type.get_shape()) == (12, 21)
    else:
        assert source_provider_builder(builder) is None
        assert result.read_bytes() == (tmp_path / "model.mlir").read_bytes()
    assert bridges == ([21] if logical and full_available else [])
    assert builder.__self__.packing_plans == ()
    builder.__self__.revalidate()
    receipt = json.loads((result.parent / "source_dense.json").read_text())
    (normalization,) = receipt["source_normalization"]
    assert normalization["rewritten"] == 0 and not normalization["routes"]
    assert normalization["discarded_candidate"]["rewritten"] == 1
    assert len(normalization["discarded_candidate"]["routes"][0]["selected_contraction_ordinals"]) == 2
    (outcome,) = normalization["outcomes"]
    assert outcome["selected_route"] == "full" and outcome["selected_packing_payload_bytes"] == 252
    assert outcome["budget_status"] == "exceeded" and outcome["reason"] == "no_completed_dense_provider"
    assert len(receipt["bindings"]) == int(full_available)


@pytest.mark.parametrize("receipt_mutation", ["unchanged", "optional_pointer", "remove_role"])
def test_provider_cannot_replace_both_live_packing_plan_and_comparison_stamp(tmp_path, receipt_mutation):
    config, _, module = configured(tmp_path, logical=False)
    owner, callbacks = [], []

    def mutate(bindings, directory):
        assert len(bindings) == 2 and len(owner[0].packing_plans) == 1
        callbacks.append(True)
        # These two in-memory fields agree after this mutation. The original
        # source receipt must still require both actual panel uses and payload.
        owner[0].packing_plans = owner[0]._packing_stamp = ()
        if receipt_mutation == "optional_pointer":
            owner[0].preparation_receipt = None
        if receipt_mutation == "remove_role":
            owner[0].pins = tuple((path, digest) for path, digest in owner[0].pins if path.name != "source_dense.json")

    config = replace(config, prepare=mutate)
    transform, builder = bind_source_lowerings((config,), None, None, None)
    owner.append(builder.__self__)
    source = tmp_path / "model.mlir"
    source.write_text(ir_text(module))
    with pytest.raises(ValueError, match="pinned source receipt"):
        apply_prepared_model_transform(source, tmp_path / "preparation", transform)
    assert callbacks == [True]
