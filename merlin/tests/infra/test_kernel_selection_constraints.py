"""Semantic shape constraints work without importing target support packages."""

import pytest

from merlin.targetgen import kernel_family_selection


@pytest.fixture
def registry():
    return kernel_family_selection


def contract(constraint):
    return {
        "qualified_families": ["strategy"],
        "experimental_families": [],
        "compiler_selection": {
            "rules": [{
                "family": "strategy",
                "semantic_ops": ["operation"],
                "dtypes": ["fp32"],
                "shape": {"D": constraint},
            }],
        },
    }


def select(registry, constraint, shape):
    return registry.select_kernel_family(
        {"op": "operation", "dtype": "fp32", "shape": shape},
        {"features": []},
        contract(constraint),
    )


@pytest.mark.parametrize(
    ("constraint", "shape", "selected"),
    [
        ({"one_of": [32, 128]}, {"D": 32}, True),
        ({"one_of": [32, 128]}, {"D": 128}, True),
        ({"one_of": [32, 128]}, {"D": 64}, False),
        ({"power_of_two": True}, {"D": 1}, True),
        ({"power_of_two": True}, {"D": 128}, True),
        ({"power_of_two": True}, {"D": 96}, False),
        ({"multiple_of_dimension": "heads"}, {"D": 12, "heads": 3}, True),
        ({"multiple_of_dimension": "heads"}, {"D": 12, "heads": 5}, False),
        ({"multiple_of_dimension": "heads"}, {"D": 12}, False),
        ({"one_of": [32]}, {}, False),
        ({"power_of_two": True}, {}, False),
        ({"multiple_of_dimension": "heads"}, {"heads": 3}, False),
        ({"one_of": [32, 128], "power_of_two": True, "multiple_of_dimension": "heads",
          "min": 64, "max": 256, "multiple_of": 16}, {"D": 128, "heads": 8}, True),
        ({"one_of": [32, 128], "min": 64}, {"D": 32}, False),
        (32, {"D": 32}, True),
        ({"eq": 32, "min": 16, "max": 64, "multiple_of": 8}, {"D": 32}, True),
    ],
)
def test_shape_constraint_selection(registry, constraint, shape, selected):
    report = select(registry, constraint, shape)
    assert report.selected_family == ("strategy" if selected else None)
    assert report.decisions[0].status == ("selected" if selected else "refused")
    assert report.decisions[0].reasons


@pytest.mark.parametrize(
    "constraint",
    [
        {"one_of": []}, {"one_of": [0]}, {"one_of": [-1]}, {"one_of": [True]},
        {"one_of": [32, "64"]}, {"one_of": [32.0]}, {"one_of": (32,)},
        {"one_of": "32"}, {"one_of": None},
        {"power_of_two": False}, {"power_of_two": 1}, {"power_of_two": "true"},
        {"power_of_two": None},
        {"multiple_of_dimension": ""}, {"multiple_of_dimension": " "},
        {"multiple_of_dimension": 2}, {"multiple_of_dimension": True},
        {"multiple_of_dimension": []}, {"multiple_of_dimension": None},
    ],
)
@pytest.mark.parametrize("shape", [{}, {"D": 32}])
def test_malformed_constraints_fail_closed(registry, constraint, shape):
    with pytest.raises(registry.KernelSelectionContractError):
        select(registry, constraint, shape)
    with pytest.raises(registry.KernelSelectionContractError):
        registry.validate_selection_contract(contract(constraint))


def test_missing_reference_is_explained(registry):
    report = select(registry, {"multiple_of_dimension": "heads"}, {"D": 32})
    assert report.selected_family is None
    assert "heads" in report.decisions[0].reasons[0]


def test_invalid_constraint_in_disabled_family_is_rejected(registry):
    payload = contract({"power_of_two": False})
    payload["qualified_families"] = []
    payload["experimental_families"] = [{"path": "strategy", "reason": "unqualified"}]
    with pytest.raises(registry.KernelSelectionContractError):
        registry.validate_selection_contract(payload)


def test_nonpositive_reference_in_direct_request_is_refused(registry):
    request = registry.KernelRequest("operation", "fp32", {"D": 32, "heads": 0})
    report = registry.select_kernel_family(
        request, {"features": []}, contract({"multiple_of_dimension": "heads"})
    )
    assert report.selected_family is None
    assert "heads" in report.decisions[0].reasons[0]


@pytest.mark.parametrize(
    ("shape", "selected"),
    [
        ({"D": 1024, "other": 2048}, True),
        ({"D": 1025, "other": 2048}, False),
        ({"D": 1024, "other": 2049}, False),
        ({"D": 1, "other": 32}, True),
        ({"D": 1, "other": 65536}, True),
        ({"D": 1, "other": 65537}, False),
        ({"D": 1024}, False),
        ({"other": 2048}, False),
    ],
)
def test_tile_product_limit(registry, shape, selected):
    report = select(
        registry, {"tile_product_max": {"dimension": "other", "tile": 32, "max": 2048}}, shape
    )
    assert report.selected_family == ("strategy" if selected else None)
    assert report.decisions[0].reasons


@pytest.mark.parametrize(
    "product",
    [
        None, [], True, {},
        {"dimension": "other", "tile": 32},
        {"dimension": "other", "tile": 32, "max": 2048, "extra": 1},
        *({"dimension": value, "tile": 32, "max": 2048} for value in ["", " ", None, 1, True]),
        *({"dimension": "other", "tile": value, "max": 2048} for value in [0, -1, True, "32", 32.0]),
        *({"dimension": "other", "tile": 32, "max": value} for value in [0, -1, True, "2048", 2048.0]),
    ],
)
@pytest.mark.parametrize("shape", [{}, {"D": 32, "other": 32}])
def test_malformed_tile_product_fails_closed(registry, product, shape):
    constraint = {"tile_product_max": product}
    with pytest.raises(registry.KernelSelectionContractError):
        select(registry, constraint, shape)
    with pytest.raises(registry.KernelSelectionContractError):
        registry.validate_selection_contract(contract(constraint))


def test_tile_product_refusal_explains_bound(registry):
    report = select(
        registry, {"tile_product_max": {"dimension": "other", "tile": 32, "max": 2048}},
        {"D": 1025, "other": 2048},
    )
    assert report.selected_family is None
    assert "2112" in report.decisions[0].reasons[0]
    assert "2048" in report.decisions[0].reasons[0]


@pytest.mark.parametrize("key", ["eq", "min", "max", "multiple_of"])
@pytest.mark.parametrize("value", [True, "32", 32.0, None, -1])
@pytest.mark.parametrize("shape", [{}, {"D": 32}])
def test_legacy_numeric_predicates_validate_without_request_dimensions(registry, key, value, shape):
    with pytest.raises(registry.KernelSelectionContractError):
        registry.validate_selection_contract(contract({key: value}))
    with pytest.raises(registry.KernelSelectionContractError):
        select(registry, {key: value}, shape)


def test_zero_divisor_is_invalid_before_a_request(registry):
    with pytest.raises(registry.KernelSelectionContractError):
        registry.validate_selection_contract(contract({"multiple_of": 0}))


@pytest.mark.parametrize(
    ("op", "dtype", "shape"),
    [
        ("matmul", "mxfp8", {"M": 256, "N": 64, "K": 2048}),
        ("matmul", "mxfp4", {"M": 256, "N": 64, "K": 2048}),
        ("bias_add", "fp32", {"rows": 32, "cols": 768}),
        ("batched_matmul", "mxfp8", {"M": 48, "N": 128, "K": 2048}),
    ],
)
def test_shared_matcher_preserves_legacy_catalog_decisions(registry, op, dtype, shape):
    import importlib.util
    import sys

    from merlin.common.paths import repo_root

    path = repo_root() / "examples/muon/support/backend/muon_kernel_selection.py"
    name = "frozen_kernel_selection_compatibility"
    spec = importlib.util.spec_from_file_location(name, path)
    legacy = importlib.util.module_from_spec(spec)
    sys.modules[name] = legacy
    try:
        spec.loader.exec_module(legacy)
    finally:
        del sys.modules[name]
    selection = registry.load_selection_contract(
        repo_root() / "examples/radiance/phase1/kernel_library/selection.yaml"
    )
    request = {"op": op, "dtype": dtype, "shape": shape}
    hardware = {"features": ["mx_mesh", "block_e8m0", "shared_memory", "simt", "exp", "tanh"]}
    assert registry.validate_selection_contract(selection) == legacy.validate_selection_contract(selection)
    assert registry.select_kernel_family(request, hardware, selection).to_dict() == legacy.select_kernel_family(
        request, hardware, selection
    ).to_dict()
