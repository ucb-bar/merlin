"""A target's declared quantization methodology, applied to a base recipe over a derived output path.

The methodology (``quantization.methodology`` in a software spec) states choices that are not hardware
facts -- calibration, accuracy recovery, the float island, reviewed per-model variants -- and says
``from_output_path`` for every hardware value. These tests derive recipes over SYNTHETIC output paths
(no rectifier, a scale per column, another clamp, no bias preload, another rounding) and check that each
recipe differs exactly where its path does, and that nothing is filled with a default.
"""

from __future__ import annotations

import copy

import pytest

from merlin.common.paths import repo_root
from merlin.targetgen import quant_methodology as QM
from merlin.targetgen import quant_recipe as QR

METHODOLOGY = {
    "schema": QM.SCHEMA,
    "scale_granularity": "from_output_path",
    "bias": "accumulator_preload_int32",
    "epilogue": "integer_domain",
    "rounding": "from_output_path",
    "clamp": "from_output_path",
    "calibration": {"activation": "mse", "weight": "minmax"},
    "recovery": {"cross_layer_equalization": True, "bias_correction": True},
    "float_island": {"dtype": "fp32", "ops": ["softmax", "rms_norm"]},
    "model_variants": [
        {
            "model": "decoder",
            "name": "int8-attention",
            "status": "reviewed",
            "reviewed_by": "test reviewer",
            "reason": "attention contractions in int8",
            "measured_accuracy": {"basis": "64 samples", "top1_agreement": 0.83},
            "changes": {
                "families": ["contraction", "activation_contraction"],
                "dynamic": {"schema": "dynamic_contractions_v1", "families": ["activation_contraction"]},
                "calibration_count": 256,
            },
        }
    ],
}


def _path(**changes) -> dict:
    readout = [
        {"kind": "activation", "function": "relu", "domain": "accumulator"},
        {"kind": "convert", "to": "f32", "rounding": "half_even"},
        {
            "kind": "scale",
            "dtype": "f32",
            "rounding": "half_even",
            "granularity": "per_command",
            "columns_per_scale": 16,
        },
        {"kind": "requant", "to": "i8", "rounding": "half_even", "clamp": [-128, 127]},
        {"kind": "pool", "reductions": ["max"], "padding": "unknown"},
    ]
    path = {
        "schema": "output_path_v1",
        "accumulator": {"dtype": "i32"},
        "accumulator_stages": [
            {"kind": "preload", "dtype": "i32", "range": [-(2**31), 2**31 - 1], "rounding": "none"},
            {"kind": "accumulate", "dtype": "i32", "operand_scale": "none"},
        ],
        "readout": readout,
        "exits": {"accumulator": True, "requant": True},
    }
    for kind, stage in changes.items():
        if stage is None:
            path["readout"] = [s for s in path["readout"] if s["kind"] != kind]
            if kind == "preload":
                path["accumulator_stages"] = [s for s in path["accumulator_stages"] if s["kind"] != "preload"]
        else:
            path["readout"] = [{**s, **stage} if s["kind"] == kind else s for s in path["readout"]]
    return path


def _base() -> dict:
    return QR.QuantRecipe(
        target="t",
        unit="mesh",
        weight=QR.TensorSpec("int8", "tensor", True, -127, 127),
        activation=QR.TensorSpec("int8", "tensor", True, -128, 127),
        families=(QR.CONTRACTION, QR.WINDOW_MEAN),
    ).to_dict() | {"accumulator_dtype": "int32"}


def _derive(path, **kw):
    return QM.derive(_base(), METHODOLOGY, path, **kw)


def test_the_reference_path_gives_column_group_scales_preloaded_biases_and_an_integer_epilogue():
    recipe = _derive(_path())
    assert recipe["status"] == QR.DERIVED and recipe["recipe_sha256"] == QR.digest(recipe)
    assert recipe["weight"]["granularity"] == "channel" and recipe["weight"]["channels_per_scale"] == 16
    assert (recipe["activation"]["quant_min"], recipe["activation"]["quant_max"]) == (-128, 127)
    assert recipe["activation"]["observer"] == "histogram" and recipe["weight"]["observer"] == "minmax"
    assert recipe["bias_domain"] == "accumulator"
    epilogue = recipe["epilogue"]
    assert epilogue["schema"] == "integer_epilogue_v1"
    assert epilogue["order"] == ["activation", "convert", "scale", "requant", "pool"]
    assert epilogue["activation"] == {"function": "relu", "domain": "accumulator"}
    assert epilogue["bias"]["mode"] == "accumulator_preload" and epilogue["bias"]["dtype"] == "int32"
    assert epilogue["requant"] == {"to": "int8", "rounding": "half_even", "clamp": [-128, 127]}
    assert epilogue["pool"]["reductions"] == ["max"]
    assert recipe["recovery"] == {"cross_layer_equalization": True, "bias_correction": True}
    assert recipe["float_island"] == {"dtype": "fp32", "ops": ["softmax", "rms_norm"]}
    assert recipe["methodology"]["output_path_sha256"] == QM.digest(_path())


def test_a_path_without_a_rectifier_leaves_relu_out_of_the_pattern():
    reference, plain = _derive(_path()), _derive(_path(activation=None))
    assert plain["status"] == QR.DERIVED
    assert plain["epilogue"]["activation"] is None and reference["epilogue"]["activation"] is not None
    assert plain["epilogue"]["order"] == ["convert", "scale", "requant", "pool"]
    assert plain["recipe_sha256"] != reference["recipe_sha256"]


def test_a_scale_per_column_gives_ungrouped_channel_scales():
    recipe = _derive(_path(scale={"granularity": "per_channel", "columns_per_scale": None}))
    assert recipe["weight"]["granularity"] == "channel" and "channels_per_scale" not in recipe["weight"]
    assert recipe["epilogue"]["scale"]["granularity"] == "per_channel"


def test_another_clamp_is_every_activation_grid_and_the_requant_range():
    recipe = _derive(_path(requant={"clamp": [-127, 127]}))
    assert (recipe["activation"]["quant_min"], recipe["activation"]["quant_max"]) == (-127, 127)
    assert recipe["epilogue"]["requant"]["clamp"] == [-127, 127]


def test_no_bias_preload_keeps_biases_on_the_host_and_says_why():
    recipe = _derive(_path(preload=None))
    assert recipe["status"] == QR.DERIVED
    assert recipe["bias_domain"] is None
    assert recipe["epilogue"]["bias"]["mode"] == "host_float"
    assert "preloads no" in recipe["epilogue"]["bias"]["why"]


def test_unexpressible_or_missing_facts_are_underivable_never_defaulted():
    rounding = _derive(_path(requant={"rounding": "half_up"}))
    assert rounding["status"] == QR.UNDERIVABLE and "rounding" in rounding["underivable"]
    columns = _derive(_path(scale={"columns_per_scale": None}))
    assert columns["status"] == QR.UNDERIVABLE and "scale.columns_per_scale" in columns["underivable"]
    absent = _derive(None)
    assert absent["status"] == QR.UNDERIVABLE and "output_path" in absent["underivable"]
    narrower = _derive(_path(requant={"to": "i4"}))
    assert narrower["status"] == QR.UNDERIVABLE and "requant.to" in narrower["underivable"]


def test_calibration_methods_name_their_observers():
    for method, observer in (({"activation": "percentile", "percentile": 99.99, "weight": "minmax"}, "p99.99"),
                             ({"activation": "minmax", "weight": "minmax"}, "minmax")):  # fmt: skip
        recipe = QM.derive(_base(), {**METHODOLOGY, "calibration": method}, _path())
        assert recipe["activation"]["observer"] == observer


def test_unproved_accumulator_and_preload_domains_are_not_realizable():
    path = _path()
    path["accumulator"]["dtype"] = "i64"
    assert "accumulator.dtype" in _derive(path)["underivable"]
    path = _path()
    path["accumulator_stages"][0].pop("range")
    assert "bias.range" in _derive(path)["underivable"]
    path = _path()
    path["readout"][0]["domain"] = "scaled"
    assert "output_path.order" in _derive(path)["underivable"]


@pytest.mark.parametrize("change", [{"model_variants": {}}, {"float_island": {"dtype": "fp32", "ops": [{}]}}])
def test_malformed_containers_raise_validation_errors(change):
    with pytest.raises(ValueError):
        QM.validate({**METHODOLOGY, **change})


def test_a_model_changes_only_through_its_reviewed_variant_row():
    shared, variant = _derive(_path()), _derive(_path(), model="decoder")
    other = _derive(_path(), model="classifier")
    assert (
        "dynamic" not in shared
        and other["recipe_sha256"] == QM.derive(_base(), METHODOLOGY, _path(), model="classifier")["recipe_sha256"]
    )
    assert variant["dynamic"]["families"] == ["activation_contraction"]
    assert variant["families"] == ["contraction", "activation_contraction"]
    assert variant["framework_capture_policy"]["calibration_count"] == 256
    assert variant["methodology"]["variants"][0]["name"] == "int8-attention"
    # A base that carries a variant block of its own does not pass it on.
    carried = copy.deepcopy(_base()) | {"dynamic": {"schema": "dynamic_contractions_v1"}}
    assert "dynamic" not in QM.derive(carried, METHODOLOGY, _path())


@pytest.mark.parametrize(
    "change, message",
    [
        ({"float_island": {"dtype": "bf16", "ops": ["softmax"]}}, "never bf16"),
        ({"float_island": {"dtype": "fp32", "ops": ["attention"]}}, "float_island.ops"),
        ({"bias": "float_on_the_accumulator"}, "bias is one of"),
        ({"rounding": "half_even"}, "rounding is one of"),
        ({"calibration": {"activation": "percentile", "weight": "minmax"}}, "states its percentile"),
        ({"recovery": {"bias_correction": True}}, "recovery states each"),
        ({"unknown": 1}, "unknown fields"),
    ],
)
def test_the_declaration_is_validated(change, message):
    with pytest.raises(ValueError, match=message):
        QM.validate({**METHODOLOGY, **change})


@pytest.mark.parametrize(
    "row_change, message",
    [
        ({"status": "unreviewed"}, "only as a reviewed row"),
        ({"reason": " "}, "reason is stated"),
        ({"measured_accuracy": {}}, "measured_accuracy is empty"),
        ({"measured_accuracy": None}, "measured_accuracy holds numbers"),
        ({"changes": {"weight": {}}}, "changes are among"),
    ],
)
def test_a_variant_needs_a_review_a_reason_and_a_measurement(row_change, message):
    row = {**METHODOLOGY["model_variants"][0], **row_change}
    with pytest.raises(ValueError, match=message):
        QM.validate({**METHODOLOGY, "model_variants": [row]})


def test_the_example_target_declares_derived_values_only():
    from merlin.targetgen.software_spec import load_software_spec

    spec = load_software_spec(repo_root() / "examples/gemmini/target/software-spec.yaml", target="gemmini")
    method = QM.validate(spec["quantization"]["methodology"])
    for key in ("scale_granularity", "rounding", "clamp"):
        assert method[key] == QM.FROM_OUTPUT_PATH
    assert method["float_island"]["dtype"] == "fp32"
    assert all(row["status"] == "reviewed" and row["measured_accuracy"] for row in method["model_variants"])
