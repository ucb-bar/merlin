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


def test_explicit_per_channel_policy_preserves_the_physical_readout_width():
    recipe = QM.derive(_base(), {**METHODOLOGY, "scale_granularity": "per_channel"}, _path())
    assert recipe["status"] == QR.DERIVED
    assert recipe["weight"]["granularity"] == "channel" and "channels_per_scale" not in recipe["weight"]
    assert recipe["epilogue"]["scale"] == {
        "dtype": "fp32",
        "rounding": "half_even",
        "granularity": "per_command",
        "columns_per_scale": 16,
        "logical_granularity": "per_channel",
    }


def test_reviewed_channel_scale_variant_changes_only_its_selected_model():
    changed = copy.deepcopy(METHODOLOGY)
    changed["model_variants"].append(
        {
            "model": "classifier",
            "name": "independent-column-scales",
            "status": "reviewed",
            "reviewed_by": "test reviewer",
            "reason": "synthetic source policy selection",
            "measured_accuracy": {"basis": "synthetic test; no production admission"},
            "changes": {"scale_granularity": "per_channel"},
        }
    )
    for model in (None, "decoder", "other"):
        assert QM.derive(_base(), changed, _path(), model=model) == _derive(_path(), model=model)
    selected = QM.derive(_base(), changed, _path(), model="classifier")
    assert "channels_per_scale" not in selected["weight"]
    assert selected["epilogue"]["scale"]["logical_granularity"] == "per_channel"
    assert selected["methodology"]["variants"][0]["changes"] == ["scale_granularity"]


def _plane_variant(**changes):
    return {
        "model": "classifier",
        "name": "static-activation-planes",
        "status": "reviewed",
        "reviewed_by": "test reviewer",
        "reason": "synthetic precision and host-bias policy selection",
        "measured_accuracy": {"basis": "synthetic test; no production admission"},
        "changes": {"static_planes": QR.static_planes_block(), "bias": "host_float", **changes},
    }


def test_reviewed_planes_and_host_bias_change_only_the_selected_model():
    method = copy.deepcopy(METHODOLOGY)
    method["model_variants"].append(_plane_variant(scale_granularity="per_channel"))
    for model in (None, "decoder", "other"):
        assert QM.derive(_base(), method, _path(), model=model) == _derive(_path(), model=model)
    recipe = QM.derive(_base(), method, _path(), model="classifier")
    assert recipe["status"] == QR.DERIVED and recipe["recipe_sha256"] == QR.digest(recipe)
    assert recipe[QR.STATIC_PLANES] == QR.static_planes_block()
    assert recipe["bias_domain"] is None and recipe["epilogue"]["bias"]["mode"] == "host_float"
    assert recipe["epilogue"]["scale"]["columns_per_scale"] == 16
    assert recipe["epilogue"]["scale"]["logical_granularity"] == "per_channel"
    assert "channels_per_scale" not in recipe["weight"]
    assert recipe["methodology"]["variants"][0]["changes"] == ["bias", "scale_granularity", "static_planes"]


def test_planes_cannot_claim_accumulator_bias_or_survive_a_later_family_change():
    row = _plane_variant(bias="accumulator_preload_int32")
    method = {**METHODOLOGY, "model_variants": [row]}
    recipe = QM.derive(_base(), method, _path(), model="classifier")
    assert recipe["status"] == QR.UNDERIVABLE
    assert "host_float bias" in recipe["underivable"][QR.STATIC_PLANES]
    assert recipe[QR.STATIC_PLANES] == row["changes"][QR.STATIC_PLANES]
    row = _plane_variant()
    drop = {**row, "name": "drop-contractions", "changes": {"families": [QR.WINDOW_MEAN]}}
    for rows in ([row, drop], [drop, row]):
        recipe = QM.derive(_base(), {**METHODOLOGY, "model_variants": rows}, _path(), model="classifier")
        assert recipe["status"] == QR.UNDERIVABLE
        assert "quantizes contractions" in recipe["underivable"][QR.STATIC_PLANES]


def test_planes_check_the_final_bias_selection_across_multiple_rows():
    row = _plane_variant()
    override = {**row, "name": "bias-placement", "changes": {"bias": "accumulator_preload_int32"}}
    method = {**METHODOLOGY, "model_variants": [row, override]}
    assert QM.derive(_base(), method, _path(), model="classifier")["status"] == QR.UNDERIVABLE
    method["model_variants"].reverse()
    assert QM.derive(_base(), method, _path(), model="classifier")["status"] == QR.DERIVED


def test_unreviewed_carried_planes_are_dropped_and_unsupported_selected_planes_are_retained_as_refusal():
    base = QR.with_static_planes(_base())
    assert QR.STATIC_PLANES not in QM.derive(base, METHODOLOGY, _path())
    base["activation"]["dtype"] = "fp32"
    method = {**METHODOLOGY, "model_variants": [_plane_variant()]}
    recipe = QM.derive(base, method, _path(), model="classifier")
    assert recipe["status"] == QR.UNDERIVABLE and QR.STATIC_PLANES in recipe["underivable"]
    assert recipe[QR.STATIC_PLANES] == QR.static_planes_block()


@pytest.mark.parametrize(
    "change",
    [
        {"schema": "unknown"},
        {"families": [QR.ACTIVATION_CONTRACTION]},
        {"planes": True},
        {"planes": 1},
        {"planes": 4},
        {"observer": "mse"},
        {"observer": "p0"},
        {"owners": []},
        {"owners": [""]},
        {"owners": [1]},
        {"owners": "front"},
        {"extra": True},
    ],
)
def test_static_plane_variant_validates_its_closed_policy(change):
    row = _plane_variant(static_planes=QR.static_planes_block() | change)
    with pytest.raises(ValueError, match="changes.static_planes"):
        QM.validate({**METHODOLOGY, "model_variants": [row]})


def test_static_plane_variant_requires_all_fields_and_a_review():
    for block in (None, {}, {k: v for k, v in QR.static_planes_block().items() if k != "owners"}):
        with pytest.raises(ValueError, match="changes.static_planes"):
            QM.validate({**METHODOLOGY, "model_variants": [_plane_variant(static_planes=block)]})
    with pytest.raises(ValueError, match="only as a reviewed row"):
        QM.validate({**METHODOLOGY, "model_variants": [_plane_variant() | {"status": "unreviewed"}]})
    with pytest.raises(ValueError, match="changes.bias"):
        QM.validate({**METHODOLOGY, "model_variants": [_plane_variant(bias="unknown")]})


def test_reviewed_plane_observer_digits_and_owner_scope_reach_the_derived_recipe():
    block = QR.static_planes_block(planes=3, observer="p99.9", owners=["front", "back", "front"])
    method = {**METHODOLOGY, "model_variants": [_plane_variant(static_planes=block)]}
    recipe = QM.derive(_base(), method, _path(), model="classifier")
    assert recipe["status"] == QR.DERIVED and recipe[QR.STATIC_PLANES] == block


@pytest.mark.parametrize(
    "scale",
    [
        {"granularity": "per_tensor"},
        {"columns_per_scale": None},
        {"columns_per_scale": False},
    ],
)
def test_per_channel_policy_does_not_invent_configurability_or_physical_width(scale):
    recipe = QM.derive(_base(), {**METHODOLOGY, "scale_granularity": "per_channel"}, _path(scale=scale))
    assert recipe["status"] == QR.UNDERIVABLE


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


def test_reviewed_recovery_changes_only_the_selected_model_and_keeps_unselected_recipe_digests():
    disabled = {name: False for name in QM.RECOVERY}
    original = {**METHODOLOGY, "recovery": disabled}
    row = {
        **_plane_variant(),
        "name": "bias-recovery",
        "changes": {
            "recovery": {**disabled, "bias_correction": True},
            "scale_granularity": "per_channel",
        },
    }
    method = {**original, "model_variants": [*original["model_variants"], row]}
    for model in (None, "decoder", "other"):
        assert QM.derive(_base(), method, _path(), model=model) == QM.derive(_base(), original, _path(), model=model)
    recipe = QM.derive(_base(), method, _path(), model="classifier")
    assert recipe["status"] == QR.DERIVED
    assert recipe["recovery"] == {"cross_layer_equalization": False, "bias_correction": True}
    assert recipe["bias_domain"] == "accumulator" and recipe["epilogue"]["bias"]["mode"] == "accumulator_preload"
    assert "channels_per_scale" not in recipe["weight"]
    assert recipe["epilogue"]["scale"]["columns_per_scale"] == 16
    assert recipe["methodology"]["variants"][0]["changes"] == ["recovery", "scale_granularity"]
    assert original["recovery"] == disabled and method["recovery"] == disabled


def test_multiple_recovery_variants_apply_in_declared_order():
    enabled = {name: True for name in QM.RECOVERY}
    disabled = {name: False for name in QM.RECOVERY}
    first = {**_plane_variant(), "name": "enable-recovery", "changes": {"recovery": enabled}}
    last = {**first, "name": "disable-recovery", "changes": {"recovery": disabled}}
    for rows, expected in [([first, last], disabled), ([last, first], enabled)]:
        recipe = QM.derive(_base(), {**METHODOLOGY, "model_variants": rows}, _path(), model="classifier")
        assert recipe["status"] == QR.DERIVED and recipe["recovery"] == expected
        assert recipe["recipe_sha256"] == QR.digest(recipe)


@pytest.mark.parametrize(
    "block",
    [
        None,
        [],
        {},
        {"bias_correction": True},
        {"cross_layer_equalization": False, "bias_correction": 1},
        {"cross_layer_equalization": "false", "bias_correction": True},
        {"cross_layer_equalization": False, "bias_correction": True, "unknown": False},
    ],
)
def test_recovery_variant_requires_the_complete_closed_boolean_policy(block):
    row = {**_plane_variant(), "changes": {"recovery": block}}
    with pytest.raises(ValueError, match="changes.recovery"):
        QM.validate({**METHODOLOGY, "model_variants": [row]})
