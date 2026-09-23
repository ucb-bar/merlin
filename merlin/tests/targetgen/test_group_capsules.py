"""The layers a model forms become generator entries: deduplicated, signed, each with a raw sibling."""

from __future__ import annotations

from fake_quant_layer import Oracle as _Oracle
from im2col_conv_layer import module as _conv

from merlin.common import mlir_query as mq
from merlin.runtime.commandbuffer import SIGNED_STIMULUS_RANGE
from merlin.targetgen import group_capsules as GC


def _stated(**layer):
    return GC.entries("synthetic", mq.parse(_conv(**layer)), weight_args={1, 2}, model="toy", oracle=_Oracle())


def test_a_closed_group_is_one_entry_with_a_bare_accumulator_sibling() -> None:
    stated = _stated(channels=2, out_channels=4, image=4, taps=3, pad=1)
    assert (stated["accelerator_groups"], stated["stated"], stated["distinct"], stated["unstated"]) == (1, 1, 1, {})
    grouped, raw = stated["entries"]
    assert grouped["entry"]["op"] == "conv2d" and grouped["entry"]["epilogue"] == ["bias_add", "acc_scale", "relu"]
    # A fused stage behaves differently below zero, so the stimulus has to be able to go there.
    assert grouped["entry"]["stimulus_range"] == list(SIGNED_STIMULUS_RANGE)
    assert grouped["entry"]["source_role"] == GC.SOURCE_ROLE and "toy" in grouped["entry"]["source_reference"]
    # The sibling is the same contraction with nothing fused: same geometry, no stage, no multiplier.
    assert raw["raw_of"] == grouped["name"] and raw["entry"]["epilogue"] == []
    assert "acc_scale" not in raw["entry"] and raw["name"].endswith("_raw")
    same = ("op", "ci", "N", "Himg", "Wimg", "kh", "kw", "stride", "padding")
    assert all(raw["entry"][key] == grouped["entry"][key] for key in same)


def test_names_are_a_function_of_the_program_not_of_the_group_index() -> None:
    first = _stated(image=4, taps=3, pad=1)["entries"][0]["name"]
    assert first == _stated(image=4, taps=3, pad=1)["entries"][0]["name"]
    assert first != _stated(image=8, taps=3, pad=1)["entries"][0]["name"]


def test_a_group_that_cannot_be_restated_is_counted_with_its_reason() -> None:
    stated = _stated(image=4, taps=3, pad=1, bias_axis=2)
    assert stated["stated"] == 0 and stated["entries"] == []
    ((reason, count),) = stated["unstated"].items()
    assert count == 1 and "bias" in reason


def test_the_covering_subset_is_one_program_per_stage_combination_the_cheapest() -> None:
    def row(name, op, stages, raw_of=None, **extents):
        entry = {"op": op, "epilogue": list(stages), **extents}
        return {"name": name, "entry": entry, **({"raw_of": raw_of} if raw_of else {})}

    stated = {
        "entries": [
            row("big", "matmul", ["bias_add", "acc_scale", "relu"], M=3136, K=256, N=64),
            row("big_raw", "matmul", [], raw_of="big", M=3136, K=256, N=64),
            row("small", "matmul", ["bias_add", "acc_scale", "relu"], M=49, K=512, N=256),
            row("small_raw", "matmul", [], raw_of="small", M=49, K=512, N=256),
            row("bias_only", "matmul", ["bias_add"], M=196, K=256, N=1024),
        ]
    }
    names = [r["name"] for r in GC.covering_subset(stated)]
    # The cheaper of the two same-combination programs, WITH its bare sibling, and the other combination.
    assert names == ["small", "small_raw", "bias_only"]


def test_every_promoted_model_layer_capsule_is_schema_valid_and_ungraded() -> None:
    # Measured: the first promoted set carried a provenance role the capsule schema does not list,
    # and the grade refused every one of them before any tier ran.
    import json

    import jsonschema
    import yaml

    from merlin.common.paths import merlin_dir
    from merlin.targetgen.target_experiment import MODEL_LAYERS_CATEGORY

    schema = json.loads((merlin_dir() / "contract/schemas/capsule.schema.json").read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema)
    found = sorted((merlin_dir() / "contract/capsules").rglob(f"{MODEL_LAYERS_CATEGORY}/*/capsule.yaml"))
    for path in found:
        capsule = yaml.safe_load(path.read_text(encoding="utf-8"))
        errors = [error.message for error in validator.iter_errors(capsule)]
        assert not errors, f"{path.parent.name}: {errors[:2]}"
        assert capsule["source_role"] == GC.SOURCE_ROLE and capsule["label"] == "dev"
    # The category's underscore is what keeps it out of the graded suite.
    assert MODEL_LAYERS_CATEGORY.startswith("_")
