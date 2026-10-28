import pytest
import yaml

from merlin.targetgen.capture_refusals import M2MUnavailable
from merlin_experiments.phase0.generation import (
    _withhold_with_qualifying_groups,
    withhold_uncapturable_micro_model as withhold,
)

MICRO = {"name": "SY_micro_model", "kind": "model", "micro_model": True, "source_role": "derived_sweep"}
DETAIL = (
    "M2MUnavailable: m2m capture produced a non-clean program for op 'model'/int8 (ok=True, opaque=1); "
    "a capsule input must be 0-opaque. Opaque ops: aten_add_Tensor=1"
)


def _refusal():
    return M2MUnavailable("m2m capture produced a non-clean program")


def test_an_uncapturable_micro_model_is_withheld_with_its_verbatim_reason():
    record = withhold(MICRO, _refusal(), detail=DETAIL, must_refuse=False)
    assert record == {
        "capsule": "SY_micro_model",
        "placements": {},
        "reason": "generation-time capture refused by the selected model2MLIR: " + DETAIL,
    }


@pytest.mark.parametrize(
    "entry, exc, must_refuse",
    [
        (MICRO, RuntimeError("writer bug"), False),
        (MICRO, ValueError("not a capture refusal"), False),
        (MICRO, _refusal(), True),
        ({**MICRO, "micro_model": False}, _refusal(), False),
        ({**MICRO, "source_role": "materialized_iteration_capture"}, _refusal(), False),
        ({"name": "M1_resnet50", "kind": "model", "cat": "model", "model": "resnet50"}, _refusal(), False),
    ],
)
def test_every_other_writer_failure_stays_fatal(entry, exc, must_refuse):
    assert withhold(entry, exc, detail="x", must_refuse=must_refuse) is None


def test_a_withheld_uncapturable_micro_model_leaves_no_partial_directory_and_takes_its_groups(tmp_path):
    out_root = tmp_path / "out"
    model = out_root / "model" / "SY_micro_model"
    group = out_root / "layers" / "G_only_mine"
    group.mkdir(parents=True)
    (group / "capsule.yaml").write_text(yaml.safe_dump({"name": "G_only_mine"}))
    model.mkdir(parents=True)
    (model / "capsule.yaml").write_text(
        yaml.safe_dump({"name": "SY_micro_model", "model_qualification": {"group_capsules": [{"capsule": "layers/G_only_mine"}]}})
    )
    record = withhold(MICRO, _refusal(), detail=DETAIL, must_refuse=False)
    out = _withhold_with_qualifying_groups(record, model, out_root=out_root, written=[], missing_ok=True)
    assert not model.exists() and not group.exists()
    assert out["qualifying_groups_withheld"] == ["layers/G_only_mine"]


def test_a_micro_model_that_wrote_nothing_is_still_withheld(tmp_path):
    out_root = tmp_path / "out"
    out_root.mkdir()
    record = withhold(MICRO, _refusal(), detail=DETAIL, must_refuse=False)
    out = _withhold_with_qualifying_groups(
        record, out_root / "model" / "SY_micro_model", out_root=out_root, written=[], missing_ok=True
    )
    assert out["capsule"] == "SY_micro_model" and "qualifying_groups_withheld" not in out
