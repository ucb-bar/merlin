"""Whole-model routing must enumerate the contractions in the parsed graph, not only its tags."""

from __future__ import annotations

import pytest

from merlin.common.paths import merlin_dir
from merlin.targetgen.capsule_source import ModelDemandIncomplete, model_op_demands_checked

pytestmark = pytest.mark.target("gemmini")


def test_real_model_demands_match_the_structural_inventory_and_refuse_a_lost_tag():
    root = merlin_dir() / "contract/capsules/model"
    expected = {"SY_model_resnet50": 54, "SY_model_smolvla": 391}
    for name, count in expected.items():
        path = root / name / "capsule.interface.mlir"
        text = path.read_text(encoding="utf-8")
        demands = model_op_demands_checked(text, "int8")
        assert sum(d.family == "contraction" for d in demands) == count, name

        if name == "SY_model_resnet50":
            lines = text.splitlines(keepends=True)
            index = next(i for i, line in enumerate(lines) if "linalg.matmul" in line and 'prov.op = "' in line)
            before, _, after = lines[index].partition('prov.op = "')
            tag, _, tail = after.partition('"')
            lines[index] = before + f'prov.unattributed = "{tag}"' + tail
            with pytest.raises(ModelDemandIncomplete, match="untagged=.*linalg.matmul"):
                model_op_demands_checked("".join(lines), "int8")
