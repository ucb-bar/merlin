"""Every compute group a real model forms on a target must be demanded by some capsule."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from merlin.common.paths import merlin_dir
from merlin.targetgen import conformance as CF
from merlin.targetgen import corpus_synth as CS

_REQUIRED = [
    {
        "signature": "matmul|bias_add+relu|i8",
        "op": "matmul",
        "epilogue": ["bias_add", "relu"],
        "dtype": "i8",
        "groups": 32,
        "observed_in": ["m"],
        "scale_granularities": ["tensor"],
    },
    {
        "signature": "matmul|bias_add|i8",
        "op": "matmul",
        "epilogue": ["bias_add"],
        "dtype": "i8",
        "groups": 20,
        "observed_in": ["m"],
        "scale_granularities": ["tensor"],
    },
]


def _capsule(root: Path, name: str, op: str, epilogue: list[str], *, label: str = "public") -> None:
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "capsule.yaml").write_text(
        yaml.safe_dump(
            {
                "name": name,
                "label": label,
                "inputs": [{"name": "A0", "dtype": "i8"}],
                "operation": {"op": op, "attributes": {"epilogue": epilogue}},
            }
        ),
        encoding="utf-8",
    )


def test_a_stage_combination_no_capsule_demands_is_uncovered(tmp_path: Path) -> None:
    _capsule(tmp_path, "bias_only", "fused_matmul_bias", ["bias_add"])
    _capsule(tmp_path, "relu_only", "matmul", ["relu"])
    gap = CF._group_gap(_REQUIRED, tmp_path)
    # Each stage is demanded somewhere; the COMBINATION is demanded nowhere.
    assert gap["uncovered"] == ["matmul|bias_add+relu|i8"]
    assert (gap["uncovered_groups"], gap["required_groups"]) == (32, 52)
    assert gap["demanded_by"] == {"matmul|bias_add|i8": ["bias_only"]}

    _capsule(tmp_path, "both", "linear", ["bias_add", "relu"])
    assert CF._group_gap(_REQUIRED, tmp_path)["uncovered"] == []
    # A capsule the public label does not include demands nothing.
    _capsule(tmp_path / "hidden_root", "hidden", "matmul", ["bias_add", "relu"], label="hidden")
    assert CF._group_gap(_REQUIRED[:1], tmp_path / "hidden_root")["uncovered"] == ["matmul|bias_add+relu|i8"]


def test_a_spec_without_the_axis_is_not_measured_rather_than_clean(tmp_path: Path) -> None:
    report = CF.uncovered({"cells": []}, tmp_path)
    assert report["groups"]["status"] == "not_measured"
    measured = CF.uncovered({"cells": [], "groups": {"required": _REQUIRED}}, tmp_path)
    assert measured["groups"]["n_covered"] == 0 and len(measured["groups"]["uncovered"]) == 2


def test_each_uncovered_combination_is_synthesized_as_one_member() -> None:
    root = merlin_dir() / "contract/capsules/conformance"
    spec_path = next(iter(sorted(root.glob("*.yaml"))), None)
    assert spec_path is not None
    doc = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    dtype = next(str(c["dtype"]) for c in doc["cells"] if c.get("family") == "contraction")
    doc["groups"] = {"required": [{**_REQUIRED[0], "dtype": dtype}, {**_REQUIRED[1], "dtype": dtype}]}
    result = CS.synthesize(doc)
    members = [e for e in result["capsules"] if (e.get("generalization") or {}).get("generalization_axis") == "groups"]
    # The two-stage combination gets a member; the single stage is the epilogue axis's already.
    assert [m["epilogue"] for m in members] == [["bias_add", "relu"]]
    assert members[0]["op"] == "matmul" and members[0]["operand_dtype"] == dtype
    assert "32 compute group(s)" in members[0]["source_reference"]

    # A combination with a stage no builder can state stops the synthesis; it is never dropped.
    doc["groups"]["required"].append(
        {
            "signature": "matmul|bias_add+softcap|x",
            "op": "matmul",
            "epilogue": ["bias_add", "softcap"],
            "dtype": dtype,
            "groups": 3,
            "observed_in": ["m"],
        }
    )
    with pytest.raises(CS.SynthesisError, match="softcap"):
        CS.synthesize(doc)
