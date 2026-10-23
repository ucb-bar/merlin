"""Every numerical fix has a reviewed call obligation; infrastructure keeps pytest."""

import hashlib
import subprocess

import pytest
import yaml
from merlin_experiments.phase0.regression_capsules import load_specs

from merlin.common.paths import repo_root

FIXES = {
    "ba786b69",
    "6e6e37a60",
    "52eb603f",
    "a9106b60",
    "67fafa9d",
    "762e2ff8",
    "6f60202b",
    "88241b20",
    "28d7c59d",
    "e4b40d31",
    "c7c4ea63",
    "7d94a537",
    "df06091e",
    "abae247e",
    "f5e7c800",
}


def inventory():
    root = repo_root()
    index = yaml.safe_load((root / "experiments/reference-data/phase0/compiler-regression-fixes.yaml").read_bytes())
    calls = [row for spec in index["specs"] for row in yaml.safe_load((root / spec).read_bytes())["calls"]]
    return index, calls


def test_all_fix_commits_map_to_capsules_or_explicit_infrastructure_tests():
    index, calls = inventory()
    assert set(index["fixes"]) == FIXES
    for commit, entry in index["fixes"].items():
        if entry["classification"] == "CAPSULE-ABLE":
            mapped = [row for row in calls if commit in row["guards"]]
            assert mapped, f"{commit} lacks a capsule obligation"
            assert all(row["overload"] and row["boundary"] for row in mapped)
        else:
            assert entry["classification"] == "INFRA-ONLY"
            assert entry["reason"] and entry["tests"]
            assert all((repo_root() / path).is_file() for path in entry["tests"])
    external = index["external_audit"]["model2mlir"]
    assert external["branch"] == "core-aten-numeric"
    assert set(external["fixes"]) == {"3d2613d", "f8ed82443", "654987203"}
    for commit in external["fixes"]:
        assert any(commit in row["guards"] for row in calls)
    assert {commit for row in calls for commit in row["guards"]} <= FIXES | set(external["fixes"])
    identities = [row["case_id"] for row in calls]
    assert len(identities) == len(set(identities))


@pytest.mark.parametrize(
    "spec",
    [
        "compiler-regression-calls.yaml",
        "compiler-regression-half-precision.yaml",
        "compiler-regression-input-layout.yaml",
        "compiler-regression-numeric.yaml",
    ],
)
def test_generic_specs_are_inputs_without_answer_surfaces(spec):
    document = load_specs(repo_root() / "experiments/reference-data/phase0" / spec)
    for row in document["calls"]:
        assert "expected" not in row and "post_arguments" not in row
        assert set(row["arguments"]) == {"args", "kwargs"}
        assert row["origin"]


def test_retained_device_ids_and_layouts_bind_existing_capsules():
    _, calls = inventory()
    device = [row for row in calls if "capsule" in row]
    for row in device:
        assert row["capsule"] == "case_" + hashlib.sha256(row["case_id"].encode()).hexdigest()
        assert all(len(item["shape"]) == len(item["stride"]) for item in row["inputs"])
    transposed = next(row for row in device if "c7c4ea63" in row["guards"])
    assert transposed["inputs"][1]["stride"] == [1, 32]
    assert transposed["sealed_release"] == "core-aten-seal-20261008-v2"


@pytest.mark.parametrize("answer", ["expected", "post_arguments", "golden"])
def test_answer_members_cannot_enter_a_regression_spec(tmp_path, answer):
    document = load_specs(repo_root() / "experiments/reference-data/phase0/compiler-regression-calls.yaml")
    document["calls"][0][answer] = {"values": [42]}
    path = tmp_path / "input.yaml"
    path.write_text(yaml.safe_dump(document))
    with pytest.raises(ValueError, match="inputs"):
        load_specs(path)


@pytest.mark.parametrize("commit", sorted(FIXES))
def test_merlin_fix_is_landed_on_current_trunk(commit):
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=repo_root(),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Merlin fix {commit} is not an ancestor of HEAD: {result.stderr}"
