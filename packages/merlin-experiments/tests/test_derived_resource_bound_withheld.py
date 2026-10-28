"""An empty capstone set is legitimate only as a reviewed, disclosed withhold, never as a silent default.

``derive_resource_bound`` used to raise whenever the staged corpus held no admitted whole-model capsule.
A rehearsal (and any run whose source-capsule withhold table withholds every model) hits that. Lifting
the stop must not turn "no capstones" into "all capstones passed": the opt-in is explicit, each withheld
capstone must be disclosed by generation with a reason, and the cohort record carries the gap.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from merlin_experiments.corpus import preparation
from merlin_experiments.spec import SpecError

from merlin.targetgen import capsule_runner
from merlin.targetgen.capsule_grade import capstones_withheld_disclosed
from merlin.targetgen.target_experiment import load_target_experiment

ROWS = [{"capsule": "SY_micro_model", "reason": "emitted program not admitted"}]


def _corpus(tmp_path, *, models=(), withheld=ROWS, extra_ops=("OP_a",)):
    corpus = tmp_path / "corpus"
    for name in extra_ops:
        d = corpus / "isa" / name
        d.mkdir(parents=True)
        (d / "capsule.yaml").write_text(yaml.safe_dump({"name": name, "kind": "operation", "label": "public"}))
    for name, qualified in models:
        d = corpus / "model" / name
        d.mkdir(parents=True)
        cap = {"name": name, "kind": "model", "label": "public"}
        if qualified:
            cap["model_qualification"] = {"status": "qualified"}
        (d / "capsule.yaml").write_text(yaml.safe_dump(cap))
    manifest = {
        "generated": [f"model/{n}" for n, _ in models],
        "placement_generation": {"gemmini": {"host_placement_not_published": withheld}},
    }
    (corpus / "MANIFEST.yaml").write_text(yaml.safe_dump(manifest))
    return corpus


@pytest.fixture(autouse=True)
def _discover(monkeypatch):
    def discover(root, labels=None):
        out = []
        for path in sorted(Path(root).rglob("capsule.yaml")):
            cap = yaml.safe_load(path.read_text())
            cap["__dir__"] = str(path.parent)
            out.append(cap)
        return out

    monkeypatch.setattr(capsule_runner, "discover_capsules", discover)


def test_no_opt_in_still_raises_even_when_generation_disclosed_the_withhold(tmp_path):
    with pytest.raises(SpecError, match="no admitted whole-model capstone"):
        preparation.derive_resource_bound(_corpus(tmp_path))


def test_opt_in_with_full_disclosure_records_the_gap_and_an_empty_required_list(tmp_path):
    bound = preparation.derive_resource_bound(_corpus(tmp_path), ["SY_micro_model"])
    assert bound["required_admitted_models"] == []
    assert bound["capstones_withheld"] == ROWS


@pytest.mark.parametrize(
    "withheld, declared, match",
    [
        ([], ["SY_micro_model"], "not disclosed"),  # nothing disclosed
        (ROWS, ["SY_micro_model", "SY_other_model"], "SY_other_model"),  # an undisclosed missing model
        ([{"capsule": "SY_micro_model", "reason": "  "}], ["SY_micro_model"], "not disclosed"),  # empty reason
        (ROWS, ["SY_micro_model", "SY_micro_model"], "exactly once"),
    ],
)
def test_opt_in_with_an_undisclosed_or_unexplained_missing_model_raises(tmp_path, withheld, declared, match):
    with pytest.raises(SpecError, match=match):
        preparation.derive_resource_bound(_corpus(tmp_path, withheld=withheld), declared)


def test_opt_in_cannot_name_a_capsule_that_is_actually_in_the_corpus(tmp_path):
    corpus = _corpus(tmp_path, models=[("SY_micro_model", True)])
    with pytest.raises(SpecError, match="present in the corpus"):
        preparation.derive_resource_bound(corpus, ["SY_micro_model"])


def test_a_corpus_with_an_admitted_model_is_unchanged(tmp_path):
    corpus = _corpus(tmp_path, models=[("SY_qualified", True)], withheld=[])
    bound = preparation.derive_resource_bound(corpus)
    assert bound == {
        "derive": preparation.DERIVED_RESOURCE_BOUND,
        "required_admitted_models": ["SY_qualified"],
        "exclude_capsules": [],
    }


# ---------------------------------------------------------------- consumers


def _descriptor(tmp_path, resource_bound):
    path = tmp_path / "target_experiment.yaml"
    path.write_text(yaml.safe_dump({"target": "probe_target", "grading": {"resource_bound": resource_bound}}))
    return path


def test_descriptor_carries_withheld_capstones_and_refuses_resource_exclusions_without_either(tmp_path):
    te = load_target_experiment(_descriptor(tmp_path, {"policy": "p", "capstones_withheld": ROWS}))
    assert te.graded_required_models == () and te.graded_capstones_withheld == (tuple(ROWS[0].values()),)
    load_target_experiment(_descriptor(tmp_path, {"policy": "p", "exclude_capsules": ["M"], "capstones_withheld": ROWS}))
    with pytest.raises(ValueError, match="admitted model capstones"):
        load_target_experiment(_descriptor(tmp_path, {"policy": "p", "exclude_capsules": ["M"]}))
    with pytest.raises(ValueError, match="capstones_withheld"):
        load_target_experiment(_descriptor(tmp_path, {"policy": "p", "capstones_withheld": [{"capsule": "X"}]}))
    with pytest.raises(ValueError, match="also required or excluded"):
        load_target_experiment(
            _descriptor(tmp_path, {"policy": "p", "required_admitted_models": ["SY_micro_model"], "capstones_withheld": ROWS})
        )


def test_empty_required_models_is_accepted_downstream_only_with_a_disclosed_withhold():
    assert capstones_withheld_disclosed({"required_admitted_models": [], "capstones_withheld": ROWS})
    assert not capstones_withheld_disclosed({"required_admitted_models": []})
    assert not capstones_withheld_disclosed({"capstones_withheld": []})
    assert not capstones_withheld_disclosed({"capstones_withheld": [{"capsule": "X", "reason": ""}]})


def test_no_shipped_descriptor_enables_the_withheld_capstone_opt_in():
    """The opt-in is a per-run, reviewed descriptor choice (PD-1), never a shipped default."""
    from merlin.common.paths import repo_root

    hits = []
    for top in ("examples", "experiments"):
        for path in (repo_root() / top).rglob("*"):
            if path.suffix in {".yaml", ".yml", ".json"} and path.is_file():
                if "allow_withheld_capstones" in path.read_text(encoding="utf-8", errors="ignore"):
                    hits.append(str(path))
    assert hits == []
