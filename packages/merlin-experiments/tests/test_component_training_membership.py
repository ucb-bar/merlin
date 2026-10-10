"""Complete original training memberships survive the ordinary report reader.

Synthetic observations exercise source/custody arithmetic only, without executing
a provider or issuing physical, runtime or independent held qualification.
"""

import json

import pytest
from merlin_experiments.phase2.component_analytical import _qualified_domains
from merlin_experiments.phase2.contracts import sha256_file

from merlin.common.jsonio import canonical_sha256 as sha
from merlin.perf.component_screen import ComponentScreenPolicy, qualify_component_screen
from merlin.perf.fast_estimate_validation import Observation
from merlin.xdsl_dialects.lowering.global_plan import CycleInterval


class SyntheticPrediction:
    def predict(self, features, *, domain_sha256):
        center = features["/count"] * 10
        return CycleInterval(center - 0.1, center + 0.1, provenance=("synthetic reader control only",))


def report():
    # One exact executable can legitimately occur on several original workloads.
    rows = [
        Observation(
            sha(count),
            sha(group),
            group,
            sha("domain"),
            {"/count": count},
            count * 10,
            (sha([group, count, "synthetic"]),),
        )
        for group in ("a", "b", "c")
        for count in range(1, 11)
    ]

    def fit(training):
        assert len(training) == len({row.id for row in training}) == 20
        assert len({row.program for row in training}) == 10
        return SyntheticPrediction()

    return qualify_component_screen(rows, fit, calibration_sha256=sha("calibration"))


def test_ordinary_reader_preserves_repeated_training_memberships_and_order(tmp_path):
    selected = report()
    for row in selected["predictions"]:
        assert len(row["training_programs"]) == 20 and len(set(row["training_programs"])) == 10
        row["training_programs"].reverse()
    assert len(selected["predictions"]) == 30
    assert selected["ranking"]["overall"]["decided"] == 135
    assert all(row["decided"] == 45 for row in selected["ranking"]["slices"].values())
    assert selected["interval_coverage"] == {"n": 30, "contains": 30, "rate": 1}
    assert selected["policy"] == ComponentScreenPolicy().to_dict()
    path = tmp_path / "complete-training.json"
    path.write_text(json.dumps(selected))
    assert _qualified_domains(path, sha256_file(path), sha("calibration")) == (sha("domain"),)


@pytest.mark.parametrize("change", ["missing", "extra", "mapping"])
def test_ordinary_reader_refuses_resigned_incomplete_or_excess_training_membership(tmp_path, change):
    selected = report()
    row = selected["predictions"][0]
    training = row["training_programs"]
    row["training_programs"] = (
        list(dict.fromkeys(training))
        if change == "missing"
        else [*training, training[0]]
        if change == "extra"
        else dict.fromkeys(training, 2)
    )
    assert len(selected["predictions"]) == 30 and selected["exposable"]
    assert selected["ranking"]["overall"]["decided"] == 135
    assert all(row["decided"] == 45 for row in selected["ranking"]["slices"].values())
    path = tmp_path / "resigned-training.json"
    path.write_text(json.dumps(selected))
    with pytest.raises(ValueError, match="component prediction training roster"):
        _qualified_domains(path, sha256_file(path), sha("calibration"))
