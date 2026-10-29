"""UNKNOWN predictions cannot inflate the ordinary calibrated-domain gate.

These fixed synthetic rows exercise report custody and arithmetic only; no
runtime/physical/held qualification is issued and no provider is executed.
"""

import json

import pytest
from merlin_experiments.phase2.component_analytical import _qualified_domains
from merlin_experiments.phase2.contracts import StageGateError, sha256_file

from merlin.common.jsonio import canonical_sha256 as sha
from merlin.perf.component_screen import (
    ComponentScreenPolicy,
    qualify_component_screen,
    validate_component_screen_report,
)
from merlin.perf.fast_estimate_validation import Observation
from merlin.xdsl_dialects.lowering.global_plan import CycleInterval


class PartialPrediction:
    def predict(self, features, *, domain_sha256):
        count = features["/count"]
        if count == 10:
            return CycleInterval.unknown("source diagnostic unknown feature")
        return CycleInterval.point(count * 10 + (0.1 if count == 1 else 0), "synthetic reader control")


def report():
    rows = [
        Observation(
            sha([group, count]),
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
    return qualify_component_screen(rows, lambda _train: PartialPrediction(), calibration_sha256=sha("calibration"))


def test_ordinary_domain_reader_keeps_unknown_predictions_unqualified(tmp_path):
    selected = report()
    assert validate_component_screen_report(selected) == selected
    assert selected["exposable"] is False and selected["ranking"]["exposable"] is True
    assert selected["interval_coverage"] == {"n": 27, "contains": 24, "rate": 24 / 27}
    assert selected["policy"] == ComponentScreenPolicy().to_dict()
    path = tmp_path / "unknown-held.json"
    path.write_text(json.dumps(selected))
    with pytest.raises(StageGateError, match="does not qualify"):
        _qualified_domains(path, sha256_file(path), sha("calibration"))


@pytest.mark.parametrize("claim", [True, 1], ids=["boolean", "numeric"])
def test_ordinary_domain_reader_refuses_resigned_unknown_coverage_inflation(tmp_path, claim):
    selected = report()
    for row in selected["predictions"]:
        if not row["prediction"]["resolved"]:
            row["contains_observation"] = claim
    selected["interval_coverage"] = {"n": 27, "contains": 27, "rate": 1}
    selected["exposable"] = True
    selected["reasons"] = []
    assert len(selected["predictions"]) == 30
    assert selected["ranking"]["overall"]["decided"] == 108
    assert all(row["decided"] == 36 for row in selected["ranking"]["slices"].values())
    assert selected["absolute_error"]["maximum_relative"] < 0.10
    path = tmp_path / "resigned-unknown-held.json"
    path.write_text(json.dumps(selected))
    with pytest.raises(ValueError, match="unresolved held prediction cannot claim error or coverage"):
        _qualified_domains(path, sha256_file(path), sha("calibration"))
