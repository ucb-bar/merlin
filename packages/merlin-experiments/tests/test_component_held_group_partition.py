"""Ordinary held-domain reader refuses split workload groups, even after re-signing.

Synthetic arithmetic exercises this reader only; no physical/runtime owner is
issued and the provider evaluation path is never launched.
"""

import json

import pytest
from merlin_experiments.phase2.component_analytical import _qualified_domains
from merlin_experiments.phase2.contracts import sha256_file

from merlin.common.jsonio import canonical_sha256 as sha
from merlin.perf import rank_validation as rank
from merlin.perf.component_screen import ComponentScreenPolicy, qualify_component_screen
from merlin.perf.fast_estimate_validation import Observation
from merlin.xdsl_dialects.lowering.global_plan import CycleInterval


class SyntheticPrediction:
    def predict(self, features, *, domain_sha256):
        center = features["/count"] * 10
        return CycleInterval(center - 0.1, center + 0.1, provenance=("synthetic reader control only",))


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
    return qualify_component_screen(rows, lambda _train: SyntheticPrediction(), calibration_sha256=sha("calibration"))


def test_ordinary_domain_reader_preserves_original_held_groups_and_minima(tmp_path):
    selected = report()
    path = tmp_path / "held.json"
    path.write_text(json.dumps(selected))
    assert _qualified_domains(path, sha256_file(path), sha("calibration")) == (sha("domain"),)
    assert selected["policy"] == ComponentScreenPolicy().to_dict()
    assert selected["ranking"]["overall"]["decided"] == 135
    assert selected["interval_coverage"] == {"n": 30, "contains": 30, "rate": 1}
    assert selected["promotion"] == "SCREENING_ONLY"


@pytest.mark.parametrize("moved", [1, 5])
def test_ordinary_domain_reader_refuses_resigned_split_workload_groups(tmp_path, moved):
    selected = report()
    rotation = {"a": "b", "b": "c", "c": "a"}
    for index, row in enumerate(selected["predictions"]):
        if index % 10 < moved:
            row["group"] = rotation[row["group"]]
    for row in selected["predictions"]:
        row["training_programs"] = [
            other["program"] for other in selected["predictions"] if other["group"] != row["group"]
        ]
    programs = [
        rank.Program(row["workload"], row["id"], row["measured_cycles"], row["group"])
        for row in selected["predictions"]
    ]
    intervals = {row["id"]: (row["prediction"]["lo"], row["prediction"]["hi"]) for row in selected["predictions"]}
    policy = ComponentScreenPolicy()
    selected["ranking"] = rank.verdict(
        rank.interval_agreement(rank.ordered_pairs(programs), intervals),
        {
            group: rank.interval_agreement(rank.ordered_pairs([p for p in programs if p.group == group]), intervals)
            for group in sorted({p.group for p in programs})
        },
        minimum_rate=policy.minimum_rank_rate,
        minimum_decided=policy.minimum_decided,
        minimum_slice_decided=policy.minimum_slice_decided,
        minimum_slices=policy.minimum_slices,
    )
    assert selected["ranking"]["exposable"]
    assert all(row["decided"] >= 20 for row in selected["ranking"]["slices"].values())
    path = tmp_path / "resigned-held.json"
    path.write_text(json.dumps(selected))
    with pytest.raises(ValueError, match="all variants of a workload must share one held-out group"):
        _qualified_domains(path, sha256_file(path), sha("calibration"))
