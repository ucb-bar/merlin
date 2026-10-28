import json

from merlin_experiments.corpus.coverage import withheld_from_generation


def _plan(tmp_path, receipt):
    (tmp_path / "coverage").mkdir()
    (tmp_path / "coverage" / "generation.json").write_text(json.dumps(receipt))
    return {"phase0_evidence_bundle": str(tmp_path)}


def test_disclosed_source_gaps_are_read_with_their_reasons_and_never_dropped(tmp_path):
    long_reason = "m2m_q.quantize.default: " + "x" * 500
    plan = _plan(tmp_path, {
        "host_placement_not_published": [{"capsule": "SY_micro_model", "reason": "host lane"}],
        "disclosed_source_gaps": [
            {"capsule": "SY_source_a", "kind": "writer_refusal", "reason": "M2MUnavailable: mixed", "reason_sha256": "a" * 64},
            {"capsule": "SY_source_b", "kind": "host_lane_admission", "reason": long_reason, "reason_sha256": "b" * 64},
        ],
    })
    out = withheld_from_generation(plan)
    assert out["SY_micro_model"] == "host lane" and out["SY_source_a"] == "M2MUnavailable: mixed"
    assert out["SY_source_b"].startswith(long_reason[:240]) and "sha256 " + "b" * 64 in out["SY_source_b"]


def test_a_source_gap_without_a_reason_is_not_a_disclosure(tmp_path):
    plan = _plan(tmp_path, {"disclosed_source_gaps": [{"capsule": "SY_source_c", "reason": "  "}, {"reason": "r"}]})
    assert withheld_from_generation(plan) == {}
