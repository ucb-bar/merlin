"""gSIM wall time is predicted before launch, from recorded campaign jobs.

RECORDED: every measured idle-gSIM job of three campaigns (``joint_gsim_measurement_budget.json`` of
gemmini-open-compiler-20261026-fresh2/3/4: measured barrier cycles and elapsed seconds), with each case's
logical read/write bytes from its structural roofline. KILLED: the fresh4 jobs the harness's cycle cap
stopped after ~12 minutes, with the cycles the lane estimated for them and their declared bytes.
"""

import json

import pytest
from test_idle_gsim_measurement import _workspace, campaign  # noqa: F401 -- the shared campaign fixture

from merlin_experiments.phase1 import gsim_wall_cap as W
from merlin_experiments.phase1 import gsim_wall_predictor as P
from merlin_experiments.phase1 import joint_control
from merlin_experiments.phase1 import joint_gsim_measurement as M
from merlin_experiments.phase1.feedback import idle_gsim as G
from merlin_experiments.phase1.feedback import idle_performance as I

BYTES = {  # case: (read, write) logical requested bytes
    "ZA_256": (131072, 65536), "ZA_512": (524288, 262144), "ZA_32x1024x1024": (1081344, 32768),
    "PR01": (65536, 1024), "PR02": (131072, 1024), "PR03": (131584, 1024), "PR04": (196608, 1024),
    "PR05": (262144, 1024), "PR06": (262656, 1024), "PR08": (524288, 1024), "JM_flow": (24640, 144000),
    "PM02": (1024, 3072), "PF00": (576, 256), "PQ04": (4352, 16384), "PQ09": (8704, 16384),
    "PM11": (1792, 12288), "PM14": (1792, 12288), "PM07": (1536, 8192),
}
RECORDED = [  # (case, measured cycles, elapsed seconds)
    ("JM_flow", 56292, 307.1), ("PR08", 181912, 472.4), ("PR01", 17695, 77.0), ("PM02", 1147, 41.6),
    ("PF00", 402, 32.8), ("PQ04", 6011, 58.0), ("PQ09", 9473, 112.5), ("PM11", 3939, 81.8), ("PM14", 3976, 50.0),
    ("PM07", 2734, 41.7), ("ZA_256", 421947, 204.4), ("PR06", 75356, 183.3), ("PR04", 55398, 144.7),
    ("PR03", 35938, 107.9), ("PR06", 75373, 183.6), ("JM_flow", 46174, 226.2), ("ZA_256", 115705, 197.3),
    ("ZA_256", 115705, 203.6), ("PR08", 138162, 328.2), ("PR05", 58127, 183.2), ("PR06", 56973, 182.6),
    ("PR04", 42587, 139.3), ("PR03", 27963, 105.9), ("ZA_512", 757200, 683.6), ("ZA_32x1024x1024", 213342, 667.2),
    ("PR02", 27976, 105.1), ("PR01", 14087, 68.2), ("PR08", 137772, 323.6), ("PR05", 57984, 173.2),
    ("ZA_256", 177007, 241.1), ("PR02", 24730, 141.1), ("PR01", 12326, 94.3), ("ZA_256", 125143, 198.5),
    ("PR01", 7567, 69.5),
]
KILLED = {  # case: (lane cycle estimate, read bytes, write bytes)
    "JT_tiny_decode_m1k2048n5728": (91779026, 11732992, 22912),
    "JT_tiny_decode_m1k6096n2048": (105669298, 12490704, 8192),
    "JT_tiny_prefill_m32k2048n32000 (LM head)": (10996550, 65601536, 4096000),
    "ZA_conv_3x3x56x56x64x64x4": (2494728, 898304, 802816),
    "ZA_gemm_3136x128x512": (2933834, 466944, 1605632),
    "ZA_schedule_reuse_gemm_6272x64x256": (1579495, 417792, 1605632),
    "PD_resnet50_m3136k64n256": (902325, 217088, 3211264),
    "PD_smolvla_prefix_encode_m1024k64n1024": (1109622, 131072, 4194304),
}
CAP = 1800


def _case(cycles, read, write):
    return {"cycles": float(cycles), "read_bytes": float(read), "write_bytes": float(write)}


def _recorded_points():
    return [(_case(cycles, *BYTES[name]), wall) for name, cycles, wall in RECORDED]


def test_the_fit_reproduces_recorded_walls_with_bytes_dominating():
    model = P.fit(_recorded_points())
    c = model["coefficients"]
    assert 20 < c["setup"] < 60  # the startup the jobs share
    assert c["write_bytes"] > c["read_bytes"] > 0  # exported output bytes cost more than loaded input
    assert 1 <= model["margin"] < 2.5  # the fit's own error spread
    within = [wall <= model["margin"] * P.predict(model, case) for case, wall in _recorded_points()]
    assert sum(within) >= 0.9 * len(within)
    # A data-heavy kernel is slower per simulated cycle than a compute-heavy one, as recorded.
    heavy, light = _case(213342, *BYTES["ZA_32x1024x1024"]), _case(421947, *BYTES["ZA_256"])
    assert P.predict(model, heavy) > 2 * P.predict(model, light)


@pytest.mark.parametrize("name", sorted(KILLED))
def test_every_job_the_cycle_cap_killed_is_predicted_over_the_cap_and_not_run(name):
    model = P.fit(_recorded_points())
    decision = P.decide(model, _case(*KILLED[name]), cap_s=CAP)
    assert decision["run"] is False and decision["projected_s"] > CAP
    assert decision["reason"].startswith("gSIM not run: projected ~")
    assert decision["reason"].endswith("> 30 min cap; use the timing model or a smaller representative case")


def test_recorded_jobs_run_with_their_projection_as_timeout():
    model = P.fit(_recorded_points())
    decision = P.decide(model, _case(757200, *BYTES["ZA_512"]), cap_s=CAP)
    assert decision["run"] and 683.6 <= decision["timeout_s"] <= CAP
    assert decision["timeout_s"] == pytest.approx(model["margin"] * decision["predicted_s"], abs=1)
    assert P.decide(None, _case(1, 1, 1), cap_s=CAP)["run"] is True  # no fit: the caller's rule stands
    assert P.decide(model, None, cap_s=CAP)["run"] is True  # no features: likewise


def _ledger_jobs():
    jobs = {}
    for index, (name, cycles, wall) in enumerate(RECORDED[:8]):
        jobs[f"j{index}"] = {"capsule": name, "outcome": "measured", "barrier_cycles": cycles, "elapsed_s": wall,
                             "wall_features": _case(1, *BYTES[name])}
    return jobs


def test_the_campaign_model_refits_with_measured_outcomes_and_seeds_from_earlier_runs(tmp_path):
    run = tmp_path / "runs" / "current"
    run.mkdir(parents=True)
    own = P.campaign_model(_ledger_jobs(), run)
    assert own["observations"] == 8 and "of this run" in own["basis"]
    # The measured cycles, not the pre-launch estimate (1 here), enter the fit.
    assert all(case["cycles"] > 1 for case, _ in P.ledger_points(_ledger_jobs()))
    earlier = tmp_path / "runs" / "earlier"
    (earlier / "qa_history").mkdir(parents=True)
    rows = [{"capsule": name, "performance": {"structural_roofline": {"logical_request_ideal_service": {
        "read": {"requested_bytes": read}, "write": {"requested_bytes": write}}}}} for name, (read, write) in BYTES.items()]
    (earlier / "qa_history" / "verdict_fast_1.json").write_text(json.dumps({"per_capsule": rows}))
    (earlier / "joint_gsim_measurement_budget.json").write_text(json.dumps({"jobs": {
        f"h{i}": {"capsule": name, "outcome": "measured", "barrier_cycles": cycles, "elapsed_s": wall}
        for i, (name, cycles, wall) in enumerate(RECORDED)}}))
    seeded = P.campaign_model({}, run)
    assert seeded["observations"] == len(RECORDED) and "of earlier runs" in seeded["basis"]


def test_the_lane_does_not_launch_a_job_projected_over_the_cap(tmp_path, campaign, capsys):  # noqa: F811
    state = json.loads((campaign / "joint_campaign.json").read_text())
    state["policy"]["gsim_measurement"] = {"max_job_wall_s": CAP}
    (campaign / "joint_campaign.json").write_text(json.dumps(state))
    with M._ledger() as data:
        data["jobs"].update(_ledger_jobs())
    ws = _workspace(tmp_path, cohort=("big",))
    # The case declares more output than any recorded job ever exported.
    reply = ws / ".qa_channel" / "resp_1_a.json"
    document = json.loads(reply.read_text())
    document["cases"][0]["performance"]["structural_roofline"]["logical_request_ideal_service"] = {
        "read": {"requested_bytes": 12_000_000}, "write": {"requested_bytes": 4_000_000}}
    reply.write_text(json.dumps(document))
    lane = G.GsimLane(ws, campaign)
    known = I.estimates(ws / ".qa_channel", with_status=True)
    assert lane.tick(ws / ".qa_channel", cohort=["big"], l2={"big": "pass"}, known=known, digest="d1") is None
    assert lane.last.startswith(W.SKIP) and "> 30 min cap" in lane.last
    assert not list((ws / ".qa_channel").glob(f"req_{M.REQUEST_PREFIX}*.json"))
    assert "[idle-gsim] big not measured: gSIM not run: projected ~" in capsys.readouterr().out


def test_a_launched_job_records_its_prediction_for_the_refit(tmp_path, campaign):  # noqa: F811
    state = json.loads((campaign / "joint_campaign.json").read_text())
    state["policy"]["gsim_measurement"] = {"max_job_wall_s": CAP}
    (campaign / "joint_campaign.json").write_text(json.dumps(state))
    with M._ledger() as data:
        data["jobs"].update(_ledger_jobs())
    ws = _workspace(tmp_path, cohort=("big",))
    reply = ws / ".qa_channel" / "resp_1_a.json"
    document = json.loads(reply.read_text())
    document["cases"][0]["performance"]["structural_roofline"]["logical_request_ideal_service"] = {
        "read": {"requested_bytes": 65536}, "write": {"requested_bytes": 4096}}
    reply.write_text(json.dumps(document))
    lane = G.GsimLane(ws, campaign)
    known = I.estimates(ws / ".qa_channel", with_status=True)
    assert lane.tick(ws / ".qa_channel", cohort=["big"], l2={"big": "pass"}, known=known, digest="d1")
    job = next(j for j in M.read()["jobs"].values() if j.get("capsule") == "big")
    assert job["wall_features"]["read_bytes"] == 65536 and 0 < job["predicted_wall_s"] < job["timeout_s"] <= CAP


def test_an_author_request_shows_its_projection_when_refused(tmp_path, campaign):  # noqa: F811
    state = json.loads((campaign / "joint_campaign.json").read_text())
    state["policy"]["gsim_correctness"] = {"capsules": ["huge", "small"], "max_job_wall_s": CAP, "max_jobs": 2,
                                           "per_job_s": 1200, "worker_budget_s": 9000}
    (campaign / "joint_campaign.json").write_text(json.dumps(state))
    with M._ledger() as data:
        data["jobs"].update(_ledger_jobs())
    (campaign / "qa_history").mkdir()

    def row(name, cycles, read, write):
        return {"capsule": name, "performance": {"estimated_whole_soc_cycles": cycles, "structural_roofline": {
            "logical_request_ideal_service": {"read": {"requested_bytes": read}, "write": {"requested_bytes": write}}}}}

    (campaign / "qa_history" / "verdict_fast_1.json").write_text(json.dumps({"per_capsule": [
        row("huge", 2933834, 466944, 1605632), row("small", 9687, 65536, 1024)]}))
    calls = []

    @joint_control.selfcheck_policy
    def selfcheck(argv=None, **kwargs):
        calls.append(argv)
        return 0

    out = tmp_path / "out.json"
    assert selfcheck(["--sim", "gsim", "--capsules", "huge", "--timeout", "1200", "--out", str(out)]) == 2
    report = json.loads(out.read_text())
    assert report["joint_policy_refused"] and report["error"].startswith("gSIM not run: projected ~")
    assert "> 30 min cap; use the timing model or a smaller representative case" in report["error"] and not calls
    assert W.request_refusal("small") is None
