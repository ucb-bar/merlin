"""Whole-model boundary profile in the measured-claims workflow: per device-group cycles, the gaps
between groups and each group against its derived roofline, for the frozen baseline and the candidate."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase2 import broker_policy as BP
from merlin_experiments.phase2 import development_feedback as DF
from merlin_experiments.phase2 import whole_model_profile as WMP
from merlin_experiments.phase2.contracts import StageGateError

from merlin.runtime.whole_model_readback import parse_group_profile

MACHINE = {
    "array_rows": 16,
    "array_cols": 16,
    "read_bytes_per_cycle": 16,
    "write_bytes_per_cycle": 16,
    "basis": {},
    "unresolved": {},
}
ROUTED = [
    {  # a 64x64x64 matmul: compute floor 1024 cycles; 8 KiB in, 16 KiB out -> movement floor 1024
        "symbol": "dev_0",
        "group": 3,
        "parallel": [64, 64],
        "reduction": [64],
        "tensor_types": ["tensor<64x64xi8>", "tensor<64x64xi8>", "tensor<64x64xi32>"],
    },
    {  # a 3x3 convolution, NHWC: M = 8*8, K = 3*3*16, N = 32
        "symbol": "dev_1",
        "group": 5,
        "parallel": [1, 8, 8, 32],
        "reduction": [3, 3, 16],
        "tensor_types": ["tensor<1x10x10x16xi8>", "tensor<3x3x16x32xi8>", "tensor<1x8x8x32xi8>"],
    },
]
CONSOLE = "\n".join(
    [
        "GROUP_ID 0 g3_dev_0",
        "GROUP_ID 1 g5_dev_1",
        "GAP 0 500",
        "GROUP 0 2048",
        "GAP 1 100",
        "GROUP 1 1500",
        "GAP 2 352",
        "METRIC group_calls 2",
        "METRIC group_calls_dropped 0",
    ]
)


def test_routed_extents_give_the_contraction_and_its_compulsory_bytes():
    contractions, read, write, _ = WMP.group_roofline_inputs(ROUTED[0])
    assert contractions == [(64, 64, 64)] and (read, write) == (8192, 16384)
    contractions, read, write, _ = WMP.group_roofline_inputs(ROUTED[1])
    assert contractions == [(64, 144, 32)] and read == 1600 + 4608 and write == 2048


def test_the_document_states_cycles_gaps_shares_and_rooflines():
    doc = WMP.boundary_document(parse_group_profile(CONSOLE), ROUTED, MACHINE)
    assert (doc["window_cycles"], doc["group_cycles"], doc["gap_cycles"]) == (4500, 3548, 952)
    first, second = doc["groups"]
    assert first["gap_before"] == 500 and first["share_of_window"] == round(2048 / 4500, 6)
    # 64x64x64: both floors are 1024 cycles (4 blocks x 4 depths x 64 rows; 16 KiB out at 16 B/cycle).
    assert first["roofline"]["roofline_cycles"] == 1024
    assert first["roofline"]["over_roofline"] == 2.0
    # 8x8 outputs x 144 taps x 32 channels: the array's floor is 1152 cycles.
    assert second["roofline"]["status"] == "derived" and second["roofline"]["limiter"] == "compute"
    assert second["roofline"]["roofline_cycles"] == 1152
    assert doc["groups_with_roofline"] == 2


def test_a_group_faster_than_its_bound_refutes_the_bound():
    fast = CONSOLE.replace("GROUP 0 2048", "GROUP 0 10")
    doc = WMP.boundary_document(parse_group_profile(fast), ROUTED, MACHINE)
    assert doc["groups"][0]["roofline"]["status"] == "refuted"
    assert doc["groups"][0]["roofline"]["over_roofline"] is None


def _inputs(tmp_path):
    capture = tmp_path / "capture"
    capture.mkdir(exist_ok=True)
    (capture / "model.mlir").write_text("module {}\n")
    return WMP.WholeModelProfileInputs(
        capture=capture,
        host_package=tmp_path,
        board_catalog=tmp_path,
        board="b",
        dts=tmp_path,
        arena_mb=64,
        datapath="int8",
    )


def _fake_compile(**kw):
    out = kw["output"]
    out.mkdir(parents=True)
    (out / "group_profile.json").write_text(json.dumps(parse_group_profile(CONSOLE)))
    (out / "sidecar.json").write_text(json.dumps({"routed": ROUTED}))
    assert kw["group_profile"] is True and kw["run"] == "gsim"
    return {
        "output": {
            "group_profile": {"path": str(out / "group_profile.json")},
            "device_sidecar": {"path": str(out / "sidecar.json")},
        }
    }


def test_an_arm_is_routed_built_run_and_profiled(tmp_path):
    plans = []
    plan = lambda *a, **kw: plans.append(kw) or {"device_routing": object()}  # noqa: E731
    doc = WMP.run_arm(
        tmp_path / "pkg",
        _inputs(tmp_path),
        target="t",
        rtl_facts=None,
        output=tmp_path / "out",
        machine=MACHINE,
        compile_fn=_fake_compile,
        plan_fn=plan,
    )
    assert doc["status"] == "measured" and doc["calls"] == 2
    assert plans[0]["device_package"] == str(tmp_path / "pkg")
    refused = WMP.run_arm(
        tmp_path / "pkg",
        _inputs(tmp_path),
        target="t",
        rtl_facts=None,
        output=tmp_path / "out2",
        machine=MACHINE,
        compile_fn=_fake_compile,
        plan_fn=lambda *a, **kw: {"device_routing": None, "device_routing_why": "none"},
    )
    assert refused == {"status": "refused", "why": "no device route: none"}


def test_the_stage_profiles_baseline_once_and_the_candidate_each_call(tmp_path, monkeypatch):
    calls = []

    def fake_arm(package, inputs, **kw):
        calls.append(kw["output"].name)
        return WMP.boundary_document(parse_group_profile(CONSOLE), ROUTED, MACHINE)

    monkeypatch.setattr(WMP, "run_arm", fake_arm)
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    feedback = DF.DevelopmentGsimFeedback(
        None,
        None,
        tmp_path,
        "a" * 64,
        SimpleNamespace(target="t"),
        {},
        tmp_path / "work",
        {},
        machine_bounds=MACHINE,
        whole_model_inputs=WMP.WholeModelProfileSet((_inputs(tmp_path),)),
    )
    first = feedback.whole_model_profile(candidate, round_index=0, call_index=1, timeout_s=10)
    feedback.whole_model_profile(candidate, round_index=0, call_index=2, timeout_s=10)
    assert calls == ["baseline", "candidate", "candidate"]
    assert set(first) == {"schema", "engine", "purpose", "programs"}
    assert set(first["programs"][0]) == {"program", "baseline", "candidate"}
    bare = DF.DevelopmentGsimFeedback(None, None, tmp_path, "a" * 64, None, {}, tmp_path / "w2", {})
    with pytest.raises(StageGateError, match="no whole-model deployment inputs"):
        bare.whole_model_profile(candidate, round_index=0, call_index=1, timeout_s=10)


def test_the_action_is_advertised_unavailable_without_deployment_inputs(tmp_path):
    from merlin_experiments.phase2 import corpus_feedback as CF

    policy = CF.CorpusFeedbackPolicy.__new__(CF.CorpusFeedbackPolicy)
    policy.feedback_evaluator = None
    assert policy.unavailable == {BP.WHOLE_MODEL_PROFILE_ACTION: BP.WHOLE_MODEL_PROFILE_UNAVAILABLE}
    policy.feedback_evaluator = SimpleNamespace(whole_model_inputs=object())
    assert policy.unavailable == {}


def test_a_refusal_carries_only_its_reason_and_forbidden_fields_are_refused():
    row = {
        "program": "p0",
        "baseline": {"status": "refused", "why": "x"},
        "candidate": {"status": "refused", "why": "y"},
    }
    ok = {"schema": WMP.SCHEMA, "engine": "gsim", "purpose": "p", "programs": [row]}
    assert WMP.validate_document(ok) == ok
    with pytest.raises(StageGateError):
        WMP.validate_document({**ok, "programs": [{**row, "candidate": {"status": "refused", "why": "y", "extra": 1}}]})
    with pytest.raises(StageGateError):
        WMP.validate_document({**ok, "programs": [{**row, "candidate": {"status": "measured", "output": 1}}]})


PRIVATE = "/operator/private-inputs/captures/net_a/model.mlir"


def test_host_paths_are_scrubbed_except_the_agents_own_tree():
    from merlin.common.path_scrub import TOKEN, scrub_host_paths

    text = f"failed reading {PRIVATE}: in /stage/_measured_candidate/src/k.py:12 and (/opt/x/y), ratio 4/5 / 2"
    out = scrub_host_paths(
        text, keep=("/agent/submission",), rewrite={"/stage/_measured_candidate": "/agent/submission"}
    )
    assert PRIVATE not in out and "/opt/x/y" not in out and out.count(TOKEN) == 2
    assert "/agent/submission/src/k.py:12" in out and "ratio 4/5 / 2" in out


def test_no_private_path_reaches_an_agent_visible_document(tmp_path):
    """A build failure naming a private capture and an operator input surfaces with both scrubbed, while
    the agent's own tree survives; the same holds for every agent-visible reason this stage writes."""
    from merlin_experiments.phase2 import feedback_metrics as FM

    def failing_compile(**kw):
        raise RuntimeError(f"cannot open {PRIVATE}; compiler {kw['package']} said {tmp_path}/pkg/out.mlir:3")

    doc = WMP.run_arm(
        tmp_path / "pkg",
        _inputs(tmp_path),
        target="t",
        rtl_facts=None,
        output=tmp_path / "o",
        machine=MACHINE,
        compile_fn=failing_compile,
        plan_fn=lambda *a, **kw: {"device_routing": object()},
        visible_root=Path("/agent/submission"),
    )
    assert doc["status"] == "refused" and "/agent/submission/out.mlir:3" in doc["why"]
    assert PRIVATE not in doc["why"] and "private-inputs" not in doc["why"]
    no_route = WMP.run_arm(
        tmp_path / "pkg",
        _inputs(tmp_path),
        target="t",
        rtl_facts=None,
        output=tmp_path / "o2",
        machine=MACHINE,
        plan_fn=lambda *a, **kw: {"device_routing": None, "device_routing_why": f"capture {PRIVATE} unroutable"},
    )
    assert PRIVATE not in no_route["why"]
    executed = FM.executed_arm({"status": "unknown", "why": f"the functional engine did not start ({PRIVATE})"})
    assert PRIVATE not in json.dumps(executed)
    cell = FM.roofline_cell({}, {"unresolved": {"movement": f"unreadable {PRIVATE}"}})
    assert PRIVATE not in json.dumps(cell)


def _record(tmp_path, programs):
    for row in programs:
        (tmp_path / row["capture"]).mkdir(parents=True, exist_ok=True)
    record = {
        "schema": WMP.INPUTS_SCHEMA,
        "host_package": ".",
        "board_catalog": ".",
        "board": "b",
        "dts": ".",
        "arena_mb": 64,
        "programs": programs,
    }
    path = tmp_path / "inputs.json"
    path.write_text(json.dumps(record))
    return path


def test_the_record_declares_a_short_list_of_public_programs(tmp_path):
    path = _record(
        tmp_path,
        [
            {"name": "residual_cnn_scale", "capture": "caps/residual_cnn_scale", "datapath": "int8"},
            {"name": "decoder_scale", "capture": "caps/decoder_scale", "datapath": "int8", "arena_mb": 256},
        ],
    )
    inputs = WMP.load_inputs(path)
    assert [p.name for p in inputs.programs] == ["residual_cnn_scale", "decoder_scale"]
    assert [p.arena_mb for p in inputs.programs] == [64, 256]


def test_a_declared_private_full_model_is_refused_operator_side(tmp_path):
    descriptor = tmp_path / "descriptor.yaml"
    descriptor.write_text(
        "phase1_gates:\n  private_full_models:\n    models: [net_a, net_b]\n"
        "    programs: {net_a: [model]}\n    source_workload_dirs: {net_a: net_a_src}\n"
    )
    public = WMP.load_inputs(
        _record(tmp_path, [{"name": "decoder_scale", "capture": "caps/decoder_scale", "datapath": "int8"}])
    )
    WMP.refuse_private_programs(public, descriptor)
    by_name = WMP.load_inputs(_record(tmp_path, [{"name": "rn", "capture": "caps/net_a", "datapath": "int8"}]))
    with pytest.raises(StageGateError, match="declared private full model"):
        WMP.refuse_private_programs(by_name, descriptor)
    hidden = tmp_path / "caps" / "innocuous"
    hidden.mkdir(parents=True)
    (hidden / "capture_receipt.json").write_text(
        json.dumps({"source": {"path": "/x/staged/applications/net_a_src/loader.py"}})
    )
    by_receipt = WMP.load_inputs(
        _record(tmp_path, [{"name": "innocuous", "capture": "caps/innocuous", "datapath": "int8"}])
    )
    with pytest.raises(StageGateError, match="declared private full model"):
        WMP.refuse_private_programs(by_receipt, descriptor)


def test_without_tensor_types_only_a_plain_contraction_is_charged_bytes():
    plain = {"parallel": [100, 12], "reduction": [147], "dtypes": ["i8", "i8", "i32"]}
    _, read, write, basis = WMP.group_roofline_inputs(plain)
    assert (read, write) == (100 * 147 + 147 * 12, 100 * 12 * 4) and "element types" in basis
    windowed = {"parallel": [1, 8, 8, 32], "reduction": [3, 3, 16], "dtypes": ["i8", "i8", "i8"]}
    assert WMP.group_roofline_inputs(windowed)[1:3] == (None, None)
