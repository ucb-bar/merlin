"""The shape panel and the per-class "vs best" line: classes from declarations, like-with-like ratios."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from merlin_experiments.phase1.feedback import case_declarations as D
from merlin_experiments.phase1.feedback import performance_brief as PB
from merlin_experiments.phase1.feedback import shape_classes as S
from merlin_experiments.phase1.feedback import shape_panel as P

EDGE = 16


def _dense(m, k, n, *, role="handauthored_compiler_test", batch=None):
    lhs, weight = ([batch, m, k], [batch, k, n]) if batch else ([m, k], [k, n])
    return {
        "source_role": role,
        "semantic": {"semantic_family": "contraction"},
        "inputs": [{"name": "W", "role": "weight", "shape": weight, "dtype": "i8"},
                   {"name": "A0", "role": "input", "shape": lhs, "dtype": "i8"}],
        "operation": {"op": "matmul", "attributes": {"lhs": "A0", "weight": "W", "out": "Y0"}},
    }


def _conv(h, w, c, k, co, *, stride=1, pad=0):
    return {
        "source_role": "derived_sweep",
        "semantic": {"semantic_family": "contraction"},
        "inputs": [{"name": "W", "role": "weight", "shape": [k * k * c, co], "dtype": "i8"},
                   {"name": "IFM", "role": "input", "shape": [1, h, w, c], "dtype": "i8"}],
        "operation": {"op": "conv2d", "attributes": {
            "ifm": "IFM", "weight": "W", "ci": c, "kh": k, "kw": k, "stride": [stride, stride],
            "padding": [pad] * 4, "dilation": [1, 1], "layout": "nhwc"}},
    }


def _classes(declaration):
    return S.classes(S.describe(declaration), EDGE)


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    monkeypatch.setattr(D, "_ROOTS", [])
    monkeypatch.setattr(D, "_TARGET", [])
    monkeypatch.setattr(D, "_CACHE", {})
    monkeypatch.setattr(S, "tile_edge", lambda target: (EDGE, "test facts arrays[mesh]"))
    monkeypatch.delenv("MERLIN_FROZEN_PUBLIC_SCALABILITY", raising=False)


def test_classes_come_from_declared_extents_relative_to_the_tile_edge():
    assert _classes(_dense(512, 512, 512)) == ("square",)
    assert _classes(_dense(3136, 64, 256)) == ("tall",)
    assert _classes(_dense(32, 1024, 1024)) == ("wide",)
    assert _classes(_dense(1, 2048, 2048)) == ("thin",)
    assert _classes(_dense(17, 33, 15)) == ("square", "ragged")
    assert _classes(_dense(48, 48, 48, role="model_derived")) == ("square", "model_layer")
    # A batched operand folds its batch into M; the family is still dense.
    assert S.describe(_dense(16, 32, 16, batch=2))["extents"] == {"m": 32, "k": 32, "n": 16}


def test_family_is_derived_from_the_declaration_not_the_name():
    conv = S.describe(_conv(8, 8, 32, 3, 16))
    assert conv["family"] == "conv" and conv["extents"] == {"m": 36, "k": 288, "n": 16}
    assert S.families(S.describe(_dense(1, 64, 64))) == ("dense", "gemv")
    assert S.families(S.describe(_dense(16, 64, 64))) == ("dense",)
    elementwise = {"semantic": {"semantic_family": "elementwise_map"}, "operation": {"op": "residual_add"},
                   "inputs": [{"name": "X0", "role": "input", "shape": [12, 48], "dtype": "i8"}]}
    assert S.describe(elementwise)["family"] == "elementwise_map"
    assert S.canonical_family("Contraction") == "dense" and S.canonical_family("elementwise") == "elementwise_map"
    # Nothing to place: a whole model has no single operand pair.
    assert S.describe({"semantic": {"semantic_family": "contraction"}, "operation": {"op": "model"},
                       "inputs": []}) is None


def _roster():
    return [
        ("core", [
            ("c_square_small", _dense(64, 64, 64)),
            ("c_square_big", _dense(256, 256, 256)),
            ("c_square_timed", _dense(128, 128, 128)),
            ("c_tall", _dense(784, 128, 64)),
            ("c_wide", _dense(12, 48, 80)),
            ("c_decode", _dense(1, 48, 80, role="model_derived")),
            ("c_ragged", _dense(141, 49, 149)),
            ("c_layer", _dense(12, 48, 48, role="model_derived")),
            ("c_conv", _conv(8, 8, 32, 3, 16)),
        ]),
        ("scalability", [("s_square_huge", _dense(1024, 1024, 1024)), ("s_tall", _dense(12544, 64, 64))]),
    ]


def test_panel_picks_one_measurable_passing_case_per_class_deterministically():
    cases = P.candidates(_roster(), family="dense", edge=EDGE)
    assert "c_conv" not in {c["case"] for c in cases}
    status = {c["case"]: "pass" for c in cases} | {"c_square_big": "fail"}
    best = {("c_square_timed", "gsim_measured"): {"cycles": 1000}}
    picks = P.select(cases, status=status, best=best, sim="gsim")
    chosen = {k: (v or {}).get("case") for k, v in picks.items()}
    # gSIM: core cases are measurable, a passing case with same-kind evidence first, then the largest work.
    assert chosen["square"] == "c_square_timed"
    assert chosen["tall"] == "c_tall"
    assert chosen["wide"] == "c_wide"
    assert chosen["thin"] == "c_decode"
    assert chosen["ragged"] == "c_ragged"
    # The held-out layer is a model-derived case no other class already stands for.
    assert chosen["model_layer"] == "c_layer"
    assert P.select(cases, status=status, best=best, sim="gsim") == picks
    # A Spike panel takes the largest work across both suites (nothing is measurable there).
    spike = P.select(cases, status={}, best={}, sim="spike")
    assert spike["square"]["case"] == "s_square_huge" and spike["tall"]["case"] == "s_tall"


def _history(tmp_path, rows):
    (tmp_path / "qa").mkdir(exist_ok=True)
    (tmp_path / "qa" / "perf_history.jsonl").write_text(
        "".join(json.dumps({"schema": "merlin.phase1.perf_history.v1", **row}) + "\n" for row in rows)
    )


def test_like_with_like_only_cycle_evidence_of_the_same_kind_forms_a_ratio(tmp_path):
    _history(tmp_path, [
        {"test_case": "a", "kind": "measured", "evidence": "measured: latest gSIM certificate", "cycles": 1000},
        {"test_case": "a", "kind": "measured", "evidence": "measured: gSIM measurement (advisory)", "cycles": 1200},
        {"test_case": "a", "kind": "modeled", "evidence": "modeled: Spike timing model (provisional)", "cycles": 90},
        {"test_case": "b", "kind": "modeled", "evidence": "modeled: Spike timing model (qualified)", "cycles": 500},
        {"test_case": "c", "kind": "modeled", "evidence": "modeled: Spike timing model (provisional)", "cycles": 70},
        {"test_case": "a", "kind": "status", "l2": "pass"},
    ])
    best = S.history_best(tmp_path)
    assert best[("a", "gsim_measured")]["cycles"] == 1000
    assert best[("a", "provisional_model")]["cycles"] == 90
    # gSIM against gSIM.
    assert S.compare("a", 1430, "gsim_measured", best, "square") == {
        "best_cycles": 1000, "ratio_vs_best": 1.43, "verdict": "regressed 1.43x vs best (square)"}
    assert S.compare("a", 800, "gsim_measured", best, "square")["verdict"] == "improved to 0.8x of best (square)"
    # A calibrated model against the same calibrated model.
    assert S.compare("b", 500, "calibrated_model", best, "thin")["ratio_vs_best"] is None  # old row lacks model identity
    # A historical provisional estimate without evaluator identity is not comparable.
    no = S.compare("c", 35, "provisional_model", best, "wide")
    assert no["ratio_vs_best"] is None and "missing complete model/facts/scope identity" in no["verdict"]
    # Spike counts are never cycles.
    assert S.compare("a", 50, "spike_count", best, "square")["ratio_vs_best"] is None
    # gSIM cycles never compare with a model's best, and vice versa.
    assert S.compare("b", 400, "gsim_measured", best, "decode")["verdict"].startswith("no earlier gsim measured")
    assert S.evidence_kind("measured: gSIM cycles of this request") == "gsim_measured"
    assert S.evidence_kind("spike count: L2 tier simulator cycles (not cycle-accurate)") == "spike_count"
    assert S.evidence_kind("modeled: Spike timing model (self-check)") == "provisional_model"
    assert S.history_status(tmp_path) == {"a": "pass"}


def _register(tmp_path, cases):
    root = tmp_path / "corpus"
    for name, declaration in cases.items():
        (root / name).mkdir(parents=True)
        (root / name / "capsule.yaml").write_text(yaml.safe_dump(declaration))
    D.register(root, target="unit-target")
    return root


def test_vs_best_line_follows_the_same_rule(tmp_path):
    _register(tmp_path, {"sq": _dense(256, 256, 256), "dec": _dense(1, 256, 256), "wide": _dense(16, 64, 512)})
    _history(tmp_path, [
        {"test_case": "sq", "kind": "measured", "evidence": "measured: latest gSIM certificate", "cycles": 100},
        {"test_case": "dec", "kind": "modeled", "evidence": "modeled: Spike timing model (provisional)", "cycles": 10},
    ])
    line = S.vs_best_line([
        ("sq", 143, "measured: gSIM cycles of this request"),
        ("dec", 5, "modeled: Spike timing model (provisional)"),
        ("wide", 900, "measured: gSIM cycles of this request"),
    ], tmp_path)
    assert "square 1.43x (sq, gsim_measured)" in line
    assert "thin: no earlier comparable provisional_model cycles (dec)" in line
    assert "wide: no earlier comparable gsim_measured cycles (wide)" in line


def test_performance_brief_labels_evidence_kind_and_carries_the_class_line(tmp_path):
    _register(tmp_path, {"sq": _dense(256, 256, 256)})
    _history(tmp_path, [{"test_case": "sq", "kind": "measured", "evidence": "measured: latest gSIM certificate",
                         "cycles": 1000}])
    row = {"capsule": "sq", "pass": True, "barrier_tier": "L3", "barrier_cycles": 1500,
           "tiers": {"L2": {"status": "pass"}},
           "performance": {"structural_roofline": {"useful_macs": 256 ** 3, "structural_peak_macs_per_cycle": 256,
                                                   "structural_compute_floor_cycles": 1000}}}
    brief = PB.build([row], workspace=tmp_path)
    entries = brief["largest_headroom"] + brief["diagnostic_controls"]
    assert entries[0]["evidence_kind"] == "gsim_measured"
    assert brief["vs_best_per_class"].endswith("square 1.5x (sq, gsim_measured)")


def test_panel_cli_grades_its_picks_as_one_ordinary_subset_and_reports_per_class(tmp_path, monkeypatch):
    root = _register(tmp_path, {"sq": _dense(256, 256, 256), "dec": _dense(1, 256, 256),
                                "tall": _dense(784, 64, 64)})
    _history(tmp_path, [
        {"test_case": "sq", "kind": "measured", "evidence": "measured: latest gSIM certificate", "cycles": 1000},
        {"test_case": "tall", "kind": "measured", "evidence": "measured: latest gSIM certificate", "cycles": 400},
    ] + [{"test_case": n, "kind": "status", "l2": "pass"} for n in ("sq", "dec", "tall")])
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MERLIN_JOINT_CONTROL", raising=False)
    seen = {}

    def run(argv, **kwargs):  # the ordinary self-check, as a stand-in
        seen["argv"] = argv
        capsules = argv[argv.index("--capsules") + 1].split(",")
        rows = []
        for name in capsules:
            cycles = {"sq": 1430, "tall": 400}.get(name)
            rows.append({"capsule": name, "pass": True, "barrier_tier": "L3" if cycles else "L2",
                         "barrier_cycles": cycles or 77})
        Path(argv[argv.index("--out") + 1]).write_text(json.dumps({"per_capsule": rows, "all_pass": True}))
        return 0

    a = argparse.Namespace(capsules="panel:dense", sim="gsim", timeout=100, out=str(tmp_path / "out.json"),
                           contract=None)
    code = P.cli(["--capsules", "panel:dense", "--sim", "gsim", "--out", a.out], a,
                 context=SimpleNamespace(target="unit-target"), root=root, run=run)
    assert code == 0
    assert seen["argv"][seen["argv"].index("--capsules") + 1] == "dec,sq,tall"
    panel = json.loads(Path(a.out).read_text())["panel"]
    assert panel["classes"]["square"]["verdict"] == "regressed 1.43x vs best (square)"
    assert panel["classes"]["tall"]["verdict"] == "matches best (tall)"
    decode = panel["classes"]["thin"]
    assert decode["evidence_kind"] == "spike_count" and decode["ratio_vs_best"] is None
    assert decode["verdict"] == "no cycle-accurate evidence for this class (thin)"
    assert panel["classes"]["wide"]["case"] is None
    assert "regressed 1.43x vs best (square)" in panel["summary"]
    # Every graded class is a shape_policy observation; only cycle evidence is complete timing.
    square = panel["classes"]["square"]["shape_policy"]
    assert square["geometry"] == {"family": "dense", "operand_shapes": [[256, 256], [256, 256]]}
    assert square["observation"]["timing_status"] == "complete" and square["observation"]["cycles"] == 1430
    assert square["identity"]["basis"] == "measured_roi"
    thin = panel["classes"]["thin"]["shape_policy"]["observation"]
    assert thin["timing_status"] == "unknown" and thin["cycles"] is None and thin["complete_roi"] is False


def test_panel_records_build_shape_policy_observations_only_with_the_authors_pins():
    from merlin.transforms.evaluated_exploration import CompletedObservation
    from merlin.transforms.pass_exploration import ExplorationGeometry

    pick = {"case": "sq", "operand_shapes": [[256, 256], [256, 256]]}
    row = {"capsule": "sq", "pass": True, "execution_digest": "a" * 64,
           "performance": {"facts_sha256": "b" * 64, "scope_sha256": "c" * 64}}
    measured = P.policy_record(pick, row, "gsim_measured", 1430, family="dense")
    pins = {name: "d" * 64 for name in P.AUTHOR_PINS} | {"artifact_format": "xdsl"}
    with pytest.raises(ValueError, match="identity pins"):
        P.completed_observation(measured)  # never minted with an invented identity
    geometry, observation = P.completed_observation(measured, **pins)
    assert type(geometry) is ExplorationGeometry and geometry.operand_shapes == ((256, 256), (256, 256))
    assert type(observation) is CompletedObservation and observation.eligible and observation.cycles == 1430
    # Incomplete provisional estimates and functional counts cannot enter cycle rankings.
    _, provisional = P.completed_observation(P.policy_record(pick, row, "provisional_model", 99, family="dense"),
                                             **pins)
    assert provisional.identity.basis == "modeled_roi" and not provisional.eligible and provisional.cycles is None
    spike = P.policy_record(pick, row, "spike_count", 99, family="dense")
    assert spike["identity"]["basis"] is None  # a Spike count states no ROI basis of its own
    with pytest.raises(ValueError, match="basis"):
        P.completed_observation(spike, **pins)
    _, counted = P.completed_observation(spike, **pins, basis="modeled_roi")
    assert not counted.eligible and counted.cycles is None


def _shim(tmp_path, *args):
    from merlin_experiments.phase1.tools import selfcheck as shim

    ws = tmp_path / "ws"
    (ws / "submission").mkdir(parents=True)
    (ws / "submission" / "manifest.yaml").write_text("name: x\n")
    shutil.copyfile(shim.__file__, ws / "agent_selfcheck.py")
    env = {k: v for k, v in os.environ.items() if not k.startswith("MERLIN_")}
    done = subprocess.run([sys.executable, str(ws / "agent_selfcheck.py"), *args, "--wait-budget", "1"],
                          cwd=ws, env=env, capture_output=True, text=True, timeout=60)
    requests = sorted((ws / ".qa_channel").glob("req_*.json"))
    return done, [json.loads(p.read_text()) for p in requests]


def test_shim_sends_the_panel_as_one_ordinary_request_on_gsim_by_default(tmp_path):
    done, requests = _shim(tmp_path, "--panel", "dense")
    assert done.returncode == 3, done.stderr
    assert len(requests) == 1
    assert requests[0]["capsules"] == "panel:dense" and requests[0]["sim"] == "gsim"
    done, requests = _shim(tmp_path / "spike", "--panel", "conv", "--sim", "spike")
    assert requests[0]["capsules"] == "panel:conv" and requests[0]["sim"] == "spike"
    done, requests = _shim(tmp_path / "both", "--panel", "dense", "--capsules", "a,b")
    assert done.returncode == 2 and not requests


def test_joint_campaign_admits_each_gsim_panel_case_under_its_own_policy(monkeypatch):
    import contextlib

    from merlin_experiments.phase1 import joint_control as J
    from merlin_experiments.phase1 import joint_gsim_measurement as M

    monkeypatch.setattr(J, "selected", lambda: {"policy": {}})
    monkeypatch.setattr(J, "remaining", lambda drain=False: 500)
    monkeypatch.setattr(M, "requested", lambda: None)
    monkeypatch.setattr(J, "correctness_capsules", lambda value=None: frozenset({"required"}))
    calls = []

    @contextlib.contextmanager
    def correctness(name):
        calls.append(("correctness", name))
        yield 300

    @contextlib.contextmanager
    def witness(name):
        calls.append(("witness", name))
        if name == "busy":
            raise ValueError("joint gSIM concurrency or aggregate worker budget exhausted")
        yield 200

    monkeypatch.setattr(J, "physical_correctness", correctness)
    monkeypatch.setattr(J, "physical_witness", witness)
    # The campaign wrapper passes the panel through; the panel admits each case itself.
    seen = []
    J.selfcheck_policy(lambda argv, **kw: seen.append(argv))(["--capsules", "panel:dense", "--sim", "gsim"])
    assert seen and "panel:dense" in seen[0] and seen[0][seen[0].index("--timeout") + 1] == "500"
    assert not calls
    with contextlib.ExitStack() as stack:
        admitted, refused, timeout = P._admit(["required", "small", "busy"], stack, 1000)
    assert admitted == ["required", "small"] and timeout == 200
    assert "campaign policy" in refused["busy"]
    assert calls == [("correctness", "required"), ("witness", "small"), ("witness", "busy")]


def test_complete_provisional_estimates_feed_policy_and_compare_only_same_domain(tmp_path):
    from merlin_experiments.phase1.feedback import perf_history
    identity = dict(complete_event_accounting=True, **{key: 'b' * 64 for key in (
        'selection_sha256', 'model_sha256', 'facts_sha256', 'comparison_domain_sha256',
        'scope_sha256', 'capsule_sha256')})
    performance = dict(identity, status='PROVISIONAL', estimated_whole_soc_cycles=100,
                       elf_sha256='a' * 64)
    fields = perf_history._evidence_fields(100, {}, performance)
    assert S.model_domain(fields['performance_identity']) == S.model_domain(performance)
    _history(tmp_path, [dict(test_case='large', kind='modeled',
        evidence='modeled: Spike timing model (provisional)', **fields)])
    best = S.history_best(tmp_path)
    compared = S.compare('large', 80, 'provisional_model', best, 'wide', domain=S.model_domain(performance))
    assert compared['ratio_vs_best'] == .8
    assert 'not measured speedup' in compared['verdict']
    for key in ('model_sha256', 'facts_sha256', 'scope_sha256', 'capsule_sha256'):
        changed = dict(performance, **{key: 'c' * 64})
        assert S.compare('large', 80, 'provisional_model', best, 'wide',
                         domain=S.model_domain(changed))['ratio_vs_best'] is None
    pick = dict(operand_shapes=[[32, 2048], [2048, 2048]])
    row = dict(pass_=True, performance=performance)
    row['pass'] = row.pop('pass_')
    record = P.policy_record(pick, row, 'provisional_model', 100, family='dense')
    pins = {name: 'd' * 64 for name in P.AUTHOR_PINS} | {'artifact_format': 'xdsl'}
    _, observed = P.completed_observation(record, **pins)
    assert observed.eligible and observed.cycles == 100
    performance['complete_event_accounting'] = False
    assert P.policy_record(pick, row, 'provisional_model', 100, family='dense')['observation']['cycles'] is None
