"""Sealed process transport controls, with synthetic grading and no hardware authority."""

from __future__ import annotations

import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from types import SimpleNamespace

import pytest
from merlin_experiments.phase1.feedback import operator_child as OC
from merlin_experiments.phase2.contracts import StageGateError
from test_compiler_profile_child import _scope, original  # noqa: F401 -- share the existing exact-owner fixture

from merlin.targetgen import capsule_runner as CR

CONTROL = """
import json,os,time
from pathlib import Path
from types import SimpleNamespace
from merlin.targetgen import capsule_runner as CR, package_runtime as P, provenance
from merlin.targetgen.capsule_common import make_run_paths
from merlin_experiments.phase1.feedback import operator_child as OC
P.load_package=lambda directory,**kwargs: SimpleNamespace(directory=Path(directory),manifest={},integrity_exempt=False)
provenance.toolchain_shas=lambda target: {'fixture':'synthetic-control-only'}
def grade(capsule, directory, **kwargs):
    assert P.active_package_executor() is not None
    assert kwargs['_defer_result_publication']
    assert set(kwargs['oracle_adapters']) == {'L2','L3'}
    root=Path(kwargs['runs_root']);name=capsule['name']
    path=root/'runs'/'synthetic'/name/'capsule_result.json'
    assert not path.exists(), 'unverified result was published'
    assert not path.with_name('run_manifest.yaml').exists()
    if capsule.get('barrier'):
        barrier=root/'barrier';barrier.mkdir(exist_ok=True)
        (barrier/name).write_text(str(os.getpid()))
        deadline=time.monotonic()+5
        while len(list(barrier.iterdir())) < 2:
            assert time.monotonic()<deadline, 'children were serialized'
            time.sleep(.01)
    paths=make_run_paths(root,name,suite='synthetic',target=capsule['target'],dtype='i8',benchmark=name)
    tiers={'L2':CR.TierResult('L2','pass',True),'L3':CR.TierResult('L3','pass',True,derived_from_rtl=True)}
    if capsule.get('missing_L3'):tiers.pop('L3')
    status='declined' if capsule.get('state_decline') else 'pass'
    if status=='declined':tiers={}
    row=CR._finalize_capsule_result(name=name,capsule=capsule,status=status,failure=None,tiers=tiers,
        trace_check_res={'status':'skipped','violations':[]},numeric={'status':'pass'},required={'L2','L3'},
        no_oracle=False,eff_target=capsule['target'],paths=paths,run_id=name,
        cfg=SimpleNamespace(target=capsule['target'],suite='synthetic',rtl_tiers={'L3'},oracle_tiers=('L2','L3')),
        result_filename='.operator_child_result.json',
        declined={'reason':'synthetic declared refusal'} if status=='declined' else None)
    assert not path.exists()
    assert not path.with_name('run_manifest.yaml').exists()
    row['synthetic_transport_control_only']=True
    return row
CR.run_capsule=grade
raise SystemExit(OC.main(sys.argv[1:]))
"""


@pytest.fixture
def setup(original, monkeypatch):  # noqa: F811 -- imported pytest fixture, shared with model-child controls
    monkeypatch.setattr(CR, "_CHILD_PREAMBLE", CR._CHILD_GUARD + CONTROL)
    monkeypatch.setattr(CR, "_config_for_target", lambda *args: SimpleNamespace(suite="synthetic"))
    monkeypatch.delenv("MERLIN_CERTIFY_BUDGET_S", raising=False)
    adapters = {"L2": CR.simulator_adapter("spike", original.target.target),
                "L3": CR.simulator_adapter("gsim", original.target.target)}
    original.adapters = adapters
    original.runs = original.private / "runs"
    original.runs.mkdir()
    popen = subprocess.Popen

    def launch(*args, **kwargs):
        process = popen(*args, **kwargs)
        communicate = process.communicate

        def checked(*values, **options):
            result = communicate(*values, **options)
            if process.returncode:
                print(result[1])  # This fixture launches synthetic authority controls only.
            return result

        process.communicate = checked
        return process

    monkeypatch.setattr(OC.subprocess, "Popen", launch)
    return original


def grade(setup, name, **extra):
    capsule = {"name": name, "kind": "isa", "label": "public", "target": setup.target.target,
               "required_oracle_tiers": ["L2", "L3"], **extra}
    return OC.grade(capsule, setup.candidate, runs_root=setup.runs, contract=None, target=setup.target.target,
        suite="synthetic", dtype="i8", adapters=setup.adapters, timeout=10, workers=2, no_oracle=False)


def test_two_fresh_children_overlap_and_publish_only_verified_required_tiers(setup):
    with _scope(setup):
        OC.validate(target=setup.target.target, adapters=setup.adapters, config=None, perf_extractor=None)
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(copy_context().run, grade, setup, name, barrier=True)
                       for name in ("first", "second")]
            rows = [future.result() for future in futures]
    assert len({row["operator_grade_process"]["pid"] for row in rows}) == 2
    assert all(row["original_compiler_child"]["schema"] == OC.CHILD.SCHEMA for row in rows)
    assert all(row["status"] == "pass" and set(row["tiers"]) == {"L2", "L3"} for row in rows)
    for row in rows:
        path = setup.runs / "runs" / "synthetic" / row["capsule"] / "capsule_result.json"
        assert json.loads(path.read_text()) == row


def test_missing_required_L3_is_incomplete_in_the_same_child_ladder(setup):
    with _scope(setup):
        row = grade(setup, "screen_only", missing_L3=True)
    assert row["status"] == "incomplete"
    assert row["failure"]["category"] == "NOT_RUN_IS_NOT_PASS"


def test_expected_refusal_is_preserved_without_invented_native_tiers(setup):
    with _scope(setup):
        row = grade(setup, "must_refuse", state_decline=True, expected={
            "outcome": "refuse", "refusal": {"reason": "unsupported declared operation", "evidence": "fixture"},
        })
    assert row["status"] == "pass" and row["expected_refusal"]["status"] == "met"
    assert row["tiers"] == {} and row["declined"]["reason"] == "synthetic declared refusal"


def test_suite_dispatch_preserves_case_completion_callback(setup, monkeypatch):
    monkeypatch.setenv(OC.ENV, "1")
    monkeypatch.setattr(CR, "load_package", lambda *args, **kwargs: object())
    monkeypatch.setattr(CR, "integrity_scan", lambda pkg: None)
    monkeypatch.setattr(CR, "build_package", lambda pkg: None)
    monkeypatch.setattr(CR, "_split_ineligible", lambda caps, target: (caps, []))
    invoked, completed = [], []

    def child(cap, package, **kwargs):
        invoked.append((cap["name"], kwargs["adapters"], kwargs["workers"]))
        return {"capsule": cap["name"], "status": "pass"}

    monkeypatch.setattr(OC, "grade", child)
    with _scope(setup):
        rows = CR._run_suite([{"name": "operator", "kind": "isa"}], setup.candidate,
            target=setup.target.target, runs_root=setup.runs, oracle_adapters=setup.adapters,
            max_workers=2, on_capsule_complete=completed.append)
    assert invoked == [("operator", setup.adapters, 1)]
    assert rows == completed == [{"capsule": "operator", "status": "pass"}]


def test_source_change_before_child_restoration_never_publishes(setup, monkeypatch):
    original_popen = subprocess.Popen

    def changed(*args, **kwargs):
        (setup.candidate / "tool.py").write_text("source changed before child restoration\n")
        return original_popen(*args, **kwargs)

    monkeypatch.setattr(OC.subprocess, "Popen", changed)
    with _scope(setup), pytest.raises(StageGateError, match="without a verified completion"):
        grade(setup, "changed")
    assert not list(setup.runs.glob("runs/*/*/capsule_result.json"))


def test_foreign_owner_and_forged_adapter_marker_cannot_select_process_grading(setup):
    with pytest.raises(StageGateError, match="original live profile"):
        OC.validate(target=setup.target.target, adapters=setup.adapters, config=None, perf_extractor=None)
    def forged(*args):
        return None
    forged._merlin_simulator_engine = "spike"
    with _scope(setup):
        with pytest.raises(StageGateError, match="original selected simulator"):
            OC.adapter_selection({"L2": forged}, target=setup.target.target)


def test_shared_budget_and_arbitrary_configuration_are_refused(setup, monkeypatch):
    with _scope(setup):
        monkeypatch.setenv("MERLIN_CERTIFY_BUDGET_S", "30")
        with pytest.raises(StageGateError, match="unlimited"):
            OC.validate(target=setup.target.target, adapters=setup.adapters, config=None, perf_extractor=None)
        monkeypatch.delenv("MERLIN_CERTIFY_BUDGET_S")
        with pytest.raises(StageGateError, match="arbitrary configuration"):
            OC.validate(target=setup.target.target, adapters=setup.adapters, config=object(), perf_extractor=None)


def test_opt_in_is_explicit_and_invalid_selection_refuses(monkeypatch):
    monkeypatch.delenv(OC.ENV, raising=False)
    assert not OC.enabled()
    monkeypatch.setenv(OC.ENV, "1")
    assert OC.enabled()
    monkeypatch.setenv(OC.ENV, "15")
    with pytest.raises(ValueError, match="requires 0 or 1"):
        OC.enabled()
