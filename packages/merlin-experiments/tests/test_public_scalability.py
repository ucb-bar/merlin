"""A separate required functional suite cannot acquire core/RTL credit."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from merlin_experiments.corpus import public_scalability as S
from merlin_experiments.phase1.feedback import public_scalability as F
from merlin_experiments.runner import fingerprint
from merlin_experiments.spec import SpecError
from test_corpus_release import _member
from test_corpus_release import release_fixture as _release_fixture

from merlin.common.digest import sha256_file
from merlin.targetgen import capsule_grade as CG
from merlin.targetgen.target_experiment import load_target_experiment


def selection(tmp_path, *, count=2):
    source = tmp_path / "selected"
    rows = []
    for n in range(count):
        name = f"case{n}"
        _member(source, "layers", name, "public")
        directory = source / "layers" / name
        p = directory / "capsule.yaml"
        cap = yaml.safe_load(p.read_text())
        cap["required_oracle_tiers"] = ["L0", "L1", "L2"]
        p.write_text(yaml.safe_dump(cap))
        rows.append(
            {"member": f"layers/{name}", "sha256": fingerprint(directory), "source_group": "a" * 64, "split": "train"}
        )
    provenance = tmp_path / "provenance.json"
    provenance.write_text('{"status":"synthetic_explicit_public_input"}\n')
    path = source / "suite.json"
    path.write_text(
        json.dumps(
            {
                "schema": S.SCHEMA,
                "target": "fixture-device",
                "origin": "operator_requested",
                "policy": S.POLICY,
                "members": rows,
                "provenance": {"path": str(provenance), "sha256": sha256_file(provenance)},
            }
        )
    )
    return path


def staged(tmp_path):
    source = selection(tmp_path)
    descriptor = tmp_path / "descriptor.yaml"
    descriptor.write_text("target: fixture-device\nsim_via: chipyard\n")
    te = load_target_experiment(descriptor)
    root = tmp_path / "frozen"
    binding = S.stage(source, root, te=te)
    doc = yaml.safe_load(descriptor.read_text())
    doc["workload_spec"] = {S.KEY: binding}
    descriptor.write_text(yaml.safe_dump(doc))
    te = load_target_experiment(descriptor)
    context = SimpleNamespace(descriptor=descriptor, target=te.target, repo=tmp_path)
    return root, S.verify(root, target=te.target), te, context


def score(names=("case0", "case1")):
    n = len(names)
    return {
        "gradeable": True,
        "n_capsules": n,
        "n_passed": n,
        "cohort_admission": {"n_admitted_capsules": n, "admitted_name_set_sha256": CG._name_set_sha256(names)},
        "per_capsule": [{"capsule": name, "status": "pass"} for name in names],
        "numeric_all_exact": True,
        "structural_evidence_all_pass": True,
        "tier_reached": {t: n for t in S.POLICY["tiers"]},
    }


def observations(names=("case0", "case1")):
    return {
        name: {
            "capsule": name,
            "status": "pass",
            "tiers": {
                "L0": {"status": "pass"},
                "L1": {"status": "pass"},
                "L2": {"status": "pass", "engine": "spike", "fidelity": "functional_model", "derived_from_rtl": False},
            },
        }
        for name in names
    }


def test_emission_error_survives_scalability_projection_without_private_receipt():
    row = {
        "capsule": "connected",
        "status": "fail",
        "tiers": {"L1": {"status": "fail"}},
        "failure": {"detail": "whole-model proof unavailable"},
        "candidate_emission_failure": {
            "detail": "no host lane declared for dtype 'fp32'; target declares ['int8_w8a8']",
            "private_receipt": {"golden": [1234567]},
        },
        "private_reference": [7654321],
    }
    observed = F._observed_result(row)
    result = F.public_result({"observed": {"connected": observed}, "score": {}})
    assert result["cases"][0]["failure_category"] == "CANDIDATE_EMISSION_FAILED"
    assert "no host lane declared" in result["cases"][0]["failure_detail"]
    assert result["cases"][0]["status"] == "fail"
    assert result["cases"][0]["tiers"] == {"L1": "fail"}
    assert "private" not in json.dumps(result)
    assert "1234567" not in json.dumps(result)
    assert "7654321" not in json.dumps(result)
    assert row["failure"]["detail"] == "whole-model proof unavailable"


def test_scalability_projection_preserves_pass_and_bounds_compiler_error():
    row = {"capsule": "case", "status": "pass", "failure": None,
           "candidate_emission_failure": "stale diagnostic"}
    assert F._observed_result(row)["failure"] is None
    row.update(status="fail", candidate_emission_failure="x" * 10000)
    assert len(F._observed_result(row)["failure"]["detail"]) <= 2000


def test_explicit_host_control_is_source_bound_and_not_accelerator_credit(tmp_path):
    path = selection(tmp_path, count=1)
    directory = path.parent / "layers/case0"
    cp = directory / "capsule.yaml"
    cap = yaml.safe_load(cp.read_text())
    cap["kind"] = "model_slice"
    cap["semantic"] = {"must_accelerate": False}
    cap["software_screen"] = {
        "status": "admitted", "program_sha256": sha256_file(directory / cap["interface_mlir"]),
        "decisions": [{"status": "admitted", "placement": "host"}],
    }
    cp.write_text(yaml.safe_dump(cap))
    doc = json.loads(path.read_text())
    doc["members"][0].update(execution_role=S.HOST_CONTROL, sha256=fingerprint(directory))
    path.write_text(json.dumps(doc))
    record = S.read(path, target="fixture-device")
    assert record["members"]["layers/case0"]["execution_role"] == S.HOST_CONTROL

    cap["semantic"]["must_accelerate"] = True
    cp.write_text(yaml.safe_dump(cap))
    doc["members"][0]["sha256"] = fingerprint(directory)
    path.write_text(json.dumps(doc))
    with pytest.raises(SpecError, match="exclusive Merlin host placement"):
        S.read(path, target="fixture-device")

    cap["semantic"]["must_accelerate"] = False
    cap["software_screen"]["program_sha256"] = "0" * 64
    cp.write_text(yaml.safe_dump(cap))
    doc["members"][0]["sha256"] = fingerprint(directory)
    path.write_text(json.dumps(doc))
    with pytest.raises(SpecError, match="source-bound"):
        S.read(path, target="fixture-device")


@pytest.mark.parametrize(
    "sim,tiers,inspection",
    [
        ("gsim", "", False),
        ("verilator", "", False),
        ("spike", "L0,L1,L2,L3", False),
        ("spike", "L0,L0", False),
        ("spike", "", True),
        ("unknown", "", False),
    ],
)
def test_server_refuses_wrong_engine_tiers_and_probes(sim, tiers, inspection):
    with pytest.raises(ValueError):
        F.request_policy(sim, tiers, inspection=inspection)


def test_server_default_and_legacy_absence(tmp_path, monkeypatch):
    assert F.request_policy() == ("spike", ["L0", "L1", "L2"])
    descriptor = tmp_path / "descriptor.yaml"
    descriptor.write_text("target: fixture-device\n")
    context = SimpleNamespace(descriptor=descriptor)
    verdict = {"all_pass": True}
    F.attach(verdict, tmp_path, context=context, output=tmp_path / "unused", timeout=1)
    assert verdict == {"all_pass": True}
    assert F.verify_issued(tmp_path / "absent", descriptor=descriptor, candidate=tmp_path)


@pytest.mark.parametrize(
    "mutation", ["subset", "extra", "duplicate", "split", "rtl", "pending", "reference", "reference_as_source"]
)
def test_strict_selection_mutations(tmp_path, mutation):
    path = selection(tmp_path)
    doc = json.loads(path.read_text())
    if mutation == "subset":
        doc["members"].pop()
    if mutation == "extra":
        doc["members"][0]["unexpected"] = True
    if mutation == "duplicate":
        doc["members"].append(doc["members"][0])
    if mutation == "split":
        doc["members"][1]["split"] = "test"
    if mutation == "rtl":
        doc["policy"] = {**S.POLICY, "engine": "gsim"}
    if mutation == "pending":
        doc["origin"] = "pending_admission"
    if mutation == "reference":
        (path.parent / "layers/case0/golden.yaml").write_text("changed")
    if mutation == "reference_as_source":
        cap_path = path.parent / "layers/case0/capsule.yaml"
        cap = yaml.safe_load(cap_path.read_text())
        cap["interface_mlir"] = "golden.yaml"
        cap_path.write_text(yaml.safe_dump(cap))
        doc["members"][0]["sha256"] = fingerprint(cap_path.parent)
    path.write_text(json.dumps(doc))
    with pytest.raises((ValueError, SpecError)):
        S.read(path, target="fixture-device")


def test_native_staging_and_frozen_transport(tmp_path, monkeypatch):
    from merlin_experiments.phase1 import corpus_inputs as CI

    from merlin.common.paths import repo_root
    from merlin.targetgen.sandbox import bwrap as BW

    root, record, te, context = staged(tmp_path)
    assert list((root / "public_sources").glob("*.mlir"))
    assert not list((root / "public_sources").rglob("golden.yaml"))
    F.install(te, root)
    assert F.selected(context) == (root, record)
    run = tmp_path / "run"
    run.mkdir()
    core = tmp_path / "core/isa"
    _member(tmp_path / "core", "isa", "core", "public")
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "out"))
    monkeypatch.setenv("MERLIN_BUNDLE_CAS", "")
    bundle, frozen = CI.stage(
        run,
        te,
        {"bundle_id": "fixture", "allowed": [], "denied": []},
        contract=repo_root() / "merlin/contract",
        capsules_root=core,
    )
    ws = tmp_path / "ws"
    BW.materialize_bundle_inputs(ws, bundle, repo=tmp_path)
    view = CI.resolve(ws, bundle, frozen, repo=tmp_path)
    assert view.scalability and fingerprint(view.scalability) == fingerprint(root)
    F.install(te, view.scalability)
    assert F.selected(context)[0] == view.scalability
    (view.scalability / "cases/case0/golden.yaml").chmod(0o600)
    (view.scalability / "cases/case0/golden.yaml").write_text("tampered")
    with pytest.raises(ValueError):
        F.selected(context)


@pytest.mark.parametrize(
    "mutation", ["none", "missing", "wrong_engine", "unrun", "partial", "missing_values", "deferred"]
)
def test_completion_is_exact_required_roster_and_actual_functional_engine(mutation):
    s, observed = score(), observations()
    if mutation == "missing":
        observed.pop("case1")
    if mutation == "wrong_engine":
        observed["case0"]["tiers"]["L2"]["engine"] = "gsim"
    if mutation == "unrun":
        observed["case0"]["tiers"]["L2"]["status"] = "not_run"
    if mutation == "partial":
        s["n_passed"] = 1
    if mutation == "missing_values":
        s["numeric_all_exact"] = False
    if mutation == "deferred":
        s["n_gated_deferred"] = 1
    assert F.complete(s, observed, names={"case0", "case1"}) is (mutation == "none")


def test_actual_observed_call_and_publication_recheck(tmp_path, monkeypatch):
    from merlin_experiments.phase1.feedback import selfcheck

    root, record, te, context = staged(tmp_path)
    F.install(te, root)
    submission = tmp_path / "submission"
    submission.mkdir()
    (submission / "compiler.py").write_text('print("fixture")\n')

    def adapter(*a, **k):
        return None

    adapter._merlin_simulator_engine = "spike"
    monkeypatch.setattr(selfcheck, "_adapters", lambda sim, *a, **k: ({"L2": adapter}, sim))
    monkeypatch.setattr(F, "_readback", lambda _: {})
    calls = []

    def ordinary_grade(package, **kwargs):
        assert kwargs["oracle_adapters"] == {"L2": adapter}
        assert kwargs["timeout"] == 120
        assert kwargs["max_workers"] == 2
        assert not kwargs.get("functional_hidden_tier")
        names = sorted(p.name for p in Path(kwargs["capsules_root"]).iterdir())
        for row in observations(names).values():
            kwargs["on_capsule_complete"](row)
        calls.append(names)
        return score(names)

    monkeypatch.setattr(CG, "grade", ordinary_grade)
    result = F.grade(
        submission, root=root, record=record, context=context, output=tmp_path / "result", timeout=120, max_workers=2
    )
    assert result["complete"] and not result["rtl_qualified"] and not result["timing_qualified"]
    assert F.verify_issued(tmp_path / "result/issued.json", descriptor=context.descriptor, candidate=submission)
    assert calls == [["case0", "case1"]]
    import importlib.util

    from merlin.common import invocation_record as I

    foreign_file = tmp_path / "foreign_producer.py"
    foreign_file.write_text("def producer(): pass\n")
    spec = importlib.util.spec_from_file_location("foreign_producer", foreign_file)
    foreign = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(foreign)
    foreign.producer.__module__ = F.__name__
    foreign.producer.__qualname__ = "_grade"
    issued = tmp_path / "result/issued.json"
    original = issued.read_bytes()
    receipt = json.loads(original)
    original_call = I.verify(Path(receipt["issuer"]["path"]))
    with I.observe_call(
        tmp_path / "foreign",
        stage="public_spike_scalability",
        function=foreign.producer,
        arguments=original_call["arguments"],
        outputs=(tmp_path / "result/qualification.json",),
    ) as observed:
        foreign.producer()
        observed.returned()
    receipt["issuer"] = {"path": str(observed.path), "sha256": sha256_file(observed.path)}
    issued.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="complete original"):
        F.verify_issued(issued, descriptor=context.descriptor, candidate=submission)
    issued.write_bytes(original)
    (submission / "compiler.py").write_text("changed")
    with pytest.raises(ValueError):
        F.verify_issued(tmp_path / "result/issued.json", descriptor=context.descriptor, candidate=submission)


def test_stock_prepare_keeps_core_denominator_and_grants_only_math(suite_release_fixture, capsys):
    from merlin_experiments.cli import main
    from merlin_experiments.corpus import release

    f = suite_release_fixture
    path = selection(f["root"])
    assert main(["run", str(f["definition"]), "--phase", "0", "--run-dir", str(f["run"])]) == 0
    capsys.readouterr()
    report = release.prepare(f["run"], f["release"], generated_only=False, public_scalability=path)
    descriptor = Path(report["descriptor"])
    te = load_target_experiment(descriptor)
    root, _ = S.selected(te)
    assert len(list(te.graded_roots())) and len(list(root.joinpath("cases").iterdir())) == 2
    bundle = yaml.safe_load(
        next((descriptor.parent / "input_bundles").glob("*/input_bundle_manifest.yaml")).read_text()
    )
    paths = [v["path"] for v in bundle["allowed"]]
    assert str(root / "public_sources") in paths
    assert str(root) not in paths and str(root / "cases") not in paths
    assert release.inspect_release(f["release"])["review_digest"] == report["review_digest"]


@pytest.mark.parametrize("joint", [False, True])
def test_prompt_selected_and_custody_checked_without_changing_legacy(tmp_path, monkeypatch, joint):
    from merlin_experiments.phase1 import task_staging as T
    from merlin_experiments.phase1 import joint_control as J

    monkeypatch.setattr(J, "selected", lambda: {"policy": {
        "duration_s": 43200, "final_model_s": 10800, "drain_s": 1800,
    }} if joint else None)

    _, _, _, context = staged(tmp_path)
    ws, run = tmp_path / "workspace", tmp_path / "run"
    ws.mkdir()
    run.mkdir()
    config = SimpleNamespace(context=context, shared_pass_requirement=None, author_profile=None)

    def original(*args, **kwargs):
        (ws / "TASK.md").write_text("Original task body\n")
        (run / "TASK.md").write_text("Original task body\n")

    monkeypatch.setattr(T, "_build_task", original)
    T.build_task("fixture", ws, run, config=config)
    body = (ws / "TASK.md").read_text()
    assert body.startswith("Original task body\n") and "--suite scalability" in body
    assert "hidden and full-model validation remain separate" in body
    assert body.count("## Joint compiler development campaign") == int(joint)
    T.verify_shared_pass_prompt(config, ws, run)
    (ws / "TASK.md").write_text("Original task body\n")
    with pytest.raises(ValueError, match="scalability instructions"):
        T.verify_shared_pass_prompt(config, ws, run)


@pytest.mark.parametrize("sim", ["gsim", "verilator"])
def test_standalone_helper_rejects_rtl_before_any_request(tmp_path, sim):
    import os
    import shutil
    import subprocess
    import sys

    from merlin_experiments.phase1.tools import selfcheck

    helper = tmp_path / "agent_selfcheck.py"
    shutil.copyfile(selfcheck.__file__, helper)
    run = subprocess.run(
        [sys.executable, "-I", "-B", str(helper), "--suite", "scalability", "--sim", sim],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=5,
        env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
    )
    assert run.returncode == 2 and "only --sim spike" in run.stderr
    assert not (tmp_path / ".qa_channel").exists()


def test_broker_bound_suite_request_is_forwarded_to_actual_worker(tmp_path, monkeypatch):
    from merlin_experiments.phase1.feedback import selfcheck

    root, _, te, context = staged(tmp_path)
    F.install(te, root)
    args = [
        "--suite",
        "scalability",
        "--submission",
        str(tmp_path / "submission"),
        "--capsules",
        "case0",
        "--sim",
        "spike",
        "--out",
        str(tmp_path / "out.json"),
    ]
    monkeypatch.setattr(selfcheck, "resolve_context", lambda *a: context)
    observed = []
    monkeypatch.setattr(F, "cli", lambda a, **kw: observed.append((a.suite, a.capsules, kw["context"])) or 17)
    # Exercise the real decorated ordinary worker up to its explicitly selected suite owner.
    assert selfcheck.main.__wrapped__(args, context=context) == 17
    assert observed == [("scalability", "case0", context)]


def test_partial_pass_and_wrong_adapter_cannot_issue_full_completion(tmp_path, monkeypatch):
    from merlin_experiments.phase1.feedback import selfcheck

    root, record, _, context = staged(tmp_path)
    submission = tmp_path / "submission"
    submission.mkdir()
    (submission / "tool.py").write_text("fixture")

    def adapter(*a, **kw):
        return None

    adapter._merlin_simulator_engine = "gsim"
    monkeypatch.setattr(selfcheck, "_adapters", lambda *a, **kw: ({"L2": adapter}, "spike"))
    monkeypatch.setattr(F, "_readback", lambda _: {})
    with pytest.raises(ValueError, match="Spike-only"):
        F.grade(
            submission,
            root=root,
            record=record,
            context=context,
            output=tmp_path / "wrong",
            timeout=120,
            names={"case0"},
        )
    assert not (tmp_path / "wrong/issued.json").exists()
    adapter._merlin_simulator_engine = "spike"

    def small_grade(*a, **kw):
        for row in observations(("case0",)).values():
            kw["on_capsule_complete"](row)
        return score(("case0",))

    monkeypatch.setattr(CG, "grade", small_grade)
    result = F.grade(
        submission, root=root, record=record, context=context, output=tmp_path / "partial", timeout=120, names={"case0"}
    )
    assert result["complete"] is False
    with pytest.raises(ValueError, match="complete original"):
        F.verify_issued(tmp_path / "partial/issued.json", descriptor=context.descriptor, candidate=submission)


@pytest.fixture
def suite_release_fixture(tmp_path, monkeypatch):
    return _release_fixture.__wrapped__(tmp_path, monkeypatch)


def test_selected_suite_missing_producer_is_a_publication_failure(tmp_path, monkeypatch):
    from merlin_experiments.phase1.feedback import capsule_certification as C

    root, _, _, context = staged(tmp_path)
    run = tmp_path / "certificate"
    run.mkdir()
    submission = run / "submission"
    submission.mkdir()
    (submission / "tool.py").write_text("fixture")
    from merlin.common.tree_hash import hash_tree

    (run / "environment.yaml").write_text("validation_scope: capsules\n")
    (run / "freeze.json").write_text(json.dumps({"submission_sha256": hash_tree(submission)["sha256"]}))
    monkeypatch.setattr(C.HCP, "resolve", lambda *a: {})
    monkeypatch.setattr(C.HCP, "verify_record", lambda *a: None)

    def phase(run, name, **kwargs):
        path = run / f"{name}.json"
        path.write_text(json.dumps({"per_capsule": [{"capsule": name}]}))
        return {"score": C.pin(path)}

    monkeypatch.setattr(C, "_phase", phase)
    with pytest.raises(FileNotFoundError):
        C._evidence(run, context.descriptor, {}, create=True)
    assert not (run / "capsule_qualification.json").exists()


def test_host_source_alias_cannot_enter_public_suite(tmp_path):
    import os

    path = selection(tmp_path)
    private = tmp_path / "protected"
    private.mkdir()
    os.link(path.parent / "layers/case0/golden.yaml", private / "answer.yaml")
    descriptor = tmp_path / "descriptor.yaml"
    descriptor.write_text("target: fixture-device\n")
    with pytest.raises(RuntimeError, match="hardlink alias"):
        S.stage(path, tmp_path / "bad-stage", te=load_target_experiment(descriptor), private_roots=(private,))


def test_stock_prepare_refuses_alias_to_explicit_private_baseline(suite_release_fixture, capsys):
    import os

    from merlin_experiments.cli import main
    from merlin_experiments.corpus import release

    f = suite_release_fixture
    path = selection(f["root"])
    golden = path.parent / "layers/case0/golden.yaml"
    golden.unlink()
    import shutil

    external = f["root"] / "separate-private-baseline"
    shutil.copytree(f["baseline"] / "hidden", external)
    os.link(external / "private_member_identity/golden.yaml", golden)
    assert main(["run", str(f["definition"]), "--phase", "0", "--run-dir", str(f["run"])]) == 0
    capsys.readouterr()
    with pytest.raises(SpecError, match="corpus preparation failed"):
        release.prepare(f["run"], f["release"], generated_only=True, private_baseline=external, public_scalability=path)
    assert "hardlink alias" in (f["release"] / "private/failure.json").read_text()
    assert not (f["release"] / "private/seal.json").exists()


def test_actionable_failure_uses_original_answer_redaction():
    rows = observations(("case0",))
    rows["case0"].update(
        status="fail",
        failure={
            "plane": "compile",
            "category": "TIMEOUT",
            "detail": "tool timeout; expected output 12345 actual 67890 at /operator/private/golden.npz",
        },
    )
    raw = {"schema": "fixture", "score": {}, "observed": rows}
    public = F.public_result(raw)
    text = json.dumps(public)
    assert public["cases"][0]["failure_plane"] == "compile"
    assert public["cases"][0]["failure_category"] == "TIMEOUT"
    assert "timeout" in public["cases"][0]["failure_detail"]
    assert "12345" not in text and "67890" not in text and "/operator/private" not in text
    assert "observed" not in public and "score" not in public
