"""Matched preparation preserves input bytes and one reservation lifecycle."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent


@pytest.fixture
def pair_module():
    spec = importlib.util.spec_from_file_location("fresh_pair_test", HERE / "fresh_campaign_pair.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) if not isinstance(value, str) else value)
    return path


@pytest.fixture
def plans(tmp_path, pair_module):
    common = tmp_path / "framework/inputs"
    source = write(common / "full-compiler/python/compiler.py", "# selected compiler source\n")
    manifest = write(
        common / "full-compiler/manifest.json",
        {
            "members": [dict(path="compiler.py", sha256=pair_module._hash(source))],
        },
    )
    ref = write(common / "public-references/reviewed/kernel.c", "/* reference only */\n")
    references = write(
        common / "public-references/reference-manifest.json",
        {
            "schema": "operator.allowed_reference_sources.v2",
            "review_id": "explicit-fixture",
            "origins": [dict(name="reviewed", files=[dict(destination="kernel.c", sha256=pair_module._hash(ref))])],
        },
    )
    inputs = write(common / "inputs-selection.json", {"source_revision": "a" * 40})
    library = write(common / "library/shared.py", "# same selected library\n")
    timing = write(common / "timing.json", {"model": "same-v90"})
    selection = write(common / "reference-selection.json", {"schema": "test-only"})
    result = []
    for label, cpus in (("raw", "2,3"), ("cca", "4,5")):
        run_id, arm = "study-fresh7" + label, tmp_path / label
        guidance = write(arm / "artifacts/author-guidance.md", "same author instructions\n")
        policy = write(
            arm / "artifacts/policy.json",
            {
                "duration_s": 16200,
                "final_model_s": 3600,
                "gsim": {"max_jobs": 2},
            },
        )
        contract = write(
            arm / "artifacts/contract.json",
            {
                "members": [dict(path="shared.py", sha256=pair_module._hash(library))],
            },
        )
        argv = [
            sys.executable,
            "-B",
            "-m",
            "selected.controller",
            "--run-id",
            run_id,
            "--sim-max-jobs",
            "2",
            "--codex-home-root",
            str(arm / "private" / ("codex-" + run_id)),
            "--operator-errata",
            str(guidance),
            "--joint-campaign",
            str(policy),
            "--compiler-library-contract",
            str(contract),
            "--compiler-library-root",
            str(library.parent),
            "--full-compiler-manifest",
            str(manifest),
            "--full-compiler-root",
            str(source.parent),
            "--max-wall-s",
            "16200",
            "--fresh-backend",
        ]
        mode = "off" if label == "raw" else "on"
        selected = pair_module._pin(selection)
        if mode == "on":
            argv += ["--reference-experiments", selected["path"], "--reference-experiments-sha256", selected["sha256"]]
        env = {
            "MERLIN_COMPUTE_CPUS": cpus,
            "MERLIN_CODEX_CPUS": cpus,
            "MERLIN_PREPARATION_CPUS": cpus,
            "MERLIN_COMPUTE_CPU_SLOTS": str(arm / "state/compute-slots"),
            "MERLIN_GSIM_SLOT_ROOT": str(arm / "state/gsim-slots"),
            "MERLIN_BUNDLE_CAS": str(arm / "state/bundle-cas"),
            "TMPDIR": str(arm / "operator"),
            "MERLIN_JOINT_PERFORMANCE_SELECTION": json.dumps(pair_module._pin(timing)),
        }
        env.update({key: "2" for key in pair_module._campaign().WORKER_KEYS})
        reserved = [
            sys.executable,
            str(HERE / "resource_reservation.py"),
            "--cpus",
            cpus,
            "--max-seconds",
            "18000",
            "--unit",
            "merlinphase1" + label,
            "--state",
            str(arm / "state/cpu-reservation.json"),
            "--log",
            str(arm / "artifacts/campaign.log"),
            "--",
            "/usr/bin/env",
            "-i",
            *(f"{k}={v}" for k, v in sorted(env.items())),
            *argv,
        ]
        plan = {
            "run_id": run_id,
            "reference_guidance": mode,
            "execution_started": False,
            "cwd": str(common.parent),
            "framework_repository": str(common.parent),
            "source_revision": "a" * 40,
            "source_release": str(common / "source-release"),
            "full_compiler_manifest_sha256": pair_module._hash(manifest),
            "treatment": "same-raw-inputs",
            "reference_manifest": pair_module._pin(references),
            "inputs_selection": pair_module._pin(inputs),
            "cpu_reservation": {"cpus": list(map(int, cpus.split(",")))},
            "argv": argv,
            "environment": env,
            "reserved_argv": reserved,
        }
        if mode == "on":
            plan["reference_experiments"] = selected
        result.append(write(arm / "artifacts/proposed-launch.json", plan))
    return result


def test_preparation_preserves_commands_and_one_outer_owner(pair_module, plans, tmp_path):
    record = pair_module.prepare_pair(*plans, output=tmp_path / "pair", unit="merlinphase1pairtest")
    assert record["execution_started"] is False
    assert record["cpus"] == [2, 3, 4, 5]
    outer = record["outer_argv"]
    assert outer[1] == str(HERE / "resource_reservation.py")
    assert outer[outer.index("--cpus") + 1] == "2,3,4,5"
    assert outer.count("--unit") == 1
    for arm in record["arms"]:
        text = Path(arm["script"]["path"]).read_text()
        assert "resource_reservation.py" not in text
        assert "ulimit -n 65536" in text
        assert "timeout -s TERM -k 60 18000 taskset -c " + ",".join(map(str, arm["cpus"])) in text
        assert json.loads(plans[arm["reference_guidance"] == "on"].read_text())["run_id"] in text
    assert any(pin["path"].endswith("reviewed/kernel.c") for pin in record["artifact_pins"])
    assert any(pin["path"].endswith("python/compiler.py") for pin in record["artifact_pins"])
    subprocess.run(["/bin/bash", "-n", str(tmp_path / "pair/run-pair.sh")], check=True)


@pytest.mark.parametrize(
    "change,match",
    [
        (lambda p: p.update(reference_guidance="off"), "off and on"),
        (lambda p: p.update(reference_manifest={"path": "/different", "sha256": "f" * 64}), "reference_manifest"),
        (lambda p: p.update(inputs_selection={"path": "/different", "sha256": "f" * 64}), "inputs_selection"),
        (lambda p: p["environment"].update(TMPDIR="/shared"), "environment"),
        (lambda p: p["argv"].extend(["--seed-submission", "/inherited"]), "argv"),
    ],
)
def test_plan_drift_refused_before_output(pair_module, plans, tmp_path, change, match):
    value = json.loads(plans[1].read_text())
    change(value)
    plans[1].write_text(json.dumps(value))
    with pytest.raises(ValueError, match=match):
        pair_module.prepare_pair(*plans, output=tmp_path / "refused", unit="merlinphase1pairtest")
    assert not (tmp_path / "refused").exists()


@pytest.mark.parametrize(
    "member", ["public-references/reviewed/kernel.c", "full-compiler/python/compiler.py", "library/shared.py"]
)
def test_selected_member_bytes_reopened(pair_module, plans, tmp_path, member):
    (tmp_path / "framework/inputs" / member).write_text("changed after preparation\n")
    with pytest.raises(ValueError, match="input member changed"):
        pair_module.prepare_pair(*plans, output=tmp_path / "refused", unit="merlinphase1pairtest")


@pytest.mark.parametrize(
    "kind,match",
    [
        ("cpu_overlap", "disjoint"),
        ("state_overlap", "namespaces overlap"),
        ("worker_overcommit", "CPU allocation"),
        ("changed_clock", "paired file bytes differ"),
        ("changed_seed", "empty backend origins"),
    ],
)
def test_effective_runtime_mismatch_refused(pair_module, plans, tmp_path, kind, match):
    raw, cca = [json.loads(path.read_text()) for path in plans]
    if kind == "cpu_overlap":
        for key in ("MERLIN_COMPUTE_CPUS", "MERLIN_CODEX_CPUS", "MERLIN_PREPARATION_CPUS"):
            cca["environment"][key] = "3,4"
        cca["cpu_reservation"]["cpus"] = [3, 4]
    elif kind == "state_overlap":
        cca["environment"]["MERLIN_BUNDLE_CAS"] = raw["environment"]["MERLIN_BUNDLE_CAS"]
    elif kind == "worker_overcommit":
        cca["environment"]["MERLIN_PRIVATE_EXECUTION_WORKERS"] = "3"
    elif kind == "changed_clock":
        args = cca["argv"]
        policy = Path(args[args.index("--joint-campaign") + 1])
        value = json.loads(policy.read_text())
        value["duration_s"] = 16201
        policy.write_text(json.dumps(value))
    else:
        cca["argv"] += ["--seed-submission", "/prior-answer"]
    wrapper = cca["reserved_argv"][: cca["reserved_argv"].index("--") + 1]
    cca["reserved_argv"] = (
        wrapper
        + ["/usr/bin/env", "-i"]
        + [f"{key}={value}" for key, value in sorted(cca["environment"].items())]
        + cca["argv"]
    )
    plans[1].write_text(json.dumps(cca))
    with pytest.raises(ValueError, match=match):
        pair_module.prepare_pair(*plans, output=tmp_path / "refused", unit="merlinphase1pairtest")
    assert not (tmp_path / "refused").exists()


def test_launch_requires_confirmation_and_rechecks_scripts(pair_module, plans, tmp_path, monkeypatch):
    pair_module.prepare_pair(*plans, output=tmp_path / "pair", unit="merlinphase1pairtest")
    receipt = tmp_path / "pair/pair.json"
    argv = ["pair", "launch", "--pair", str(receipt), "--sha256", pair_module._hash(receipt)]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("no launch authorized"))
    assert pair_module.main() == 2
    (tmp_path / "pair/run-raw.sh").write_text("changed\n")
    with pytest.raises(ValueError, match="input changed"):
        pair_module.main()


def test_supervisor_reaps_failed_arms_process_groups(pair_module, tmp_path):
    """Exercise bounded local processes; never invoke a campaign or reservation."""
    child = tmp_path / "child.pid"
    marker = tmp_path / "sibling-survived"
    raw = write(tmp_path / "raw.sh", f"#!/bin/bash\nsleep 30 &\necho $! > {child}\nsleep 0.2\nexit 7\n")
    cca = write(tmp_path / "cca.sh", f"#!/bin/bash\nsleep 3\ntouch {marker}\n")
    script = write(
        tmp_path / "pair.sh", pair_module._supervisor([raw, cca], [tmp_path / "raw.log", tmp_path / "cca.log"])
    )
    completed = subprocess.run(["/bin/bash", str(script)], capture_output=True, text=True, timeout=5)
    assert completed.returncode == 7
    assert not marker.exists()
    pid = int(child.read_text())
    try:
        state = Path(f"/proc/{pid}/stat").read_text().split()[2]
    except FileNotFoundError:
        state = None
    assert state in (None, "Z"), "the failed arm's child must no longer execute"


def test_supervisor_waits_for_both_successful_arms(pair_module, tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    raw = write(tmp_path / "raw.sh", f"#!/bin/bash\ntouch {first}\n")
    cca = write(tmp_path / "cca.sh", f"#!/bin/bash\nsleep 0.1\ntouch {second}\n")
    script = write(
        tmp_path / "pair.sh", pair_module._supervisor([raw, cca], [tmp_path / "raw.log", tmp_path / "cca.log"])
    )
    completed = subprocess.run(["/bin/bash", str(script)], capture_output=True, text=True, timeout=5)
    assert completed.returncode == 0
    assert first.exists() and second.exists()


def _rewrite(plan):
    wrapper = plan["reserved_argv"][: plan["reserved_argv"].index("--") + 1]
    plan["reserved_argv"] = (
        wrapper
        + ["/usr/bin/env", "-i"]
        + [f"{key}={value}" for key, value in sorted(plan["environment"].items())]
        + plan["argv"]
    )


def _seed_review(plan):
    seed, origin = plan["seed"], plan["origin"]
    review = {**origin, "schema": "operator.joint_seed_review.v1", "approved": True}
    review.pop("review", None)
    review["derived_from"] = {
        **seed["from"], "digest": seed["from_digest"], "stripped": seed["stripped"],
        "run": seed["run"]["run_id"],
    }
    path = Path(plan["argv"][plan["argv"].index("--joint-campaign") + 1])
    origin["review"] = {"path": str(path.with_name("seed-review.json"))}
    write(Path(origin["review"]["path"]), review)
    import hashlib

    origin["review"]["sha256"] = hashlib.sha256(Path(origin["review"]["path"]).read_bytes()).hexdigest()
    policy = json.loads(path.read_text())
    policy["compiler_origin"] = origin
    write(path, policy)
    _rewrite(plan)


@pytest.fixture
def seeded_plans(plans, tmp_path, pair_module):
    from merlin.common.tree_hash import hash_tree

    original = tmp_path / "original-seed"
    write(original / "backend.py", "pass\n")
    addendum = pair_module._pin(write(tmp_path / "seed-guidance.md", "continue the graded seed\n"))
    run = {"run_id": "previous-run", "run_dir": str(tmp_path / "previous/run"),
           "workspace": str(tmp_path / "previous/workspace"), "framework": str(tmp_path / "previous/export")}
    for path in plans:
        plan = json.loads(path.read_text())
        out = path.parent
        source = out / "seed-compiler"
        write(source / "backend.py", "pass\n")
        evidence = out / "seed-evidence"
        write(evidence / "RULES.md", "keep the successful prior rules\n")
        epin = {"path": str(evidence), "sha256": hash_tree(evidence)["sha256"]}
        origin = {
            "schema": "merlin.joint_reviewed_seed.v1", "review_id": plan["run_id"],
            "source": str(source), "sha256": hash_tree(source)["sha256"],
            "inherit_clock": False, "inherit_results": False, "variants": [],
            "inherit_certificates_for_identical_executables": True,
            "previous_session": {k: run[k] for k in ("run_dir", "workspace")}, "seed_evidence": epin,
        }
        plan["origin"] = origin
        plan["seed"] = {
            "from": {"path": str(original), "sha256": hash_tree(original)["sha256"]},
            "from_digest": "hash_tree", "stripped": [], "source": str(source),
            "graded_source_sha256": origin["sha256"],
            "carry": dict.fromkeys(("certificates", "results", "rules", "handoff"), True),
            "run": run, "evidence": {"pin": epin}, "guidance_addendum": addendum,
            "certificates": {"judge": {"identical": True, "changed": [], "missing": []}},
        }
        plan["argv"].remove("--fresh-backend")
        plan["argv"] += ["--seed-submission", str(source), "--sandbox", "bwrap"]
        plan["environment"].update(MERLIN_TIER_CERT_PORTABLE="1", MERLIN_SEEDED_GRADE=run["run_id"])
        _seed_review(plan)
        write(path, plan)
    return plans


def test_matched_seeded_cca_pair_preserves_seed_and_common_baseline(pair_module, seeded_plans, tmp_path):
    before = [json.loads(path.read_text()) for path in seeded_plans]
    record = pair_module.prepare_pair(*seeded_plans, output=tmp_path / "pair", unit="merlinphase1seedpair")
    assert record["execution_started"] is False
    assert record["cpus"] == [2, 3, 4, 5]
    for arm, original in zip(record["arms"], before, strict=True):
        text = Path(arm["script"]["path"]).read_text()
        assert "--seed-submission " + original["seed"]["source"] in text
        assert "--fresh-backend" not in text
        assert Path(original["seed"]["source"], "backend.py").read_text() == "pass\n"
    assert [json.loads(path.read_text()) for path in seeded_plans] == before
    pins = {pin["path"] for pin in record["artifact_pins"]}
    assert all(plan["origin"]["review"]["path"] in pins for plan in before)


@pytest.mark.parametrize("drift", ("mixed_origin", "from", "source", "carry", "evidence", "judge", "hidden_policy"))
def test_seeded_pair_rejects_matched_claim_with_different_effective_inputs(
    pair_module, seeded_plans, tmp_path, drift,
):
    from merlin.common.tree_hash import hash_tree

    path = seeded_plans[1]
    plan = json.loads(path.read_text())
    if drift == "mixed_origin":
        plan.pop("seed")
    elif drift == "from":
        other = tmp_path / "different-original"
        write(other / "backend.py", "different seed\n")
        plan["seed"]["from"] = {"path": str(other), "sha256": hash_tree(other)["sha256"]}
    elif drift == "source":
        source = Path(plan["seed"]["source"])
        write(source / "backend.py", "different staged program\n")
        plan["seed"]["graded_source_sha256"] = plan["origin"]["sha256"] = hash_tree(source)["sha256"]
    elif drift == "carry":
        plan["seed"]["carry"]["rules"] = False
    elif drift == "evidence":
        evidence = Path(plan["origin"]["seed_evidence"]["path"])
        write(evidence / "RULES.md", "different prior evidence\n")
        plan["origin"]["seed_evidence"]["sha256"] = hash_tree(evidence)["sha256"]
        plan["seed"]["evidence"]["pin"] = dict(plan["origin"]["seed_evidence"])
    elif drift == "judge":
        plan["seed"]["certificates"]["judge"]["missing"] = ["different-grader.py"]
    else:
        policy_path = Path(plan["argv"][plan["argv"].index("--joint-campaign") + 1])
        policy = json.loads(policy_path.read_text())
        policy["hidden_cases"] = ["different-test-panel"]
        write(policy_path, policy)
    if drift != "mixed_origin":
        _seed_review(plan)
    write(path, plan)
    mismatch = "mix fresh and seeded|paired seed source, carry policy, evidence or effective judge"
    with pytest.raises(ValueError, match=mismatch):
        pair_module.prepare_pair(*seeded_plans, output=tmp_path / "refused", unit="merlinphase1seedpair")
    assert not (tmp_path / "refused").exists()


def test_seeded_pair_reopens_source_before_a_confirmed_launch(pair_module, seeded_plans, tmp_path, monkeypatch):
    pair_module.prepare_pair(*seeded_plans, output=tmp_path / "pair", unit="merlinphase1seedpair")
    receipt = tmp_path / "pair/pair.json"
    plan = json.loads(seeded_plans[1].read_text())
    write(Path(plan["seed"]["source"]) / "backend.py", "changed after preparation\n")
    monkeypatch.setattr(sys, "argv", ["pair", "launch", "--pair", str(receipt), "--sha256",
                                     pair_module._hash(receipt), "--confirm"])
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("changed seed must not launch"))
    with pytest.raises(ValueError, match="seed source bytes changed"):
        pair_module.main()


@pytest.mark.parametrize("variant", ("same", "different", "raw_only", "cca_only"))
def test_guided_map_may_be_extra_but_shared_map_bytes_must_match(pair_module, plans, tmp_path, variant):
    pins = []
    for index, label in enumerate(("raw", "cca")):
        body = {"map": "same"} if variant != "different" or index == 0 else {"map": "another"}
        pins.append(pair_module._pin(write(tmp_path / label / "artifacts/REFERENCE_MAP.json", body)))
    for index, path in enumerate(plans):
        plan = json.loads(path.read_text())
        if (variant != "raw_only" or index == 0) and (variant != "cca_only" or index == 1):
            plan["environment"]["MERLIN_REFERENCE_MAP"] = json.dumps(pins[index])
            plan["reference_map"] = pins[index]
        _rewrite(plan)
        path.write_text(json.dumps(plan))
    if variant in {"same", "cca_only"}:
        record = pair_module.prepare_pair(*plans, output=tmp_path / "pair", unit="merlinphase1pairtest")
        expected = pins if variant == "same" else pins[1:]
        assert {pin["path"] for pin in expected} <= {pin["path"] for pin in record["artifact_pins"]}
        if variant == "cca_only":
            assert pins[0]["path"] not in {pin["path"] for pin in record["artifact_pins"]}
        return
    with pytest.raises(ValueError, match="reference maps differ"):
        pair_module.prepare_pair(*plans, output=tmp_path / "refused", unit="merlinphase1pairtest")


def test_guided_map_bytes_are_reopened_and_runtime_requires_a_receipt(pair_module, plans, tmp_path):
    cca = json.loads(plans[1].read_text())
    path = write(tmp_path / "cca/artifacts/REFERENCE_MAP.json", {"map": "selected"})
    cca["reference_map"] = pair_module._pin(path)
    cca["environment"]["MERLIN_REFERENCE_MAP"] = json.dumps(cca["reference_map"])
    _rewrite(cca)
    plans[1].write_text(json.dumps(cca))
    path.write_text(json.dumps({"map": "changed"}))
    with pytest.raises(ValueError, match="selected file changed"):
        pair_module.prepare_pair(*plans, output=tmp_path / "changed", unit="merlinphase1pairtest")
    path.write_text(json.dumps({"map": "selected"}))
    cca.pop("reference_map")
    cca["environment"].pop("MERLIN_REFERENCE_MAP")  # Leave the unrecorded pin in the actual wrapper.
    plans[1].write_text(json.dumps(cca))
    with pytest.raises(ValueError, match="reference map differs from its plan receipt"):
        pair_module.prepare_pair(*plans, output=tmp_path / "unrecorded", unit="merlinphase1pairtest")


def test_guided_arm_may_withhold_tasks_only_with_its_receipt(pair_module, plans, tmp_path):
    cca = json.loads(plans[1].read_text())
    cca["argv"] += ["--reference-experiments-tasks", "off"]
    _rewrite(cca)
    plans[1].write_text(json.dumps(cca))
    with pytest.raises(ValueError, match="task serving"):
        pair_module.prepare_pair(*plans, output=tmp_path / "refused", unit="merlinphase1pairtest")
    cca["reference_experiment_tasks"] = "off"
    plans[1].write_text(json.dumps(cca))
    pair_module.prepare_pair(*plans, output=tmp_path / "pair", unit="merlinphase1pairtest")
    raw = json.loads(plans[0].read_text())
    raw["argv"] += ["--reference-experiments-tasks", "off"]
    raw["reference_experiment_tasks"] = "off"
    _rewrite(raw)
    plans[0].write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="task serving"):
        pair_module.prepare_pair(*plans, output=tmp_path / "refused2", unit="merlinphase1pairtest")
