"""The fresh campaign entry point renders the reviewed spec exactly and launches nothing."""

import copy
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

HERE = Path(__file__).resolve().parent
FRESH2 = HERE / "fresh-campaign-fresh2.yaml"


@pytest.fixture
def F():
    spec = importlib.util.spec_from_file_location("fresh_campaign_test", HERE / "fresh_campaign.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _git_repo(root: Path) -> str:
    root.mkdir(parents=True)

    def run(*args):
        return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True)

    run("init", "-q")
    (root / "README").write_text("fixture\n")
    run("add", "README")
    run("-c", "user.name=t", "-c", "user.email=t@example.invalid", "-c", "commit.gpgsign=false", "commit", "-qm", "x")
    return run("rev-parse", "HEAD").stdout.strip()


def _write(path: Path, value) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) if not isinstance(value, str) else value)
    return path


@pytest.fixture
def fixture_spec(tmp_path):
    """A complete small spec over local fixture inputs, derived from the fresh2 spec's shape."""
    revision = _git_repo(tmp_path / "repo")
    inputs = tmp_path / "framework/inputs"
    _write(inputs / "full-compiler/manifest.json", {"members": []})
    (inputs / "full-compiler/python").mkdir(parents=True)
    release = inputs / "source-release"
    for name in (
        "payload/experiment/target_experiment.yaml",
        "payload/experiment/input_bundles/b/input_bundle_manifest.yaml",
        "private/seal.json",
    ):
        _write(release / name, "{}")
    _write(inputs / "public-references/reference-manifest.json", {"members": [{"path": "upstream/a.py"}]})
    policy = {
        "version": 1,
        "duration_s": 43200,
        "final_model_s": 10800,
        "drain_s": 900,
        "compiler_execution_mode": "snapshot_external_v1",
        "gsim": {"max_jobs": 20, "per_job_s": 1200, "worker_budget_s": 864000},
        "compiler_origin": {"review_id": "seeded"},
    }
    base = _write(tmp_path / "base-policy.json", policy)
    _write(tmp_path / "scope.json", {"excluded": ["b_case", "a_case", "b_case"]})
    _write(tmp_path / "guidance.md", "guidance\n")
    _write(tmp_path / "timing.json", {"model": "selected"})
    spec = yaml.safe_load(FRESH2.read_text())
    spec["variables"] = {
        "TASK": str(tmp_path / "task"),
        "REPO": str(tmp_path / "repo"),
        "PYTHON": "/selected/python",
        "FRAMEWORK": str(tmp_path / "framework"),
        "INPUTS": str(inputs),
        "RELEASE": str(release),
        "ARTIFACTS": str(tmp_path / "launch-out"),
        "RUN_STATE": "${TASK}/.tmp/${RUN_ID}",
        "PRIVATE_ROOT": "${TASK}/private",
        "LAUNCH_MODULES": str(HERE),
    }
    spec["run"].update(run_id="fixture-run", unit="fixtureunit", cpus="3,4", max_wall_s=7200, final_model_s=7200)
    plan = spec["plan"]
    plan.update(
        library_selection=str(tmp_path / "library.json"),
        guidance=str(tmp_path / "guidance.md"),
        timing_selection=str(tmp_path / "timing.json"),
        policy={
            "base": {"path": str(base), "sha256": hashlib.sha256(base.read_bytes()).hexdigest()},
            "performance_scope": str(tmp_path / "scope.json"),
        },
    )
    plan["record"].pop("immutable_input_repository")
    plan["argv"] = [
        "${PYTHON}",
        "-m",
        "merlin_experiments.phase1",
        "--descriptor",
        "${RELEASE}/payload/experiment/target_experiment.yaml",
        "--bundle-manifest",
        "${RELEASE}/payload/experiment/input_bundles/b/input_bundle_manifest.yaml",
        "--corpus-seal",
        "${RELEASE}/private/seal.json",
        "--compiler-library-root",
        "${LIBRARY_ROOT}",
        "--compiler-library-contract",
        "${ARTIFACTS}/selected-library-contract.json",
        "--compiler-library-contract-sha256",
        "${LIBRARY_CONTRACT_SHA256}",
        "--codex-home-root",
        "${PRIVATE_ROOT}/codex-${RUN_ID}",
        "--max-wall-s",
        "${DURATION_S}",
        "--sim-max-jobs",
        "${SIM_MAX_JOBS}",
        "--operator-errata",
        "${ARTIFACTS}/author-guidance.md",
        "--run-id",
        "${RUN_ID}",
        "--joint-campaign",
        "${ARTIFACTS}/policy.json",
        "--full-compiler-root",
        "${INPUTS}/full-compiler/python",
        "--full-compiler-manifest",
        "${INPUTS}/full-compiler/manifest.json",
        "--full-compiler-manifest-sha256",
        "${FULL_COMPILER_MANIFEST_SHA256}",
        "--max-rate-limit-waits",
        "${RATE_LIMIT_WAITS}",
        "--fresh-backend",
        "--author-token-unbounded",
    ]
    plan["environment"] = {
        "MERLIN_COMPUTE_CPUS": "${CPUS}",
        "MERLIN_CODEX_CPUS": "${CPUS}",
        "MERLIN_PREPARATION_CPUS": "${CPUS}",
        "MERLIN_COMPUTE_CPU_SLOTS": "${RUN_STATE}/compute-slots",
        "MERLIN_GSIM_SLOT_ROOT": "${RUN_STATE}/gsim-slots",
        "MERLIN_GRADE_WORKERS": "20",
        "MERLIN_CAPSULE_ORDER": "information-first",
        "MERLIN_JOINT_PERFORMANCE_SELECTION": "${PERFORMANCE_SELECTION_PIN}",
        "MERLIN_BUNDLE_CAS": "${RUN_STATE}/bundle-cas",
        "HOME": "/selected/home",
    }
    path = tmp_path / "spec.yaml"
    path.write_text(yaml.safe_dump(spec, sort_keys=False))
    return SimpleNamespace(path=path, spec=spec, root=tmp_path, revision=revision)


@pytest.fixture
def library(F, monkeypatch, tmp_path):
    contract = SimpleNamespace(sha256="c" * 64, record=lambda: {"members": ["generic"]})
    (tmp_path / "library-root").mkdir()
    selected = SimpleNamespace(root=tmp_path / "library-root", contract=contract)
    monkeypatch.setattr(F, "restore_library", lambda path: selected)
    return selected


def _plan(F, fixture, **options):
    ctx = F.Context(fixture.path, export_id=fixture.revision[:12], **options.pop("context", {}))
    return ctx, F.build_plan(ctx, **options)


def test_variables_expand_override_and_fail_closed(F, fixture_spec, tmp_path):
    ctx = F.Context(fixture_spec.path, export_id="abc", overrides=["ARTIFACTS=/elsewhere"])
    assert ctx.x("${RUN_STATE}/x") == f"{tmp_path}/task/.tmp/fixture-run/x"
    assert ctx.x("${ARTIFACTS}") == "/elsewhere"
    assert ctx.x("${PREFLIGHT_RUN_ID}") == "fixture-run-preflight"
    with pytest.raises(F.SpecError, match="undefined"):
        ctx.x("${NOT_DECLARED}")
    with pytest.raises(F.SpecError, match="declared variable"):
        F.Context(fixture_spec.path, overrides=["NOT_DECLARED=1"])
    spec = copy.deepcopy(fixture_spec.spec)
    spec["variables"].update(A="${B}", B="${A}")
    cyclic = tmp_path / "cyclic.yaml"
    cyclic.write_text(yaml.safe_dump(spec))
    with pytest.raises(F.SpecError, match="cycle"):
        F.Context(cyclic).x("${A}")
    spec["variables"]["RUN_ID"] = "shadow"
    cyclic.write_text(yaml.safe_dump(spec))
    with pytest.raises(F.SpecError, match="built-in"):
        F.Context(cyclic)


def test_plan_renders_fresh_unseeded_launch(F, fixture_spec, library):
    ctx, record = _plan(F, fixture_spec)
    out = fixture_spec.root / "launch-out"
    policy = json.loads((out / "policy.json").read_text())
    assert "compiler_origin" not in policy
    assert (policy["duration_s"], policy["final_model_s"], policy["drain_s"]) == (7200, 5400, 900)
    assert policy["gsim"]["worker_budget_s"] == 7200 * 20  # the harness default for this clock
    assert policy["performance_scope"] == {"excluded": ["a_case", "b_case"]}
    argv = record["argv"]
    assert argv[-2:] == ["--fresh-backend", "--author-token-unbounded"]
    assert "--seed-submission" not in argv and "--author-helper-effort" not in argv
    assert argv[argv.index("--max-wall-s") + 1] == "7200"
    assert argv[argv.index("--compiler-library-contract-sha256") + 1] == "c" * 64
    env = record["environment"]
    assert env["MERLIN_CAPSULE_ORDER"] == "information-first"
    assert env["MERLIN_BUNDLE_CAS"] == str(fixture_spec.root / "task/.tmp/fixture-run/bundle-cas")
    pin = json.loads(env["MERLIN_JOINT_PERFORMANCE_SELECTION"])
    assert pin == record["timing_selection"] and Path(pin["path"]).parent == out
    reserved = record["reserved_argv"]
    wrapper = reserved[: reserved.index("--")]
    assert wrapper[1] == str(HERE / "resource_reservation.py")
    assert wrapper[wrapper.index("--cpus") + 1] == "3,4"
    assert wrapper[wrapper.index("--max-seconds") + 1] == str(7200 + 1800)
    assert wrapper[wrapper.index("--unit") + 1] == "fixtureunit"
    assert wrapper[wrapper.index("--log") + 1] == str(out / "campaign.log")
    inner = reserved[reserved.index("--") + 1 :]
    assert inner[:2] == ["/usr/bin/env", "-i"] and inner[-len(argv) :] == argv
    assert {item.split("=", 1)[0] for item in inner[2 : -len(argv)]} >= set(env)
    assert record["execution_started"] is False and record["source_revision"] == fixture_spec.revision
    assert record["cpu_reservation"]["cpus"] == [3, 4]
    assert record["launch_spec"]["sha256"] == hashlib.sha256(fixture_spec.path.read_bytes()).hexdigest()
    for key in ("MERLIN_COMPUTE_CPU_SLOTS", "MERLIN_GSIM_SLOT_ROOT"):
        assert (Path(env[key]).stat().st_mode & 0o777) == 0o700
    assert json.loads((out / "proposed-launch.json").read_text()) == record


def test_helper_effort_passes_through(F, fixture_spec, library):
    _, record = _plan(F, fixture_spec, helper_effort="low")
    assert record["argv"][-2:] == ["--author-helper-effort", "low"]
    spec = copy.deepcopy(fixture_spec.spec)
    spec["run"]["author_helper_effort"] = "medium"
    fixture_spec.path.write_text(yaml.safe_dump(spec))
    _, record = _plan(F, fixture_spec)
    assert record["argv"][-2:] == ["--author-helper-effort", "medium"]


def test_worker_cap_bounds_parallel_settings(F, fixture_spec, library):
    spec = copy.deepcopy(fixture_spec.spec)
    spec["run"]["workers"] = 6
    fixture_spec.path.write_text(yaml.safe_dump(spec))
    _, record = _plan(F, fixture_spec)
    assert record["environment"]["MERLIN_GRADE_WORKERS"] == "6"
    assert record["argv"][record["argv"].index("--sim-max-jobs") + 1] == "6"
    policy = json.loads((fixture_spec.root / "launch-out/policy.json").read_text())
    assert policy["gsim"]["max_jobs"] == 6


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda s, root: s["plan"]["argv"].extend(["--seed-submission", "x"]), "must not pass"),
        (lambda s, root: s["plan"]["argv"].remove("--fresh-backend"), "requires --fresh-backend"),
        (lambda s, root: s["plan"]["argv"].append(str(root / "seed-compiler")), "stale origin"),
        (lambda s, root: s["plan"]["environment"].update(MERLIN_CODEX_CPUS="1"), "reserved CPU list"),
        (
            lambda s, root: _write(
                root / "framework/inputs/public-references/reference-manifest.json",
                {"members": [{"path": "measured-exo/kernel.py"}]},
            ),
            "forbidden members",
        ),
    ],
)
def test_plan_refuses_unfresh_or_private_inputs(F, fixture_spec, library, mutate, message):
    spec = copy.deepcopy(fixture_spec.spec)
    mutate(spec, fixture_spec.root)
    fixture_spec.path.write_text(yaml.safe_dump(spec))
    with pytest.raises(F.SpecError, match=message):
        _plan(F, fixture_spec)
    assert not (fixture_spec.root / "launch-out/proposed-launch.json").exists()


def test_preflight_scripts_render_the_plan_command(F, fixture_spec, library):
    _, record = _plan(F, fixture_spec)
    preflight = F.preflight_script(record, run=False).splitlines()
    assert preflight[:3] == ["#!/bin/bash", "set -e", f"cd {record['cwd']}"]
    assert any(line.startswith("mkdir -p ") and "fixture-run-preflight/compute-slots" in line for line in preflight)
    assert preflight[-1].startswith("exec taskset -c 3,4 /usr/bin/env -i ")
    assert preflight[-1].endswith("--author-token-unbounded --preflight-only")
    assert "--run-id fixture-run-preflight" in preflight[-1]
    run = F.preflight_script(record, run=True).splitlines()
    assert run[-2] == "ulimit -n 65536"
    assert run[-1].startswith("exec timeout -s TERM -k 60 9000 taskset -c 3,4 /usr/bin/env -i ")
    assert "--preflight-only" not in run[-1] and "--run-id fixture-run " in run[-1]


def test_launch_requires_confirmation(F, fixture_spec, library, capsys):
    ctx, _ = _plan(F, fixture_spec)
    assert F.cmd_launch(ctx, SimpleNamespace(confirm=False)) == 2
    record = json.loads((fixture_spec.root / "launch-out/proposed-launch.json").read_text())
    assert record["execution_started"] is False


def test_error_markers_match_ordered_fragments(F):
    lines = ["ok", "Grade FAILED for x", "lease refused", "refusing stale lease", "Traceback (most recent"]
    markers = [["grade failed"], ["traceback"], ["refus", "lease"]]
    assert F.error_lines(lines, markers) == ["Grade FAILED for x", "refusing stale lease", "Traceback (most recent"]


def test_fresh2_spec_declares_the_reviewed_selection(F):
    spec = yaml.safe_load(FRESH2.read_text())
    inputs, plan = spec["inputs"], spec["plan"]
    assert inputs["library_version"] == 20
    assert inputs["common_substrate"] == "merlin.common_compiler_substrate.v13"
    assert "measured-exo/" in inputs["forbidden_reference_prefixes"]
    assert "measured-exo" not in inputs["public_references"]
    assert plan["environment"]["MERLIN_CAPSULE_ORDER"] == "information-first"
    assert {"--fresh-backend", "--author-token-unbounded"} <= set(plan["argv"])
    assert plan["environment"]["MERLIN_BUNDLE_CAS"].startswith("${RUN_STATE}/")
    ctx = F.Context(FRESH2, export_id="0" * 12)
    ctx.computed.update(dict.fromkeys(("DURATION_S", "RATE_LIMIT_WAITS", "SIM_MAX_JOBS", "LIBRARY_ROOT"), "1"))
    ctx.computed.update(
        dict.fromkeys(("LIBRARY_CONTRACT_SHA256", "FULL_COMPILER_MANIFEST_SHA256", "PERFORMANCE_SELECTION_PIN"), "1")
    )
    for name in ("export", "inputs", "seal", "plan", "preflight", "check"):
        assert "${" not in json.dumps(ctx.x_tree(spec[name]))
    assert ctx.x("${ARTIFACTS}").endswith("/open-fresh/fresh2")


def test_next_spec_extends_fresh2_with_the_builder_settings(F):
    with pytest.raises(F.SpecError, match="--run-id"):
        F.Context(HERE / "fresh-campaign-next.yaml")
    ctx = F.Context(HERE / "fresh-campaign-next.yaml", run_id="gemmini-open-compiler-20261027-fresh3")
    base = F.load_spec(FRESH2)
    assert ctx.spec["plan"]["argv"] == base["plan"]["argv"]
    assert ctx.spec["plan"]["policy"]["base"]["path"].endswith("/open-fresh/fresh4/policy.json")
    assert ctx.spec["plan"]["policy"]["performance_scope"] is None
    assert ctx.spec["plan"]["policy"]["defaults"] == {
        "prohibited_instruction_roles": ["loop_descriptor"],
        "gsim_measurement": {"long_max_jobs": 4, "max_job_wall_s": 1800},
        "gsim": {"max_job_wall_s": 1800},
        "gsim_correctness": {"max_job_wall_s": 1800},
    }
    assert ctx.run["author_helper_effort"] == "medium" and ctx.unit is None
    assert ctx.spec["seal"]["review_note"] is None and ctx.spec["seal"]["reviewed_by"]
    assert ctx.x("${ARTIFACTS}").endswith("/open-fresh/fresh3")
    # The next run selects the withholding author SDK under its current number.
    assert ctx.spec["inputs"]["library_version"] == 22
    assert ctx.spec["inputs"]["common_substrate"] == "merlin.common_compiler_substrate.v15"
    assert ctx.spec["inputs"]["output"] == base["inputs"]["output"]
    assert ctx.x(ctx.spec["plan"]["environment"]["MERLIN_JOINT_CORRECTNESS_ROSTER"]) == ctx.x(
        ctx.spec["inputs"]["environment"]["MERLIN_JOINT_CORRECTNESS_ROSTER"])
    assert "original-roster-r6.json" in ctx.x(ctx.spec["plan"]["environment"]["MERLIN_JOINT_CORRECTNESS_ROSTER"])

    assert ctx.x(ctx.spec["plan"]["guidance"]).endswith("/fresh3-guidance.md")
    pin = json.loads(ctx.x(ctx.spec["inputs"]["environment"]["MERLIN_JOINT_PERFORMANCE_SELECTION"]))
    assert pin["path"] == ctx.x(ctx.spec["plan"]["timing_selection"])
    assert pin["path"].endswith("/prospective-performance-selection-v83/performance-selection.json")
    assert pin["sha256"] == "5a219811e8edbe66197b57cacb84008833abcdc9ee6d73a36a81d4f0eb0788ef"
    changed = F.Context(
        HERE / "fresh-campaign-next.yaml",
        run_id="fixture-next",
        overrides=["NEXT_TIMING_SELECTION=/selected/runtime.json", "NEXT_TIMING_SELECTION_SHA256=" + "a" * 64],
    )
    other = json.loads(changed.x(changed.spec["inputs"]["environment"]["MERLIN_JOINT_PERFORMANCE_SELECTION"]))
    assert other == {"path": changed.x(changed.spec["plan"]["timing_selection"]), "sha256": "a" * 64}
    assert other["path"] == "/selected/runtime.json"


def test_extended_spec_adds_policy_defaults_and_helper_effort(F, fixture_spec, library):
    spec = copy.deepcopy(fixture_spec.spec)
    base = json.loads(Path(spec["plan"]["policy"]["base"]["path"]).read_text())
    base["gsim_measurement"] = {"max_jobs": 2}
    Path(spec["plan"]["policy"]["base"]["path"]).write_text(json.dumps(base))
    spec["plan"]["policy"]["base"]["sha256"] = hashlib.sha256(json.dumps(base).encode()).hexdigest()
    fixture_spec.path.write_text(yaml.safe_dump(spec))
    child = fixture_spec.root / "child.yaml"
    child.write_text(
        yaml.safe_dump(
            {
                "extends": fixture_spec.path.name,
                "run": {"run_id": "fixture-next", "author_helper_effort": "medium"},
                "plan": {
                    "policy": {
                        "defaults": {
                            "gsim_measurement": {"long_max_jobs": 4, "max_jobs": 9},
                            "prohibited_instruction_roles": ["loop_descriptor"],
                        }
                    }
                },
            }
        )
    )
    ctx = F.Context(child, export_id=fixture_spec.revision[:12])
    record = F.build_plan(ctx)
    assert record["argv"][-2:] == ["--author-helper-effort", "medium"]
    policy = json.loads((fixture_spec.root / "launch-out/policy.json").read_text())
    # A default fills only what the base policy leaves unselected.
    assert policy["gsim_measurement"] == {"max_jobs": 2, "long_max_jobs": 4}
    # The instruction rule reaches the joint policy every grader of the run reads.
    assert policy["prohibited_instruction_roles"] == ["loop_descriptor"]
    assert record["run_id"] == "fixture-next"
    parent = {"path": str(fixture_spec.path), "sha256": hashlib.sha256(fixture_spec.path.read_bytes()).hexdigest()}
    assert record["launch_spec"]["extends"] == [parent]


def test_fresh2_spec_reproduces_the_recorded_launch(F, tmp_path):
    """Read-only against the recorded launch and its export; every write goes to tmp_path."""
    RECORDED = F.Context(FRESH2, export_id="0" * 12).path("${ARTIFACTS}")
    if not (RECORDED / "proposed-launch.json").exists():
        pytest.skip("fresh2 record is not on this host")
    recorded = json.loads((RECORDED / "proposed-launch.json").read_text())
    ctx = F.Context(
        FRESH2,
        export_id=Path(recorded["framework_repository"]).name,
        overrides=[
            f"ARTIFACTS={tmp_path}/launch-out",
            f"RUN_STATE={tmp_path}/state",
            f"PRIVATE_ROOT={tmp_path}/private",
            f"LAUNCH_MODULES={HERE}",
        ],
    )
    if not ctx.path("${INPUTS}/author-library/compiler-library-selection.json").exists():
        pytest.skip("fresh2 export inputs are not on this host")
    planned = F.build_plan(ctx)
    swaps = {
        f"{tmp_path}/launch-out": str(RECORDED),
        f"{tmp_path}/state": str(Path(recorded["environment"]["MERLIN_BUNDLE_CAS"]).parent),
        f"{tmp_path}/private": str(Path(recorded["argv"][recorded["argv"].index("--codex-home-root") + 1]).parent),
        str(HERE): recorded["framework_repository"] + "/examples/gemmini/phase1/launch",
    }

    def norm(value):
        if isinstance(value, str):
            for old, new in swaps.items():
                value = value.replace(old, new)
            return value
        if isinstance(value, list):
            return [norm(item) for item in value]
        if isinstance(value, dict):
            return {key: norm(item) for key, item in value.items()}
        return value

    planned = norm(planned)
    assert planned["argv"] == recorded["argv"]
    assert planned["environment"] == recorded["environment"]
    # reservation_argv sorts the env -i assignments; fresh2 kept v8's order.
    assert sorted(planned["reserved_argv"]) == sorted(recorded["reserved_argv"])
    differing = {key for key in set(planned) | set(recorded) if planned.get(key) != recorded.get(key)}
    # execution_started: set by launch; launch_spec: new provenance; full_compiler_manifest_sha256:
    # fresh2 carried v8's stale value, the plan records the manifest its argv pins; capsules: the
    # sealed release's case counts, which plans record since fresh2.
    assert differing == {"reserved_argv", "execution_started", "launch_spec", "full_compiler_manifest_sha256", "capsules"}
    assert (
        planned["full_compiler_manifest_sha256"]
        == recorded["argv"][recorded["argv"].index("--full-compiler-manifest-sha256") + 1]
    )
    for name in ("policy.json", "selected-library-contract.json", "performance-selection.json", "author-guidance.md"):
        assert (tmp_path / "launch-out" / name).read_bytes() == (RECORDED / name).read_bytes(), name


def test_export_archives_head_and_copies_local_configuration(F, fixture_spec, tmp_path):
    config = tmp_path / "config"
    _write(config / ".env", "SECRET=1\n")
    _write(config / "out/host/package/descriptor.json", {"package": True})
    spec = copy.deepcopy(fixture_spec.spec)
    spec["variables"]["FRAMEWORK"] = str(tmp_path / "frameworks/${EXPORT_ID}")
    spec["export"] = {
        "config_source": str(config),
        "copies": [{"path": ".env", "mode": "0600"}, {"path": "out/host/package"}],
    }
    fixture_spec.path.write_text(yaml.safe_dump(spec))
    assert F.cmd_export(F.Context(fixture_spec.path), SimpleNamespace()) == 0
    export = tmp_path / "frameworks" / fixture_spec.revision[:12]
    assert (export / "README").read_text() == "fixture\n"
    assert (export / ".env").stat().st_mode & 0o777 == 0o600
    assert json.loads((export / "out/host/package/descriptor.json").read_text()) == {"package": True}
    with pytest.raises(F.SpecError, match="export exists"):
        F.cmd_export(F.Context(fixture_spec.path), SimpleNamespace())
    (tmp_path / "repo/README").write_text("edited\n")
    with pytest.raises(F.SpecError, match="uncommitted"):
        F.cmd_export(F.Context(fixture_spec.path), SimpleNamespace())


def test_v2_reference_staging_preserves_review_and_rejects_forbidden_origins(F, tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    leaf = source / "kernel.py"
    leaf.write_text("public_example = True\n")
    (source / "unselected.py").write_text("private = True\n")
    document = {"schema": "operator.allowed_reference_sources.v2", "review_id": "launch-test",
        "origins": [{"name": "upstream", "kind": "public_repository", "container": ".",
          "identity": "1" * 40, "license": "MIT", "public_url": "https://example.org/source",
          "allowed_purpose": "reference_reading", "files": [{"member": "kernel.py",
          "destination": "kernel.py", "sha256": hashlib.sha256(leaf.read_bytes()).hexdigest()}]}]}
    manifest = _write(source / "reference-manifest.json", document)
    staged = tmp_path / "staged"
    assert F.stage_launch_references(manifest, staged, ["measured-exo/"]) == manifest
    assert (staged / "upstream/kernel.py").read_bytes() == leaf.read_bytes()
    assert not (staged / "unselected.py").exists()
    assert F.reference_members(document) == [{"path": "upstream/kernel.py"}]
    with pytest.raises(F.SpecError, match="forbidden"):
        F.stage_launch_references(manifest, tmp_path / "bad", ["upstream/"])


def test_additional_source_selections_are_pinned(F, tmp_path):
    selected = _write(tmp_path / "selection.json", {"member": "public"})
    pin = {"path": str(selected), "sha256": hashlib.sha256(selected.read_bytes()).hexdigest()}
    ctx = SimpleNamespace(path=Path)
    result = F.extra_source_selections(ctx, {"public_program_suite": pin, "host_capability_selection": pin})
    assert result == {"public_program_suite": selected, "host_capability_selection": (selected, pin["sha256"])}
    selected.write_text("changed")
    with pytest.raises(F.SpecError, match="selection changed"):
        F.extra_source_selections(ctx, {"public_program_suite": pin})


def test_plan_counts_match_sealed_release_not_historical_template(F, fixture_spec, library):
    inputs = fixture_spec.root / "framework/inputs"
    release = inputs / "source-release"
    _write(inputs / "inspect.json", {"release": str(release), "review_digest": "review",
        "counts": {"public_admitted": 159, "public_scalability": {"source_cases": 117},
                   "public_programs": {"source_cases": 8}}})
    _write(release / "private/seal.json", {"review_digest": "review"})
    _, record = _plan(F, fixture_spec)
    assert record["capsules"]["total_public"] == 284
    assert record["capsules"]["connected_programs"] == 8
    assert "original_correctness" not in record["capsules"]
    _write(release / "private/seal.json", {"review_digest": "other"})
    with pytest.raises(F.SpecError, match="sealed inspection"):
        _plan(F, fixture_spec)
