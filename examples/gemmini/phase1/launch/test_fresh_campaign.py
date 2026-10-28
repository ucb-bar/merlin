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


@pytest.mark.parametrize("damage", [None, "engine", "library", "source"])
def test_plan_pins_recorder_and_rejects_changed_execution_inputs(F, fixture_spec, library, damage):
    root = fixture_spec.root
    engine = _write(root / "spike", "engine bytes")
    extension = _write(root / "recorder/library.so", "extension bytes")
    source = _write(root / "framework/src/merlin/perf/spike_command_recorder.cc", "source bytes")
    manifest = _write(root / "recorder/manifest.json", {
        "schema": "merlin.command_recorder.v1",
        "engine": {"path": str(engine), "sha256": F.sha256(engine)},
        "library": {"path": str(extension), "sha256": F.sha256(extension)},
        "source_sha256": F.sha256(source),
    })
    fixture_spec.spec["ordering"] = {"command_recorder": {
        "engine": str(engine), "include": str(root / "include"), "output": str(manifest.parent)}}
    fixture_spec.path.write_text(yaml.safe_dump(fixture_spec.spec))
    ctx, record = _plan(F, fixture_spec)
    assert json.loads(record["environment"]["MERLIN_COMMAND_RECORDER"]) == {
        "path": str(manifest), "sha256": F.sha256(manifest)}
    if damage is not None:
        {"engine": engine, "library": extension, "source": source}[damage].write_text("changed")
        with pytest.raises(F.SpecError, match="command recorder unavailable"):
            F.command_recorder_pin(ctx)


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


def test_fresh7_spec_selects_r7_inputs_with_existing_runtime_and_clock(F):
    ctx = F.Context(HERE / "fresh-campaign-fresh7.yaml")
    parent = F.load_spec(HERE / "fresh-campaign-next.yaml")
    assert ctx.run_id == "gemmini-open-compiler-20261027-fresh7"
    assert ctx.unit == "merlinphase1fresh20261027b"
    assert ctx.run["max_wall_s"] == parent["run"]["max_wall_s"] == 16200
    assert ctx.run["final_model_s"] == 3600
    assert ctx.spec["inputs"]["corpus"].endswith("fresh7/corpus-source-release")
    assert ctx.spec["plan"]["policy"]["performance_scope"] is None
    assert "command_recorder" in ctx.spec["ordering"]
    assert ctx.spec["inputs"]["public_program_suite"] == parent["inputs"]["public_program_suite"]


def test_short_policy_keeps_requested_clock_and_nested_windows(F):
    seen = []

    def validate(value):
        seen.append(value["duration_s"])
        value["gsim"]["worker_budget_s"] = value["duration_s"]
        return value

    value = F.edited_policy(
        {"version": 1, "drain_s": 900}, max_wall_s=2100, final_model_s=300,
        workers=1, excluded=None, validate=validate,
    )
    assert (value["duration_s"], value["final_model_s"], value["drain_s"]) == (2100, 300, 300)
    assert seen == [2100]


def test_plan_renders_fresh_unseeded_launch(F, fixture_spec, library):
    ctx, record = _plan(F, fixture_spec)
    assert "--reference-experiments" not in record["argv"]
    assert "reference_experiments" not in record
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


@pytest.mark.parametrize("damage", (None, "selection", "packet"))
def test_reference_guidance_is_explicit_pinned_and_reopened(F, fixture_spec, library, damage):
    from merlin_experiments.phase1.feedback import reference_experiments as reference

    from merlin.verify.lean_contract import document_digest

    root = fixture_spec.root
    context = dict(compiler_sha256="a" * 64, source_sha256="b" * 64,
                   environment_sha256="c" * 64, reference_sha256="d" * 64,
                   visibility="reference_guided", evaluator=dict(identity="test-owner", sha256="e" * 64))
    packet = dict(schema=reference.PACKET_SCHEMA, context=context,
                  execution_context_identity=document_digest(context), tasks=[])
    packet["identity"] = document_digest(packet)
    path = _write(root / "reference-packet.json", packet)
    selection = _write(root / "reference-selection.json", dict(schema=reference.SCHEMA,
        packet=dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()), current_context=context))
    pin = dict(path=str(selection), sha256=hashlib.sha256(selection.read_bytes()).hexdigest())
    fixture_spec.spec["plan"]["reference_experiments"] = pin
    fixture_spec.path.write_text(yaml.safe_dump(fixture_spec.spec))
    if damage:
        (selection if damage == "selection" else path).write_text("{}")
        with pytest.raises(F.SpecError, match="reference experiment selection is invalid"):
            _plan(F, fixture_spec)
    else:
        _, record = _plan(F, fixture_spec)
        assert record["reference_experiments"] == pin
        argv = record["argv"]
        assert argv[argv.index("--reference-experiments") + 1] == str(selection)
        assert argv[argv.index("--reference-experiments-sha256") + 1] == pin["sha256"]

        # The same source/reference selection can be used with the mechanism disabled.
        off_ctx, disabled = _plan(F, fixture_spec, context={"reference_guidance": "off"})
        assert disabled["reference_guidance"] == "off" and "reference_experiments" not in disabled
        assert "--reference-experiments" not in disabled["argv"]
        without_guidance = list(argv)
        for flag in ("--reference-experiments", "--reference-experiments-sha256"):
            pos = without_guidance.index(flag)
            del without_guidance[pos:pos + 2]
        assert disabled["argv"] == without_guidance
        assert disabled["source_release"] == record["source_release"]
        assert disabled["timing_selection"] == record["timing_selection"]
        assert F.prepared_plan(off_ctx) == disabled
        with pytest.raises(F.SpecError, match="selected mode"):
            F.prepared_plan(F.Context(fixture_spec.path, export_id=fixture_spec.revision[:12],
                                     reference_guidance="on"))
        # Disabled execution must not acquire a dependency on optional derived evidence.
        path.write_text("{}")
        _, disabled = _plan(F, fixture_spec, context={"reference_guidance": "off"})
        assert "--reference-experiments" not in disabled["argv"]


def test_guidance_on_requires_a_selected_reference(F, fixture_spec, library):
    with pytest.raises(F.SpecError, match="on requires a pinned"):
        _plan(F, fixture_spec, context={"reference_guidance": "on"})
    parsed = F.parser().parse_args(["plan", "--spec", str(fixture_spec.path), "--reference-guidance", "off"])
    assert parsed.reference_guidance == "off"


def test_matched_specs_share_inputs_and_budgets_with_isolated_equal_cpu_pools(F):
    raw = F.Context(HERE / "fresh-campaign-fresh7-raw.yaml", export_id="a" * 12)
    cca = F.Context(HERE / "fresh-campaign-fresh7-cca.yaml", export_id="a" * 12)
    assert (raw.reference_guidance, cca.reference_guidance) == ("off", "on")
    for key in ("FRAMEWORK", "INPUTS", "RELEASE", "FRESH7_REFERENCE_EXPERIMENTS"):
        assert raw.lookup(key) == cca.lookup(key)
    assert raw.path(raw.section("inputs")["public_references"]) == cca.path(cca.section("inputs")["public_references"])
    assert raw.run["workers"] == cca.run["workers"] == 10
    assert raw.run["max_wall_s"] == cca.run["max_wall_s"]
    left, right = set(F.cpu_list(raw.cpus)), set(F.cpu_list(cca.cpus))
    assert len(left) == len(right) == 10 and not left.intersection(right)
    for key in ("PRIVATE_ROOT", "OPERATOR_TMP", "ARTIFACTS", "RUN_STATE"):
        assert raw.lookup(key) != cca.lookup(key)


def test_helper_effort_passes_through(F, fixture_spec, library):
    _, record = _plan(F, fixture_spec, helper_effort="low")
    assert record["argv"][-2:] == ["--author-helper-effort", "low"]
    spec = copy.deepcopy(fixture_spec.spec)
    spec["run"]["author_helper_effort"] = "medium"
    fixture_spec.path.write_text(yaml.safe_dump(spec))
    _, record = _plan(F, fixture_spec)
    assert record["argv"][-2:] == ["--author-helper-effort", "medium"]


def test_prepared_plan_cannot_silently_select_another_mode(F, fixture_spec, library):
    ctx, record = _plan(F, fixture_spec)
    assert F.prepared_plan(ctx) == record
    fixture_spec.spec["plan"]["reference_experiments"] = {"path": "/unselected", "sha256": "a" * 64}
    fixture_spec.path.write_text(yaml.safe_dump(fixture_spec.spec))
    changed = F.Context(fixture_spec.path, export_id=fixture_spec.revision[:12])
    with pytest.raises(F.SpecError, match="selected mode"):
        F.prepared_plan(changed)


def test_unselected_unit_keeps_the_wrapper_name(F, fixture_spec, library):
    spec = copy.deepcopy(fixture_spec.spec)
    spec["run"]["unit"] = None
    fixture_spec.path.write_text(yaml.safe_dump(spec))
    _, record = _plan(F, fixture_spec)
    wrapper = record["reserved_argv"][: record["reserved_argv"].index("--")]
    assert wrapper[wrapper.index("--unit") + 1].startswith("merlinphase1")
    _, record = _plan(F, fixture_spec, context={"unit": "merlinphase1chosen"})
    wrapper = record["reserved_argv"][: record["reserved_argv"].index("--")]
    assert wrapper[wrapper.index("--unit") + 1] == "merlinphase1chosen"


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


def test_next_spec_extends_fresh2_with_the_builder_settings(F, tmp_path):
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
    assert ctx.spec["inputs"]["library_version"] == 23
    assert ctx.spec["inputs"]["common_substrate"] == "merlin.common_compiler_substrate.v16"
    assert ctx.spec["inputs"]["output"] == base["inputs"]["output"]
    assert ctx.x(ctx.spec["plan"]["environment"]["MERLIN_JOINT_CORRECTNESS_ROSTER"]) == ctx.x(
        ctx.spec["inputs"]["environment"]["MERLIN_JOINT_CORRECTNESS_ROSTER"]
    )
    assert "original-roster-r6.json" in ctx.x(ctx.spec["plan"]["environment"]["MERLIN_JOINT_CORRECTNESS_ROSTER"])

    # Engine and lane caps follow the reserved CPU list, also when --cpus selects another one.
    environment = ctx.spec["plan"]["environment"]
    for key in F.WORKER_KEYS:
        assert ctx.x(environment[key]) == str(len(F.cpu_list(ctx.cpus))) == "20"
    narrow = F.Context(HERE / "fresh-campaign-next.yaml", run_id="fixture-narrow", cpus="9,10,11,12")
    assert {narrow.x(narrow.spec["plan"]["environment"][key]) for key in F.WORKER_KEYS} == {"4"}
    redefined = _write(
        tmp_path / "redefined.yaml",
        {"extends": str(HERE / "fresh-campaign-next.yaml"), "variables": {"CPU_COUNT": "3"}},
    )
    with pytest.raises(F.SpecError, match="built-in"):
        F.Context(redefined, run_id="fixture-builtin")

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


def test_fresh2_spec_reproduces_the_recorded_launch(F, tmp_path, monkeypatch):
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
    # This reconstructs historical argv, rather than assessing present runtime
    # readiness. Its immutable export predates the import_probe hook; the live
    # capture-interpreter check has its own test below.
    monkeypatch.setattr(F, "check_capture_interpreter", lambda environment, *, framework: None)
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
    # sealed release's case counts, which plans record since fresh2. Reference mode
    # and the common input/reference pins are explicit metadata for matched arms.
    assert differing == {
        "reserved_argv",
        "execution_started",
        "launch_spec",
        "full_compiler_manifest_sha256",
        "capsules",
        "reference_guidance",
        "reference_manifest",
        "inputs_selection",
    }
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
    from merlin.targetgen.provenance import source_revision

    assert source_revision(export) == fixture_spec.revision
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
    document = {
        "schema": "operator.allowed_reference_sources.v2",
        "review_id": "launch-test",
        "origins": [
            {
                "name": "upstream",
                "kind": "public_repository",
                "container": ".",
                "identity": "1" * 40,
                "license": "MIT",
                "public_url": "https://example.org/source",
                "allowed_purpose": "reference_reading",
                "files": [
                    {
                        "member": "kernel.py",
                        "destination": "kernel.py",
                        "sha256": hashlib.sha256(leaf.read_bytes()).hexdigest(),
                    }
                ],
            }
        ],
    }
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
    _write(
        inputs / "inspect.json",
        {
            "release": str(release),
            "review_digest": "review",
            "counts": {
                "public_admitted": 159,
                "public_scalability": {"source_cases": 117},
                "public_programs": {"source_cases": 8},
            },
        },
    )
    _write(release / "private/seal.json", {"review_digest": "review"})
    _, record = _plan(F, fixture_spec)
    assert record["capsules"]["total_public"] == 284
    assert record["capsules"]["connected_programs"] == 8
    assert "original_correctness" not in record["capsules"]
    _write(release / "private/seal.json", {"review_digest": "other"})
    with pytest.raises(F.SpecError, match="sealed inspection"):
        _plan(F, fixture_spec)


def test_ordering_selection_must_bind_the_campaign_facts_contract_and_readers(F, tmp_path):
    framework = tmp_path / "framework"
    reader = framework / "packages/p/src/pkg/reader.py"
    reader.parent.mkdir(parents=True)
    reader.write_text("READER = 1\n")
    contract, facts, other = (_write(tmp_path / name, name) for name in ("contract.yaml", "facts.json", "other"))
    selection = {
        "base_contract": {"path": str(contract), "sha256": F.sha256(contract)},
        "base_facts": {"path": str(facts), "sha256": F.sha256(facts)},
        "readers": {"pkg.reader": F.sha256(reader)},
    }
    path = _write(tmp_path / "selection.json", selection)
    environment = {
        F.ORDERING_ENV: json.dumps({"path": str(path), "sha256": F.sha256(path)}),
        "MERLIN_TARGET_CONTRACT": str(contract),
        "MERLIN_RTL_FACTS": str(facts),
    }
    F.check_ordering_selection(environment, framework=framework)
    F.check_ordering_selection({}, framework=framework)  # no selection: nothing to bind
    with pytest.raises(F.SpecError, match="base_contract"):
        F.check_ordering_selection({**environment, "MERLIN_TARGET_CONTRACT": str(other)}, framework=framework)
    reader.write_text("READER = 2\n")
    with pytest.raises(F.SpecError, match="different reader"):
        F.check_ordering_selection(environment, framework=framework)


def test_capture_interpreter_must_run_the_full_model_child(F, tmp_path):
    import sys

    F.check_capture_interpreter({}, framework=tmp_path)  # no capture interpreter selected: nothing to probe
    # The interpreter running these tests has the child's dependencies.
    F.check_capture_interpreter({F.CAPTURE_PYTHON_ENV: sys.executable}, framework=tmp_path)
    broken = tmp_path / "python"
    broken.write_text("#!/bin/sh\necho \"ModuleNotFoundError: No module named 'jsonschema'\" >&2\nexit 1\n")
    broken.chmod(0o755)
    with pytest.raises(F.SpecError, match="No module named 'jsonschema'"):
        F.check_capture_interpreter({F.CAPTURE_PYTHON_ENV: str(broken)}, framework=tmp_path)
    with pytest.raises(F.SpecError, match="cannot be probed"):
        F.check_capture_interpreter({F.CAPTURE_PYTHON_ENV: str(tmp_path / "missing")}, framework=tmp_path)


def test_plan_pins_the_derived_ordering_selection(F, fixture_spec, library):
    root = fixture_spec.root
    facts, contract = _write(root / "facts.json", "facts"), _write(root / "contract.yaml", "contract")
    selection = {
        "base_facts": {"path": str(facts), "sha256": F.sha256(facts)},
        "base_contract": {"path": str(contract), "sha256": F.sha256(contract)},
        "readers": {},
    }
    spec = copy.deepcopy(fixture_spec.spec)
    spec["plan"]["environment"].update(MERLIN_RTL_FACTS=str(facts), MERLIN_TARGET_CONTRACT=str(contract))
    spec["plan"]["ordering_selection"] = str(root / "derived/selection.json")
    spec["ordering"] = {"target": "t", "declarations": str(contract)}
    fixture_spec.path.write_text(yaml.safe_dump(spec))
    # A fresh ordering invocation precedes build_plan's performance pin binding.
    fresh = F.Context(fixture_spec.path, export_id=fixture_spec.revision[:12])
    assert F.ordering_inputs(fresh)["facts"] == facts
    assert "PERFORMANCE_SELECTION_PIN" not in fresh.computed
    assert F.ordering_inputs(fresh)["contract"] == contract
    with pytest.raises(F.SpecError, match="run `fresh_campaign.py ordering` first"):
        _plan(F, fixture_spec)
    _write(root / "derived/selection.json", selection)
    ctx, record = _plan(F, fixture_spec)
    pin = json.loads(record["environment"][F.ORDERING_ENV])
    assert pin["path"] == str(root / "launch-out/completion-order-selection.json")
    assert pin["sha256"] == F.sha256(root / "derived/selection.json")
    # The ordering step reuses a selection bound to these bytes and refuses one bound to others.
    assert F.cmd_ordering(ctx, None) == 0
    contract.write_text("another contract")
    with pytest.raises(F.SpecError, match="binds other facts or a different contract"):
        F.cmd_ordering(ctx, None)


def _reference_map(root, manifest_sha):
    from merlin_experiments.phase1.feedback import reference_map

    document = {"schema": reference_map.SCHEMA, "reference_manifest_sha256": manifest_sha, "entries": [
        {"id": "e1", "file": "listing/model.txt", "locator": {"lines": [1, 9]}, "model_context": "fixture",
         "operation": {"kind": "matmul", "dims": {"M": 4, "K": 8, "N": 16}, "epilogue": [], "dtypes": {}},
         "techniques": ["weights once"], "uses_hardware_loops": True, "confidence": "verified", "evidence": "x"}],
        "coverage": {}, "not_covered": []}
    path = _write(root / "operator-map" / "REFERENCE_MAP.json", document)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


@pytest.mark.parametrize("damage", (None, "bytes", "manifest", "unset"))
def test_reference_map_is_pinned_verified_and_staged_through_the_environment(F, fixture_spec, library, damage):
    root = fixture_spec.root
    manifest = root / "framework/inputs/public-references/reference-manifest.json"
    pin = _reference_map(root, F.sha256(manifest) if damage != "manifest" else "0" * 64)
    selected = dict(pin)
    if damage == "unset":
        fixture_spec.spec["variables"]["MAP_SHA256"] = "set-to-the-sha256-of-REFERENCE_MAP.json"
        selected["sha256"] = "${MAP_SHA256}"
    fixture_spec.spec["plan"]["reference_map"] = selected
    fixture_spec.path.write_text(yaml.safe_dump(fixture_spec.spec))
    if damage == "bytes":
        Path(pin["path"]).write_text("{}")
    if damage is not None:
        match = {"bytes": "reference map", "manifest": "another reference manifest", "unset": "MAP_SHA256"}[damage]
        with pytest.raises(F.SpecError, match=match):
            _plan(F, fixture_spec)
        return
    _, record = _plan(F, fixture_spec)
    staged = json.loads(record["environment"]["MERLIN_REFERENCE_MAP"])
    assert staged == record["reference_map"] and staged["sha256"] == pin["sha256"]
    assert Path(staged["path"]) == fixture_spec.root / "launch-out" / "REFERENCE_MAP.json"
    assert F.sha256(staged["path"]) == pin["sha256"]
    # The same map for the guidance-off arm: identical bytes, no guidance flags.
    _, off = _plan(F, fixture_spec, context={"reference_guidance": "off"})
    assert json.loads(off["environment"]["MERLIN_REFERENCE_MAP"])["sha256"] == pin["sha256"]
    assert "--reference-experiments" not in off["argv"]


def test_withheld_reference_tasks_keep_the_selection_and_add_one_flag(F, fixture_spec, library):
    from merlin_experiments.phase1.feedback import reference_experiments as reference

    from merlin.verify.lean_contract import document_digest

    root = fixture_spec.root
    context = dict(compiler_sha256="a" * 64, source_sha256="b" * 64, environment_sha256="c" * 64,
                   reference_sha256="d" * 64, visibility="reference_guided",
                   evaluator=dict(identity="test-owner", sha256="e" * 64))
    packet = dict(schema=reference.PACKET_SCHEMA, context=context,
                  execution_context_identity=document_digest(context), tasks=[])
    packet["identity"] = document_digest(packet)
    path = _write(root / "reference-packet.json", packet)
    selection = _write(root / "reference-selection.json", dict(schema=reference.SCHEMA,
        packet=dict(path=str(path), sha256=F.sha256(path)), current_context=context))
    fixture_spec.spec["plan"].update(reference_experiments=dict(path=str(selection), sha256=F.sha256(selection)),
                                     reference_experiment_tasks="off")
    fixture_spec.path.write_text(yaml.safe_dump(fixture_spec.spec))
    _, record = _plan(F, fixture_spec)
    argv = record["argv"]
    assert argv[argv.index("--reference-experiments-tasks") + 1] == "off"
    assert "--reference-experiments" in argv and record["reference_experiment_tasks"] == "off"
    _, raw = _plan(F, fixture_spec, context={"reference_guidance": "off"})
    assert "--reference-experiments-tasks" not in raw["argv"] and "reference_experiment_tasks" not in raw
    fixture_spec.spec["plan"]["reference_experiment_tasks"] = "never"
    fixture_spec.path.write_text(yaml.safe_dump(fixture_spec.spec))
    with pytest.raises(F.SpecError, match="on or off"):
        _plan(F, fixture_spec)


def test_fresh7ref_arms_share_one_reference_map_and_only_the_guided_arm_withholds_tasks(F):
    raw = F.Context(HERE / "fresh-campaign-fresh7ref-raw.yaml", export_id="a" * 12)
    cca = F.Context(HERE / "fresh-campaign-fresh7ref-cca.yaml", export_id="a" * 12)
    assert (raw.reference_guidance, cca.reference_guidance) == ("off", "on")
    pins = [ctx.x_tree(ctx.spec["plan"]["reference_map"]) for ctx in (raw, cca)]
    assert pins[0] == pins[1]
    assert pins[0]["path"].endswith("preparation/fresh7-reference-map/REFERENCE_MAP.json")
    assert cca.spec["plan"]["reference_experiment_tasks"] == "off"
    assert raw.spec["plan"].get("reference_experiment_tasks", "on") == "on"
    assert raw.lookup("FRESH7_REFERENCE_MAP_SHA256") == cca.lookup("FRESH7_REFERENCE_MAP_SHA256")


def test_fresh8_arms_share_kernels_without_baselines_and_only_guided_maps(F):
    raw = F.Context(HERE / "fresh-campaign-fresh8ref-raw.yaml", export_id="a" * 12)
    cca = F.Context(HERE / "fresh-campaign-fresh8ref-cca.yaml", export_id="a" * 12)
    # Operator decision: the reference map belongs to the guided arm only.
    assert raw.spec["plan"]["reference_map"] is None
    pin = cca.x_tree(cca.spec["plan"]["reference_map"])
    assert pin["path"].endswith("preparation/fresh8-reference-map/REFERENCE_MAP.json")
    assert raw.spec["plan"]["reference_experiments"] is None
    references = raw.lookup("MATCHED_RAW_REFERENCES")
    assert references == cca.lookup("MATCHED_RAW_REFERENCES") and references.endswith("fresh8-reference-kernels-only")
    # Operator decision: no measured performance baselines or targets reach either arm.
    assert "baseline" not in references and "performance" not in references


def test_fresh8_arms_share_raw_inputs_objective_and_budgets_with_guided_extras(F):
    raw = F.Context(HERE / "fresh-campaign-fresh8ref-raw.yaml", export_id="a" * 12)
    cca = F.Context(HERE / "fresh-campaign-fresh8ref-cca.yaml", export_id="a" * 12)
    assert raw.lookup("MATCHED_RAW_REFERENCES") == cca.lookup("MATCHED_RAW_REFERENCES")
    assert raw.lookup("INPUTS") == cca.lookup("INPUTS")
    # Base instructions must satisfy the actual pair check; guided-only staging
    # supplies the map/tool instructions, rather than two drifting base prompts.
    guidance = [ctx.path(ctx.spec["plan"]["guidance"]) for ctx in (raw, cca)]
    assert guidance[0] == guidance[1] and guidance[0].is_file()
    assert guidance[0].read_bytes() == guidance[1].read_bytes()
    assert "REFERENCE_MAP" not in guidance[0].read_text()
    for key in ("max_wall_s", "final_model_s", "workers", "sim_max_jobs"):
        assert raw.run[key] == cca.run[key]
    assert len(F.cpu_list(raw.cpus)) == len(F.cpu_list(cca.cpus))
    assert set(F.cpu_list(raw.cpus)).isdisjoint(F.cpu_list(cca.cpus))
    assert raw.reference_guidance == "off" and cca.reference_guidance == "on"
    assert raw.spec["plan"]["reference_map"] is None and cca.spec["plan"]["reference_map"]
    assert cca.spec["plan"]["reference_experiment_tasks"] == "off"


def _sandbox_state(root: Path) -> Path:
    """Preflight state shaped like the launcher's: read-only sandbox input copies beside the workspace."""
    (root / "workspace").mkdir(parents=True)
    (root / "workspace/notes.md").write_text("kept\n")
    inputs = root / "bundle_inputs"
    for member in ("external/tool/bin", "repo/src"):
        (inputs / member).mkdir(parents=True)
    (inputs / "external/tool/bin/opt").write_bytes(b"x" * 4096)
    (inputs / "repo/src/a.py").write_text("a = 1\n")
    outside = root.parent / f"{root.name}-outside.txt"
    outside.write_text("never followed\n")
    (inputs / "repo/src/link").symlink_to(outside)
    (inputs / "snapshot.json").write_text("{}")
    for directory in sorted((p for p in inputs.rglob("*") if p.is_dir() and not p.is_symlink()), reverse=True):
        directory.chmod(0o555)
    inputs.chmod(0o500)
    return outside


def test_drop_inputs_removes_listed_copies_and_keeps_the_snapshot(F, tmp_path):
    state = tmp_path / "preflight"
    outside = _sandbox_state(state)
    freed = F.drop_inputs(state, ["bundle_inputs/external", "bundle_inputs/repo", "bundle_inputs/absent"])
    assert freed == 4096 + len("a = 1\n") + len(str(outside))
    assert sorted(p.name for p in (state / "bundle_inputs").iterdir()) == ["snapshot.json"]
    assert (state / "workspace/notes.md").is_file() and outside.is_file()
    with pytest.raises(F.SpecError, match="inside its state"):
        F.drop_inputs(state, ["../elsewhere"])


@pytest.mark.parametrize("returncode", [0, 1])
def test_preflight_drops_sandbox_inputs_on_retire_and_once_it_passes(F, fixture_spec, library, monkeypatch,
                                                                     returncode):
    ctx, _ = _plan(F, fixture_spec)
    operator = fixture_spec.root / "operator"
    section = dict(ctx.section("preflight"), retire_into=str(operator / "old-preflight"),
                   script=str(operator / "preflight.sh"), log=str(operator / "preflight.log"))
    sections = ctx.section
    monkeypatch.setattr(ctx, "section", lambda name: section if name == "preflight" else sections(name))
    item = next(i for i in section["retire"] if i["as"] == "ws")
    assert item["drop"] == ["bundle_inputs/external", "bundle_inputs/repo"]
    live = ctx.path(item["path"])
    _sandbox_state(live)

    def run(argv, **kwargs):
        _sandbox_state(live)  # the new preflight stages its own sandbox
        return SimpleNamespace(returncode=returncode)

    monkeypatch.setattr(F, "host_health_line", lambda: "HOST HEALTH ok")

    monkeypatch.setattr(F.subprocess, "run", run)
    assert F.cmd_preflight(ctx, SimpleNamespace(run=False, execute=True)) == returncode
    retired = next(ctx.path(section["retire_into"]).iterdir()) / "ws"
    assert sorted(p.name for p in (retired / "bundle_inputs").iterdir()) == ["snapshot.json"]
    assert (retired / "workspace/notes.md").is_file()
    kept = sorted(p.name for p in (live / "bundle_inputs").iterdir())
    assert kept == (["snapshot.json"] if returncode == 0 else ["external", "repo", "snapshot.json"])
