"""The checked-in launch config renders operator inputs that verify clean, and every drift is named.

Each test copies the real Gemmini launch and release configs into a temporary checkout and stands up
a synthetic operator world below a temporary task root: every pinned file exists with fixture bytes
(re-pinned in the copy), the numerics lineage and the reviewed release records agree, and the
controller is a harmless sleeping stand-in. Nothing here reads a real prep directory or starts a run.
"""

from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml
from merlin_experiments.phase1 import launch_config as lc

from merlin.common.digest import sha256_file
from merlin.common.paths import repo_root

LAUNCH = "examples/gemmini/phase1/launch/el4-fire-aligned.yaml"
RELEASE = "examples/gemmini/phase0/release/fire-aligned-v10.yaml"


def _write(path: Path, data: bytes | str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data if isinstance(data, bytes) else data.encode())
    return path


def _json(path: Path, value: object) -> Path:
    return _write(path, json.dumps(value, indent=2))


def _checkout(tmp_path: Path, edit=None) -> Path:
    """Copy both configs (and the errata) with their relative layout; optionally edit the launch YAML."""
    root = tmp_path / "checkout"
    for relative in (Path(LAUNCH).parent, Path(RELEASE).parent):
        shutil.copytree(repo_root() / relative, root / relative)
    if edit is not None:
        path = root / LAUNCH
        document = yaml.safe_load(path.read_text())
        edit(document)
        path.write_text(yaml.safe_dump(document, sort_keys=False))
    return root / LAUNCH


class World:
    """A complete synthetic operator world for one copied config."""

    def __init__(self, tmp_path: Path):
        self.tmp = tmp_path
        self.task = tmp_path / "work" / "tasks" / "fixture-task"
        self.prep = self.task / ".tmp" / "prep"
        self.home = tmp_path / "home"
        self.config_path = _checkout(tmp_path, self._local_manager)
        self.overrides = {"PYTHON": sys.executable}
        self._materialize()

    @staticmethod
    def _local_manager(document: dict) -> None:
        # The controller stand-in runs on a CPU this test process may use.
        document["cpu"]["manager"] = [min(os.sched_getaffinity(0))]

    def build(self, **kwargs) -> lc.Launch:
        overrides = {**self.overrides, **kwargs.pop("overrides", {})}
        return lc.build(
            lc.load(self.config_path),
            task_root=self.task,
            prep=self.prep,
            overrides=overrides,
            environ={"HOME": str(self.home)},
            **kwargs,
        )

    def _repin(self, launch: lc.Launch) -> None:
        """Give every pinned path fixture bytes and record their digests in the copied configs."""
        launch_doc = yaml.safe_load(self.config_path.read_text())
        release_path = self.config_path.parent / launch_doc["release_config"]
        release_doc = yaml.safe_load(release_path.read_text())
        digests = {}
        for label, (path, _digest) in launch.pins.items():
            if label in ("errata", "instruction_selection.approval"):
                continue
            target = Path(path)
            if not target.exists():
                _write(target, f"fixture bytes for {label}\n")
            digests[label] = sha256_file(target)
        for section in ("runtime_pins", "target_selection"):
            for name, pin in launch_doc[section].items():
                pin["sha256"] = digests[f"{section}.{name}"]
        launch_doc["inputs"]["oracle_timing"]["sha256"] = digests["inputs.oracle_timing"]
        selection = launch_doc["instruction_selection"]
        selection["sha256"] = digests["instruction_selection"]
        approval = _json(
            Path(launch.pins["instruction_selection.approval"][0]),
            {"selection": launch.pins["instruction_selection"][0], "selection_sha256": selection["sha256"]},
        )
        selection["approval"]["sha256"] = sha256_file(approval)
        for key in ("kernel_timing_selection", "lean_toolchain_manifest", "phase1_policy_descriptor"):
            release_doc["prepare"][key]["sha256"] = digests[f"release.prepare.{key}"]
        release_doc["seal"]["post_experiment_descriptor"]["sha256"] = digests["release.seal.post_experiment_descriptor"]
        self._numerics(launch, launch_doc)
        self.config_path.write_text(yaml.safe_dump(launch_doc, sort_keys=False))
        release_path.write_text(yaml.safe_dump(release_doc, sort_keys=False))

    def _numerics(self, launch: lc.Launch, launch_doc: dict) -> None:
        numerics = launch_doc["numerics"]
        approved = launch.path(numerics["approved"]["bundle"])
        numerics["approved"]["manifest_sha256"] = sha256_file(_json(approved / "MANIFEST.json", {"approved": True}))
        signoff = numerics["approved"]["signoff"]
        _write(
            launch.path(signoff["source"]),
            f"intro\n**{signoff['marker']}: {numerics['approved']['id']} "
            f"`{numerics['approved']['manifest_sha256']}` is {signoff['verdict']}.**\n",
        )

    def _materialize(self) -> None:
        self._repin(self.build())
        launch = self.build()
        x = launch.expand
        numerics = launch.document["numerics"]
        bundle = x(numerics["launch_bundle"])
        self.manifest_sha = sha256_file(_json(Path(bundle) / "MANIFEST.json", {"launch": True}))
        _json(
            launch.path(numerics["reuse_proof"]),
            {
                "parent_bundle": x(numerics["approved"]["bundle"]),
                "parent_manifest_sha256": numerics["approved"]["manifest_sha256"],
                "changed_original_files": [],
            },
        )
        _write(
            launch.path(launch.document["inputs"]["private_full_model_spec"]),
            yaml.safe_dump({"numerics_bundle": {"path": bundle}}),
        )
        release = launch.config.release
        root = launch.path(release["prepare"]["output"])
        _json(
            launch.path(numerics["finalization_report"]),
            {"bundle": bundle, "release": str(root), "manifest_sha256": self.manifest_sha},
        )
        _json(
            launch.path(numerics["verify_report"]),
            {"status": "pass", "bundle": {"verify": {"manifest_sha256": self.manifest_sha}}},
        )
        prepare = release["prepare"]
        _json(
            root / "private" / "preparation.json",
            {
                "source_run": x(prepare["run_dir"]),
                "assembly": {"mode": "generated_only", "capsule_scope": {"kind": prepare.get("capsule_scope", "all")}},
                "phase1_policy": {"sha256": prepare["phase1_policy_descriptor"]["sha256"]},
                "scaffolding": {
                    "bundle_generation": {"selection": {"variants": list(prepare["selected_variants"])}},
                    "capability_contract": {
                        "kernel_timing_selection": {"sha256": prepare["kernel_timing_selection"]["sha256"]}
                    },
                },
            },
        )
        seal = release["seal"]
        _json(launch.path(seal["seal"]), {"review": {"reviewed_by": seal["reviewed_by"], "note": seal["review_note"]}})
        _write(launch.path(seal["descriptor"]), "target: fixture\n")
        self.bundle_manifest = launch.path(launch.document["inputs"]["bundle_manifest"])
        self.grant({"allowed": [{"path": "merlin/contract/VERSION", "mode": "ro"}], "host_inputs": []})
        _write(root / "payload" / "corpus" / "public" / "capsule.mlir", "module {}\n")
        self.launch = launch
        self.written = lc.render(launch, self.prep, plan_template=self._template())

    def grant(self, rows: dict) -> None:
        _write(self.bundle_manifest, yaml.safe_dump({"bundle_id": self.build().document["identity"]["bundle"], **rows}))

    def _template(self) -> Path:
        return _json(self.tmp / "plan-template.json", {"state": "authorized", "manifest_sha256": self.manifest_sha})

    def verify(self, **kwargs) -> lc.Report:
        return lc.verify(self.build(run_id=kwargs.pop("run_id", None)), **kwargs)


def _kinds(report: lc.Report) -> list[tuple[str, str]]:
    return [(drift.kind, drift.subject) for drift in report.drifts]


@pytest.fixture
def world(tmp_path):
    return World(tmp_path)


# --------------------------------------------------------------------------- schema and consistency


def test_checked_in_config_loads_and_resolves_without_machine_literals(tmp_path):
    config = lc.load(repo_root() / LAUNCH)
    assert config.release is not None and config.release["target"] == config.document["identity"]["target"]
    task_root = tmp_path / "work" / "tasks" / "t"  # WORKSPACE_ROOT is two levels above the task root
    launch = lc.build(config, task_root=task_root, prep=tmp_path / "p", environ={"HOME": str(tmp_path / "h")})
    assert launch.argv[1:3] == ("-m", "merlin_experiments.phase1")
    flags = lc.split_argv(launch.argv)[1]
    assert flags["--effort"] == ["high"] and flags["--max-wall-s"] == ["18000"] == flags["--round-timeout"]
    assert flags["--codex-max-subagents"] == ["16"] and flags["--grade-interval"] == ["120"]
    assert flags["--sim-max-jobs"] == ["18"] and "--seed-submission" not in flags and "--resume" not in flags
    assert launch.environment["MERLIN_COMPUTE_CPUS"] == ",".join(str(cpu) for cpu in range(6, 24))
    assert launch.environment["MERLIN_CODEX_CPUS"] == "4,5" and launch.manager_cpus == (0,)
    assert launch.environment["MERLIN_SELFCHECK_MAX_CAPSULES"] == "auto"
    assert launch.environment["MERLIN_CAPSULE_ORDER"] == "measured-longest-first"
    forbidden = set(config.document["environment_policy"]["forbidden"])
    assert not forbidden & set(launch.environment), "a per-engine cap would bypass the shared pool"
    # Every machine root is a variable: no checked-in text names this machine's roots.
    for relative in (LAUNCH, RELEASE):
        text = (repo_root() / relative).read_text()
        assert "/work/" not in text and "/home/" not in text and "/tmp/" not in text
    for value in [*launch.argv, *map(str, launch.environment.values())]:
        if value.startswith("/") and "<inherited>" not in value:
            assert value.startswith(str(tmp_path)), value


def test_config_excludes_its_own_directories_and_the_operator_patches():
    # merlin/tests/infra/test_launch_config_not_author_visible.py proves the sandbox side.
    config = lc.load(repo_root() / LAUNCH)
    excluded = {path.rstrip("/"): entry["id"] for entry in config.document["exclusions"] for path in entry["paths"]}
    assert {str(Path(LAUNCH).parent), str(Path(RELEASE).parent)} <= set(excluded)
    assert excluded["examples/gemmini/phase1/patches"] == "conv-input-residency-r39"


def test_capsule_launch_has_explicit_external_owner_and_keeps_private_host_selection(tmp_path):
    def edit(document):
        document["validation_scope"] = "capsules"
        document["external_validation"] = {"budget_s": 7200, "start_after": "capsule_owner_terminal",
                                           "submission": "verified_frozen_capsule_handoff"}
        document["inputs"]["private_full_model_spec"] = "${PREP}/private-models.yaml"
    path = _checkout(tmp_path, edit)
    launch = lc.build(lc.load(path), task_root=tmp_path, prep=tmp_path, environ={"HOME": "/h"})
    flags = lc.split_argv(launch.argv)[1]
    assert flags["--validation-scope"] == ["capsules"]
    assert flags["--private-full-model-spec"] == [str(tmp_path / "private-models.yaml")]
    assert "--instruction-selection" in flags and "--seed-submission" not in flags
    document = yaml.safe_load(path.read_text())
    document.pop("external_validation")
    path.write_text(yaml.safe_dump(document))
    with pytest.raises(lc.ConfigError, match="external_validation"):
        lc.load(path)


def test_legacy_launch_argv_does_not_add_validation_scope(tmp_path):
    launch = lc.build(lc.load(repo_root() / LAUNCH), task_root=tmp_path, prep=tmp_path,
                      environ={"HOME": "/h"})
    assert "--validation-scope" not in launch.argv


def test_launch_renders_explicit_all_author_token_target(tmp_path):
    config = _checkout(tmp_path, lambda doc: doc["author"].update(
        token_target=40_000_000, token_reserve=8_000_000))
    launch = lc.build(lc.load(config), task_root=tmp_path, prep=tmp_path, environ={"HOME": "/h"})
    for flag, expected in (("--author-token-target", "40000000"), ("--author-token-reserve", "8000000")):
        assert launch.argv[launch.argv.index(flag) + 1] == expected


@pytest.mark.parametrize("live_conformance", [False, True])
def test_rendered_argv_is_accepted_by_the_installed_controller_parser(tmp_path, live_conformance):
    from merlin_experiments.phase1 import options

    parser = options.build_parser()
    # The installed entrypoint adds its own required inputs; read them statically, never run it.
    source = Path(options.__file__).with_name("__main__.py").read_text()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument" and node.args:
            flag = node.args[0]
            if isinstance(flag, ast.Constant) and flag.value not in parser._option_string_actions:
                parser.add_argument(flag.value)
    config = _checkout(tmp_path, lambda doc: doc["author"].update(live_conformance_feedback=live_conformance))
    launch = lc.build(lc.load(config), task_root=tmp_path, prep=tmp_path, environ={"HOME": "/h"})
    parsed = parser.parse_args(list(launch.argv[3:]))
    assert parsed.codex_max_subagents == 16 and parsed.sandbox == "bwrap" and parsed.schedule == "continuous"
    assert parsed.live_conformance_feedback is live_conformance


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda d: d.update(unknown_section={}), "Additional properties"),
        (lambda d: d["environment"]["threads"].update(PYTHONUNBUFFERED=1), "PYTHONUNBUFFERED"),
        (lambda d: d["identity"].update(treatment="baseline"), "identity"),
        (lambda d: d["author"].update(driver="claudecode"), "max_subagents"),
        (lambda d: d["environment"]["paths"].update(MERLIN_SPIKE="x"), "more than once"),
        (lambda d: d["environment"]["caches"].update(MERLIN_GSIM_MAX_SLOTS="4"), "forbidden"),
        (lambda d: d["errata"].update(sha256="0" * 64), "errata source"),
        (lambda d: d["cpu"].update(sim_max_jobs=0), "sim_max_jobs"),
    ],
)
def test_schema_and_consistency_refusals(tmp_path, edit, message):
    with pytest.raises(lc.ConfigError, match=message):
        lc.load(_checkout(tmp_path, edit))


def test_variable_resolution_refuses_cycles_unknowns_and_builtin_overrides(tmp_path):
    config = lc.load(_checkout(tmp_path, lambda d: d["variables"].update(A="${B}", B="${A}")))
    with pytest.raises(lc.ConfigError, match="cycle"):
        lc.build(config, task_root=tmp_path, prep=tmp_path, environ={"HOME": "/h"})
    config = lc.load(repo_root() / LAUNCH)
    with pytest.raises(lc.ConfigError, match="undeclared"):
        lc.build(config, task_root=tmp_path, prep=tmp_path, overrides={"NOT_DECLARED": "1"}, environ={"HOME": "/h"})
    with pytest.raises(lc.ConfigError, match="own options"):
        lc.build(config, task_root=tmp_path, prep=tmp_path, overrides={"PREP": "/x"}, environ={"HOME": "/h"})
    with pytest.raises(lc.ConfigError, match="HOME"):
        lc.build(config, task_root=tmp_path, prep=tmp_path, environ={})
    earlier = lc.build(config, task_root="/r", prep="/p", overrides={"RELEASE_VERSION": "18"}, environ={"HOME": "/h"})
    assert lc.split_argv(earlier.argv)[1]["--descriptor"][0].endswith(
        "-timed-v18/payload/experiment/target_experiment.yaml"
    )


def _full_model_verification() -> dict:
    return {
        "comparison": "quantized-reference", "atol": 2e-6, "rtol": 1e-5,
        "split": "visible", "input": "visible/v00",
        "readiness_receipt": "${PREP}/readiness.json", "readiness_sha256": "a" * 64,
    }


def test_full_model_policy_is_operator_configuration_not_native_controller_flags(tmp_path):
    policy = _full_model_verification()
    config = lc.load(_checkout(tmp_path, lambda doc: doc.update(full_model_verification=policy)))
    launch = lc.build(config, task_root=tmp_path, prep=tmp_path, environ={"HOME": "/h"})
    assert launch.document["full_model_verification"] == policy
    assert "--full-comparison" not in launch.argv
    assert launch.path(policy["readiness_receipt"]) == tmp_path / "readiness.json"


@pytest.mark.parametrize("change", [
    {"comparison": "unmeasured"}, {"atol": -1}, {"rtol": float("nan")},
    {"atol": float("inf")}, {"input": "visible/v01"},
    {"readiness_sha256": "missing"}, {"unknown_option": True},
])
def test_full_model_policy_refuses_unbound_or_invalid_comparisons(tmp_path, change):
    policy = {**_full_model_verification(), **change}
    with pytest.raises(lc.ConfigError):
        lc.load(_checkout(tmp_path, lambda doc: doc.update(full_model_verification=policy)))


@pytest.mark.parametrize("scope", [None, "judged_only", "all_declared"])
def test_all_declared_output_scope_requires_an_operator_only_pinned_plan(tmp_path, scope):
    policy = _full_model_verification()
    if scope is not None:
        policy["output_scope"] = scope
    if scope == "all_declared":
        policy["all_output_plan"] = {"path": "${TASK_ROOT}/private/all-output-plan.json", "sha256": "b" * 64}
    config = lc.load(_checkout(tmp_path, lambda doc: doc.update(full_model_verification=policy)))
    assert config.document["full_model_verification"] == policy


@pytest.mark.parametrize(
    "change",
    [
        {"output_scope": "all_declared"},
        {"output_scope": "anything"},
        {"all_output_plan": {"path": "/private/plan.json", "sha256": "b" * 64}},
        {"output_scope": "all_declared", "all_output_plan": {"path": "/private/plan.json", "sha256": "not-a-digest"}},
    ],
)
def test_output_scope_rejects_unbound_or_inapplicable_plan(tmp_path, change):
    with pytest.raises(lc.ConfigError):
        lc.load(
            _checkout(
                tmp_path, lambda doc: doc.update(full_model_verification={**_full_model_verification(), **change})
            )
        )


def test_environment_script_round_trips_through_the_parser(world):
    values, problems = lc.parse_environment_script((world.prep / "environment.sh").read_text())
    assert problems == []
    assert values == world.launch.environment
    hand_written = "export PATH=\"/x/bin:$PATH\"\nexport A=1  # note\nexport B='two words'\n"
    assert lc.parse_environment_script(hand_written) == ({"PATH": lc.Prepend("/x/bin"), "A": "1", "B": "two words"}, [])
    assert lc.parse_environment_script("PATH=/x\nexport C=a b\n")[1] == ["line 1: 'PATH=/x'", "line 2: 'export C=a b'"]


# --------------------------------------------------------------------------- render -> verify


def test_render_then_verify_is_clean_and_render_is_idempotent(world):
    report = world.verify()
    assert report.drifts == [], [drift.line() for drift in report.drifts]
    assert any("release payload content" in subject for subject in report.checked)
    assert (world.prep / "operator-errata.md").read_bytes() == (
        world.config_path.parent / "errata/fresh-empty.md"
    ).read_bytes()
    assert lc.render(world.build(), world.prep, plan_template=world._template()) == world.written
    plan = json.loads((world.prep / "phase1-launch-plan.json").read_text())
    assert plan["state"] == "authorized" and plan["seed_submission"] is None and plan["argv"] == list(world.launch.argv)
    call = json.loads((world.prep / "release-prepare.json").read_text())
    args, kwargs = lc.prepare_arguments(call)
    assert args[1] == world.launch.path(world.launch.config.release["prepare"]["output"])
    assert kwargs["selected_levels"] == ("EL4",) and kwargs["kernel_timing_selection"][0].is_absolute()


def test_render_refuses_to_replace_different_bytes(world):
    environment = world.prep / "environment.sh"
    environment.write_text(environment.read_text() + "export EXTRA=1\n")
    with pytest.raises(lc.ConfigError, match="refusing to replace"):
        lc.render(world.build(), world.prep)
    lc.render(world.build(), world.prep, force=True)
    assert "EXTRA" not in environment.read_text()


def _edit_plan(world, edit) -> None:
    path = world.prep / "phase1-launch-plan.json"
    plan = json.loads(path.read_text())
    edit(plan)
    path.write_text(json.dumps(plan))


def test_changed_flag_is_one_named_drift(world):
    def effort(plan):
        plan["argv"][plan["argv"].index("--effort") + 1] = "xhigh"

    _edit_plan(world, effort)
    report = world.verify()
    assert _kinds(report) == [("flag_value", "plan argv --effort")]
    assert report.drifts[0].expected == "high" and report.drifts[0].actual == "xhigh"


def test_missing_extra_and_seed_flags_are_drifts(world):
    def edit(plan):
        argv = plan["argv"]
        index = argv.index("--codex-max-subagents")
        del argv[index : index + 2]
        argv += ["--seed-submission", "/old/compiler"]
        plan["seed_submission"] = "/old/compiler"

    _edit_plan(world, edit)
    assert sorted(_kinds(world.verify())) == [
        ("extra_flag", "plan argv --seed-submission"),
        ("missing_flag", "plan argv --codex-max-subagents"),
        ("plan_field", "plan seed_submission"),
    ]


def test_extra_cap_variable_and_missing_variable_are_drifts(world):
    path = world.prep / "environment.sh"
    lines = [line for line in path.read_text().splitlines() if not line.startswith("export MERLIN_CAPSULE_ORDER=")]
    path.write_text("\n".join([*lines, "export MERLIN_GSIM_MAX_SLOTS=4", "export MERLIN_UNDECLARED=1"]) + "\n")
    assert sorted(_kinds(world.verify())) == [
        ("extra_env", "environment file MERLIN_UNDECLARED"),
        ("forbidden_env", "environment file MERLIN_GSIM_MAX_SLOTS"),
        ("missing_env", "environment file MERLIN_CAPSULE_ORDER"),
    ]


def test_search_path_must_keep_the_inherited_value_in_a_file(world):
    path = world.prep / "environment.sh"
    text = path.read_text().replace("${PATH:+:$PATH}", "")
    path.write_text(text)
    assert _kinds(world.verify()) == [("env_value", "environment file PATH")]


def test_wrong_runtime_hash_is_a_drift(world):
    spike = Path(world.launch.pins["runtime_pins.spike"][0])
    spike.write_bytes(spike.read_bytes() + b"rebuilt")
    report = world.verify()
    assert _kinds(report) == [("hash", "runtime_pins.spike sha256")]


def test_numerics_lineage_and_release_records_are_checked(world):
    launch = world.launch
    proof = launch.path(launch.document["numerics"]["reuse_proof"])
    proof.write_text(json.dumps({**json.loads(proof.read_text()), "changed_original_files": ["captures/a.bin"]}))
    seal = launch.path(launch.config.release["seal"]["seal"])
    seal.write_text(json.dumps({"review": {"reviewed_by": "someone else", "note": "different"}}))
    assert sorted(_kinds(world.verify())) == [
        ("lineage", "reuse proof changed original files"),
        ("value", "release seal review note"),
        ("value", "release seal reviewed_by"),
    ]


def test_exclusions_cover_grants_references_and_payload_bytes(world):
    world.grant({"allowed": [{"path": "examples/gemmini/phase1/", "mode": "ro"}], "host_inputs": []})
    payload = world.launch.path(world.launch.config.release["prepare"]["output"]) / "payload"
    shutil.copy(world.prep / "operator-errata.md", payload / "ERRATA.md")
    kinds = _kinds(world.verify())
    assert ("exclusion", "conv-input-residency-r39 reachable from bundle allowed") in kinds
    assert ("exclusion", "operator-launch-config reachable from bundle allowed") in kinds
    assert ("exclusion", "operator errata bytes in release payload") in kinds
    assert len(kinds) == 3


def test_missing_prepared_inputs_are_listed_not_raised(world):
    (world.prep / "phase1-launch-plan.json").unlink()
    (world.prep / "environment.sh").unlink()
    assert sorted(_kinds(world.verify())) == [("missing_file", "environment file"), ("missing_file", "launch plan")]


# --------------------------------------------------------------------------- live controller


def _controller(world, tmp_path, run_id, *, extra_env=None, cpu=None):
    """A sleeping stand-in started with the rendered argv, environment and affinity."""
    fake = tmp_path / "standin"
    _write(fake / "merlin_experiments" / "__init__.py", "")
    _write(fake / "merlin_experiments" / "phase1" / "__init__.py", "")
    _write(fake / "merlin_experiments" / "phase1" / "__main__.py", "import time\ntime.sleep(60)\n")
    launch = world.build(run_id=run_id)
    environment = {"HOME": str(world.home), "PATH": os.environ.get("PATH", "/usr/bin")}
    for name, value in launch.environment.items():
        if isinstance(value, lc.Prepend):
            inherited = environment.get(name)
            environment[name] = value.prefix + (":" + inherited if inherited else "")
        else:
            environment[name] = value
    environment.update(launch.launch_environment)
    environment.update(extra_env or {})
    selected = cpu if cpu is not None else launch.manager_cpus[0]
    process = subprocess.Popen(
        list(launch.argv),
        cwd=fake,
        env=environment,
        preexec_fn=lambda: os.sched_setaffinity(0, {selected}),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.monotonic() + 10
    while (
        time.monotonic() < deadline
        and b"merlin_experiments.phase1" not in Path(f"/proc/{process.pid}/cmdline").read_bytes()
    ):
        time.sleep(0.05)
    return process


def test_live_controller_matches_and_drifts_are_named(world, tmp_path):
    run_id = "fixture-run-v1"
    clean = _controller(world, tmp_path, run_id)
    try:
        assert lc._process_run_id(clean.pid) == run_id
        report = world.verify(run_id=run_id, controller_pid=clean.pid)
        assert report.drifts == [], [drift.line() for drift in report.drifts]
    finally:
        clean.kill()
        clean.wait()
    cpus = sorted(os.sched_getaffinity(0))
    other = next((cpu for cpu in cpus if cpu != world.launch.manager_cpus[0]), None)
    drifted = _controller(world, tmp_path, run_id, extra_env={"MERLIN_GRADE_WORKERS": "8"}, cpu=other)
    try:
        kinds = _kinds(world.verify(run_id=run_id, controller_pid=drifted.pid))
    finally:
        drifted.kill()
        drifted.wait()
    assert ("forbidden_env", f"controller {drifted.pid} environment MERLIN_GRADE_WORKERS") in kinds
    if other is not None:
        assert ("affinity", f"controller {drifted.pid} affinity") in kinds


def test_cli_verify_exit_codes(world, capsys):
    common = ["--config", str(world.config_path), "--task-root", str(world.task), "--prep", str(world.prep)]
    common += [f"--var=PYTHON={sys.executable}"]
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("HOME", str(world.home))
        assert lc.main(["verify", *common]) == 0
        assert "0 drift(s)" in capsys.readouterr().out
        _edit_plan(world, lambda plan: plan["argv"].append("--resume"))
        assert lc.main(["verify", *common, "--json"]) == 1
        record = json.loads(capsys.readouterr().out)
        assert record["status"] == "drift" and record["drifts"][0]["subject"] == "plan argv --resume"
        assert lc.main(["verify", *common, "--report", str(world.prep / "r.json")]) == 2


def test_launch_renders_opt_in_token_admission(tmp_path):
    path = _checkout(tmp_path, lambda doc: doc['author'].update(
        token_target=40_000_000, token_reserve=8_000_000,
        token_admission=True, token_turn_allowance=750_000))
    launch = lc.build(lc.load(path), task_root=tmp_path, prep=tmp_path, environ={"HOME": "/h"})
    assert '--author-token-admission' in launch.argv
    assert launch.argv[launch.argv.index('--author-token-turn-allowance') + 1] == '750000'
