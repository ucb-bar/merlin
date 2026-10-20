"""Original opt-in prompt, task and child transports; no author/compiler execution."""

import hashlib
import json
import subprocess
import sys
import threading
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace

import pytest
import yaml
from merlin_experiments.phase1 import compiler_library as C
from merlin_experiments.phase1 import task_staging as T
from merlin_experiments.phase2.compiler_library_state import freeze_selection
from test_phase1_task_staging import config

from merlin.targetgen import generate_prompt as P
from merlin.targetgen.compiler_library import CompilerLibrarySelection, freeze_compiler_library, scoped_compiler_library
from merlin.targetgen.shared_pass_context import active_shared_pass_requirement, scoped_shared_pass_requirement
from merlin.targetgen.shared_pass_contract import shared_pass_requirement, validate_shared_pass_contract


def library(root, modules=P.SHARED_PASS_PROMPT_APIS):
    sources = {}
    for module in modules:
        parts = module.split(".")
        for i in range(1, len(parts)):
            sources["/".join(parts[:i]) + "/__init__.py"] = ".".join(parts[:i])
        sources[module.replace(".", "/") + ".py"] = module
    for name in sources:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Neutral access fixture; no algorithm/effect qualification.\n")
    contract = freeze_compiler_library(
        root, review_id="neutral-v1-access", public_modules=modules, sources=tuple(sources.items())
    )
    return CompilerLibrarySelection(contract, root)


def admitted(tmp_path):
    selected = library(tmp_path / "original")
    conf = config(tmp_path / "operator")
    ws, run = conf.context.repo / "workspace", conf.context.repo / "run"
    ws.mkdir()
    run.mkdir()
    freeze_selection(selected, run / "compiler_library/python")
    contract = run / "compiler_library/contract.json"
    contract.write_bytes(C._contract_bytes(selected))
    contract.chmod(0o444)
    return replace(conf, shared_pass_requirement=shared_pass_requirement(), shared_pass_library=selected), ws, run


def stage(conf, ws, run, monkeypatch, route="authored"):
    if route == "model":
        conf = replace(conf, experiment="full")
        # Exercise the actual special-body branch without loading a target or model.
        conf = replace(conf, context=replace(conf.context, target="gemmini"))
        monkeypatch.setattr(T, "full_model_task", lambda *a: "original complete-model task\n")
        arm = "merlin_assisted"
    else:
        te = SimpleNamespace(resource_path=lambda _: conf.context.repo / "authored.md", target="neutral", sim_via={})
        monkeypatch.setattr(T, "load_target_experiment", lambda _: te)
        monkeypatch.setattr(T, "load_capability_manifest", lambda _: {})
        monkeypatch.setattr(T, "task_runtime_scope_block", lambda *a, **k: "\noriginal runtime scope\n")
        monkeypatch.setattr(T, "task_tool_inventory_block", lambda *a: "\noriginal tool inventory\n")
        if route == "authored":
            (conf.context.repo / "authored.md").write_text("original authored task\n")
        else:
            monkeypatch.setattr(P, "render_prompt", lambda *a, **k: "original generated task\n")
        arm = "raw_baseline"
    if route == "generated_full":
        from merlin.targetgen import capsule_runner

        conf = replace(conf, experiment="full")
        monkeypatch.setattr(capsule_runner, "qa_loop_adapters", lambda *a, **k: {})
        monkeypatch.setattr(capsule_runner, "qa_checkpoint_adapters", lambda *a, **k: {})
        monkeypatch.setattr(T.LG, "declared_loop_tiers", lambda *a: ())
        monkeypatch.setattr(T.A, "policy_roots", lambda *a: ())
    T.build_task(arm, ws, run, config=conf)
    return conf


def slots():
    return dict(
        target="neutral_accel",
        corpus_rel="public/corpus/",
        corpus_families=[],
        sim_tiers={"L1": "screen"},
        endpoint_kind="command_buffer",
        screen_tier="L1",
        screen_sim="screen",
        tool_stem="neutral-opt",
        kernel_symbol="neutral_kernel",
        endpoint_desc="declared artifact",
        emit_framing="declared artifact",
        emit_symbol_note="",
        grading_model="original numerical gate",
        isa_facts="selected facts",
        isa_spec="",
        dram_contract="",
        termination_contract="",
    )


@pytest.mark.parametrize(
    "experiment,digest",
    [
        ("full", "370bbcbc0267f45f56ec86133091a9d51a0b25fae1ea85ff03d51deaa5dd3401"),
        ("realistic", "ed007bc337c7d0441eb48b970f8ed7f697eeccae7fc935889b20aaf3f794c095"),
    ],
)
def test_legacy_render_exact_prechange_bytes(monkeypatch, experiment, digest):
    monkeypatch.setattr(P, "prompt_slots", lambda *a: slots())
    monkeypatch.setattr(P, "shared_pass_prompt_record", lambda *a, **k: pytest.fail("legacy feature access"))
    te = SimpleNamespace(target="neutral_accel")
    value = P.render_prompt(
        te,
        {"endpoint": {"kind": "command_buffer"}},
        experiment,
        granted_tools=frozenset(),
        shared_pass_requirement=None,
    )
    assert hashlib.sha256(value.encode()).hexdigest() == digest


def test_selected_prompt_template_is_structurally_valid_and_answer_free(tmp_path, monkeypatch):
    selected = library(tmp_path / "library")
    requirement = shared_pass_requirement()
    template = P.shared_pass_manifest_template(requirement)
    checked = validate_shared_pass_contract(template, requirement=requirement)
    assert len(checked["contract"]["families"]) == 9
    assert all(row["status"] == "unknown" for row in checked["contract"]["families"])
    assert checked["contract"]["controls"]["default_policy"] == []
    monkeypatch.setattr(P, "prompt_slots", lambda *a: slots())
    text = P.render_prompt(
        SimpleNamespace(target="neutral_accel"),
        {},
        shared_pass_requirement=requirement,
        shared_pass_library=selected,
        granted_tools=P.SHARED_PASS_PROMPT_APIS,
    )
    assert selected.contract.sha256 in text and str(selected.root) not in text
    assert "during Phase 1 iteration" in text and "actual transformed IR" in text
    assert "Phase 2 tunes only legal alternatives" in text
    assert '"shared_pass_contract"' in text and '"max_repairs"' in text
    assert "qualification remains an independent checker" in text


@pytest.mark.parametrize("fault", ["missing_library", "missing_api", "no_grants", "partial_grants", "stale"])
def test_optin_refuses_unavailable_original_access_before_prompt_reader(tmp_path, monkeypatch, fault):
    selected = library(
        tmp_path / "library", P.SHARED_PASS_PROMPT_APIS[:-1] if fault == "missing_api" else P.SHARED_PASS_PROMPT_APIS
    )
    grants = P.SHARED_PASS_PROMPT_APIS
    if fault == "missing_library":
        selected = None
    elif fault == "no_grants":
        grants = None
    elif fault == "partial_grants":
        grants = grants[:-1]
    elif fault == "stale":
        (selected.root / "merlin/sched/ir/kernel.py").write_text("# changed\n")
    monkeypatch.setattr(P, "prompt_slots", lambda *a: pytest.fail("prompt reader before access validation"))
    with pytest.raises(ValueError):
        P.render_prompt(
            None,
            None,
            shared_pass_requirement=shared_pass_requirement(),
            shared_pass_library=selected,
            granted_tools=grants,
        )


@pytest.mark.parametrize("route", ["authored", "generated", "generated_full", "model"])
def test_every_actual_task_body_receives_same_original_requirement(tmp_path, monkeypatch, route):
    conf, ws, run = admitted(tmp_path)
    conf = stage(conf, ws, run, monkeypatch, route)
    requirement = T.verify_shared_pass_prompt(conf, ws, run)
    assert requirement == shared_pass_requirement()
    assert (ws / "TASK.md").read_bytes() == (run / "TASK.md").read_bytes()
    assert (ws / "TASK.md").read_text().count("## Selected shared-pass compiler requirements (v1)") == 1
    record = json.loads((run / "shared_pass_selection.json").read_bytes())
    assert record["library"]["contract_sha256"] == conf.shared_pass_library.contract.sha256
    for name in ("shared_pass_requirement.json", "shared_pass_selection.json"):
        assert not (run / name).stat().st_mode & 0o222


@pytest.mark.parametrize(
    "fault", ["drop", "delete_both_and_drop", "change_requirement", "change_library", "record", "task"]
)
def test_resume_never_adds_drops_or_rebases_original_requirement(tmp_path, monkeypatch, fault):
    conf, ws, run = admitted(tmp_path)
    stage(conf, ws, run, monkeypatch)
    if fault in {"drop", "delete_both_and_drop"}:
        conf = replace(conf, shared_pass_requirement=None)
        if fault == "delete_both_and_drop":
            (run / "shared_pass_requirement.json").unlink()
            (run / "shared_pass_selection.json").unlink()
    elif fault == "change_requirement":
        changed = shared_pass_requirement()
        changed["limits"]["max_repairs"] = 0
        conf = replace(conf, shared_pass_requirement=changed)
    elif fault == "change_library":
        conf = replace(conf, shared_pass_library=library(tmp_path / "foreign"))
        (conf.shared_pass_library.root / "merlin/sched/ir/kernel.py").write_text("# stale\n")
    elif fault == "record":
        file = run / "shared_pass_selection.json"
        file.chmod(0o644)
        file.write_text("{}")
    else:
        (ws / "TASK.md").write_text("lost original selected requirement")
    with pytest.raises(ValueError):
        T.verify_shared_pass_prompt(conf, ws, run)


def test_cannot_add_requirement_to_legacy_run(tmp_path):
    conf, ws, run = admitted(tmp_path)
    (ws / "TASK.md").write_text("legacy")
    (run / "TASK.md").write_text("legacy")
    with pytest.raises(ValueError, match="cannot resume"):
        T.verify_shared_pass_prompt(conf, ws, run)


def child_setup(tmp_path, monkeypatch):
    conf, ws, run = admitted(tmp_path)
    stage(conf, ws, run, monkeypatch)
    snapshot = CompilerLibrarySelection(conf.shared_pass_library.contract, run / "compiler_library/python")
    return snapshot, run / "compiler_library/contract.json", run / "shared_pass_requirement.json"


def arguments(snapshot, contract, requirement):
    with scoped_compiler_library(snapshot), scoped_shared_pass_requirement(requirement):
        token = C._INITIAL.set((snapshot, contract))
        try:
            return C.child_arguments()
        finally:
            C._INITIAL.reset(token)


def test_original_child_restores_requirement_and_background_thread(tmp_path, monkeypatch):
    snapshot, contract, path = child_setup(tmp_path, monkeypatch)
    requirement = shared_pass_requirement()
    flags = arguments(snapshot, contract, requirement)
    assert len(flags) == 10 and flags[-3] == str(path)
    calls = []

    @C.selected_child
    def child(argv):
        assert argv == ["--ordinary-input", "value"]
        assert active_shared_pass_requirement() == requirement
        body, join = C.background_target(lambda: calls.append(active_shared_pass_requirement()))
        thread = threading.Thread(target=body)
        thread.start()
        thread.join()
        join()
        calls.append(C.child_arguments())
        return 7

    assert child([*flags, "--ordinary-input", "value"]) == 7
    assert calls == [requirement, flags]
    assert active_shared_pass_requirement() is None


@pytest.mark.parametrize("fault", ["omitted", "sha", "foreign", "writable", "late_format"])
def test_child_refuses_missing_foreign_or_changed_requirement(tmp_path, monkeypatch, fault):
    snapshot, contract, path = child_setup(tmp_path, monkeypatch)
    flags = list(arguments(snapshot, contract, shared_pass_requirement()))
    reached = []

    @C.selected_child
    def child(argv):
        reached.append(True)
        if fault == "late_format":
            path.chmod(0o644)
            path.write_text(json.dumps(shared_pass_requirement()))
            path.chmod(0o444)

    if fault == "omitted":
        flags = flags[:-4]
    elif fault == "sha":
        flags[-1] = "0" * 64
    elif fault == "foreign":
        flags[-3] = str(path.with_name("foreign.json"))
    elif fault == "writable":
        path.chmod(0o644)
    with pytest.raises(ValueError):
        child(flags)
    assert reached == ([True] if fault == "late_format" else [])


@pytest.mark.parametrize("selected", [False, True])
def test_installed_cli_threads_optin_without_changing_legacy_options(tmp_path, monkeypatch, selected):
    from merlin_experiments.phase1 import __main__ as CLI
    from merlin_experiments.phase1 import context, controller

    manifest = tmp_path / "manifest.yaml"
    manifest.write_text("{}")
    seen = []
    monkeypatch.setattr(context, "load_context", lambda *a, **k: "original context")
    monkeypatch.setattr(controller, "run", lambda *a, **k: seen.append((a, k)) or 9)
    argv = [
        "--run-id",
        "neutral",
        "--model",
        "explicit",
        "--descriptor",
        str(tmp_path / "descriptor"),
        "--repo",
        str(tmp_path),
        "--bundle-manifest",
        str(manifest),
        "--bundle",
        "selected",
        "--oracle-timing",
        str(tmp_path / "timing"),
    ]
    assert CLI.main([*argv, *(["--require-shared-passes"] if selected else [])]) == 9
    options = seen[0][0][1]
    assert not hasattr(options, "require_shared_passes")
    assert seen[0][1].get("require_shared_passes", False) is selected
    if not selected:
        assert "require_shared_passes" not in seen[0][1]


def test_actual_controller_executes_under_original_requirement_before_author_callback(tmp_path, monkeypatch):
    from merlin_experiments.phase1 import authoring, controller, runtime_environment, session
    from merlin_experiments.phase1.options import parse_options

    conf, ws, run = admitted(tmp_path)
    snapshot = CompilerLibrarySelection(conf.shared_pass_library.contract, run / "compiler_library/python")
    options = parse_options(["--run-id", "neutral", "--model", "explicit", "--no-oracle"])
    prepared = SimpleNamespace(workspace=ws, run_dir=run)
    monkeypatch.setattr(session, "validate_options", lambda *a: None)
    monkeypatch.setattr(
        runtime_environment, "prepare_runtime_environment", lambda *a, **k: SimpleNamespace(refusal=None, account={})
    )
    monkeypatch.setattr(runtime_environment, "applied_environment", lambda *a: nullcontext())

    def prepare(request, transport, stager, **kw):
        # Actual task wrapper, original snapshot; no worker or agent authoring.
        stage(conf, ws, run, monkeypatch)
        return prepared

    monkeypatch.setattr(session, "prepare", prepare)

    from contextlib import contextmanager

    @contextmanager
    def execution(actual):
        assert actual is prepared
        with scoped_compiler_library(snapshot):
            token = C._INITIAL.set((snapshot, run / "compiler_library/contract.json"))
            try:
                yield
            finally:
                C._INITIAL.reset(token)

    monkeypatch.setattr(C, "execution", execution)
    seen = []

    def execute(actual, runtime):
        seen.append(active_shared_pass_requirement())
        assert C.child_arguments()[-3] == str(run / "shared_pass_requirement.json")
        # The existing ordinary intake sees the original active requirement,
        # before any candidate command is invoked.
        from merlin.targetgen.package_runtime import CertFailure, load_package

        with pytest.raises(CertFailure, match="shared_pass_contract"):
            load_package(ws / "submission")
        return 0

    (ws / "submission").mkdir()
    value = P.shared_pass_manifest_template(shared_pass_requirement())
    value.pop("shared_pass_contract")
    value.update(
        artifact_type="mlir_oot_target_backend",
        language="python",
        authoring={"mode": "hand_curated"},
        integrity_exempt=False,
        entrypoints={"tool": "driver.py"},
    )
    for name in (
        "parse",
        "lower_interface_to_target",
        "emit_command_buffer",
        "emit_target_artifact",
        "lower_target_to_llvm",
    ):
        value["commands"][name] = {"argv": ["{tool}", name, "{input_mlir}", "{output_json}"]}
    (ws / "submission/driver.py").write_text("# Never executed\n")
    (ws / "submission/manifest.yaml").write_text(yaml.safe_dump(value))
    monkeypatch.setattr(authoring, "execute", execute)
    assert (
        controller.run(
            conf.context,
            options,
            bundle_manifest=conf.bundle_dir / "manifest.yaml",
            bundle_id=conf.bundle_id,
            oracle_timing=tmp_path / "timing",
            base_environment={},
            compiler_library_selection=conf.shared_pass_library,
            require_shared_passes=True,
        )
        == 0
    )
    assert seen == [shared_pass_requirement()]


def test_custom_task_stager_refuses_before_original_admission(tmp_path):
    from merlin_experiments.phase1.controller import run
    from merlin_experiments.phase1.treatments import Treatment

    with pytest.raises(ValueError, match="custom treatment task stager"):
        run(
            None,
            None,
            bundle_manifest=tmp_path / "manifest",
            bundle_id="neutral",
            oracle_timing=tmp_path / "timing",
            require_shared_passes=True,
            treatment=Treatment(stage_task=lambda *a: pytest.fail("custom stager ran")),
        )


def test_real_fresh_python_child_enters_original_requirement(tmp_path, monkeypatch):
    snapshot, contract, path = child_setup(tmp_path, monkeypatch)
    flags = arguments(snapshot, contract, shared_pass_requirement())
    program = r"""
import json,sys
from merlin_experiments.phase1.compiler_library import selected_child,child_arguments
from merlin.targetgen.shared_pass_context import active_shared_pass_requirement
@selected_child
def main(argv):
    assert argv == ["original-input"]
    assert len(active_shared_pass_requirement()["families"]) == 9
    assert tuple(child_arguments()) == tuple(sys.argv[1:-1])
    print(json.dumps(active_shared_pass_requirement()))
main(sys.argv[1:])
assert active_shared_pass_requirement() is None
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", program, *flags, "original-input"],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout) == shared_pass_requirement()


@pytest.mark.parametrize("mutation", ["admission", "execution", "drop_deleted_records"])
def test_controller_original_selection_survives_early_and_late_callbacks(tmp_path, monkeypatch, mutation):
    from merlin_experiments.phase1 import authoring, controller, runtime_environment, session
    from merlin_experiments.phase1.options import parse_options

    conf, ws, run = admitted(tmp_path)
    stage(conf, ws, run, monkeypatch)
    selected = mutation != "drop_deleted_records"
    if not selected:
        (run / "shared_pass_requirement.json").unlink()
        (run / "shared_pass_selection.json").unlink()
    captured = []
    original_callbacks = T.callbacks

    def callbacks(actual):
        captured.append(actual)
        return original_callbacks(actual)

    monkeypatch.setattr(T, "callbacks", callbacks)
    monkeypatch.setattr(session, "validate_options", lambda *a: None)
    monkeypatch.setattr(
        runtime_environment, "prepare_runtime_environment", lambda *a, **k: SimpleNamespace(refusal=None, account={})
    )
    monkeypatch.setattr(runtime_environment, "applied_environment", lambda *a: nullcontext())
    prepared = SimpleNamespace(workspace=ws, run_dir=run)

    def prepare(*a, **k):
        if mutation == "admission":
            captured[0].shared_pass_requirement["limits"]["max_repairs"] = 0
        return prepared

    monkeypatch.setattr(session, "prepare", prepare)
    monkeypatch.setattr(C, "execution", lambda *a: nullcontext())
    called = []

    def execute(*a):
        called.append(True)
        assert active_shared_pass_requirement() == shared_pass_requirement()
        captured[0].shared_pass_requirement["limits"]["max_repairs"] = 0
        return 0

    monkeypatch.setattr(authoring, "execute", execute)
    options = parse_options(["--run-id", "neutral", "--model", "explicit", "--no-oracle", "--resume"])
    with pytest.raises(ValueError, match="original shared-pass requirement changed|resume refused"):
        controller.run(
            conf.context,
            options,
            bundle_manifest=conf.bundle_dir / "manifest.yaml",
            bundle_id=conf.bundle_id,
            oracle_timing=tmp_path / "timing",
            base_environment={},
            compiler_library_selection=conf.shared_pass_library,
            require_shared_passes=selected,
        )
    assert called == ([True] if mutation == "execution" else [])


def test_nested_sealed_model_child_refuses_before_owner_or_process(tmp_path, monkeypatch):
    from merlin_experiments.phase1.feedback import compiler_profile_child as child
    from merlin_experiments.phase2.contracts import StageGateError

    monkeypatch.setattr(child, "active_compiler_passes", lambda: pytest.fail("owner callback before refusal"))
    monkeypatch.setattr(child, "_prepare_model_child", lambda **k: pytest.fail("sealed child transport created"))
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("child process launched"))
    with scoped_shared_pass_requirement(shared_pass_requirement()):
        with pytest.raises(StageGateError, match="not supported by sealed fresh model/operator children"):
            with child.prepare_model_child(spec_path=tmp_path / "missing", environment={}, package_dir=tmp_path):
                pytest.fail("unsupported sealed child admitted")


def test_nested_sealed_model_child_default_is_inert(tmp_path, monkeypatch):
    from merlin_experiments.phase1.feedback import compiler_profile_child as child

    marker, calls = object(), []
    monkeypatch.setattr(child, "active_compiler_passes", lambda: None)
    monkeypatch.setattr(child, "_prepare_model_child", lambda **k: calls.append(k) or nullcontext(marker))
    with child.prepare_model_child(spec_path=tmp_path / "spec", environment={}, package_dir=tmp_path) as actual:
        assert actual is marker
    assert len(calls) == 1
