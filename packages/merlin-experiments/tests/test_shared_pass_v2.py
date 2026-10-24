"""Prospective v2 structural-only tests; no author, compiler or hardware job."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
from test_shared_pass_authoring import admitted, library, stage

from merlin.targetgen import generate_prompt as P
from merlin.targetgen import shared_pass_contract as S


def manifest(*, required=()):
    requirement = S.shared_pass_requirement(version=2, required_supported=required)
    value = P.shared_pass_manifest_template(requirement)
    for row in value["shared_pass_contract"]["families"]:
        if row["name"] in required:
            row.update(status="supported", reason="Declared interface only; original qualification remains separate")
            row["boundary"] = "device" if row["name"] == "buffering_pipelining" else "computation"
    controls = value["shared_pass_contract"]["controls"]
    controls["default_policy"] = sorted(required, key=lambda n: n == "buffering_pipelining")
    for name in required:
        controls["choices"][name]["enabled"] = [False, True]
    return requirement, value


def test_v1_remains_exact_and_v2_buffering_is_explicit():
    v1, v2 = S.shared_pass_requirement(), S.shared_pass_requirement(version=2)
    assert v1["schema"] == "merlin.shared_pass_requirement.v1" and len(v1["families"]) == 9
    assert v2["schema"] == "merlin.shared_pass_requirement.v2" and len(v2["families"]) == 10
    assert set(v2["families"]) - set(v1["families"]) == {"buffering_pipelining"}
    assert {
        "physical_ranges",
        "alias_lifetime",
        "dependency_scheduling",
        "temporary_allocation",
        "completion",
        "resource_bounds",
    } == set(v2["families"]["buffering_pipelining"])
    with pytest.raises(ValueError, match="unknown families"):
        S.shared_pass_requirement(required_supported=("buffering_pipelining",))


@pytest.mark.parametrize("version", [True, False, 0, 3, "2"])
def test_closed_version(version):
    with pytest.raises(ValueError, match="version"):
        S.shared_pass_requirement(version=version)


def test_v2_supported_fusion_and_pipeline_default_controls_are_mandatory():
    required = ("multi_output_fusion", "buffering_pipelining")
    requirement, value = manifest(required=required)
    assert S.validate_shared_pass_contract(value, requirement=requirement)["contract"]["controls"][
        "default_policy"
    ] == list(required)
    assert S.validate_shared_pass_requirement(requirement) == requirement
    value["shared_pass_contract"]["controls"]["default_policy"].remove("buffering_pipelining")
    with pytest.raises(ValueError, match="every supported family"):
        S.validate_shared_pass_contract(value, requirement=requirement)


@pytest.mark.parametrize("kind", ["resources", "completion"])
def test_v2_boundary_needs_explicit_resource_completion_contracts(kind):
    requirement, value = manifest(required=("buffering_pipelining",))
    value["shared_pass_contract"]["boundaries"]["device"]["contracts"].pop(kind)
    with pytest.raises(ValueError, match="boundary contracts"):
        S.validate_shared_pass_contract(value, requirement=requirement)


@pytest.mark.parametrize("fault", ["schema", "unknown", "no_row", "no_completion", "disabled"])
def test_v2_no_silent_requirement_downgrade(fault):
    requirement, value = manifest(required=("multi_output_fusion", "buffering_pipelining"))
    declaration = value["shared_pass_contract"]
    row = next(row for row in declaration["families"] if row["name"] == "buffering_pipelining")
    if fault == "schema":
        declaration["schema"] = "merlin.shared_pass_contract.v1"
    elif fault == "unknown":
        row["status"] = "unknown"
    elif fault == "no_row":
        declaration["families"].remove(row)
    elif fault == "no_completion":
        row["requirements"].remove("completion")
    else:
        declaration["controls"]["choices"]["buffering_pipelining"]["enabled"] = [False]
    with pytest.raises(ValueError):
        S.validate_shared_pass_contract(value, requirement=requirement)


def test_v2_actual_task_and_child_records_keep_original_required_interfaces(tmp_path, monkeypatch):
    from merlin_experiments.phase1 import compiler_library as C
    from merlin_experiments.phase1 import task_staging as T

    from merlin.targetgen.compiler_library import CompilerLibrarySelection, scoped_compiler_library
    from merlin.targetgen.shared_pass_context import active_shared_pass_requirement, scoped_shared_pass_requirement

    conf, ws, run = admitted(tmp_path)
    requirement = S.shared_pass_requirement(
        version=2, required_supported=("multi_output_fusion", "buffering_pipelining")
    )
    conf = replace(conf, shared_pass_requirement=requirement)
    stage(conf, ws, run, monkeypatch)
    assert T.verify_shared_pass_prompt(conf, ws, run) == requirement
    assert "(v2)" in (ws / "TASK.md").read_text()
    snapshot = CompilerLibrarySelection(conf.shared_pass_library.contract, run / "compiler_library/python")
    with scoped_compiler_library(snapshot), scoped_shared_pass_requirement(requirement):
        token = C._INITIAL.set((snapshot, run / "compiler_library/contract.json"))
        try:
            flags = C.child_arguments()
        finally:
            C._INITIAL.reset(token)

    @C.selected_child
    def child(argv):
        assert active_shared_pass_requirement() == requirement

    child(flags)
    (run / "shared_pass_requirement.json").unlink()
    (run / "shared_pass_selection.json").unlink()
    with pytest.raises(ValueError, match="removed"):
        T.verify_shared_pass_prompt(replace(conf, shared_pass_requirement=None), ws, run)


@pytest.mark.parametrize("fault", [None, "oversized", "tampered", "duplicate"])
def test_full_model_v2_resource_staging_preserves_exact_block_and_base_budget(tmp_path, monkeypatch, fault):
    from merlin_experiments.phase1 import resource_prompt
    from merlin_experiments.phase1 import task_staging as T

    conf, ws, run = admitted(tmp_path)
    conf = replace(
        conf,
        context=replace(conf.context, target="gemmini"),
        experiment="full",
        bundle_id="merlin_assisted_rtlchecks_public_v0",
        shared_pass_requirement=S.shared_pass_requirement(version=2),
    )
    monkeypatch.setattr(T, "load_target_experiment", lambda _: object())
    T.build_task(
        "merlin_assisted",
        ws,
        run,
        config=conf,
        task_scope={"required_public_dev_capsules": 153, "held_out_capsules": 8},
    )
    block = P.shared_pass_prompt_block(T._shared_pass_record(conf, run))
    body = (ws / "TASK.md").read_text()
    assert len(body.encode()) > 12 * 1024 and len(body.replace(block, "").encode()) < 10_000
    if fault == "oversized":
        body += "x" * (12 * 1024)
    elif fault == "tampered":
        body = body.replace(block, block.replace("The original host", "An unrelated host", 1))
    elif fault == "duplicate":
        body += block
    for root in (ws, run):
        (root / "TASK.md").write_text(body)
    if fault is not None:
        with pytest.raises(RuntimeError if fault == "oversized" else ValueError):
            resource_prompt.append_resource_guidance(ws, run, object(), task_staging_config=conf)
        return
    from merlin.targetgen.public_codegen_budget import validate_policy

    policy = validate_policy(
        {
            "required": True,
            "timeout_s": 120,
            "capsules": ["public_a"],
            "device_ir_budget": {"mode": "enforce", "max_code_bytes": 1048576},
        }
    )
    resource_prompt.append_resource_guidance(
        ws,
        run,
        object(),
        public_codegen_policy=policy,
        task_staging_config=conf,
    )
    final = (ws / "TASK.md").read_text()
    assert final.count(block) == 1 and final.startswith(body)
    assert "max_code_bytes=1048576" in final and "mandatory for completion: **True**" in final
    assert len(final.replace(block, "").encode()) <= 12 * 1024
    assert final == (run / "TASK.md").read_text()
    assert T.verify_shared_pass_prompt(conf, ws, run) == conf.shared_pass_requirement


def test_installed_cli_forwards_exact_v2_original_selection(tmp_path, monkeypatch):
    from merlin_experiments.phase1 import __main__ as CLI
    from merlin_experiments.phase1 import context, controller

    (tmp_path / "manifest.yaml").write_text("{}")
    seen = []
    monkeypatch.setattr(context, "load_context", lambda *a, **k: "original")
    monkeypatch.setattr(controller, "run", lambda *a, **k: seen.append(k) or 0)
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
        str(tmp_path / "manifest.yaml"),
        "--bundle",
        "selected",
        "--oracle-timing",
        str(tmp_path / "timing"),
        "--require-shared-passes",
        "--shared-pass-contract-version",
        "2",
        "--require-supported-shared-pass",
        "multi_output_fusion",
        "--require-supported-shared-pass",
        "buffering_pipelining",
    ]
    assert CLI.main(argv) == 0
    assert seen[0]["shared_pass_contract_version"] == 2
    assert seen[0]["required_supported_shared_passes"] == ("multi_output_fusion", "buffering_pipelining")
    seen.clear()
    with pytest.raises(SystemExit):
        CLI.main([arg for arg in argv if arg != "--require-shared-passes"])
    assert seen == []


def test_controller_builds_original_v2_requirement_before_admission(tmp_path, monkeypatch):
    from contextlib import nullcontext

    from merlin_experiments.phase1 import controller, runtime_environment, session, task_staging
    from merlin_experiments.phase1.options import parse_options

    selected = library(tmp_path / "library")
    seen = []
    callbacks = task_staging.callbacks
    monkeypatch.setattr(
        task_staging, "callbacks", lambda config: seen.append(config.shared_pass_requirement) or callbacks(config)
    )
    monkeypatch.setattr(session, "validate_options", lambda *a: None)
    monkeypatch.setattr(session, "prepare", lambda *a, **k: 0)
    monkeypatch.setattr(
        runtime_environment, "prepare_runtime_environment", lambda *a, **k: SimpleNamespace(refusal=None, account={})
    )
    monkeypatch.setattr(runtime_environment, "applied_environment", lambda *a: nullcontext())
    options = parse_options(["--run-id", "neutral", "--model", "explicit", "--no-oracle"])
    context = SimpleNamespace(
        readback_policy=None, candidate_policy=None, descriptor=tmp_path / "descriptor", target="neutral"
    )
    required = ("multi_output_fusion", "buffering_pipelining")
    assert (
        controller.run(
            context,
            options,
            bundle_manifest=tmp_path / "manifest",
            bundle_id="selected",
            oracle_timing=tmp_path / "timing",
            base_environment={},
            compiler_library_selection=selected,
            require_shared_passes=True,
            shared_pass_contract_version=2,
            required_supported_shared_passes=required,
        )
        == 0
    )
    assert seen == [S.shared_pass_requirement(version=2, required_supported=required)]


def test_requirement_tool_writes_exact_v2_original_selection(tmp_path):
    import json

    from merlin.targetgen.oot_starterkit.passes import main

    path = tmp_path / "requirement.json"
    assert (
        main(
            [
                "requirement",
                "--contract-version",
                "2",
                "--require-supported",
                "buffering_pipelining",
                "--output",
                str(path),
            ]
        )
        == 0
    )
    assert json.loads(path.read_text()) == S.shared_pass_requirement(
        version=2, required_supported=("buffering_pipelining",)
    )
    with pytest.raises(SystemExit):
        main(["requirement", "--require-supported", "buffering_pipelining"])
