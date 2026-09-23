"""Installed compiler-policy construction with synthetic resources, never tool execution."""

import copy
import hashlib
import importlib.util
import shutil
import socket
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase2 import agent_workspace as AW
from merlin_experiments.phase2 import campaign as C
from merlin_experiments.phase2 import contracts as D
from merlin_experiments.phase2 import emission_analysis as EA
from merlin_experiments.phase2 import qualification_policy as Q
from merlin_experiments.phase2.functional_inputs import FrozenFunctionalInputs
from merlin_experiments.phase2.portfolio_sandbox import PortfolioSandboxFactory, rebind_compiler_sandbox

from merlin.perf.analysis_worker import IsolatedAnalysisWorker
from merlin.targetgen.sandbox import toolchain as TC
from merlin.targetgen.sandbox.answer_surfaces import AnswerSurface


@pytest.fixture(autouse=True)
def no_execution(monkeypatch, tmp_path):
    def refused(*args, **kwargs):
        pytest.fail("compiler-policy tests cannot execute tools or listeners")

    monkeypatch.setattr(subprocess, "Popen", refused)
    monkeypatch.setattr(socket.socket, "bind", refused)
    monkeypatch.setenv("MERLIN_BUNDLE_CAS", str(tmp_path / "cas"))


@pytest.fixture
def case(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "portfolio_sandbox_fixture", Path(__file__).with_name("portfolio_analysis_fixtures.py")
    )
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    case = helper.build_case(tmp_path / "experiment", monkeypatch)

    def write(path, payload=b"synthetic", mode=0o644):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        path.chmod(mode)
        return path

    repo = tmp_path / "tool-context"
    repo.mkdir()
    paths = TC.ToolchainPaths(
        repo, *(str(tmp_path / name) for name in ("venv", "llvm", "compat", "clang", "uv")), python_import_roots=()
    )
    for root in (paths.venv, paths.llvm, paths.compat_lib, paths.clang_bin, paths.clang_resource, paths.uv_python):
        write(Path(root) / "synthetic-tool", mode=0o755)
    sim = tmp_path / "sim"
    write(sim / "bin/sim", mode=0o755)
    secret = write(sim / "answers/private", b"not candidate data")
    selection = C.PackageSandboxInputs(
        paths,
        TC.SimToolchain(
            bind_paths=(str(sim),),
            path_dirs=(str(sim / "bin"),),
            ld_dirs=(),
            env_extra={},
            probes=(TC.ToolProbe("synthetic", "sim --version", str(sim)),),
        ),
        "",
        (AnswerSurface("private", secret.parent, "dir", "oracle"),),
    )
    root = tmp_path / "agent"
    payload = b"public interface"
    write(root / "interface", payload)
    digest = hashlib.sha256(payload).hexdigest()
    aggregate = hashlib.sha256(f"interface\0{digest}\0{len(payload)}\n".encode()).hexdigest()
    manifest = write(
        root / "agent_input_manifest.json",
        D.canonical_json({"files": [{"path": "interface", "sha256": digest, "n_bytes": len(payload)}]}),
    )
    agent = AW.AgentInputSnapshot(root, manifest, D.sha256_file(manifest), aggregate, 1, len(payload))
    marker = write(tmp_path / "functional.json", b"{}")
    functional = FrozenFunctionalInputs(tmp_path, marker, D.sha256_file(marker), "a" * 64, ())
    # Phase-1 V4 provenance admission is separate. Keep real AW policy construction,
    # selected tool mounts, answer masks, agent-input checks and frozen-tool verification.
    monkeypatch.setattr(AW, "_private_functional_surfaces", lambda *_: [])
    base_calls = []
    empty = tmp_path / "public-shared-view"
    empty.mkdir()

    def base(workspace, bundle, **kwargs):
        base_calls.append((workspace, kwargs))
        return [
            "bwrap",
            "--bind",
            str(workspace),
            str(workspace),
            "--ro-bind",
            str(empty),
            str(case.inputs.compiler_shared_source_root),
        ]

    monkeypatch.setattr(AW.BW, "base_argv", base)
    case.selection = selection
    case.options = dict(
        target_experiment=SimpleNamespace(target="synthetic", path=write(tmp_path / "target.json", b"{}")),
        agent_inputs=agent,
        frozen_functional=functional,
        frozen_corpus_manifest=write(tmp_path / "corpus.json", b"{}"),
    )
    case.secret = secret
    case.base_calls = base_calls
    return case


def factory(case, *, frozen=False):
    selection = case.selection
    if frozen:
        selection = Q.restore(case.root, Q.freeze(case.root, case.options["target_experiment"], selection))
    return PortfolioSandboxFactory(case.owner, sandbox_inputs=selection, **case.options)


def scratch(case, name):
    path = case.root / name
    path.mkdir()
    return path


@pytest.mark.parametrize("frozen", [False, True])
def test_two_arm_cache_rebinds_revision_and_scratch_with_detached_records(case, frozen):
    owner = factory(case, frozen=frozen)
    first = owner(case.inputs.baseline, case.candidate, scratch(case, "scratch1"))
    pristine = copy.deepcopy(first)
    assert set(first) == {"baseline", "candidate"}
    assert len(case.base_calls) == 2
    first["candidate"]["command_prefix"].append("returned-record-tamper")
    first["candidate"]["compiler_dependencies"]["shared_sources"].clear()
    (case.candidate / "revision.txt").write_text("new revision")
    second_scratch = scratch(case, "scratch2")
    second = owner(case.inputs.baseline, case.candidate, second_scratch)
    assert len(case.base_calls) == 2
    for arm, record in second.items():
        assert record["policy_reuse"]["answer_masks_changed"] is False
        assert record["scratch_path"] == str(second_scratch)
        assert "returned-record-tamper" not in record["command_prefix"]
        assert record["compiler_dependencies"]["shared_sources"]
        assert not AW.BW.is_exposed(record["command_prefix"], case.secret)
        assert record["compiler_dependencies"]["shared_source_root"] == str(case.inputs.compiler_shared_source_root)
    assert (
        second["candidate"]["compiler_dependencies"]["candidate_sha256"]
        != pristine["candidate"]["compiler_dependencies"]["candidate_sha256"]
    )
    second["candidate"]["answer_surfaces"].clear()
    third = owner(case.inputs.baseline, case.candidate, scratch(case, "scratch3"))
    assert third["candidate"]["answer_surfaces"]


def test_schema_source_overlays_and_worker_configuration_are_explicit(case):
    owner = factory(case)
    policies = owner(case.inputs.baseline, case.candidate, scratch(case, "scratch"))
    schema = case.inputs.contract_root / "schemas/command_buffer.schema.json"
    for record in policies.values():
        argv = record["command_prefix"]
        index = argv.index(str(schema))
        assert argv[index - 1 : index + 2] == ["--ro-bind", str(schema), "/compiler-api/command_buffer.schema.json"]
        assert record["overlay_trees"]
        for root, digest in record["overlay_trees"].items():
            path = Path(root)
            assert D.exact_tree_record(path)["sha256"] == digest
            # The inventory parent is not mounted; each copied mount subtree is read-only.
            assert all(not member.stat().st_mode & 0o222 for member in path.rglob("*"))
            for subtree in path.iterdir():
                assert argv[argv.index(str(subtree)) - 1] == "--ro-bind"
            assert any(member.name == "helper.py" for member in path.rglob("*"))
    owner.install_worker(output=case.root / "workers")
    worker = case.owner.analyzer
    assert isinstance(worker, IsolatedAnalysisWorker)
    assert worker.analysis_source == Path(EA.__file__).resolve()
    assert worker.contract_root == case.inputs.contract_root
    assert worker.sandbox_factory is owner
    from merlin_experiments.frozen_python import inherited_python_command

    assert worker.python_command is inherited_python_command
    replacement = object()
    case.owner.analyzer = replacement
    owner.install_worker(output=case.root / "unused-workers")
    assert case.owner.analyzer is replacement


@pytest.mark.parametrize("mutation", ["overlap", "dependency", "overlay", "mask"])
def test_rebind_refuses_invalid_retained_grants(case, mutation):
    owner = factory(case)
    cached = owner(case.inputs.baseline, case.candidate, scratch(case, "scratch"))["candidate"]
    dependencies = copy.deepcopy(cached["compiler_dependencies"])
    next_scratch = scratch(case, "next-scratch")
    if mutation == "overlap":
        next_scratch = case.candidate
    elif mutation == "dependency":
        dependencies["candidate_sha256"] = "0" * 64
    elif mutation == "overlay":
        root = Path(next(iter(cached["overlay_trees"])))
        member = next(path for path in root.rglob("*") if path.is_file())
        member.chmod(0o644)
        member.write_text("changed")
    else:
        cached["answer_surfaces"] = []
    with pytest.raises(ValueError):
        rebind_compiler_sandbox(cached, package=case.candidate, scratch=next_scratch, dependencies=dependencies)


def test_frozen_tool_tamper_refuses_even_on_cache_hit(case):
    record = Q.freeze(case.root, case.options["target_experiment"], case.selection)
    selected = Q.restore(case.root, record)
    owner = PortfolioSandboxFactory(case.owner, sandbox_inputs=selected, **case.options)
    owner(case.inputs.baseline, case.candidate, scratch(case, "scratch"))
    root = Path(record["snapshot"]["path"])
    file = next(path for path in root.rglob("*") if path.is_file())
    file.chmod(0o644)
    file.write_text("tampered tool")
    file.chmod(0o444)
    with pytest.raises(C.CampaignGateError):
        owner(case.inputs.baseline, case.candidate, scratch(case, "next-scratch"))
    assert len(case.base_calls) == 2


@pytest.mark.parametrize("early_mask", [False, True])
@pytest.mark.parametrize("overlap", ["candidate", "scratch"])
def test_cache_hit_cannot_rebind_inside_frozen_tool_destination(case, monkeypatch, overlap, early_mask):
    if early_mask:
        original_base = AW.BW.base_argv

        def masked_base(*args, **kwargs):
            return [*original_base(*args, **kwargs), "--tmpfs", str(case.secret.parent)]

        monkeypatch.setattr(AW.BW, "base_argv", masked_base)
    owner = factory(case, frozen=True)
    owner(case.inputs.baseline, case.candidate, scratch(case, "scratch"))
    tool_child = Path(case.selection.sim.bind_paths[0]) / "new-child"
    candidate = case.candidate
    next_scratch = scratch(case, "next-scratch")
    if overlap == "candidate":
        shutil.copytree(candidate, tool_child)
        candidate = tool_child
    else:
        tool_child.mkdir()
        next_scratch = tool_child
    with pytest.raises((D.StageGateError, ValueError), match="frozen tool"):
        owner(case.inputs.baseline, candidate, next_scratch)
    assert len(case.base_calls) == 2


@pytest.mark.parametrize("changed", ["schema", "controller"])
def test_selected_source_policy_change_invalidates_both_cached_arms(case, changed):
    owner = factory(case)
    owner(case.inputs.baseline, case.candidate, scratch(case, "scratch"))
    path = (
        case.inputs.controller_source
        if changed == "controller"
        else case.inputs.contract_root / "schemas/command_buffer.schema.json"
    )
    path.write_text("# changed synthetic controller\n" if changed == "controller" else '{"changed_schema": true}')
    records = owner(case.inputs.baseline, case.candidate, scratch(case, "next-scratch"))
    assert len(case.base_calls) == 4
    assert all("policy_reuse" not in record for record in records.values())


def test_agent_input_drift_refuses_before_cache_lookup_or_policy_build(case):
    owner = factory(case)
    owner(case.inputs.baseline, case.candidate, scratch(case, "scratch"))
    (case.options["agent_inputs"].root / "interface").write_bytes(b"changed public input")
    with pytest.raises(D.StageGateError, match="input changed"):
        owner(case.inputs.baseline, case.candidate, scratch(case, "next-scratch"))
    assert len(case.base_calls) == 2
