"""Safe per-artifact host choices run only under original sealed permissions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from merlin.common.paths import repo_root
from merlin.llvmlower.host_optimization_request import HostOptimizationRequest
from merlin.targetgen import host_optimization_admission as H
from merlin.targetgen.compiler_profile_admission import original_compiler_imports
from merlin.targetgen.contract import compile as compiler
from merlin.targetgen.contract.build_recipe import HarnessBuildRecipe, KernelStackFramePolicy
from merlin.targetgen.contract.build_service import BuildOnlyService, file_digest
from merlin.targetgen.contract.schemas import ContractViolation, validate
from merlin.targetgen.sandbox import bwrap
from merlin.targetgen.tool_profile import COMMON_SUBSTRATE_V3, attach_common_substrate

_SOURCE = "module { llvm.func @kernel() { llvm.return } }"
_TARGET = "host_choice_fixture"


def _renderer(*_args, **_kwargs):
    raise AssertionError("object transport cannot execute a harness")


def _original(tmp_path, *, policy=None, status="reviewed", software=True, operator_contract=False, compiler=None):
    repo = repo_root()
    source = repo / "src"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    spec = tmp_path / "software.yaml"
    spec.write_text(yaml.safe_dump({
        "schema": "merlin.software_spec.v1", "status": status, "target": _TARGET,
        **({"host_optimization": policy} if policy is not None else {}),
    }))
    bundle = {"arm": "raw_baseline", "tools": [], "allowed": [], "denied": [], "host_inputs": []}
    attach_common_substrate(
        bundle, version=COMMON_SUBSTRATE_V3, python_source_root=source, host_index_bits=64,
        software_spec_path=spec if software else None, target=_TARGET,
    )
    for row in bundle["allowed"]:
        if row["path"].startswith("merlin/python/"):
            row["path"] = str(source / row["path"].removeprefix("merlin/python/"))
    contract = tmp_path / "target_contract.yaml"
    if operator_contract:
        contract.write_text(yaml.safe_dump({"name": _TARGET, "harness_abi": {"entry_symbol": "kernel"}}))
        bundle["host_inputs"].append({"path": str(contract), "note": "original operator capability selection"})
    archive = tmp_path / "archive.yaml"
    archive.write_text(yaml.safe_dump(bundle))
    bwrap.materialize_bundle_inputs(workspace, bundle, repo=repo)
    script = tmp_path / "unused.ld"
    script.write_text("SECTIONS {}\n")
    clang = Path(compiler).resolve() if compiler is not None else tmp_path / "unused-compiler"
    if compiler is None:
        clang.write_text("identity-only CPU control; never an executable compiler\n")
    recipe = HarnessBuildRecipe(
        clang, (), (), script, 0, cflags=("-march=rv64gc", "-mabi=lp64d"),
        kernel_stack_frame=KernelStackFramePolicy("kernel", 1024),
    )
    service = BuildOnlyService(
        _TARGET, recipe, _renderer,
        tuple((str(path), file_digest(path)) for path in (Path(__file__).resolve(), clang, script)),
    )
    return SimpleNamespace(repo=repo, workspace=workspace, bundle_manifest=archive, service=service, contract=contract)


def _scope(owner):
    return original_compiler_imports(
        workspace=owner.workspace, bundle_manifest=owner.bundle_manifest, repo=owner.repo
    )


def _policy(**extra):
    return {
        "schema": H.POLICY_SCHEMA, "permitted_stages": ["llvm"],
        "permitted_optimizations": ["static_arena"], "sequential_execution": True,
        "successful_allocations": True, "non_reentrant_entry": True, **extra,
    }


def _buffer(source=_SOURCE, **choices):
    return {"abi_version": "0.1", "target": _TARGET, "commands": [],
            "tensors": {"Y": {"shape": [1], "dtype": "i32", "role": "output"}}, "host_optimization_request":
            HostOptimizationRequest(hashlib.sha256(source.encode()).hexdigest(), static_arena=True, **choices).record()}


def test_schema_admits_only_bounded_choices_not_permissions():
    cb = _buffer()
    validate(cb, "command_buffer")
    for key, value in (("index_bits", 32), ("sequential_execution", True), ("pipeline", "anything")):
        with pytest.raises(ContractViolation):
            validate(
                {**cb, "host_optimization_request": {**cb["host_optimization_request"], key: value}}, "command_buffer"
            )


@pytest.mark.parametrize("premise", ["sequential_execution", "successful_allocations", "non_reentrant_entry"])
def test_original_runtime_permission_must_be_explicit(tmp_path, premise):
    policy = _policy()
    del policy[premise]
    owner = _original(tmp_path, policy=policy)
    with _scope(owner), pytest.raises(ValueError, match="premises|non-reentrant"):
        with H.selected_buffer_choices(_buffer(), _SOURCE, target=_TARGET, build_service=owner.service):
            pytest.fail("an identity granted missing runtime permission")
    assert H.active_selection() is None


@pytest.mark.parametrize("settings", [{}, {"software": False}, {"policy": _policy(), "status": "draft"}])
def test_missing_original_software_permissions_refuse(tmp_path, settings):
    owner = _original(tmp_path, **settings)
    with _scope(owner), pytest.raises(ValueError, match="reviewed|permission schema"):
        with H.selected_buffer_choices(_buffer(), _SOURCE, target=_TARGET, build_service=owner.service):
            pytest.fail("unadmitted policy")


def test_actual_source_request_and_live_scope_are_mandatory(tmp_path):
    owner = _original(tmp_path, policy=_policy())
    with pytest.raises(ValueError, match="original compiler admission"):
        with H.selected_buffer_choices(_buffer(), _SOURCE, target=_TARGET, build_service=owner.service):
            pass
    with _scope(owner):
        with pytest.raises(ValueError, match="terminal artifact"):
            with H.selected_buffer_choices(_buffer(), _SOURCE + "\n", target=_TARGET, build_service=owner.service):
                pass
        with pytest.raises((RuntimeError, FileNotFoundError)):
            with H.selected_buffer_choices(_buffer(), _SOURCE, target=_TARGET):
                pass
        with H.selected_buffer_choices(_buffer(), _SOURCE, target=_TARGET, build_service=owner.service) as chosen:
            assert chosen.context.entry == "kernel" and H.active_selection() is chosen
            assert chosen.context.index_bits == 64
    assert H.active_selection() is None


def test_choices_cannot_mutate_midbuild_or_nest(tmp_path):
    owner = _original(tmp_path, policy=_policy())
    cb = _buffer()
    with _scope(owner), pytest.raises(ValueError, match="changed during"):
        with H.selected_buffer_choices(cb, _SOURCE, target=_TARGET, build_service=owner.service):
            with pytest.raises(ValueError, match="replace"):
                with H.selected_buffer_choices(cb, _SOURCE, target=_TARGET, build_service=owner.service):
                    pass
            cb["host_optimization_request"]["static_arena"] = False
    assert H.active_selection() is None


def test_normal_object_path_consumes_original_per_artifact_choice(tmp_path, monkeypatch):
    from merlin.common import invocation_record
    from merlin.llvmlower import toolchain

    clang = Path(toolchain.clang())
    if not toolchain.mlir_translate().is_file() or not clang.is_file():
        pytest.skip("requires the explicitly selected original LLVM translator")
    owner = _original(tmp_path, policy=_policy(), compiler=clang)
    monkeypatch.setattr(compiler, "link_elf", lambda _cb, obj, _work, **_kwargs: obj)
    with _scope(owner):
        obj = compiler.compile_lowered_to_elf(
            _buffer(), _SOURCE, tmp_path / "build", target=_TARGET, inputs={}, _build_service=owner.service,
        )
    assert obj.read_bytes().startswith(b"\x7fELF")
    report = json.loads((obj.parent / "kernel.host_optimization.json").read_text())
    assert report["selection"]["context"]["non_reentrant_entry"] is True
    assert report["products"]["object"]["sha256"] == file_digest(obj)
    assert report["formal_proof"] == "not_issued" and report["runtime"] == "not_established"
    for path in obj.parent.rglob("invocation.json"):
        invocation_record.verify(path)


def test_operator_path_consumes_only_the_original_selected_contract(tmp_path, monkeypatch):
    from merlin.targetgen import target_registry

    owner = _original(tmp_path, policy=_policy(), operator_contract=True)
    selected = SimpleNamespace(
        capability_contract_path=owner.contract,
        load_contract=lambda: yaml.safe_load(owner.contract.read_bytes()),
    )
    monkeypatch.setattr(target_registry, "resolve", lambda target: selected)
    with _scope(owner):
        with H.selected_buffer_choices(_buffer(), _SOURCE, target=_TARGET) as chosen:
            assert chosen.context.entry == "kernel"
        old = owner.contract.read_bytes()
        owner.contract.write_text(yaml.safe_dump({"name": _TARGET, "harness_abi": {"entry_symbol": "another"}}))
        with pytest.raises(ValueError, match="capability contract"):
            with H.selected_buffer_choices(_buffer(), _SOURCE, target=_TARGET):
                pass
        owner.contract.write_bytes(old)


def test_normal_operator_build_does_not_reuse_unbound_legacy_cache(tmp_path, monkeypatch):
    from merlin.runtime.backends import base as backends
    from merlin.targetgen import build_cache, target_registry

    owner = _original(tmp_path, policy=_policy(), operator_contract=True)
    monkeypatch.setattr(target_registry, "resolve", lambda target: SimpleNamespace(
        capability_contract_path=owner.contract,
        load_contract=lambda: yaml.safe_load(owner.contract.read_bytes()),
    ))
    monkeypatch.setattr(backends, "get_backend", lambda target: SimpleNamespace())
    monkeypatch.setattr(build_cache, "reuse", lambda *_args: pytest.fail("unbound legacy cache was consulted"))
    observed = []

    def object_build(source, work, **kwargs):
        assert source == _SOURCE and H.active_selection().request.static_arena
        observed.append(kwargs)
        work.mkdir()
        obj = work / "kernel.o"
        obj.write_bytes(b"build protocol fixture; no machine qualification")
        return obj

    monkeypatch.setattr(compiler, "llvm_mlir_to_object", object_build)
    monkeypatch.setattr(compiler, "link_elf", lambda _cb, obj, _work, **_kwargs: obj)
    with _scope(owner):
        obj = compiler.compile_lowered_to_elf(_buffer(), _SOURCE, tmp_path / "operator", target=_TARGET, inputs={})
    assert observed == [{"target": _TARGET}] and obj.is_file()
