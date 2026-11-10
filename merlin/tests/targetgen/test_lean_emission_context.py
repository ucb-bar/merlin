"""Ordinary source/object custody controls, with native/proof edges stubbed.

The original archive/snapshot/import admission is real. The host class checker
returns unsupported or a deliberate refusal; these tests establish no math,
numerical, machine-code or hardware qualification.
"""

from __future__ import annotations

import json

import pytest
import yaml
from test_lean_emission_chain import emission as emission  # noqa: F401

from merlin.common.paths import repo_root
from merlin.targetgen import capsule_common as CC
from merlin.targetgen.common_transforms import CommonTransformService
from merlin.targetgen.sandbox import bwrap
from merlin.targetgen.tool_profile import (
    COMMON_SUBSTRATE_V3,
    attach_common_substrate,
)

_ORIGINAL_VERIFY = CommonTransformService._verify


def _owner(emission, monkeypatch, version=COMMON_SUBSTRATE_V3):
    # The tests intentionally rewrite frozen payloads, including restored mtimes.
    monkeypatch.setenv("MERLIN_BUNDLE_CAS", "")
    monkeypatch.setattr(CommonTransformService, "_verify", _ORIGINAL_VERIFY)
    repo = repo_root()
    workspace = emission["tmp"] / "workspace"
    workspace.mkdir()
    bundle = {"arm": "raw_baseline", "tools": [], "allowed": [], "denied": [], "host_inputs": []}
    attach_common_substrate(bundle, version=version, python_source_root=repo / "src", host_index_bits=64)
    for row in bundle["allowed"]:
        if row["path"].startswith("merlin/python/"):
            row["path"] = str(repo / "src" / row["path"].removeprefix("merlin/python/"))
    archive = emission["tmp"] / "frozen.yaml"
    archive.write_text(yaml.safe_dump(bundle))
    bwrap.materialize_bundle_inputs(workspace, bundle, repo=repo)
    return CommonTransformService(workspace=workspace, bundle_manifest=archive, repo=repo)


@pytest.fixture
def ordinary(emission, monkeypatch):
    from merlin.llvmlower import codegen, pipeline
    from merlin.targetgen import host_optimization_admission

    owner = _owner(emission, monkeypatch)

    # The actual original ABI selector is independently covered by its own
    # frozen-contract controls; this fixture supplies only the native stub ABI.
    def original_entry(admission, target, build_service):
        assert admission.workspace == owner.workspace and build_service is None and target is None
        admission.verify()
        return "fixture_entry", None

    monkeypatch.setattr(host_optimization_admission, "_original_entry", original_entry)
    monkeypatch.setattr(pipeline, "lower_to_llvm_ir", lambda *_a, **_k: "define void @fixture_entry() { ret void }\n")

    def object_stub(source, output, *_args, **_kwargs):
        output.write_bytes(b"native boundary stub; not executable")
        return output

    monkeypatch.setattr(codegen, "compile_ll", object_stub)
    return emission, owner


def _scope(owner):
    from merlin_experiments.phase1.feedback.compiler_profile_runtime import original_compiler_imports

    return original_compiler_imports(owner)


def _lower(fixture, name="generated", **options):
    return CC.lower_interface(
        fixture["package"],
        fixture["original_source"],
        fixture["tmp"] / name,
        contract=fixture["contract"],
        timeout=30,
        **options,
    )


def _reports(root):
    return [json.loads(path.read_text()) for path in sorted(root.rglob("emission_proof_chain.json"))]
