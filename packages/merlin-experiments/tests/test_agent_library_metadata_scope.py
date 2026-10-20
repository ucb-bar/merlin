"""Real small original snapshot/import owners; no author or target execution."""

import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from merlin_experiments.phase2 import compiler_libraries as L
from merlin_experiments.phase2 import functional_inputs as FI
from merlin_experiments.phase2.agent_workspace import AgentSandboxPolicy
from merlin_experiments.phase2.contracts import sha256_file
from test_compiler_libraries import selected
from test_compiler_profile_child import original as original

from merlin.common import bundle_snapshot as S
from merlin.common.cpu_pool import current_thread_owns_cpu_lease
from merlin.targetgen import compiler_profile_admission as API
from merlin.targetgen.compiler_library import scoped_compiler_library
from merlin.targetgen.sandbox import bwrap as B


@pytest.fixture
def admitted(original, tmp_path):
    owner = original.owner
    manifest = owner.bundle_manifest
    environment = manifest.parent / "environment.yaml"
    environment.write_text(
        yaml.safe_dump(
            {
                "bundle_input_snapshot": B.snapshot_record(owner.workspace),
                "workspace_path": str(owner.workspace),
                "bundle_manifest_file": manifest.name,
                "bundle_manifest_sha256": sha256_file(manifest),
                "resolved_tools": owner.bundle["tools"],
                "compiler_tool_profile": owner.profile,
            }
        )
    )
    functional = SimpleNamespace(
        host_provenance={
            "environment": str(environment),
            "environment_sha256": sha256_file(environment),
            "bundle_manifest": str(manifest),
            "bundle_manifest_sha256": sha256_file(manifest),
        }
    )
    library_root = tmp_path / "reviewed"
    library_root.mkdir()
    selection, candidate = selected(library_root)
    policy = AgentSandboxPolicy(
        (),
        (),
        "available_not_an_isolation_claim",
        True,
        True,
        True,
        package=candidate,
        compiler_library_selection=selection,
        compiler_library_functional=functional,
    )
    return original, functional, selection, policy


def test_actual_metadata_getters_share_only_full_entry_and_exit_reads(admitted, monkeypatch):
    original, functional, selection, policy = admitted
    calls, verifications, provenance_scopes = [], [], []
    digest = S.hashlib.file_digest
    provenance = FI._verify_private_functional_provenance
    verify = API._Admission.verify

    def counted(stream, *args):
        calls.append(Path(stream.name))
        return digest(stream, *args)

    def checked_provenance(value):
        provenance_scopes.append(S._ACTIVE_VERIFICATION.get())
        return provenance(value)

    def checked_admission(self):
        verifications.append(S._ACTIVE_VERIFICATION.get())
        return verify(self)

    monkeypatch.setattr(S.hashlib, "file_digest", counted)
    monkeypatch.setattr(FI, "_verify_private_functional_provenance", checked_provenance)
    monkeypatch.setattr(API._Admission, "verify", checked_admission)
    with scoped_compiler_library(selection):
        # The original unbatched lexical operation is a comparison control.
        checked_provenance(functional.host_provenance)
        with API.original_compiler_imports(
            workspace=original.owner.workspace, bundle_manifest=original.owner.bundle_manifest, repo=original.owner.repo
        ):
            expected = L.apply(policy)
        assert calls.count(original.frozen) == 6
        calls.clear()
        verifications.clear()
        result = L.apply_agent(policy)
        assert result == expected
        assert calls.count(original.frozen) == 3  # original provenance + fresh scope entry/exit
        assert len(verifications) == 4 and all(row is not None for row in verifications)
        assert provenance_scopes == [None, None]
        assert S._ACTIVE_VERIFICATION.get() is None
        assert API.active_compiler_imports() is None
        # A later call reopens all three reads; no successful result is cached.
        calls.clear()
        assert L.apply_agent(policy) == result
        assert calls.count(original.frozen) == 3


@pytest.mark.parametrize("changed", ["payload", "marker", "manifest"])
def test_final_change_refuses_already_constructed_mount_metadata(admitted, monkeypatch, changed):
    original, _, selection, policy = admitted
    apply = L.apply

    def changed_after_apply(value):
        result = apply(value)
        path = {
            "payload": original.frozen,
            "marker": S.bundle_snapshot_root(original.owner.workspace) / S.MARKER,
            "manifest": original.owner.bundle_manifest,
        }[changed]
        before = path.stat()
        path.chmod(0o600)
        data = path.read_bytes()
        path.write_bytes(bytes([data[0] ^ 1]) + data[1:])
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
        return result

    monkeypatch.setattr(L, "apply", changed_after_apply)
    with scoped_compiler_library(selection), pytest.raises((RuntimeError, ValueError)):
        L.apply_agent(policy)
    assert S._ACTIVE_VERIFICATION.get() is None
    assert API._issued_compiler_imports() is None


def test_scope_releases_cpu_before_metadata_body_and_no_selection_keeps_default(admitted, monkeypatch):
    _, _, selection, policy = admitted
    with scoped_compiler_library(selection), L.original_agent_sources(policy.compiler_library_functional):
        assert not current_thread_owns_cpu_lease()
        assert API.active_compiler_imports() is not None
    assert S._ACTIVE_VERIFICATION.get() is None

    def unexpected(*args, **kwargs):
        pytest.fail("default policy entered original payload/import scope")

    monkeypatch.setattr(S, "verified_snapshot_scope", unexpected)
    default = replace(policy, compiler_library_selection=None, compiler_library_functional=None)
    assert L.apply_agent(default) is default
