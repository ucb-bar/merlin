"""Exact archived inventories are replay data; prospective additions select V8."""

import copy
import hashlib
import json

import pytest

from merlin.compile.scheduling.export import (
    SOURCE_BINDING_MODULES,
    SOURCE_BINDING_PROTOCOL,
    export_shared_scheduling,
    public_modules,
)
from merlin.compile.scheduling.library_sources import library_modules
from merlin.targetgen import tool_profile_shared as S


def declaration(modules, public, *, binding=False):
    return {
        "schema": "merlin.shared_scheduling_library.v2" if binding else "merlin.shared_scheduling_library.v1",
        **({"source_binding_protocol": SOURCE_BINDING_PROTOCOL} if binding else {}),
        "members": [{"module": name} for name in (*modules, None, None)],
        "public_modules": list(public),
    }


def test_frozen_default_inventories_match_original_parent_exactly():
    assert (
        hashlib.sha256(json.dumps(library_modules(6)).encode()).hexdigest()
        == "e4f3e0c56b3b98f2490800c13a4d3afe98e3e634a389c38cab248360f71f675c"
    )
    assert (
        hashlib.sha256(json.dumps(public_modules(6)).encode()).hexdigest()
        == "adac0b1aa0614f9c14d1ff712e4e88fc15fa18dd53aa9a1a6ab7e64a02b45e58"
    )
    assert (
        hashlib.sha256(json.dumps(library_modules(7)).encode()).hexdigest()
        == "52c116f2eee5dd49c5c241b13e3106029e83abe21864fb3e593fee79cfdb8f8d"
    )
    assert (
        hashlib.sha256(json.dumps(public_modules(7)).encode()).hexdigest()
        == "38b62e889184f2e6eeffe667e547bbe16d64b26e2151e10c296e36a431827797"
    )


@pytest.mark.parametrize(
    "omitted",
    [
        (),
        ("merlin.compile.scheduling.window_transfer",),
        ("merlin.compile.scheduling.window_transfer", "merlin.compile.scheduling.llvm_program"),
    ],
)
@pytest.mark.parametrize("binding", [False, True])
def test_exact_incoming_archive_retains_its_own_public_grants(omitted, binding):
    modules = tuple(name for name in S._INCOMING_V6_MODULES if name not in omitted)
    public = tuple(name for name in S._INCOMING_V6_PUBLIC if name not in omitted)
    if binding:
        modules, public = (*modules, *SOURCE_BINDING_MODULES), (*public, *SOURCE_BINDING_MODULES)
    saved = declaration(modules, public, binding=binding)
    assert S._release_inventory(saved, version=6) == (modules, public)
    assert (
        S.profile_inventory(
            {"common_substrate": "merlin.common_compiler_substrate.v6", "shared_scheduling_library": saved}, (), ()
        )[1]
        == public
    )
    assert not set(SOURCE_BINDING_MODULES) & set(public_modules(6))
    # Removing an arbitrary ordinary leaf is not one of the exact archived rosters.
    saved["members"].pop(
        next(i for i, row in enumerate(saved["members"]) if row["module"] == "merlin.compile.scheduling.copy_pipeline")
    )
    with pytest.raises(ValueError, match="inventory"):
        S._release_inventory(saved, version=6)


@pytest.mark.parametrize("version", [6, 7])
@pytest.mark.parametrize("binding", [False, True])
def test_new_v8_release_cannot_be_retagged_as_a_frozen_release(tmp_path, version, binding):
    document = export_shared_scheduling(tmp_path / "new", version=8, source_binding=binding)
    with pytest.raises(ValueError, match="API"):
        S._release_inventory(document, version=version)
    changed = copy.deepcopy(document)
    changed["schema"] = "merlin.shared_scheduling_library.v1" if version == 6 else "merlin.shared_scheduling_library.v2"
    if version == 6:
        changed.pop("inventory_version")
    else:
        changed["inventory_version"] = 7
    # V8 complete source order matches a historical incoming V6 archive ONLY
    # when no new V8 marker remains. Existing original profile hashes still
    # forbid replacing one saved grant with that different archived selection.
    if version == 6 and not binding:
        assert S._release_inventory(changed, version=6) == (library_modules(8), public_modules(8))
    else:
        with pytest.raises(ValueError, match="API|inventory"):
            S._release_inventory(changed, version=version)


@pytest.mark.parametrize("version", [6, 7])
def test_source_binding_requires_new_explicit_inventory_before_writes(tmp_path, version):
    path = tmp_path / "forbidden"
    with pytest.raises(ValueError, match="explicit inventory version 8"):
        export_shared_scheduling(path, version=version, source_binding=True)
    assert not path.exists()


@pytest.mark.parametrize("original_version", [6, 7])
def test_archive_recognition_never_replaces_original_frozen_profile(tmp_path, original_version):
    from merlin.common.paths import repo_root
    from merlin.targetgen.tool_profile import attach_common_substrate, inherited_tool_profile, verify_tool_profile
    from merlin.targetgen.tool_profile_shared import prepare_shared_source_release

    original_root, incoming_root = tmp_path / "original", tmp_path / "incoming"
    prepare_shared_source_release(
        original_root, python_source_root=repo_root() / "src", license_directory=repo_root(), version=original_version
    )
    archived = prepare_shared_source_release(
        incoming_root, python_source_root=repo_root() / "src", license_directory=repo_root(), version=8
    )
    # Build structurally exact historical metadata using disposable current
    # sources. This is a grant-continuity test, not archived-byte qualification.
    archived["schema"] = "merlin.shared_scheduling_library.v1"
    archived.pop("inventory_version")
    archived["library_sha256"] = hashlib.sha256(
        json.dumps({k: v for k, v in archived.items() if k != "library_sha256"}, sort_keys=True).encode()
    ).hexdigest()
    (incoming_root / "shared-library.json").write_text(json.dumps(archived, indent=2) + "\n")

    def bundle(root, version):
        result = dict(arm="raw_baseline", tools=[], allowed=[], denied=[], host_inputs=[])
        attach_common_substrate(result, version=f"merlin.common_compiler_substrate.v{version}", python_source_root=root)
        assert verify_tool_profile(result) is not None
        return result

    original = bundle(original_root, original_version)
    different_archive = bundle(incoming_root, 6)
    with pytest.raises(ValueError, match="Phase 2 changed"):
        inherited_tool_profile(original, different_archive)
