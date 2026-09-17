"""Hostile and end-to-end tests for the v4 public/private build boundary."""
from __future__ import annotations

import json
import hashlib
import os
import shutil
import struct
from pathlib import Path

import pytest

from merlin.compare.paper_build_bundle import (
    TOOL_ROLES,
    VerifiedBuildBarrier,
    build_and_relink_synthetic,
    load_multi_toolchain_authority,
    load_private_session_bundle,
    materialize_private_session_bundle,
    materialize_synthetic_public_closure,
    run_after_barrier,
    snapshot_public_build_bundle,
    verify_build_barrier,
    verify_public_build_bundle,
    write_multi_toolchain_authority,
)
from merlin.compare.paper_session_abi import (
    InputEndpoint,
    InputFrame,
    decode_response,
    encode_request,
)
from merlin.llvmlower.toolchain import clang


def _tools() -> dict[str, str]:
    names = {
        "linker": "ld",
        "cmake": "cmake", "ninja": "ninja",
    }
    resolved = {role: shutil.which(name) for role, name in names.items()}
    missing = sorted(role for role, path in resolved.items() if path is None)
    if missing:
        pytest.skip(f"host build tools are unavailable: {missing}")
    return {
        "c_compiler": str(clang()), "cxx_compiler": str(clang()),
        **{role: str(path) for role, path in resolved.items()},
    }


def _host_target() -> dict[str, object]:
    return {
        "name": "host_tracer_x86_64", "target_triple": "x86_64-unknown-linux-gnu",
        "march": "x86-64", "mabi": "sysv", "features": [], "elf_class": 64,
        "elf_machine": 62, "elf_osabi": 0, "elf_flags_mask": 0xFFFFFFFF,
        "elf_flags_value": 0,
    }


def _compiler_resources() -> dict[str, Path]:
    compiler = Path(clang()).resolve()
    roots = sorted((compiler.parent.parent / "lib/clang").iterdir())
    if not roots:
        pytest.skip("Merlin clang resource directory is unavailable")
    return {"compiler_resource_dir": roots[-1]}


def _request(descriptor) -> bytes:
    values = (2, 1, 4, 5)
    return encode_request(descriptor, [
        InputFrame(InputEndpoint(program, input_index), step, struct.pack(">Q", value))
        for (program, input_index, step), value
        in zip(descriptor.required_input_keys, values, strict=True)
    ])


def _setup(tmp_path: Path, *, c_compiler: str | None = None):
    closure = materialize_synthetic_public_closure(tmp_path / "public" / "build_closure")
    public_manifest = tmp_path / "public" / "public_build_bundle.json"
    public = snapshot_public_build_bundle(closure, public_manifest)
    sysroot = tmp_path / "toolchain" / "sysroot"
    (sysroot / "include").mkdir(parents=True)
    (sysroot / "include" / "identity.h").write_text(
        "/* synthetic public sysroot identity */\n", encoding="utf-8")
    tools = _tools()
    if c_compiler is not None:
        tools["c_compiler"] = c_compiler
    authority_path = tmp_path / "toolchain" / "authority.json"
    authority = write_multi_toolchain_authority(
        authority_path, tools=tools, sysroot=sysroot,
        static_libraries={"public_anchor": closure / "lib" / "libpublic_anchor.a"},
        tree_resources=_compiler_resources(), file_resources={}, target_abi=_host_target())
    receipt = build_and_relink_synthetic(
        public_manifest, authority_path, tmp_path / "rebuilt")
    barrier = verify_build_barrier(public_manifest, authority_path, receipt)
    return public, authority, receipt, barrier


def test_public_closure_multi_tool_authority_and_private_barrier_end_to_end(tmp_path):
    public, authority, receipt, barrier = _setup(tmp_path)
    assert {tool.role for tool in authority.tools} == TOOL_ROLES
    assert authority.target_abi.to_dict() == _host_target()
    assert {resource.name for resource in authority.tree_resources} == {
        "compiler_resource_dir"}
    assert {row["path"] for row in public.files} == {
        "descriptor/session_descriptor.json", "lib/libpublic_anchor.a",
        "resource_roles.json",
        "sources/model_session.c", "sources/runner.c", "sources/support.cc",
    }
    assert [row[0] for row in authority.static_libraries] == ["public_anchor"]
    assert [row["path"] for row in authority.sysroot_files] == ["include/identity.h"]

    request = _request(barrier.descriptor)
    reference = run_after_barrier(barrier, request)
    private_manifest = materialize_private_session_bundle(
        tmp_path / "private", request=request, reference_response=reference,
        descriptor=barrier.descriptor)
    private = load_private_session_bundle(private_manifest.parent, barrier=barrier)
    actual = run_after_barrier(barrier, private.request)
    assert actual == private.reference_response
    decoded = decode_response(actual, expected_descriptor=barrier.descriptor)
    assert [struct.unpack(">Q", frame.payload)[0] for frame in decoded.outputs] == [7, 25, 80]
    assert receipt.is_file()


def test_private_bundle_cannot_be_touched_before_a_verified_barrier(tmp_path):
    private_root = tmp_path / "not-readable-yet"
    with pytest.raises(PermissionError, match="barrier is required before private I/O"):
        load_private_session_bundle(private_root, barrier=None)
    with pytest.raises(TypeError, match="only from verify_build_barrier"):
        VerifiedBuildBarrier(
            public_manifest=tmp_path / "public", authority_path=tmp_path / "authority",
            receipt_path=tmp_path / "receipt", receipt_sha256="0" * 64,
            runner=tmp_path / "runner", composite_object=tmp_path / "object",
            descriptor=None, _seal=object(),  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("relative", [
    "sources/model_session.c", "lib/libpublic_anchor.a",
])
def test_omitted_public_file_or_static_library_revokes_private_access(tmp_path, relative):
    public, _authority, _receipt, barrier = _setup(tmp_path)
    private_root = tmp_path / "private"
    materialize_private_session_bundle(
        private_root, request=b"request", reference_response=b"response",
        descriptor=barrier.descriptor)
    (public.closure_root / relative).unlink()
    with pytest.raises(ValueError, match="public build closure tree differs"):
        load_private_session_bundle(private_root, barrier=barrier)


def test_replaced_compiler_revokes_the_verified_barrier_before_private_io(tmp_path):
    wrapper = tmp_path / "tools" / "cc-wrapper"
    wrapper.parent.mkdir(parents=True)
    real_cc = _tools()["c_compiler"]
    wrapper.write_text(f'#!/bin/sh\nexec "{real_cc}" "$@"\n', encoding="utf-8")
    wrapper.chmod(0o755)
    _public, _authority, _receipt, barrier = _setup(tmp_path, c_compiler=str(wrapper))
    private_root = tmp_path / "private"
    materialize_private_session_bundle(
        private_root, request=b"request", reference_response=b"response",
        descriptor=barrier.descriptor)
    wrapper.write_text(
        f'#!/bin/sh\n# replaced after build\nexec "{real_cc}" "$@"\n', encoding="utf-8")
    wrapper.chmod(0o755)
    with pytest.raises(ValueError, match="tool identity differs.*c_compiler"):
        load_private_session_bundle(private_root, barrier=barrier)


def test_public_manifest_traversal_and_tree_symlinks_fail_closed(tmp_path):
    closure = materialize_synthetic_public_closure(tmp_path / "public" / "closure")
    manifest_path = tmp_path / "public" / "public_build_bundle.json"
    snapshot_public_build_bundle(closure, manifest_path)
    document = json.loads(manifest_path.read_text(encoding="ascii"))
    document["closure_root"] = "../outside"
    manifest_path.write_text(json.dumps(document), encoding="ascii")
    with pytest.raises(ValueError, match="without traversal"):
        verify_public_build_bundle(manifest_path)

    other = materialize_synthetic_public_closure(tmp_path / "symlinked" / "closure")
    os.symlink(other / "sources" / "model_session.c", other / "sources" / "alias.c")
    with pytest.raises(ValueError, match="contains a symlink"):
        snapshot_public_build_bundle(
            other, tmp_path / "symlinked" / "public_build_bundle.json")


def test_private_payload_traversal_and_symlink_are_rejected_after_barrier(tmp_path):
    _public, _authority, _receipt, barrier = _setup(tmp_path)
    private_root = tmp_path / "private"
    manifest_path = materialize_private_session_bundle(
        private_root, request=b"request", reference_response=b"response",
        descriptor=barrier.descriptor)
    document = json.loads(manifest_path.read_text(encoding="ascii"))
    document["request"]["path"] = "../request.bin"
    manifest_path.write_text(json.dumps(document), encoding="ascii")
    with pytest.raises(ValueError, match="without traversal"):
        load_private_session_bundle(private_root, barrier=barrier)

    manifest_path = materialize_private_session_bundle(
        private_root, request=b"request", reference_response=b"response",
        descriptor=barrier.descriptor)
    (private_root / "request.bin").unlink()
    os.symlink(private_root / "reference_response.bin", private_root / "request.bin")
    with pytest.raises(ValueError, match="cannot be a symlink"):
        load_private_session_bundle(private_root, barrier=barrier)


def test_public_closure_rejects_private_streams_and_references(tmp_path):
    closure = materialize_synthetic_public_closure(tmp_path / "public" / "closure")
    (closure / "session_inputs.npz").write_bytes(b"synthetic private stream")
    with pytest.raises(ValueError, match="resource roles do not exactly cover"):
        snapshot_public_build_bundle(
            closure, tmp_path / "public" / "public_build_bundle.json")


def test_refreshed_receipt_cannot_cover_substituted_runner(tmp_path):
    public, authority, receipt, _barrier = _setup(tmp_path)
    raw = json.loads(receipt.read_text(encoding="ascii"))
    runner = receipt.parent / raw["outputs"]["runner"]["path"]
    runner.write_bytes(Path("/bin/true").read_bytes())
    runner.chmod(0o755)
    raw["outputs"]["runner"]["sha256"] = hashlib.sha256(runner.read_bytes()).hexdigest()
    raw["outputs"]["runner"]["size"] = runner.stat().st_size
    receipt.write_text(
        json.dumps(raw, sort_keys=True, separators=(",", ":")) + "\n", encoding="ascii")
    with pytest.raises(ValueError, match="independent clean replay"):
        verify_build_barrier(public.manifest_path, authority.path, receipt)


def test_undeclared_build_output_revokes_barrier(tmp_path):
    public, authority, receipt, _barrier = _setup(tmp_path)
    (receipt.parent / "undeclared-cache.o").write_bytes(b"cache")
    with pytest.raises(ValueError, match="output graph has omitted or extra paths"):
        verify_build_barrier(public.manifest_path, authority.path, receipt)


def test_arbitrary_private_observation_bytes_are_not_admitted(tmp_path):
    _public, _authority, _receipt, barrier = _setup(tmp_path)
    private_root = tmp_path / "private"
    materialize_private_session_bundle(
        private_root, request=b"observations.bin", reference_response=b"not-a-response",
        descriptor=barrier.descriptor)
    with pytest.raises(ValueError, match="no MRLNSES2 magic"):
        load_private_session_bundle(private_root, barrier=barrier)


def test_authority_rejects_omitted_library_and_detects_sysroot_drift(tmp_path):
    public, authority, _receipt, _barrier = _setup(tmp_path)
    raw = json.loads(authority.path.read_text(encoding="ascii"))
    raw["static_libraries"] = []
    authority.path.write_text(json.dumps(raw), encoding="ascii")
    omitted = load_multi_toolchain_authority(authority.path)
    with pytest.raises(ValueError, match="exact bound public_anchor"):
        build_and_relink_synthetic(
            public.manifest_path, omitted.path, tmp_path / "rebuild-without-library")

    # Restore a valid authority, then change one file in its tree resource.
    authority = write_multi_toolchain_authority(
        authority.path, tools={tool.role: tool.path for tool in authority.tools},
        sysroot=authority.sysroot,
        static_libraries={"public_anchor": public.closure_root / "lib" / "libpublic_anchor.a"},
        tree_resources=_compiler_resources(), file_resources={}, target_abi=_host_target())
    (authority.sysroot / "include" / "identity.h").write_text("drift\n", encoding="utf-8")
    with pytest.raises(ValueError, match="sysroot tree differs"):
        load_multi_toolchain_authority(authority.path)


@pytest.mark.parametrize("relative", ["sources/support.cc", "sources/model_session.c"])
def test_self_labeled_or_private_authored_source_cannot_reach_barrier(tmp_path, relative):
    closure = materialize_synthetic_public_closure(tmp_path / "public" / "closure")
    path = closure / relative
    path.write_text(path.read_text(encoding="utf-8") + "/* private value: 8675309 */\n",
                    encoding="utf-8")
    public_manifest = tmp_path / "public" / "public_build_bundle.json"
    snapshot_public_build_bundle(closure, public_manifest)
    sysroot = tmp_path / "toolchain/sysroot"
    sysroot.mkdir(parents=True)
    authority = write_multi_toolchain_authority(
        tmp_path / "toolchain/authority.json", tools=_tools(), sysroot=sysroot,
        static_libraries={"public_anchor": closure / "lib/libpublic_anchor.a"},
        tree_resources=_compiler_resources(), file_resources={}, target_abi=_host_target())
    with pytest.raises(ValueError, match="differs from deterministic producer output"):
        build_and_relink_synthetic(public_manifest, authority.path, tmp_path / "rebuilt")


def test_self_labeled_observation_resource_cannot_reach_barrier(tmp_path):
    closure = materialize_synthetic_public_closure(tmp_path / "public" / "closure")
    observation = closure / "observations.bin"
    observation.write_text("private samples", encoding="utf-8")
    roles = json.loads((closure / "resource_roles.json").read_text(encoding="ascii"))
    roles["resources"].append({"path": "observations.bin", "role": "c_source"})
    roles["resources"] = sorted(roles["resources"], key=lambda row: row["path"])
    (closure / "resource_roles.json").write_text(
        json.dumps(roles, sort_keys=True, separators=(",", ":")) + "\n", encoding="ascii")
    public_manifest = tmp_path / "public/public_build_bundle.json"
    snapshot_public_build_bundle(closure, public_manifest)
    sysroot = tmp_path / "toolchain/sysroot"
    sysroot.mkdir(parents=True)
    authority = write_multi_toolchain_authority(
        tmp_path / "toolchain/authority.json", tools=_tools(), sysroot=sysroot,
        static_libraries={"public_anchor": closure / "lib/libpublic_anchor.a"},
        tree_resources=_compiler_resources(), file_resources={}, target_abi=_host_target())
    with pytest.raises(ValueError, match="exact path-role graph"):
        build_and_relink_synthetic(public_manifest, authority.path, tmp_path / "rebuilt")


def test_identity_query_is_sealed_and_tree_resource_drift_is_detected(
        tmp_path, monkeypatch):
    closure = materialize_synthetic_public_closure(tmp_path / "public/closure")
    config = tmp_path / "toolchain/config"
    config.mkdir(parents=True)
    (config / "driver.cfg").write_text("pinned\n", encoding="ascii")
    driver_file = tmp_path / "toolchain/driver.conf"
    driver_file.write_text("pinned-file\n", encoding="ascii")
    sysroot = tmp_path / "toolchain/sysroot"
    sysroot.mkdir(parents=True)
    wrapper = tmp_path / "toolchain/version-tool"
    wrapper.write_text(
        "#!/bin/sh\n"
        "[ -z \"${MERLIN_IDENTITY_POISON+x}\" ] || exit 71\n"
        "[ \"$(pwd)\" = / ] || exit 72\n"
        "if read line; then exit 73; fi\n"
        "echo sealed-version-1\n",
        encoding="ascii")
    wrapper.chmod(0o755)
    tools = _tools()
    tools["cmake"] = str(wrapper)
    monkeypatch.setenv("MERLIN_IDENTITY_POISON", "ambient")
    monkeypatch.chdir(tmp_path)
    authority = write_multi_toolchain_authority(
        tmp_path / "toolchain/authority.json", tools=tools, sysroot=sysroot,
        static_libraries={"public_anchor": closure / "lib/libpublic_anchor.a"},
        tree_resources={**_compiler_resources(), "driver_config": config},
        file_resources={"driver_config_file": driver_file}, target_abi=_host_target())
    assert load_multi_toolchain_authority(authority.path).tool("cmake").version_line == (
        "sealed-version-1")
    (config / "driver.cfg").write_text("drift\n", encoding="ascii")
    with pytest.raises(ValueError, match="tree-resource identity differs.*driver_config"):
        load_multi_toolchain_authority(authority.path)
    (config / "driver.cfg").write_text("pinned\n", encoding="ascii")
    driver_file.write_text("file-drift\n", encoding="ascii")
    with pytest.raises(ValueError, match="file-resource identity differs.*driver_config_file"):
        load_multi_toolchain_authority(authority.path)
