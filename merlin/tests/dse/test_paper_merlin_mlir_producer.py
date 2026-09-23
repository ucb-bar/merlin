"""Genuine Merlin-lowering tracer for the v4 production object recipe."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from merlin.compare.paper_build_bundle import (
    _elf_identity,
    load_private_session_bundle,
    snapshot_public_build_bundle,
    write_multi_toolchain_authority,
)
from merlin.compare.paper_measurement_freeze import (
    _session_trajectory,
    write_capture_measurement_source_receipt,
)
from merlin.compare.paper_merlin_mlir_producer import (
    build_merlin_mlir_model_object,
    snapshot_synthetic_merlin_mlir_bundle,
    verify_merlin_mlir_build_barrier,
)
from merlin.compare.paper_model_object_builder import (
    MERLIN_OBJECT_BUILD_ARGV,
    MERLIN_RECIPE,
    merlin_session_resources,
    regenerate_model_object,
    stage_compiler_input,
    write_merlin_compiler_input,
)
from merlin.compare.paper_session_abi import (
    InputEndpoint,
    InputFrame,
    decode_request,
    decode_response,
    encode_request,
)
from merlin.llvmlower.toolchain import available, clang, m2m_python


def _tools() -> dict[str, str | Path]:
    names = {
        "cxx_compiler": "cc",
        "linker": "ld.lld",
        "host_linker": "ld",
        "cmake": "cmake",
        "ninja": "ninja",
    }
    found = {role: shutil.which(name) for role, name in names.items()}
    missing = [role for role, path in found.items() if path is None]
    if missing or not available():
        pytest.skip(f"genuine Merlin/toolchain prerequisites unavailable: {missing}")
    return {
        "c_compiler": clang(),
        "mlir_lowering_python": m2m_python(),
        **{role: str(path) for role, path in found.items()},
    }


def _sysroot(root: Path) -> Path:
    include = root / "include"
    include.mkdir(parents=True)
    (include / "stddef.h").write_text("typedef __SIZE_TYPE__ size_t;\n", encoding="ascii")
    (include / "stdint.h").write_text(
        "typedef __INT64_TYPE__ int64_t; typedef __UINT64_TYPE__ uint64_t;\n", encoding="ascii"
    )
    (include / "stdlib.h").write_text(
        "#include <stddef.h>\nvoid *malloc(size_t); void free(void *);\n", encoding="ascii"
    )
    (include / "string.h").write_text(
        "#include <stddef.h>\nvoid *memcpy(void *,const void *,size_t);\n", encoding="ascii"
    )
    return root


def _k1_target() -> dict[str, object]:
    return {
        "name": "k1_rvv_lp64d",
        "target_triple": "riscv64-unknown-elf",
        "march": "rv64gcv",
        "mabi": "lp64d",
        "features": ["c", "g", "v"],
        "elf_class": 64,
        "elf_machine": 243,
        "elf_osabi": 0,
        "elf_flags_mask": 0xFFFFFFFF,
        "elf_flags_value": 0x5,
    }


def _resource_closure() -> tuple[dict[str, Path], dict[str, Path]]:
    compiler = Path(clang()).resolve()
    compiler_roots = sorted((compiler.parent.parent / "lib/clang").iterdir())
    query = subprocess.run(
        [
            str(m2m_python()),
            "-I",
            "-c",
            "import json,pathlib,sys,sysconfig;"
            "P=pathlib.Path;site=P(sysconfig.get_path('purelib'));"
            "trees={'lowering_stdlib':str(P(sysconfig.get_path('stdlib')).resolve()),"
            "'lowering_numpy':str((site/'numpy').resolve()),"
            "'lowering_torch_mlir':str((site/'torch_mlir').resolve()),"
            "'lowering_yaml':str((site/'yaml').resolve()),"
            "'lowering_xdsl':str((site/'xdsl').resolve()),"
            "'lowering_immutabledict':str((site/'immutabledict').resolve()),"
            "'lowering_distutils_hack':str((site/'_distutils_hack').resolve())};"
            "base=[site/'typing_extensions.py',site/'_virtualenv.py',"
            "site/'_cuda_bindings_redirector.py',P(sys.prefix)/'pyvenv.cfg'];"
            "more=sorted(site.glob('*.pth'))+sorted(site.glob('__editable__*.py'));"
            "files={('lowering_config_%03d'%i):str(p.resolve()) for i,p in enumerate(base+more)};"
            "print(json.dumps({'trees':trees,'files':files},sort_keys=True))",
        ],
        capture_output=True,
        check=True,
        cwd="/",
        stdin=subprocess.DEVNULL,
        env={
            "LANG": "C",
            "LC_ALL": "C",
            "TZ": "UTC",
            "PATH": "",
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
    )
    raw = json.loads(query.stdout)
    trees = {name: Path(path) for name, path in raw["trees"].items()}
    trees["compiler_resource_dir"] = compiler_roots[-1]
    files = {name: Path(path) for name, path in raw["files"].items()}
    return trees, files


def _setup(tmp_path: Path, *, constant: int = 2, tools: dict[str, str | Path] | None = None):
    public = snapshot_synthetic_merlin_mlir_bundle(
        tmp_path / "public/closure", tmp_path / "public/manifest.json", prefill_constant=constant
    )
    trees, files = _resource_closure()
    authority = write_multi_toolchain_authority(
        tmp_path / "toolchain/authority.json",
        tools=tools or _tools(),
        sysroot=_sysroot(tmp_path / "toolchain/sysroot"),
        static_libraries={"public_anchor": public.closure_root / "lib/libpublic_anchor.a"},
        tree_resources=trees,
        file_resources=files,
        target_abi=_k1_target(),
    )
    build = build_merlin_mlir_model_object(public.manifest_path, authority.path, tmp_path / "rebuilt")
    return public, authority, build


def _request(descriptor) -> bytes:
    values = (2, 1, 4, 5)
    return encode_request(
        descriptor,
        [
            InputFrame(InputEndpoint(program, input_index), step, struct.pack("=q", value))
            for (program, input_index, step), value in zip(descriptor.required_input_keys, values, strict=True)
        ],
    )


def _run(build) -> tuple[int, ...]:
    import subprocess

    completed = subprocess.run(
        [str(build.runner), str(build.runner.parent)], input=_request(build.descriptor), capture_output=True, timeout=30
    )
    assert completed.returncode == 0
    response = decode_response(completed.stdout, expected_descriptor=build.descriptor)
    return tuple(struct.unpack("=q", frame.payload)[0] for frame in response.outputs)


def _capture(root: Path) -> Path:
    for name in ("prefill", "decode"):
        (root / "stages" / name).mkdir(parents=True)
    np = pytest.importorskip("numpy")
    np.savez(root / "stages/prefill/session_inputs.npz", seed=np.asarray([[2]], dtype=np.int64))
    np.savez(root / "stages/decode/session_inputs.npz", delta=np.asarray([[1], [4], [5]], dtype=np.int64))
    reference = np.asarray([[13], [43], [134]], dtype=np.int64)
    np.savez(root / "stages/decode/quality.npz", output=reference)
    children = {
        "prefill": {
            "version": 1,
            "kind": "synthetic_merlin_recurrent",
            "paper_ready": True,
            "stages": ["prefill"],
            "steps": 1,
            "stage_schedule": [{"name": "prefill", "steps": 1, "execution": "compiled", "timed": True}],
            "inputs": "session_inputs.npz",
            "states": [],
            "streams": [{"name": "value", "input_arg": 0, "key": "seed"}],
            "quality": {"scope": "trajectory", "output_index": 0},
        },
        "decode": {
            "version": 1,
            "kind": "synthetic_merlin_recurrent",
            "paper_ready": True,
            "stages": ["decode"],
            "steps": 3,
            "stage_schedule": [{"name": "decode", "steps": 3, "execution": "compiled_recurrent", "timed": True}],
            "inputs": "session_inputs.npz",
            "states": [{"name": "state", "input_arg": 1, "output_index": 0}],
            "streams": [{"name": "value", "input_arg": 0, "key": "delta"}],
            "quality": {
                "scope": "trajectory",
                "output_index": 0,
                "golden": "quality.npz",
                "key": "output",
                "reference": "eager_fp32",
                "metric": "top1_agreement",
                "reference_sha256": hashlib.sha256(reference.tobytes()).hexdigest(),
            },
        },
    }
    for name, value in children.items():
        (root / "stages" / name / "session_contract.yaml").write_text(yaml.safe_dump(value), encoding="utf-8")
    (root / "session_contract.yaml").write_text(
        yaml.safe_dump(
            {
                "version": 2,
                "kind": "synthetic_merlin_recurrent",
                "paper_ready": True,
                "stages": ["prefill", "decode"],
                "stage_schedule": [
                    {"name": "prefill", "steps": 1, "execution": "compiled", "timed": True},
                    {"name": "decode", "steps": 3, "execution": "compiled_recurrent", "timed": True},
                ],
                "programs": [
                    {"name": "prefill", "bundle": "stages/prefill", "steps": 1},
                    {"name": "decode", "bundle": "stages/decode", "steps": 3},
                ],
                "bindings": [
                    {
                        "name": "state_seed",
                        "from": {"program": "prefill", "output_index": 0},
                        "to": {"program": "decode", "input_arg": 1},
                    }
                ],
                "states": ["state"],
                "streams": [],
                "quality": {"scope": "trajectory", "program": "decode"},
            }
        ),
        encoding="utf-8",
    )
    return root


def test_registry_merlin_recipe_replays_and_deep_retains_byte_identical_object(tmp_path):
    public, authority, build = _setup(tmp_path / "portable")
    source_identity, capture_identity, artifact_identity = "a" * 64, "b" * 64, "c" * 64
    compiler_input = write_merlin_compiler_input(
        tmp_path / "portable/compiler-input.json",
        public_manifest=public.manifest_path,
        producer_authority=authority.path,
        producer_receipt=build.receipt,
        source_identity_sha256=source_identity,
        capture_sha256=capture_identity,
        runtime_artifact_sha256=artifact_identity,
    )
    regenerated = tmp_path / "regenerated.o"
    receipt = regenerate_model_object(
        recipe=MERLIN_RECIPE,
        registry_id="merlin_compile_v1",
        target="k1",
        compiler_input=compiler_input,
        tool=Path("/bin/true"),
        output=regenerated,
        source_identity_sha256=source_identity,
        capture_sha256=capture_identity,
        runtime_artifact_sha256=artifact_identity,
    )
    producer_receipt = json.loads(build.receipt.read_text(encoding="ascii"))
    runner_output = producer_receipt["outputs"]["session_runner_source"]
    expected_capacity = (
        13
        + len(build.descriptor.canonical_bytes)
        + 4
        + 8 * len(build.descriptor.calls)
        + 4
        + build.descriptor.output.frames * (20 + 8)
    )
    assert f"response_capacity = {expected_capacity}ULL" in (build.receipt.parent / runner_output["path"]).read_text(
        encoding="utf-8"
    )
    assert merlin_session_resources(compiler_input).runner_source == (build.receipt.parent / runner_output["path"])
    assert regenerated.read_bytes() == build.composite_object.read_bytes()
    assert receipt == {
        "recipe": MERLIN_RECIPE,
        "compiler_input_sha256": hashlib.sha256(compiler_input.read_bytes()).hexdigest(),
        "generated_source_sha256": producer_receipt["outputs"]["session_adapter_source"]["sha256"],
        "object_build_argv": MERLIN_OBJECT_BUILD_ARGV,
        "model_object_sha256": hashlib.sha256(regenerated.read_bytes()).hexdigest(),
    }

    retained_input = stage_compiler_input(compiler_input, tmp_path / "retained/compiler-input", recipe=MERLIN_RECIPE)
    retained_runner = merlin_session_resources(retained_input).runner_source
    assert retained_runner.is_relative_to(retained_input.parent)
    assert retained_runner.read_bytes() == (build.receipt.parent / runner_output["path"]).read_bytes()
    replayed = tmp_path / "retained/replayed.o"
    replay_receipt = regenerate_model_object(
        recipe=MERLIN_RECIPE,
        registry_id="merlin_compile_v1",
        target="k1",
        compiler_input=retained_input,
        tool=Path("/bin/false"),
        output=replayed,
        source_identity_sha256=source_identity,
        capture_sha256=capture_identity,
        runtime_artifact_sha256=artifact_identity,
    )
    assert replayed.read_bytes() == regenerated.read_bytes()
    assert replay_receipt == receipt

    from merlin.compare import paper_model_object_builder
    from merlin.compare.paper_contract_registry import _package_receipt

    builder = Path(paper_model_object_builder.__file__)
    resource_hashes = {
        "compiler_input": receipt["compiler_input_sha256"],
        "model_object": receipt["model_object_sha256"],
        "object_builder": hashlib.sha256(builder.read_bytes()).hexdigest(),
        "runner": producer_receipt["outputs"]["session_runner_source"]["sha256"],
        "build_tool": hashlib.sha256(Path("/bin/true").read_bytes()).hexdigest(),
        "session_descriptor": hashlib.sha256(
            (public.closure_root / "descriptor/session_descriptor.json").read_bytes()
        ).hexdigest(),
    }
    package = tmp_path / "portable/package.json"
    package.write_text(
        json.dumps(
            {
                "schema_version": 3,
                "kind": "paper_backend_package_receipt_v3",
                "status": "finalized",
                "registry_id": "merlin_compile_v1",
                "build_adapter": "merlin_session_abi_c_v1",
                "cell": {"model": "synthetic", "backend": "merlin", "precision": "fp32"},
                "package_identity_sha256": "d" * 64,
                "compiler_or_framework_source_sha256": source_identity,
                "capture_sha256": capture_identity,
                "runtime_artifact_sha256": artifact_identity,
                "runner_source_sha256": resource_hashes["runner"],
                "model_object_sha256": resource_hashes["model_object"],
                "compiler_input_sha256": resource_hashes["compiler_input"],
                "object_builder_source_sha256": resource_hashes["object_builder"],
                "object_recipe": MERLIN_RECIPE,
                "object_build_argv": MERLIN_OBJECT_BUILD_ARGV,
                "generated_model_source_sha256": receipt["generated_source_sha256"],
                "build_tool_sha256": resource_hashes["build_tool"],
                "build_source_identity_sha256": hashlib.sha256(
                    json.dumps(
                        {
                            "compiler_input": resource_hashes["compiler_input"],
                            "model_object": resource_hashes["model_object"],
                            "object_builder": resource_hashes["object_builder"],
                            "runner": resource_hashes["runner"],
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest(),
                "build_argv": [
                    "{tool}",
                    "-O2",
                    "-std=c11",
                    "{source:runner}",
                    "{source:model_object}",
                    "-o",
                    "{output}",
                ],
                "result_executable_sha256": "e" * 64,
                "finalized_at": "2026-08-31T00:00:00Z",
                "session_protocol": "MRLNSES2",
                "session_descriptor_sha256": resource_hashes["session_descriptor"],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    cell = SimpleNamespace(
        key="synthetic/merlin/fp32",
        precision="fp32",
        model=SimpleNamespace(name="synthetic", artifacts={"fp32": {"sha256": capture_identity}}),
        backend=SimpleNamespace(name="merlin"),
    )
    assert (
        _package_receipt(
            package,
            registry_id="merlin_compile_v1",
            cell=cell,
            source_identity=source_identity,
            package_identity="d" * 64,
            runtime_artifact_sha256=artifact_identity,
            resource_hashes=resource_hashes,
            target="k1",
            regeneration=receipt,
        )["session_protocol"]
        == "MRLNSES2"
    )

    original_runner = build.receipt.parent / runner_output["path"]
    alternate_runner = build.receipt.parent / "alternate-session-runner.c"
    alternate_runner.write_bytes(original_runner.read_bytes())
    original_runner.unlink()
    original_runner.symlink_to(alternate_runner.name)
    with pytest.raises(ValueError, match="omits or changes"):
        merlin_session_resources(compiler_input)


def test_registry_merlin_compiler_input_is_closed_and_canonical(tmp_path):
    compiler_input = tmp_path / "compiler-input.json"
    compiler_input.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "paper_merlin_mlir_compiler_input_v1",
                "compiler_or_framework_source_sha256": "a" * 64,
                "capture_sha256": "b" * 64,
                "runtime_artifact_sha256": "c" * 64,
                "public_manifest": {"path": "public.json", "sha256": "d" * 64},
                "producer_authority": {"path": "authority.json", "sha256": "e" * 64},
                "producer_receipt": {"path": "receipt.json", "sha256": "f" * 64},
                "private_inputs": "must never be accepted",
            }
        ),
        encoding="ascii",
    )
    with pytest.raises(ValueError, match="closed"):
        regenerate_model_object(
            recipe=MERLIN_RECIPE,
            registry_id="merlin_compile_v1",
            target="k1",
            compiler_input=compiler_input,
            tool=Path("/bin/true"),
            output=tmp_path / "model.o",
            source_identity_sha256="a" * 64,
            capture_sha256="b" * 64,
            runtime_artifact_sha256="c" * 64,
        )


def test_mrlnses2_capture_runner_and_controller_stdout_path_are_descriptor_bound(tmp_path, monkeypatch):
    from merlin.compare import paper_measurement_controller as controller

    _public, _authority, build = _setup(tmp_path / "build")
    capture = _capture(tmp_path / "capture")
    request, reference, _sources = _session_trajectory(capture, build.descriptor)
    source_receipt = write_capture_measurement_source_receipt(
        capture, model="synthetic", precision="fp32", observations=3
    )
    source_document = json.loads(source_receipt.read_text(encoding="utf-8"))
    assert source_document["schema_version"] == 2
    assert source_document["session_descriptor_sha256"] == build.descriptor.sha256
    assert source_document["session_request_sha256"] == hashlib.sha256(request).hexdigest()
    request_path = tmp_path / "request.bin"
    request_path.write_bytes(request)
    decoded_request = decode_request(request, expected_descriptor=build.descriptor)
    manifest = tmp_path / "session-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "kind": "paper_session_request_v2",
                "protocol": "MRLNSES2",
                "session_kind": "synthetic_merlin_recurrent",
                "observations": 3,
                "descriptor_sha256": build.descriptor.sha256,
                "inputs": {"session_request": hashlib.sha256(request).hexdigest()},
                "records": [
                    {
                        "program": frame.endpoint.program,
                        "input": frame.endpoint.input,
                        "step": frame.step,
                        "payload_sha256": hashlib.sha256(frame.payload).hexdigest(),
                    }
                    for frame in decoded_request.frames
                ],
            }
        ),
        encoding="utf-8",
    )
    assert (
        controller._session_manifest(
            tmp_path,
            {"path": manifest.name, "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()},
            {"kind": "synthetic_merlin_recurrent", "observations": 3},
            {"session_request": request_path},
            descriptor=build.descriptor,
        )["protocol"]
        == "MRLNSES2"
    )
    core = min(os.sched_getaffinity(0))
    monkeypatch.setattr(
        controller,
        "_proc_state",
        lambda _pid, requested, _previous: (
            4096,
            set(requested),
            1,
            set(requested),
            {},
            {value: 1 for value in requested},
        ),
    )
    observation = tmp_path / "must-not-be-written.bin"
    row, observed, stdout, stderr = controller._run_iteration(
        [str(build.runner), str(build.receipt.parent)],
        cwd=tmp_path,
        environment={"LANG": "C", "LC_ALL": "C", "TZ": "UTC", "PATH": ""},
        core_ids=[core],
        observation=observation,
        timeout=10,
        phase="measured",
        index=0,
        request=request,
    )
    correctness, quality = controller._oracle(
        observed,
        reference,
        {
            "session_abi": {"protocol": "MRLNSES2"},
            "oracle": {
                "kind": "int64_top1",
                "metric": "top1_agreement",
                "threshold": 1.0,
                "scope": "trajectory",
                "steps": 3,
            },
        },
        descriptor=build.descriptor,
    )
    assert observed == stdout and stderr == b"" and not observation.exists()
    assert decode_response(observed, expected_descriptor=build.descriptor).executed_calls
    assert row["observation_sha256"] == hashlib.sha256(observed).hexdigest()
    assert correctness["gate_ok"] and quality["value"] == 1.0


def test_genuine_mlir_prefill_recurrent_build_exports_k1_session_object(tmp_path):
    public, authority, build = _setup(tmp_path)
    barrier = verify_merlin_mlir_build_barrier(public.manifest_path, authority.path, build.receipt)
    elf = _elf_identity(barrier.composite_object)
    assert (elf["class"], elf["type"], elf["machine"]) == (64, 1, 243)
    assert (elf["osabi"], elf["flags"]) == (0, 0x5)
    assert elf["global_definitions"].count("merlin_paper_session_v1") == 1
    assert _run(build) == (13, 43, 134)
    generated = (build.receipt.parent / "session_adapter.c").read_text(encoding="utf-8")
    assert "merlin_run_multi_with" in generated
    assert "merlin_commit_state" in generated
    assert "model_name" not in generated
    receipt = json.loads(build.receipt.read_text(encoding="ascii"))
    assert receipt["target_abi"] == _k1_target()
    lower = receipt["recipe"]["lower_program_0"]
    assert lower[-4:] == ["riscv64-unknown-elf", "rv64gcv", "lp64d", "c,g,v"]
    compile_runtime = receipt["recipe"]["compile_runtime_riscv"]
    assert "--target=riscv64-unknown-elf" in compile_runtime
    assert "-march=rv64gcv" in compile_runtime and "-mabi=lp64d" in compile_runtime
    assert any(value.startswith("-resource-dir=") for value in compile_runtime)
    assert any(value.startswith("--sysroot=") for value in compile_runtime)
    assert {resource.name for resource in authority.tree_resources} == {
        "compiler_resource_dir",
        "lowering_stdlib",
        "lowering_numpy",
        "lowering_torch_mlir",
        "lowering_yaml",
        "lowering_xdsl",
        "lowering_immutabledict",
        "lowering_distutils_hack",
    }
    assert authority.file_resources

    # Even with refreshed output hashes, wrong RISC-V ABI flags fail before replay.
    raw = bytearray(build.composite_object.read_bytes())
    raw[48:52] = (0).to_bytes(4, "little")
    build.composite_object.write_bytes(raw)
    row = receipt["outputs"]["composite_object"]
    row["sha256"] = hashlib.sha256(raw).hexdigest()
    row["size"] = len(raw)
    build.receipt.write_text(json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n", encoding="ascii")
    with pytest.raises(ValueError, match="not ELF64 EM_RISCV"):
        verify_merlin_mlir_build_barrier(public.manifest_path, authority.path, build.receipt)


def test_public_mlir_change_changes_real_object_and_session_dataflow(tmp_path):
    _p0, _a0, first = _setup(tmp_path / "first", constant=2)
    _p1, _a1, second = _setup(tmp_path / "second", constant=5)
    assert first.composite_object.read_bytes() != second.composite_object.read_bytes()
    assert _run(first) == (13, 43, 134)
    assert _run(second) == (22, 70, 215)


def test_refreshed_cached_object_substitution_fails_clean_replay(tmp_path):
    public, authority, build = _setup(tmp_path)
    receipt = json.loads(build.receipt.read_text(encoding="ascii"))
    row = receipt["outputs"]["program_0_riscv_object"]
    target = build.receipt.parent / row["path"]
    replacement = build.receipt.parent / receipt["outputs"]["program_1_riscv_object"]["path"]
    target.write_bytes(replacement.read_bytes())
    row["sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
    row["size"] = target.stat().st_size
    build.receipt.write_text(json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n", encoding="ascii")
    with pytest.raises(ValueError, match="independent clean replay"):
        verify_merlin_mlir_build_barrier(public.manifest_path, authority.path, build.receipt)


@pytest.mark.parametrize("kind", ["public_program", "generated_object", "extra_output"])
def test_omitted_program_or_generated_object_fails_closed(tmp_path, kind):
    public, authority, build = _setup(tmp_path)
    if kind == "public_program":
        (public.closure_root / "programs/1/model.mlir").unlink()
        pattern = "public build closure tree differs"
    else:
        if kind == "generated_object":
            receipt = json.loads(build.receipt.read_text(encoding="ascii"))
            (build.receipt.parent / receipt["outputs"]["program_1_riscv_object"]["path"]).unlink()
            pattern = "output identity differs"
        else:
            (build.receipt.parent / "undeclared-cache.o").write_bytes(b"cache")
            pattern = "output graph has omitted or extra paths"
    with pytest.raises(ValueError, match=pattern):
        verify_merlin_mlir_build_barrier(public.manifest_path, authority.path, build.receipt)


def test_tool_drift_and_private_resource_attempt_fail_before_private_io(tmp_path):
    tools = _tools()
    wrapper = tmp_path / "tool/cc-wrapper"
    wrapper.parent.mkdir(parents=True)
    real = str(tools["cxx_compiler"])
    wrapper.write_text(f'#!/bin/sh\nexec "{real}" "$@"\n', encoding="ascii")
    wrapper.chmod(0o755)
    tools["cxx_compiler"] = wrapper
    public, authority, build = _setup(tmp_path / "build", tools=tools)
    with pytest.raises(PermissionError, match="barrier is required"):
        load_private_session_bundle(tmp_path / "private", barrier=None)

    closure = snapshot_synthetic_merlin_mlir_bundle(
        tmp_path / "private-attempt/closure", tmp_path / "private-attempt/manifest.json"
    ).closure_root
    (closure / "observations.bin").write_text("private frames", encoding="ascii")
    roles = json.loads((closure / "resource_roles.json").read_text(encoding="ascii"))
    roles["resources"].append({"path": "observations.bin", "role": "c_source"})
    roles["resources"].sort(key=lambda row: row["path"])
    (closure / "resource_roles.json").write_text(
        json.dumps(roles, sort_keys=True, separators=(",", ":")) + "\n", encoding="ascii"
    )
    rebound = snapshot_public_build_bundle(closure, tmp_path / "private-attempt/rebound_manifest.json")
    trees, files = _resource_closure()
    private_authority = write_multi_toolchain_authority(
        tmp_path / "private-attempt/authority.json",
        tools=tools,
        sysroot=_sysroot(tmp_path / "private-attempt/sysroot"),
        static_libraries={"public_anchor": closure / "lib/libpublic_anchor.a"},
        tree_resources=trees,
        file_resources=files,
        target_abi=_k1_target(),
    )
    with pytest.raises(ValueError, match="exact path-role graph"):
        build_merlin_mlir_model_object(rebound.manifest_path, private_authority.path, tmp_path / "private-attempt/out")

    wrapper.write_text(f'#!/bin/sh\n# drift\nexec "{real}" "$@"\n', encoding="ascii")
    wrapper.chmod(0o755)
    with pytest.raises(ValueError, match="tool identity differs.*cxx_compiler"):
        verify_merlin_mlir_build_barrier(public.manifest_path, authority.path, build.receipt)
