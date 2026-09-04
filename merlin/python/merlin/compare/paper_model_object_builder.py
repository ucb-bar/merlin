"""Registry-owned regeneration of paper model objects.

Backend packages may cache a model object, but that object is never accepted as build authority.
The controller regenerates it from a frozen compiler/framework input using one closed recipe and
requires byte identity before linking.  The Merlin recipe is backed by the independently replayed
whole-session producer.  ExecuTorch uses a separately cross-built, receipt-sealed RISC-V session
executable.  The board controller verifies those exact bytes; it does not claim an on-board rebuild
with the x86-hosted cross compiler and does not accept supplied C or arbitrary argv.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

UNIT_TEST_RECIPE = "unit_test_affine_descriptor_v1"
MERLIN_RECIPE = "merlin_mlir_model_object_v1"
EXECUTORCH_RECIPE = "executorch_aot_model_object_v1"
OBJECT_BUILD_ARGV = [
    "{tool}",
    "-O2",
    "-std=c11",
    "-c",
    "{generated_source}",
    "-o",
    "{output}",
]
MERLIN_OBJECT_BUILD_ARGV = [
    "verify_merlin_mlir_build_barrier",
    "{public_manifest}",
    "{producer_authority}",
    "{producer_receipt}",
    "{output}",
]
EXECUTORCH_OBJECT_BUILD_ARGV = [
    "verify_executorch_sealed_session",
    "{compiler_input}",
    "{producer_receipt}",
    "{output}",
]
MERLIN_COMPILER_INPUT_KIND = "paper_merlin_mlir_compiler_input_v1"
_MERLIN_INPUT_FIELDS = {
    "schema_version",
    "kind",
    "compiler_or_framework_source_sha256",
    "capture_sha256",
    "runtime_artifact_sha256",
    "public_manifest",
    "producer_authority",
    "producer_receipt",
}
_PRODUCTION_RECIPES = {
    "merlin_compile_v1": MERLIN_RECIPE,
    "executorch_v1": EXECUTORCH_RECIPE,
}


@dataclass(frozen=True)
class MerlinSessionResources:
    descriptor_path: Path
    runner_source: Path
    descriptor: object


@dataclass(frozen=True)
class ExecuTorchSessionResources:
    """Closed host-produced run package selected by an ExecuTorch compiler input."""

    compiler_input: Path
    package_root: Path
    package_metadata: Path
    producer_receipt: Path
    runner: Path
    public_files: tuple[Path, ...]
    private_files: tuple[Path, ...]
    identities: Mapping[str, Any]
    session_manifest_sha256: str


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _is_sha(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def expected_recipe(registry_id: str, target: str) -> str:
    if target == "unit-test" and registry_id in _PRODUCTION_RECIPES:
        return UNIT_TEST_RECIPE
    try:
        return _PRODUCTION_RECIPES[registry_id]
    except KeyError as error:
        raise ValueError("model-object registry id is unsupported") from error


def _closed(value: object, fields: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"{label} is not a closed {sorted(fields)} mapping")
    return value


def _has_symlink_component(root: Path, relative: Path) -> bool:
    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            return True
    return False


def object_build_argv(recipe: str) -> list[str]:
    """Return the closed, receipt-facing operation for one registry recipe."""
    if recipe == MERLIN_RECIPE:
        return list(MERLIN_OBJECT_BUILD_ARGV)
    if recipe == EXECUTORCH_RECIPE:
        return list(EXECUTORCH_OBJECT_BUILD_ARGV)
    if recipe == UNIT_TEST_RECIPE:
        return list(OBJECT_BUILD_ARGV)
    raise ValueError("model-object recipe is unsupported")


def _safe_ref(root: Path, value: object, label: str) -> Path:
    ref = _closed(value, {"path", "sha256"}, label)
    relative = Path(str(ref["path"]))
    if (
        relative.is_absolute()
        or not relative.parts
        or ".." in relative.parts
        or relative.as_posix() != str(ref["path"])
    ):
        raise ValueError(f"{label} must be a normalized relative path")
    path = (root / relative).resolve()
    if (
        _has_symlink_component(root, relative)
        or not path.is_relative_to(root.resolve())
        or not path.is_file()
        or not _is_sha(ref["sha256"])
        or _sha(path) != ref["sha256"]
    ):
        raise ValueError(f"{label} identity differs or is unsafe")
    return path


def _load_canonical(path: Path, fields: set[str], label: str) -> Mapping[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} is absent or unsafe")
    raw = path.read_bytes()
    try:
        document = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is invalid JSON") from error
    document = _closed(document, fields, label)
    if raw != _canonical(document) + b"\n":
        raise ValueError(f"{label} is not canonical JSON")
    return document


def _receipt_rows(root: Path, value: object, label: str, *, verify_bytes: bool) -> tuple[Path, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty closed file list")
    paths: list[Path] = []
    previous = ""
    for index, raw in enumerate(value):
        row = _closed(raw, {"path", "sha256", "size"}, f"{label}[{index}]")
        relative = Path(str(row["path"]))
        path_text = relative.as_posix()
        if (relative.is_absolute() or not relative.parts or ".." in relative.parts
                or path_text != str(row["path"]) or path_text <= previous
                or not _is_sha(row["sha256"]) or type(row["size"]) is not int
                or row["size"] < 0):
            raise ValueError(f"{label}[{index}] is unsafe or unordered")
        previous = path_text
        path = (root / relative).absolute()
        if verify_bytes and (
                _has_symlink_component(root, relative) or path.is_symlink() or not path.is_file()
                or path.stat().st_size != row["size"] or _sha(path) != row["sha256"]):
            raise ValueError(f"{label}[{index}] identity differs")
        paths.append(path.resolve() if verify_bytes else path)
    return tuple(paths)


def executorch_session_resources(
    compiler_input: str | Path, *, include_private: bool = False,
) -> ExecuTorchSessionResources:
    """Verify a producer handoff without opening private session tensors by default."""
    from merlin.baselines import executorch_session as session

    compiler_input = Path(compiler_input).absolute()
    input_document = _load_canonical(compiler_input, {
        "schema_version", "kind", "compiler_or_framework_source_sha256", "capture_sha256",
        "capture_session_identity_sha256", "producer_receipt", "package_metadata",
    }, "ExecuTorch compiler input")
    if (input_document["schema_version"] != 1
            or input_document["kind"] != session.PAPER_COMPILER_INPUT_KIND
            or not all(_is_sha(input_document[name]) for name in (
                "compiler_or_framework_source_sha256", "capture_sha256",
                "capture_session_identity_sha256"))):
        raise ValueError("ExecuTorch compiler input schema or frozen identities differ")
    root = compiler_input.parent.resolve()
    producer = _safe_ref(root, input_document["producer_receipt"], "ExecuTorch producer receipt")
    metadata_path = _safe_ref(root, input_document["package_metadata"], "ExecuTorch package metadata")
    receipt = _load_canonical(producer, {
        "schema_version", "kind", "status", "producer_id", "producer_source_sha256", "target",
        "model", "variant", "xnnpack", "session_package_schema", "session_manifest_sha256",
        "runner", "runner_architecture", "public_files", "private_files", "identities",
        "identity_sha256",
    }, "ExecuTorch producer receipt")
    identities = _closed(receipt["identities"], {
        "capture_sha256", "capture_session_identity_sha256", "framework_source_sha256",
        "build_environment_sha256", "build_invocation_environment_sha256", "executorch_identity",
        "model2mlir_identity", "toolchain_identity", "external_model_source",
    }, "ExecuTorch producer identities")
    architecture = _closed(
        receipt["runner_architecture"], {"elf_class", "endianness", "machine", "machine_id"},
        "ExecuTorch runner architecture")
    if (receipt["schema_version"] != 1
            or receipt["kind"] != session.PAPER_PRODUCER_RECEIPT_KIND
            or receipt["status"] != "finalized" or receipt["producer_id"] != session.PAPER_PRODUCER_ID
            or receipt["producer_source_sha256"] != _sha(Path(session.__file__))
            or receipt["target"] != "k1-rv64gcv-linux" or receipt["xnnpack"] is not True
            or receipt["session_package_schema"] != session.PACKAGE_SCHEMA
            or architecture != {"elf_class": 64, "endianness": "little", "machine": "riscv",
                                    "machine_id": 243}
            or not _is_sha(receipt["session_manifest_sha256"])
            or not _is_sha(receipt["identity_sha256"])
            or receipt["identity_sha256"] != hashlib.sha256(_canonical(identities)).hexdigest()
            or identities["framework_source_sha256"]
            != input_document["compiler_or_framework_source_sha256"]
            or identities["capture_sha256"] != input_document["capture_sha256"]
            or identities["capture_session_identity_sha256"]
            != input_document["capture_session_identity_sha256"]):
        raise ValueError("ExecuTorch producer receipt identity differs")
    public = _receipt_rows(root, receipt["public_files"], "ExecuTorch public files", verify_bytes=True)
    private = _receipt_rows(
        root, receipt["private_files"], "ExecuTorch private files", verify_bytes=include_private)
    if set(public) & set(private) or metadata_path not in public:
        raise ValueError("ExecuTorch producer file classification is invalid")
    runner_row = _closed(receipt["runner"], {"path", "sha256", "size"}, "ExecuTorch runner")
    runner_matches = [
        path for path in public if path.relative_to(root).as_posix() == runner_row["path"]]
    if (len(runner_matches) != 1 or _sha(runner_matches[0]) != runner_row["sha256"]
            or runner_matches[0].stat().st_size != runner_row["size"]):
        raise ValueError("ExecuTorch producer runner differs from its public graph")
    header = runner_matches[0].read_bytes()[:20]
    if (len(header) < 20 or header[:4] != b"\x7fELF" or header[4] != 2 or header[5] != 1
            or int.from_bytes(header[18:20], "little") != 243):
        raise ValueError("ExecuTorch producer runner is not ELF64 little-endian RISC-V")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (metadata.get("schema") != session.PACKAGE_SCHEMA
            or metadata.get("model") != receipt["model"]
            or metadata.get("variant") != receipt["variant"]
            or metadata.get("xnnpack") is not True
            or metadata.get("capture_sha256") != identities["capture_sha256"]
            or metadata.get("capture_session_identity_sha256")
            != identities["capture_session_identity_sha256"]
            or metadata.get("framework_source_sha256") != identities["framework_source_sha256"]
            or metadata.get("build_environment_sha256") != identities["build_environment_sha256"]
            or metadata.get("build_invocation_environment_sha256")
            != identities["build_invocation_environment_sha256"]):
        raise ValueError("ExecuTorch package metadata differs from producer receipt")
    if include_private:
        package = session.load_session_package(root)
        if (package.runner != runner_matches[0]
                or package.capture_sha256 != identities["capture_sha256"]
                or package.capture_session_identity_sha256
                != identities["capture_session_identity_sha256"]):
            raise ValueError("ExecuTorch full package differs from its producer receipt")
    return ExecuTorchSessionResources(
        compiler_input.resolve(), root, metadata_path, producer, runner_matches[0], public, private,
        dict(identities), str(receipt["session_manifest_sha256"]))


def _load_merlin_input(path: Path) -> tuple[Mapping[str, Any], dict[str, Path]]:
    if not path.is_file() or path.is_symlink():
        raise ValueError("Merlin compiler input is absent or unsafe")
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Merlin compiler input is invalid JSON") from exc
    value = _closed(value, _MERLIN_INPUT_FIELDS, "Merlin compiler input")
    if raw != _canonical(value) + b"\n":
        raise ValueError("Merlin compiler input is not canonical JSON")
    identities = (
        value["compiler_or_framework_source_sha256"],
        value["capture_sha256"],
        value["runtime_artifact_sha256"],
    )
    if (
        value["schema_version"] != 1
        or value["kind"] != MERLIN_COMPILER_INPUT_KIND
        or not all(_is_sha(identity) for identity in identities)
    ):
        raise ValueError("Merlin compiler input schema or frozen identities differ")
    root = path.parent.resolve()
    refs = {
        name: _safe_ref(root, value[name], f"Merlin compiler input {name}")
        for name in ("public_manifest", "producer_authority", "producer_receipt")
    }
    return value, refs


def write_merlin_compiler_input(
    path: str | Path,
    *,
    public_manifest: str | Path,
    producer_authority: str | Path,
    producer_receipt: str | Path,
    source_identity_sha256: str,
    capture_sha256: str,
    runtime_artifact_sha256: str,
) -> Path:
    """Write the closed handoff consumed by ``merlin_mlir_model_object_v1``.

    Every referenced producer input/output must be below the descriptor directory.  This makes the
    complete compiler input portable and permits exact deep retention without copying unrelated
    files from a producer workspace.
    """
    path = Path(path).resolve()
    root = path.parent
    refs: dict[str, dict[str, str]] = {}
    for name, raw in {
        "public_manifest": public_manifest,
        "producer_authority": producer_authority,
        "producer_receipt": producer_receipt,
    }.items():
        candidate = Path(raw).absolute()
        if candidate.is_symlink() or not candidate.is_file():
            raise ValueError(f"Merlin {name} must be a regular file below the compiler input")
        candidate = candidate.resolve()
        if not candidate.is_relative_to(root):
            raise ValueError(f"Merlin {name} must be a regular file below the compiler input")
        refs[name] = {"path": candidate.relative_to(root).as_posix(), "sha256": _sha(candidate)}
    value = {
        "schema_version": 1,
        "kind": MERLIN_COMPILER_INPUT_KIND,
        "compiler_or_framework_source_sha256": source_identity_sha256,
        "capture_sha256": capture_sha256,
        "runtime_artifact_sha256": runtime_artifact_sha256,
        **refs,
    }
    if not all(
        _is_sha(value[name])
        for name in ("compiler_or_framework_source_sha256", "capture_sha256", "runtime_artifact_sha256")
    ):
        raise ValueError("Merlin compiler input identities must be lowercase SHA-256 values")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical(value) + b"\n")
    _load_merlin_input(path)
    return path


def _merlin_graph(path: Path) -> dict[Path, Path]:
    """Enumerate, without private I/O, the exact portable producer graph."""
    _value, refs = _load_merlin_input(path)
    from .paper_build_bundle import verify_public_build_bundle

    public = verify_public_build_bundle(refs["public_manifest"])
    receipt = json.loads(refs["producer_receipt"].read_text(encoding="ascii"))
    if not isinstance(receipt, Mapping) or not isinstance(receipt.get("outputs"), Mapping):
        raise ValueError("Merlin producer receipt has no closed output graph")
    root = path.parent.resolve()
    files = {path.resolve(): path.resolve()}

    def add(candidate: Path, label: str) -> None:
        candidate = candidate.resolve()
        if not candidate.is_relative_to(root) or candidate.is_symlink() or not candidate.is_file():
            raise ValueError(f"{label} escapes the portable Merlin compiler input")
        files[candidate] = candidate

    for name, candidate in refs.items():
        add(candidate, f"Merlin {name}")
    for row in public.files:
        add(public.closure_root / str(row["path"]), "Merlin public closure resource")
    for name, raw in receipt["outputs"].items():
        row = _closed(raw, {"path", "sha256", "size"}, f"Merlin producer output {name}")
        relative = Path(str(row["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Merlin producer output escapes its receipt")
        receipt_root = refs["producer_receipt"].parent
        candidate = (receipt_root / relative).resolve()
        if (
            _has_symlink_component(receipt_root, relative)
            or not candidate.is_relative_to(receipt_root)
            or not candidate.is_file()
            or _sha(candidate) != row["sha256"]
            or candidate.stat().st_size != row["size"]
        ):
            raise ValueError(f"Merlin producer output identity differs for {name}")
        add(candidate, f"Merlin producer output {name}")
    return {source: source.relative_to(root) for source in files}


def merlin_session_resources(compiler_input: str | Path) -> MerlinSessionResources:
    """Resolve the producer-bound public runner and descriptor from a closed Merlin input."""
    compiler_input = Path(compiler_input).absolute()
    if compiler_input.is_symlink():
        raise ValueError("Merlin compiler input is absent or unsafe")
    _value, refs = _load_merlin_input(compiler_input.resolve())
    from .paper_build_bundle import verify_public_build_bundle
    from .paper_session_abi import descriptor_from_dict

    public = verify_public_build_bundle(refs["public_manifest"])
    descriptor_path = public.closure_root / "descriptor/session_descriptor.json"
    try:
        receipt = json.loads(refs["producer_receipt"].read_text(encoding="ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Merlin producer receipt is invalid JSON") from exc
    if not isinstance(receipt, Mapping) or not isinstance(receipt.get("outputs"), Mapping):
        raise ValueError("Merlin producer receipt has no closed output graph")
    runner = _closed(
        receipt["outputs"].get("session_runner_source"),
        {"path", "sha256", "size"},
        "Merlin producer session runner output",
    )
    relative = Path(str(runner["path"]))
    if (
        relative.is_absolute()
        or not relative.parts
        or ".." in relative.parts
        or relative.as_posix() != str(runner["path"])
    ):
        raise ValueError("Merlin producer session runner path is unsafe")
    receipt_root = refs["producer_receipt"].parent
    runner_source = (receipt_root / relative).resolve()
    if (
        descriptor_path.is_symlink()
        or not descriptor_path.is_file()
        or _has_symlink_component(receipt_root, relative)
        or not runner_source.is_file()
        or not runner_source.is_relative_to(receipt_root)
        or not _is_sha(runner["sha256"])
        or _sha(runner_source) != runner["sha256"]
        or runner_source.stat().st_size != runner["size"]
    ):
        raise ValueError("Merlin producer omits or changes its session descriptor or runner")
    descriptor = descriptor_from_dict(json.loads(descriptor_path.read_text(encoding="ascii")))
    return MerlinSessionResources(descriptor_path, runner_source, descriptor)


def stage_compiler_input(
    source: str | Path, destination: str | Path, *, recipe: str, include_private: bool = True,
) -> Path:
    """Retain a compiler input and, for Merlin, its exact transitive producer graph."""
    source, destination = Path(source).absolute(), Path(destination).resolve()
    if source.is_symlink():
        raise ValueError("frozen compiler input is absent or unsafe")
    source = source.resolve()
    if recipe == EXECUTORCH_RECIPE:
        resources = executorch_session_resources(source, include_private=include_private)
        destination_root = destination.parent
        destination_root.mkdir(parents=True, exist_ok=True)
        for original in (
                resources.package_metadata, resources.producer_receipt,
                *resources.public_files, *(resources.private_files if include_private else ())):
            relative = original.relative_to(resources.package_root)
            target = destination_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copy2(original, target)
        shutil.copy2(source, destination)
        executorch_session_resources(destination, include_private=include_private)
        return destination
    if recipe != MERLIN_RECIPE:
        if source.is_symlink() or not source.is_file():
            raise ValueError("frozen compiler input is absent or unsafe")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        return destination
    graph = _merlin_graph(source)
    destination_root = destination.parent
    for original, relative in sorted(graph.items(), key=lambda item: item[1].as_posix()):
        target = destination_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, target)
    staged_source = destination_root / source.name
    if destination != staged_source:
        shutil.copy2(staged_source, destination)
        staged_source.unlink()
    _merlin_graph(destination)
    return destination


def _unit_test_source(
    compiler_input: Path, *, source_identity_sha256: str, capture_sha256: str, runtime_artifact_sha256: str
) -> str:
    descriptor = _closed(
        json.loads(compiler_input.read_text(encoding="utf-8")),
        {
            "schema_version",
            "kind",
            "compiler_or_framework_source_sha256",
            "capture_sha256",
            "runtime_artifact_sha256",
            "work_iterations",
        },
        "unit-test compiler input",
    )
    work = descriptor["work_iterations"]
    if (
        descriptor["schema_version"] != 1
        or descriptor["kind"] != "paper_unit_test_affine_compiler_input_v1"
        or descriptor["compiler_or_framework_source_sha256"] != source_identity_sha256
        or descriptor["capture_sha256"] != capture_sha256
        or descriptor["runtime_artifact_sha256"] != runtime_artifact_sha256
        or type(work) is not int
        or not 1000 <= work <= 10_000_000
    ):
        raise ValueError("unit-test compiler input differs from its frozen package identities")
    return f"""#include <stddef.h>
#include <stdio.h>
int merlin_paper_step(const char *artifact_path, const unsigned char *input,
                      size_t input_size, unsigned char *output, size_t *output_size) {{
  if (input_size != sizeof(float) || *output_size < 2 * sizeof(float)) return 2;
  FILE *artifact = fopen(artifact_path, "rb");
  float weight = 0.0f, bias = 0.0f;
  if (!artifact) return 3;
  if (fscanf(artifact, "/* %f %f */", &weight, &bias) != 2) return 4;
  fclose(artifact);
  float value, result[2];
  __builtin_memcpy(&value, input, sizeof(value));
  result[0] = value * weight + bias; result[1] = value;
  __builtin_memcpy(output, &result, sizeof(result));
  *output_size = sizeof(result);
  volatile unsigned long work = 1;
  for (unsigned long i = 1; i < {work}UL; ++i) work = work * 33u + i;
  return (int)(work & 0u);
}}
"""


def regenerate_model_object(
    *,
    recipe: str,
    registry_id: str,
    target: str,
    compiler_input: Path,
    tool: Path,
    output: Path,
    source_identity_sha256: str,
    capture_sha256: str,
    runtime_artifact_sha256: str,
    timeout_seconds: float = 120,
) -> dict[str, object]:
    """Regenerate a model object using only a closed, shipped recipe.

    No private benchmark input or reference path is accepted by this interface.  That makes it
    possible for freeze/contract construction to finish this check before private I/O is opened.
    """
    expected = expected_recipe(registry_id, target)
    if recipe != expected:
        raise ValueError("model-object recipe differs from the registry-owned recipe")
    if recipe == EXECUTORCH_RECIPE:
        resources = executorch_session_resources(compiler_input, include_private=False)
        identities = resources.identities
        if (identities["framework_source_sha256"] != source_identity_sha256
                or identities["capture_sha256"] != capture_sha256
                or _sha(resources.package_metadata) != runtime_artifact_sha256):
            raise ValueError("ExecuTorch compiler input differs from its frozen package identities")
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(resources.runner, output)
        output.chmod(resources.runner.stat().st_mode & 0o777)
        return {
            "recipe": recipe,
            "compiler_input_sha256": _sha(compiler_input),
            "generated_source_sha256": _sha(resources.producer_receipt),
            "object_build_argv": object_build_argv(recipe),
            "model_object_sha256": _sha(output),
        }
    if not compiler_input.is_file() or compiler_input.is_symlink():
        raise ValueError("frozen compiler input is absent or unsafe")
    if recipe == MERLIN_RECIPE:
        descriptor, refs = _load_merlin_input(compiler_input)
        if (
            descriptor["compiler_or_framework_source_sha256"] != source_identity_sha256
            or descriptor["capture_sha256"] != capture_sha256
            or descriptor["runtime_artifact_sha256"] != runtime_artifact_sha256
        ):
            raise ValueError("Merlin compiler input differs from its frozen package identities")
        from .paper_merlin_mlir_producer import verify_merlin_mlir_build_barrier

        barrier = verify_merlin_mlir_build_barrier(
            refs["public_manifest"], refs["producer_authority"], refs["producer_receipt"]
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(barrier.composite_object, output)
        receipt = json.loads(refs["producer_receipt"].read_text(encoding="ascii"))
        generated = _closed(
            receipt["outputs"]["session_adapter_source"], {"path", "sha256", "size"}, "Merlin generated session adapter"
        )
        return {
            "recipe": recipe,
            "compiler_input_sha256": _sha(compiler_input),
            "generated_source_sha256": generated["sha256"],
            "object_build_argv": object_build_argv(recipe),
            "model_object_sha256": _sha(output),
        }
    if not tool.is_file() or tool.is_symlink() or not tool.read_bytes().startswith(b"\x7fELF"):
        raise ValueError("model-object build tool is not a bound ELF")
    source_text = _unit_test_source(
        compiler_input,
        source_identity_sha256=source_identity_sha256,
        capture_sha256=capture_sha256,
        runtime_artifact_sha256=runtime_artifact_sha256,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="merlin-paper-object-source-") as temporary:
        generated_source = Path(temporary) / "registry-generated-model.c"
        generated_source.write_text(source_text, encoding="utf-8")
        argv = [str(tool), "-O2", "-std=c11", "-c", str(generated_source), "-o", str(output)]
        completed = subprocess.run(argv, capture_output=True, timeout=timeout_seconds, check=False)
        if completed.returncode or not output.is_file():
            raise ValueError(
                "registry-owned model-object regeneration failed: " + completed.stderr.decode(errors="replace")[-500:]
            )
        generated_sha = _sha(generated_source)
    return {
        "recipe": recipe,
        "compiler_input_sha256": _sha(compiler_input),
        "generated_source_sha256": generated_sha,
        "object_build_argv": object_build_argv(recipe),
        "model_object_sha256": _sha(output),
    }


__all__ = [
    "EXECUTORCH_RECIPE",
    "EXECUTORCH_OBJECT_BUILD_ARGV",
    "ExecuTorchSessionResources",
    "MERLIN_COMPILER_INPUT_KIND",
    "MERLIN_OBJECT_BUILD_ARGV",
    "MERLIN_RECIPE",
    "MerlinSessionResources",
    "OBJECT_BUILD_ARGV",
    "UNIT_TEST_RECIPE",
    "expected_recipe",
    "executorch_session_resources",
    "merlin_session_resources",
    "object_build_argv",
    "regenerate_model_object",
    "stage_compiler_input",
    "write_merlin_compiler_input",
]
