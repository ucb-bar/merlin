"""Execute existing products under an explicit deployment and runtime selection."""

from __future__ import annotations

import hashlib
import json
import math
import shlex
import shutil
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import uuid4

from merlin.compiler.compilation import Artifact, Compilation, Deployment

# ---- service -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class RunOptions:
    deployment: Deployment | None = None
    harts: int | None = None
    timeout: float = 300
    role: str = "executable"
    mode: str = "execute"
    output: Path | None = None
    arguments: tuple[str, ...] = ()
    inputs: dict[str, Path] = field(default_factory=dict)
    entry: str | None = None

    def __post_init__(self) -> None:
        if self.harts is not None and (type(self.harts) is not int or self.harts < 1):
            raise ValueError("harts must be a positive integer")
        if isinstance(self.timeout, bool) or not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("timeout must be positive and finite")
        if self.mode not in {"execute", "prepare"}:
            raise ValueError("runtime mode must be execute or prepare")
        if type(self.arguments) is not tuple or any(
            type(value) is not str or "\x00" in value for value in self.arguments
        ):
            raise ValueError("runtime arguments must be a tuple of strings without NUL")


@dataclass(frozen=True)
class RunRequest:
    artifact: Artifact
    deployment: Deployment
    harts: int
    timeout: float
    arguments: tuple[str, ...] = ()


@dataclass(frozen=True)
class RunResult:
    runner: str
    artifact_sha256: str
    status: str
    stdout: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def _execution_identity(request, options):
    def path(value):
        if not isinstance(value, Path):
            raise TypeError("runtime identity contains an unsupported value")
        return str(value)

    # Serialize copies rather than retaining aliases to caller-owned mappings
    # or the frozen dataclasses' mutable reachable objects.
    return json.dumps(
        {"request": asdict(request), "options": asdict(options)}, sort_keys=True, allow_nan=False, default=path
    )


def run_artifact(product, *, runner, options: RunOptions | None = None):
    options = options or RunOptions()
    # FireSim jobs are explicitly prepared for external execution. Refuse before
    # resolving any plugin, launcher, cloud client, or infrastructure adapter.
    if runner == "firesim":
        if options.mode != "prepare":
            raise ValueError("FireSim supports prepare/collect here; execution is external")
        if options.role != "executable":
            raise ValueError("external preparation selects an explicit entry path, not a runtime role override")

        if isinstance(product, Compilation):
            external = product
        elif isinstance(product, Artifact):
            external = product
        else:
            path = Path(product)
            external = (
                Compilation.load(path)
                if path.is_dir() or path.name == "compilation.json"
                else Artifact.from_file(path, kind="elf", role="executable")
            )
        contract = external.deployment if isinstance(external, Compilation) else options.deployment
        if isinstance(external, Compilation) and options.deployment is not None and options.deployment != contract:
            raise ValueError("runtime cannot override the compiled deployment contract")
        if contract is None:
            raise ValueError("external execution requires an explicit Deployment contract")
        if options.harts is not None and options.harts != contract.harts:
            raise ValueError("runtime hart count differs from the image contract")
        if options.output is None:
            raise ValueError("FireSim preparation requires an explicit output directory")
        return prepare(
            external,
            output=options.output,
            artifact=options.entry,
            arguments=options.arguments,
            inputs=options.inputs,
            **({"deployment": contract} if isinstance(external, Artifact) else {}),
        )
    if options.mode != "execute":
        raise ValueError("prepare mode requires an external runtime such as FireSim")
    if options.inputs or options.output is not None or options.entry is not None:
        raise ValueError("this runtime adapter does not support external preparation options")
    compilation = None
    if isinstance(product, Compilation):
        compilation = product
    elif not isinstance(product, Artifact):
        path = Path(product)
        if path.is_dir() or path.name == "compilation.json":
            compilation = Compilation.load(path)
        else:
            if options.deployment is None:
                raise ValueError("an external ELF requires an explicit Deployment contract")
            product = Artifact.from_file(path, kind="elf", role=options.role)
    if compilation is not None:
        compilation.verify()
        choices = [a for a in compilation.artifacts if a.role == options.role]
        if len(choices) != 1:
            raise ValueError(f"expected one artifact with role {options.role!r}, found {len(choices)}")
        product = choices[0]
        if options.deployment is not None and options.deployment != compilation.deployment:
            raise ValueError("runtime cannot override the compiled deployment contract")
        deployment = compilation.deployment
    else:
        deployment = options.deployment
    if deployment is None:
        raise ValueError("runtime requires a deployment contract; artifact kind does not imply a platform")
    harts = options.harts if options.harts is not None else deployment.harts
    if harts != deployment.harts:
        raise ValueError("runtime hart count differs from the image contract; rebuild or supply its actual contract")
    product.verify()
    request = RunRequest(product, deployment, harts, options.timeout, options.arguments)
    original = _execution_identity(request, options)
    original_path, original_digest = product.path, product.sha256

    def check_inputs():
        try:
            unchanged = _execution_identity(request, options) == original
        except (TypeError, ValueError):
            unchanged = False
        if not unchanged or request.artifact.path.resolve() != original_path:
            raise ValueError("runtime callback changed the original execution request or options")
        request.artifact.verify()

    if runner == "spike":
        from merlin.runtime.execution import spike_runner as provider

        name = "spike"
    else:
        from merlin.compiler.backend.providers import resolve

        name, provider = resolve(runner, group="merlin.runners", methods=("validate", "run"))
    check_inputs()
    provider.validate(request)
    check_inputs()
    result = provider.run(request)
    check_inputs()
    if not isinstance(result, RunResult) or result.runner != name or result.artifact_sha256 != original_digest:
        raise ValueError("runtime result does not match the selected artifact and runner")
    return result


# ---- external ----------------------------------------------------------------------------------
# Prepare portable execution handoffs and collect externally supplied observations.
#
# There is no process launch, queue client, farm configuration or provisioning in
# this module. Matching hashes bind files to a request; they do not authenticate
# an operator's execution claim or establish numerical or timing correctness.


@dataclass(frozen=True)
class PreparedRun:
    directory: Path
    request_sha256: str
    status: str = "prepared"


@dataclass(frozen=True)
class ExternalObservation:
    directory: Path
    request_sha256: str
    reported_exit_code: int
    artifacts: tuple[dict, ...]
    status: str = "collected"


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def _member(root: Path, name: str) -> Path:
    if not isinstance(name, str) or not name or "\\" in name or ":" in name or "\x00" in name:
        raise ValueError("handoff files require plain relative POSIX paths")
    relative = PurePosixPath(name)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in name.split("/")):
        raise ValueError("handoff file escapes its package")
    path = root.joinpath(*relative.parts)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("handoff file escapes its package")
    return path


def _write(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _copy(source: Path, destination: Path, expected: str | None = None) -> dict:
    if not source.is_file():
        raise ValueError(f"handoff input is not a file: {source}")
    before = _digest(source)
    if expected is not None and before != expected:
        raise ValueError(f"handoff input changed: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    if _digest(source) != before or _digest(destination) != before:
        raise ValueError(f"handoff input changed while copying: {source}")
    return {"sha256": before, "bytes": destination.stat().st_size}


def _preparation_identity(product):
    record = (
        {"directory": str(product.directory), "compilation": product.to_dict()}
        if isinstance(product, Compilation)
        else asdict(product)
    )
    return json.dumps(record, sort_keys=True, allow_nan=False, default=str)


def prepare(
    compilation: Compilation | Artifact,
    *,
    output: str | Path,
    artifact: str | None = None,
    arguments: tuple[str, ...] = (),
    inputs: Mapping[str, Path] | None = None,
    deployment: Deployment | None = None,
) -> PreparedRun:
    """Package an admitted image for an externally managed FireSim environment.

    Linux images get a guest invocation script. Bare-metal and Zephyr images
    get a boot description; they are never invoked as Linux shell programs.
    Input files retain explicit relative names and are not interpreted as tensors.
    """
    original = compilation
    if not isinstance(original, (Compilation, Artifact)):
        raise TypeError("external preparation requires a Compilation or an explicit Artifact")
    original.verify()
    original_identity = _preparation_identity(original)
    if isinstance(compilation, Compilation):
        compilation = Compilation.load(compilation.directory)
        if _preparation_identity(compilation) != original_identity:
            raise ValueError("compiled manifest differs from the caller's selected build")
        manifest_digest = _digest(compilation.manifest)
        if deployment is not None and deployment != compilation.deployment:
            raise ValueError("external preparation cannot override the compiled deployment")
        deployment = compilation.deployment
        artifacts = compilation.artifacts
        source_directory = compilation.directory
    elif isinstance(compilation, Artifact):
        compilation.verify()
        artifacts = (compilation,)
        source_directory = compilation.path.parent
    if deployment is None or deployment.platform not in {"linux", "baremetal", "zephyr"}:
        raise ValueError("external execution requires an explicit Linux, bare-metal or Zephyr deployment contract")
    deployment_identity = asdict(deployment)
    arguments = tuple(arguments)
    if not all(isinstance(value, str) and "\x00" not in value for value in arguments):
        raise ValueError("guest arguments must be strings without NUL")
    if deployment.platform != "linux" and arguments:
        raise ValueError("boot images do not admit Linux command-line arguments")

    def check_inputs():
        original.verify()
        if _preparation_identity(original) != original_identity or asdict(deployment) != deployment_identity:
            raise ValueError("external preparation changed the original selected input")
        if isinstance(compilation, Compilation):
            current = Compilation.load(source_directory)
            if _preparation_identity(current) != original_identity or _digest(current.manifest) != manifest_digest:
                raise ValueError("compiled manifest changed during external preparation")

    check_inputs()
    images = [item for item in artifacts if item.kind == "elf"]
    if artifact is not None:
        images = [item for item in images if item.path.relative_to(source_directory).as_posix() == artifact]
    if len(images) != 1:
        raise ValueError("select exactly one ELF entry image for the external workload")
    directory = Path(output).resolve()
    if directory.exists():
        raise FileExistsError(f"prepared workload requires a fresh directory: {directory}")
    declared_inputs = dict(inputs or {})
    for name in declared_inputs:
        _member(directory / "inputs", name)
    directory.mkdir(parents=True)
    files = []
    for item in artifacts:
        relative = item.path.relative_to(source_directory).as_posix()
        name = "payload/" + relative
        metadata = _copy(item.path, _member(directory, name), item.sha256)
        files.append({"path": name, "kind": item.kind, "role": item.role, **metadata})
    for name, source in declared_inputs.items():
        name = "inputs/" + name
        files.append({"path": name, "kind": "data", "role": "input", **_copy(Path(source), _member(directory, name))})
    if isinstance(compilation, Compilation):
        manifest = compilation.directory / "compilation.json"
        files.append(
            {
                "path": "build.json",
                "kind": "json",
                "role": "build-record",
                **_copy(manifest, directory / "build.json", manifest_digest),
            }
        )
    else:
        record = directory / "external-image.json"
        _write(
            record,
            {
                "schema": "merlin.external_image.v1",
                "sha256": compilation.sha256,
                "deployment": asdict(deployment),
                "compilation_provenance": "not_supplied",
                "deployment_authority": "caller",
                "qualification": "not_established",
            },
        )
        files.append(
            {
                "path": record.name,
                "kind": "json",
                "role": "external-image-record",
                "sha256": _digest(record),
                "bytes": record.stat().st_size,
            }
        )
    entry = "payload/" + images[0].path.relative_to(source_directory).as_posix()
    if deployment.platform == "linux":
        script = directory / "invoke.sh"
        script.write_text(
            '#!/bin/sh\nset -eu\ncd -- "$(dirname -- "$0")"\nexec '
            + " ".join(shlex.quote(value) for value in ("./" + entry, *arguments))
            + "\n",
            encoding="utf-8",
        )
        script.chmod(0o755)
        _member(directory, entry).chmod(0o755)
        files.append(
            {
                "path": "invoke.sh",
                "kind": "shell",
                "role": "guest-invocation",
                "sha256": _digest(script),
                "bytes": script.stat().st_size,
            }
        )
    request = {
        "schema": "merlin.external_request.v1",
        "id": uuid4().hex,
        "runner": "firesim",
        "status": "prepared",
        "deployment": asdict(deployment),
        "entry": entry,
        "invocation": "guest-command" if deployment.platform == "linux" else "boot-image",
        "arguments": list(arguments),
        "files": files,
        "result_protocol": "merlin.external_result.v1",
        "execution": "external_operator_required",
        "qualification": "not_established",
    }
    # Reopening a self-consistent replacement does not renew the original
    # selection. Compare its receipt as well as the selected input bytes.
    check_inputs()
    _write(directory / "request.json", request)
    return PreparedRun(directory, _digest(directory / "request.json"))


def _open(prepared: PreparedRun | str | Path) -> tuple[Path, str, dict]:
    directory = Path(prepared.directory if isinstance(prepared, PreparedRun) else prepared).resolve()
    path = directory / "request.json"
    digest = _digest(path)
    if isinstance(prepared, PreparedRun) and prepared.request_sha256 != digest:
        raise ValueError("prepared request identity changed")
    request = json.loads(path.read_text(encoding="utf-8"))
    if request.get("schema") != "merlin.external_request.v1" or request.get("status") != "prepared":
        raise ValueError("not a prepared external workload")
    for item in request["files"]:
        if _digest(_member(directory, item["path"])) != item["sha256"]:
            raise ValueError(f"prepared workload file changed: {item['path']}")
    return directory, digest, request


def collect(prepared: PreparedRun | str | Path, *, results: str | Path) -> ExternalObservation:
    """Retain matched external bytes without issuing an execution or quality verdict.

    The operator supplies result.json with schema, request_sha256, integer
    exit_code and artifacts (relative path, sha256 and optional role). Missing,
    conflicting or altered results fail; they are never counted as a pass.
    """
    directory, request_digest, _ = _open(prepared)
    root = Path(results).resolve()
    report_path = root / "result.json"
    report_digest = _digest(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema") != "merlin.external_result.v1" or report.get("request_sha256") != request_digest:
        raise ValueError("external results do not match this prepared request")
    if type(report.get("exit_code")) is not int:
        raise ValueError("external results require an explicit integer exit_code")
    records = report.get("artifacts")
    if not isinstance(records, list) or not records:
        raise ValueError("external results require observed output artifacts")
    seen = set()
    for item in records:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise ValueError("invalid external result artifact")
        name = item["path"]
        source = _member(root, name)
        if name in seen or name == "result.json" or name == "observation.json":
            raise ValueError("duplicate or reserved external result artifact")
        seen.add(name)
        if _digest(source) != item.get("sha256"):
            raise ValueError(f"external result content changed: {name}")
    destination = directory / "observations"
    destination.mkdir()
    retained = []
    for item in records:
        name = item["path"]
        retained.append(
            {
                "path": name,
                "role": item.get("role", "output"),
                **_copy(_member(root, name), _member(destination, name), item["sha256"]),
            }
        )
    _copy(report_path, destination / "result.json", report_digest)
    _open(PreparedRun(directory, request_digest))
    _write(
        destination / "observation.json",
        {
            "schema": "merlin.external_observation.v1",
            "status": "collected",
            "request_sha256": request_digest,
            "reported_exit_code": report["exit_code"],
            "artifacts": retained,
            "execution_authority": "external_operator_report",
            "qualification": "not_established",
        },
    )
    return ExternalObservation(destination, request_digest, report["exit_code"], tuple(retained))
