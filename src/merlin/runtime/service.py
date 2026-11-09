"""Execute existing products under an explicit deployment and runtime selection."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from merlin.compiler.artifact import Artifact, Compilation, Deployment


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
        from .external import prepare

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
        from . import spike as provider

        name = "spike"
    else:
        from merlin.compiler.backend.service import resolve

        name, provider = resolve(runner, group="merlin.runners", methods=("validate", "run"))
    check_inputs()
    provider.validate(request)
    check_inputs()
    result = provider.run(request)
    check_inputs()
    if not isinstance(result, RunResult) or result.runner != name or result.artifact_sha256 != original_digest:
        raise ValueError("runtime result does not match the selected artifact and runner")
    return result
