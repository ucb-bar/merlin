"""Public compile, inspect, and run services shared with the installed CLI.

Compilation requires an explicitly selected provider. SystemCompiler discovers
regions from admitted input; PlannedCompiler consumes a selected plan. Framework
capture and compiler-only stages retain their own admission. C image linking is
optional and caller-selected; none of these services establishes execution or
numerical correctness.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from merlin.compiler.artifact import Artifact, Compilation, Deployment
from merlin.compiler.frontend.staged import StagedProgram
from merlin.compiler.frontend.torch.export import SavedExport
from merlin.compiler.request import CompileOptions, CompileRequest, Source

__all__ = [
    "Artifact",
    "Compilation",
    "CompileOptions",
    "CompileRequest",
    "Deployment",
    "Source",
    "SavedExport",
    "StagedProgram",
    "collect",
    "compile",
    "inspect",
    "inspect_backends",
    "inspect_spec",
    "run",
]


def compile(
    source: Any,
    *,
    backend: str | Any,
    output: str | Path,
    options: CompileOptions | None = None,
    example_inputs: tuple | None = None,
    capture_output: str | Path | None = None,
) -> Compilation:
    """Compile source using one explicitly selected provider and fresh output.

    A string is source text; use Path for a source file. Provider validation
    rejects unknown options before compilation. Multiple products may have any
    declared artifact kind, including device binaries or command streams.
    PyTorch modules, FX graphs and functions take example_inputs=(...); an
    ExportedProgram or .pt2 Path carries its own inputs. Python objects are frozen before import in a
    separate capture workspace, defaulting to OUTPUT.capture. A SavedExport uses
    its explicit workspace. StagedProgram selects existing compiler-only inputs
    and requires explicit provider admission of their precision policy. It runs
    no capture or quantization. Framework imports are lazy for ordinary MLIR users.
    """
    from merlin.compiler.backend.service import resolve
    from merlin.compiler.invocation import invoke

    target = Path(output).resolve()
    if target.exists():
        raise FileExistsError(f"compilation requires a fresh output directory: {target}")
    if isinstance(source, StagedProgram) and (
        target.is_relative_to(source.directory.resolve()) or source.directory.resolve().is_relative_to(target)
    ):
        raise ValueError("compiler stage and compilation output require separate directories")
    workspace = (
        source.capture
        if isinstance(source, SavedExport)
        else Path(capture_output).absolute()
        if capture_output is not None
        else target.with_name(target.name + ".capture")
    )
    if (
        isinstance(source, SavedExport)
        or not isinstance(source, (str, Path, Source))
        or isinstance(source, Path)
        and source.suffix.lower() == ".pt2"
    ):
        if workspace.resolve().is_relative_to(target) or target.is_relative_to(workspace.resolve()):
            raise ValueError("capture evidence and compilation output require separate directories")
    if isinstance(source, Path) and source.suffix.lower() == ".pt2":
        if example_inputs is not None:
            raise ValueError("a .pt2 archive already carries its example inputs")
        source = SavedExport(source, capture=workspace)
        capture_output = None
    selected_options = options or CompileOptions()
    parameters = dict(selected_options.parameters)
    budget = parameters.pop("frontend.max-initialization-bytes", 64 * 1024 * 1024)
    if type(budget) is not int or budget < 0:
        raise ValueError("frontend.max-initialization-bytes must be a nonnegative integer")
    captured = isinstance(source, (SavedExport, StagedProgram)) or not isinstance(source, (str, Path, Source))
    if "frontend.max-initialization-bytes" in selected_options.parameters and not captured:
        raise ValueError("frontend.max-initialization-bytes requires a captured framework program")
    if parameters != selected_options.parameters:
        from dataclasses import replace

        selected_options = replace(selected_options, parameters=parameters)
    frozen = None
    if isinstance(source, (str, Path, Source, SavedExport, StagedProgram)):
        if example_inputs is not None or capture_output is not None:
            raise ValueError("example_inputs and capture_output apply to Python framework objects")
    else:
        from merlin.compiler.frontend.torch.program import freeze

        source, frozen = freeze(
            source,
            example_inputs=example_inputs,
            workspace=workspace,
        )
    prepared = None
    preparation = None
    if isinstance(source, StagedProgram):
        from merlin.compiler.frontend.staged import prepare as prepare_staged

        source, preparation = prepare_staged(source, max_initialization_bytes=budget)
    if isinstance(source, SavedExport):
        from merlin.compiler.frontend.torch.export import prepare

        source, stage, record, bound = prepare(
            source,
            max_initialization_bytes=budget,
            original_program=frozen,
        )
        prepared = stage, record, bound
    if isinstance(source, Path):
        source = Source(source.read_text(encoding="utf-8"), origin=str(source.resolve()))
    elif isinstance(source, str):
        source = Source(source)
    if not isinstance(source, Source):
        raise TypeError("this compilation boundary accepts Source, text, or Path")
    request = CompileRequest(source, Path(output), selected_options, preparation=preparation)
    if request.output.exists():
        raise FileExistsError(f"compilation requires a fresh output directory: {request.output}")
    name, provider = resolve(backend, group="merlin.backends", methods=("validate", "compile"))
    result = invoke(request, name, provider)
    if preparation is not None:
        from merlin.compiler.frontend.torch.export import publish

        try:
            preparation.revalidate(source)
            record = preparation.record()
            record["compiler_mlir_sha256"] = source.sha256
            record["argument_binding"] = preparation.bound.record
            result = publish(result, preparation.stage.directory, record, preparation.bound)
        except Exception:
            result.manifest.unlink(missing_ok=True)
            raise
    if prepared is not None:
        from merlin.compiler.frontend.torch.export import publish

        try:
            result = publish(result, *prepared)
        except Exception:
            result.manifest.unlink(missing_ok=True)
            raise
    return result


def inspect(compilation: str | Path | Compilation) -> dict[str, Any]:
    """Read and verify product hashes without importing a backend or executing it."""
    result = compilation if isinstance(compilation, Compilation) else Compilation.load(compilation)
    result.verify()
    return result.to_dict()


def inspect_backends() -> dict[str, Any]:
    """List installed backend names and declared origins without loading code."""
    from merlin.compiler.backend.service import inventory

    return inventory()


def inspect_spec(path: str | Path, *, expanded: bool = False) -> dict[str, Any]:
    """Inspect authored semantics and source identities without compiling.

    Expanded inspection includes the resolved definitions and compatible legacy
    view. Inspection reports projection obligations; it grants no numerical or
    hardware qualification and does not execute authored Python.
    """
    if type(expanded) is not bool:
        raise ValueError("expanded specification inspection requires a Boolean")
    from merlin.target.spec import load_spec

    return load_spec(path).record(expanded=expanded)


def run(artifact: str | Path | Compilation | Artifact, *, runner: str | Any, options=None):
    """Execute an existing admitted product; this operation never compiles it."""
    from merlin.runtime.service import run_artifact

    return run_artifact(artifact, runner=runner, options=options)


def collect(prepared, *, results):
    """Collect externally produced observations without submitting runtime work."""
    from merlin.runtime.external import collect as collect_external

    return collect_external(prepared, results=results)
