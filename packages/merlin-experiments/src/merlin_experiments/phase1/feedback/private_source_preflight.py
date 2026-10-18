"""Compiler-independent preflight of an operator-private full-model specification.

The private full-model gate runs once, after a run's authoring window, yet much of
what it refuses does not depend on the compiler: whether each capture is the
preselected sealed run output and still re-verifies against its sealed receipt,
whether its frontend trace attributes every literal range, and whether the selected
software, host and capability declarations decide every source operation.  An
input that fails there fails the gate for every compiler.

This runs exactly those checks -- :func:`.private_full_models.run` with
``source_only=True``, under the same selected RTL facts, authored-source freeze,
execution build options and optional native result-format selection a formal
grade would use -- with no candidate, and reports each model's first refusal.
It grants nothing: a passing preflight is not a gate result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REPORT = "private_full_model_source_preflight.json"


def preflight(
    spec: str | Path,
    *,
    descriptor: str | Path,
    out: str | Path,
    target: str,
    native_result_format_selection: str | Path | None = None,
) -> dict[str, Any]:
    """Run the gate's compiler-independent checks for every declared model into a fresh ``out``."""
    from . import private_full_model_execution as execution
    from . import private_full_models as models
    from . import private_source_freeze
    from .private_facts import selected_input_facts
    from .private_result_format_selection import freeze_for_run

    spec, out = Path(spec).absolute(), Path(out).absolute()
    if out.exists() or out.is_symlink():
        raise ValueError("source preflight needs a fresh private output directory")
    out.mkdir(mode=0o700, parents=True)
    sources = out / "sources"
    freeze = private_source_freeze.stage(spec, sources, target=target)
    selection, _record = freeze_for_run(native_result_format_selection, target=target, run_dir=out)
    required_models = models.requirements_for(descriptor)
    required_programs = models.program_requirements_for(descriptor)
    gate = execution.gate_for(descriptor, required_programs=required_programs)
    with selected_input_facts(spec, target=target, required_models=required_models):
        return models.run(
            None,
            spec,
            target=target,
            required_models=required_models,
            required_programs=required_programs,
            loader_env_requirements=models.loader_env_requirements_for(descriptor),
            out=out,
            source_freeze=freeze,
            source_freeze_root=sources,
            build_options=execution.build_options(gate) if gate is not None else None,
            native_result_format_selection=selection,
            source_only=True,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="merlin-experiment private-preflight",
        description="Check an operator-private full-model specification before a launch, without a compiler.",
    )
    parser.add_argument("--spec", required=True, type=Path, help="operator-private full-model specification")
    parser.add_argument("--descriptor", required=True, type=Path, help="target experiment descriptor of the launch")
    parser.add_argument("--target", required=True, help="target the specification and descriptor select")
    parser.add_argument("--out", required=True, type=Path, help="fresh operator-private output directory")
    parser.add_argument(
        "--native-result-format-selection",
        default="",
        metavar="FILE",
        help="the launch's original negative native output-format selection, when it passes one",
    )
    args = parser.parse_args(argv)
    result = preflight(
        args.spec,
        descriptor=args.descriptor,
        out=args.out,
        target=args.target,
        native_result_format_selection=args.native_result_format_selection,
    )
    for item in result["models"]:
        print(f"{item['model']}: {item['status']}" + (f" -- {item['reason']}" if item.get("reason") else ""))
    passed = result["source_preflight_passed"] is True
    print(json.dumps({"source_preflight_passed": passed, "report": str(Path(args.out).absolute() / REPORT)}))
    return 0 if passed else 1
