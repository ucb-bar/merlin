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

The CLI emits private stage records on stderr and atomically replaces one bounded
progress JSON file in its output directory. API callers can opt into the same
journal or a callback. These observations never grant admission, and stop before
the caller hashes a completed preparation for reuse.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPORT = "private_full_model_source_preflight.json"
PROGRESS = "private_full_model_source_progress.json"


def _notify(progress, stage, **details):
    """Observational callbacks cannot alter admission or a formal verdict."""
    if progress is not None:
        try:
            progress({"stage": stage, **details})
        except Exception:
            pass


def _observer(callback, path):
    """Keep one bounded, atomic private record; never change an admission result."""
    started, sequence = time.monotonic(), 0

    def observe(event):
        nonlocal sequence
        sequence += 1
        record = {
            "schema": "merlin.phase1.private_source_progress.v1",
            "sequence": sequence,
            "updated_at": datetime.now(UTC).isoformat(),
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }
        for key in ("stage", "model", "program", "status"):
            if isinstance(event.get(key), str):
                record[key] = event[key][:128]
        for key in ("current", "total"):
            if type(event.get(key)) is int and event[key] >= 0:
                record[key] = event[key]
        if path is not None:
            temporary = path.with_name(path.name + ".pending")
            try:
                fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
                with os.fdopen(fd, "w") as stream:
                    json.dump(record, stream, sort_keys=True)
                    stream.write("\n")
                os.replace(temporary, path)
            except OSError:
                pass
        if callback is not None:
            try:
                callback(record)
            except Exception:
                pass

    return observe


def preflight(
    spec: str | Path,
    *,
    descriptor: str | Path,
    out: str | Path,
    target: str,
    native_result_format_selection: str | Path | None = None,
    progress=None,
    progress_path: str | Path | None = None,
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
    observe = _observer(progress, Path(progress_path) if progress_path is not None else None)
    sources = out / "sources"
    observe({"stage": "source_freeze"})
    freeze = private_source_freeze.stage(spec, sources, target=target)
    observe({"stage": "descriptor"})
    selection, _record = freeze_for_run(native_result_format_selection, target=target, run_dir=out)
    required_models = models.requirements_for(descriptor)
    required_programs = models.program_requirements_for(descriptor)
    gate = execution.gate_for(descriptor, required_programs=required_programs)
    observe({"stage": "selected_facts"})
    with selected_input_facts(spec, target=target, required_models=required_models):
        result = models.run(
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
            **({"progress": observe} if progress is not None or progress_path is not None else {}),
        )
    observe({"stage": "complete", "status": "source_inputs_verified" if result["source_preflight_passed"] else "fail"})
    return result


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
        "--progress-file", type=Path, help="atomic private stage record (default: OUT/" + PROGRESS + ")"
    )
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
        progress=lambda event: print(json.dumps(event, sort_keys=True), file=sys.stderr, flush=True),
        progress_path=args.progress_file or args.out / PROGRESS,
    )
    for item in result["models"]:
        print(f"{item['model']}: {item['status']}" + (f" -- {item['reason']}" if item.get("reason") else ""))
    passed = result["source_preflight_passed"] is True
    print(json.dumps({"source_preflight_passed": passed, "report": str(Path(args.out).absolute() / REPORT)}))
    return 0 if passed else 1
