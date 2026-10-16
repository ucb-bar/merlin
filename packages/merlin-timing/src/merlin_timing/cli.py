"""spike-timing: portable empirical fitting, held-out checks and probe planning."""

from __future__ import annotations

import argparse
import json
import sys
import tarfile

from . import archive as A
from . import contracts as C
from . import feedback as B
from . import fitting as F
from . import validation as V


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="spike-timing", description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("fit", help="fit a diagnostic nonnegative empirical screen")
    fit.add_argument("--dataset", required=True)
    fit.add_argument("--maximum-condition", type=float, default=1e6)
    fit.add_argument("--no-fixed", action="store_true", help="explicitly omit fixed invocation cost")
    validate = commands.add_parser("validate", help="validate held source groups without refitting on them")
    validate.add_argument("--dataset", required=True)
    validate.add_argument("--model", help="required for empirical_screen; unused for event_timing")
    validate.add_argument("--prediction-kind", choices=("empirical_screen", "event_timing"), default="empirical_screen")
    compare = commands.add_parser(
        "compare", help="compare predictions; overlapping/unqualified intervals remain unknown"
    )
    compare.add_argument("--model", required=True)
    compare.add_argument("--cases", required=True)
    compare.add_argument("--qualification", help="receipt with original dataset and its exact validation report")
    native = commands.add_parser(
        "compare-native", help="compare original native event reports using independently qualified residual intervals"
    )
    native.add_argument("--cases", required=True)
    native.add_argument("--qualification", help="receipt with original event dataset and its exact validation report")
    probe = commands.add_parser("probe-plan", help="export missing original-measurement requests without launching")
    probe.add_argument("--dataset", required=True)
    probe.add_argument("--model")
    inspect = commands.add_parser("archive-inspect", help="stream and verify an immutable Phase 1 capsule timing tar")
    prepare = commands.add_parser(
        "archive-prepare", help="prepare reviewed original-ELF observation requests without launching"
    )
    for command in (inspect, prepare):
        command.add_argument("--archive", required=True)
        command.add_argument("--archive-sha256", required=True)
    prepare.add_argument("--binding", required=True, help="reviewed counter, source split and exact execution bindings")
    attach = commands.add_parser("archive-attach", help="join original native reports to the exact prepared plan")
    attach.add_argument("--plan", required=True)
    attach.add_argument("--reports", required=True)
    handoff = commands.add_parser(
        "archive-handoff", help="freeze original labels, native predictions and held-out errors"
    )
    handoff.add_argument("--plan", required=True)
    handoff.add_argument("--dataset", required=True)
    handoff.add_argument("--spike-executable", required=True)
    handoff.add_argument("--spike-configuration", required=True)
    handoff.add_argument("--model-artifact", required=True)
    handoff.add_argument("--training-code", required=True)
    for command in (fit, validate, compare, native, probe, inspect, prepare, attach, handoff):
        command.add_argument("--output", required=True)
    return root


def main(argv: list[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    try:
        if arguments.command == "fit":
            result = F.fit(
                C.read(arguments.dataset),
                include_fixed=not arguments.no_fixed,
                maximum_condition=arguments.maximum_condition,
            )
        elif arguments.command == "validate":
            if arguments.prediction_kind == "event_timing" and arguments.model is not None:
                raise C.ContractError("event_timing uses original native model/report digests, not a fitted screen")
            result = V.validate(
                C.read(arguments.dataset),
                C.read(arguments.model) if arguments.model else None,
                prediction_kind=arguments.prediction_kind,
            )
        elif arguments.command == "compare":
            result = B.compare(
                C.read(arguments.model),
                C.read(arguments.cases),
                qualification=C.read(arguments.qualification) if arguments.qualification else None,
            )
        elif arguments.command == "compare-native":
            result = B.compare_native(
                C.read(arguments.cases),
                qualification=C.read(arguments.qualification) if arguments.qualification else None,
            )
        elif arguments.command == "probe-plan":
            result = B.probe_plan(C.read(arguments.dataset), C.read(arguments.model) if arguments.model else None)
        elif arguments.command in {"archive-inspect", "archive-prepare"}:
            result = A.inspect_archive(arguments.archive, arguments.archive_sha256)
            if arguments.command == "archive-prepare":
                result = A.prepare(result, C.read(arguments.binding))
        elif arguments.command == "archive-attach":
            result = A.attach(C.read(arguments.plan), C.read(arguments.reports))
        else:
            result = A.handoff(
                C.read(arguments.plan),
                C.read(arguments.dataset),
                {
                    key + "_sha256": A.file_identity(getattr(arguments, key))
                    for key in ("spike_executable", "spike_configuration", "model_artifact", "training_code")
                },
            )
        if arguments.command.startswith("archive-"):
            # Preserve immutable intake/handoff outputs; never overwrite one.
            with open(arguments.output, "x", encoding="utf-8") as stream:
                stream.write(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
        else:
            C.write(arguments.output, result)
        return 0
    except (C.ContractError, OSError, ValueError, TypeError, KeyError, RecursionError, tarfile.TarError) as exc:
        print(f"spike-timing: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
