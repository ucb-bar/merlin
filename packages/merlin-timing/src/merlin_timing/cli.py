"""spike-timing: portable empirical fitting, held-out checks and probe planning."""

from __future__ import annotations

import argparse
import json
import sys
import tarfile

from . import archive as A
from . import capacities as K
from . import contracts as C
from . import design as D
from . import feedback as B
from . import fitting as F
from . import native_sensitivity as S
from . import replay as R
from . import validation as V


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="spike-timing", description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("fit", help="fit a diagnostic nonnegative empirical screen")
    fit.add_argument("--dataset", required=True)
    fit.add_argument("--maximum-condition", type=float, default=1e6)
    fit.add_argument("--no-fixed", action="store_true", help="explicitly omit fixed invocation cost")
    fit.add_argument(
        "--loss",
        choices=("absolute_squared", "relative_squared"),
        default="absolute_squared",
        help="explicit v3 relative loss balances training cases across latency scales",
    )
    fit.add_argument(
        "--training-intake",
        choices=("all_declared", "eligible_records_v1"),
        default="all_declared",
        help="explicit v2 eligible-record intake preserves unavailable cases and all held denominators",
    )
    validate = commands.add_parser("validate", help="validate held source groups without refitting on them")
    validate.add_argument("--dataset", required=True)
    validate.add_argument("--model", help="required for empirical_screen; unused for event_timing")
    validate.add_argument("--prediction-kind", choices=("empirical_screen", "event_timing"), default="empirical_screen")
    sensitivity = commands.add_parser(
        "native-sensitivity", help="diagnose TRAIN-only native price responses without fitting or launching"
    )
    sensitivity.add_argument("--dataset", required=True)
    sensitivity.add_argument("--evaluations", required=True)
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
    replay = commands.add_parser("archive-replay-plan", help="freeze a new-model replay without relabeling old reports")
    replay.add_argument("--plan", required=True)
    replay.add_argument("--model-artifact", required=True)
    replay.add_argument("--spike-executable", required=True)
    replay.add_argument("--provider-library", action="append", required=True)
    attach = commands.add_parser("archive-attach", help="join original native reports to the exact prepared plan")
    attach.add_argument("--plan", required=True)
    attach.add_argument("--reports", required=True)
    design = commands.add_parser("archive-design-check", help="check measured variant pairs before native tuning")
    design.add_argument("--plan", required=True)
    handoff = commands.add_parser(
        "archive-handoff", help="freeze original labels, native predictions and held-out errors"
    )
    handoff.add_argument("--plan", required=True)
    handoff.add_argument("--dataset", required=True)
    handoff.add_argument("--spike-executable", required=True)
    handoff.add_argument("--spike-configuration", required=True)
    handoff.add_argument("--model-artifact", required=True)
    handoff.add_argument("--training-code", required=True)
    capacities = commands.add_parser(
        "apply-capacities", help="place RTL-derived structural facts into a device profile by declared role"
    )
    capacities.add_argument("--profile", required=True)
    capacities.add_argument("--facts", required=True, help=f"{K.SCHEMA} document")
    for command in (
        fit,
        validate,
        sensitivity,
        compare,
        native,
        probe,
        inspect,
        prepare,
        replay,
        attach,
        design,
        handoff,
        capacities,
    ):
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
                training_intake=arguments.training_intake,
                loss=arguments.loss,
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
        elif arguments.command == "native-sensitivity":
            result = S.analyze(C.read(arguments.dataset), C.read(arguments.evaluations))
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
        elif arguments.command == "archive-replay-plan":
            result = R.prepare(
                C.read(arguments.plan), arguments.model_artifact, arguments.spike_executable, arguments.provider_library
            )
        elif arguments.command == "archive-attach":
            result = A.attach(C.read(arguments.plan), C.read(arguments.reports))
        elif arguments.command == "archive-design-check":
            result = D.check(C.read(arguments.plan))
        elif arguments.command == "apply-capacities":
            result = K.apply_files(arguments.profile, arguments.facts)
        else:
            result = A.handoff(
                C.read(arguments.plan),
                C.read(arguments.dataset),
                {
                    key + "_sha256": A.file_identity(getattr(arguments, key))
                    for key in ("spike_executable", "spike_configuration", "model_artifact", "training_code")
                },
            )
        if arguments.command.startswith("archive-") or arguments.command == "apply-capacities":
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
