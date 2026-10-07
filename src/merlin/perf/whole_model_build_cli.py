"""``python -m merlin.perf.whole_model_build``: build a whole-model program, or grade a run against its oracle.

The command line of :func:`merlin.perf.whole_model_build.build`, kept apart from it so the builder module
holds the build alone. Beside the build's own options it takes the compile-debugging ones
(:mod:`merlin.compile.debug`: ``--list-stages``, ``--dump-ir-after``, ``--dump-ir-before``,
``--stop-after``, ``--trace-dir``) and ``--only-group``, which builds a PARTIAL program
(:mod:`merlin.perf.whole_model_partial`) that is never graded or measured as the whole model.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from merlin.common.compile_trace import StopAfterStage

from .whole_model_build import ON_VENDOR, WholeModelBuildError, build, grade
from .whole_model_partial import MARKER


def main(argv: Sequence[str] | None = None) -> int:
    from merlin.compile import debug

    parser = argparse.ArgumentParser(
        prog="merlin-whole-model-build",
        description="Build a model capsule as one runnable program from a compiler package's own kernels, "
        "with its per-group attribution and an independent oracle; or grade a run's UART against one.",
    )
    parser.add_argument(
        "--list-stages", action="store_true", help="list every stage --stop-after/--dump-ir-* can name, and exit"
    )
    sub = parser.add_subparsers(dest="command")
    make = sub.add_parser("build", help="build the whole-model ELF, attribution and oracle")
    make.add_argument(
        "--package",
        type=Path,
        help="the compiler package directory; omit it for the target's library program (every group vendor)",
    )
    make.add_argument("--capsule", required=True, type=Path, help="the model capsule directory")
    make.add_argument("--target", required=True, help="the target the package lowers for")
    make.add_argument("--machine", required=True, help="the hardware-registry entry the program is built for")
    make.add_argument("--header", required=True, type=Path, help="that machine's vendor parameter header")
    make.add_argument("--header-sha256", help="the header's digest, when the registry declares none for the machine")
    make.add_argument("--out", type=Path, help="the product directory (default: under out/artifacts/perf-bench)")
    make.add_argument("--no-oracle", action="store_true", help="skip the reference recomputation")
    make.add_argument(
        "--verify",
        choices=("on_target", "host_dump", "local", "words"),
        default="on_target",
        help="check each group's output on the core (digests; 'local' also recomputes each exact group's "
        "reference on the core from its actual inputs), or leave it to a host-side reader of a memory dump",
    )
    make.add_argument("--timeout", type=int, default=600, help="seconds per package entrypoint call")
    make.add_argument("--jobs", type=int, help="parallel package calls and object builds")
    make.add_argument(
        "--decline",
        action="append",
        default=[],
        metavar="OP_OR_GROUP",
        help="route every group of this op (e.g. residual_add), or one group by index (e.g. g33), to the "
        "target's library even where the package answered it (repeatable); each is recorded as a "
        "caller_declined group",
    )
    make.add_argument(
        "--harness-override",
        action="append",
        default=[],
        type=Path,
        metavar="FILE",
        help="replace the harness file of this name in the build's copy of the harness tree (repeatable); "
        "each is recorded by digest",
    )
    make.add_argument(
        "--allow-passes",
        action="store_true",
        help="run the package's own whole_model_passes (its manifest) over the capsule's interface before "
        "splitting it into groups, using the transformed module only if it verifies as computing the same "
        "function; off by default",
    )
    make.add_argument(
        "--phase0-recipe",
        type=Path,
        help="the Phase 0 recipe whose datapath block the corpus is built under; required with a package",
    )
    make.add_argument("--descriptor", type=Path, help="the target descriptor the experiment loads")
    make.add_argument(
        "--prohibited-role",
        action="append",
        default=[],
        metavar="ROLE",
        help="an instruction role the linked program must not contain (repeatable)",
    )
    make.add_argument(
        "--allow-regions",
        action="store_true",
        help="offer the package a legal run of consecutive groups as one kernel, in addition to each group "
        "alone; a package that never asks for a region sees no change; off by default",
    )
    make.add_argument(
        "--only-group",
        action="append",
        default=[],
        metavar="gN[,gM]",
        help="ask the package for these groups only; every other group keeps the target's library call. "
        "The result is marked a PARTIAL build and is never graded, gated or measured as the whole model",
    )
    debug.add_arguments(make, list_stages=False)
    check = sub.add_parser("grade", help="grade a run's UART against a build's oracle")
    check.add_argument("--uart", required=True, type=Path)
    check.add_argument("--oracle", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.list_stages:
        return debug.list_stages(args)
    if args.command is None:
        parser.error("name a command: build or grade (or pass --list-stages)")
    if args.command == "grade":
        verdict = grade(args.uart.read_text(encoding="utf-8", errors="replace"), json.loads(args.oracle.read_text()))
        print(json.dumps(verdict, indent=1))
        return 0 if verdict["quotable"] else 1
    trace = debug.request_from_args(make, args, target=args.target, workload=Path(args.capsule).name)
    try:
        with debug.opened(trace, ["merlin-whole-model-build", *(sys.argv[1:] if argv is None else argv)]):
            record = build(
                args.package,
                args.capsule,
                target=args.target,
                machine=args.machine,
                header=args.header,
                header_sha256=args.header_sha256,
                out=args.out,
                oracle=not args.no_oracle,
                verify=args.verify,
                timeout=args.timeout,
                jobs=args.jobs,
                decline=args.decline,
                harness_overrides=args.harness_override,
                allow_passes=args.allow_passes,
                allow_regions=args.allow_regions,
                prohibited_roles=args.prohibited_role,
                phase0_recipe=args.phase0_recipe,
                descriptor=args.descriptor,
                only_groups=args.only_group,
            )
    except StopAfterStage as stop:  # a requested stop: the IR is written, no program; not an error
        return debug.stopped("merlin-whole-model-build", stop)
    except WholeModelBuildError as refusal:
        print(f"not built: {refusal}", file=sys.stderr)
        return 2
    if debug.not_reached(trace):
        print(f"not stopped: {debug.not_reached(trace)}", file=sys.stderr)
        return 1
    counts = record["attribution"]["counts"]
    print(f"built {record['elf']} ({record['elf_sha256'][:12]})")
    print(f"groups: {counts}")
    for row in record["attribution"]["per_group"]:
        if row["on"] == ON_VENDOR:
            print(f"  g{row['group']} ({row.get('op')}) -> vendor [{row.get('cause')}]: {str(row.get('why'))[:120]}")
    if record["oracle"]:
        print(f"oracle: argmax={record['oracle']['argmax']} golden={record['oracle']['golden_argmax']}")
    if record.get(MARKER):
        print(f"PARTIAL BUILD (only {', '.join(record[MARKER]['only_groups'])}): not the whole model; never graded")
    if trace is not None:
        print(f"trace: {Path(trace.directory).absolute() / 'trace.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
