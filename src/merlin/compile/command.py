"""The ``merlin-compile`` command line: its arguments, the combinations it refuses, and its report.

:func:`merlin.compile_cli.main` builds the parser here, checks the parsed arguments here, dispatches to
the front-door functions it defines itself, and prints the result here. The argument handling lives in
this module so that the front door keeps only what callers look up and tests replace (this package's
AGENT.md); nothing here compiles anything.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path


def parser(
    rvv_dtypes: Sequence[str], runs: Sequence[str], rvv_run: str, sustained: Sequence[str]
) -> argparse.ArgumentParser:
    """Every ``merlin-compile`` argument. The vocabularies are the dispatcher's (``compile_cli``): the whole-model
    lane's datatypes, where a result can run, the whole-model lane's default, and the sustained-mode runs."""
    ap = argparse.ArgumentParser(
        prog="merlin-compile",
        description="Compile saved models or OOT capsules; inspect model readiness without claiming execution.",
    )
    ap.add_argument(
        "--workload",
        required=False,
        help="rvv: a captured model name (bitvla, openvla, rdt2, …); "
        "an OOT target: a capsule name (A2_single_tile_matmul, …); "
        "not needed for --model-preflight (which selects an explicit bundle)",
    )
    # --target choices = rvv (whole-model) + every registered OOT target, auto-discovered via the
    # target registry (in-tree references + MERLIN_TARGET_PATH). Registering a dialect package makes
    # `--target=<name>` work with no code change here.
    try:
        from ..targetgen.target_registry import all_targets

        _oot_targets = sorted(all_targets())
    except Exception as exc:  # noqa: BLE001 — an unreadable registry offers no OOT target, and says so
        print(
            f"[merlin-compile] target registry unreadable ({type(exc).__name__}: {exc}); only "
            f"--target rvv is available",
            file=sys.stderr,
        )
        _oot_targets = []
    ap.add_argument(
        "--target",
        choices=["rvv", *_oot_targets],
        default="rvv",
        help="rvv (whole-model) or any registered OOT target (auto-discovered)",
    )
    ap.add_argument("--dtype", choices=list(rvv_dtypes), default="fp32", help="rvv only")
    ap.add_argument(
        "--harts",
        type=int,
        default=1,
        help="rvv+zephyr: harts to fan the model across (>1 builds the multicore "
        "OpenMP image; needs a matching SoC/sim)",
    )
    ap.add_argument(
        "--iters",
        type=int,
        default=1,
        help=f"rvv: timed inference iterations (sustained mode; {'/'.join(sustained)})",
    )
    ap.add_argument("--warmup", type=int, default=0, help="rvv: untimed warmup iterations before the timed ones")
    ap.add_argument(
        "--run",
        choices=list(runs),
        default=None,
        help=f"where to run after compiling (default: rvv→{rvv_run}, an OOT target→spike; 'none' = compile only)",
    )
    ap.add_argument(
        "--verify",
        dest="verify",
        action="store_true",
        default=True,
        help="gate the run output vs the golden (default on)",
    )
    ap.add_argument("--no-verify", dest="verify", action="store_false")
    ap.add_argument(
        "--no-capture",
        dest="capture",
        action="store_false",
        default=True,
        help="rvv: do NOT auto-capture a missing bundle (fail with the capture command instead)",
    )
    ap.add_argument("--package", default=None, help="override the codegen/OOT package dir")
    ap.add_argument(
        "--board",
        help="board name for RVV execution or an explicitly selected bare-metal --model-build",
    )
    ap.add_argument(
        "--corpus-descriptor",
        type=Path,
        help="explicit OOT corpus descriptor; prefer a released descriptor (this CLI does not verify its seal)",
    )
    ap.add_argument(
        "--model-preflight",
        action="store_true",
        help="read-only OOT model analysis: compare declared routes with groups in a captured program",
    )
    ap.add_argument("--model-build", action="store_true", help="build one saved capture as a bare-metal ELF")
    ap.add_argument("--capture-bundle", help="explicit saved capture for --model-preflight or --model-build")
    ap.add_argument("--board-catalog", type=Path, help="explicit board catalog for --model-build")
    ap.add_argument("--host-dts", type=Path, help="byte-pinned elaborated host DTS for --model-build")
    ap.add_argument("--output", type=Path, help="fresh generated output directory for --model-build")
    ap.add_argument("--arena-mb", type=int, help="explicit model arena size for --model-build")
    ap.add_argument("--reference-file", help="explicit in-capture .npy reference for model execution")
    ap.add_argument("--rtl-facts", type=Path, help="selected RTL facts for native --model-build execution")
    ap.add_argument(
        "--deployment-dtype",
        help="exact target operand format for --model-preflight (e.g. int8, bf16, fp8_e4m3)",
    )
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--json", action="store_true", help="emit the result dict as JSON")
    ap.add_argument(
        "--list-passes",
        action="store_true",
        help="list the optional lowering passes (stage, exactness, default, effect) and exit",
    )
    ap.add_argument(
        "--pass",
        dest="passes",
        action="append",
        default=[],
        metavar="NAME",
        help="enable an optional lowering pass by name (repeatable; see --list-passes)",
    )
    ap.add_argument(
        "--no-pass",
        dest="no_passes",
        action="append",
        default=[],
        metavar="NAME",
        help="disable an optional lowering pass that is on by default (repeatable)",
    )
    from .debug import add_arguments

    add_arguments(ap)
    return ap


def pass_selection(ap: argparse.ArgumentParser, a: argparse.Namespace):
    """The optional-pass selection ``--pass``/``--no-pass`` name (``ap.error`` on an invalid one)."""
    from ..llvmlower import optional_passes

    try:
        return optional_passes.Selection.of(a.passes, a.no_passes)
    except optional_passes.PassSelectionError as exc:
        ap.error(str(exc))


def list_passes(a: argparse.Namespace) -> int:
    """``--list-passes``: the registry as a table, or as JSON with ``--json``."""
    from ..llvmlower import optional_passes

    print(json.dumps(optional_passes.describe(), indent=2) if a.json else optional_passes.table())
    return 0


def validate(ap: argparse.ArgumentParser, a: argparse.Namespace) -> None:
    """Refuse the argument combinations no workflow accepts (``ap.error`` exits)."""
    if a.model_build and a.model_preflight:
        ap.error("--model-build and --model-preflight are separate workflows")
    if a.corpus_descriptor is not None and (a.target == "rvv" or a.model_preflight or a.model_build):
        ap.error("--corpus-descriptor applies only to OOT capsule compilation")
    if a.board is not None and a.target != "rvv" and not a.model_build:
        ap.error("--board applies only to RVV spike/Zephyr/Verilator execution")
    if a.capture_bundle is not None and not (a.model_preflight or a.model_build):
        ap.error("--capture-bundle requires --model-preflight or --model-build")
    build_only_inputs = (a.board_catalog, a.host_dts, a.output, a.arena_mb, a.reference_file, a.rtl_facts)
    if any(value is not None for value in build_only_inputs) and not a.model_build:
        ap.error("bare-metal build inputs require --model-build")
    if a.model_build:
        if not all((a.capture_bundle, a.package, a.board_catalog, a.board, a.host_dts, a.output, a.arena_mb)):
            ap.error(
                "--model-build requires --capture-bundle, --package (host), --board-catalog, "
                "--board, --host-dts, --output and --arena-mb"
            )
        if a.harts != 1 or a.iters != 1 or a.warmup != 0:
            ap.error("--model-build currently supports one hart and one inference")
        if a.run not in (None, "none", "spike", "gsim", "verilator"):
            ap.error("--model-build supports --run none, spike, gsim or verilator")
        if a.run not in (None, "none") and (not a.verify or not a.reference_file):
            ap.error("model execution requires --reference-file and complete-output verification")
        if a.run in ("gsim", "verilator") and not a.rtl_facts:
            ap.error("native model execution requires --rtl-facts")
    elif a.target == "rvv" and a.run == "gsim":
        ap.error("RVV gsim execution requires an explicit --model-build and matching board")

    if a.model_preflight and (a.target == "rvv" or not a.capture_bundle or not a.deployment_dtype):
        ap.error("--model-preflight requires an OOT --target, --capture-bundle, and --deployment-dtype")
    if not (a.model_preflight or a.model_build) and not a.workload:
        ap.error("--workload is required for compilation")
    if a.model_preflight or a.model_build:
        # The explicit bundle selects the model. An optional --workload supplied
        # out of habit must not become a second, possibly conflicting selector.
        a.workload = Path(a.capture_bundle).name


def report(a: argparse.Namespace, res: dict) -> int:
    """Print the result (``--json`` or a summary) and return the process exit status."""
    if a.json:
        print(json.dumps(res, indent=2, default=str))
    else:
        print(
            f"\n[merlin-compile] {a.target}:{a.workload}"
            f"{':' + a.dtype if a.target == 'rvv' else ''} → status={res.get('status')}"
            + (f"  gate_ok={res['verify'].get('gate_ok')}" if res.get("verify") else "")
            + (f"  reason={res.get('reason') or res.get('error')}" if res.get("reason") or res.get("error") else "")
        )
        for k in ("binary", "cycles", "vlen", "bundle", "package", "lowering_passes", "trace"):
            if res.get(k) is not None:
                print(f"    {k}: {res[k]}")
    return 0 if res.get("status") in ("compiled", "ran", "verified", "verified_complete_output") else 1
