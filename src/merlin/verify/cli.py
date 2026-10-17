"""``merlin-verify`` — the command line for the verification layers.

The ``compile`` subcommand is the one an experiment agent runs on its own output. It takes the
``interface`` program the backend was handed and the ``command_buffer.json`` it produced, and answers
whether the second computes what the first specified, **for every input at that shape**. It needs no
in-tree lowering and no simulator, so it works on a submission from a backend nobody has seen.

The verdict is advisory by design: it tells the agent what is wrong and hands back the concrete inputs
that expose it, without changing what counts as a pass. Three outcomes, and the middle one is the
reason this is safe to put in front of an agent:

* ``verified``  — no input at this shape distinguishes the two. Exit 0.
* ``refuted``   — here are inputs that do. Exit 1, and the counterexample is printed.
* ``abstained`` — this checker cannot model something the buffer uses. Exit 2, NOT a failure of the
  backend. An abstention reported as a defect would penalise correct work for our incompleteness.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

#: Exit codes, chosen so a shell caller can distinguish the three verdicts without parsing text.
EXIT_VERIFIED = 0
EXIT_REFUTED = 1
EXIT_ABSTAINED = 2


def _load_interface(path: Path):
    """Parse an ``interface``-plane module, with every in-tree dialect the pipeline can emit loaded.

    NOTE the surface this accepts: the IN-TREE ``interface.*`` dialect, which is what the pipeline
    prints and what ``merlin-opt`` reads. A capsule's own ``capsule.interface.mlir`` uses the
    ``merlin_iface`` grammar with CUSTOM assembly, which xDSL has no parser for — so a capsule file
    cannot be passed here today. That is a real limitation of this entry point, not of the checker:
    the validation itself only needs the two encodings, and the spec side could equally come from a
    capsule's declared ``operation`` block. Stated rather than discovered at the command line.
    """
    from xdsl.parser import Parser

    from merlin.xdsl_dialects import _common, contract, interface, runtime, schedule

    # Each dialect module exposes its Dialect INSTANCE as <NAME>_DIALECT; `Dialect` is the xDSL class
    # it imported, which is not what the context wants.
    mods = (contract, schedule, interface, runtime)
    dialects = [getattr(m, f"{m.DIALECT_NAME.upper()}_DIALECT") for m in mods]
    ctx = _common.make_context(*dialects)
    return Parser(ctx, path.read_text(encoding="utf-8"), str(path)).parse_module()


def _report(verdict, *, shape_note: str = "") -> int:
    print(f"verdict: {verdict.status}{shape_note}")
    if verdict.status == "unsat":
        print(
            "VERIFIED — the command buffer computes what the interface program specified, "
            "for every input at this shape."
        )
        return EXIT_VERIFIED
    if verdict.status == "sat":
        print("REFUTED — these inputs make the buffer disagree with the program it was given:")
        for name, value in sorted((verdict.model_values or {}).items()):
            print(f"    {name} = {value}")
        if not verdict.model_values:
            print("    (no model recovered — report this, a refutation without a counterexample is an assertion)")
        return EXIT_REFUTED
    print("ABSTAINED — no verdict within the budget. This says nothing about the backend.")
    return EXIT_ABSTAINED


def cmd_compile(args) -> int:
    """Validate an emitted command buffer against the interface program it came from."""
    from .refine import validate_compilation
    from .smt_semantics import UnsupportedSemantics

    cb: dict[str, Any] = json.loads(Path(args.command_buffer).read_text(encoding="utf-8"))
    module = _load_interface(Path(args.interface))
    try:
        verdict = validate_compilation(module, cb, acc_width=args.acc_width, timeout_ms=args.timeout_ms)
    except UnsupportedSemantics as exc:
        # Incompleteness in THIS checker, not a defect in the backend. Say so plainly, because the
        # difference decides whether an agent should change its code or ignore the message.
        print(f"verdict: abstained\nABSTAINED — {exc}")
        print("This is a limitation of the checker, not a defect in the command buffer.")
        return EXIT_ABSTAINED
    return _report(verdict)


def cmd_faults(args) -> int:
    """Run the seeded fault corpus past every layer (the detection matrix)."""
    from .evaluate import main as evaluate_main

    argv = [
        "--m",
        str(args.m),
        "--k",
        str(args.k),
        "--n",
        str(args.n),
        "--reuse",
        str(args.reuse),
        "--timeout-ms",
        str(args.timeout_ms),
    ]
    if args.json:
        argv.append("--json")
    if args.write:
        argv.append("--write")
    return evaluate_main(argv)


def cmd_lattice(args) -> int:
    """Verify a target's DERIVED extent lattice, the same one the capsule corpus is built from."""
    from .lattice import main as lattice_main

    argv = ["--target", args.target, "--timeout-ms", str(args.timeout_ms)]
    if args.json:
        argv.append("--json")
    if args.write:
        argv.append("--write")
    return lattice_main(argv)


def cmd_capture_coverage(args) -> int:
    """Inventory the source-side SMT subset without treating it as a proof."""
    from .model_coverage import audit_capture

    print(json.dumps([audit_capture(path) for path in args.captures], indent=2, sort_keys=True))
    return 0


def cmd_compile_receipt(args) -> int:
    """Recheck one exact interface→command-buffer pair and write a replayable receipt."""
    from .receipts import verify_transformation

    source = _load_interface(Path(args.interface))
    target = json.loads(Path(args.command_buffer).read_text(encoding="utf-8"))
    receipt = verify_transformation(
        "interface_to_command_buffer",
        source,
        target,
        translator=args.translator,
        expected_translator_sha256=args.translator_sha256,
        acc_width=args.acc_width,
        timeout_ms=args.timeout_ms,
    )
    rendered = json.dumps(receipt.to_dict(), indent=2, sort_keys=True) + "\n"
    if args.output:
        with Path(args.output).open("x", encoding="utf-8") as stream:
            stream.write(rendered)
    else:
        print(rendered, end="")
    return EXIT_VERIFIED if receipt.verified else EXIT_REFUTED if receipt.status == "refuted" else EXIT_ABSTAINED


def _emit_receipt(record: dict[str, Any], output: str | None) -> None:
    rendered = json.dumps(record, indent=2, sort_keys=True) + "\n"
    if output:
        with Path(output).open("x", encoding="utf-8") as stream:
            stream.write(rendered)
    else:
        print(rendered, end="")


def cmd_scalar_receipt(args) -> int:
    """Validate two exact MLIR files in the supported pure scalar-integer subset."""
    from .scalar_ir import verify_scalar_transform

    receipt = verify_scalar_transform(
        Path(args.before).read_bytes(),
        Path(args.after).read_bytes(),
        entry=args.entry,
        timeout_ms=args.timeout_ms,
    )
    _emit_receipt(receipt.to_dict(), args.output)
    return EXIT_VERIFIED if receipt.verified else EXIT_REFUTED if receipt.status == "refuted" else EXIT_ABSTAINED


def cmd_qualify_scalar_receipt(args) -> int:
    """Replay a saved scalar proof against the exact before/after file bytes."""
    from .scalar_ir import ScalarTransformReceipt, qualify_scalar_receipt

    try:
        record = json.loads(Path(args.receipt).read_text(encoding="utf-8"))
        receipt = ScalarTransformReceipt.from_dict(record)
        qualified = qualify_scalar_receipt(receipt, Path(args.before).read_bytes(), Path(args.after).read_bytes())
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"unqualified: invalid receipt or artifact: {exc}")
        return EXIT_ABSTAINED
    print("qualified" if qualified else "unqualified: proof replay or byte identity failed")
    return EXIT_VERIFIED if qualified else EXIT_ABSTAINED


def cmd_outline_receipt(args) -> int:
    """Check beta-equivalence of one actual pure outlining pass output."""
    from .outline_ir import verify_outline_transform

    receipt = verify_outline_transform(Path(args.before).read_bytes(), Path(args.after).read_bytes(), entry=args.entry)
    _emit_receipt(receipt.to_dict(), args.output)
    return EXIT_VERIFIED if receipt.verified else EXIT_REFUTED if receipt.status == "mismatch" else EXIT_ABSTAINED


def cmd_qualify_outline_receipt(args) -> int:
    """Replay a saved outlining receipt against the exact before/after bytes."""
    from .outline_ir import OutlineTransformReceipt, qualify_outline_receipt

    try:
        record = json.loads(Path(args.receipt).read_text(encoding="utf-8"))
        receipt = OutlineTransformReceipt.from_dict(record)
        qualified = qualify_outline_receipt(receipt, Path(args.before).read_bytes(), Path(args.after).read_bytes())
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"unqualified: invalid receipt or artifact: {exc}")
        return EXIT_ABSTAINED
    print("qualified" if qualified else "unqualified: proof replay or byte identity failed")
    return EXIT_VERIFIED if qualified else EXIT_ABSTAINED


def _chunk_sizes(text: str) -> tuple[int, ...]:
    try:
        return tuple(int(part) for part in text.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"chunks must be comma-separated integers, not {text!r}") from exc


def cmd_split_reduction(args) -> int:
    """Prove or refute one signed split-K accumulation at the declared widths, for every input."""
    from dataclasses import asdict

    from .split_reduction import maximum_safe_chunk, verify_split_reduction

    try:
        verdict = verify_split_reduction(
            k=args.k,
            operand_width=args.operand_width,
            partial_width=args.partial_width,
            output_width=args.output_width,
            chunks=args.chunks,
            timeout_ms=args.timeout_ms,
        )
        bound = maximum_safe_chunk(args.operand_width, args.partial_width)
    except ValueError as exc:
        print(json.dumps({"status": "invalid", "reason": str(exc)}, indent=2))
        return EXIT_ABSTAINED
    print(json.dumps({**asdict(verdict), "maximum_safe_chunk": bound}, indent=2))
    return {"verified": EXIT_VERIFIED, "refuted": EXIT_REFUTED}.get(verdict.status, EXIT_ABSTAINED)


def main(argv: list[str] | None = None) -> int:
    from .lean.cli import add_commands

    ap = argparse.ArgumentParser(prog="merlin-verify", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    add_commands(sub)

    c = sub.add_parser("compile", help="validate a command buffer against its interface program")
    c.add_argument("--interface", required=True, help="the .mlir the backend was handed")
    c.add_argument("--command-buffer", required=True, help="the command_buffer.json it produced")
    c.add_argument("--acc-width", type=int, default=32)
    c.add_argument("--timeout-ms", type=int, default=60_000)
    c.set_defaults(fn=cmd_compile)

    f = sub.add_parser("faults", help="run the seeded fault corpus past every layer")
    for name in ("m", "k", "n"):
        f.add_argument(f"--{name}", type=int, default=4)
    f.add_argument("--reuse", type=int, default=2)
    f.add_argument("--timeout-ms", type=int, default=60_000)
    f.add_argument("--json", action="store_true")
    f.add_argument("--write", action="store_true")
    f.set_defaults(fn=cmd_faults)

    lattice = sub.add_parser("lattice", help="verify a target's derived extent lattice")
    lattice.add_argument("--target", required=True)
    lattice.add_argument("--timeout-ms", type=int, default=300_000)
    lattice.add_argument("--json", action="store_true")
    lattice.add_argument("--write", action="store_true")
    lattice.set_defaults(fn=cmd_lattice)

    coverage = sub.add_parser("capture-coverage", help="inventory captured MLIR against the bounded SMT subset")
    coverage.add_argument("captures", nargs="+", type=Path)
    coverage.set_defaults(fn=cmd_capture_coverage)

    receipt = sub.add_parser(
        "compile-receipt", help="verify an exact interface/command-buffer pair with a pinned translator"
    )
    receipt.add_argument("--interface", required=True)
    receipt.add_argument("--command-buffer", required=True)
    receipt.add_argument("--translator", required=True)
    receipt.add_argument("--translator-sha256")
    receipt.add_argument("--acc-width", type=int, default=32)
    receipt.add_argument("--timeout-ms", type=int, default=60_000)
    receipt.add_argument("--output", help="write a new receipt file; omit for stdout")
    receipt.set_defaults(fn=cmd_compile_receipt)

    scalar = sub.add_parser("scalar-receipt", help="prove an exact pure scalar-integer MLIR transformation")
    scalar.add_argument("--before", required=True, help="exact before-pass MLIR file")
    scalar.add_argument("--after", required=True, help="exact after-pass MLIR file")
    scalar.add_argument("--entry", default="forward", help="entry function to compare")
    scalar.add_argument("--timeout-ms", type=int, default=30_000)
    scalar.add_argument("--output", help="write a new receipt file; omit for stdout")
    scalar.set_defaults(fn=cmd_scalar_receipt)

    qualify_scalar = sub.add_parser(
        "qualify-scalar-receipt", help="replay a saved scalar proof on exact before/after MLIR files"
    )
    qualify_scalar.add_argument("--receipt", required=True, help="saved scalar receipt JSON")
    qualify_scalar.add_argument("--before", required=True, help="exact before-pass MLIR file")
    qualify_scalar.add_argument("--after", required=True, help="exact after-pass MLIR file")
    qualify_scalar.set_defaults(fn=cmd_qualify_scalar_receipt)

    outline = sub.add_parser("outline-receipt", help="check exact pure outlining by call expansion")
    outline.add_argument("--before", required=True, help="exact before-pass MLIR file")
    outline.add_argument("--after", required=True, help="exact after-pass MLIR file")
    outline.add_argument("--entry", default="forward", help="entry function to compare")
    outline.add_argument("--output", help="write a new receipt file; omit for stdout")
    outline.set_defaults(fn=cmd_outline_receipt)

    qualify_outline = sub.add_parser("qualify-outline-receipt", help="replay a saved pure outlining receipt")
    qualify_outline.add_argument("--receipt", required=True, help="saved outline receipt JSON")
    qualify_outline.add_argument("--before", required=True, help="exact before-pass MLIR file")
    qualify_outline.add_argument("--after", required=True, help="exact after-pass MLIR file")
    qualify_outline.set_defaults(fn=cmd_qualify_outline_receipt)

    split = sub.add_parser(
        "split-reduction", help="prove a signed split-K accumulation at declared widths for every input"
    )
    split.add_argument("--k", type=int, required=True, help="reduction length of the source contraction")
    split.add_argument("--operand-width", type=int, required=True, help="signed operand bits")
    split.add_argument("--partial-width", type=int, required=True, help="signed bits of each chunk's accumulator")
    split.add_argument("--output-width", type=int, required=True, help="signed bits of the aggregated output")
    split.add_argument("--chunks", type=_chunk_sizes, required=True, help="chunk sizes covering K, e.g. 31,31,10")
    split.add_argument("--timeout-ms", type=int, default=30_000)
    split.set_defaults(fn=cmd_split_reduction)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
