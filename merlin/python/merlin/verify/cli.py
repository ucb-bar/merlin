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
        print("VERIFIED — the command buffer computes what the interface program specified, "
              "for every input at this shape.")
        return EXIT_VERIFIED
    if verdict.status == "sat":
        print("REFUTED — these inputs make the buffer disagree with the program it was given:")
        for name, value in sorted((verdict.model_values or {}).items()):
            print(f"    {name} = {value}")
        if not verdict.model_values:
            print("    (no model recovered — report this, a refutation without a counterexample "
                  "is an assertion)")
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
        verdict = validate_compilation(module, cb, acc_width=args.acc_width,
                                       timeout_ms=args.timeout_ms)
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

    argv = ["--m", str(args.m), "--k", str(args.k), "--n", str(args.n),
            "--reuse", str(args.reuse), "--timeout-ms", str(args.timeout_ms)]
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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="merlin-verify", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

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

    l = sub.add_parser("lattice", help="verify a target's derived extent lattice")
    l.add_argument("--target", required=True)
    l.add_argument("--timeout-ms", type=int, default=300_000)
    l.add_argument("--json", action="store_true")
    l.add_argument("--write", action="store_true")
    l.set_defaults(fn=cmd_lattice)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
