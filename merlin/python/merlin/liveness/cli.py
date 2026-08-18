"""``merlin-liveness`` CLI — screen a program for silicon liveness/precondition hazards.

Answers "would this stall or fault on real silicon?" from the target's OWN derived facts (never a
per-target literal), failing closed on what cannot be derived. Consumes a decoded RoCC
``instruction_trace`` (from ``merlin-targetgen`` / ``rocc_decode``) and/or image-audit flags.

Usage:
  merlin-liveness --target gemmini --trace path/to/instruction_trace.json --address-model pointer_args
  merlin-liveness --target <t> --trace <t.json> --dram-bytes 0x40000000 --persist
  merlin-liveness --target <t> --trace <t.json> --hostless --has-htif   # host-assist precondition

Exit code: 2 if any FAULT finding (unrunnable on silicon); 1 if --strict and any STALL; else 0.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .oracle import Program, assess, persist
from .report import Severity


def _int(s: str | None) -> int | None:
    if s is None:
        return None
    return int(s, 0)  # accepts 0x… and decimal


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="merlin-liveness",
        description="HW-agnostic silicon liveness/precondition screening (would it stall/fault on silicon?).")
    ap.add_argument("--target", required=True, help="target name (facts are derived from its RTL/manifest)")
    ap.add_argument("--trace", type=Path, help="decoded RoCC instruction_trace.json")
    ap.add_argument("--name", default=None, help="program label (default: derived from --trace path)")
    ap.add_argument("--address-model", default=None,
                    help="harness DRAM addressing convention (e.g. pointer_args)")
    ap.add_argument("--dram-bytes", default=None,
                    help="DRAM window size (int, accepts 0x…); enables the upper-bound address check")
    ap.add_argument("--hostless", action="store_true",
                    help="the delivery substrate has no fesvr host (enables the HTIF-hang precondition)")
    ap.add_argument("--has-htif", dest="has_htif", action="store_true",
                    help="image audit: a .htif section is present")
    ap.add_argument("--has-tohost", dest="has_tohost", action="store_true",
                    help="image audit: a tohost symbol is present")
    ap.add_argument("--declared-vlen", default=None,
                    help="VLEN the build declared (VECTOR_MAX_LEN / -march …_zvlNNNb)")
    ap.add_argument("--hw-vlen", default=None,
                    help="the target board's hardware VLEN in bits (enables the vlen-match check)")
    ap.add_argument("--uses-medany", action="store_true",
                    help="the chosen compile path builds with -mcmodel=medany")
    ap.add_argument("--image-span-bytes", default=None,
                    help="linked image symbol span in bytes (enables the medany ±2GB-window check)")
    ap.add_argument("--strict", action="store_true", help="exit nonzero on STALL as well as FAULT")
    ap.add_argument("--persist", action="store_true",
                    help="write the report under out/artifacts/liveness/<target>/")
    ap.add_argument("--json", action="store_true", help="emit the full report as JSON")
    args = ap.parse_args(argv)

    trace = None
    if args.trace:
        trace = json.loads(Path(args.trace).read_text(encoding="utf-8"))
    name = args.name or (Path(args.trace).parent.name if args.trace else "program")

    # host-assist audit is only meaningful when the substrate is stated.
    hostless = True if args.hostless else None
    has_htif = True if args.has_htif else (None if hostless is None else False)
    has_tohost = True if args.has_tohost else (None if hostless is None else False)

    prog = Program(
        name=name, trace=trace,
        address_model=args.address_model, dram_bytes=_int(args.dram_bytes),
        hostless=hostless, has_htif=has_htif, has_tohost=has_tohost,
        declared_vlen=_int(args.declared_vlen), hw_vlen=_int(args.hw_vlen),
        uses_medany=(True if args.uses_medany else None),
        image_span_bytes=_int(args.image_span_bytes))
    report = assess(prog, args.target)

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(f"merlin-liveness: {name} vs {args.target} → verdict={report.verdict.upper()}")
        if report.resource_peaks:
            print(f"  peaks: {report.resource_peaks}")
        for f in report.findings:
            where = f" @ {f.where}" if f.where else ""
            print(f"  [{f.severity.value:7s}] {f.rule}{where}: {f.message}")
        if not report.findings:
            print("  (no findings)")

    if args.persist:
        out = persist(report)
        print(f"  wrote {out}")

    if any(f.severity == Severity.FAULT for f in report.findings):
        return 2
    if args.strict and any(f.severity == Severity.STALL for f in report.findings):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
