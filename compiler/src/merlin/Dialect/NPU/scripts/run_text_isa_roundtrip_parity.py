#!/usr/bin/env python3
"""End-to-end text-ISA round-trip parity harness.

Goal: prove the bridge from a high-level "kernel sequence" description to a
numerically-correct simulator run via the text-ISA intermediate. For each
fixture in the kernel golden suite (Piece A):

  1. Stitch the kernel's instructions via ``stitch_kernels([kernel])``.
  2. Serialize to text ISA via ``instruction_to_text``.
  3. Parse the text back via ``npu_model.software.text_isa_loader``.
  4. Build a Program from the parsed instructions (plus the fixture inputs /
     golden) and run it through the simulator.
  5. Assert ``torch.allclose`` at the fixture's tolerance.

Exits 1 on any unexpected failure; xfails are reported but don't fail the
run. This is the minimal version of Piece E of the testing plan —
specifically, it proves step D.1 (the text-ISA parser) reconstructs the
same instruction stream the in-memory path uses. The compiler-side step
(iree-compile → npu-translate → parse) is pending Piece D.2, which needs
the native-kernel-lowering manifest to include the new variant names.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[6]


def main() -> int:
    root = _repo_root()
    sys.path.insert(0, str(root / "third_party" / "npu_model"))
    sys.path.insert(0, str(root))

    import npu_model  # noqa: F401
    import torch
    from npu_model.configs.hardware.default import DefaultHardwareConfig
    from npu_model.configs.programs._kernel_fixtures import (
        REGISTRY as FIXTURES,
    )
    from npu_model.configs.programs._stitch import (
        instruction_to_text,
        stitch_kernels,
    )
    from npu_model.logging import LoggerConfig
    from npu_model.simulation import Simulation
    from npu_model.software import Program
    from npu_model.software.text_isa_loader import parse_text_isa

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kernel",
        action="append",
        help="Run only the named kernel (may be repeated).",
    )
    parser.add_argument("--max-cycles", type=int, default=50_000)
    args = parser.parse_args()

    kernel_names = sorted(FIXTURES)
    if args.kernel:
        kernel_names = [k for k in kernel_names if k in args.kernel]
        if not kernel_names:
            print("ERROR: no matching fixtures", file=sys.stderr)
            return 2

    header = f"{'kernel':<22}  {'result':<6}  {'max_diff':>12}  {'cycles':>8}"
    print(header)
    print("-" * len(header))

    failures: list[str] = []
    xfails: list[str] = []
    passes = 0
    for kernel in kernel_names:
        fixture = FIXTURES[kernel]
        in_memory = stitch_kernels([kernel], base_address=0)
        text_blob = "\n".join(instruction_to_text(i) for i in in_memory)
        parsed = parse_text_isa(text_blob)

        # Build a throwaway Program subclass with the parsed instructions.
        prog_cls = type(
            f"TextIsaRoundtrip_{kernel}",
            (Program,),
            {
                "instructions": parsed,
                "memory_regions": list(fixture.inputs),
                "golden_result": (fixture.output_addr, fixture.expected),
            },
        )
        prog = prog_cls()
        sim = Simulation(
            hardware_config=DefaultHardwareConfig(),
            logger_config=LoggerConfig(filename="/tmp/text_isa_roundtrip.json"),
            program=prog,
        )
        with contextlib.redirect_stdout(io.StringIO()):
            sim.run(max_cycles=args.max_cycles)

        base, golden = prog.golden_result
        size = golden.numel() * golden.element_size()
        mem = sim.core.arch_state.read_dram(base, size).view(golden.dtype).reshape(golden.shape).clone()
        rtol, atol = fixture.tolerance
        ok = torch.allclose(mem.float(), golden.float(), rtol=rtol, atol=atol)
        diff = (mem.float() - golden.float()).abs().max().item()
        cycles = sim.cycle_count

        if fixture.xfail_reason:
            status = "XFAIL" if not ok else "XPASS"
            if ok:
                # XPASS from the round-trip path: same as suite's XPASS.
                failures.append(f"{kernel} (XPASS via roundtrip)")
            else:
                xfails.append(kernel)
        else:
            status = "PASS" if ok else "FAIL"
            if ok:
                passes += 1
            else:
                failures.append(kernel)
        print(f"{kernel:<22}  {status:<6}  {diff:>12.6f}  {cycles:>8d}")

    print("-" * len(header))
    if xfails:
        print(f"{len(xfails)} xfail(s): {', '.join(xfails)}")
    if failures:
        print(f"{len(failures)} unexpected failure(s): {', '.join(failures)}")
        return 1
    print(f"text-ISA round-trip parity: {passes}/{len(kernel_names)} PASS " f"({len(xfails)} xfail)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
