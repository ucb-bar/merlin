#!/usr/bin/env python3
"""Numerical parity harness for emitted NPU text ISA matmul paths.

Current scope:
  - Identity matmul parity check on MRF register output.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import torch

MERLIN_ROOT = Path(__file__).resolve().parents[6]
NPU_MODEL_ROOT = MERLIN_ROOT / "third_party" / "npu_model"

if str(NPU_MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(NPU_MODEL_ROOT))

from model_npu.configs.hardware.default import DefaultHardwareConfig  # noqa: E402
from model_npu.configs.isa_definition import *  # noqa: E402, F401, F403
from model_npu.logging import LoggerConfig  # noqa: E402
from model_npu.simulation import Simulation  # noqa: E402
from model_npu.software.instruction import Instruction  # noqa: E402
from model_npu.software.program import Program  # noqa: E402

LINE_RE = re.compile(r"^([a-zA-Z0-9_.]+)(?:\s+(.*))?$")
KV_RE = re.compile(r"([a-zA-Z0-9_]+)\s*=\s*([+-]?[0-9]+)")


class _Program(Program):
    instructions: list[Instruction] = []
    memory_regions: list[tuple[int, torch.Tensor]] = []


def parse_isa_lines(lines: list[str]) -> list[Instruction]:
    instructions: list[Instruction] = []
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = LINE_RE.match(line)
        if not m:
            raise ValueError(f"Invalid ISA line: {line}")
        mnemonic = m.group(1)
        rest = m.group(2) or ""
        args = {k: int(v) for k, v in KV_RE.findall(rest)}
        instructions.append(Instruction(mnemonic=mnemonic, args=args))
    return instructions


def build_identity_regions(m: int, k: int, n: int) -> list[tuple[int, torch.Tensor]]:
    lhs = torch.eye(m, k, dtype=torch.float8_e4m3fn)
    rhs = torch.eye(k, n, dtype=torch.float8_e4m3fn)
    return [(0, lhs), (2048, rhs)]


def compute_reference(m: int, k: int, n: int) -> torch.Tensor:
    lhs = torch.eye(m, k, dtype=torch.float8_e4m3fn)
    rhs = torch.eye(k, n, dtype=torch.float8_e4m3fn)
    return (lhs.to(torch.float16) @ rhs.to(torch.float16)).to(torch.bfloat16)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run NPU numerical parity check")
    parser.add_argument("isa_file", type=Path, help="Path to text ISA file")
    parser.add_argument("--m", type=int, default=64)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--result-reg", type=int, default=0)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--max-cycles", type=int, default=10000)
    args = parser.parse_args()

    instructions = parse_isa_lines(args.isa_file.read_text().splitlines())
    if not any(inst.mnemonic.startswith("matmul.") for inst in instructions):
        raise RuntimeError("No matmul instruction found in ISA; parity check not applicable")

    _Program.instructions = instructions
    _Program.memory_regions = build_identity_regions(args.m, args.k, args.n)

    sim = Simulation(
        hardware_config=DefaultHardwareConfig(),
        logger_config=LoggerConfig(filename=str(NPU_MODEL_ROOT / "compiler" / "parity_trace.json")),
        program=_Program(),
    )
    sim.run(max_cycles=args.max_cycles)

    actual = sim.core.arch_state.read_mrf_bf16(args.result_reg)[: args.m, : args.n]
    expected = compute_reference(args.m, args.k, args.n)

    max_abs = (actual.to(torch.float32) - expected.to(torch.float32)).abs().max().item()
    print(f"max_abs_error={max_abs:.6g} (atol={args.atol})")

    if max_abs > args.atol:
        print("Numerical parity FAILED")
        return 1

    print("Numerical parity PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
