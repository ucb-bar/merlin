# AGENT.md — merlin/targets/k1_cpu

## Purpose

Curated CPU-host target definition for the eight-hart SpacemiT K1/X60 board.  This package gives
TargetGen a CPU dialect plan that can be composed with an accelerator target while the Linux/K1
runner supplies the silicon performance authority.

## Invariants

- Keep observed silicon facts separate from compiler policy.
- The dialect describes CPU-side scalar, RVV, memory, and parallel work; it must not contain model
  names or paper-network shapes.
- A fixed VLEN optimization is legal only after the runtime probe reports the same VLEN.
- Generated dialect/package output belongs under `out/`, never in this directory.

## Testing expectations

Validate both contracts against the schemas, compile `examples/probe.c` with the configured K1
toolchain, and run it on the board before promoting a compiler freeze.
