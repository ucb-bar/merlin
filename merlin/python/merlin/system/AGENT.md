# AGENT.md — merlin/python/merlin/system

## Purpose

The configuration we compile FOR: one host, N devices, and the **link** joining each device to the
host. Everywhere else a "target" is one flat name that means the device; this package is where the
host exists as a first-class thing and where "how do I reach this device" has one answer instead of
five.

## What belongs here

- `model.py` — `System` / `Host` / `Device` / `Link` and their closed vocabularies
  (`COMMAND_TRANSPORTS`, `OPERAND_PLACEMENTS`, `ADDRESS_TRANSLATIONS`).
- `derive.py` — building one from facts the repo ALREADY derives (board descriptor, capability
  manifest, RTL facts). Nothing here discovers a new fact.

## The two rules

- **Derived or `None`.** Never default a fact into existence. A default is indistinguishable from a
  measurement at the call site and silently yields a wrong address or a wrong transport. Every
  underived axis is `None` with a note in `evidence` recording what was consulted, and
  `Link.unknowns()` / `System.unknowns()` let a caller fail closed.
- **No target names.** Keyed on derived properties (endpoint kind, decoder facts, declared
  interfaces) — never on which target it happens to be. Adding a target means adding no code here.

## Why `Link` has four axes

`endpoint_kind` answers four independent questions with one token — what artifact to emit, how a
command reaches the device, where operands live and who moves them, and which oracle grades. Two
targets can share the token and still differ: measured here, two `external_backend` devices derive
different operand placement and different address grounding. The axes separate so a hybrid device (a
mesh reached by `.insn` whose operands arrive by DMA) is describable at all.

## What does not belong here

- Placement/cost decisions — those consume a `System`, they do not live in it.
- Codegen or lowering. This package answers *what the machine is*, never *what to emit*.
- Anything read from an experiment run. Facts are generated during experiments and gitignored;
  resolve them through the existing derivers, and treat absence as `None`.
