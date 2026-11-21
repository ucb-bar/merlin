# AGENT.md — merlin/targets/k1_cpu/contracts

## Purpose

Curated K1 CPU target contract and CPU-host dialect plan.

## Invariants

- Only hardware/runtime facts and target-level compiler obligations belong here.
- Do not encode a paper model, layer name, checkpoint, or selected search result.
- Keep the scalar path available as a correctness oracle, not as an unreported performance fallback.
