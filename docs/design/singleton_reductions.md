---
title: Parallel scalar execution of singleton reductions
kind: design
status: current
owner: core
last_verified: 2026-10-18
---

# Parallel scalar execution of singleton reductions

`merlin.xdsl_dialects.lowering.singleton_reductions` exposes statically
single-iteration tensor reductions as parallel scalar arithmetic before
placement. This prevents an outer product expressed as a contraction from
unnecessarily requiring a matrix provider's numerical approximation.

The transformation returns a fresh verified module. It requires a pure tensor
`linalg.generic`, projected dimension maps, consistent positive static extents,
unit extent for every reduction iterator, and one output whose dimensions cover
every parallel iterator exactly once. Dynamic extents, effectful bodies,
external captures, symbols, compound maps, and `linalg.index` observers refuse.
Removed iteration dimensions become constant-zero indices in the input maps;
remaining dimensions are renumbered without changing output iteration order.

The entire scalar body, accumulator initializer, input and result types, flags,
and SSA connections survive. In particular, a multiply followed by addition
to zero remains two arithmetic operations with the original rounding rules.
The pass does not replace it with a bare multiply or assume finite operands.

An optional selector chooses among proven candidates and cannot override
legality. The pass grants no target capability, approximate numerical policy,
profitability verdict, or authority to drop source operations. Callers must
account explicitly for transformed source obligations and subsequent placement.
The target adapter owns numerical policy selection and accelerator avoidance;
the singleton proof and rewrite are target independent.

Regression checks are in `merlin/tests/ir/test_singleton_reductions.py` and
cover arbitrary initializers, nonunit and dynamic reductions, reduction-axis
renumbering, index observers, inconsistent maps, explicit selection, and
immutability of the original module. Provider integration must additionally
execute representative transformed programs and check complete outputs.
