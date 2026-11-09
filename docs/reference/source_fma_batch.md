---
title: Independent source FMA batch permission
kind: reference
status: draft
owner: compiler
last_verified: 2026-10-06
related: []
code_refs:
  - src/merlin/llvmlower/source_fma_batch.py
  - src/merlin/runtime/host_arithmetic.py
  - src/merlin/runtime/native/c/prepared_polynomial_batch.h
---

# Independent source FMA batch permission

An explicit `SourceFmaBatchContract` permits a separately supplied implementation
to schedule eight independent finite binary32 source FMA chains together. Every
lane retains its operand order, addend and original single rounding. This does
not permit reassociating the Horner recurrence or changing its enclosure.

The caller supplies proofs of finite operands/results, distinct private operand
and product storage, stable rounding, gradual underflow, nontrapping arithmetic,
and unobserved exception flags and errno. Missing obligations refuse emission.
`emit_source_fma_batch_permission` selects an independently supplied mathematical
hook. CPU instruction selection and register constraints belong to an explicitly
selected host implementation. Merlin shares those implementations independently
of accelerator backends; target wrappers retain only ABI/symbol bindings.

The current consumer is the explicit four-cell polynomial alternative. Its
fraction and polynomial arrays are distinct private locals; the admitted source
plan bounds every Horner prefix. Each original lane reduction still happens in
source order after the independent endpoint chains. Unsupported preparation uses
the existing checked scalar path. Without the new permission, emitted arithmetic
and compiled default provider bytes remain unchanged.

This is an experimental scheduling option. Numerical qualification and functional
instruction counts do not establish hardware latency or whole-model gains.
