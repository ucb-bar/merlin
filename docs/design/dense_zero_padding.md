---
title: Private dense zero padding
kind: design
status: current
owner: core
last_verified: 2026-10-19
code_refs: [src/merlin/llvmlower/dense_padding.py, merlin/tests/runtime/test_dense_padding.py]
---

# Private dense zero padding

`DenseZeroPadding` separates a logical dense shape from a consumer's physical
shape. Dimensions are static and positive, the logical tensor fits inside the
physical tensor, and implicit axes must be distinct valid dimensions. Default
padding materializes the complete physical tensor with scalar zero assignment
and a row-major copy of the logical tensor.

A provider may explicitly declare axes on which its consumer performs exactly
the required positive-zero extension. Those axes retain their logical extent
in the materialized input. For example, a logical `(2, 3)` tensor with physical
shape `(4, 5)` and implicit axis `0` needs only a `(2, 5)` staging tensor. The
consumer then receives the logical row count and physical row stride separately.
When all padding is implicit, the original tensor can be consumed directly and
the plan reports no copy scratch.

`emit_dense_zero_padding` generates the remaining scalar zero/copy loops into
caller-owned disjoint storage. It neither allocates buffers nor infers a
consumer capability from its name or target. Input/output accessibility,
element type, actual strides, nonaliasing and complete storage lifetime remain
caller obligations. Distinct generated C names do not prove pointer disjointness.

The provider must independently qualify its implicit-zero representation,
indexing, exceptional-value behavior and completion. Packed formats may share
bits between adjacent logical rows; writing a padded row must preserve live
neighboring bits. A plan supplies no device residency, scheduling, arithmetic
approximation or profitability permission. Explicit selection and measurement
remain necessary.
