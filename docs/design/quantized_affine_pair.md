---
title: "Design: exact quantized affine pair certificates"
kind: design
status: draft
owner: core
last_verified: 2026-10-08
related: [verified_compiler_roadmap]
code_refs:
  - src/merlin/llvmlower/quantized_affine_pair.py
  - merlin/tests/ir/test_quantized_affine_pair.py
  - merlin/tests/ir/test_quantized_affine_source_word.py
---

# Exact quantized affine pair certificates

`quantized_affine_pair` compares an explicitly supplied integer predictor with the
complete signed byte domain of an ordered source expression. Every one of the
65,536 operand pairs is checked. Source scales and the predictor scale are finite
positive binary32 values; signed integer coefficients must prove that every sum
fits signed i32. ReLU selection is explicit.

The source contract is separate binary32 left and right products, addition,
optional ReLU, multiplication by the binary32 output reciprocal, ties-even
rounding, and signed byte saturation. Its floating operations require RN-even
with gradual underflow. The predictor contract is the exact integer affine sum,
binary32 conversion, binary32 multiplication, ties-even rounding, and saturation.
The certificate binds both tables, their complete difference bitmap, and the
arithmetic contracts. Changed certificates are refused before C emission.

The optional C correction scans the original source inputs and changes only pairs
whose certified bitmap entry differs. Each changed result replays the original
ordered floating operations. A packed path derives a safe unary interval and
invariant bits of exceptional predictor outputs from the complete difference
table. Word predicates may flag additional lanes; an exact bitmap check filters
them before replay. Final scalar tails and unaligned inputs use byte accesses.

## Caller and provider obligations

The caller must preserve the original two input arrays until correction finishes.
Prediction output storage must not overlap either input. The emitted packed guard
requires little endian storage and includes a compile-time endian assertion.
Correction compilation and execution must preserve the stated floating contract.
These are ABI and numeric obligations, rather than alias or floating environment
facts inferred by this utility.

A target provider must independently close its implementation of the predictor
contract. A software table is insufficient evidence for device readout behavior.
Target instruction selection, schedules, resource proofs, and execution belong
in the provider. This module neither inserts a model rewrite nor selects a device
implementation by default.

## Performance admission

Full-domain exception counts describe correctness, not runtime frequency or cost.
The certificate leaves runtime ambiguity frequency unknown. Admission requires
measuring the complete prediction, transfers, scan, and replay against the exact
control, then preserving the normal whole-model accuracy and artifact gates.
Sparse numerical exceptions can still require an expensive scan of every input.
The first provider experiment rejected standalone CPU correction despite a much
faster predictor because the complete path was slower. Optimizers must retain
that distinction when considering correction fused with an already required
host operation or a different target implementation.

An explicit source-word guard narrows the scan without changing the certificate.
From the sparse pair predicate it derives the raw bytes that every correction
pair must contain on one operand axis, choosing the axis with fewer distinct
bytes as a code-size ranking. An aligned eight-lane source word that holds none
of them cannot contain a correction, so it skips the pair checks and reads neither
the other source nor the predicted output; a word with a possible hit checks its
lanes with the exact pair predicate. The guard is exact over the complete domain,
including adjacent-borrow cases of the any-byte test, but its hit rate on real
inputs is unknown, so it carries the same performance admission as the scan.
