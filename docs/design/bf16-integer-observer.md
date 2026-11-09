---
title: Exact BF16 integer observations
kind: design
status: current
owner: compiler
last_verified: 2026-11-09
related: []
code_refs:
  - src/merlin/llvmlower/bf16_integer_observer.py
  - src/merlin/llvmlower/source_attention_frontier.py
  - merlin/runtime/c/bf16_quant_frontier.h
---

# Exact BF16 integer observations

`emit_source_attention_frontier(..., bf16_integer_observer=contract)` explicitly
selects a target independent integer observation of the owned BF16 quantizer.
The default emitter and runtime header remain unchanged. This choice introduces
no approximation, model selection or accelerator instruction.

## Proof

The source computes its original row extrema, BF16 scale, epsilon maximum,
reciprocal and `BF16(x * inverse)` product. The specialization retains those
operations and every original row validation, refusal and publication rule.
Only the product's subsequent nearest-even round, BF16 conversions, addition
of positive zero and integer clamp are replaced by integer decoding.

Every finite unsaturated BF16 product rounds to an integer in `[-128, 128]`.
Each of these integers is exactly representable in BF16, so the intervening
BF16 conversions leave its integer observation unchanged. Products with an
exponent at least seven saturate under any admitted clamp inside `[-128, 127]`.
Products below one half round to zero; one half ties to zero. Remaining products
use the eight-bit significand, discarded remainder, midpoint and integer parity
to implement nearest-even rounding exactly. Negative products apply the sign
after rounding their magnitude.

Positive and negative zero have the same integer observation, including clamps
that exclude zero. Subnormal products round to zero. The direct helper maps
infinities to their saturation limit and a NaN to the upper bound, matching the
original standard min/max expression. The row implementation still rejects
nonfinite source endpoints before publishing any output.

## Effects and composition

`BF16IntegerObserverContract` requires stable RNE execution, nontrapping
arithmetic, unobserved exception flags, an integer-only returned observation,
a pure quantizer with no library effects, standard fixed representation copies,
and unobserved copy interposition. A typed pure source `math.roundeven` operation
can supply the value-operation obligation. An arbitrary interposed `nearbyintf`
call cannot. Existing scale permissions and an approximate product policy do
not establish this separate source observation contract.

The helper copies the authoritative runtime header and refuses if its owned
quantizer seam changes. It replaces the first owned guarded include, preserving
later includes. Apply it after finite-point row specialization when both choices
are selected; the first header definition then supplies the integer observer
to every row user. It does not change other min/max, copy or rounding operations.

## Verification and admission

Native tests compare all 65,536 BF16 product words under eight clamp plans,
including intervals excluding zero, against the original quantizer. Eight
inverse values exercise the complete source product DAG. Mixed rows, half ties,
signed zeros, subnormals, saturation, nonfinite values, unsupported environments,
invalid plans, output guards and metadata are checked with undefined behavior
sanitization. Every contract permission is required explicitly.

Whole-model and complete target-group receipts belong to their qualification
owner. Unit tests grant no performance or production admission. Functional
instruction counts must not be reported as accelerator or FPGA cycle counts.

`bf16_integer_observer.c_header` specializes the owned runtime quantizer;
`prepare_bf16_integer_observer` inserts it into the selected emitted source.
The shared frontier emitter validates composition order and the explicit contract.
This boundary supplies portable host code, while target products and instruction
selection remain OOT responsibilities.
