---
title: Exact integer radix reconstruction before binary64 conversion
kind: design
status: current
owner: compiler
last_verified: 2026-10-06
related: []
code_refs:
  - src/merlin/llvmlower/radix_product_groups.py
  - src/merlin/llvmlower/radix_integer_reconstruct.py
  - merlin/tests/ir/test_radix_integer_reconstruct.py
---

# Exact integer radix reconstruction

An explicitly selected integer group consumer can replace repeated binary64
integer conversion, power-of-two multiplication and addition with signed-i64
weighted updates, followed by one binary64 conversion. The existing canonical
radix product plan proves every group fits signed i32 and the sum of absolute
weighted terms is at most 2 to the power 53. Every weighted product and prefix
therefore fits i64 and is an exactly representable binary64 integer.

The original consumer must use RNE and a positive-zero seed. Exact cancellation
then produces positive zero, matching conversion of the final integer zero.
External scales, source floating operations, certificates and fallback replay
are unchanged and require their existing independent contracts.

The helper rederives the canonical plan and refuses modified groups, weights or
range proofs. It uses defined signed multiplication by a positive constant;
shifting a negative signed value is never used. Signed-i64 addition cannot
overflow under the rederived prefix bound. A separate owned integer scratch
buffer avoids pointer type punning and must remain disjoint from the complete
immutable i32 source and binary64 destination. Its explicit restrict qualifiers
depend on this caller-proved nonoverlap and lifetime obligation.

Each group case keeps its proven constant inside the update loop. This gives
ordinary CPU code generation an opportunity to simplify power-of-two
multiplication without a runtime weight lookup. Actual emitted instructions and
complete costs still require inspection and measurement for the selected CPU.

The caller resets scratch for every independent reconstruction, performs the
original complete group sequence and converts only after the final update.
Scratch may be reused after conversion when no remaining consumer needs it.
Every allocated element is initialized and the final conversion writes every
logical output. A plan cap may cover several source reduction extents only when
each actual extent is independently bound and checked against that cap.

An explicit alternative initializes fresh scratch directly from the fully
written first group. The canonical group's weight is one, so this exactly
replaces the positive-zero seed and first weighted update. Only subsequent
groups are accumulated afterward. It removes the zero-fill pass and first
prefix read while retaining the same disjoint storage, full-writer and lifetime
obligations. It grants no permission to omit or replay the first producer.

The additional scratch allocation, initialization, integer traffic, final
conversion and increased live memory must be measured together with producer
and device execution. A numeric proof alone gives no performance guarantee.
This shared helper selects no device, workload or automatic production route;
device schedules and resource legality remain in the selected OOT backend.

Tests compile the actual C and check complete raw binary64 prefixes, exact zero
cancellation, bounded tails, positive and negative near-limit sums, dirty guard
words and floating flags. UBSan independently checks that the emitted negative
products and additions use defined C arithmetic.
