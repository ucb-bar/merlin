---
title: "Prepared word-space softmax interval domain"
kind: reference
status: current
owner: core
last_verified: 2026-11-09
related: [ordered_fma_certificates, quantized_host_optimizations]
code_refs:
  - src/merlin/llvmlower/source_attention_frontier.py
  - src/merlin/runtime/native/c/prepared_softmax_interval.h
  - src/merlin/runtime/native/c/monotone_bit_polynomial.h
---

# Prepared word-space softmax interval domain

`word_interval_enclosure=True` may compose with `prepare_softmax_domain=True`
only through the separate `merlin_softmax_word_domain_prepare` admission.
The existing source-q domain and default emitted program are unchanged.

The source-q uniform probability cap encloses every original source point
word, including all rounded Horner operations and the final encoding FMA.
The word-space helper can add at most `word_budget` to that positive binary32
encoding. Admission adds this budget using uint64 arithmetic and applies the
same finite-positive clamp as the word helper. It then executes the source
positive lane accumulation, binary tree, and chunk FMA schedule on the cap,
using alpha=1. Every prefix must remain finite. Overflow, unsupported plans,
unknown environments, or unadmitted active spans retain checked execution.

This proof uses immutable admitted endpoints, a stable RNE environment,
nontrapping arithmetic and unobserved exception flags. All-masked tiles retain
zero lanes; cutoff crossings can include zero. No signed PV operation is
admitted by this positive softmax domain. Runtime source operations are not
reassociated. Wider intervals may increase exact replay, so numerical validity
does not establish profitability.

Independent tests cover three coefficient/domain plans, cutoff crossings,
signed zeros, subnormals, word-budget overflow, non-RNE admission, masks and
unknown inputs. Small complete executors exercise both basic and fully prepared
word/softmax/integer-reconstruction combinations, including refusal paths.

The emitter chooses the separate word-domain admission when both options are
selected. `prepared_softmax_interval.h` owns that admission and the positive
consumer operations; `monotone_bit_polynomial.h` owns the word enclosure.
Keeping these owners distinct prevents a word-budget bound from being treated
as permission for unrelated signed reductions.
