# AGENT.md — merlin/python/merlin/kernels/decode

## Purpose

Robust, non-regex decoders for the kernel-mining pipeline.

## Modules

- `clang_ast.py` — Typed C-intrinsic source extractor — the cross-check for the asm-lifted CCA.
- `memory.py` — Memory-traffic / packing facet over the decoded RVV ``InsnStream`` — the data-movement view the
- `objdump.py`, `rocc.py`, `insn_header.py` — compatibility aliases of
  `merlin.compile.analysis.{disassembly,rocc_commands,intrinsics_header}`, which own the
  tokenizer (`word_of`, unknown-mnemonic vocabulary), fact-bound RoCC decoding and intrinsics
  header tables. Generic RoCC and derived-ISA decoding must not import an accelerator-specific
  decoder for these.
- `rvv.py` — RVV (vector) semantic decoder — the first per-ISA instantiation over ``decode/objdump``.

<!-- Purpose/Modules derived from docstrings via build_tools/scripts/gen_package_docs.py.
     Add hand-written notes (invariants, gotchas) below. -->
