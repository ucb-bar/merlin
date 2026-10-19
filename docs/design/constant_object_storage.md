---
title: Immutable constant object storage
kind: design
status: current
owner: core
last_verified: 2026-10-18
related: [architecture, private_tensor_transport]
code_refs: [src/merlin/llvmlower/constant_object.py, merlin/tests/ir/test_constant_object.py]
---

# Immutable constant object storage

Large immutable operands can occupy several copies of the same physical disk
storage: raw packed files, a readonly data object and a linked catalog object.
`reuse_constant_object` lets a caller share a prior data object's inode after
checking its complete current contents against the newly derived operands.
It creates no global cache and never rewrites the original object.

The provider supplies a fresh small reference object assembled with the selected
target options. Merlin reads public ELF serialization and requires identical
object identity and non-data section metadata. Target ABI and instruction facts
remain provider-owned. Nonempty executable or other allocated sections and all
relocations refuse. The selected constant section must be readonly and complete.

Each `ConstantBlob` declares its exported symbol, regular input file and alignment
in assembly order. Merlin checks every operand byte, required zero padding, total
section extent and exact symbol membership and offsets. Undefined, extra and
non-data symbols refuse. After hardlinking to a fresh destination it reopens the
complete object digest, inode and operand digests. Input mutation or a different
target reference prevents a successful receipt. Callers retain the existing
immutable-file lifetime requirement after the build.

This establishes only identical stored constants and their symbol layout. The
caller must still derive packing from source operands and qualify its numerical
contract, compile current kernels, link a fresh catalog, admit the final image
and measure actual execution. An object reuse receipt is not a new compiler
invocation, inference speedup or proof that runtime weight reads are shared.
