# Exact captured source regions

`merlin.llvmlower.source_region.extract_source_region` extracts a contiguous
current SSA span. Free source values become arguments in first-use order and
every externally observed member result becomes an output in source order.
The original graph and SSA objects remain available for an independent operand
binder and numerical reference. Optional literal and static view dependencies
copy the exact original operations; they do not infer values or dtype support.
The view domain is the existing `tensor_views` pass's closed immutable domain.

`outline_sibling_contractions` applies the existing exact signed zero-seeded
contraction reader to independent roots with the same actual LHS SSA. It copies
those original roots and zero initializers into a source callee. Future operand
preparations may move only when the existing immutable tensor purity checker
admits them and every crossed nonmember operation. Original epilogues, result
types, quantization order and all observers remain in their source owner.
Unknown effects, a changed initializer, different LHS values or dependent roots
refuse. Source order remains fixed among the selected contractions.

These APIs establish source correspondence only. The outlined host still defines
its original integer callee: it is not yet a device executable. Selecting a
candidate implementation additionally needs the original borrowed-buffer ABI,
complete writes, source and target command verification, completion/lifetime
premises, ordinary object/link, independent numeric execution, and the original
source custody. No receipt grants those premises or a whole-model closure claim.

A captured region's operands should be resolved from the original prefix and
only its required manifest arguments. Large weight blobs can remain read-only
and pinned in place while selected tensors are memory mapped. An independent
reference executes the original source SSA with its original quantization;
matching a reconstructed component graph is insufficient source custody.

## Source-owned sibling host compilation

`prepare_sibling_source_host` applies the existing immutable view CSE and
quant_ext lowering, then the exact sibling outline. It replaces only the
selected raw roots with a plural fresh writer. The original quantizers, layout
operations, mixed result types and every live observation stay in the owner.
The caller supplies separately selected borrowed effects; no source shape or
successful build grants those effects.

The prospective `merlin.owned_sibling_source.v1` capsule route reopens its two
local hash-bound source/plan documents after ordinary cohort relocation. Fresh
independent source eligibility must equal the frozen accelerator carrier census
and contain exactly the proved signed sibling roots. The author receives only
the rederived raw-core interface through the ordinary package entrypoints. The
actual closed emitted helper and completion are mechanically retained; the
original host is compiled separately through normal MLIR lowering and linked
around them. No candidate host wrapper is linked.

The public pointer ABI remains the complete original input/result ABI. The
private accelerator command buffer is retained separately. Its MAC count is a
lower bound on whole-call work; uncounted original host work stays unknown.
Canonical inputs and every mixed output come from the original source oracle,
not from the candidate command interpreter. The ordinary complete-output native,
trace, stack/memory and required-tier gates remain mandatory. This region route
provides no whole-model gate credit or full capture/source-closure assertion.

The initial host transport requires an independently observed 64-bit ranked
software ABI and the existing bounded bare-metal allocator/MLIR runtime. A
source-owned aligned BSS workspace covers checked constant allocation demand
plus the allocator's header/alignment overhead; unknown or loop-dependent
allocation demand refuses. The original runtime supplies roundeven semantics.
This software configuration is not a new accelerator or asynchronous law.
