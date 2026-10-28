/-
REVIEWED SPEC (design §4.1, §5.3). The only place an attribute may be declared irrelevant to an
op's meaning. Changing this file changes what every claim covers; it requires human review and a
`SPEC_LOCK` update.

* `semanticsFree k`: attribute `k` never affects the value an op computes (provenance, profiling
  and cache annotations). Ops may carry these without becoming unsupported, and boundaries may add,
  drop or change them.
* `abiOnly k`: attribute `k` affects only how the trusted downstream lowering spells a function's
  calling convention, not the function's meaning at the tensor level. A boundary that changes one is
  reported as an outside-claim change handled by the trusted lowering (design §10), never as verified.
-/
namespace MerlinVerify.Spec.Annotations

/-- Provenance (`prov.*`) and the structured-op indexing-map cache. -/
def semanticsFree (k : String) : Bool :=
  k.startsWith "prov." || k == "linalg.memoized_indexing_maps"

/-- Calling-convention attributes of `func.func` honored by MLIR's LLVM lowering. -/
def abiOnly (k : String) : Bool :=
  k == "llvm.emit_c_interface"

end MerlinVerify.Spec.Annotations
