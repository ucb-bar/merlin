/-
Build-time tests of the module-level check and the outside-claim accounting.
-/
import MerlinVerify.Check.Module
import MerlinVerify.Tests.C1

namespace MerlinVerify.Tests.Module

open MerlinVerify MerlinVerify.Check MerlinVerify.Tests.Semantics MerlinVerify.Tests.C1

def withFuncAttrs (m : Module) (extra : List (String × Attr)) : Module :=
  { m with ops := m.ops.map fun
      | .mk n r o a p g s => if n == "func.func" then .mk n r o (a ++ extra) p g s else .mk n r o a p g s }

-- Adding `llvm.emit_c_interface` (ABI-only, per the reviewed annotation spec): every function still
-- verifies, and the change is reported as ABI-only, not as unverified.
#guard moduleCheck src (withFuncAttrs src [("llvm.emit_c_interface", .unit)])
#guard (outsideChanges src (withFuncAttrs src [("llvm.emit_c_interface", .unit)])).abi ==
  ["func.func @f: ABI attributes"]
#guard (outsideChanges src (withFuncAttrs src [("llvm.emit_c_interface", .unit)])).unverified == []

-- A provenance attribute is semantics-free: no report at all.
#guard (outsideChanges src (withFuncAttrs src [("prov.fqn", .str "x")])).unverified == []
#guard (outsideChanges src (withFuncAttrs src [("prov.fqn", .str "x")])).abi == []

-- Any other attribute change on the function op is unverified.
#guard (outsideChanges src (withFuncAttrs src [("passthrough", .unit)])).unverified ==
  ["func.func @f: attributes"]

-- An added top-level op is unverified; duplicate keys make pairing ambiguous.
#guard (outsideChanges src { src with ops := src.ops ++ [decl "dev"] }).unverified == ["added func.func @dev"]
#guard (outsideChanges { src with ops := src.ops ++ src.ops } src).unverified ==
  ["ambiguous top-level ops (duplicate name/symbol keys)"]

-- A wrong body still fails the per-function check.
#guard !moduleCheck srcSub tgtSwap

end MerlinVerify.Tests.Module
