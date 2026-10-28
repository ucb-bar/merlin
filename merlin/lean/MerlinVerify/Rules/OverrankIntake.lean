/- Representation alignment for an invalid source that cannot be parsed by upstream MLIR.
No boolean-repair operation or shared exporter/semantics is changed. -/
import MerlinVerify.Sem.Support
namespace MerlinVerify.Rules.OverrankIntake
open MerlinVerify
/-- xDSL renders the one-bit all-ones integer literal as -1; MLIR's exporter uses 1.
Both decode to exactly the same initialized one-bit scalar. This is a spelling issue,
not a boolean conversion repair or a new numerical law. -/
theorem i1_literal (indexBits : Nat) :
    scalarOfAttr indexBits (.int (-1) (.int 1)) = scalarOfAttr indexBits (.int 1 (.int 1)) := by
  cbv
end MerlinVerify.Rules.OverrankIntake
