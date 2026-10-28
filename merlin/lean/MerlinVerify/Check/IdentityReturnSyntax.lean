/- A bounded whole-block comparator. Only returned values may differ extensionally;
the candidate never replaces a tensor by an exact alias in an arbitrary continuation. -/
import MerlinVerify.Rules.IdentityReturn
import MerlinVerify.Check.RulesLinalg
import MerlinVerify.Check.Alpha
import MerlinVerify.Check.AlphaFast

namespace MerlinVerify.Check.IdentityReturnSyntax

open MerlinVerify MerlinVerify.Sem.Ops MerlinVerify.Sem.Ops.Linalg

private def valueDecEq (a b : Value) : Decidable (a = b) :=
  decidable_of_iff (a.id = b.id ∧ a.ty = b.ty) (by cases a; cases b; simp)
local instance : DecidableEq Value := valueDecEq

def DefaultTerm (name : String) (s : Op) : Prop :=
  s.name = name ∧ s.results = [] ∧ s.attrs = [] ∧ s.props = [] ∧
    s.regions = [] ∧ s.successors = []

instance (name : String) (s : Op) : Decidable (DefaultTerm name s) := by
  unfold DefaultTerm
  infer_instance

structure Generic where
  input : ValueId
  destination : ValueId
  result : Value
  shape : List Nat
  elt : Ty
  arg : Value
  old : Value
  term : Op

def Generic.region (g : Generic) : Region := .mk [.mk [g.arg, g.old] [g.term]]

def candidate (ib : Nat) (s : Op) : Option Generic := do
  let [input, destination] := s.operands | none
  let [result] := s.results | none
  let [.mk [.mk [arg, old] [term]]] := s.regions | none
  let some (.tensor shape elt) := vtyOf ib result.ty | none
  pure ⟨input, destination, result, shape, elt, arg, old, term⟩

theorem candidate_regions {ib : Nat} {s : Op} {g : Generic} (h : candidate ib s = some g) :
    s.regions = [g.region] := by
  unfold candidate at h
  split at h
  · split at h
    · split at h
      · next input destination ho result hr arg old term hregions =>
        split at h
        · cases h
          exact hregions
        · cases h
      · cases h
    · cases h
  · cases h

/-- Every semantic fact is rechecked against the actual header and the current source types.
Entry types are used only through the separate runtime EnvTyped premise of soundness. -/
def Legal (ib : Nat) (Γ : TEnv) (s : Op) (g : Generic) : Prop :=
  s.name = "linalg.generic" ∧ unmodelledAttr? s.blind = none ∧
  s.operands = [g.input, g.destination] ∧ s.results = [g.result] ∧
  s.successors = [] ∧
  attrDenseArray? s "operandSegmentSizes" = some [1, 1] ∧
  attrAffineMaps? s "indexing_maps" = some [identityMap g.shape.length, identityMap g.shape.length] ∧
  iterKinds? s = some (List.replicate g.shape.length .parallel) ∧
  vtyOf ib g.result.ty = some (.tensor g.shape g.elt) ∧
  Γ.lookup g.input = some (.tensor g.shape g.elt) ∧
  Γ.lookup g.destination = some (.tensor g.shape g.elt) ∧
  Γ.lookup g.result.id = none ∧
  vtyOf ib g.arg.ty = some (.scalar g.elt) ∧ vtyOf ib g.old.ty = some (.scalar g.elt) ∧
  g.arg.id ≠ g.old.id ∧ DefaultTerm "linalg.yield" g.term ∧ g.term.operands = [g.arg.id]

instance (ib : Nat) (Γ : TEnv) (s : Op) (g : Generic) : Decidable (Legal ib Γ s g) := by
  unfold Legal
  infer_instance

/-- The generic result may be returned in any position, with repeated or unchanged arguments.
All other source return operands must already be in scope and map through the actual rho. -/
def returnId (Γ : TEnv) (ρ : Rho) (result input : ValueId) (id : ValueId) : Option ValueId :=
  if id = result then ρ.find input else
    if (Γ.lookup id).isSome then ρ.find id else none

end MerlinVerify.Check.IdentityReturnSyntax
