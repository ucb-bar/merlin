/-
The capture is fully modelled (design §4.5): `supportCheck` + `supportCheck_sound` give
`FuncNotStuck` on well-typed inputs, which turns the per-boundary conditional refinements
(`FuncRefinesC`) into strict refinement (`FuncRefinesC.strict`).

The typing rule for Merlin's op semantics is assembled here from per-family rules; each family
rule comes with its safety proof. An op the rule does not cover is refused, so `supportCheck`
fails closed: a program with an op outside the proved fragment is reported as not fully modelled.
-/
import MerlinVerify.Check.Progress
import MerlinVerify.Sem.Ops

namespace MerlinVerify.Check

open MerlinVerify MerlinVerify.Sem.Ops

/-! ### Reducing `opSem` to a family -/

theorem opSem_modelled {ctx : Ctx} {call : CallSem} {op : Op} {args : List RVal}
    {regions : List (Ctx → RegionSem)} (h : unmodelledAttr? op.blind = none) :
    opSem ctx call op args regions =
      dispatch ctx (restrictCall (calleeOf op.blind) call) op.blind args regions := by
  simp [opSem, h]

@[simp] theorem Op.blind_name (op : Op) : op.blind.name = op.name := by cases op; rfl
@[simp] theorem Op.blind_results (op : Op) :
    op.blind.results = op.results.map fun r => { r with id := 0 } := by cases op; rfl
@[simp] theorem Op.blind_attr? (op : Op) (k : String) : op.blind.attr? k = op.attr? k := by
  cases op; rfl

/-! ### Family rules -/

/-- `cf.assert %c`: one `i1` operand, no results, no regions. The dialect test is a hypothesis the
proof rewrites with (string functions are checked at run time, never evaluated in proofs). -/
def cfRule (op : Op) (argTys resTys : List VTy) : Option (List RegionSig) :=
  if dialectOf op.name = "cf" ∧ op.name = "cf.assert" ∧ argTys = [.scalar (.int 1)] ∧ resTys = [] then
    some []
  else none

theorem cfRule_sound (ib : Nat) (op : Op) (argTys resTys : List VTy) (sigs : List RegionSig)
    (hm : unmodelledAttr? op.blind = none) (h : cfRule op argTys resTys = some sigs) :
    OpSafe opSem ib op argTys resTys sigs := by
  simp only [cfRule] at h
  split at h
  · next hc =>
    obtain ⟨hd, hn, rfl, rfl⟩ := hc
    intro ctx call args regions _ hargs _
    obtain _ | ⟨hv, hnil⟩ := hargs
    cases hnil
    rename_i v
    rw [opSem_modelled hm]
    simp only [dispatch, Op.blind_name, hd]
    simp only [Cf.sem, Op.blind_name, hn]
    cases v with
    | tensor t => simp [RVal.HasTy] at hv
    | scalar x =>
      cases x with
      | poison => exact ⟨trivial, fun rs hrs => by cases hrs⟩
      | some sc =>
        cases sc with
        | flt k b => simp [RVal.HasTy, Val.HasTy, Scalar.conforms] at hv
        | bits w b =>
          simp only [RVal.HasTy, Val.HasTy, Scalar.conforms, beq_iff_eq] at hv
          subst hv
          by_cases hb : b = 1
          · simp only [hb, ↓reduceIte]
            exact ⟨trivial, fun rs hrs => by cases hrs; exact List.Forall₂.nil⟩
          · simp only [hb, ↓reduceIte]
            exact ⟨trivial, fun rs hrs => by cases hrs⟩
  · cases h

end MerlinVerify.Check
