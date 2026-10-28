/- Exact internal headers for eligible fixed-width casts. The actual operations and exports
remain unchanged; checked defaults and provenance have one deterministic representation. -/
import MerlinVerify.Rules.CastCancel

namespace MerlinVerify.Rules.IntegerCastHeader

open MerlinVerify MerlinVerify.Check MerlinVerify.Sem.Ops IntegerCast

/-- Preserve the operation's name, result bindings, operands and structural header. Legal
candidates have no regions or successors; only independently checked defaults are erased. -/
def canonical : Op → Op
  | .mk name results operands _ _ regions successors =>
    .mk name results operands [] [] regions successors

theorem canonical_legal {Γ : TEnv} {s : Op} {f : Fact} (h : Legal Γ s f) : Legal Γ (canonical s) f := by
  obtain ⟨hd, hn, _, ha, hr, hty, hl, hregs, hsucc, _, _, hw⟩ := h
  cases s
  exact ⟨hd, hn, rfl, ha, hr, hty, hl, hregs, hsucc, rfl, rfl, hw⟩

theorem canonical_step {ib : Nat} {Γ : TEnv} {s : Op} {f : Fact}
    (h : Legal Γ s f) {ctx : Ctx} {call : CallSem} {env : Env} (he : EnvTyped ib Γ env) :
    stepOp opSem ctx call env (canonical s) = stepOp opSem ctx call env s := by
  have legal := h
  obtain ⟨_, _, _, _, _, _, hlookup, _⟩ := legal
  obtain ⟨actual, hv, ht⟩ := he _ _ hlookup
  cases actual with
  | tensor _ => cases ht
  | scalar value =>
    have typed : Val.HasTy ctx.indexBits (.int f.inputWidth) value := int_type_change ht
    rw [cast_exec (canonical_legal h) ctx call env hv typed, cast_exec h ctx call env hv typed]

/-- Only absent attributes or the actual typed enum default are canonical defaults.
Malformed effective properties cannot hide behind a harmless-looking discardable attribute. -/
def defaultAttributes (s : Op) : Bool := decide (
  (s.attr? "overflowFlags" = none ∨ s.attr? "overflowFlags" = some (.enum "arith" "overflow" "none")) ∧
  (s.attr? "fastmath" = none ∨ s.attr? "fastmath" = some (.enum "arith" "fastmath" "none")) ∧
  s.attr? "nonNeg" = none ∧ s.attr? "isExact" = none ∧ s.attr? "disjoint" = none)

/-- Read/Legal checks actual types and the actual attribute precedence. The strict default
policy also refuses promises that a particular cast's semantics happens to ignore. -/
def operation (Γ : TEnv) (s : Op) : Op :=
  if CastCancel.admissible s && defaultAttributes s then
    match read Γ s with
    | some _ => canonical s
    | none => s
  else s

theorem operation_spec (Γ : TEnv) (s : Op) :
    (operation Γ s).isTerminator = s.isTerminator ∧
    (operation Γ s).results = s.results ∧
    ∀ ib ctx call env, EnvTyped ib Γ env →
      stepOp opSem ctx call env (operation Γ s) = stepOp opSem ctx call env s := by
  unfold operation
  split
  · cases hr : read Γ s with
    | none => exact ⟨rfl, rfl, fun _ _ _ _ _ => rfl⟩
    | some f =>
      refine ⟨?_, ?_, fun _ _ _ _ he => canonical_step (read_legal hr) he⟩
      · cases s; rfl
      · cases s; rfl
  · exact ⟨rfl, rfl, fun _ _ _ _ _ => rfl⟩

end MerlinVerify.Rules.IntegerCastHeader
