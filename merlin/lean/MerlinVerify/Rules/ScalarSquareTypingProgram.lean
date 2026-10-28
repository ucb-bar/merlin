/- Whole-module result preservation, including internal calls by fuel induction. External
contracts assert only successful-result typing; they do not assert progress or poison freedom. -/
import MerlinVerify.Check.ScalarSquareTyping

namespace MerlinVerify.Rules.ScalarSquareTyping
open MerlinVerify MerlinVerify.Check MerlinVerify.Sem.Ops ScalarSquareSSA ScalarSquareClaims

def signatureOf? (ib : Nat) (op : Op) : Option Signature := do
  let some (.type (.function ats rts)) := op.attr? "function_type" | none
  return (← ats.mapM (vtyOf ib), ← rts.mapM (vtyOf ib))

def signatures (ib : Nat) (source : Module) : Signatures :=
  source.ops.filterMap fun op => do
    let f ← op.asFunc?
    return (f.name, ← signatureOf? ib op)

def functionCheck (ib : Nat) (table : Signatures) (f : Func) : Bool :=
  match table.lookup f.name with
  | none => false
  | some sig => if f.isDecl then true else regionCheck ib table [] f.body ⟨sig.1, sig.2, 0⟩

def programCheck (ib : Nat) (source : Module) : Bool :=
  source.ops.all (fun op => op.asFunc?.isSome && (signatureOf? ib op).isSome &&
    decide (op.results = [] ∧ op.operands = [] ∧ op.successors = [] ∧ op.regions.length ≤ 1)) &&
    decide ((source.funcs.map (·.name)).Nodup) &&
    source.funcs.all (functionCheck ib (signatures ib source))

/-- Explicit boundary for foreign implementations. No contracts are required when the module
has no declarations. An external function may still be stuck, UB, or return poison. -/
def ExternalResults (ib : Nat) (source : Module) (ctx : Ctx) : Prop :=
  ∀ name f sig, source.func? name = some f → f.isDecl = true →
    (signatures ib source).lookup name = some sig → ∀ args, HasTys ib args sig.1 →
      ∀ results, ctx.ext name args = .ok results → HasTys ib results sig.2

theorem function_checked {ib : Nat} {source : Module} (h : programCheck ib source = true)
    {name : String} {f : Func} (found : source.func? name = some f) :
    f.name = name ∧ functionCheck ib (signatures ib source) f = true := by
  have member := List.mem_of_find?_eq_some found
  have named : f.name = name := by simpa using List.find?_some found
  simp only [programCheck, Bool.and_eq_true] at h
  exact ⟨named, List.all_eq_true.1 h.2 f member⟩

theorem calls_preserve {ib : Nat} {source : Module} (h : programCheck ib source = true)
    (ctx : Ctx) (hib : ctx.indexBits = ib) (external : ExternalResults ib source ctx) :
    ∀ fuel, CallTyped ib (signatures ib source) (evalFunc opSem ctx source fuel) := by
  intro fuel
  induction fuel with
  | zero => intro _ _ _ _ _ _ impossible; cases impossible
  | succ fuel ih =>
    intro name sig hs args ha results executed
    cases found : source.func? name with
    | none => simp [evalFunc, found] at executed
    | some f =>
      have checked := function_checked h found
      have hs' : (signatures ib source).lookup f.name = some sig := by rw [checked.1]; exact hs
      by_cases declaration : f.isDecl = true
      · exact external name f sig found declaration hs args ha results
          (by simpa [evalFunc, found, declaration] using executed)
      · have body : regionCheck ib (signatures ib source) [] f.body ⟨sig.1, sig.2, 0⟩ = true := by
          simpa [functionCheck, hs', declaration] using checked.2
        exact regionCheck_preserves ib _ [] f.body _ _ [] body
          (fun _ _ impossible => by cases impossible) ih ctx hib args ha results
          (by simpa [evalFunc, found, declaration] using executed)

theorem arguments_types {ib : Nat} : ∀ (args : List Value) (types : List VTy),
    args.mapM (fun a => vtyOf ib a.ty) = some types → ∀ values,
      Arguments ib args values ↔ HasTys ib values types
  | [], _, h, _ => by
    simp only [List.mapM_nil, Option.pure_def, Option.some.injEq] at h
    subst h
    simp only [Arguments, HasTys, List.forall₂_nil_left_iff, List.forall₂_nil_right_iff]
  | arg :: args, types, h, values => by
    simp only [List.mapM_cons] at h
    cases ht : vtyOf ib arg.ty with
    | none => simp [ht] at h
    | some ty =>
      cases hts : args.mapM (fun a => vtyOf ib a.ty) with
      | none => simp [ht, hts] at h
      | some ts =>
        simp only [ht, hts, Option.pure_def, bind, Option.bind, Option.some.injEq] at h
        subst types
        cases values with
        | nil => simp [Arguments, HasTys]
        | cons value values =>
          simp only [Arguments, HasTys, List.forall₂_cons, HasType, ht, Option.some.injEq,
            exists_eq_left']
          exact and_congr Iff.rfl (arguments_types args ts hts values)

theorem calls_admitted {ib : Nat} {source : Module} (h : programCheck ib source = true) :
    CallAdmits ib (signatures ib source) (Inputs source ib) := by
  intro name sig hs args ha
  cases found : source.func? name with
  | none => simp [Inputs, found]
  | some f =>
    by_cases declaration : f.isDecl = true
    · simp [Inputs, found, declaration]
    · have checked := function_checked h found
      have body : regionCheck ib (signatures ib source) [] f.body ⟨sig.1, sig.2, 0⟩ = true := by
        simpa [functionCheck, checked.1, hs, declaration] using checked.2
      cases hb : f.body with
      | mk blocks =>
        rw [hb] at body
        cases blocks with
        | nil => cases body
        | cons block rest =>
          cases block with
          | mk formals ops =>
            cases rest with
            | cons b bs => simp [hb, regionCheck] at body
            | nil =>
              have typed : formals.mapM (fun a => vtyOf ib a.ty) = some sig.1 :=
                by
                  simp only [regionCheck, Bool.and_eq_true, beq_iff_eq] at body
                  exact body.1
              simpa [Inputs, found, declaration, ScalarSquareComposition.BlockDomain, hb, Region.blocks, Block.args] using
                (arguments_types formals sig.1 typed args).mpr ha

#print axioms calls_preserve
end MerlinVerify.Rules.ScalarSquareTyping
