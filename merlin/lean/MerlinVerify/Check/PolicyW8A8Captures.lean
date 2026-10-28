/- Recompute captured float constants from the actual executed prefix, with SSA shadowing. -/
import MerlinVerify.Check.ConstantFacts

namespace MerlinVerify.Check.PolicyW8A8Captures

open MerlinVerify MerlinVerify.Sem.Ops

def facts (ib : Nat) (pre : List Op) : List ConstantFacts.Fact :=
  pre.foldl (ConstantFacts.advance ib) []

private theorem fold_sound (ib : Nat) (ctx : Ctx) (call : CallSem)
    (hi : ctx.indexBits = ib) (pre : List Op) :
    ∀ (known : List ConstantFacts.Fact) (env env' : Env),
      BindingFacts.Valid ConstantFacts.model known env →
      execOps opSem ctx call env pre = .ok env' →
      BindingFacts.Valid ConstantFacts.model (pre.foldl (ConstantFacts.advance ib) known) env' := by
  induction pre with
  | nil => intro known env env' hf hs; cases hs; exact hf
  | cons op rest ih =>
    intro known env env' hf hs
    by_cases ht : op.isTerminator = true
    · simp [execOps, ht] at hs
    · have ht' : op.isTerminator = false := Bool.eq_false_iff.2 ht
      rw [execOps_cons_step _ _ _ _ _ _ ht'] at hs
      cases hstep : stepOp opSem ctx call env op with
      | ub => simp [hstep] at hs
      | stuck reason => simp [hstep] at hs
      | ok mid =>
        simp only [hstep, Outcome.bind_ok] at hs
        exact ih _ mid env' (ConstantFacts.advance_sound hf hi hstep) hs

/-- Learned constants follow actual execution, including rebindings and nonconstant producers. -/
theorem facts_sound (ib : Nat) (ctx : Ctx) (call : CallSem) (hi : ctx.indexBits = ib)
    (pre : List Op) (env env' : Env) (hs : execOps opSem ctx call env pre = .ok env') :
    BindingFacts.Valid ConstantFacts.model (facts ib pre) env' :=
  fold_sound ib ctx call hi pre [] env env' (by intro f hf; cases hf) hs

def matchesFloat (id : Nat) (bits : FBits .f32) (f : ConstantFacts.Fact) : Bool :=
  match f.value with
  | .scalar (.some (.flt .f32 actual)) => f.id == id && actual == bits
  | _ => false

def hasKnownFloat (known : List ConstantFacts.Fact) (id : Nat) (bits : FBits .f32) : Bool :=
  known.any (matchesFloat id bits)

/-- A streaming checker may reuse proved facts instead of rescanning the prefix per body. -/
theorem hasKnownFloat_sound {known : List ConstantFacts.Fact} {id : Nat} {bits : FBits .f32}
    (h : hasKnownFloat known id bits = true) {env : Env}
    (valid : BindingFacts.Valid ConstantFacts.model known env) :
    env.get? id = some (.scalar (.some (.flt .f32 bits))) := by
  obtain ⟨f, hf, hm⟩ := List.any_eq_true.1 h
  have hv := valid f hf
  obtain ⟨fid, fv⟩ := f
  cases fv with
  | tensor t => simp [matchesFloat] at hm
  | scalar v =>
    cases v with
    | poison => simp [matchesFloat] at hm
    | some s =>
      cases s with
      | bits w x => simp [matchesFloat] at hm
      | flt k x =>
        cases k <;> simp only [matchesFloat] at hm
        all_goals try cases hm
        simp only [Bool.and_eq_true, beq_iff_eq] at hm
        obtain ⟨rfl, rfl⟩ := hm
        exact hv

def hasFloat (ib : Nat) (pre : List Op) (id : Nat) (bits : FBits .f32) : Bool :=
  hasKnownFloat (facts ib pre) id bits

/-- A checked capture is a runtime binding, not a value asserted by a certificate. -/
theorem hasFloat_sound {ib : Nat} {pre : List Op} {id : Nat} {bits : FBits .f32}
    (h : hasFloat ib pre id bits = true) (ctx : Ctx) (call : CallSem)
    (hi : ctx.indexBits = ib) (env env' : Env)
    (hs : execOps opSem ctx call env pre = .ok env') :
    env'.get? id = some (.scalar (.some (.flt .f32 bits))) :=
  hasKnownFloat_sound h (facts_sound ib ctx call hi pre env env' hs)

#print axioms facts_sound
#print axioms hasKnownFloat_sound
#print axioms hasFloat_sound

end MerlinVerify.Check.PolicyW8A8Captures
