/- Exact typed scalar bitcast identities and round trips. One shared tail traversal owns facts,
alias validation, escaping bindings and total producer deletion. -/
import MerlinVerify.Rules.BitcastCancelKernel
import MerlinVerify.Check.TailNormalize
import MerlinVerify.Check.TypedAlias
import MerlinVerify.Check.TypedDead

namespace MerlinVerify.Check.BitcastCancel

open MerlinVerify MerlinVerify.Sem.Ops
open Rules.ScalarBitcast Rules.BitcastCancel

private def valueDecEq (a b : Value) : Decidable (a = b) :=
  decidable_of_iff (a.id = b.id ∧ a.ty = b.ty) (by cases a; cases b; simp)
local instance : DecidableEq Value := valueDecEq

structure Cast where
  root : ValueId
  source : Ty
  target : Ty
  result : Value

/-- This stricter identity policy also refuses arithmetic promises that the raw-bit dispatcher
currently ignores. Successful typed source execution remains the source of every origin fact. -/
def Legal (Γ : TEnv) (s : Op) (p : Cast) : Prop :=
  s.name = "arith.bitcast" ∧ unmodelledAttr? s.blind = none ∧ Plain s ∧
  hasUnit s "isExact" = false ∧ hasUnit s "disjoint" = false ∧
  s.regions = [] ∧ s.successors = [] ∧ s.operands = [p.root] ∧ s.results = [p.result] ∧
  Γ.lookup p.root = some (.scalar p.source) ∧ p.result.ty = p.target ∧ Compatible p.source p.target

instance (Γ : TEnv) (s : Op) (p : Cast) : Decidable (Legal Γ s p) := by
  unfold Legal; infer_instance

def read (Γ : TEnv) (s : Op) : Option Cast := do
  let [root] := s.operands | none
  let [result] := s.results | none
  let some (.scalar src) := Γ.lookup root | none
  let p := Cast.mk root src result.ty result
  if Legal Γ s p then some p else none

theorem read_legal {Γ : TEnv} {s : Op} {p : Cast} (h : read Γ s = some p) : Legal Γ s p := by
  unfold read at h
  split at h <;> try cases h
  split at h <;> try cases h
  split at h <;> try cases h
  dsimp only at h
  split at h
  · cases h; assumption
  · cases h

/-- Supported source types contain no index-width dependency. -/
theorem type_change {src dst : Ty} {a b : Nat} {value : Val}
    (hc : Compatible src dst) (ht : Val.HasTy a src value) : Val.HasTy b src value := by
  cases src <;> cases dst <;> simp only [Compatible] at hc
  all_goals cases value with
    | poison => trivial
    | some scalar => cases scalar <;> simp_all [Val.HasTy, Scalar.conforms]

theorem cast_exec {ib : Nat} {Γ : TEnv} {s : Op} {p : Cast}
    (h : Legal Γ s p) {env : Env} (he : EnvTyped ib Γ env) (ctx : Ctx) (call : CallSem) :
    ∃ value, env.get? p.root = some (.scalar value) ∧ Val.HasTy 0 p.source value ∧
      stepOp opSem ctx call env s = .ok ((p.result.id, .scalar (reinterpret p.target value)) :: env) := by
  obtain ⟨hn, hm, _, _, _, hregs, _, ho, hr, hl, hty, hc⟩ := h
  obtain ⟨actual, ha, ht⟩ := he _ _ hl
  cases actual with
  | tensor _ => cases ht
  | scalar value =>
    have hv : Val.HasTy ib p.source value := ht
    refine ⟨value, ha, type_change hc hv, ?_⟩
    have hs := meaning_exact hn hm (resultElt_eq hr hty hc) hc ctx call [] (type_change hc hv)
    unfold stepOp
    simp only [ho, Env.getAll, Outcome.mapM, ha, Outcome.ofOption, Outcome.bind_ok,
      hregs, evalRegions, hs, hr, Env.bindAll]
    simp

structure Fact where
  root : ValueId
  current : ValueId
  source : Ty
  target : Ty
  deriving DecidableEq

def Holds (f : Fact) (env : Env) : Prop :=
  ∃ value, Val.HasTy 0 f.source value ∧ Compatible f.source f.target ∧
    env.get? f.root = some (.scalar value) ∧
    env.get? f.current = some (.scalar (reinterpret f.target value))

def model : BindingFacts.Model Fact where
  dependencies f := [f.root, f.current]
  holds := Holds
  congr f env env' h hf := by
    obtain ⟨value, ht, hc, hr, hv⟩ := hf
    exact ⟨value, ht, hc, (h f.root (by simp)).symm.trans hr,
      (h f.current (by simp)).symm.trans hv⟩

def learn (Γ : TEnv) (s : Op) : Option Fact := do
  let p ← read Γ s
  if p.root ≠ p.result.id then some ⟨p.root, p.result.id, p.source, p.target⟩ else none

theorem learn_sound {ib : Nat} {Γ : TEnv} {s : Op} {f : Fact}
    (h : learn Γ s = some f) {ctx : Ctx} {call : CallSem} {env env' : Env}
    (he : EnvTyped ib Γ env) (hs : stepOp opSem ctx call env s = .ok env') : Holds f env' := by
  unfold learn at h
  cases hp : read Γ s with
  | none => simp [hp] at h
  | some p =>
    simp only [hp, bind, Option.bind] at h
    split at h
    · next hne =>
      cases h
      have hc := read_legal hp
      obtain ⟨value, hv, ht, hstep⟩ := cast_exec hc he ctx call
      rw [hstep] at hs; cases hs
      obtain ⟨_, _, _, _, _, _, _, _, _, _, _, hcompatible⟩ := hc
      refine ⟨value, ht, hcompatible, ?_, ?_⟩
      · simpa [Env.get?, List.lookup, show (p.root == p.result.id) = false by simpa using hne] using hv
      · simp [Env.get?, List.lookup]
    · cases h

def advance (Γ : TEnv) (facts : List Fact) (s : Op) : List Fact :=
  let retained := BindingFacts.forget model (s.results.map (·.id)) facts
  match learn Γ s with
  | some f => f :: retained
  | none => retained

theorem advance_sound {ib : Nat} {Γ : TEnv} {facts : List Fact} {s : Op}
    {ctx : Ctx} {call : CallSem} {env env' : Env}
    (he : EnvTyped ib Γ env) (hf : BindingFacts.Valid model facts env)
    (hs : stepOp opSem ctx call env s = .ok env') : BindingFacts.Valid model (advance Γ facts s) env' := by
  have retained := BindingFacts.forget_step model hf hs
  unfold advance
  cases hl : learn Γ s with
  | none => exact retained
  | some f =>
    intro g hg
    rcases List.mem_cons.1 hg with rfl | hg
    · exact learn_sound hl he hs
    · exact retained g hg

/-- An integer identity aliases its own input. Cross-kind cancellation additionally checks
an exact current-binding origin and returns to its original type, including the float format. -/
def origin (Γ : TEnv) (facts : List Fact) (p : Cast) : Option ValueId :=
  if p.source = p.target then some p.root else
    (facts.find? fun f =>
      f.current == p.root && f.target == p.source && f.source == p.target &&
      decide (Γ.lookup f.root = some (.scalar f.source))).map (·.root)

theorem origin_sound {Γ : TEnv} {facts : List Fact} {p : Cast} {root : ValueId}
    (ho : origin Γ facts p = some root) (hc : Compatible p.source p.target)
    {env : Env} (hf : BindingFacts.Valid model facts env) {value : Val}
    (hv : env.get? p.root = some (.scalar value)) (ht : Val.HasTy 0 p.source value) :
    ∃ original, env.get? root = some (.scalar original) ∧ reinterpret p.target value = original := by
  unfold origin at ho
  split at ho
  · next heq =>
    cases ho
    refine ⟨value, hv, ?_⟩
    cases hsrc : p.source <;> cases hdst : p.target <;>
      simp only [hsrc, hdst, Compatible] at hc <;> simp [hsrc, hdst] at heq
    next a b =>
      cases heq
      exact integer_identity (hsrc ▸ ht)
  · cases hfind : facts.find? (fun f =>
      f.current == p.root && f.target == p.source && f.source == p.target &&
      decide (Γ.lookup f.root = some (.scalar f.source))) with
    | none => simp [hfind] at ho
    | some f =>
      simp only [hfind, Option.map_some, Option.some.injEq] at ho
      subst root
      have checks := List.find?_some hfind
      simp only [Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at checks
      obtain ⟨⟨⟨hcurrent, htarget⟩, hsource⟩, _⟩ := checks
      obtain ⟨original, htype, hcompatible, hroot, hvalue⟩ := hf f (List.mem_of_find?_eq_some hfind)
      refine ⟨original, hroot, ?_⟩
      rw [hcurrent] at hvalue
      have heq : value = reinterpret f.target original := by
        rw [hv] at hvalue
        exact RVal.scalar.inj (Option.some.inj hvalue)
      rw [heq, ← hsource]
      exact roundtrip hcompatible htype

-- The executable alias proposal is validated against the entire continuation.
def eliminate (Γ : TEnv) (facts : List Fact) (s : Op) (tail : List Op) : Option (List Op) := do
  let p ← read Γ s
  let root ← origin Γ facts p
  TypedAlias.rename? Γ p.result.id root tail


theorem eliminate_sound {ib : Nat} {Γ : TEnv} {facts : List Fact} {s : Op} {tail renamed : List Op}
    (h : eliminate Γ facts s tail = some renamed) {ctx : Ctx} {call : CallSem} {env : Env}
    (he : EnvTyped ib Γ env) (hf : BindingFacts.Valid model facts env) :
    ∃ env', stepOp opSem ctx call env s = .ok env' ∧
      evalOps opSem ctx call env renamed = evalOps opSem ctx call env' tail := by
  unfold eliminate at h
  cases hp : read Γ s with
  | none => simp [hp] at h
  | some p =>
    simp only [hp, bind, Option.bind] at h
    cases ho : origin Γ facts p with
    | none => simp [ho] at h
    | some root =>
      simp only [ho] at h
      have hc := read_legal hp
      obtain ⟨value, hv, ht, hs⟩ := cast_exec hc he ctx call
      obtain ⟨_, _, _, _, _, _, _, _, _, _, _, hcompatible⟩ := hc
      obtain ⟨original, hr, heq⟩ := origin_sound ho hcompatible hf hv ht
      rw [heq] at hs
      exact ⟨_, hs, TypedAlias.rename_sound h hr⟩

def drop? (Γ : TEnv) (s : Op) (tail : List Op) : Bool :=
  (read Γ s).isSome && TypedDead.unused Γ s tail

theorem drop_sound {ib : Nat} {Γ : TEnv} {s : Op} {tail : List Op}
    (h : drop? Γ s tail = true) {env : Env} (he : EnvTyped ib Γ env) (ctx : Ctx) (call : CallSem) :
    ∃ env', stepOp opSem ctx call env s = .ok env' ∧
      evalOps opSem ctx call env' tail = evalOps opSem ctx call env tail := by
  simp only [drop?, Bool.and_eq_true] at h
  obtain ⟨p, hp⟩ := Option.isSome_iff_exists.1 h.1
  have hc := read_legal hp
  obtain ⟨value, _, _, hs⟩ := cast_exec hc he ctx call
  obtain ⟨_, _, _, _, _, _, _, _, hr, _⟩ := hc
  exact TypedDead.drop_sound h.2 ⟨p.result, _, hr, hs⟩

def finish (Γ : TEnv) (facts : List Fact) (s : Op) (tail : List Op) : List Op :=
  if drop? Γ s tail then tail else (eliminate Γ facts s tail).getD (s :: tail)

theorem finish_sound {ib : Nat} {Γ : TEnv} {facts : List Fact} {s : Op} {tail : List Op}
    {ctx : Ctx} {call : CallSem} {env : Env} (ht : s.isTerminator = false)
    (he : EnvTyped ib Γ env) (hf : BindingFacts.Valid model facts env) :
    evalOps opSem ctx call env (finish Γ facts s tail) = evalOps opSem ctx call env (s :: tail) := by
  unfold finish
  split
  · next hd =>
    obtain ⟨env', hs, heq⟩ := drop_sound hd he ctx call
    rw [evalOps_cons_step _ _ _ _ _ _ ht, hs, Outcome.bind_ok]
    exact heq.symm
  · cases helim : eliminate Γ facts s tail with
    | none => rfl
    | some renamed =>
      simp only [Option.getD_some]
      obtain ⟨env', hs, heq⟩ := eliminate_sound helim he hf (ctx := ctx) (call := call)
      rw [evalOps_cons_step _ _ _ _ _ _ ht, hs, Outcome.bind_ok]
      exact heq

def family : TailNormalize.Family Fact model where
  advance _ Γ facts s := advance Γ facts s
  advance_sound _ _ _ _ _ _ _ _ he hf _ hs := advance_sound he hf hs
  finish _ Γ facts _ s tail := finish Γ facts s tail
  finish_sound _ _ _ _ _ _ _ _ _ ht he hf _ := finish_sound ht he hf

def normalize (ib : Nat) (Γ : TEnv) (facts : List Fact) : List Op → Option (List Op) :=
  TailNormalize.normalize family ib Γ facts []

theorem normalize_sound {ib : Nat} (R : Rulebook opSem ib) (d : Nat) :
    ∀ Γ facts ss ns yields ctx call env,
      checkOps R d Γ ss yields = true → normalize ib Γ facts ss = some ns →
      EnvTyped ib Γ env → BindingFacts.Valid model facts env →
      ctx.indexBits = ib → d ≤ ctx.loopIndex.length →
      evalOps opSem ctx call env ns = evalOps opSem ctx call env ss := by
  intro Γ facts ss ns yields ctx call env hc hn he hf hi hd
  exact TailNormalize.normalize_sound family R d Γ facts [] ss ns yields ctx call env hc hn he hf hi hd

/-- Reuse the existing structured-region walker centrally; every local block starts empty. -/
def base (ib : Nat) : SplatNormalizer.Normalizer ib := TailNormalize.base family ib []

end MerlinVerify.Check.BitcastCancel
