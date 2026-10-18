/- Exact fixed-width scalar neutral identities. Current literal values come from successful
source execution, and continuation aliases reuse the shared constant-fact tail normalizer. -/
import MerlinVerify.Check.ConstantAlias
import MerlinVerify.Rules.IntegerIdentity
import MerlinVerify.Rules.Commutative

namespace MerlinVerify.Rules.IntegerNeutral

open MerlinVerify MerlinVerify.Check MerlinVerify.Sem.Ops

inductive Kind where
  | add | sub | mul | band | bor | bxor | minU | maxU | minS | maxS
  deriving DecidableEq, Repr

def Kind.name : Kind → String
  | .add => "arith.addi" | .sub => "arith.subi" | .mul => "arith.muli"
  | .band => "arith.andi" | .bor => "arith.ori" | .bxor => "arith.xori"
  | .minU => "arith.minui" | .maxU => "arith.maxui"
  | .minS => "arith.minsi" | .maxS => "arith.maxsi"

def kinds : List Kind := [.add,.sub,.mul,.band,.bor,.bxor,.minU,.maxU,.minS,.maxS]

def kind? (name : String) : Option Kind := kinds.find? (fun k => k.name == name)

def Kind.neutral (k : Kind) (w : Nat) : BitVec w :=
  match k with
  | .mul => 1
  | .band | .minU => BitVec.allOnes w
  | .minS => BitVec.intMax w
  | .maxS => BitVec.intMin w
  | _ => 0

def Kind.reverseOK (k : Kind) (rev : Bool) : Prop := k ≠ .sub ∨ rev = false
instance (k : Kind) (rev : Bool) : Decidable (k.reverseOK rev) := by unfold Kind.reverseOK; infer_instance

theorem minU_neutral {w : Nat} (x : BitVec w) :
    (if x.ult (BitVec.allOnes w) then x else BitVec.allOnes w) = x := by
  apply BitVec.eq_of_toNat_eq
  have hx := x.isLt
  simp only [BitVec.ult_eq_decide, decide_eq_true_eq, BitVec.toNat_allOnes]
  split_ifs
  · rfl
  · simp only [BitVec.toNat_allOnes]; omega

theorem maxU_neutral {w : Nat} (x : BitVec w) : (if x.ult 0 then 0 else x) = x := by
  simp [BitVec.ult_eq_decide]

theorem minS_neutral {w : Nat} (x : BitVec w) :
    (if x.slt (BitVec.intMax w) then x else BitVec.intMax w) = x := by
  apply BitVec.eq_of_toInt_eq
  have hx : x.toInt ≤ 2 ^ (w - 1) - 1 := BitVec.toInt_le
  simp only [BitVec.slt_eq_decide, decide_eq_true_eq, BitVec.toInt_intMax]
  split_ifs
  · rfl
  · simp only [BitVec.toInt_intMax]; omega

theorem maxS_neutral {w : Nat} (x : BitVec w) :
    (if x.slt (BitVec.intMin w) then BitVec.intMin w else x) = x := by
  apply BitVec.eq_of_toInt_eq
  have hx := BitVec.toInt_intMin_le x
  simp only [BitVec.slt_eq_decide, decide_eq_true_eq]
  split_ifs <;> omega

theorem minU_neutral_left {w : Nat} (x : BitVec w) :
    (if (BitVec.allOnes w).ult x then BitVec.allOnes w else x) = x :=
  (Commutative.unsigned_min_comm _ _).trans (minU_neutral x)
theorem maxU_neutral_left {w : Nat} (x : BitVec w) : (if (0 : BitVec w).ult x then x else 0) = x :=
  (Commutative.unsigned_max_comm _ _).trans (maxU_neutral x)
theorem minS_neutral_left {w : Nat} (x : BitVec w) :
    (if (BitVec.intMax w).slt x then BitVec.intMax w else x) = x :=
  (Commutative.signed_min_comm _ _).trans (minS_neutral x)
theorem maxS_neutral_left {w : Nat} (x : BitVec w) :
    (if (BitVec.intMin w).slt x then x else BitVec.intMin w) = x :=
  (Commutative.signed_max_comm _ _).trans (maxS_neutral x)

/-- Runtime scalar width evidence is essential. Poison remains poison, while every actual
same-width bitvector is returned exactly; no floating or index identity is introduced. -/
theorem meaning {ib w : Nat} {s : Op} {r : Value} {value : Val}
    (kind : Kind) (rev : Bool) (hrev : kind.reverseOK rev)
    (hd : dialectOf s.name = "arith") (hn : s.name = kind.name)
    (hm : unmodelledAttr? s.blind = none) (hr : s.results = [r]) (ht : r.ty = .int w)
    (hf : overflowFlags s = []) (hv : Val.HasTy ib (.int w) value) (ctx : Ctx) (call : CallSem) :
    opSem ctx call s (IntegerIdentity.ordered rev (.scalar value)
      (.scalar (.some (.bits w (kind.neutral w))))) [] = .ok [.scalar value] := by
  rw [opSem_modelled hm]
  simp only [dispatch, Op.blind_name, hd]
  have hf' : overflowFlags s.blind = [] := by simpa [overflowFlags, attrEnum?] using hf
  rcases hasTy_intWidth (ib := ib) (t := .int w) rfl hv with rfl | ⟨x,rfl⟩
  all_goals
    cases kind <;> cases rev <;>
      simp_all [Kind.name, Kind.neutral, Kind.reverseOK, IntegerIdentity.ordered,
        Arith.sem, Arith.binOp, Arith.pureN, Arith.resultElt, Op.blind_results,
        liftEager, Arith.intBin, Arith.arithWithFlags, Arith.bv,
        minU_neutral, minS_neutral, maxS_neutral,
        minU_neutral_left, minS_neutral_left, maxS_neutral_left]
  all_goals
    intro h
    apply BitVec.eq_of_toNat_eq
    simp_all [BitVec.ult_eq_decide]

private def valueDecEq (a b : Value) : Decidable (a = b) :=
  decidable_of_iff (a.id = b.id ∧ a.ty = b.ty) (by cases a; cases b; simp)
local instance : DecidableEq Value := valueDecEq

def Legal (kind : Kind) (rev : Bool) (Γ : TEnv) (s : Op)
    (input constant : ValueId) (r : Value) (w : Nat) : Prop :=
  dialectOf s.name = "arith" ∧ s.name = kind.name ∧ unmodelledAttr? s.blind = none ∧
  s.operands = IntegerIdentity.ordered rev input constant ∧ s.results = [r] ∧ r.ty = .int w ∧
  Γ.lookup input = some (.scalar (.int w)) ∧ s.regions = [] ∧ s.successors = [] ∧
  overflowFlags s = [] ∧ Arith.fastmathKey s = "0" ∧ hasUnit s "nonNeg" = false ∧
  hasUnit s "isExact" = false ∧ hasUnit s "disjoint" = false ∧ kind.reverseOK rev ∧
  freshIds (SplatNormalize.diagonal Γ) [r.id] = true

instance (kind : Kind) (rev : Bool) (Γ : TEnv) (s : Op) (x c : ValueId) (r : Value) (w : Nat) :
    Decidable (Legal kind rev Γ s x c r w) := by unfold Legal; infer_instance

def proposal (rev : Bool) (s : Op) : Option (Kind × ValueId × ValueId × Value × Nat) := do
  let kind ← kind? s.name
  let [a,b] := s.operands | none
  let [r] := s.results | none
  let .int w := r.ty | none
  pure (kind, if rev then b else a, if rev then a else b, r, w)

def eliminateAt (ib : Nat) (rev : Bool) (Γ : TEnv) (facts : List ConstantFacts.Fact)
    (s : Op) (tail : List Op) : Option (List Op) := do
  let (kind,input,constant,r,w) ← proposal rev s
  let fact ← facts.find? (fun f => f.id == constant)
  if Legal kind rev Γ s input constant r w ∧
      ConstantAlias.matching ib Γ r (.some (.bits w (kind.neutral w))) fact = true then
    TypedAlias.rename? Γ r.id input tail
  else none

theorem eliminateAt_sound {ib : Nat} {rev : Bool} {Γ : TEnv} {facts : List ConstantFacts.Fact}
    {s : Op} {tail renamed : List Op} (h : eliminateAt ib rev Γ facts s tail = some renamed)
    {ctx : Ctx} {call : CallSem} {env : Env}
    (he : EnvTyped ib Γ env) (hf : BindingFacts.Valid ConstantFacts.model facts env) :
    ∃ env', stepOp opSem ctx call env s = .ok env' ∧
      evalOps opSem ctx call env renamed = evalOps opSem ctx call env' tail := by
  unfold eliminateAt at h
  cases hp : proposal rev s with
  | none => simp [hp] at h
  | some item =>
    obtain ⟨kind,input,constant,r,w⟩ := item
    simp only [hp, bind, Option.bind] at h
    cases hc : facts.find? (fun f => f.id == constant) with
    | none => simp [hc] at h
    | some fact =>
      simp only [hc] at h
      split at h
      · next checks =>
        obtain ⟨legal,hmatch⟩ := checks
        obtain ⟨hd,hn,hm,ho,hr,ht,hlookup,hregions,_,hflags,_,_,_,_,hrev,_⟩ := legal
        have hid : fact.id = constant := by simpa using List.find?_some hc
        have hconstant : env.get? constant = some (.scalar (.some (.bits w (kind.neutral w)))) := by
          have hv := hf fact (List.mem_of_find?_eq_some hc)
          change env.get? fact.id = some fact.value at hv
          simpa [hid, ConstantAlias.matching_value hmatch] using hv
        obtain ⟨actual,hinput,htyped⟩ := he _ _ hlookup
        cases actual with
        | tensor _ => cases htyped
        | scalar value =>
          have hargs : env.getAll s.operands = .ok (IntegerIdentity.ordered rev (.scalar value)
              (.scalar (.some (.bits w (kind.neutral w))))) := by
            cases rev <;> simp [ho, IntegerIdentity.ordered, Env.getAll, Outcome.mapM,
              hinput, hconstant, Outcome.ofOption]
          have hstep : stepOp opSem ctx call env s = .ok ((r.id,.scalar value) :: env) := by
            unfold stepOp
            rw [hargs, Outcome.bind_ok, hregions]
            simp only [evalRegions]
            rw [meaning kind rev hrev hd hn hm hr ht hflags htyped ctx call]
            simp [Env.bindAll, hr]
          exact ⟨_,hstep,TypedAlias.rename_sound h hinput⟩
      · cases h

def eliminate (ib : Nat) (Γ : TEnv) (facts : List ConstantFacts.Fact) (s : Op) (tail : List Op) :
    Option (List Op) := (eliminateAt ib false Γ facts s tail).orElse fun _ => eliminateAt ib true Γ facts s tail

theorem eliminate_sound {ib : Nat} {Γ : TEnv} {facts : List ConstantFacts.Fact}
    {s : Op} {tail renamed : List Op} (h : eliminate ib Γ facts s tail = some renamed)
    {ctx : Ctx} {call : CallSem} {env : Env}
    (he : EnvTyped ib Γ env) (hf : BindingFacts.Valid ConstantFacts.model facts env) :
    ∃ env', stepOp opSem ctx call env s = .ok env' ∧
      evalOps opSem ctx call env renamed = evalOps opSem ctx call env' tail := by
  unfold eliminate at h
  cases hfirst : eliminateAt ib false Γ facts s tail with
  | some result =>
    simp only [hfirst, Option.orElse_some, Option.some.injEq] at h
    subst result
    exact eliminateAt_sound hfirst he hf
  | none =>
    simp only [hfirst, Option.orElse_none] at h
    exact eliminateAt_sound h he hf

/-- Extend the existing scalar-constant family in the same exact tail walk. Its source facts,
alias logic and total dead-constant cleanup remain unchanged. No extra region walker is added. -/
def family : TailNormalize.Family ConstantFacts.Fact ConstantFacts.model where
  advance := ConstantAlias.family.advance
  advance_sound := ConstantAlias.family.advance_sound
  finish ib Γ facts _ s tail := (eliminate ib Γ facts s tail).getD (ConstantAlias.finish ib Γ facts s tail)
  finish_sound ib Γ facts _ s tail ctx call env ht he hf hi := by
    cases h : eliminate ib Γ facts s tail with
    | none => exact ConstantAlias.finish_sound ht hf hi
    | some renamed =>
      simp only [Option.getD_some]
      obtain ⟨env',hs,heq⟩ := eliminate_sound h he hf (ctx := ctx) (call := call)
      rw [evalOps_cons_step _ _ _ _ _ _ ht, hs, Outcome.bind_ok]
      exact heq

def normalize (ib : Nat) (Γ : TEnv) (facts : List ConstantFacts.Fact) : List Op → Option (List Op) :=
  TailNormalize.normalize family ib Γ facts []

/-- Runtime width and current literal bindings justify each alias. The shared tail induction
preserves the complete source execution, including poison, captures, and failures. -/
theorem normalize_sound {ib : Nat} (R : Rulebook opSem ib) (d : Nat) :
    ∀ Γ facts ss ns yields ctx call env,
      checkOps R d Γ ss yields = true → normalize ib Γ facts ss = some ns →
      EnvTyped ib Γ env → BindingFacts.Valid ConstantFacts.model facts env →
      ctx.indexBits = ib → d ≤ ctx.loopIndex.length →
      evalOps opSem ctx call env ns = evalOps opSem ctx call env ss := by
  intro Γ facts ss ns yields ctx call env hc hn he hf hi hd
  exact TailNormalize.normalize_sound family R d Γ facts [] ss ns yields ctx call env hc hn he hf hi hd

def base (ib : Nat) : SplatNormalizer.Normalizer ib := TailNormalize.base family ib []

end MerlinVerify.Rules.IntegerNeutral
