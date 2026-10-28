/- Exact widening followed by partial truncation. Existing cast facts establish actual scalar
origins; CastCancel supplies the same checked tail aliases and total dead-cast cleanup. -/
import MerlinVerify.Rules.IntegerCastHeader

namespace MerlinVerify.Rules.IntegerCastPartial

open MerlinVerify MerlinVerify.Check MerlinVerify.Sem.Ops IntegerCast

/-- Discarding high bits of a widened signed value preserves precisely the requested bits. -/
theorem signed_truncate {a b c : Nat} (x : BitVec a) (h : c ≤ b) :
    (x.signExtend b).setWidth c = x.signExtend c := by
  apply BitVec.eq_of_getLsbD_eq
  intro i hi
  have hib : i < b := Nat.lt_of_lt_of_le hi h
  simp only [BitVec.getLsbD_setWidth, BitVec.getLsbD_signExtend, hi, hib, decide_true, Bool.true_and]

def resultKind (first : Kind) (a c : Nat) : Kind := if c < a then .trunc else first

def Compatible (first : Kind) (a b c : Nat) : Prop :=
  (first = .zero ∨ first = .sign) ∧ a < b ∧ c < b ∧ c ≠ a
instance (first : Kind) (a b c : Nat) : Decidable (Compatible first a b c) := by
  unfold Compatible; infer_instance

/-- Both poison and every actual same-width initialized scalar retain their exact final value. -/
theorem meaning {first : Kind} {a b c ib : Nat} {value : Val}
    (h : Compatible first a b c) (ht : Val.HasTy ib (.int a) value) :
    Kind.trunc.apply c (first.apply b value) = (resultKind first a c).apply c value := by
  rcases hasTy_intWidth (ib := ib) (t := .int a) rfl ht with rfl | ⟨x,rfl⟩
  · cases resultKind first a c <;> cases first <;> rfl
  · obtain ⟨hkind,_,hcb,_⟩ := h
    rcases hkind with rfl | rfl
    · by_cases hc : c < a <;>
        simp [Kind.apply, resultKind, hc, BitVec.setWidth_setWidth_of_le x (Nat.le_of_lt hcb)]
    · by_cases hc : c < a
      · simp [Kind.apply, resultKind, hc, signed_truncate x (Nat.le_of_lt hcb),
          BitVec.signExtend_eq_setWidth_of_le x (Nat.le_of_lt hc)]
      · simp [Kind.apply, resultKind, hc, signed_truncate x (Nat.le_of_lt hcb)]

def combined (first last : Fact) : Fact :=
  { last with
    root := first.root
    inputWidth := first.inputWidth
    kind := resultKind first.kind first.inputWidth last.outputWidth }

def direct (s : Op) (first last : Fact) : Op :=
  .mk (combined first last).kind.name s.results [first.root] s.attrs s.props s.regions s.successors

def Checks (Γ : TEnv) (facts : List Fact) (s : Op) (first last : Fact) : Prop :=
  Legal Γ s last ∧ first ∈ facts ∧ first.result.id = last.root ∧
  first.outputWidth = last.inputWidth ∧ last.kind = .trunc ∧
  Compatible first.kind first.inputWidth first.outputWidth last.outputWidth ∧
  Legal Γ (direct s first last) (combined first last)
instance (Γ : TEnv) (facts : List Fact) (s : Op) (first last : Fact) :
    Decidable (Checks Γ facts s first last) := by unfold Checks; infer_instance

/-- The checked original and direct casts return the same binding in the entire environment.
Origin facts carry runtime values and widths; annotations alone never justify composition. -/
theorem replacement_sound {ib : Nat} {Γ : TEnv} {facts : List Fact} {s : Op} {first last : Fact}
    (h : Checks Γ facts s first last) {ctx : Ctx} {call : CallSem} {env : Env}
    (he : EnvTyped ib Γ env) (hf : BindingFacts.Valid model facts env) :
    stepOp opSem ctx call env (direct s first last) = stepOp opSem ctx call env s := by
  obtain ⟨hs,hm,hid,hw,htrunc,hcompat,hdirect⟩ := h
  obtain ⟨value,hvalue,hroot,hresult⟩ := hf first hm
  have hlegal := hs
  obtain ⟨_,_,_,_,_,_,hinput,_,_,_,_,_⟩ := hlegal
  obtain ⟨actual,ha,hty⟩ := he _ _ hinput
  have hav : actual = .scalar (first.kind.apply first.outputWidth value) := by
    rw [← hid,hresult] at ha
    exact (Option.some.inj ha).symm
  have htyped : Val.HasTy ctx.indexBits (.int last.inputWidth)
      (first.kind.apply first.outputWidth value) := by
    exact IntegerCast.int_type_change (by simpa [hav,RVal.HasTy] using hty)
  rw [IntegerCast.cast_exec hdirect ctx call env hroot (IntegerCast.int_type_change hvalue),
    IntegerCast.cast_exec hs ctx call env (by simpa [hid] using hresult) htyped]
  simp only [combined]
  rw [htrunc,meaning hcompat hvalue]

def proposal (Γ : TEnv) (facts : List Fact) (s : Op) : Option (Fact × Fact) := do
  let last ← IntegerCast.read Γ s
  let first ← facts.find? fun f => f.result.id == last.root &&
    f.outputWidth == last.inputWidth && last.kind == .trunc &&
    decide (Compatible f.kind f.inputWidth f.outputWidth last.outputWidth)
  pure (first,last)

def rewrite (Γ : TEnv) (facts : List Fact) (s : Op) : Op :=
  match proposal Γ facts s with
  | some (first,last) => if Checks Γ facts s first last then direct s first last else s
  | none => s

theorem rewrite_spec (Γ : TEnv) (facts : List Fact) (s : Op) :
    (rewrite Γ facts s).isTerminator = s.isTerminator ∧
    (rewrite Γ facts s).results = s.results ∧
    ∀ ib ctx call env, EnvTyped ib Γ env → BindingFacts.Valid model facts env →
      stepOp opSem ctx call env (rewrite Γ facts s) = stepOp opSem ctx call env s := by
  unfold rewrite
  split
  · next first last _ =>
    split
    · next hc =>
      have hn : s.isTerminator = false := by
        have hchecks := hc
        obtain ⟨hsource,_,_,_,htrunc,_,_⟩ := hchecks
        obtain ⟨_,hname,_,_,_,_,_,_,_,_,_,_⟩ := hsource
        rw [htrunc] at hname
        simp [Op.isTerminator,terminatorNames,hname,Kind.name]
      have hd : (direct s first last).isTerminator = false := by
        cases hk : (combined first last).kind <;>
          simp [direct,Op.isTerminator,terminatorNames,Kind.name,hk,Op.name]
      exact ⟨hd.trans hn.symm,rfl,fun _ _ _ _ he hf => replacement_sound hc he hf⟩
    · exact ⟨rfl,rfl,fun _ _ _ _ _ _ => rfl⟩
  · exact ⟨rfl,rfl,fun _ _ _ _ _ _ => rfl⟩

/-- Reuse the existing cast cancellation/dead-binding family in the same tail walk. Partial
rewrites retain all result IDs and every escaped intermediate; captured origin facts start empty. -/
def family : TailNormalize.Family Fact model where
  advance := CastCancel.family.advance
  advance_sound := CastCancel.family.advance_sound
  finish _ Γ facts _ s tail :=
    CastCancel.finish Γ facts (rewrite Γ facts (IntegerCastHeader.operation Γ s)) tail
  finish_sound ib Γ facts _ s tail ctx call env ht he hf _ := by
    have header := IntegerCastHeader.operation_spec Γ s
    have spec := rewrite_spec Γ facts (IntegerCastHeader.operation Γ s)
    have hterm := spec.1.trans (header.1.trans ht)
    refine (CastCancel.finish_sound hterm he hf).trans ?_
    rw [evalOps_cons_step _ _ _ _ _ _ hterm,
      evalOps_cons_step _ _ _ _ _ _ ht, spec.2.2 ib ctx call env he hf,
      header.2.2 ib ctx call env he]

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

def base (ib : Nat) : SplatNormalizer.Normalizer ib := TailNormalize.base family ib []

end MerlinVerify.Rules.IntegerCastPartial
