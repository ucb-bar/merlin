/- Exact scalar widening/truncation round trips. Actual successful source prefixes establish
the origin value; checked continuation aliases preserve poison, captures and escaping values. -/
import MerlinVerify.Rules.IntegerCastDead
import MerlinVerify.Check.TypedAlias
import MerlinVerify.Check.TailNormalize
import MerlinVerify.Check.BodyNormalize
import MerlinVerify.Check.RegionDepth
import MerlinVerify.Check.TypedNormalizeC1

namespace MerlinVerify.Rules.CastCancel

open MerlinVerify MerlinVerify.Check MerlinVerify.Sem.Ops IntegerCast

/-- Narrowing a genuine zero/sign extension back to its actual root width restores all bits,
including the zero-width mathematical boundary and poison. Executable candidates require legal
strictly increasing MLIR integer widths separately. -/
theorem roundtrip {kind : Kind} {a b ib : Nat} {value : Val}
    (hk : kind = .zero ∨ kind = .sign) (hw : a ≤ b)
    (ht : Val.HasTy ib (.int a) value) :
    Kind.trunc.apply a (kind.apply b value) = value := by
  rcases hasTy_intWidth (ib := ib) (t := .int a) rfl ht with rfl | ⟨x, rfl⟩
  · rfl
  · rcases hk with rfl | rfl
    · simp [Kind.apply, BitVec.setWidth_setWidth_of_le x hw]
    · have hs : (x.signExtend b).setWidth a = x := by
        ext i hi
        simp only [BitVec.getElem_setWidth]
        rw [BitVec.getLsbD_signExtend]
        simp [hi, show i < b by omega]
      simp [Kind.apply, hs]

/-- This family deliberately excludes every arithmetic modifier, including modifiers ignored
by a particular cast's frozen semantics. Unknown attributes are checked by IntegerCast.read. -/
def admissible (s : Op) : Bool :=
  Arith.fastmathKey s.blind == "0" && overflowFlags s == [] &&
    !hasUnit s "nonNeg" && !hasUnit s "isExact" && !hasUnit s "disjoint"

def advance (Γ : TEnv) (facts : List Fact) (s : Op) : List Fact :=
  if admissible s then IntegerCast.advance Γ facts s
  else BindingFacts.forget model (s.results.map (·.id)) facts

theorem advance_sound {ib : Nat} {Γ : TEnv} {facts : List Fact} {s : Op}
    {ctx : Ctx} {call : CallSem} {env env' : Env}
    (he : EnvTyped ib Γ env) (hf : BindingFacts.Valid model facts env)
    (hi : ctx.indexBits = ib) (hs : stepOp opSem ctx call env s = .ok env') :
    BindingFacts.Valid model (advance Γ facts s) env' := by
  unfold advance
  split
  · exact IntegerCast.advance_sound he hf hi hs
  · exact BindingFacts.forget_step model hf hs

def Matches (Γ : TEnv) (first last : Fact) : Prop :=
  first.result.id = last.root ∧ first.outputWidth = last.inputWidth ∧
  last.outputWidth = first.inputWidth ∧ (first.kind = .zero ∨ first.kind = .sign) ∧
  first.inputWidth < first.outputWidth ∧ last.kind = .trunc ∧
  Γ.lookup first.root = some (.scalar (.int first.inputWidth)) ∧
  freshIds (SplatNormalize.diagonal Γ) [last.result.id] = true

instance (Γ : TEnv) (first last : Fact) : Decidable (Matches Γ first last) := by
  unfold Matches; infer_instance

def eliminate (Γ : TEnv) (facts : List Fact) (s : Op) (tail : List Op) : Option (List Op) := do
  if !admissible s then none else do
    let last ← IntegerCast.read Γ s
    let first ← facts.find? fun f => decide (Matches Γ f last)
    TypedAlias.rename? Γ last.result.id first.root tail

theorem eliminate_sound {ib : Nat} {Γ : TEnv} {facts : List Fact}
    {s : Op} {tail renamed : List Op} (h : eliminate Γ facts s tail = some renamed)
    {ctx : Ctx} {call : CallSem} {env : Env}
    (he : EnvTyped ib Γ env) (hf : BindingFacts.Valid model facts env) :
    ∃ env', stepOp opSem ctx call env s = .ok env' ∧
      evalOps opSem ctx call env renamed = evalOps opSem ctx call env' tail := by
  unfold eliminate at h
  split at h
  · cases h
  · cases hl : IntegerCast.read Γ s with
    | none => simp [hl] at h
    | some last =>
      simp only [hl, bind, Option.bind] at h
      cases hm : facts.find? (fun f => decide (Matches Γ f last)) with
      | none => simp [hm] at h
      | some first =>
        simp only [hm] at h
        have hm' : decide (Matches Γ first last) = true :=
          List.find?_some (p := fun f : Fact => decide (Matches Γ f last)) hm
        have hc : Matches Γ first last := of_decide_eq_true hm'
        obtain ⟨hid, hw, hout, hk, hwiden, hlast, _, _⟩ := hc
        obtain ⟨value, hvalue, hroot, hresult⟩ := hf first (List.mem_of_find?_eq_some hm)
        have hs := IntegerCast.read_legal hl
        obtain ⟨actual, ha, hty⟩ := he _ _ hs.2.2.2.2.2.2.1
        have hav : actual = .scalar (first.kind.apply first.outputWidth value) := by
          rw [← hid, hresult] at ha
          exact (Option.some.inj ha).symm
        have hty' : Val.HasTy ctx.indexBits (.int last.inputWidth)
            (first.kind.apply first.outputWidth value) := by
          have ht : Val.HasTy ib (.int last.inputWidth)
              (first.kind.apply first.outputWidth value) := by simpa [hav, RVal.HasTy] using hty
          exact int_type_change ht
        have hstep := cast_exec hs ctx call env (by simpa [hid] using hresult) hty'
        have hr : last.kind.apply last.outputWidth (first.kind.apply first.outputWidth value) = value := by
          rw [hlast, hout]
          exact roundtrip hk (Nat.le_of_lt hwiden) hvalue
        rw [hr] at hstep
        exact ⟨_, hstep, TypedAlias.rename_sound h hroot⟩

def drop? (Γ : TEnv) (s : Op) (tail : List Op) : Bool :=
  admissible s && IntegerCastDead.drop? Γ s tail

theorem drop_sound {ib : Nat} {Γ : TEnv} {s : Op} {tail : List Op}
    (h : drop? Γ s tail = true) {env : Env} (he : EnvTyped ib Γ env)
    (ctx : Ctx) (call : CallSem) :
    ∃ env', stepOp opSem ctx call env s = .ok env' ∧
      evalOps opSem ctx call env' tail = evalOps opSem ctx call env tail := by
  simp only [drop?, Bool.and_eq_true] at h
  exact IntegerCastDead.drop_sound h.2 he ctx call

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
  advance_sound _ _ _ _ _ _ _ _ he hf hi hs := advance_sound he hf hi hs
  finish _ Γ facts _ s tail := finish Γ facts s tail
  finish_sound _ _ _ _ _ _ _ _ _ ht he hf _ := finish_sound ht he hf

def normalize (ib : Nat) (Γ : TEnv) (facts : List Fact) : List Op → Option (List Op) :=
  TailNormalize.normalize family ib Γ facts []

/-- The actual typed input domain is essential. No arbitrary runtime value or tensor value is
identified with a scalar merely because its source annotation claims an integer type. -/
theorem normalize_sound {ib : Nat} (R : Rulebook opSem ib) (d : Nat) :
    ∀ Γ facts ss ns yields ctx call env,
      checkOps R d Γ ss yields = true → normalize ib Γ facts ss = some ns →
      EnvTyped ib Γ env → BindingFacts.Valid model facts env →
      ctx.indexBits = ib → d ≤ ctx.loopIndex.length →
      evalOps opSem ctx call env ns = evalOps opSem ctx call env ss := by
  intro Γ facts ss ns yields ctx call env hc hn he hf hi hd
  exact TailNormalize.normalize_sound family R d Γ facts [] ss ns yields ctx call env hc hn he hf hi hd

/-- Every nested block starts without origin facts. The structured walker supplies the actual
local runtime typing, while alpha validation handles aliases captured by later nested regions. -/
def base (ib : Nat) : SplatNormalizer.Normalizer ib := TailNormalize.base family ib []

def normalizer : TypedNormalizer where
  run ib d Γ ops hints yields :=
    (BodyNormalize.recursive (base ib) (max (RegionDepth.ops ops) (RegionDepth.ops hints))).run
      d Γ [] ops yields
  sound ib d Γ ops hints normalized yields ctx call env hc hn he hi hd :=
    (BodyNormalize.recursive (base ib) (max (RegionDepth.ops ops) (RegionDepth.ops hints))).sound
      d Γ [] ops normalized yields ctx call env hc hn he (by intro f hf; cases hf) hi hd

def check (ib : Nat) (src tgt : Module) (fn : String) (argTys resTys : List VTy) : Bool :=
  TypedNormalizeC1.check normalizer ib src tgt fn argTys resTys

theorem check_sound {ib : Nat} {src tgt : Module} {fn : String} {argTys resTys : List VTy}
    (h : check ib src tgt fn argTys resTys = true) :
    FuncRefinesC opSem (fun ctx => ctx.indexBits = ib) (fun args => HasTys ib args argTys)
      tgt fn src fn := TypedNormalizeC1.check_sound normalizer h

end MerlinVerify.Rules.CastCancel
