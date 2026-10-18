/- Exact scalar selection from a preceding tensor insertion at proved closed coordinates.
The shared state traversal retains escaping insertions and starts each block without facts. -/
import MerlinVerify.Check.InsertExtractFacts
import MerlinVerify.Check.ConstantAlias

namespace MerlinVerify.Check.InsertExtract

open MerlinVerify MerlinVerify.Sem.Ops

structure Facts where
  points : List InsertExtractFacts.Fact := []
  constants : List ConstantFacts.Fact := []

def Valid (facts : Facts) (env : Env) : Prop :=
  BindingFacts.Valid InsertExtractFacts.model facts.points env ∧
    BindingFacts.Valid ConstantFacts.model facts.constants env

def advance (ib : Nat) (Γ : TEnv) (facts : Facts) (s : Op) : Facts :=
  ⟨InsertExtractFacts.advance Γ facts.constants facts.points s,
    Rules.IndexFold.advance ib facts.constants s⟩

theorem advance_sound {ib : Nat} {Γ : TEnv} {facts : Facts} {s : Op} {ctx : Ctx} {call : CallSem}
    {env env' : Env} (he : EnvTyped ib Γ env) (hf : Valid facts env) (hi : ctx.indexBits = ib)
    (hs : stepOp opSem ctx call env s = .ok env') : Valid (advance ib Γ facts s) env' := by
  exact ⟨InsertExtractFacts.advance_sound he hf.2 hf.1 hs, Rules.IndexFold.advance_sound hf.2 hi hs⟩

private def valueDecEq (a b : Value) : Decidable (a = b) :=
  decidable_of_iff (a.id = b.id ∧ a.ty = b.ty) (by cases a; cases b; simp)
local instance : DecidableEq Value := valueDecEq

def Legal (Γ : TEnv) (s : Op) (f : InsertExtractFacts.Fact) (indices : List ValueId) (r : Value) : Prop :=
  s.name = "tensor.extract" ∧ unmodelledAttr? s.blind = none ∧ s.regions = [] ∧ s.successors = [] ∧
  s.operands = f.tensor :: indices ∧ s.results = [r] ∧ r.ty = f.element ∧
  Γ.lookup f.scalar = some (.scalar f.element) ∧ Γ.lookup f.tensor = some (.tensor f.shape f.element) ∧
  indices.mapM (Γ.lookup ·) = some (List.replicate f.shape.length (.scalar .index))

instance (Γ : TEnv) (s : Op) (f : InsertExtractFacts.Fact) (indices : List ValueId) (r : Value) :
    Decidable (Legal Γ s f indices r) := by unfold Legal; infer_instance

def eliminate (Γ : TEnv) (facts : Facts) (s : Op) (tail : List Op) : Option (List Op) := do
  if s.name == "tensor.extract" then
    let [r] := s.results | none
    let tensor :: indices := s.operands | none
    let coords ← InsertExtractFacts.positions facts.constants indices
    let f ← facts.points.find? fun f =>
      f.tensor == tensor && f.coords == coords && decide (Legal Γ s f indices r)
    TypedAlias.rename? Γ r.id f.scalar tail
  else none

theorem eliminate_sound {Γ : TEnv} {facts : Facts} {s : Op} {tail renamed : List Op}
    (h : eliminate Γ facts s tail = some renamed) {ctx : Ctx} {call : CallSem} {env : Env}
    (hf : Valid facts env) :
    ∃ env', stepOp opSem ctx call env s = .ok env' ∧
      evalOps opSem ctx call env renamed = evalOps opSem ctx call env' tail := by
  unfold eliminate at h
  split at h
  · split at h
    · next r hr =>
      split at h
      · next tensor indices _ =>
        cases hc : InsertExtractFacts.positions facts.constants indices with
        | none => simp [hc] at h
        | some coords =>
          simp only [hc, bind, Option.bind] at h
          cases hp : facts.points.find? (fun f =>
            f.tensor == tensor && f.coords == coords && decide (Legal Γ s f indices r)) with
          | none => simp [hp] at h
          | some f =>
            simp only [hp] at h
            have checks := List.find?_some hp
            simp only [Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at checks
            obtain ⟨⟨_, hcoords⟩, hn, hm, hregs, _, ho, _, _, _, _, _⟩ := checks
            obtain ⟨t, value, ht, hv, hshape, _, hlen, hb, hpoint⟩ := hf.1 f (List.mem_of_find?_eq_some hp)
            obtain ⟨args, hi, hic⟩ := InsertExtractFacts.positions_sound hc hf.2
            rw [← hcoords] at hic
            have ha : env.getAll s.operands = .ok (.tensor t :: args) := by
              simp only [ho, Env.getAll, Outcome.mapM, ht, Outcome.ofOption, Outcome.bind_ok]
              change (env.getAll indices).bind (fun rest => .ok (.tensor t :: rest)) = _
              rw [hi]; rfl
            have hs : stepOp opSem ctx call env s = .ok ((r.id, .scalar value) :: env) := by
              unfold stepOp
              rw [ha, hregs]
              simp only [Outcome.bind_ok, evalRegions,
                Rules.InsertExtract.extract_sem hn hm hic (hshape.symm ▸ hlen) (hshape.symm ▸ hb) ctx call,
                hpoint]
              simp [Env.bindAll, hr]
            exact ⟨_, hs, TypedAlias.rename_sound h hv⟩
      · cases h
    · cases h
  · cases h

def drop? (Γ : TEnv) (facts : Facts) (s : Op) (tail : List Op) : Bool :=
  (InsertExtractFacts.checked Γ facts.constants s).isSome && TypedDead.unused Γ s tail

theorem drop_sound {ib : Nat} {Γ : TEnv} {facts : Facts} {s : Op} {tail : List Op}
    (h : drop? Γ facts s tail = true) {ctx : Ctx} {call : CallSem} {env : Env}
    (he : EnvTyped ib Γ env) (hf : Valid facts env) :
    ∃ env', stepOp opSem ctx call env s = .ok env' ∧
      evalOps opSem ctx call env' tail = evalOps opSem ctx call env tail := by
  simp only [drop?, Bool.and_eq_true] at h
  obtain ⟨p, hp⟩ := Option.isSome_iff_exists.1 h.1
  have hc := InsertExtractFacts.checked_spec hp
  obtain ⟨t, value, _, _, _, hs⟩ := InsertExtractFacts.write_exec hc he hf.2 ctx call
  exact TypedDead.drop_sound h.2 ⟨p.result, _, hc.2.2.2.2.2.1, hs⟩

def finish (ib : Nat) (Γ : TEnv) (facts : Facts) (s : Op) (tail : List Op) : List Op :=
  if drop? Γ facts s tail then tail else
    (eliminate Γ facts s tail).getD (ConstantAlias.finish ib Γ facts.constants s tail)

theorem finish_sound {ib : Nat} {Γ : TEnv} {facts : Facts} {s : Op} {tail : List Op}
    {ctx : Ctx} {call : CallSem} {env : Env} (ht : s.isTerminator = false)
    (he : EnvTyped ib Γ env) (hf : Valid facts env) (hi : ctx.indexBits = ib) :
    evalOps opSem ctx call env (finish ib Γ facts s tail) = evalOps opSem ctx call env (s :: tail) := by
  unfold finish
  split
  · next hdrop =>
    obtain ⟨env', hs, heq⟩ := drop_sound hdrop he hf (ctx := ctx) (call := call)
    rw [evalOps_cons_step _ _ _ _ _ _ ht, hs, Outcome.bind_ok]
    exact heq.symm
  · cases helim : eliminate Γ facts s tail with
    | none => exact ConstantAlias.finish_sound ht hf.2 hi
    | some renamed =>
      simp only [Option.getD_some]
      obtain ⟨env', hs, heq⟩ := eliminate_sound helim hf (ctx := ctx) (call := call)
      rw [evalOps_cons_step _ _ _ _ _ _ ht, hs, Outcome.bind_ok]
      exact heq

def family : StateNormalize.Family Facts Valid where
  initial := ⟨[], []⟩
  initial_valid _ := by constructor <;> intro f hf <;> cases hf
  advance := advance
  advance_sound _ _ _ _ _ _ _ _ he hf hi hs := advance_sound he hf hi hs
  finish ib Γ facts _ s tail := finish ib Γ facts s tail
  finish_sound _ _ _ _ _ _ _ _ _ ht he hf hi := finish_sound ht he hf hi

def normalize (ib : Nat) (Γ : TEnv) (facts : Facts) : List Op → Option (List Op) :=
  StateNormalize.normalize family ib Γ facts []

theorem normalize_sound {ib : Nat} (R : Rulebook opSem ib) (d : Nat) :
    ∀ Γ facts ss ns yields ctx call env,
      checkOps R d Γ ss yields = true → normalize ib Γ facts ss = some ns →
      EnvTyped ib Γ env → Valid facts env → ctx.indexBits = ib → d ≤ ctx.loopIndex.length →
      evalOps opSem ctx call env ns = evalOps opSem ctx call env ss := by
  intro Γ facts ss ns yields ctx call env hc hn he hf hi hd
  exact StateNormalize.normalize_sound family R d Γ facts [] ss ns yields ctx call env hc hn he hf hi hd

def base (ib : Nat) : SplatNormalizer.Normalizer ib := StateNormalize.base family ib []

end MerlinVerify.Check.InsertExtract
