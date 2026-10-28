/- Rank-one scalar selection joins actual constructor and closed-index binding facts. Every
alias validates its complete continuation; empty-tensor, negative and out-of-bounds reads refuse. -/
import MerlinVerify.Check.FromElementsFacts
import MerlinVerify.Check.ConstantFacts
import MerlinVerify.Rules.IndexFold
import MerlinVerify.Check.TypedAlias
import MerlinVerify.Check.TypedDead
import MerlinVerify.Check.BodyNormalize
import MerlinVerify.Check.StateNormalize
import MerlinVerify.Check.RegionDepth
import MerlinVerify.Check.TypedNormalizeC1

namespace MerlinVerify.Check.FromElementsSelect

open MerlinVerify MerlinVerify.Sem.Ops

structure Facts where
  elements : List FromElementsFacts.Fact := []
  constants : List ConstantFacts.Fact := []

def Valid (facts : Facts) (env : Env) : Prop :=
  BindingFacts.Valid FromElementsFacts.model facts.elements env ∧
    BindingFacts.Valid ConstantFacts.model facts.constants env

def indexProducer (s : Op) : Bool :=
  s.name == "arith.constant" && match s.results with
  | [r] => r.ty == .index
  | _ => false

/-- Reuse exact successful-source index replay. Intermediate values need no materialization
hints, and arbitrary writes invalidate the same current-binding facts. -/
def advance (ib : Nat) (facts : Facts) (s : Op) : Facts :=
  ⟨FromElementsFacts.advance facts.elements s, Rules.IndexFold.advance ib facts.constants s⟩

theorem advance_sound {ib : Nat} {facts : Facts} {s : Op} {ctx : Ctx} {call : CallSem}
    {env env' : Env} (hf : Valid facts env) (hi : ctx.indexBits = ib)
    (hs : stepOp opSem ctx call env s = .ok env') : Valid (advance ib facts s) env' := by
  exact ⟨FromElementsFacts.advance_sound hf.1 hs, Rules.IndexFold.advance_sound hf.2 hi hs⟩

private def valueDecEq (a b : Value) : Decidable (a = b) :=
  decidable_of_iff (a.id = b.id ∧ a.ty = b.ty) (by cases a; cases b; simp)
local instance : DecidableEq Value := valueDecEq

def Legal (Γ : TEnv) (s : Op) (f : FromElementsFacts.Fact) (c : ConstantFacts.Fact) (r : Value) : Prop :=
  s.name = "tensor.extract" ∧ unmodelledAttr? s.blind = none ∧ s.regions = [] ∧ s.successors = [] ∧
  s.operands = [f.tensor, c.id] ∧ s.results = [r] ∧ r.ty = f.element ∧
  Γ.lookup f.scalar = some (.scalar f.element) ∧
  Γ.lookup f.tensor = some (.tensor [f.size] f.element) ∧ Γ.lookup c.id = some (.scalar .index) ∧
  match c.value with
  | .scalar index => Sem.Ops.Tensor.indexVal? index = some (Int.ofNat f.coordinate)
  | _ => False

instance (Γ : TEnv) (s : Op) (f : FromElementsFacts.Fact) (c : ConstantFacts.Fact) (r : Value) :
    Decidable (Legal Γ s f c r) := by unfold Legal; split <;> infer_instance

def eliminate (Γ : TEnv) (facts : Facts) (s : Op) (tail : List Op) : Option (List Op) := do
  if s.name = "tensor.extract" then
    let [r] := s.results | none
    let [_, index] := s.operands | none
    let c ← facts.constants.find? (fun c => c.id == index)
    let f ← facts.elements.find? (fun f => decide (Legal Γ s f c r))
    TypedAlias.rename? Γ r.id f.scalar tail
  else none

theorem eliminate_sound {Γ : TEnv} {facts : Facts} {s : Op} {tail renamed : List Op}
    (h : eliminate Γ facts s tail = some renamed) {ctx : Ctx} {call : CallSem} {env : Env}
    (hf : Valid facts env) :
    s.isTerminator = false ∧ ∃ env', stepOp opSem ctx call env s = .ok env' ∧
      evalOps opSem ctx call env renamed = evalOps opSem ctx call env' tail := by
  unfold eliminate at h
  split at h
  · next hn =>
    split at h
    · next r hr =>
      split at h
      · next tensorId indexId ho =>
        cases hc : facts.constants.find? (fun c => c.id == indexId) with
        | none => simp [hc] at h
        | some c =>
          simp only [hc, bind, Option.bind] at h
          cases he : facts.elements.find? (fun f => decide (Legal Γ s f c r)) with
          | none => simp [he] at h
          | some f =>
            simp only [he] at h
            have hl : Legal Γ s f c r := by simpa using List.find?_some he
            obtain ⟨_, hm, hregs, _, hop, _, _, _, _, _, hi⟩ := hl
            obtain ⟨t, value, ht, hp, hshape, _, hb, hv⟩ := hf.1 f (List.mem_of_find?_eq_some he)
            have hconstant := hf.2 c (List.mem_of_find?_eq_some hc)
            cases hvindex : c.value with
            | tensor _ => simp only [hvindex] at hi
            | scalar index =>
              simp only [hvindex] at hi
              have hindex : env.get? c.id = some (.scalar index) := by
                simpa [ConstantFacts.model, hvindex] using hconstant
              have hs : stepOp opSem ctx call env s = .ok ((r.id, .scalar value) :: env) := by
                unfold stepOp
                rw [hop, hregs]
                simp only [Env.getAll, Outcome.mapM, ht, hindex, Outcome.ofOption, Outcome.bind_ok,
                  evalRegions, Rules.FromElements.extract_sem hn hm hshape hb hi ctx call, hv]
                simp [Env.bindAll, hr]
              exact ⟨by simp [Op.isTerminator, terminatorNames, hn], _, hs, TypedAlias.rename_sound h hp⟩
      · cases h
    · cases h
  · cases h

/-- Runtime scalar inputs, including poison, make a well-formed constructor total. -/
def Total (Γ : TEnv) (s : Op) : Prop :=
  s.name = "tensor.from_elements" ∧ unmodelledAttr? s.blind = none ∧ s.regions = [] ∧ s.successors = [] ∧
  match s.results, Sem.Ops.Tensor.resultTensorTy s with
  | [_], some ([size], element) => s.operands.mapM (Γ.lookup ·) = some (List.replicate size (.scalar element))
  | _, _ => False

instance (Γ : TEnv) (s : Op) : Decidable (Total Γ s) := by unfold Total; split <;> infer_instance

theorem total_exec {ib : Nat} {Γ : TEnv} {s : Op} (h : Total Γ s) {env : Env}
    (he : EnvTyped ib Γ env) (ctx : Ctx) (call : CallSem) :
    ∃ r value, s.results = [r] ∧ stepOp opSem ctx call env s = .ok ((r.id, value) :: env) := by
  obtain ⟨hn, hm, hregs, _, h⟩ := h
  split at h
  · next r size element hr ht =>
    obtain ⟨args, hargs, htys⟩ := EnvTyped.getAll he _ _ h
    have hrep : List.replicate size (VTy.scalar element) =
        (List.replicate size element).map (liftTy none) := by simp [liftTy]
    rw [hrep] at htys
    obtain ⟨values, rfl, hvalues⟩ := hasTys_lift_scalar htys
    have hlen : values.length = size := by rw [List.Forall₂.length_eq hvalues, List.length_replicate]
    refine ⟨r, .tensor (Rules.FromElements.tensor element values), hr, ?_⟩
    unfold stepOp
    rw [hregs, hargs]
    simp [evalRegions, Rules.FromElements.constructor_sem hn hm ht values hlen ctx call, Env.bindAll, hr]
  · contradiction

def drop? (ib : Nat) (Γ : TEnv) (s : Op) (tail : List Op) : Bool :=
  TypedDead.unused Γ s tail &&
    (decide (Total Γ s) || (indexProducer s && (ConstantFacts.value? ib s).isSome))

theorem drop_sound {ib : Nat} {Γ : TEnv} {s : Op} {tail : List Op}
    (h : drop? ib Γ s tail = true) {ctx : Ctx} (hi : ctx.indexBits = ib) (call : CallSem) {env : Env}
    (he : EnvTyped ib Γ env) :
    ∃ env', stepOp opSem ctx call env s = .ok env' ∧
      evalOps opSem ctx call env' tail = evalOps opSem ctx call env tail := by
  simp only [drop?, Bool.and_eq_true, Bool.or_eq_true, decide_eq_true_eq] at h
  apply TypedDead.drop_sound h.1
  rcases h.2 with ht | hc
  · exact total_exec ht he ctx call
  · obtain ⟨value, hv⟩ := Option.isSome_iff_exists.1 hc.2
    obtain ⟨r, hr, hs⟩ := ConstantFacts.value_exec hv hi call env
    exact ⟨r, value, hr, hs⟩

def finish (ib : Nat) (Γ : TEnv) (facts : Facts) (s : Op) (tail : List Op) : List Op :=
  if drop? ib Γ s tail then tail else (eliminate Γ facts s tail).getD (s :: tail)

theorem finish_sound {ib : Nat} {Γ : TEnv} {facts : Facts} {s : Op} {tail : List Op}
    {ctx : Ctx} {call : CallSem} {env : Env} (ht : s.isTerminator = false)
    (he : EnvTyped ib Γ env) (hf : Valid facts env) (hi : ctx.indexBits = ib) :
    evalOps opSem ctx call env (finish ib Γ facts s tail) = evalOps opSem ctx call env (s :: tail) := by
  unfold finish
  split
  · next hdrop =>
    obtain ⟨env', hs, heq⟩ := drop_sound hdrop hi call he
    rw [evalOps_cons_step _ _ _ _ _ _ ht, hs, Outcome.bind_ok]
    exact heq.symm
  · cases helim : eliminate Γ facts s tail with
    | none => rfl
    | some renamed =>
      simp only [Option.getD_some]
      obtain ⟨_, env', hs, heq⟩ := eliminate_sound helim hf (ctx := ctx) (call := call)
      rw [evalOps_cons_step _ _ _ _ _ _ ht, hs, Outcome.bind_ok]
      exact heq

/-- The existing product fact state enters the common traversal directly. Neither component
contains entry facts, and successful original execution justifies both components after each step. -/
def family : StateNormalize.Family Facts Valid where
  initial := ⟨[], []⟩
  initial_valid _ := by constructor <;> intro f hf <;> cases hf
  advance ib _ facts s := advance ib facts s
  advance_sound _ _ _ _ _ _ _ _ _ hf hi hs := advance_sound hf hi hs
  finish ib Γ facts _ s tail := finish ib Γ facts s tail
  finish_sound _ _ _ _ _ _ _ _ _ ht he hf hi := finish_sound ht he hf hi

def normalize (ib : Nat) (Γ : TEnv) (facts : Facts) : List Op → Option (List Op) :=
  StateNormalize.normalize family ib Γ facts []

/-- Source progress establishes actual intermediate types, pointwise values and index facts.
Alias checks preserve the entire normalized continuation, including every poison and failure outcome. -/
theorem normalize_sound {ib : Nat} (R : Rulebook opSem ib) (d : Nat) :
    ∀ Γ facts ss ns yields ctx call env,
      checkOps R d Γ ss yields = true → normalize ib Γ facts ss = some ns →
      EnvTyped ib Γ env → Valid facts env →
      ctx.indexBits = ib → d ≤ ctx.loopIndex.length →
      evalOps opSem ctx call env ns = evalOps opSem ctx call env ss := by
  intro Γ facts ss ns yields ctx call env hc hn he hf hi hd
  exact StateNormalize.normalize_sound family R d Γ facts [] ss ns yields ctx call env hc hn he hf hi hd

/-- Local provenance starts empty at each block. Structured lifting supplies the actual typed
scope, while every scalar alias and deleted producer carries its independent equality proof. -/
def base (ib : Nat) : SplatNormalizer.Normalizer ib := StateNormalize.base family ib []

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

end MerlinVerify.Check.FromElementsSelect
