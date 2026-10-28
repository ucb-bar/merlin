/- Closed scalar index comparisons replay actual current bits at the checked index width.
The typing environment limits applicability; exact runtime facts justify the computation. -/
import MerlinVerify.Rules.IndexFold

namespace MerlinVerify.Rules.IndexCompare

open MerlinVerify MerlinVerify.Check MerlinVerify.Sem.Ops

def boolResult (s : Op) : Bool :=
  match s.results with
  | [{ ty := .int 1, .. }] => true
  | _ => false

def indexOperands (Γ : TEnv) (s : Op) : Bool :=
  match s.operands with
  | [x,y] => Γ.lookup x == some (.scalar .index) && Γ.lookup y == some (.scalar .index)
  | _ => false

def Eligible (Γ : TEnv) (s : Op) : Prop :=
  s.name = "arith.cmpi" ∧ dialectOf s.name = "arith" ∧ s.isTerminator = false ∧
    s.regions = [] ∧ s.successors = [] ∧ unmodelledAttr? s.blind = none ∧
    boolResult s = true ∧ indexOperands Γ s = true ∧
    (attrInt? s.blind "predicate" >>= Arith.cmpiPred).isSome = true ∧
    overflowFlags s = [] ∧ Arith.fastmathKey s = "0" ∧ hasUnit s "nonNeg" = false ∧
    hasUnit s "isExact" = false ∧ hasUnit s "disjoint" = false

instance (Γ : TEnv) (s : Op) : Decidable (Eligible Γ s) := by unfold Eligible; infer_instance

theorem result_one {s : Op} (h : boolResult s = true) : ∃ r, s.results = [r] := by
  unfold boolResult at h
  split at h
  · next id hr => exact ⟨⟨id,.int 1⟩,hr⟩
  · cases h

theorem context_independent {ib : Nat} {Γ : TEnv} {s : Op} (h : Eligible Γ s)
    (ctx : Ctx) (call : CallSem) (args : List RVal) :
    opSem ctx call s args [] = opSem (ConstantFacts.reference ib) (ConstantFacts.reference ib).ext s args [] := by
  obtain ⟨hn,hd,_,_,_,hm,_⟩ := h
  rw [opSem_modelled hm, opSem_modelled hm]
  simp only [dispatch, Op.blind_name, hd]
  simp [Arith.sem, Op.blind_name, hn]

def value? (ib : Nat) (Γ : TEnv) (facts : List ConstantFacts.Fact) (s : Op) : Option RVal := do
  if Eligible Γ s then
    let args ← IndexFold.arguments facts s.operands
    match opSem (ConstantFacts.reference ib) (ConstantFacts.reference ib).ext s args [] with
    | .ok [value] => some value
    | _ => none
  else none

theorem value_eligible {ib : Nat} {Γ : TEnv} {facts : List ConstantFacts.Fact} {s : Op} {value : RVal}
    (h : value? ib Γ facts s = some value) : Eligible Γ s := by
  unfold value? at h
  split at h
  · assumption
  · cases h

theorem value_exec {ib : Nat} {Γ : TEnv} {facts : List ConstantFacts.Fact} {s : Op} {value : RVal}
    (h : value? ib Γ facts s = some value) {env : Env}
    (hf : BindingFacts.Valid ConstantFacts.model facts env) {ctx : Ctx} (call : CallSem) :
    Eligible Γ s ∧ ∃ r, s.results = [r] ∧ stepOp opSem ctx call env s = .ok ((r.id,value) :: env) := by
  unfold value? at h
  split at h
  · next he =>
    cases ha : IndexFold.arguments facts s.operands with
    | none => simp [ha] at h
    | some args =>
      simp only [ha, bind, Option.bind] at h
      split at h
      · next v hv =>
        cases h
        have legal := he
        obtain ⟨_,_,_,hregions,_,_,hresult,_⟩ := he
        obtain ⟨r,hr⟩ := result_one hresult
        refine ⟨legal,r,hr,?_⟩
        unfold stepOp
        rw [hregions, IndexFold.arguments_sound hf _ _ ha]
        simp only [Outcome.bind_ok, evalRegions,
          context_independent (ib := ib) legal ctx call args, hv]
        simp [Env.bindAll, hr]
      · cases h
  · cases h

def replacement? (ib : Nat) (Γ : TEnv) (facts : List ConstantFacts.Fact) (hints : List Op) (s : Op) : Option Op := do
  let .scalar value ← value? ib Γ facts s | none
  IndexFold.tryHints ib s value hints

theorem replacement_spec {ib : Nat} {Γ : TEnv} {facts : List ConstantFacts.Fact}
    {hints : List Op} {s v : Op} (h : replacement? ib Γ facts hints s = some v) :
    v.results = s.results ∧ ∃ value, value? ib Γ facts s = some (.scalar value) ∧
      ConstantFacts.value? ib v = some (.scalar value) := by
  unfold replacement? at h
  cases hs : value? ib Γ facts s with
  | none => simp [hs] at h
  | some result =>
    simp only [hs, bind, Option.bind] at h
    cases result with
    | tensor _ => cases h
    | scalar value =>
      obtain ⟨hr,hv⟩ := IndexFold.tryHints_spec ib s value hints v h
      exact ⟨hr,value,rfl,hv⟩

theorem replacement_sound {ib : Nat} {Γ : TEnv} {facts : List ConstantFacts.Fact}
    {hints : List Op} {s v : Op} (h : replacement? ib Γ facts hints s = some v) :
    v.isTerminator = s.isTerminator ∧ v.results = s.results ∧
    ∀ ctx call env, BindingFacts.Valid ConstantFacts.model facts env → ctx.indexBits = ib →
      stepOp opSem ctx call env v = stepOp opSem ctx call env s := by
  obtain ⟨hr,value,hs,hv⟩ := replacement_spec h
  have hconstant := (ConstantFacts.value_spec hv).1
  obtain ⟨_,_,hterm,_⟩ := value_eligible hs
  refine ⟨?_,hr,?_⟩
  · have htarget : v.isTerminator = false := by simp [Op.isTerminator, terminatorNames, hconstant.1]
    exact htarget.trans hterm.symm
  · intro ctx call env hf hi
    obtain ⟨_,r,hrs,hstep⟩ := value_exec hs hf (ctx := ctx) call
    obtain ⟨r',hrv,htarget⟩ := ConstantFacts.value_exec hv hi call env
    have : r' = r := by simpa [hr,hrs] using hrv.symm
    subst r'
    exact htarget.trans hstep.symm

/-- Compare results do not enter the index facts. The six-op arithmetic learner remains unchanged. -/
def family : FactNormalize.Family ConstantFacts.Fact ConstantFacts.model where
  rewrite ib Γ facts hints s := (replacement? ib Γ facts hints s).getD (IndexFold.rewrite ib facts hints s)
  rewrite_sound ib Γ facts hints s := by
    cases h : replacement? ib Γ facts hints s with
    | some v =>
      obtain ⟨ht,hr,hs⟩ := replacement_sound h
      exact ⟨ht,hr,fun ctx call env _ hf hi => hs ctx call env hf hi⟩
    | none =>
      simpa only [IndexFold.family, h, Option.getD_none] using IndexFold.family.rewrite_sound ib Γ facts hints s
  advance ib _ facts s := IndexFold.advance ib facts s
  advance_sound _ _ _ _ _ _ _ _ _ hf hi hs := IndexFold.advance_sound hf hi hs

def normalizer : TypedNormalizer where
  run ib d Γ ss hints yields :=
    (BodyNormalize.recursive (FactNormalize.base family ib (DenseExtract.hintOps hints))
      (max (RegionDepth.ops ss) (RegionDepth.ops hints))).run d Γ [] ss yields
  sound ib d Γ ss hints ns yields ctx call env hc hn he hi hd :=
    (BodyNormalize.recursive (FactNormalize.base family ib (DenseExtract.hintOps hints))
      (max (RegionDepth.ops ss) (RegionDepth.ops hints))).sound d Γ [] ss ns yields ctx call env
      hc hn he (by intro f hf; cases hf) hi hd

end MerlinVerify.Rules.IndexCompare
