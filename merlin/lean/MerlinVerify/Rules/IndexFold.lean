/- Closed index arithmetic uses current exact runtime constant facts at the checked index width.
The target supplies materialization syntax only; every candidate is independently evaluated. -/
import MerlinVerify.Check.ConstantFacts
import MerlinVerify.Check.FactNormalize
import MerlinVerify.Check.DenseExtract

namespace MerlinVerify.Rules.IndexFold

open MerlinVerify MerlinVerify.Check MerlinVerify.Sem.Ops

def names : List String :=
  ["arith.addi", "arith.subi", "arith.muli", "arith.andi", "arith.ori", "arith.xori"]

def indexResult (s : Op) : Bool :=
  match s.results with
  | [{ ty := .index, .. }] => true
  | _ => false

def Eligible (s : Op) : Prop :=
  s.name ∈ names ∧ dialectOf s.name = "arith" ∧ s.isTerminator = false ∧
    s.regions = [] ∧ s.successors = [] ∧ unmodelledAttr? s.blind = none ∧ indexResult s = true

instance (s : Op) : Decidable (Eligible s) := by unfold Eligible; infer_instance

def producer (s : Op) : Bool := s.name == "arith.constant" && indexResult s

theorem result_one {s : Op} (h : indexResult s = true) : ∃ r, s.results = [r] := by
  unfold indexResult at h
  split at h
  · next id hr => exact ⟨⟨id,.index⟩,hr⟩
  · cases h

theorem context_independent {ib : Nat} {s : Op} (h : Eligible s)
    (ctx : Ctx) (call : CallSem) (args : List RVal) :
    opSem ctx call s args [] = opSem (ConstantFacts.reference ib) (ConstantFacts.reference ib).ext s args [] := by
  obtain ⟨hn, hd, _, _, _, hm, _⟩ := h
  rw [opSem_modelled hm, opSem_modelled hm]
  simp only [dispatch, Op.blind_name, hd]
  simp only [names, List.mem_cons, List.not_mem_nil, or_false] at hn
  rcases hn with hn | hn | hn | hn | hn | hn
  all_goals simp [Arith.sem, Op.blind_name, hn]

def lookup (facts : List ConstantFacts.Fact) (id : ValueId) : Option RVal :=
  (facts.find? (fun f => f.id == id)).map (·.value)

theorem lookup_sound {facts : List ConstantFacts.Fact} {id : ValueId} {value : RVal}
    (h : lookup facts id = some value) {env : Env}
    (hf : BindingFacts.Valid ConstantFacts.model facts env) : env.get? id = some value := by
  unfold lookup at h
  cases hm : facts.find? (fun f => f.id == id) with
  | none => simp [hm] at h
  | some f =>
    simp only [hm, Option.map_some, Option.some.injEq] at h
    have hid : f.id = id := by simpa using List.find?_some hm
    subst value
    simpa only [ConstantFacts.model, hid] using hf f (List.mem_of_find?_eq_some hm)

def arguments (facts : List ConstantFacts.Fact) (ids : List ValueId) : Option (List RVal) :=
  ids.mapM (lookup facts)

theorem arguments_sound {facts : List ConstantFacts.Fact} {env : Env}
    (hf : BindingFacts.Valid ConstantFacts.model facts env) :
    ∀ ids args, arguments facts ids = some args → env.getAll ids = .ok args
  | [], _, h => by simp [arguments] at h; cases h; rfl
  | x :: xs, _, h => by
    simp only [arguments, List.mapM_cons, bind, Option.bind] at h
    cases hx : lookup facts x with
    | none => simp [hx] at h
    | some value =>
      simp only [hx] at h
      cases ht : xs.mapM (lookup facts) with
      | none => simp [ht] at h
      | some tail =>
        simp only [ht, Option.pure_def, Option.some.injEq] at h
        subst h
        have hv := lookup_sound hx hf
        have ih := arguments_sound hf xs tail ht
        simp only [Env.getAll, Outcome.mapM, hv, Outcome.ofOption, Outcome.bind_ok]
        change (env.getAll xs).bind (fun rest => .ok (value :: rest)) = .ok (value :: tail)
        rw [ih]; rfl

/-- Replay source operations independently of all target hints. A successful intermediate result
is learned even when the final target has no matching constant for that intermediate. -/
def value? (ib : Nat) (facts : List ConstantFacts.Fact) (s : Op) : Option RVal := do
  if Eligible s then
    let args ← arguments facts s.operands
    match opSem (ConstantFacts.reference ib) (ConstantFacts.reference ib).ext s args [] with
    | .ok [value] => some value
    | _ => none
  else none

theorem value_eligible {ib : Nat} {facts : List ConstantFacts.Fact} {s : Op} {value : RVal}
    (h : value? ib facts s = some value) : Eligible s := by
  unfold value? at h
  split at h
  · assumption
  · cases h

theorem value_exec {ib : Nat} {facts : List ConstantFacts.Fact} {s : Op} {value : RVal}
    (h : value? ib facts s = some value) {env : Env}
    (hf : BindingFacts.Valid ConstantFacts.model facts env) {ctx : Ctx} (call : CallSem) :
    Eligible s ∧ ∃ r, s.results = [r] ∧ stepOp opSem ctx call env s = .ok ((r.id,value) :: env) := by
  unfold value? at h
  split at h
  · next he =>
    cases ha : arguments facts s.operands with
    | none => simp [ha] at h
    | some args =>
      simp only [ha, bind, Option.bind] at h
      split at h
      · next v hv =>
        cases h
        have legal := he
        obtain ⟨_,_,_,hregs,_,_,hindex⟩ := he
        obtain ⟨r, hr⟩ := result_one hindex
        refine ⟨legal,r,hr,?_⟩
        unfold stepOp
        rw [hregs, arguments_sound hf _ _ ha]
        simp only [Outcome.bind_ok, evalRegions,
          context_independent (ib := ib) legal ctx call args, hv]
        simp [Env.bindAll, hr]
      · cases h
  · cases h

def advance (ib : Nat) (facts : List ConstantFacts.Fact) (s : Op) : List ConstantFacts.Fact :=
  if producer s then ConstantFacts.advance ib facts s else
    let retained := BindingFacts.forget ConstantFacts.model (s.results.map (·.id)) facts
    match value? ib facts s, s.results with
    | some value, [r] => ⟨r.id,value⟩ :: retained
    | _, _ => retained

theorem advance_sound {ib : Nat} {facts : List ConstantFacts.Fact} {s : Op}
    {ctx : Ctx} {call : CallSem} {env env' : Env}
    (hf : BindingFacts.Valid ConstantFacts.model facts env) (hi : ctx.indexBits = ib)
    (hs : stepOp opSem ctx call env s = .ok env') :
    BindingFacts.Valid ConstantFacts.model (advance ib facts s) env' := by
  unfold advance
  split
  · exact ConstantFacts.advance_sound hf hi hs
  · have retained := BindingFacts.forget_step ConstantFacts.model hf hs
    split
    · next value r hv hr =>
      intro f hmem
      rcases List.mem_cons.1 hmem with rfl | hmem
      · obtain ⟨_,r',hr',hstep⟩ := value_exec hv hf (ctx := ctx) call
        have : r' = r := by simpa [hr] using hr'.symm
        subst r'
        rw [hstep] at hs
        cases hs
        simp [ConstantFacts.model, Env.get?, List.lookup]
      · exact retained f hmem
    · exact retained

def candidate (s hint : Op) : Op := .mk hint.name s.results [] hint.attrs hint.props [] []

def tryHints (ib : Nat) (s : Op) (value : Val) : List Op → Option Op
  | [] => none
  | hint :: hints =>
    if hint.results.map (·.ty) == s.results.map (·.ty) then
      let v := candidate s hint
      if DenseExtract.scalar? ib v = some value then some v else tryHints ib s value hints
    else tryHints ib s value hints

theorem tryHints_spec (ib : Nat) (s : Op) (value : Val) : ∀ hints v,
    tryHints ib s value hints = some v →
    v.results = s.results ∧ ConstantFacts.value? ib v = some (.scalar value)
  | [], _, h => by cases h
  | hint :: hints, v, h => by
    simp only [tryHints] at h
    split at h
    · split at h
      · next hv => cases h; exact ⟨rfl,DenseExtract.scalar_spec hv⟩
      · exact tryHints_spec ib s value hints v h
    · exact tryHints_spec ib s value hints v h

def replacement? (ib : Nat) (facts : List ConstantFacts.Fact) (hints : List Op) (s : Op) : Option Op := do
  let .scalar value ← value? ib facts s | none
  tryHints ib s value hints

theorem replacement_spec {ib : Nat} {facts : List ConstantFacts.Fact} {hints : List Op} {s v : Op}
    (h : replacement? ib facts hints s = some v) :
    v.results = s.results ∧ ∃ value, value? ib facts s = some (.scalar value) ∧
      ConstantFacts.value? ib v = some (.scalar value) := by
  unfold replacement? at h
  cases hs : value? ib facts s with
  | none => simp [hs] at h
  | some result =>
    simp only [hs, bind, Option.bind] at h
    cases result with
    | tensor _ => cases h
    | scalar value =>
      obtain ⟨hr,hv⟩ := tryHints_spec ib s value hints v h
      exact ⟨hr,value,rfl,hv⟩

/-- A candidate with different actual checked-width bits cannot be selected. -/
theorem refuses_wrong_value {ib : Nat} {s v : Op} {value : Val} {hints : List Op}
    (hwrong : ConstantFacts.value? ib v ≠ some (.scalar value)) : tryHints ib s value hints ≠ some v := by
  intro h
  exact hwrong (tryHints_spec ib s value hints v h).2

theorem replacement_sound {ib : Nat} {facts : List ConstantFacts.Fact} {hints : List Op} {s v : Op}
    (h : replacement? ib facts hints s = some v) :
    v.isTerminator = s.isTerminator ∧ v.results = s.results ∧
    ∀ ctx call env, BindingFacts.Valid ConstantFacts.model facts env → ctx.indexBits = ib →
      stepOp opSem ctx call env v = stepOp opSem ctx call env s := by
  obtain ⟨hr,value,hs,hv⟩ := replacement_spec h
  have he := (ConstantFacts.value_spec hv).1
  obtain ⟨_,_,hterm,_,_,_,_⟩ := value_eligible hs
  refine ⟨?_,hr,?_⟩
  · have htarget : v.isTerminator = false := by simp [Op.isTerminator, terminatorNames, he.1]
    exact htarget.trans hterm.symm
  · intro ctx call env hf hi
    obtain ⟨_,r,hrs,hstep⟩ := value_exec hs hf (ctx := ctx) call
    obtain ⟨r',hrv,htarget⟩ := ConstantFacts.value_exec hv hi call env
    have : r' = r := by simpa [hr,hrs] using hrv.symm
    subst r'
    exact htarget.trans hstep.symm

def rewrite (ib : Nat) (facts : List ConstantFacts.Fact) (hints : List Op) (s : Op) : Op :=
  (replacement? ib facts hints s).getD s

def family : FactNormalize.Family ConstantFacts.Fact ConstantFacts.model where
  rewrite ib _ facts hints s := rewrite ib facts hints s
  rewrite_sound ib Γ facts hints s := by
    unfold rewrite
    cases h : replacement? ib facts hints s with
    | none => exact ⟨rfl,rfl,fun _ _ _ _ _ _ => rfl⟩
    | some v =>
      obtain ⟨ht,hr,hs⟩ := replacement_sound h
      exact ⟨ht,hr,fun ctx call env _ hf hi => hs ctx call env hf hi⟩
  advance ib _ facts s := advance ib facts s
  advance_sound _ _ _ _ _ _ _ _ _ hf hi hs := advance_sound hf hi hs

def normalizer : TypedNormalizer where
  run ib d Γ ss hints yields :=
    (BodyNormalize.recursive (FactNormalize.base family ib (DenseExtract.hintOps hints))
      (max (RegionDepth.ops ss) (RegionDepth.ops hints))).run d Γ [] ss yields
  sound ib d Γ ss hints ns yields ctx call env hc hn he hi hd :=
    (BodyNormalize.recursive (FactNormalize.base family ib (DenseExtract.hintOps hints))
      (max (RegionDepth.ops ss) (RegionDepth.ops hints))).sound d Γ [] ss ns yields ctx call env
      hc hn he (by intro f hf; cases hf) hi hd

end MerlinVerify.Rules.IndexFold
