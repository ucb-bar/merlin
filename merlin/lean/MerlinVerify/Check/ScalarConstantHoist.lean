/- Exact movement of one total scalar constant from a leading structured-body prefix.
The operation and both continuations retain checked scopes; no annotation invents a value. -/
import MerlinVerify.Check.ConstantFacts
import MerlinVerify.Check.StateNormalize
import MerlinVerify.Rules.MotionIndependent
import MerlinVerify.Check.EnvBindingCommute

namespace MerlinVerify.Check.ScalarConstantHoist

open MerlinVerify MerlinVerify.Sem.Ops MerlinVerify.Sem.Ops.Linalg

/-- Only the actual successful scalar result of the frozen constant decoder can be hoisted. -/
def scalarValue? (ib : Nat) (constant : Op) : Option Val := do
  match ← ConstantFacts.value? ib constant with
  | .scalar value => some value
  | .tensor _ => none

theorem scalarValue_spec {ib : Nat} {constant : Op} {value : Val}
    (h : scalarValue? ib constant = some value) :
    ConstantFacts.value? ib constant = some (.scalar value) := by
  unfold scalarValue? at h
  cases hv : ConstantFacts.value? ib constant with
  | none => simp [hv] at h
  | some actual =>
    cases actual with
    | tensor _ => simp [hv] at h
    | scalar actual => simpa [hv] using h

/-- Index constants are excluded from this initial family. No loop-dependent operation or
floating arithmetic is evaluated; float constants contribute only their actual raw bits. -/
def fixedScalar (ty : Ty) : Prop := match ty with
  | .int width => 0 < width
  | .float _ => True
  | _ => False
instance (ty : Ty) : Decidable (fixedScalar ty) := by unfold fixedScalar; split <;> infer_instance

structure Source where
  args : List Value
  constant : Op
  tail : List Op

def source? (s : Op) : Option Source := match s.regions with
  | [.mk [.mk args (constant :: tail)]] => some ⟨args, constant, tail⟩
  | _ => none

theorem source_spec {s : Op} {source : Source} (h : source? s = some source) :
    s.regions = [.mk [.mk source.args (source.constant :: source.tail)]] := by
  unfold source? at h
  split at h
  · cases h; assumption
  · cases h

structure Plan where
  result : Value
  value : Val
  types : List VTy
  inputs : List VTy
  outputs : List VTy
  maps : List AffineMap
  iters : List IterKind
  ranges : List Nat

def signature (p : Plan) : RegionSig :=
  ⟨(p.inputs ++ p.outputs).map VTy.eltScalar, p.outputs.map VTy.eltScalar, p.iters.length⟩

def bodyEnv (Γ : TEnv) (source : Source) (p : Plan) : TEnv :=
  (p.result.id, .scalar p.result.ty) :: (source.args.map (·.id)).zip (signature p).args ++ Γ

def resultRho (Γ : TEnv) (s : Op) : Rho :=
  (s.results.map (·.id)).zip (s.results.map (·.id)) ++ SplatNormalize.diagonal Γ

private def valueDecEq (a b : Value) : Decidable (a = b) :=
  decidable_of_iff (a.id = b.id ∧ a.ty = b.ty) (by cases a; cases b; simp)
local instance : DecidableEq Value := valueDecEq

/-- Every semantic, source-scope, region-layout and continuation premise is independently
recomputed. The added outer binding must be unobservable outside the original generic body. -/
def Checks (ib : Nat) (Γ : TEnv) (s : Op) (tail : List Op) (source : Source) (p : Plan) : Prop :=
  s.name = "linalg.generic" ∧ unmodelledAttr? s.blind = none ∧ s.successors = [] ∧
  scalarValue? ib source.constant = some p.value ∧ source.constant.results = [p.result] ∧
  fixedScalar p.result.ty ∧ p.value ≠ .poison ∧
  s.operands.mapM (Γ.lookup ·) = some p.types ∧ splitInsOuts s p.types = some (p.inputs, p.outputs) ∧
  attrAffineMaps? s "indexing_maps" = some p.maps ∧ iterKinds? s = some p.iters ∧
  structuredRanges? p.maps p.iters.length (p.inputs ++ p.outputs) = some p.ranges ∧
  (∀ ty ∈ p.outputs, ty.isTensor = true) ∧
  checkRegion (merlinBook ib) Γ (.mk [.mk source.args (source.constant :: source.tail)]) (signature p) = true ∧
  p.result.id ∉ source.args.map (·.id) ∧ p.result.id ∉ s.operands ∧ p.result.id ∉ s.results.map (·.id) ∧
  freshIds (SplatNormalize.diagonal Γ) [p.result.id] = true ∧
  freshIds (SplatNormalize.diagonal Γ) (s.results.map (·.id)) = true ∧
  alphaOps (SplatNormalize.diagonal (bodyEnv Γ source p)) source.tail source.tail = true ∧
  alphaOps (resultRho Γ s) tail tail = true

instance (ib : Nat) (Γ : TEnv) (s : Op) (tail : List Op) (source : Source) (p : Plan) :
    Decidable (Checks ib Γ s tail source p) := by unfold Checks; infer_instance

/-- The internal target retains the generic header, results and block arguments exactly. -/
def stripped (s : Op) (source : Source) : Op :=
  SplatNested.withRegions s [.mk [.mk source.args source.tail]]


/-- The particular original constant produces the same scalar in every checked context. -/
theorem constant_step {ib : Nat} {constant : Op} {result : Value} {value : Val}
    (hv : scalarValue? ib constant = some value) (hr : constant.results = [result])
    {ctx : Ctx} (hi : ctx.indexBits = ib) (call : CallSem) (env : Env) :
    stepOp opSem ctx call env constant = .ok ((result.id, .scalar value) :: env) := by
  obtain ⟨r, hresult, hs⟩ := ConstantFacts.value_exec (scalarValue_spec hv) hi call env
  have heq : r = result := by simpa [hr] using hresult.symm
  subst r
  exact hs

theorem constant_nonterminator {ib : Nat} {constant : Op} {value : Val}
    (hv : scalarValue? ib constant = some value) : constant.isTerminator = false := by
  have hn := (ConstantFacts.value_spec (scalarValue_spec hv)).1.1
  simp [Op.isTerminator, terminatorNames, hn]

/-- Binding the checked scalar before the generic rather than after its region arguments
preserves every body lookup. Alpha validation covers all nested captures and later rebindings. -/
theorem body_equal {ib : Nat} {Γ : TEnv} {source : Source} {p : Plan}
    (hv : scalarValue? ib source.constant = some p.value)
    (hr : source.constant.results = [p.result])
    (hc : checkRegion (merlinBook ib) Γ
      (.mk [.mk source.args (source.constant :: source.tail)]) (signature p) = true)
    (hfresh : p.result.id ∉ source.args.map (·.id))
    (ha : alphaOps (SplatNormalize.diagonal (bodyEnv Γ source p)) source.tail source.tail = true)
    (call : CallSem) (env : Env) :
    Rules.StructuredBody.Equal ib (signature p)
      (fun ctx => evalRegion opSem ctx call env (.mk [.mk source.args (source.constant :: source.tail)]))
      (fun ctx => evalRegion opSem ctx call ((p.result.id, .scalar p.value) :: env)
        (.mk [.mk source.args source.tail])) := by
  intro ctx hib _ values hvalues
  simp only [checkRegion, Bool.and_eq_true, beq_iff_eq] at hc
  have hlen : source.args.length = values.length :=
    (optionMapM_length _ _ _ hc.1).symm.trans (HasTys.length hvalues).symm
  simp only [evalRegion, evalBlocks, evalBlock, bindAll_eq_ok hlen, Outcome.bind_ok]
  rw [evalOps_cons_step _ _ _ _ _ _ (constant_nonterminator hv),
    constant_step hv hr hib call _, Outcome.bind_ok]
  apply alphaOps_sound call call _ source.tail source.tail ctx _ _ ha
  · intro x y hxy
    exact (SplatNormalize.diagonal_refl (bodyEnv Γ source p)
      ((p.result.id, .scalar p.value) :: ((source.args.map (·.id)).zip values ++ env)) x y hxy).trans
      (EnvBindingCommute.get?_singleton_zip env p.result.id (.scalar p.value)
        (source.args.map (·.id)) values hfresh y)
  · intro _ _; rfl


/-- The structured-loop proof compares the two actual captured environments on the source's
checked runtime body domain. Target failures and zero-iteration behavior are preserved exactly. -/
theorem generic_sem {ib : Nat} {Γ : TEnv} {s : Op} {tail : List Op} {source : Source} {p : Plan}
    (h : Checks ib Γ s tail source p)
    (hsource : s.regions = [.mk [.mk source.args (source.constant :: source.tail)]])
    {ctx : Ctx} (hi : ctx.indexBits = ib) (call : CallSem) {env : Env} (he : EnvTyped ib Γ env)
    {args : List RVal} (ha : HasTys ib args p.types) :
    opSem ctx call (stripped s source) args
      (evalRegions opSem call ((p.result.id, .scalar p.value) :: env) (stripped s source).regions) =
    opSem ctx call s args (evalRegions opSem call env s.regions) := by
  obtain ⟨hn, hm, _, hv, hr, _, _, _, hsplit, hmaps, hiters, hlayout, hout,
    hcheck, hargs, _, _, _, _, halpha, _⟩ := h
  have hsafe := checkRegion_safe (merlinBook ib) Γ
    (.mk [.mk source.args (source.constant :: source.tail)]) (signature p) call env hcheck he
  have hbody := body_equal hv hr hcheck hargs halpha call env
  have hsem := SplatNested.semantic_equal hn hm hsplit hmaps hiters hlayout hout hi call ha hsafe hbody
  simp only [stripped, SplatNested.withRegions_regions, hsource, evalRegions]
  rw [opSem_blind (op' := s) (SplatNested.withRegions_blind _ _)]
  exact hsem.symm

/-- A total closed scalar producer may execute before the generic only when every newly
visible binding is checked to be irrelevant to the surrounding program and nested captures. -/
theorem hoist_sound {ib : Nat} {Γ : TEnv} {s : Op} {tail : List Op} {source : Source} {p : Plan}
    (h : Checks ib Γ s tail source p)
    (hsource : s.regions = [.mk [.mk source.args (source.constant :: source.tail)]])
    {ctx : Ctx} (hi : ctx.indexBits = ib) (call : CallSem) {env : Env} (he : EnvTyped ib Γ env) :
    evalOps opSem ctx call env (source.constant :: stripped s source :: tail) =
      evalOps opSem ctx call env (s :: tail) := by
  have checks := h
  obtain ⟨hn, _, _, hv, hr, _, _, htypes, _, _, _, _, _, _, _, hop, _, hconst, hresults, _, htail⟩ := checks
  have hterm : s.isTerminator = false := by simp [Op.isTerminator, terminatorNames, hn]
  obtain ⟨args, hargs, htyped⟩ := he.getAll _ _ htypes
  have hargs' : Env.getAll ((p.result.id, .scalar p.value) :: env) s.operands = .ok args := by
    rw [Rules.MotionIndependent.getAll_cons_not_mem env p.result.id (.scalar p.value) s.operands hop]
    exact hargs
  rw [evalOps_cons_step _ _ _ _ _ _ (constant_nonterminator hv), constant_step hv hr hi call env,
    Outcome.bind_ok, evalOps_cons_step _ _ _ _ _ _ (by simpa [stripped] using hterm),
    evalOps_cons_step _ _ _ _ _ _ hterm]
  simp only [stepOp, stripped, SplatNested.withRegions_operands, SplatNested.withRegions_results,
    hargs, hargs', Outcome.bind_ok]
  have hsem := generic_sem h hsource hi call he htyped
  simp only [stripped] at hsem
  rw [hsem]
  cases hs : opSem ctx call s args (evalRegions opSem call env s.regions) with
  | ub => rfl
  | stuck _ => rfl
  | ok values =>
    simp only [Outcome.bind_ok, Env.bindAll]
    by_cases hlen : s.results.length = values.length
    · simp only [hlen, ↓reduceIte, Outcome.bind_ok]
      have hc := Rules.DequantSite.Corr.temporary (SplatNormalize.diagonal_refl Γ env)
        ((freshIds_sound hconst).2 p.result.id (by simp)) (.scalar p.value)
      have hf := freshIds_sound hresults
      have hids : (s.results.map (·.id)).length = values.length := by simpa using hlen
      have hext := Corr.extend hc (s.results.map (·.id)) (s.results.map (·.id)) values hids hids hf.1 hf.2
      exact (alphaOps_sound call call _ tail tail ctx _ _ htail hext (fun _ _ => rfl)).symm
    · simp [hlen]


/-- Parsing proposes a layout and actual closed payload; Checks validates them independently. -/
def candidate (ib : Nat) (Γ : TEnv) (s : Op) (source : Source) : Option Plan := do
  let [result] := source.constant.results | none
  let value ← scalarValue? ib source.constant
  let types ← s.operands.mapM (Γ.lookup ·)
  let (inputs, outputs) ← splitInsOuts s types
  let maps ← attrAffineMaps? s "indexing_maps"
  let iters ← iterKinds? s
  let ranges ← structuredRanges? maps iters.length (inputs ++ outputs)
  pure ⟨result, value, types, inputs, outputs, maps, iters, ranges⟩

def checked (ib : Nat) (Γ : TEnv) (s : Op) (tail : List Op) : Option (Source × Plan) := do
  let source ← source? s
  let p ← candidate ib Γ s source
  if Checks ib Γ s tail source p then some (source, p) else none

theorem checked_spec {ib : Nat} {Γ : TEnv} {s : Op} {tail : List Op} {source : Source} {p : Plan}
    (h : checked ib Γ s tail = some (source, p)) :
    s.regions = [.mk [.mk source.args (source.constant :: source.tail)]] ∧ Checks ib Γ s tail source p := by
  unfold checked at h
  cases hs : source? s with
  | none => simp [hs] at h
  | some source' =>
    simp only [hs, bind, Option.bind] at h
    cases hp : candidate ib Γ s source' with
    | none => simp [hp] at h
    | some p' =>
      simp only [hp] at h
      split at h
      · next hc =>
        cases h
        exact ⟨source_spec hs, hc⟩
      · cases h

/-- Hoist one leading scalar constant per generic. This bounded kernel does not retry a whole
normalizer or claim that multiple separated constants have all been moved. -/
def finish (ib : Nat) (Γ : TEnv) (s : Op) (tail : List Op) : List Op :=
  match checked ib Γ s tail with
  | some (source, _) => source.constant :: stripped s source :: tail
  | none => s :: tail

theorem finish_sound {ib : Nat} {Γ : TEnv} {s : Op} {tail : List Op} {ctx : Ctx} {call : CallSem}
    {env : Env} (he : EnvTyped ib Γ env) (hi : ctx.indexBits = ib) :
    evalOps opSem ctx call env (finish ib Γ s tail) = evalOps opSem ctx call env (s :: tail) := by
  unfold finish
  cases hc : checked ib Γ s tail with
  | none => rfl
  | some pair =>
    obtain ⟨source, p⟩ := pair
    obtain ⟨hsource, hchecks⟩ := checked_spec hc
    exact hoist_sound hchecks hsource hi call he

def family : StateNormalize.Family Unit (fun _ _ => True) where
  initial := ()
  initial_valid _ := trivial
  advance _ _ _ _ := ()
  advance_sound _ _ _ _ _ _ _ _ _ _ _ _ := trivial
  finish ib Γ _ _ s tail := finish ib Γ s tail
  finish_sound _ _ _ _ _ _ _ _ _ _ he _ hi := finish_sound he hi

def normalize (ib : Nat) (Γ : TEnv) : List Op → Option (List Op) :=
  StateNormalize.normalize family ib Γ () []

theorem normalize_sound {ib : Nat} (R : Rulebook opSem ib) (d : Nat) :
    ∀ Γ ss ns yields ctx call env,
      checkOps R d Γ ss yields = true → normalize ib Γ ss = some ns →
      EnvTyped ib Γ env → ctx.indexBits = ib → d ≤ ctx.loopIndex.length →
      evalOps opSem ctx call env ns = evalOps opSem ctx call env ss := by
  intro Γ ss ns yields ctx call env hc hn he hi hd
  exact StateNormalize.normalize_sound family R d Γ () [] ss ns yields ctx call env hc hn he trivial hi hd

/-- The shared structured traversal supplies each actual local Γ. No new traversal is defined. -/
def base (ib : Nat) : SplatNormalizer.Normalizer ib := StateNormalize.base family ib []

end MerlinVerify.Check.ScalarConstantHoist
