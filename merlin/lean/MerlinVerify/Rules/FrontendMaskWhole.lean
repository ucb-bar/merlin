/- Whole-program execution of the captured mask-selection cut. -/
import MerlinVerify.Rules.FrontendMaskTemplate
import MerlinVerify.Rules.ReshapeIndex

namespace MerlinVerify.Rules.FrontendMaskWhole
open MerlinVerify MerlinVerify.Check
open FrontendMaskRuntime FrontendMaskBounds FrontendMaskProgram FrontendMaskExecution
open FrontendMaskCount FrontendMaskCountProgram FrontendMaskTemplate
open Sem.Ops

theorem unmodelled_free (op : Op)
    (h : (op.props ++ op.attrs).all (fun p => Spec.Annotations.semanticsFree p.1) = true) :
    unmodelledAttr? op = none := by
  unfold unmodelledAttr?
  apply List.findSome?_eq_none_iff.mpr
  intro pair hp
  have hf := List.all_eq_true.mp h pair hp
  simp only [hf, Bool.or_true, ↓reduceIte]

theorem metadata_ok : MetadataFacts FrontendMaskTemplate.metadata := by
  unfold MetadataFacts
  refine ⟨?_, ?_, ?_, ?_⟩
  · cbv
  · cbv
  · apply unmodelled_free; cbv
  · apply unmodelled_free; cbv

/-- A row-major input representation. Integer input words are arbitrary, initialized i64 values. -/
def row (shape : List Nat) (elt : Ty) (f : Nat → Val) : MerlinVerify.Tensor :=
  Tensor.canon shape elt (fun p => f (linearIndex shape p))

theorem row_reshape (old new : List Nat) (elt : Ty) (f : Nat → Val)
    (hs : shapeSize old = shapeSize new) :
    ReshapeIndex.reshape (row old elt f) new elt = row new elt f := by
  apply Tensor.canon_ext
  intro p hp
  have hk : linearIndex new p < shapeSize old := by rw [hs]; exact linearIndex_lt _ _ hp
  have hj := delinearize_inBounds old _ hk
  simp only [row, Tensor.canon, hj, ↓reduceIte]
  rw [ReshapeIndex.linear_delinearize old _ hk]

theorem row_flat (elt : Ty) (f : Nat → Val) {n i : Nat} (hi : i < n) :
    (row [n] elt f).get [i] = f i := by
  simp [row, Tensor.canon, one_inBounds.mpr hi, linearIndex]

/-- Operation fields are read from the decoded AST; the validity premise is checked for the
actual reassociation and runtime shape. No annotation supplies a numerical or bounds law. -/
theorem reshape_exec (limit : Nat) (ctx : Ctx) (call : CallSem) (op : Op)
    (t : MerlinVerify.Tensor) (new : List Nat) (elt : Ty) (groups : List (List Nat))
    (hn : op.name = "tensor.collapse_shape" ∨ op.name = "tensor.expand_shape")
    (hm : unmodelledAttr? op.blind = none)
    (hr : Tensor.resultTensorTy op.blind = some (new, elt))
    (hg : Tensor.reassociation? op.blind = some groups)
    (he : t.elt = elt)
    (hv : (if op.name = "tensor.collapse_shape" then Tensor.reassocValid groups t.shape new
      else Tensor.reassocValid groups new t.shape) = true) :
    FrontendMaskRuntime.sem limit ctx call op [.tensor t] [] =
      .ok [.tensor (ReshapeIndex.reshape t new elt)] := by
  rcases hn with hn | hn
  all_goals
    have hd : dialectOf op.name = "tensor" := by rw [hn]; cbv
    rw [tensor_eq limit ctx call op hd (by rw [hn]; decide)]
    simp only [opSem, hm, dispatch, Op.blind_name, hn]
    have hc : dialectOf "tensor.collapse_shape" = "tensor" := by cbv
    have hx : dialectOf "tensor.expand_shape" = "tensor" := by cbv
    simp only [hc, hx, Tensor.sem, Op.blind_name, hn, hr, hg]
    simpa [hn, he, ReshapeIndex.reshape] using hv

theorem index_constant (limit : Nat) (ctx : Ctx) (call : CallSem) (op : Op) (n : Nat)
    (hn : op.name = "arith.constant") (hm : unmodelledAttr? op.blind = none)
    (hr : op.blind.results = [{id := 0, ty := .index}])
    (hv : op.blind.attr? "value" = some (.int n .index)) (hib : ctx.indexBits = 64) :
    FrontendMaskRuntime.sem limit ctx call op [] [] = .ok [index n] := by
  have hd : dialectOf op.name = "arith" := by rw [hn]; cbv
  rw [arith_eq limit ctx call op hd]
  simp only [opSem, hm, dispatch, Op.blind_name, hn]
  have hc : dialectOf "arith.constant" = "arith" := by cbv
  simp [hc, Arith.sem, Op.blind_name, hn, hv, hr, scalarOfAttr, hib, index]
  cases ctx with
  | mk model ext ib loopIndex =>
    dsimp at hib ⊢
    subst ib
    rfl

theorem for_op (limit : Nat) (ctx : Ctx) (call : CallSem) (op : Op)
    (hn : op.name = "scf.for") (hm : unmodelledAttr? op.blind = none)
    (args : List RVal) (regions : List (Ctx → RegionSem)) :
    FrontendMaskRuntime.sem limit ctx call op args regions = forSem limit ctx args regions := by
  have hd : dialectOf "scf.for" = "scf" := by cbv
  simp only [FrontendMaskRuntime.sem, hm, Op.blind_name, hn, hd, ↓reduceIte]

theorem row_op (limit : Nat) (ctx : Ctx) (call : CallSem) (op : Op)
    (old new : List Nat) (elt : Ty) (groups : List (List Nat)) (f : Nat → Val)
    (hn : op.name = "tensor.collapse_shape" ∨ op.name = "tensor.expand_shape")
    (hm : unmodelledAttr? op.blind = none)
    (hr : Tensor.resultTensorTy op.blind = some (new, elt))
    (hg : Tensor.reassociation? op.blind = some groups)
    (hv : (if op.name = "tensor.collapse_shape" then Tensor.reassocValid groups old new
      else Tensor.reassocValid groups new old) = true) (hs : shapeSize old = shapeSize new) :
    FrontendMaskRuntime.sem limit ctx call op [.tensor (row old elt f)] [] = .ok [.tensor (row new elt f)] := by
  rw [reshape_exec limit ctx call op (row old elt f) new elt groups hn hm hr hg rfl hv,
    row_reshape old new elt f hs]

/-- Bounds and initialization transport through the final reshape. -/
theorem reshape_defined (t : MerlinVerify.Tensor) (new : List Nat) (elt : Ty)
    (hd : RVal.Defined (.tensor t)) (hs : shapeSize t.shape = shapeSize new) :
    RVal.Defined (.tensor (ReshapeIndex.reshape t new elt)) := by
  intro p hp
  change Idx.InBounds new p at hp
  have hk : linearIndex new p < shapeSize t.shape := by rw [hs]; exact linearIndex_lt _ _ hp
  have hi := delinearize_inBounds t.shape _ hk
  simpa [ReshapeIndex.reshape, Tensor.canon, hp] using hd _ hi

theorem captured_corr (mask values : MerlinVerify.Tensor) (env : Env) (mid vid one : Nat)
    (hm : env.get? mid = some (.tensor mask)) (hv : env.get? vid = some (.tensor values))
    (ho : env.get? one = some (index 1)) :
    Corr [(0, mid), (1, vid), (2, one)] (capturedEnv mask values) env := by
  intro x y h
  by_cases h0 : x = 0
  · subst x; simp [Rho.find, List.lookup] at h; subst y; exact hm.symm
  · by_cases h1 : x = 1
    · subst x; simp [Rho.find, List.lookup] at h; subst y; exact hv.symm
    · by_cases h2 : x = 2
      · subst x; simp [Rho.find, List.lookup] at h; subst y; exact ho.symm
      · simp only [Rho.find, List.lookup, beq_eq_false_iff_ne.mpr h0,
          beq_eq_false_iff_ne.mpr h1, beq_eq_false_iff_ne.mpr h2] at h
        cases h

theorem op_step {limit : Nat} {ctx : Ctx} {call : CallSem} {op : Op} {env : Env}
    {args results : List RVal} (tail : List Op)
    (hterm : op.isTerminator = false) (hread : env.getAll op.operands = .ok args)
    (hsem : FrontendMaskRuntime.sem limit ctx call op args
      (evalRegions (FrontendMaskRuntime.sem limit) call env op.regions) = .ok results) :
    evalOps (FrontendMaskRuntime.sem limit) ctx call env (op :: tail) =
      (env.bindAll op.results results).bind fun next =>
        evalOps (FrontendMaskRuntime.sem limit) ctx call next tail := by
  rw [evalOps_cons_step _ _ _ _ _ _ hterm]
  simp only [stepOp, hread, Outcome.bind_ok, hsem]

local macro "advance " op:ident " using " law:term : tactic => `(tactic|
  (rw [op_step (op := $op) _ (by cbv) (by rfl) $law]
   rw [bindAll_eq_ok (by rfl)]
   simp only [Outcome.bind_ok]
   conv_lhs =>
     arg 4
     simp only [$op:ident, Op.results, List.map_cons, List.map_nil, List.zip_cons_cons,
       List.zip_nil_left, List.cons_append, List.nil_append]))



set_option maxRecDepth 20000 in
set_option maxHeartbeats 8000000 in
theorem body_exec (limit : Nat) (ctx : Ctx) (call : CallSem)
    (bits : Nat → Bool) (values seed : Nat → BitVec 64)
    (hib : ctx.indexBits = 64) (hu : FrontendBoolFptosi.UneCorrect ctx.M) (hl : 1024 ≤ limit) :
    ∃ out : MerlinVerify.Tensor,
      evalRegion (FrontendMaskRuntime.sem limit) ctx call [] FrontendMaskTemplate.body
        [.tensor (row [1,32,32] (.int 1) (fun i => Arith.boolVal (bits i))),
         .tensor (row [1,1024] (.int 64) (fun i => .some (.bits 64 (values i)))),
         .tensor (row [1,1024] (.int 64) (fun i => .some (.bits 64 (seed i))))] =
        .ok [.tensor out, index (countFrom bits 1024 0)] ∧
      out.shape = [1,1024] ∧ RVal.Defined (.tensor out) := by
  let bf : Nat → Val := fun i => Arith.boolVal (bits i)
  let vf : Nat → Val := fun i => .some (.bits 64 (values i))
  let sf : Nat → Val := fun i => .some (.bits 64 (seed i))
  let mask := row [1024] (.int 1) bf
  let vals := row [1024] (.int 64) vf
  let initial := row [1024] (.int 64) sf
  obtain ⟨cap, hcap⟩ : ∃ cap, countFrom bits 1024 0 = cap := ⟨_, rfl⟩
  let buf := MerlinVerify.Tensor.full [cap] (.int 64) .poison
  obtain ⟨compact, hgather⟩ : ∃ compact,
      gather bits (fun i => vals.get [i]) 1024 0 buf 0 = compact := ⟨_, rfl⟩
  obtain ⟨final, hscatter⟩ : ∃ final,
      scatter bits compact.1 1024 0 initial 0 = final := ⟨_, rfl⟩
  have hn : 1024 < 2^63 := by decide
  have hcle : cap ≤ 1024 := by rw [← hcap]; exact countFrom_le bits 1024 0
  have hc : cap < 2^63 := by omega
  have hm : MaskMatches mask bits 1024 := ⟨rfl, fun i hi => row_flat _ bf hi⟩
  have hgi := gather_invariant bits (fun i => vals.get [i]) 1024 0 buf 0 cap rfl
    (by intro i hi; omega) (by rw [hcap]; omega) (by
      intro i _ hi
      have hi' : i < 1024 := by omega
      rw [show vals = row [1024] (.int 64) vf from rfl, row_flat _ vf hi']
      simp [vf])
  dsimp only at hgi
  rw [hgather, hcap] at hgi
  have hgc : compact.2 = cap := by simpa only [Nat.zero_add] using hgi.2.1
  have hgp : PrefixInitialized compact.1 cap := by simpa only [hgc] using hgi.2.2
  have hseed : RVal.Defined (.tensor initial) := by
    intro point hp
    change Idx.InBounds [1024] point at hp
    simp [initial, row, MerlinVerify.Tensor.canon, hp, sf]
  have hsi := scatter_invariant bits compact.1 1024 0 initial 0 cap 1024
    hgi.1 hgp rfl hseed (by omega) (by rw [hcap]; omega)
  dsimp only at hsi
  rw [hscatter, hcap] at hsi
  have hsc : final.2 = cap := by simpa only [Nat.zero_add] using hsi.2.1
  have hse : final.1.elt = .int 64 := by rw [← hscatter]; exact scatter_elt bits compact.1 1024 0 initial 0
  let out := ReshapeIndex.reshape final.1 [1,1024] (.int 64)
  refine ⟨out, ?_, rfl, reshape_defined final.1 [1,1024] (.int 64) hsi.2.2 ?_⟩
  swap
  · rw [hsi.1]; decide
  have h_maskCollapse0 := row_op limit ctx call maskCollapse0 [1, 32, 32] [1024] (.int 1) [[0, 1, 2]] bf
    (by left; rfl) (by cbv) (by cbv) (by cbv) (by cbv) (by decide)
  have h_maskExpand0 := row_op limit ctx call maskExpand0 [1024] [1, 1024] (.int 1) [[0, 1]] bf
    (by right; rfl) (by cbv) (by cbv) (by cbv) (by cbv) (by decide)
  have h_valuesCollapse := row_op limit ctx call valuesCollapse [1, 1024] [1024] (.int 64) [[0, 1]] vf
    (by left; rfl) (by cbv) (by cbv) (by cbv) (by cbv) (by decide)
  have h_maskFlatten0 := row_op limit ctx call maskFlatten0 [1, 1024] [1024] (.int 1) [[0, 1]] bf
    (by left; rfl) (by cbv) (by cbv) (by cbv) (by cbv) (by decide)
  have h_maskCollapse1 := row_op limit ctx call maskCollapse1 [1, 32, 32] [1024] (.int 1) [[0, 1, 2]] bf
    (by left; rfl) (by cbv) (by cbv) (by cbv) (by cbv) (by decide)
  have h_maskExpand1 := row_op limit ctx call maskExpand1 [1024] [1, 1024] (.int 1) [[0, 1]] bf
    (by right; rfl) (by cbv) (by cbv) (by cbv) (by cbv) (by decide)
  have h_seedCollapse := row_op limit ctx call seedCollapse [1, 1024] [1024] (.int 64) [[0, 1]] sf
    (by left; rfl) (by cbv) (by cbv) (by cbv) (by cbv) (by decide)
  have h_maskFlatten1 := row_op limit ctx call maskFlatten1 [1, 1024] [1024] (.int 1) [[0, 1]] bf
    (by left; rfl) (by cbv) (by cbv) (by cbv) (by cbv) (by decide)
  have h_gatherLower := index_constant limit ctx call gatherLower 0 rfl (by cbv) rfl (by cbv) hib
  have h_gatherStep := index_constant limit ctx call gatherStep 1 rfl (by cbv) rfl (by cbv) hib
  have h_gatherUpper := index_constant limit ctx call gatherUpper 1024 rfl (by cbv) rfl (by cbv) hib
  have h_scatterLower := index_constant limit ctx call scatterLower 0 rfl (by cbv) rfl (by cbv) hib
  have h_scatterStep := index_constant limit ctx call scatterStep 1 rfl (by cbv) rfl (by cbv) hib
  have h_scatterUpper := index_constant limit ctx call scatterUpper 1024 rfl (by cbv) rfl (by cbv) hib
  have hempty : FrontendMaskRuntime.sem limit ctx call compactEmpty [index cap] [] = .ok [.tensor buf] := by
    rw [sem_blind (show compactEmpty.blind = FrontendMaskExecution.emptyOp.blind from rfl)]
    exact empty_alloc limit ctx call cap hib hc
  have h_gather : ∀ env : Env,
      Corr [(0,0),(1,40),(2,10)] (capturedEnv mask vals) env →
      FrontendMaskRuntime.sem limit ctx call gatherLoop
        [index 0, index 1024, index 1, .tensor buf, index 0]
        (evalRegions (FrontendMaskRuntime.sem limit) call env gatherLoop.regions) =
        .ok [.tensor compact.1, index cap] := by
    intro env he
    rw [for_op limit ctx call gatherLoop rfl (by cbv)]
    change forSem limit ctx ([index 0, index 1024, index 1] ++ [.tensor buf, index 0])
      [fun inner => evalRegion (FrontendMaskRuntime.sem limit) inner call env gatherLoop.regions.head!] = _
    rw [for_carried limit ctx 1024 _ _ hib hn hl]
    simpa only [hgather, hgc] using accepted_gather_loop limit ctx call env mask vals bits 1024 cap
      (actual := gatherLoop.regions.head!) (maskId := 0) (valuesId := 40) (oneId := 10)
      (by cbv) hu he hm rfl hn hc buf rfl (by rw [hcap]; omega)
  have h_scatter : ∀ env : Env,
      Corr [(0,42),(1,13),(2,16)] (capturedEnv mask compact.1) env →
      FrontendMaskRuntime.sem limit ctx call scatterLoop
        [index 0, index 1024, index 1, .tensor initial, index 0]
        (evalRegions (FrontendMaskRuntime.sem limit) call env scatterLoop.regions) =
        .ok [.tensor final.1, index cap] := by
    intro env he
    rw [for_op limit ctx call scatterLoop rfl (by cbv)]
    change forSem limit ctx ([index 0, index 1024, index 1] ++ [.tensor initial, index 0])
      [fun inner => evalRegion (FrontendMaskRuntime.sem limit) inner call env scatterLoop.regions.head!] = _
    rw [for_carried limit ctx 1024 _ _ hib hn hl]
    simpa only [hscatter, hsc] using accepted_scatter_loop limit ctx call env mask compact.1 bits 1024 cap
      (actual := scatterLoop.regions.head!) (maskId := 42) (valuesId := 13) (oneId := 16)
      (by cbv) hu he hm hgi.1 hn hc initial rfl (by rw [hcap]; omega)
  have h_result := reshape_exec limit ctx call resultExpand final.1 [1,1024] (.int 64) [[0,1]]
    (by right; rfl) (by cbv) (by cbv) (by cbv) hse (by rw [hsi.1]; cbv)
  simp only [FrontendMaskTemplate.body, evalRegion, evalBlocks, evalBlock, Env.bindAll,
    List.length_cons, List.length_nil, ↓reduceIte, List.map_cons, List.map_nil,
    List.zip_cons_cons, List.zip_nil_left, List.cons_append, List.nil_append, Outcome.bind_ok]
  simp only [operations, List.cons_append, List.nil_append]
  advance maskCollapse0 using h_maskCollapse0
  advance maskExpand0 using h_maskExpand0
  advance valuesCollapse using h_valuesCollapse
  advance maskFlatten0 using h_maskFlatten0
  rw [count_prefix_exec 1024 limit countMap.attrs countMap.props metadata (by cbv) metadata_ok
    ctx call mask bits hm hib]
  simp only [countEnv, hcap, List.cons_append, List.nil_append, afterCount]
  advance gatherLower using h_gatherLower
  advance gatherStep using h_gatherStep
  advance gatherUpper using h_gatherUpper
  advance compactEmpty using hempty
  advance gatherLoop using (by
    apply h_gather
    apply captured_corr <;> rfl)
  advance maskCollapse1 using h_maskCollapse1
  advance maskExpand1 using h_maskExpand1
  advance seedCollapse using h_seedCollapse
  advance maskFlatten1 using h_maskFlatten1
  advance scatterLower using h_scatterLower
  advance scatterStep using h_scatterStep
  advance scatterUpper using h_scatterUpper
  advance scatterLoop using (by
    apply h_scatter
    apply captured_corr <;> rfl)
  advance resultExpand using h_result
  rfl

#print axioms body_exec

def inputs (bits : Nat → Bool) (values seed : Nat → BitVec 64) : List RVal :=
  [.tensor (row [1,32,32] (.int 1) (fun i => Arith.boolVal (bits i))),
   .tensor (row [1,1024] (.int 64) (fun i => .some (.bits 64 (values i)))),
   .tensor (row [1,1024] (.int 64) (fun i => .some (.bits 64 (seed i))))]

theorem subject_find : subject.func? "subject" =
    some {name := "subject", body := FrontendMaskTemplate.body, isPrivate := false, isDecl := false} := by cbv

theorem template_denote (limit : Nat) (ctx : Ctx) (bits : Nat → Bool) (values seed : Nat → BitVec 64)
    (hib : ctx.indexBits = 64) (hu : FrontendBoolFptosi.UneCorrect ctx.M) (hl : 1024 ≤ limit) :
    ∃ out : MerlinVerify.Tensor,
      denote (FrontendMaskRuntime.sem limit) ctx subject "subject" (inputs bits values seed) =
        .ok [.tensor out, index (countFrom bits 1024 0)] ∧
      out.shape = [1,1024] ∧ RVal.Defined (.tensor out) := by
  obtain ⟨out, he, hs, hd⟩ := body_exec limit ctx
    (evalFunc (FrontendMaskRuntime.sem limit) ctx subject 1) bits values seed hib hu hl
  refine ⟨out, ?_, hs, hd⟩
  rw [denote, show subject.funcs.length = 1 from by cbv, evalFunc, subject_find]
  exact he

/-- All operations, regions, SSA definitions/uses and input/output types in the actual cut are
checked. Additional top-level operations or functions are explicitly unsupported. -/
def check (actual : Module) : Bool :=
  actual.ops.length == 1 && decide (actual.attrs = subject.attrs ∧ actual.props = subject.props) &&
    FrontendBoolSimulation.functions [] subject.funcs actual.funcs

/-- Accepted actual MLIR programs execute successfully and return initialized elements on the
declared initialized cut domain. The approved f32 instance discharges the generic comparison
premise. This is not a successful-execution theorem about the full SmolVLA function. -/
theorem check_sound {actual : Module} (h : check actual = true)
    (limit : Nat) (base : FloatModel) (ctx : Ctx) (bits : Nat → Bool)
    (values seed : Nat → BitVec 64) (hib : ctx.indexBits = 64) (hl : 1024 ≤ limit) :
    ∃ out : MerlinVerify.Tensor,
      denote (FrontendMaskRuntime.sem limit) {ctx with M := FrontendF32Comparison.model base}
        actual "subject" (inputs bits values seed) = .ok [.tensor out, index (countFrom bits 1024 0)] ∧
      out.shape = [1,1024] ∧ RVal.Defined (.tensor out) := by
  simp only [check, Bool.and_eq_true] at h
  have he := FrontendBoolSimulationWith.evalFunc_sound (FrontendMaskRuntime.sem limit)
    (FrontendBoolSimulationWith.candidateLaws limit) h.2
    {ctx with M := FrontendF32Comparison.model base} (FrontendF32Comparison.model_correct base)
  rw [FrontendBoolSimulationWith.policy_nil] at he
  have hlen := FrontendBoolSimulation.functions_length subject.funcs actual.funcs h.2
  obtain ⟨out, hout, hs, hd⟩ := template_denote limit {ctx with M := FrontendF32Comparison.model base}
    bits values seed hib (FrontendF32Comparison.model_correct base) hl
  refine ⟨out, ?_, hs, hd⟩
  simpa only [denote, he, hlen] using hout

#print axioms check_sound
end MerlinVerify.Rules.FrontendMaskWhole
