/- Complete source/target operation and SSA checking for per-tensor quantizer lowering. -/
import MerlinVerify.Rules.QuantizeSiteLemmas
import MerlinVerify.Check.ExactSyntax

namespace MerlinVerify.Rules.QuantizeSite

open MerlinVerify MerlinVerify.Check MerlinVerify.Sem.Ops MerlinVerify.Sem.Ops.Linalg
open QuantizeTensor QuantizeRuntime QuantizeLowering QuantizeSiteLemmas

private def valueDecEq (a b : Value) : Decidable (a = b) :=
  decidable_of_iff (a.id = b.id ∧ a.ty = b.ty) (by cases a; cases b; simp)
local instance : DecidableEq Value := valueDecEq

structure Site where
  src : Op
  invEmpty : Op
  inverse : Op
  empty : Op
  lower : Op
  upper : Op
  generic : Op
  rs : Value
  r0 : Value
  r1 : Value
  r2 : Value
  rl : Value
  rh : Value
  rg : Value
  x : Nat
  scale : Nat
  zero : Nat
  c : Quantize.Config
  inverseRegion : Region
  region : Region
  maps : List AffineMap
  iters : List IterKind

def targets (s : Site) : List Op := [s.invEmpty, s.inverse, s.empty, s.lower, s.upper, s.generic]
def ids (s : Site) : List Nat := [s.r0.id, s.r1.id, s.r2.id, s.rl.id, s.rh.id, s.rg.id]

def Frame (s : Site) : Prop :=
  s.src.results = [s.rs] ∧ s.invEmpty.results = [s.r0] ∧ s.inverse.results = [s.r1] ∧
  s.empty.results = [s.r2] ∧ s.lower.results = [s.rl] ∧ s.upper.results = [s.rh] ∧
  s.generic.results = [s.rg] ∧ s.src.regions = [] ∧ s.invEmpty.regions = [] ∧
  s.inverse.regions = [s.inverseRegion] ∧ s.empty.regions = [] ∧ s.lower.regions = [] ∧
  s.upper.regions = [] ∧ s.generic.regions = [s.region]

def frameCheck (s : Site) : Bool :=
  decide (s.src.results = [s.rs] ∧ s.invEmpty.results = [s.r0] ∧ s.inverse.results = [s.r1] ∧
    s.empty.results = [s.r2] ∧ s.lower.results = [s.rl] ∧ s.upper.results = [s.rh] ∧
    s.generic.results = [s.rg]) &&
  ExactSyntax.regions s.src.regions [] && ExactSyntax.regions s.invEmpty.regions [] &&
  ExactSyntax.regions s.inverse.regions [s.inverseRegion] && ExactSyntax.regions s.empty.regions [] &&
  ExactSyntax.regions s.lower.regions [] && ExactSyntax.regions s.upper.regions [] &&
  ExactSyntax.regions s.generic.regions [s.region]

theorem frameCheck_sound {s : Site} (h : frameCheck s = true) : Frame s := by
  simp only [frameCheck, Bool.and_eq_true, decide_eq_true_eq] at h
  rcases h with ⟨⟨⟨⟨⟨⟨⟨results, hs⟩, h0⟩, h1⟩, h2⟩, hl⟩, hh⟩, hg⟩
  rcases results with ⟨rs, r0, r1, r2, rl, rh, rg⟩
  exact ⟨rs, r0, r1, r2, rl, rh, rg, ExactSyntax.regions_sound _ _ hs,
    ExactSyntax.regions_sound _ _ h0, ExactSyntax.regions_sound _ _ h1,
    ExactSyntax.regions_sound _ _ h2, ExactSyntax.regions_sound _ _ hl,
    ExactSyntax.regions_sound _ _ hh, ExactSyntax.regions_sound _ _ hg⟩

def Checks (ρ : Rho) (s : Site) : Prop :=
  frameCheck s = true ∧
  (s.src :: targets s).all (fun op => !op.isTerminator && decide (unmodelledAttr? op.blind = none) &&
    op.successors.isEmpty) = true ∧
  Quantize.config? s.src = some s.c ∧ s.src.operands.mapM ρ.find = some [s.x, s.scale, s.zero] ∧
  s.invEmpty.name = "tensor.empty" ∧ s.inverse.name = "linalg.generic" ∧
  s.empty.name = "tensor.empty" ∧ s.lower.name = "arith.constant" ∧
  s.upper.name = "arith.constant" ∧ s.generic.name = "linalg.generic" ∧
  s.invEmpty.operands = [] ∧ s.inverse.operands = [s.scale, s.r0.id] ∧ s.empty.operands = [] ∧
  s.lower.operands = [] ∧ s.upper.operands = [] ∧
  s.generic.operands = [s.x, s.r1.id, s.zero, s.r2.id] ∧
  Sem.Ops.Tensor.resultTensorTy s.invEmpty = some ([], .float .f32) ∧
  Sem.Ops.Tensor.resultTensorTy s.empty = some (s.c.shape, .int 8) ∧
  s.r1.ty.staticTensor? = some ([], .float .f32) ∧ s.rg.ty.staticTensor? = some (s.c.shape, .int 8) ∧
  s.lower.attr? "value" = some (.float 0xc3000000 (.float .f32)) ∧
  s.upper.attr? "value" = some (.float 0x42fe0000 (.float .f32)) ∧
  attrDenseArray? s.inverse "operandSegmentSizes" = some [1, 1] ∧
  attrAffineMaps? s.inverse "indexing_maps" = some [identityMap 0, identityMap 0] ∧
  iterKinds? s.inverse = some [] ∧
  attrDenseArray? s.generic "operandSegmentSizes" = some [3, 1] ∧
  attrAffineMaps? s.generic "indexing_maps" = some s.maps ∧ iterKinds? s.generic = some s.iters ∧
  s.maps = [identityMap s.c.shape.length, broadcast s.c.shape.length,
    broadcast s.c.shape.length, identityMap s.c.shape.length] ∧
  s.iters = List.replicate s.c.shape.length .parallel ∧
  structuredRanges? s.maps s.iters.length (inputTypes s.c.shape) = some s.c.shape ∧
  alphaRegion [] QuantizeTensor.inverseRegion s.inverseRegion = true ∧
  (alphaRegion [(100, s.rl.id), (101, s.rh.id)] (QuantizeTensor.region 32) s.region = true ∨
   alphaRegion [(100, s.rl.id), (101, s.rh.id)] (QuantizeTensor.region 64) s.region = true) ∧
  freshIds ρ (ids s) = true

set_option synthInstance.maxSize 4096 in
instance (ρ : Rho) (s : Site) : Decidable (Checks ρ s) := by
  unfold Checks
  infer_instance

theorem step_result {ctx : Ctx} {call : CallSem} {env : Env} {op : Op} {r : Value}
    {args : List RVal} {value : RVal} (hr : op.results = [r])
    (fetch : env.getAll op.operands = .ok args)
    (operation : opSem ctx call op args (evalRegions opSem call env op.regions) = .ok [value]) :
    stepOp opSem ctx call env op = .ok ((r.id, value) :: env) := by
  unfold stepOp
  rw [fetch, Outcome.bind_ok, operation]
  simp [Env.bindAll, hr]

/-- Every temporary and both literal captures are checked against the actual target IR.
The surrounding correspondence supplies only the original input values. -/
theorem target_sound {ρ : Rho} {s : Site} (checked : Checks ρ s)
    (ctx : Ctx) (call : CallSem) (envS envT : Env) (corresponds : Corr ρ envS envT)
    (w : Nat) (data scale zero : MerlinVerify.Tensor)
    (fetch : envS.getAll s.src.operands = .ok [.tensor data, .tensor scale, .tensor zero])
    (valid : Quantize.validInputs w s.c data scale zero = true) :
    ∃ envT', execOps opSem ctx call envT (targets s) = .ok envT' ∧
      Corr ((s.rs.id, s.rg.id) :: ρ)
        ((s.rs.id, .tensor (Quantize.result w s.c ctx.M data scale zero)) :: envS) envT' := by
  rcases checked with ⟨frame, headers, configured, mapped, n0, n1, n2, nl, nh, ng,
    a0, a1, a2, al, ah, ag, t0, t2, _, _, vl, vh, seg1, maps1, it1, segg, mapsg, itg,
    layout, _, geometry, inverseBody, body, fresh⟩
  rcases frameCheck_sound frame with ⟨rs, r0, r1, r2, rl, rh, rg, srcRegions, regs0, regs1, regs2, regsl, regsh, regsg⟩
  have header (op : Op) (member : op ∈ s.src :: targets s) :
      op.isTerminator = false ∧ unmodelledAttr? op.blind = none := by
    have h := List.all_eq_true.1 headers op member
    simp only [Bool.and_eq_true, Bool.not_eq_eq_eq_not, Bool.not_true, decide_eq_true_eq] at h
    exact ⟨h.1.1, h.1.2⟩
  have h0 := header s.invEmpty (by simp [targets])
  have h1 := header s.inverse (by simp [targets])
  have h2 := header s.empty (by simp [targets])
  have hl := header s.lower (by simp [targets])
  have hh := header s.upper (by simp [targets])
  have hg := header s.generic (by simp [targets])
  have unused := (freshIds_sound fresh).2
  have distinct := (freshIds_sound fresh).1
  let initialInverse := MerlinVerify.Tensor.full [] (.float .f32) .poison
  let inverse := inverseTensor ctx.M scale
  let initial := MerlinVerify.Tensor.full s.c.shape (.int 8) .poison
  let lo : RVal := .scalar (.some (.flt .f32 0xc3000000#32))
  let hi : RVal := .scalar (.some (.flt .f32 0x42fe0000#32))
  let e1 : Env := (s.r0.id, RVal.tensor initialInverse) :: envT
  let e2 : Env := (s.r1.id, RVal.tensor inverse) :: e1
  let e3 : Env := (s.r2.id, RVal.tensor initial) :: e2
  let e4 : Env := (s.rl.id, lo) :: e3
  let e5 : Env := (s.rh.id, hi) :: e4
  let out := Quantize.result w s.c ctx.M data scale zero
  let e6 : Env := (s.rg.id, RVal.tensor out) :: e5
  have c1 : Corr ρ envS e1 := DequantSite.Corr.temporary corresponds (unused _ (by simp [ids])) _
  have c2 : Corr ρ envS e2 := DequantSite.Corr.temporary c1 (unused _ (by simp [ids])) _
  have c3 : Corr ρ envS e3 := DequantSite.Corr.temporary c2 (unused _ (by simp [ids])) _
  have c4 : Corr ρ envS e4 := DequantSite.Corr.temporary c3 (unused _ (by simp [ids])) _
  have c5 : Corr ρ envS e5 := DequantSite.Corr.temporary c4 (unused _ (by simp [ids])) _
  have inputs1 : e1.getAll [s.x, s.scale, s.zero] = .ok [.tensor data, .tensor scale, .tensor zero] := by
    rw [← Corr.getAll c1 _ _ mapped]
    exact fetch
  have inputs5 : e5.getAll [s.x, s.scale, s.zero] = .ok [.tensor data, .tensor scale, .tensor zero] := by
    rw [← Corr.getAll c5 _ _ mapped]
    exact fetch
  obtain ⟨_, scaleValue, _⟩ := getAll_three inputs1
  obtain ⟨dataValue, _, zeroValue⟩ := getAll_three inputs5
  have d12 : s.r1.id ≠ s.r2.id := by intro h; simp [ids, h, List.nodup_cons] at distinct
  have d1l : s.r1.id ≠ s.rl.id := by intro h; simp [ids, h, List.nodup_cons] at distinct
  have d1h : s.r1.id ≠ s.rh.id := by intro h; simp [ids, h, List.nodup_cons] at distinct
  have d2l : s.r2.id ≠ s.rl.id := by intro h; simp [ids, h, List.nodup_cons] at distinct
  have d2h : s.r2.id ≠ s.rh.id := by intro h; simp [ids, h, List.nodup_cons] at distinct
  have dlh : s.rl.id ≠ s.rh.id := by intro h; simp [ids, h, List.nodup_cons] at distinct
  have b12 : (s.r1.id == s.r2.id) = false := by simpa using d12
  have b1l : (s.r1.id == s.rl.id) = false := by simpa using d1l
  have b1h : (s.r1.id == s.rh.id) = false := by simpa using d1h
  have b2l : (s.r2.id == s.rl.id) = false := by simpa using d2l
  have b2h : (s.r2.id == s.rh.id) = false := by simpa using d2h
  have blh : (s.rl.id == s.rh.id) = false := by simpa using dlh
  have step0 : stepOp opSem ctx call envT s.invEmpty = .ok e1 := by
    apply step_result (args := []) r0 (by simp [a0, Env.getAll, Outcome.mapM])
    rw [regs0]
    have hd : dialectOf s.invEmpty.name = "tensor" := by rw [n0]; cbv
    exact empty_sem ctx call s.invEmpty [] (.float .f32) h0.2 hd n0 t0
  have step1 : stepOp opSem ctx call e1 s.inverse = .ok e2 := by
    apply step_result (args := [.tensor scale, .tensor initialInverse]) r1
    · have ownValue : e1.get? s.r0.id = some (.tensor initialInverse) := by simp [e1, Env.get?]
      simp [a1, Env.getAll, Outcome.mapM, scaleValue, ownValue, Outcome.ofOption]
    · rw [regs1]
      exact generic_one ctx call s.inverse (.tensor scale) initialInverse inverse _ h1.2 n1
        seg1 maps1 it1 (by simp [r1])
        (inverse_from_valid valid inverseBody ctx call e1)
  have step2 : stepOp opSem ctx call e2 s.empty = .ok e3 := by
    apply step_result (args := []) r2 (by simp [a2, Env.getAll, Outcome.mapM])
    rw [regs2]
    have hd : dialectOf s.empty.name = "tensor" := by rw [n2]; cbv
    exact empty_sem ctx call s.empty s.c.shape (.int 8) h2.2 hd n2 t2
  have stepl : stepOp opSem ctx call e3 s.lower = .ok e4 :=
    constant_step ctx call e3 s.lower s.rl 0xc3000000#32 hl.2 nl rl al regsl vl
  have steph : stepOp opSem ctx call e4 s.upper = .ok e5 :=
    constant_step ctx call e4 s.upper s.rh 0x42fe0000#32 hh.2 nh rh ah regsh vh
  have inverseValue : e5.get? s.r1.id = some (.tensor inverse) := by
    simp [e5, e4, e3, e2, Env.get?, List.lookup, b12, b1l, b1h]
  have initialValue : e5.get? s.r2.id = some (.tensor initial) := by
    simp [e5, e4, e3, Env.get?, List.lookup, b2l, b2h]
  have capturesKnown : Corr [(100, s.rl.id), (101, s.rh.id)]
      (captures 0xc3000000#32 0x42fe0000#32) e5 := by
    intro x y mapped
    simp only [Rho.find, List.lookup] at mapped
    split at mapped
    · next hx =>
      simp only [Option.some.injEq] at mapped
      subst y
      have hx' : x = 100 := by simpa using hx
      subst x
      simp [captures, e5, e4, lo, hi, Env.get?, List.lookup, blh]
    · split at mapped
      · next hx =>
        simp only [Option.some.injEq] at mapped
        subst y
        have hx' : x = 101 := by simpa using hx
        subst x
        simp [captures, e5, lo, hi, Env.get?, List.lookup]
      · cases mapped
  have stepg : stepOp opSem ctx call e5 s.generic = .ok e6 := by
    apply step_result (args := [.tensor data, .tensor inverse, .tensor zero, .tensor initial]) rg
    · simp [ag, Env.getAll, Outcome.mapM, Outcome.ofOption, dataValue, inverseValue, zeroValue, initialValue]
    · rw [regsg]
      apply DequantSite.generic_sem ctx call s.generic [.tensor data, .tensor inverse, .tensor zero]
        initial _ s.maps s.iters out hg.2 (by rw [ng]; cbv) ng segg rfl mapsg itg (by simp [rg])
      rcases body with body | body
      · exact data_from_valid valid body ctx call e5 capturesKnown layout (structuredRanges?_spec geometry)
      · exact data_from_valid valid body ctx call e5 capturesKnown layout (structuredRanges?_spec geometry)
  refine ⟨e6, ?_, ?_⟩
  · rw [targets, execOps_cons_step _ _ _ _ _ _ h0.1, step0, Outcome.bind_ok]
    rw [execOps_cons_step _ _ _ _ _ _ h1.1, step1, Outcome.bind_ok]
    rw [execOps_cons_step _ _ _ _ _ _ h2.1, step2, Outcome.bind_ok]
    rw [execOps_cons_step _ _ _ _ _ _ hl.1, stepl, Outcome.bind_ok]
    rw [execOps_cons_step _ _ _ _ _ _ hh.1, steph, Outcome.bind_ok]
    rw [execOps_cons_step _ _ _ _ _ _ hg.1, stepg, Outcome.bind_ok]
    rfl
  · exact Corr.extend c5 [s.rs.id] [s.rg.id] [.tensor out] rfl rfl (by simp)
      (fun y hy => by simpa using unused y (by simp_all [ids]))

end MerlinVerify.Rules.QuantizeSite
