/- Exact semantic witnesses for the syntax-selected parallel fusion candidate. -/
import MerlinVerify.Check.ParallelFusionSyntax

namespace MerlinVerify.Check.ParallelFusionProof

open MerlinVerify MerlinVerify.Sem.Ops MerlinVerify.Sem.Ops.Linalg
open ParallelFusionBody ParallelFusionRuntime ParallelFusionSyntax
open MerlinVerify.Rules.ParallelFusionScalar

def produced (g : Generic) (input : Tensor) (pt : Idx) : Val :=
  g.body.primitive.kind.value g.width (input.get pt) (input.get pt)

def composed (p c : Generic) (input : Tensor) (pt : Idx) : Val :=
  c.body.primitive.kind.value c.width (produced p input pt) (produced p input pt)

/-- Execution of each actual source generic is total on typed elements, including poison. -/
theorem generic_total {ib : Nat} {g : Generic} (hg : g.Valid)
    (hregions : g.operation.regions = [g.body.region])
    {input init : Tensor} (hi : input.shape = g.shape) (ho : init.shape = g.shape)
    (hv : ∀ pt, Idx.InBounds g.shape pt → Val.HasTy ib (.int g.width) (input.get pt))
    (ctx : Ctx) (call : CallSem) (env : Env) :
    opSem ctx call g.operation [.tensor input, .tensor init]
      (evalRegions opSem call env g.operation.regions) =
      .ok [.tensor (Tensor.canon g.shape init.elt (produced g input))] := by
  rw [hregions]
  simp only [evalRegions]
  apply generic_success hg.1
  apply Rules.ParallelFusion.identity_generic ctx g.shape [input] init
  · intro t ht
    simp only [List.singleton_append, List.mem_cons, List.not_mem_nil, or_false] at ht
    rcases ht with rfl | rfl <;> assumption
  · intro pt hp
    obtain ⟨hprimitive, hop, htn, hto⟩ := Body.valid_core hg.2.2.1
    exact repeated_point hprimitive hop htn hto hi ho hp (hv pt hp) ctx call env

theorem fused_total {ib : Nat} {p c : Generic} (hp : p.Valid) (hc : c.Valid)
    (hs : c.shape = p.shape) (hw : c.width = p.width)
    {input init : Tensor} (hi : input.shape = p.shape) (ho : init.shape = p.shape)
    (hv : ∀ pt, Idx.InBounds p.shape pt → Val.HasTy ib (.int p.width) (input.get pt))
    (ctx : Ctx) (call : CallSem) (env : Env) :
    opSem ctx call (fused p c) [.tensor input, .tensor init]
      (evalRegions opSem call env (fused p c).regions) =
      .ok [.tensor (Tensor.canon p.shape init.elt (composed p c input))] := by
  have hblind : (fused p c).blind = c.operation.blind := by simp [fused]
  have hregions : (fused p c).regions = [fusedRegion p c] := by simp [fused]
  rw [hregions]
  simp only [evalRegions]
  rw [opSem_blind hblind]
  apply generic_success hc.1
  rw [hs]
  apply Rules.ParallelFusion.identity_generic ctx p.shape [input] init
  · intro t ht
    simp only [List.singleton_append, List.mem_cons, List.not_mem_nil, or_false] at ht
    rcases ht with rfl | rfl <;> assumption
  · intro pt hpt
    obtain ⟨hpbody, hop, _, _⟩ := Body.valid_core hp.2.2.1
    obtain ⟨hcbody, _, htn, hto⟩ := Body.valid_core hc.2.2.1
    rw [hw] at hcbody
    unfold composed
    rw [hw]
    exact fused_point hpbody hcbody hop htn hto hi ho hpt (hv pt hpt) ctx call env

theorem empty_total {ib : Nat} {p : Generic}
    (ht : vtyOf ib p.result.ty = some (.tensor p.shape (.int p.width)))
    (ctx : Ctx) (call : CallSem) :
    opSem ctx call (empty p) [] [] = .ok [.tensor (Tensor.full p.shape (.int p.width) .poison)] := by
  have hs := staticTensor_of_vty ht
  have hm : unmodelledAttr? (empty p).blind = none := by rfl
  have hn : (empty p).name = "tensor.empty" := rfl
  have hd : dialectOf (empty p).name = "tensor" := by
    rw [hn]
    cbv
  have hresult : Sem.Ops.Tensor.resultTensorTy (empty p).blind = some (p.shape, .int p.width) := by
    rw [resultTensorTy_blind]
    simpa [Sem.Ops.Tensor.resultTensorTy, empty, Op.results] using hs
  rw [tensor_sem_of hm hd]
  simp only [Sem.Ops.Tensor.sem, Op.blind_name, hn, hresult]

theorem step_ok {op : Op} {r : Value} {value : RVal} {args : List RVal}
    {env : Env} {ctx : Ctx} {call : CallSem}
    (ha : env.getAll op.operands = .ok args) (hr : op.results = [r])
    (hs : opSem ctx call op args (evalRegions opSem call env op.regions) = .ok [value]) :
    stepOp opSem ctx call env op = .ok ((r.id, value) :: env) := by
  simp [stepOp, ha, hs, Env.bindAll, hr, Outcome.bind_ok]

theorem diagonal_pair (Γ : TEnv) (env : Env) {id : ValueId}
    (hf : (SplatNormalize.diagonal Γ).notImage id) (left right : RVal) :
    Corr (SplatNormalize.diagonal Γ) ((id, left) :: env) ((id, right) :: env) := by
  intro x y h
  have he : x = y := by
    induction Γ with
    | nil => simp [SplatNormalize.diagonal, Rho.find] at h
    | cons p ps ih =>
      simp only [SplatNormalize.diagonal, List.map_cons, Rho.find, List.lookup] at h
      split at h
      · simp_all only [beq_iff_eq, Option.some.injEq]
      · exact ih (fun q hq => hf q (List.mem_cons_of_mem _ hq)) h
  have hne : y ≠ id := by
    obtain ⟨p, hp, hpy⟩ := lookup_mem_image h
    exact hpy ▸ hf p hp
  have hb : (y == id) = false := by simpa using hne
  rw [he]
  simp only [Env.get?, List.lookup, hb]

/-- Both source loops and the actual fresh destination are evaluated. The vanished producer
binding may differ only where the recursive continuation alpha check proves it unobservable. -/
theorem pair_sound {ib : Nat} {Γ : TEnv} {p c : Generic} {tail : List Op}
    (hl : Legal Γ p c tail)
    (hpO : p.operation.operands = [p.input, p.destination])
    (hcO : c.operation.operands = [c.input, c.destination])
    (hpR : p.operation.regions = [p.body.region])
    (hcR : c.operation.regions = [c.body.region])
    (hpT : vtyOf ib p.result.ty = some (.tensor p.shape (.int p.width)))
    {ctx : Ctx} {call : CallSem} {env : Env} (he : EnvTyped ib Γ env) :
    evalOps opSem ctx call env (empty p :: fused p c :: tail) =
      evalOps opSem ctx call env (p.operation :: c.operation :: tail) := by
  obtain ⟨hp, hc, hshape, hwidth, hinput, hself, hdest, hI, hPI, hCI, hfresh, htail⟩ := hl
  obtain ⟨input, hInput, hi, _, hInputTyped⟩ := tensor_lookup he hI
  obtain ⟨pinit, hPinit, hps, hpe, _⟩ := tensor_lookup he hPI
  obtain ⟨cinit, hCinit, hcs, hce, _⟩ := tensor_lookup he hCI
  let pv : Tensor := Tensor.canon p.shape pinit.elt (produced p input)
  let qv : Tensor := Tensor.canon p.shape cinit.elt (composed p c input)
  let fresh : Tensor := Tensor.full p.shape (.int p.width) .poison
  let sourceEnv : Env := (p.result.id, RVal.tensor pv) :: env
  let targetEnv : Env := (p.result.id, RVal.tensor fresh) :: env
  have hP := generic_total hp hpR hi hps hInputTyped ctx call env
  have hPV : ∀ pt, Idx.InBounds c.shape pt → Val.HasTy ib (.int c.width) (pv.get pt) := by
    intro pt hpt
    have hpt' : Idx.InBounds p.shape pt := hshape ▸ hpt
    simpa [pv, Tensor.canon, produced, hpt', hwidth] using
      value_typed ib p.body.primitive.kind p.width (input.get pt) (input.get pt)
  have hC := generic_total hc hcR (input := pv) (init := cinit)
    (by simpa [pv, Tensor.canon] using hshape.symm) (hcs.trans hshape.symm) hPV ctx call sourceEnv
  have hq : Tensor.canon c.shape cinit.elt (produced c pv) = qv := by
    dsimp [qv]
    rw [hshape]
    apply Tensor.canon_ext
    intro pt hpt
    simp [produced, composed, pv, Tensor.canon, hpt]
  rw [hq] at hC
  have hE := empty_total hpT ctx call
  have hF := fused_total hp hc hshape hwidth hi (init := fresh) rfl hInputTyped ctx call targetEnv
  have hqF : Tensor.canon p.shape fresh.elt (composed p c input) = qv := by
    simp [fresh, qv, Tensor.full, Tensor.canon, hce]
  rw [hqF] at hF
  have hPargs : env.getAll p.operation.operands = .ok [.tensor input, .tensor pinit] := by
    simp [hpO, Env.getAll, Outcome.mapM, Outcome.ofOption, hInput, hPinit]
  have hCInput : sourceEnv.get? c.input = some (.tensor pv) := by
    simp [sourceEnv, Env.get?, List.lookup, hinput]
  have hCDest : sourceEnv.get? c.destination = some (.tensor cinit) := by
    have hkeep : sourceEnv.get? c.destination = env.get? c.destination := by
      have hb : (c.destination == p.result.id) = false := by simpa using hdest
      simp only [sourceEnv, Env.get?, List.lookup, hb]
    exact hkeep.trans hCinit
  have hCargs : sourceEnv.getAll c.operation.operands = .ok [.tensor pv, .tensor cinit] := by
    simp [hcO, Env.getAll, Outcome.mapM, Outcome.ofOption, hCInput, hCDest]
  have hFInput : targetEnv.get? p.input = some (.tensor input) := by
    have hkeep : targetEnv.get? p.input = env.get? p.input := by
      have hb : (p.input == p.result.id) = false := by simpa using hself
      simp only [targetEnv, Env.get?, List.lookup, hb]
    exact hkeep.trans hInput
  have hFEmpty : targetEnv.get? p.result.id = some (.tensor fresh) := by
    simp [targetEnv, Env.get?, List.lookup]
  have hFargs : targetEnv.getAll (fused p c).operands = .ok [.tensor input, .tensor fresh] := by
    simp only [fused, SplatNested.withRegions_operands, withOperands_operands]
    simp [Env.getAll, Outcome.mapM, Outcome.ofOption, hFInput, hFEmpty]
  have hsP := step_ok hPargs hp.1.2.2.2.2.2 hP
  have hsC := step_ok hCargs hc.1.2.2.2.2.2 hC
  have hsE : stepOp opSem ctx call env (empty p) = .ok targetEnv := by
    apply step_ok (args := []) (r := p.result) (by rfl) (by rfl)
    exact hE
  have hFres : (fused p c).results = [c.result] := by
    simpa [fused] using hc.1.2.2.2.2.2
  have hsF := step_ok hFargs hFres hF
  have htP : p.operation.isTerminator = false := by
    simp [Op.isTerminator, terminatorNames, hp.1.1]
  have htC : c.operation.isTerminator = false := by
    simp [Op.isTerminator, terminatorNames, hc.1.1]
  have htE : (empty p).isTerminator = false := by
    simp [Op.isTerminator, terminatorNames, empty, Op.name]
  have htF : (fused p c).isTerminator = false := by simpa [fused] using htC
  rw [evalOps_cons_step _ _ _ _ _ _ htE, hsE]
  simp only [Outcome.bind_ok]
  rw [evalOps_cons_step _ _ _ _ _ _ htF, hsF]
  simp only [Outcome.bind_ok]
  rw [evalOps_cons_step _ _ _ _ _ _ htP, hsP]
  simp only [Outcome.bind_ok]
  change _ = evalOps opSem ctx call sourceEnv (c.operation :: tail)
  rw [evalOps_cons_step _ _ _ _ _ _ htC, hsC]
  simp only [Outcome.bind_ok]
  have hfp : (SplatNormalize.diagonal Γ).notImage p.result.id :=
    (freshIds_sound hfresh).2 _ (by simp)
  have hfc : (SplatNormalize.diagonal Γ).notImage c.result.id :=
    (freshIds_sound hfresh).2 _ (by simp)
  have hcorr := Corr.extend (diagonal_pair Γ env hfp (.tensor pv) (.tensor fresh))
    [c.result.id] [c.result.id] [.tensor qv] rfl rfl (by simp)
    (by intro y hy; simp only [List.mem_singleton] at hy; subst y; exact hfc)
  exact (alphaOps_sound call call _ tail tail ctx _ _ htail hcorr (fun _ _ => rfl)).symm

end MerlinVerify.Check.ParallelFusionProof
