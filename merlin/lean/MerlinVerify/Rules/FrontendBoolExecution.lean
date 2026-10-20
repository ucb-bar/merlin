import MerlinVerify.Rules.FrontendBoolFptosi
import MerlinVerify.Sem.Ops
import MerlinVerify.Check.Step
import MerlinVerify.Check.Support

namespace MerlinVerify.Rules.FrontendBoolExecution

open MerlinVerify MerlinVerify.Spec.FrontendIntent
open MerlinVerify.Rules.FrontendBoolFptosi

/-- The two operations emitted at a supported site, without provenance annotations. -/
def zero (z : Nat) : Op :=
  .mk "arith.constant" [{ id := z, ty := .float .f32 }] []
    [("value", .float 0 (.float .f32))] [] [] []

theorem zero_sem (ctx : Ctx) (call : CallSem) (z : Nat) :
    Sem.Ops.opSem ctx call (zero z) [] [] =
      .ok [.scalar (.some (.flt .f32 0))] := by
  have hd : Sem.Ops.dialectOf "arith.constant" = "arith" := by cbv
  simp [Sem.Ops.opSem, Sem.Ops.unmodelledAttr?, Sem.Ops.interpretedAttrs,
    Sem.Ops.dispatch, hd, Op.blind, Op.attr?, Op.name, Op.props, Op.attrs,
    Op.results, Op.successors, Sem.Ops.Arith.sem, zero, scalarOfAttr]

/-- The same local semantic claim for an exported comparison carrying provenance. The proof reads
its actual interpreted key and modeled-attribute check rather than assuming annotations are inert. -/
def sourceVal : Option (FBits .f32) → Val
  | none => .poison
  | some x => .some (.flt .f32 x)

theorem cmp_sem_checked (ctx : Ctx) (call : CallSem) (op : Op)
    (hm : Sem.Ops.unmodelledAttr? op.blind = none)
    (hd : Sem.Ops.dialectOf op.name = "arith")
    (hn : op.name = "arith.cmpf")
    (hr : Sem.Ops.Arith.resultElt op.blind = some (.int 1))
    (hk : Sem.Ops.Arith.floatKey op.blind = uneKey)
    (hune : UneCorrect ctx.M) (x : FBits .f32) :
    Sem.Ops.opSem ctx call op
      [.scalar (.some (.flt .f32 x)), .scalar (.some (.flt .f32 0))] [] =
      .ok [.scalar (f32ToBool (some x))] := by
  rw [Check.opSem_modelled hm]
  simp only [Sem.Ops.dispatch, Check.Op.blind_name, hd]
  simp [Sem.Ops.Arith.sem, Check.Op.blind_name, hn, Sem.Ops.Arith.binOp,
    Sem.Ops.Arith.pureN, hr, liftEager, Sem.Ops.Arith.boolVal, f32ToBool]
  rw [hk]
  have hz0 : (0#FKind.f32.width : FBits .f32) = 0 := by rfl
  rw [hz0, hune x]
  by_cases hz : x.toNat % 2^31 = 0 <;> simp [hz]

theorem cmp_sem_checked_poison (ctx : Ctx) (call : CallSem) (op : Op)
    (hm : Sem.Ops.unmodelledAttr? op.blind = none)
    (hd : Sem.Ops.dialectOf op.name = "arith")
    (hn : op.name = "arith.cmpf")
    (hr : Sem.Ops.Arith.resultElt op.blind = some (.int 1)) :
    Sem.Ops.opSem ctx call op
      [.scalar .poison, .scalar (.some (.flt .f32 0))] [] =
      .ok [.scalar .poison] := by
  rw [Check.opSem_modelled hm]
  simp only [Sem.Ops.dispatch, Check.Op.blind_name, hd]
  simp [Sem.Ops.Arith.sem, Check.Op.blind_name, hn, Sem.Ops.Arith.binOp,
    Sem.Ops.Arith.pureN, hr, liftEager]

theorem cmp_sem_checked_option (ctx : Ctx) (call : CallSem) (op : Op)
    (hm : Sem.Ops.unmodelledAttr? op.blind = none)
    (hd : Sem.Ops.dialectOf op.name = "arith")
    (hn : op.name = "arith.cmpf")
    (hr : Sem.Ops.Arith.resultElt op.blind = some (.int 1))
    (hk : Sem.Ops.Arith.floatKey op.blind = uneKey)
    (hune : UneCorrect ctx.M) (x : Option (FBits .f32)) :
    Sem.Ops.opSem ctx call op
      [.scalar (sourceVal x), .scalar (.some (.flt .f32 0))] [] =
      .ok [.scalar (f32ToBool x)] := by
  cases x with
  | none => simpa [sourceVal, f32ToBool] using cmp_sem_checked_poison ctx call op hm hd hn hr
  | some bits => simpa [sourceVal] using cmp_sem_checked ctx call op hm hd hn hr hk hune bits

/-- Continuation theorem for the actual exported comparison op, including its provenance.
The premises are syntactic and semantic facts recomputed from that op by the boundary checker. -/
theorem checked_site_then_tail (ctx : Ctx) (call : CallSem) (env : Env) (tail : List Op)
    (op : Op) (source z result : Nat) (x : Option (FBits .f32))
    (hm : Sem.Ops.unmodelledAttr? op.blind = none)
    (hd : Sem.Ops.dialectOf op.name = "arith")
    (hn : op.name = "arith.cmpf")
    (hr : Sem.Ops.Arith.resultElt op.blind = some (.int 1))
    (hk : Sem.Ops.Arith.floatKey op.blind = uneKey)
    (hoperands : op.operands = [source, z])
    (hresults : op.results = [{ id := result, ty := .int 1 }])
    (hregions : op.regions = [])
    (hune : UneCorrect ctx.M) (hsource : env.get? source = some (.scalar (sourceVal x)))
    (hfresh : source ≠ z) :
    evalOps Sem.Ops.opSem ctx call env (zero z :: op :: tail) =
      evalOps Sem.Ops.opSem ctx call
        ((result, .scalar (f32ToBool x)) :: (z, .scalar (.some (.flt .f32 0))) :: env) tail := by
  rw [Check.evalOps_cons_step _ _ _ _ _ _ (by rfl)]
  simp only [Check.stepOp]
  have hread : env.getAll (zero z).operands = .ok [] := by rfl
  have hregs : evalRegions Sem.Ops.opSem call env (zero z).regions = [] := by rfl
  rw [hread]
  simp only [Outcome.bind_ok, hregs]
  rw [zero_sem]
  simp only [Outcome.bind_ok]
  have hbind : env.bindAll (zero z).results [.scalar (.some (.flt .f32 0))] =
      .ok ((z, .scalar (.some (.flt .f32 0))) :: env) := by rfl
  rw [hbind]
  simp only [Outcome.bind_ok]
  have ht : op.isTerminator = false := by simp [Op.isTerminator, hn, terminatorNames]
  rw [Check.evalOps_cons_step _ _ _ _ _ _ ht]
  simp only [Check.stepOp]
  let env' : Env := (z, .scalar (.some (.flt .f32 0))) :: env
  have hread2 : env'.getAll op.operands =
      .ok [.scalar (sourceVal x), .scalar (.some (.flt .f32 0))] := by
    have hs : env.lookup source = some (.scalar (sourceVal x)) := hsource
    have heq : (source == z) = false := by simpa [beq_iff_eq] using hfresh
    have hl : List.lookup source ((z, .scalar (.some (.flt .f32 0))) :: env) =
        env.lookup source := by simp [List.lookup, heq]
    simp only [env', Env.getAll, Env.get?, hoperands, Outcome.mapM,
      Outcome.ofOption, Outcome.bind_ok]
    rw [hl, hs]
    simp [List.lookup]
  have hregs2 : evalRegions Sem.Ops.opSem call env' op.regions = [] := by
    simp [hregions, evalRegions]
  rw [hread2]
  simp only [Outcome.bind_ok]
  rw [hregs2]
  rw [cmp_sem_checked_option ctx call op hm hd hn hr hk hune x]
  simp only [Outcome.bind_ok]
  have hbind2 : env'.bindAll op.results [.scalar (f32ToBool x)] =
      .ok ((result, .scalar (f32ToBool x)) :: env') := by simp [Env.bindAll, hresults]
  rw [hbind2]
  rfl

/-- Tensor lifting of the same poison-aware A2 contract. Each point may independently be poison,
NaN, a signed zero, or another f32 bit pattern. -/
def pointwiseResult (M : FloatModel) (shape : List Nat)
    (x : Idx → Option (FBits .f32)) : Tensor :=
  Tensor.canon shape (.int 1) fun p =>
    match x p with
    | none => .poison
    | some bits => Sem.Ops.Arith.boolVal (M.cmp uneKey .f32 bits 0)

theorem pointwise_a2 (M : FloatModel) (hune : UneCorrect M) (shape : List Nat)
    (x : Idx → Option (FBits .f32)) :
    pointwiseResult M shape x = f32ToBoolTensor shape x := by
  unfold pointwiseResult f32ToBoolTensor
  congr 1
  funext p
  cases hp : x p with
  | none => simp [f32ToBool]
  | some bits =>
    have hu := hune bits
    have hz0 : (0#FKind.f32.width : FBits .f32) = 0 := by rfl
    by_cases hz : bits.toNat % 2^31 = 0
    · simp only [hz, ↓reduceIte] at hu
      simp [f32ToBool, hz, Sem.Ops.Arith.boolVal]
      simpa only [hz0] using hu
    · simp only [hz, ↓reduceIte] at hu
      simp [f32ToBool, hz, Sem.Ops.Arith.boolVal]
      simpa only [hz0] using hu

#print axioms checked_site_then_tail
#print axioms pointwise_a2

end MerlinVerify.Rules.FrontendBoolExecution
