/- Per-tensor dequantization at any zero-point width. `arith.sitofp` converts at the width of the
value it reads, so the per-tensor body `sitofp(q) - sitofp(z)` then `* s` computes the contract for
the zero point that actually arrives, whatever integer type its block argument is annotated with.
This supplies the i64 per-tensor case of the C1 site rule beside the existing i32 proofs. -/
import MerlinVerify.Rules.DequantSource

namespace MerlinVerify.Rules.DequantPerTensor

open MerlinVerify MerlinVerify.Sem.Ops MerlinVerify.Sem.Ops.Linalg
open MerlinVerify.Rules.DequantScalar MerlinVerify.Rules.DequantTensor MerlinVerify.Rules.DequantSource

/-- Arguments: q, scale, zero point (at width `w`), unused destination element. -/
def inputsAt (w : Nat) (q : Option (BitVec 8)) (z : Option (BitVec w)) (s : Option (FBits .f32))
    (init : Val) : Env :=
  [(0, .scalar (intValue 8 q)), (1, .scalar (floatValue s)), (2, .scalar (intValue w z)),
   (3, .scalar init)]

private theorem dialect_sitofp : Sem.Ops.dialectOf "arith.sitofp" = "arith" := by cbv
private theorem dialect_subf : Sem.Ops.dialectOf "arith.subf" = "arith" := by cbv
private theorem dialect_mulf : Sem.Ops.dialectOf "arith.mulf" = "arith" := by cbv

private theorem floatKey_default (name : String) (id : Nat) (ty : Ty) (args : List Nat) :
    Sem.Ops.Arith.floatKey (.mk name [{ id, ty }] args [] [] [] []) = Spec.Dequant.defaultKey name := by
  apply congrArg (fun modifiers => ({ op := name, modifiers } : FloatOpKey))
  simp only [Sem.Ops.Arith.fastmathKey, Sem.Ops.Arith.attrKey, Op.attr?, Op.props, Op.attrs, List.lookup]
  cbv

private theorem floatKey_explicitNone (name : String) (id : Nat) (args : List Nat) :
    Sem.Ops.Arith.floatKey (.mk name [{ id, ty := .float .f32 }] args
      [("fastmath", .enum "arith" "fastmath" "none")] [] [] []) = Spec.Dequant.defaultKey name := by
  apply congrArg (fun modifiers => ({ op := name, modifiers } : FloatOpKey))
  simp only [Sem.Ops.Arith.fastmathKey, Sem.Ops.Arith.attrKey,
    Op.attr?, Op.props, Op.attrs, List.lookup]
  cbv

set_option maxHeartbeats 2000000 in
set_option maxRecDepth 4096 in
/-- Exact per-tensor body execution at zero-point width `w`, poison operands included. -/
theorem body_sound_at (w : Nat) (ctx : Ctx) (call : CallSem) (q : Option (BitVec 8))
    (z : Option (BitVec w)) (s : Option (FBits .f32)) (init : Val) :
    evalOps Sem.Ops.opSem ctx call (inputsAt w q z s init) perTensorBody =
      .ok [.scalar (Spec.Dequant.perTensorValueAt w ctx.M q z s)] := by
  cases q <;> cases z <;> cases s <;>
  simp [inputsAt, intValue, floatValue, Spec.Dequant.perTensorValueAt, Spec.Dequant.perTensorAt,
    evalOps, perTensorBody, op, floatOp, yield, Env.getAll, Env.get?, Env.bindAll,
    List.lookup, Outcome.mapM, Outcome.ofOption, Sem.Ops.opSem, Sem.Ops.unmodelledAttr?,
    Sem.Ops.interpretedAttrs, Sem.Ops.dispatch,
    Op.blind, Op.name, Op.results, Op.attrs, Op.props, Op.successors,
    dialect_sitofp, dialect_subf, dialect_mulf, Sem.Ops.Arith.sem,
    Sem.Ops.Arith.unOp, Sem.Ops.Arith.binOp, Sem.Ops.Arith.pureN, Sem.Ops.Arith.resultElt,
    liftEager, Sem.Ops.Arith.fltBin, floatKey_default, floatKey_explicitNone,
    Op.isTerminator, terminatorNames]

/-- A per-tensor body: four block arguments and the per-tensor operations, under any typing. -/
def bodyMatches (args : List Value) (ops : List Op) : Bool :=
  decide (args.length = 4) && Check.freshIds [] (args.map (·.id)) &&
    Check.alphaOps (argMap args) perTensorBody ops

/-- The block-argument types the per-tensor site accepts: an i32 or an i64 zero point. -/
def checkBodyAt (args : List Value) (ops : List Op) : Bool :=
  (decide (args.map (·.ty) = [.int 8, .float .f32, .int 32, .float .f32]) ||
    decide (args.map (·.ty) = [.int 8, .float .f32, .int 64, .float .f32])) &&
  Check.freshIds [] (args.map (·.id)) && Check.alphaOps (argMap args) perTensorBody ops

theorem checkBody_matches {args : List Value} {ops : List Op} (h : checkBody false args ops = true) :
    bodyMatches args ops = true := by
  simp only [checkBody, body, Bool.false_eq_true, ↓reduceIte, Bool.and_eq_true, decide_eq_true_eq] at h
  have hl : args.length = 4 := by simpa using congrArg List.length h.1.1
  simp only [bodyMatches, Bool.and_eq_true, decide_eq_true_eq]
  exact ⟨⟨hl, h.1.2⟩, h.2⟩

theorem checkBodyAt_matches {args : List Value} {ops : List Op} (h : checkBodyAt args ops = true) :
    bodyMatches args ops = true := by
  simp only [checkBodyAt, Bool.and_eq_true, Bool.or_eq_true, decide_eq_true_eq] at h
  have hl : args.length = 4 := by
    rcases h.1.1 with h | h <;> simpa using congrArg List.length h
  simp only [bodyMatches, Bool.and_eq_true, decide_eq_true_eq]
  exact ⟨⟨hl, h.1.2⟩, h.2⟩

/-- Region execution at the zero-point width of the values that arrive. -/
theorem region_sound_at {args : List Value} {ops : List Op} (h : bodyMatches args ops = true)
    (w : Nat) (ctx : Ctx) (call : CallSem) (env : Env) (q : Option (BitVec 8))
    (z : Option (BitVec w)) (s : Option (FBits .f32)) (init : Val) :
    evalRegion Sem.Ops.opSem ctx call env (.mk [.mk args ops])
      [.scalar (intValue 8 q), .scalar (floatValue s), .scalar (intValue w z), .scalar init] =
      .ok [.scalar (Spec.Dequant.perTensorValueAt w ctx.M q z s)] := by
  simp only [bodyMatches, Bool.and_eq_true, decide_eq_true_eq] at h
  obtain ⟨⟨hl, hfresh⟩, halpha⟩ := h
  have hf := (Check.freshIds_sound hfresh).1
  have hc := Check.Corr.extend
    (ρ := []) (envS := []) (envT := env) (fun _ _ he => by simp [Check.Rho.find] at he)
    [0, 1, 2, 3] (args.map (·.id))
    [.scalar (intValue 8 q), .scalar (floatValue s), .scalar (intValue w z), .scalar init]
    (by rfl) (by simpa using hl) hf (fun _ _ _ he => by simp at he)
  have hc' : Check.Corr (argMap args) (inputsAt w q z s init)
      ((args.map (·.id)).zip
        [.scalar (intValue 8 q), .scalar (floatValue s), .scalar (intValue w z), .scalar init] ++ env) := by
    simpa [argMap, inputsAt] using hc
  have he := Check.alphaOps_sound call call (argMap args) perTensorBody ops ctx
    (inputsAt w q z s init) _ halpha hc' (fun _ _ => rfl)
  simpa [evalRegion, evalBlocks, evalBlock, Env.bindAll, hl, ← he] using body_sound_at w ctx call q z s init

/-- The per-tensor loop nest at zero-point width `w`. -/
theorem checked_generic_sound_at {axis : Nat} {maps : List AffineMap} {iters : List IterKind}
    {args : List Value} {ops : List Op} (w : Nat)
    (ctx : Ctx) (call : CallSem) (env : Env) (qv sv zv : RVal) (t : Tensor)
    (hl : checkLayout false axis t.shape [] maps iters = true)
    (hb : bodyMatches args ops = true)
    (hshapes : [qv, sv, zv, .tensor t].map shapeOf = (operandTypes t.shape []).map Check.VTy.shape)
    (q : Idx → Option (BitVec 8)) (z : Option (BitVec w)) (s : Option (FBits .f32))
    (hq : ∀ p, Idx.InBounds t.shape p →
      readAt qv (identityMap t.shape.length) p = .ok (intValue 8 (q p)))
    (hs : ∀ p, Idx.InBounds t.shape p →
      readAt sv (parameterMap false t.shape.length axis) p = .ok (floatValue s))
    (hz : ∀ p, Idx.InBounds t.shape p →
      readAt zv (parameterMap false t.shape.length axis) p = .ok (intValue w z)) :
    genericSem ctx maps iters [qv, sv, zv] [.tensor t]
      (fun ctx => evalRegion Sem.Ops.opSem ctx call env (.mk [.mk args ops])) =
      .ok [.tensor (Tensor.canon t.shape t.elt
        (fun p => Spec.Dequant.perTensorValueAt w ctx.M (q p) z s))] := by
  obtain ⟨-, -, hm, hi, hstruct⟩ := checkLayout_sound hl
  have hd : maps.any (fun m => m.numDims ≠ iters.length || m.numSyms ≠ 0) = false := by
    rw [List.any_eq_false]
    intro m hm
    simp [(hstruct.dims m hm).1, (hstruct.dims m hm).2]
  have hr : (maps.zip ([qv, sv, zv, .tensor t].map shapeOf)).any
      (fun p => p.1.results.length ≠ p.2.length) = false := by
    rw [hshapes, List.zip_map_right, List.any_map, List.any_eq_false]
    intro p hp
    have hin := List.forall₂_iff_zip.1 hstruct.inBounds |>.2 hp
    simp only [Check.mapInBounds, Bool.and_eq_true, beq_iff_eq] at hin
    simp [Function.comp, hin.1]
  have hranges : loopRanges maps ([qv, sv, zv, .tensor t].map shapeOf) iters.length = .ok t.shape := by
    rw [hshapes]
    exact Check.mapM_ofOption _ _ _ _ hstruct.loops
  subst hm
  subst hi
  apply genericSem_parallel1 ctx _ _ _ t _ t.shape _ (by rfl)
    (by simpa [mapsFor] using hd) (by simpa [mapsFor] using hr) (by simpa [mapsFor] using hranges)
    rfl (by rfl)
  intro p hp
  simp [bodyAt, mapsFor, hq p hp, hs p hp, hz p hp, readAt_identity hp, region_sound_at hb]

/-- Actual runtime tensors: a per-tensor generic computes the per-tensor contract at the zero-point
width of the source's inputs. -/
theorem generic_result_sound_at {c : Dequant.Config} {q s z t : Tensor}
    {maps : List AffineMap} {iters : List IterKind} {args : List Value} {ops : List Op} (w : Nat)
    (ctx : Ctx) (call : CallSem) (env : Env)
    (hv : Dequant.validPerTensorAt w c q s z = true) (ht : t.shape = c.shape) (he : t.elt = .float .f32)
    (hl : checkLayout c.channel c.axis c.shape (Dequant.paramShape c) maps iters = true)
    (hb : bodyMatches args ops = true) :
    genericSem ctx maps iters [.tensor q, .tensor s, .tensor z] [.tensor t]
      (fun ctx => evalRegion opSem ctx call env (.mk [.mk args ops])) =
      .ok [.tensor (Dequant.resultPerTensorAt w c ctx.M q s z)] := by
  simp only [Dequant.validPerTensorAt, Bool.and_eq_true, decide_eq_true_eq, Bool.not_eq_eq_eq_not,
    Bool.not_true] at hv
  obtain ⟨⟨⟨⟨hch, hqshape, hsshape, hzshape⟩, hq⟩, hs⟩, hz⟩ := hv
  have hparam : Dequant.paramShape c = [] := by simp [Dequant.paramShape, hch]
  rw [hparam, hch] at hl
  have hempty : Idx.InBounds [] ([] : Idx) := (Idx.inBounds_nil_iff []).2 rfl
  have hqread (p : Idx) (hp : Idx.InBounds t.shape p) :
      readAt (.tensor q) (identityMap t.shape.length) p = .ok (intValue 8 (Dequant.bits 8 (q.get p))) := by
    have hqp : Idx.InBounds q.shape p := by simpa [hqshape, ht] using hp
    rw [bits_roundtrip (tensorTyped_get hq hqp)]
    simpa [ht, hqshape] using readAt_identity hqp
  have hsread (p : Idx) (_ : Idx.InBounds t.shape p) :
      readAt (.tensor s) (parameterMap false t.shape.length c.axis) p =
        .ok (floatValue (Dequant.float32 (s.get []))) := by
    rw [float_roundtrip (tensorTyped_get hs (hsshape ▸ hempty))]
    simp [readAt, parameterMap, evalMap, toIdx, hsshape]
  have hzread (p : Idx) (_ : Idx.InBounds t.shape p) :
      readAt (.tensor z) (parameterMap false t.shape.length c.axis) p =
        .ok (intValue w (Dequant.bits w (z.get []))) := by
    rw [bits_roundtrip (tensorTyped_get hz (hzshape ▸ hempty))]
    simp [readAt, parameterMap, evalMap, toIdx, hzshape]
  simpa [Dequant.resultPerTensorAt, Spec.Dequant.perTensorAtTensor, ht, he] using
    checked_generic_sound_at w ctx call env (.tensor q) (.tensor s) (.tensor z) t
      (by simpa [ht] using hl) hb
      (by simp [operandTypes, Check.VTy.shape, shapeOf, hqshape, hsshape, hzshape, ht])
      (fun p => Dequant.bits 8 (q.get p)) (Dequant.bits w (z.get [])) (Dequant.float32 (s.get []))
      hqread hsread hzread

/-- At width 32 the per-tensor contract is the existing i32 contract. -/
theorem resultPerTensorAt_32 {c : Dequant.Config} (hch : c.channel = false) (M : FloatModel)
    (q s z : Tensor) : Dequant.resultPerTensorAt 32 c M q s z = Dequant.result c M q s z := by
  have hv (a : Option (BitVec 8)) (b : Option (BitVec 32)) (e : Option (FBits .f32)) :
      Spec.Dequant.perTensorValueAt 32 M a b e = Spec.Dequant.scalarValue false M a b e := by
    cases a <;> cases b <;> cases e <;> rfl
  simp [Dequant.resultPerTensorAt, Dequant.result, Spec.Dequant.perTensorAtTensor, Spec.Dequant.tensor,
    Spec.Dequant.parameterIndex, hch, hv]

theorem validInputs_perTensor {c : Dequant.Config} {q s z : Tensor} (hch : c.channel = false)
    (hv : Dequant.validInputs c q s z = true) : Dequant.validPerTensorAt 32 c q s z = true := by
  simp only [Dequant.validInputs, Dequant.paramShape, hch, Bool.false_eq_true, ↓reduceIte,
    Bool.and_eq_true, decide_eq_true_eq] at hv
  simp only [Dequant.validPerTensorAt, hch, Bool.not_false, Bool.true_and, Bool.and_eq_true,
    decide_eq_true_eq]
  exact ⟨⟨⟨hv.1.1.1, hv.1.1.2⟩, hv.1.2⟩, hv.2⟩

/-- The body a dequantization site accepts: the granularity's i32 body, or, per-tensor, the body
under an i32 or i64 zero-point annotation. -/
def bodyOK (channel : Bool) (args : List Value) (ops : List Op) : Bool :=
  checkBody channel args ops || (!channel && checkBodyAt args ops)

/-- Every successful source execution is computed by an accepted generic, in any environment. -/
theorem generic_of_run {c : Dequant.Config} {vals result : List RVal}
    {maps : List AffineMap} {iters : List IterKind} {args : List Value} {ops : List Op}
    (ctx : Ctx) (call : CallSem) (env : Env) (hrun : Dequant.run c ctx.M vals = .ok result)
    (hl : checkLayout c.channel c.axis c.shape (Dequant.paramShape c) maps iters = true)
    (hb : bodyOK c.channel args ops = true) :
    ∃ q s z out, vals = [.tensor q, .tensor s, .tensor z] ∧ result = [.tensor out] ∧
      genericSem ctx maps iters [.tensor q, .tensor s, .tensor z]
        [.tensor (Tensor.full c.shape (.float .f32) .poison)]
        (fun ctx => evalRegion opSem ctx call env (.mk [.mk args ops])) = .ok [.tensor out] := by
  unfold Dequant.run at hrun
  split at hrun
  · next q s z =>
    refine ⟨q, s, z, ?_⟩
    simp only [bodyOK, Bool.or_eq_true, Bool.and_eq_true, Bool.not_eq_eq_eq_not, Bool.not_true] at hb
    split at hrun
    · next hv =>
      cases hrun
      refine ⟨_, rfl, rfl, ?_⟩
      rcases hb with hb | ⟨hch, hb⟩
      · exact generic_result_sound ctx call env hv rfl rfl hl hb
      · rw [← resultPerTensorAt_32 hch]
        exact generic_result_sound_at 32 ctx call env (validInputs_perTensor hch hv) rfl rfl hl
          (checkBodyAt_matches hb)
    · split at hrun
      · next hv64 =>
        cases hrun
        refine ⟨_, rfl, rfl, ?_⟩
        have hch : c.channel = false := by
          simp only [Dequant.validPerTensorAt, Bool.and_eq_true, Bool.not_eq_eq_eq_not,
            Bool.not_true] at hv64
          exact hv64.1.1.1.1
        have hm : bodyMatches args ops = true := by
          rcases hb with hb | ⟨_, hb⟩
          · rw [hch] at hb; exact checkBody_matches hb
          · exact checkBodyAt_matches hb
        exact generic_result_sound_at 64 ctx call env hv64 rfl rfl hl hm
      · cases hrun
  · cases hrun

theorem resultPerTensorAt_typed (ib w : Nat) (c : Dequant.Config) (M : FloatModel) (q s z : Tensor) :
    Check.RVal.HasTy ib (.tensor (Dequant.resultPerTensorAt w c M q s z)) (.tensor c.shape (.float .f32)) := by
  refine ⟨rfl, rfl, fun p hp => ?_⟩
  simp only [Dequant.resultPerTensorAt, Spec.Dequant.perTensorAtTensor, Tensor.canon, hp, ↓reduceIte]
  unfold Spec.Dequant.perTensorValueAt
  split <;> simp [Check.Val.HasTy, Scalar.conforms]

/-- Only the two dequantizations have a source contract. -/
theorem config_name {op : Op} {c : Dequant.Config} (h : Dequant.config? op = some c) :
    op.name = "quant_ext.dequantize_per_tensor" ∨ op.name = "quant_ext.dequantize_per_channel" := by
  by_cases h1 : op.name = "quant_ext.dequantize_per_tensor"
  · exact Or.inl h1
  by_cases h2 : op.name = "quant_ext.dequantize_per_channel"
  · exact Or.inr h2
  exfalso
  have e1 : (op.name == "quant_ext.dequantize_per_tensor") = false := by simpa using h1
  have e2 : (op.name == "quant_ext.dequantize_per_channel") = false := by simpa using h2
  unfold Dequant.config? at h
  simp only [e1, e2, Bool.false_eq_true, ↓reduceIte] at h
  repeat' split at h
  all_goals cases h

#print axioms body_sound_at
#print axioms generic_result_sound_at
#print axioms generic_of_run

end MerlinVerify.Rules.DequantPerTensor
