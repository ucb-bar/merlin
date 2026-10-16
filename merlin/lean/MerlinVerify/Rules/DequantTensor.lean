/- Tensor lifting of the poison-aware scalar contract. This module does not register a C1 rule. -/
import MerlinVerify.Rules.DequantScalar
import MerlinVerify.Sem.Facts
import MerlinVerify.Check.RulesLinalg

namespace MerlinVerify.Rules.DequantTensor

open MerlinVerify MerlinVerify.Rules.DequantScalar MerlinVerify.Sem.Ops.Linalg

/-- Region execution closes the scalar theorem over any enclosing SSA environment. -/
theorem region_sound {channel : Bool} {args : List Value} {ops : List Op}
    (h : checkBody channel args ops = true) (ctx : Ctx) (call : CallSem) (env : Env)
    (q : Option (BitVec 8)) (z : Option (BitVec 32)) (s : Option (FBits .f32)) (init : Val) :
    evalRegion Sem.Ops.opSem ctx call env (.mk [.mk args ops])
      [.scalar (intValue 8 q), .scalar (floatValue s), .scalar (intValue 32 z), .scalar init] =
      .ok [.scalar (Spec.Dequant.scalarValue channel ctx.M q z s)] := by
  have hh := h
  simp only [checkBody, Bool.and_eq_true] at hh
  have hty := of_decide_eq_true hh.1.1
  have hl : args.length = 4 := by simpa using congrArg List.length hty
  have hf := (Check.freshIds_sound hh.1.2).1
  have hc := Check.Corr.extend
    (ρ := []) (envS := []) (envT := env) (fun _ _ he => by simp [Check.Rho.find] at he)
    [0, 1, 2, 3] (args.map (·.id))
    [.scalar (intValue 8 q), .scalar (floatValue s), .scalar (intValue 32 z), .scalar init]
    (by rfl) (by simpa using hl) hf (fun _ _ _ he => by simp at he)
  have hc' : Check.Corr (argMap args) (poisonInputs q z s init)
      ((args.map (·.id)).zip
        [.scalar (intValue 8 q), .scalar (floatValue s), .scalar (intValue 32 z), .scalar init] ++ env) := by
    simpa [argMap, poisonInputs] using hc
  simpa [evalRegion, evalBlocks, evalBlock, Env.bindAll, hl] using
    checkBody_poison_sound h ctx call q z s init _ hc'

/-- The parameter map comes from source granularity and axis, never from the candidate body. -/
def parameterMap (channel : Bool) (rank axis : Nat) : AffineMap :=
  { numDims := rank, numSyms := 0, results := if channel then [.dim axis] else [] }

def mapsFor (channel : Bool) (rank axis : Nat) : List AffineMap :=
  [identityMap rank, parameterMap channel rank axis,
    parameterMap channel rank axis, identityMap rank]

def operandTypes (shape paramShape : List Nat) : List Check.VTy :=
  [.tensor shape (.int 8), .tensor paramShape (.float .f32),
    .tensor paramShape (.int 32), .tensor shape (.float .f32)]

/-- Static layout check, separate from the scalar-body check. All axes, shapes, maps, loop
kinds and access bounds are recomputed. A per-tensor axis is unused. -/
def checkLayout (channel : Bool) (axis : Nat) (shape paramShape : List Nat)
    (maps : List AffineMap) (iters : List IterKind) : Bool :=
  decide (channel = false ∨ axis < shape.length) &&
  decide (paramShape = if channel then (shape[axis]?).toList else []) &&
  decide (maps = mapsFor channel shape.length axis) &&
  decide (iters = List.replicate shape.length .parallel) &&
  (Check.structuredRanges? maps iters.length (operandTypes shape paramShape) == some shape)

/-- Every accepted layout has the source's indexing and independently proved total access bounds. -/
theorem checkLayout_sound {channel : Bool} {axis : Nat} {shape paramShape : List Nat}
    {maps : List AffineMap} {iters : List IterKind}
    (h : checkLayout channel axis shape paramShape maps iters = true) :
    (channel = false ∨ axis < shape.length) ∧
    paramShape = (if channel then (shape[axis]?).toList else []) ∧
    maps = mapsFor channel shape.length axis ∧
    iters = List.replicate shape.length .parallel ∧
    Check.StructuredOk maps iters.length (operandTypes shape paramShape) shape := by
  simp only [checkLayout, Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at h
  exact ⟨h.1.1.1.1, h.1.1.1.2, h.1.1.2, h.1.2, Check.structuredRanges?_spec h.2⟩

/-- Lift to the entire loop nest, including zero extents and arbitrary initial output elements.
The hypotheses expose every layout/access obligation still required of a future site checker;
none is inferred from the success of the scalar-body matcher. -/
theorem generic_sound {channel : Bool} {args : List Value} {ops : List Op}
    (h : checkBody channel args ops = true) (ctx : Ctx) (call : CallSem) (env : Env)
    (qv sv zv : RVal) (t : Tensor) (m : AffineMap)
    (q : Idx → Option (BitVec 8)) (z : Idx → Option (BitVec 32))
    (s : Idx → Option (FBits .f32))
    (hdims : [identityMap t.shape.length, m, m, identityMap t.shape.length].any
      (fun m => m.numDims ≠ t.shape.length || m.numSyms ≠ 0) = false)
    (hrank : ([identityMap t.shape.length, m, m, identityMap t.shape.length].zip
      ([qv, sv, zv, .tensor t].map shapeOf)).any
      (fun p => p.1.results.length ≠ p.2.length) = false)
    (hranges : loopRanges [identityMap t.shape.length, m, m, identityMap t.shape.length]
      ([qv, sv, zv, .tensor t].map shapeOf) t.shape.length = .ok t.shape)
    (hq : ∀ p, Idx.InBounds t.shape p →
      readAt qv (identityMap t.shape.length) p = .ok (intValue 8 (q p)))
    (hs : ∀ p, Idx.InBounds t.shape p → readAt sv m p = .ok (floatValue (s p)))
    (hz : ∀ p, Idx.InBounds t.shape p → readAt zv m p = .ok (intValue 32 (z p))) :
    genericSem ctx [identityMap t.shape.length, m, m, identityMap t.shape.length]
      (List.replicate t.shape.length .parallel) [qv, sv, zv] [.tensor t]
      (fun ctx => evalRegion Sem.Ops.opSem ctx call env (.mk [.mk args ops])) =
      .ok [.tensor (Tensor.canon t.shape t.elt
        (fun p => Spec.Dequant.scalarValue channel ctx.M (q p) (z p) (s p)))] := by
  apply genericSem_parallel1 ctx _ _ _ t _ t.shape _ (by rfl)
    (by simpa using hdims) hrank (by simpa using hranges) rfl (by rfl)
  intro p hp
  simp [bodyAt, hq p hp, hs p hp, hz p hp, readAt_identity hp,
    region_sound h]

/-- The checked tensor result. Static geometry obligations are discharged by `checkLayout`;
the pointwise hypotheses identify typed values at the source's broadcast/projected indices. -/
theorem checked_generic_sound {channel : Bool} {axis : Nat} {paramShape : List Nat}
    {maps : List AffineMap} {iters : List IterKind} {args : List Value} {ops : List Op}
    (ctx : Ctx) (call : CallSem) (env : Env) (qv sv zv : RVal) (t : Tensor)
    (hl : checkLayout channel axis t.shape paramShape maps iters = true)
    (hb : checkBody channel args ops = true)
    (ht : Check.HasTys ctx.indexBits [qv, sv, zv, .tensor t] (operandTypes t.shape paramShape))
    (q : Idx → Option (BitVec 8)) (z : Idx → Option (BitVec 32))
    (s : Idx → Option (FBits .f32))
    (hq : ∀ p, Idx.InBounds t.shape p →
      readAt qv (identityMap t.shape.length) p = .ok (intValue 8 (q p)))
    (hs : ∀ p, Idx.InBounds t.shape p →
      readAt sv (parameterMap channel t.shape.length axis) p = .ok (floatValue (s p)))
    (hz : ∀ p, Idx.InBounds t.shape p →
      readAt zv (parameterMap channel t.shape.length axis) p = .ok (intValue 32 (z p))) :
    genericSem ctx maps iters [qv, sv, zv] [.tensor t]
      (fun ctx => evalRegion Sem.Ops.opSem ctx call env (.mk [.mk args ops])) =
      .ok [.tensor (Tensor.canon t.shape t.elt
        (fun p => Spec.Dequant.scalarValue channel ctx.M (q p) (z p) (s p)))] := by
  obtain ⟨-, -, hm, hi, hstruct⟩ := checkLayout_sound hl
  have hshapes := Check.shapes_typed ht
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
  exact generic_sound hb ctx call env qv sv zv t _ q z s
    (by simpa [mapsFor] using hd) (by simpa [mapsFor] using hr)
    (by simpa [mapsFor] using hranges) hq hs hz

end MerlinVerify.Rules.DequantTensor
