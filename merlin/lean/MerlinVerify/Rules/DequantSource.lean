/- Connect actual runtime tensors in the source semantics to the proved generic loop. -/
import MerlinVerify.Rules.DequantTensor

namespace MerlinVerify.Rules.DequantSource

open MerlinVerify MerlinVerify.Sem.Ops MerlinVerify.Sem.Ops.Linalg
open MerlinVerify.Rules.DequantScalar MerlinVerify.Rules.DequantTensor

/-- Decoding/re-encoding cannot change a well-typed integer element, including poison. -/
theorem bits_roundtrip {w : Nat} {v : Val} (h : v.conforms 0 (.int w) = true) :
    intValue w (Dequant.bits w v) = v := by
  cases v with
  | poison => rfl
  | some v =>
    cases v with
    | flt k x => simp [Val.conforms, Scalar.conforms] at h
    | bits w' x =>
      simp only [Val.conforms, Scalar.conforms, beq_iff_eq] at h
      subst h
      simp [Dequant.bits, intValue]

theorem float_roundtrip {v : Val} (h : v.conforms 0 (.float .f32) = true) :
    floatValue (Dequant.float32 v) = v := by
  cases v with
  | poison => rfl
  | some v =>
    cases v with
    | bits w x => simp [Val.conforms, Scalar.conforms] at h
    | flt k x =>
      cases k <;> simp_all [Val.conforms, Scalar.conforms, Dequant.float32, floatValue]

theorem tensorTyped_get {t : Tensor} {ty : Ty} (h : Dequant.tensorTyped t ty = true)
    {p : Idx} (hp : Idx.InBounds t.shape p) : (t.get p).conforms 0 ty = true := by
  simp only [Dequant.tensorTyped, Bool.and_eq_true] at h
  exact List.all_eq_true.1 h.2 p ((mem_allPoints _ _).2 hp)

theorem parameterIndex_bounds {channel : Bool} {axis : Nat} {shape : List Nat}
    (ha : channel = false ∨ axis < shape.length) {p : Idx} (hp : Idx.InBounds shape p) :
    Idx.InBounds (if channel then (shape[axis]?).toList else [])
      (Spec.Dequant.parameterIndex channel axis p) := by
  cases channel with
  | false => exact (Idx.inBounds_nil_iff []).2 rfl
  | true =>
    have haxis : axis < shape.length := ha.resolve_left (by decide)
    have hpaxis : axis < p.length := by rw [hp.1]; exact haxis
    simpa [Spec.Dequant.parameterIndex, List.getElem?_eq_getElem haxis,
      List.getElem?_eq_getElem hpaxis] using
      (Idx.inBounds_cons_cons _ _ _ _).2
        ⟨hp.2 axis hpaxis haxis, (Idx.inBounds_nil_iff []).2 rfl⟩

/-- No abstract tensor representation hypothesis remains: the operands are the source's actual
runtime tensors, whose observable elements and shapes have been checked independently. -/
theorem generic_result_sound {c : Dequant.Config} {q s z t : Tensor}
    {maps : List AffineMap} {iters : List IterKind} {args : List Value} {ops : List Op}
    (ctx : Ctx) (call : CallSem) (env : Env)
    (hv : Dequant.validInputs c q s z = true) (ht : t.shape = c.shape) (he : t.elt = .float .f32)
    (hl : checkLayout c.channel c.axis c.shape (Dequant.paramShape c) maps iters = true)
    (hb : checkBody c.channel args ops = true) :
    genericSem ctx maps iters [.tensor q, .tensor s, .tensor z] [.tensor t]
      (fun ctx => evalRegion opSem ctx call env (.mk [.mk args ops])) =
      .ok [.tensor (Dequant.result c ctx.M q s z)] := by
  simp only [Dequant.validInputs, Bool.and_eq_true, decide_eq_true_eq] at hv
  obtain ⟨⟨⟨⟨hqshape, hsshape, hzshape⟩, hq⟩, hs⟩, hz⟩ := hv
  have layout := checkLayout_sound hl
  have hindex (p : Idx) (hp : Idx.InBounds c.shape p) :=
    parameterIndex_sound layout.1 layout.2.1 hp
  have hbounds (p : Idx) (hp : Idx.InBounds c.shape p) :
      Idx.InBounds (Dequant.paramShape c) (Spec.Dequant.parameterIndex c.channel c.axis p) :=
    parameterIndex_bounds layout.1 hp
  have hqread (p : Idx) (hp : Idx.InBounds t.shape p) :
      readAt (.tensor q) (identityMap t.shape.length) p = .ok (intValue 8 (Dequant.bits 8 (q.get p))) := by
    have hqp : Idx.InBounds q.shape p := by simpa [hqshape, ht] using hp
    rw [bits_roundtrip (tensorTyped_get hq hqp)]
    simpa [ht, hqshape] using readAt_identity hqp
  have hsread (p : Idx) (hp : Idx.InBounds t.shape p) :
      readAt (.tensor s) (parameterMap c.channel t.shape.length c.axis) p =
        .ok (floatValue (Dequant.float32 (s.get (Spec.Dequant.parameterIndex c.channel c.axis p)))) := by
    have hp' : Idx.InBounds c.shape p := ht ▸ hp
    have hip : Idx.InBounds s.shape (Spec.Dequant.parameterIndex c.channel c.axis p) :=
      hsshape ▸ hbounds p hp'
    rw [float_roundtrip (tensorTyped_get hs hip)]
    simp [readAt, ht, hsshape, hindex p hp']
  have hzread (p : Idx) (hp : Idx.InBounds t.shape p) :
      readAt (.tensor z) (parameterMap c.channel t.shape.length c.axis) p =
        .ok (intValue 32 (Dequant.bits 32 (z.get (Spec.Dequant.parameterIndex c.channel c.axis p)))) := by
    have hp' : Idx.InBounds c.shape p := ht ▸ hp
    have hip : Idx.InBounds z.shape (Spec.Dequant.parameterIndex c.channel c.axis p) :=
      hzshape ▸ hbounds p hp'
    rw [bits_roundtrip (tensorTyped_get hz hip)]
    simp [readAt, ht, hzshape, hindex p hp']
  simpa [Dequant.result, Spec.Dequant.tensor, ht, he] using
    checked_generic_sound ctx call env (.tensor q) (.tensor s) (.tensor z) t
      (by simpa [ht] using hl) hb
      (by simp [operandTypes, Check.VTy.shape, shapeOf, hqshape, hsshape, hzshape, ht])
      (fun p => Dequant.bits 8 (q.get p))
      (fun p => Dequant.bits 32 (z.get (Spec.Dequant.parameterIndex c.channel c.axis p)))
      (fun p => Dequant.float32 (s.get (Spec.Dequant.parameterIndex c.channel c.axis p)))
      hqread hsread hzread

end MerlinVerify.Rules.DequantSource
