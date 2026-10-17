/- Internal canonical form of uniform generic inputs. The scalar is the actual splat payload;
no new value or numeric law is introduced, and this form is never emitted by the compiler. -/
import MerlinVerify.Rules.SplatInputs

namespace MerlinVerify.Rules.SplatScalar

open MerlinVerify MerlinVerify.Check MerlinVerify.Sem.Ops.Linalg
open SplatInputs

theorem read_equal {ib : Nat} {ranges pt : List Nat} (slot : Slot) {v : RVal}
    (hv : RVal.HasTy ib v slot.payloadTy)
    (hm : mapInBounds ranges (slot.map false) (slot.ty false).shape = true)
    (hp : Idx.InBounds ranges pt) :
    readAt (slot.value false v) (slot.map false) pt = readAt v (slot.map true) pt := by
  rw [SplatInputs.read_equal slot hv hm hp]
  cases slot with
  | keep => rfl
  | splat shape elt map =>
    cases v with
    | tensor _ => cases hv
    | scalar x => simp [Slot.value, Slot.map, readAt, Tensor.full, Tensor.canon,
        evalMap, toIdx, Idx.InBounds]

theorem reads_equal {ib : Nat} {ranges pt : List Nat} (hp : Idx.InBounds ranges pt) :
    ∀ {slots : List Slot} {payloads : List RVal},
      HasTys ib payloads (slots.map Slot.payloadTy) →
      MapsInBounds ranges (slots.map (Slot.map false)) (slots.map (Slot.ty false)) →
      StructuredInputs.reads (inputs false slots payloads) (slots.map (Slot.map false)) pt =
        StructuredInputs.reads payloads (slots.map (Slot.map true)) pt
  | [], [], _, _ => rfl
  | slot :: slots, v :: vs, .cons hv ht, .cons hm hms => by
    simp only [inputs, List.zip_cons_cons, List.map_cons, StructuredInputs.reads,
      List.mapM_cons, Outcome.bindOp_eq]
    rw [read_equal slot hv hm hp]
    have ih := reads_equal hp ht hms
    unfold StructuredInputs.reads inputs at ih
    rw [ih]

def checkLayout (slots : List Slot) (outMaps : List AffineMap) (iters : List IterKind)
    (outTys : List VTy) : Option (List Nat) := do
  let ranges ← structuredRanges? (slots.map (Slot.map false) ++ outMaps) iters.length
    (slots.map (Slot.ty false) ++ outTys)
  if structuredRanges? (slots.map (Slot.map true) ++ outMaps) iters.length
      (slots.map Slot.payloadTy ++ outTys) = some ranges then some ranges else none

theorem checkLayout_sound {slots : List Slot} {outMaps : List AffineMap} {iters : List IterKind}
    {outTys : List VTy} {ranges : List Nat}
    (h : checkLayout slots outMaps iters outTys = some ranges)
    {ib : Nat} {payloads : List RVal} {outs : List Tensor}
    (hi : HasTys ib payloads (slots.map Slot.payloadTy))
    (ho : HasTys ib (outs.map RVal.tensor) outTys) (ctx : Ctx) (body : Ctx → RegionSem) :
    genericSem ctx (slots.map (Slot.map false) ++ outMaps) iters
        (inputs false slots payloads) (outs.map RVal.tensor) body =
      genericSem ctx (slots.map (Slot.map true) ++ outMaps) iters
        payloads (outs.map RVal.tensor) body := by
  unfold checkLayout at h
  cases hs : structuredRanges? (slots.map (Slot.map false) ++ outMaps) iters.length
      (slots.map (Slot.ty false) ++ outTys) with
  | none => simp [hs] at h
  | some sourceRanges =>
    simp only [hs, bind, Option.bind] at h
    split at h
    · next ht =>
      cases h
      have hs := structuredRanges?_spec hs
      have ht := structuredRanges?_spec ht
      apply StructuredInputs.genericSem_congr hs ht (inputs_typed false hi) hi ho
      · have hl : payloads.length = slots.length := by simpa using HasTys.length hi
        simp [inputs_length hi, hl]
      · intro pt hp
        have hb := List.forall₂_take slots.length hs.inBounds
        have hb' : MapsInBounds ranges (slots.map (Slot.map false)) (slots.map (Slot.ty false)) := by
          simpa [List.take_append, List.length_map] using hb
        have hl : payloads.length = slots.length := by simpa using HasTys.length hi
        simpa [inputs_length hi, hl] using reads_equal hp hi hb'
    · cases h

end MerlinVerify.Rules.SplatScalar
