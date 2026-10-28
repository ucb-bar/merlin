/- Splat input compaction: full uniform tensors become rank-zero uniform tensors with empty
indexing maps. Both geometries are checked independently; the body and outputs are retained.
This is a semantic kernel, not yet a checker for arbitrary before/after SSA programs. -/
import MerlinVerify.Rules.StructuredInputs

namespace MerlinVerify.Rules.SplatInputs

open MerlinVerify MerlinVerify.Check MerlinVerify.Sem.Ops.Linalg

/-- Each input either remains unchanged or is constructed from one actual scalar payload.
Uniformity is supplied by tensor.splat construction, never inferred from a tensor's type. -/
inductive Slot where
  | keep (ty : VTy) (map : AffineMap)
  | splat (shape : List Nat) (elt : Ty) (map : AffineMap)

def Slot.payloadTy : Slot → VTy
  | .keep ty _ => ty
  | .splat _ elt _ => .scalar elt

def Slot.ty (compact : Bool) : Slot → VTy
  | .keep ty _ => ty
  | .splat shape elt _ => .tensor (if compact then [] else shape) elt

def Slot.map (compact : Bool) : Slot → AffineMap
  | .keep _ map => map
  | .splat _ _ map => if compact then { map with results := [] } else map

def Slot.value (compact : Bool) : Slot → RVal → RVal
  | .keep _ _, v => v
  | .splat shape elt _, .scalar v => .tensor (Tensor.full (if compact then [] else shape) elt v)
  | .splat .., v => v

def inputs (compact : Bool) (slots : List Slot) (payloads : List RVal) : List RVal :=
  (slots.zip payloads).map fun (slot, v) => slot.value compact v

theorem value_typed (compact : Bool) {ib : Nat} (slot : Slot) {v : RVal}
    (h : RVal.HasTy ib v slot.payloadTy) : RVal.HasTy ib (slot.value compact v) (slot.ty compact) := by
  cases slot with
  | keep => exact h
  | splat shape elt map =>
    cases v with
    | tensor _ => cases h
    | scalar x => exact hasTy_full h

theorem inputs_typed (compact : Bool) {ib : Nat} :
    ∀ {slots : List Slot} {payloads : List RVal}, HasTys ib payloads (slots.map Slot.payloadTy) →
      HasTys ib (inputs compact slots payloads) (slots.map (Slot.ty compact))
  | [], [], _ => .nil
  | _ :: _, _ :: _, .cons hv ht => .cons (value_typed compact _ hv) (inputs_typed compact ht)

theorem inputs_length {ib : Nat} {slots : List Slot} {payloads : List RVal}
    (h : HasTys ib payloads (slots.map Slot.payloadTy)) (compact : Bool) :
    (inputs compact slots payloads).length = slots.length := by
  have hlen : payloads.length = slots.length := by simpa using HasTys.length h
  simp [inputs, hlen]

/-- A checked original access to a uniform tensor reads its payload, including poison. The
compacted rank-zero tensor reads that same payload at the empty coordinate. -/
theorem read_equal {ib : Nat} {ranges pt : List Nat} (slot : Slot) {v : RVal}
    (hv : RVal.HasTy ib v slot.payloadTy)
    (hm : mapInBounds ranges (slot.map false) (slot.ty false).shape = true)
    (hp : Idx.InBounds ranges pt) :
    readAt (slot.value false v) (slot.map false) pt =
      readAt (slot.value true v) (slot.map true) pt := by
  cases slot with
  | keep => rfl
  | splat shape elt map =>
    cases v with
    | tensor _ => cases hv
    | scalar x =>
      change mapInBounds ranges map shape = true at hm
      obtain ⟨i, hi, hb⟩ := mapInBounds_sound hm hp
      have hread : readAt (.tensor (Tensor.full shape elt x)) map pt = .ok x := by
        simp only [readAt, Tensor.full, Tensor.canon, hi, hb, ↓reduceIte]
      simp only [Slot.map, Slot.value, Bool.false_eq_true, ↓reduceIte]
      rw [hread]
      simp [readAt, Tensor.full, Tensor.canon, evalMap, toIdx, Idx.InBounds]

theorem reads_equal {ib : Nat} {ranges pt : List Nat} (hp : Idx.InBounds ranges pt) :
    ∀ {slots : List Slot} {payloads : List RVal},
      HasTys ib payloads (slots.map Slot.payloadTy) →
      MapsInBounds ranges (slots.map (Slot.map false)) (slots.map (Slot.ty false)) →
      StructuredInputs.reads (inputs false slots payloads) (slots.map (Slot.map false)) pt =
        StructuredInputs.reads (inputs true slots payloads) (slots.map (Slot.map true)) pt
  | [], [], _, _ => rfl
  | slot :: slots, v :: vs, .cons hv ht, .cons hm hms => by
    simp only [inputs, List.zip_cons_cons, List.map_cons, StructuredInputs.reads,
      List.mapM_cons, Outcome.bindOp_eq]
    rw [read_equal slot hv hm hp]
    have ih := reads_equal hp ht hms
    unfold StructuredInputs.reads inputs at ih
    rw [ih]

/-- Accept only independently checked source and compacted layouts with equal loop ranges.
Removing an extent-determining operand cannot silently change the iteration space. -/
def checkLayout (slots : List Slot) (outMaps : List AffineMap) (iters : List IterKind)
    (outTys : List VTy) : Option (List Nat) := do
  let ranges ← structuredRanges? (slots.map (Slot.map false) ++ outMaps) iters.length
    (slots.map (Slot.ty false) ++ outTys)
  if structuredRanges? (slots.map (Slot.map true) ++ outMaps) iters.length
      (slots.map (Slot.ty true) ++ outTys) = some ranges then some ranges else none

theorem checkLayout_sound {slots : List Slot} {outMaps : List AffineMap} {iters : List IterKind}
    {outTys : List VTy} {ranges : List Nat}
    (h : checkLayout slots outMaps iters outTys = some ranges)
    {ib : Nat} {payloads : List RVal} {outs : List Tensor}
    (hi : HasTys ib payloads (slots.map Slot.payloadTy))
    (ho : HasTys ib (outs.map RVal.tensor) outTys) (ctx : Ctx) (body : Ctx → RegionSem) :
    genericSem ctx (slots.map (Slot.map false) ++ outMaps) iters
        (inputs false slots payloads) (outs.map RVal.tensor) body =
      genericSem ctx (slots.map (Slot.map true) ++ outMaps) iters
        (inputs true slots payloads) (outs.map RVal.tensor) body := by
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
      apply StructuredInputs.genericSem_congr hs ht (inputs_typed false hi) (inputs_typed true hi) ho
      · simp [inputs_length hi]
      · intro pt hp
        have hb := List.forall₂_take slots.length hs.inBounds
        have hb' : MapsInBounds ranges (slots.map (Slot.map false)) (slots.map (Slot.ty false)) := by
          simpa [List.take_append, List.length_map] using hb
        simpa [inputs_length hi] using reads_equal hp hi hb'
    · cases h

end MerlinVerify.Rules.SplatInputs
