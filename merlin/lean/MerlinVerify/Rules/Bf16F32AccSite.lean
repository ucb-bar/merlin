/-
Bridge from the reference transformer's emitted syntax to the contract. The accumulating
`linalg.generic` that `Bf16F32AccReference.emit` builds, with any admitted semantics-free
annotations, dispatches through the frozen `Sem.Ops.opSem` to exactly the contract's
accumulator tensor (`Bf16F32AccTensor.accumulate_sound`). Nothing here is assumed of the output
being checked; it states what the reference that output must refine computes.
-/
import MerlinVerify.Rules.Bf16F32AccReference

namespace MerlinVerify.Rules.Bf16F32AccSite

open MerlinVerify MerlinVerify.Sem.Ops.Linalg Bf16F32AccContract Bf16F32AccBody Bf16F32AccTensor
  Bf16F32AccReference
open MerlinVerify.Check (alphaRegion)

/-- Removing annotations never changes the value of a key the semantics may read. -/
theorem lookup_semantic (attrs : List (String × Attr)) {key : String}
    (hk : Spec.Annotations.semanticsFree key = false) :
    (semantic attrs).lookup key = attrs.lookup key := by
  induction attrs with
  | nil => rfl
  | cons entry rest ih =>
    obtain ⟨name, value⟩ := entry
    by_cases hf : Spec.Annotations.semanticsFree name = true
    · have hne : key ≠ name := by
        intro he
        rw [he, hf] at hk
        cases hk
      have hb : (key == name) = false := by simpa using hne
      simp [semantic, free, hf, List.lookup, hb, ← ih]
    · have hf' : Spec.Annotations.semanticsFree name = false := by simpa using hf
      by_cases he : key = name
      · subst he
        simp [semantic, free, hf', List.lookup]
      · have hb : (key == name) = false := by simpa using he
        simp [semantic, free, hf', List.lookup, hb, ← ih]

theorem decode_maps (maps : List AffineMap) :
    (maps.map Attr.affineMap).mapM (fun | .affineMap m => some m | _ => none) = some maps := by
  induction maps with
  | nil => rfl
  | cons m rest ih => simp [ih]

theorem decode_iters (iters : List IterKind) :
    (iters.map iterAttr).mapM (fun
      | .enum "linalg" "iterator_type" "parallel" => some IterKind.parallel
      | .enum "linalg" "iterator_type" "reduction" => some .reduction
      | _ => none) = some iters := by
  induction iters with
  | nil => rfl
  | cons it rest ih => cases it <;> simp [iterAttr, ih]

theorem affineMaps_of {op : Op} {maps : List AffineMap}
    (h : op.attr? "indexing_maps" = some (.array (maps.map .affineMap))) :
    attrAffineMaps? op "indexing_maps" = some maps := by
  unfold attrAffineMaps?
  rw [h]
  exact decode_maps maps

theorem iterKinds_of {op : Op} {iters : List IterKind}
    (h : op.attr? "iterator_types" = some (.array (iters.map iterAttr))) : iterKinds? op = some iters := by
  unfold iterKinds?
  rw [h]
  exact decode_iters iters

private theorem key_order :
    ("iterator_types" == "indexing_maps") = false ∧ ("operandSegmentSizes" == "indexing_maps") = false ∧
    ("operandSegmentSizes" == "iterator_types") = false := by
  refine ⟨?_, ?_, ?_⟩ <;> decide

private theorem dialect_linalg : Sem.Ops.dialectOf "linalg.generic" = "linalg" := by cbv

private theorem free_keys :
    Spec.Annotations.semanticsFree "indexing_maps" = false ∧
    Spec.Annotations.semanticsFree "iterator_types" = false ∧
    Spec.Annotations.semanticsFree "operandSegmentSizes" = false := by
  refine ⟨?_, ?_, ?_⟩ <;> cbv

/-- Every attribute of an admitted operation is read by the generic semantics or annotation. -/
theorem modelled (ins : Nat) (maps : List AffineMap) (iters : List IterKind)
    {attrs : List (String × Attr)} (hattrs : semantic attrs = generic ins maps iters) (result : Value) :
    Sem.Ops.unmodelledAttr? (Op.mk "linalg.generic" [result] [] attrs [] [] []) = none := by
  unfold Sem.Ops.unmodelledAttr?
  rw [List.findSome?_eq_none_iff]
  intro entry member
  obtain ⟨name, value⟩ := entry
  simp only [Op.props, Op.attrs, Op.name, List.nil_append] at member ⊢
  by_cases hf : Spec.Annotations.semanticsFree name = true
  · simp [hf]
  · have hs : (name, value) ∈ semantic attrs := by
      simp only [semantic, free, List.mem_filter]
      exact ⟨member, by simpa using hf⟩
    rw [hattrs] at hs
    simp only [generic, List.mem_cons, Prod.mk.injEq, List.not_mem_nil, or_false] at hs
    rcases hs with ⟨rfl, _⟩ | ⟨rfl, _⟩ | ⟨rfl, _⟩ <;> simp [Sem.Ops.interpretedAttrs]

/-- The emitted accumulating generic computes clauses 1–4 of the contract. -/
theorem accumulating_op_sound (ctx : Ctx) (call callR : CallSem) (env : Env)
    (iters : List IterKind) {k : FKind} (license : License) (g : Geometry)
    (geometry : Check.StructuredOk [g.leftMap, g.rightMap, g.outMap] iters.length
      [.tensor g.left (.float k), .tensor g.right (.float k), .tensor g.out (.float .f32)] g.ranges)
    {attrs : List (String × Attr)}
    (hattrs : semantic attrs = generic 2 [g.leftMap, g.rightMap, g.outMap] iters)
    (result : Value) (operands : List Nat) {region : Region}
    (matched : alphaRegion [] (accRegion k k license) region)
    (A : Idx → Option (FBits k)) (B : Idx → Option (FBits k)) :
    Sem.Ops.opSem ctx call (.mk "linalg.generic" [result] operands attrs [] [region] [])
      [.tensor (floats k g.left A), .tensor (floats k g.right B),
        .tensor (Tensor.full g.out (.float .f32) (value .f32 (some zero)))]
      [fun c => evalRegion Sem.Ops.opSem c callR env region] =
      .ok [.tensor (Tensor.canon g.out (.float .f32)
        (fun cell => value .f32 (accumulate ctx.M license g A B cell)))] := by
  obtain ⟨fm, fi, fo⟩ := free_keys
  have read : ∀ key, Spec.Annotations.semanticsFree key = false →
      attrs.lookup key = (generic 2 [g.leftMap, g.rightMap, g.outMap] iters).lookup key := by
    intro key hk
    rw [← lookup_semantic attrs hk, hattrs]
  have hm := read "indexing_maps" fm
  have hi := read "iterator_types" fi
  have ho := read "operandSegmentSizes" fo
  obtain ⟨k1, k2, k3⟩ := key_order
  simp only [generic, List.lookup, beq_self_eq_true, k1, k2, k3] at hm hi ho
  have hloop := accumulate_sound (ib := 0) ctx callR env iters license g geometry matched A B
  generalize (fun c => evalRegion Sem.Ops.opSem c callR env region) = body at hloop ⊢
  let blind := Op.mk "linalg.generic" [{ result with id := 0 }] [] attrs [] [] []
  have attr : ∀ key, blind.attr? key = attrs.lookup key := fun _ => rfl
  have hMaps : attrAffineMaps? blind "indexing_maps" = some [g.leftMap, g.rightMap, g.outMap] :=
    affineMaps_of (by rw [attr, hm])
  have hIters : iterKinds? blind = some iters := iterKinds_of (by rw [attr, hi])
  have hSeg : attrDenseArray? blind "operandSegmentSizes" = some [2, 1] := by
    unfold attrDenseArray?
    rw [attr, ho]
    rfl
  have hBlind : (Op.mk "linalg.generic" [result] operands attrs [] [region] []).blind = blind := rfl
  unfold Sem.Ops.opSem
  rw [hBlind, modelled 2 _ iters hattrs]
  simp only [Sem.Ops.dispatch, blind, dialect_linalg, Op.name, Sem.Ops.Linalg.sem, splitInsOuts]
  simp only [blind] at hSeg hMaps hIters
  rw [hSeg, hMaps, hIters]
  simp [hloop, Op.results]

/-- The emitted final-conversion generic computes clause 5 on every cell, over any destination. -/
theorem final_op_sound (ctx : Ctx) (call callR : CallSem) (env : Env) (kc : FKind) (shape : List Nat)
    (accumulated : Idx → Option (FBits .f32))
    (geometry : Check.StructuredOk [identityMap shape.length, identityMap shape.length]
      (List.replicate shape.length IterKind.parallel).length
      [.tensor shape (.float .f32), .tensor shape (.float kc)] shape)
    {attrs : List (String × Attr)}
    (hattrs : semantic attrs = generic 1 [identityMap shape.length, identityMap shape.length]
      (List.replicate shape.length .parallel))
    (result : Value) (operands : List Nat) {region : Region}
    (matched : alphaRegion [] (truncRegion kc) region) (initial : Idx → Val) :
    Sem.Ops.opSem ctx call (.mk "linalg.generic" [result] operands attrs [] [region] [])
      [.tensor (Tensor.canon shape (.float .f32) (fun cell => value .f32 (accumulated cell))),
        .tensor ⟨shape, .float kc, initial⟩]
      [fun c => evalRegion Sem.Ops.opSem c callR env region] =
      .ok [.tensor (Tensor.canon shape (.float kc)
        (fun cell => value kc ((accumulated cell).map (trunc ctx.M kc))))] := by
  obtain ⟨fm, fi, fo⟩ := free_keys
  obtain ⟨k1, k2, k3⟩ := key_order
  have read : ∀ key, Spec.Annotations.semanticsFree key = false →
      attrs.lookup key = (generic 1 [identityMap shape.length, identityMap shape.length]
        (List.replicate shape.length .parallel)).lookup key := by
    intro key hk
    rw [← lookup_semantic attrs hk, hattrs]
  have hm := read "indexing_maps" fm
  have hi := read "iterator_types" fi
  have ho := read "operandSegmentSizes" fo
  simp only [generic, List.lookup, beq_self_eq_true, k1, k2, k3] at hm hi ho
  have hloop := truncate_sound ctx callR env (List.replicate shape.length .parallel) kc shape accumulated
    geometry matched initial
  generalize (fun c => evalRegion Sem.Ops.opSem c callR env region) = body at hloop ⊢
  let blind := Op.mk "linalg.generic" [{ result with id := 0 }] [] attrs [] [] []
  have attr : ∀ key, blind.attr? key = attrs.lookup key := fun _ => rfl
  have hMaps : attrAffineMaps? blind "indexing_maps" =
      some [identityMap shape.length, identityMap shape.length] := affineMaps_of (by rw [attr, hm])
  have hIters : iterKinds? blind = some (List.replicate shape.length .parallel) :=
    iterKinds_of (by rw [attr, hi])
  have hSeg : attrDenseArray? blind "operandSegmentSizes" = some [1, 1] := by
    unfold attrDenseArray?
    rw [attr, ho]
    rfl
  have hBlind : (Op.mk "linalg.generic" [result] operands attrs [] [region] []).blind = blind := rfl
  unfold Sem.Ops.opSem
  rw [hBlind, modelled 1 _ _ hattrs]
  simp only [Sem.Ops.dispatch, blind, dialect_linalg, Op.name, Sem.Ops.Linalg.sem, splitInsOuts]
  simp only [blind] at hSeg hMaps hIters
  rw [hSeg, hMaps, hIters]
  simp [hloop, Op.results]

#print axioms lookup_semantic
#print axioms modelled
#print axioms accumulating_op_sound
#print axioms final_op_sound

end MerlinVerify.Rules.Bf16F32AccSite
