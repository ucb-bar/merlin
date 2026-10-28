/- Refined facts through structured reads, writes and canonicalization. Elementwise is a
proved property of a refinement model, not a certificate assertion. -/
import MerlinVerify.Check.DefinedContext
import MerlinVerify.Check.DefinedLoops

namespace MerlinVerify.Check.Definedness.Refined

open MerlinVerify MerlinVerify.Sem.Ops.Linalg

def Elementwise {Info : Type} (M : Model Info) : Prop :=
  ∀ info tensor, M.holds info (.tensor tensor) ↔
    ∀ point, Idx.InBounds tensor.shape point → M.holds info (.scalar (tensor.get point))

def elementFact {Info : Type} (fact : Fact Info) : Fact Info :=
  ⟨Definedness.elementFact fact.base, fact.info⟩

variable {Info : Type} {M : Model Info}

theorem splitInsOuts_refined {ib : Nat} {op : Op} {args : List RVal}
    {facts ins outs : List (Fact Info)} (split : splitInsOuts op facts = some (ins, outs))
    (holds : FactsHold M ib args facts) : ∃ xs ys, splitInsOuts op args = some (xs, ys) ∧
      FactsHold M ib xs ins ∧ FactsHold M ib ys outs := by
  have count := List.Forall₂.length_eq holds
  unfold splitInsOuts at split ⊢
  split at split
  · next ni no _ =>
    split at split
    · next sizes =>
      simp only [Option.some.injEq, Prod.mk.injEq] at split
      obtain ⟨rfl, rfl⟩ := split
      refine ⟨args.take ni.toNat, args.drop ni.toNat, ?_, List.forall₂_take _ holds,
        List.forall₂_drop _ holds⟩
      rw [ite_eq_left ⟨sizes.1, sizes.2.1, by rw [count]; exact sizes.2.2⟩]
    · cases split
  · cases split

theorem readAt_refined (elementwise : Elementwise M) {ib : Nat} {ranges point : List Nat}
    (bounds : Idx.InBounds ranges point) {value : RVal} {fact : Fact Info} {map : AffineMap}
    (holds : fact.Holds M ib value) (mapped : mapInBounds ranges map fact.base.ty.shape = true) :
    ∃ result, readAt value map point = .ok result ∧ (elementFact fact).Holds M ib (.scalar result) := by
  obtain ⟨result, executed, base⟩ := Definedness.readAt_total bounds holds.1 mapped
  refine ⟨result, executed, base, ?_⟩
  cases value with
  | scalar value =>
    cases ty : fact.base.ty with
    | tensor _ _ =>
      have typed := holds.1.1
      simp only [ty, RVal.HasTy] at typed
    | scalar elt =>
      simp only [ty, mapInBounds, VTy.shape, List.length_nil, beq_iff_eq, List.zip_nil_right,
        List.all_nil, Bool.and_true, List.length_eq_zero_iff] at mapped
      simp only [readAt, mapped, List.isEmpty_nil, ↓reduceIte, Outcome.ok.injEq] at executed
      exact executed ▸ holds.2
  | tensor tensor =>
    cases ty : fact.base.ty with
    | scalar _ =>
      have typed := holds.1.1
      simp only [ty, RVal.HasTy] at typed
    | tensor shape elt =>
      have typed := holds.1.1
      simp only [ty, RVal.HasTy] at typed
      have shapeEq : tensor.shape = shape := typed.1
      obtain ⟨index, decoded, inBounds⟩ := mapInBounds_sound mapped bounds
      simp only [ty, VTy.shape] at decoded inBounds
      simp only [readAt, shapeEq, decoded, Outcome.ok.injEq] at executed
      exact executed ▸ (elementwise _ _).1 holds.2 index (shapeEq ▸ inBounds)

theorem mapM_read_refined (elementwise : Elementwise M) {ib : Nat} {ranges point : List Nat}
    (bounds : Idx.InBounds ranges point) {α : Type} (toValue : α → RVal)
    (read : α × AffineMap → Outcome Val)
    (same : ∀ value map, read (value, map) = readAt (toValue value) map point) :
    ∀ (values : List α) (maps : List AffineMap) (facts : List (Fact Info)),
      FactsHold M ib (values.map toValue) facts →
      MapsInBounds ranges maps (facts.map (·.base.ty)) →
      ∃ results, (values.zip maps).mapM read = .ok results ∧
        FactsHold M ib (results.map RVal.scalar) (facts.map elementFact)
  | [], [], [], _, _ => ⟨[], rfl, .nil⟩
  | value :: values, map :: maps, fact :: facts, .cons head tail, .cons mapped rest => by
    obtain ⟨result, executed, holds⟩ := readAt_refined elementwise bounds head mapped
    obtain ⟨results, remaining, valid⟩ := mapM_read_refined elementwise bounds toValue read same
      values maps facts tail rest
    exact ⟨result :: results, by simp only [List.zip_cons_cons, List.mapM_cons, Outcome.bindOp_eq,
      same, executed, remaining, Outcome.bind_ok, Outcome.pure_eq], .cons holds valid⟩
  | [], _ :: _, _, holds, maps => by cases holds; cases maps
  | _ :: _, [], facts, holds, maps => by cases facts; cases holds; cases maps

theorem writeAt_refined (elementwise : Elementwise M) {ib : Nat} {ranges point : List Nat}
    (bounds : Idx.InBounds ranges point) {tensor : Tensor} {fact : Fact Info} {map : AffineMap}
    {value : Val} (initialized : fact.base.initialized = true) (holds : fact.Holds M ib (.tensor tensor))
    (mapped : mapInBounds ranges map fact.base.ty.shape = true)
    (result : (elementFact fact).Holds M ib (.scalar value)) :
    ∃ output, writeAt tensor map point value = .ok output ∧ fact.Holds M ib (.tensor output) := by
  have initial : Definedness.Fact.Holds ib (.tensor tensor) ⟨fact.base.ty, true⟩ :=
    ⟨holds.1.1, fun _ => holds.1.2 initialized⟩
  obtain ⟨output, executed, base⟩ := Definedness.writeAt_total bounds initial mapped result.1.1
    (result.1.2 initialized)
  refine ⟨output, executed, ⟨base.1, fun _ => base.2 rfl⟩, ?_⟩
  cases ty : fact.base.ty with
  | scalar _ =>
    have typed := initial.1
    simp only [ty, RVal.HasTy] at typed
  | tensor shape elt =>
    have typed := holds.1.1
    simp only [ty, RVal.HasTy] at typed
    have shapeEq : tensor.shape = shape := typed.1
    obtain ⟨index, decoded, _inBounds⟩ := mapInBounds_sound mapped bounds
    simp only [ty, VTy.shape] at decoded
    simp only [writeAt, shapeEq, decoded, Outcome.ok.injEq] at executed
    subst output
    apply (elementwise _ _).2
    intro cell inBounds
    change Idx.InBounds tensor.shape cell at inBounds
    simp only [Tensor.set]
    split
    · exact result.2
    · exact (elementwise _ _).1 holds.2 cell inBounds

theorem mapM_write_refined (elementwise : Elementwise M) {ib : Nat} {ranges point : List Nat}
    (bounds : Idx.InBounds ranges point) (write : Val × Tensor × AffineMap → Outcome Tensor)
    (same : ∀ value tensor map, write (value, tensor, map) = writeAt tensor map point value) :
    ∀ (values : List Val) (tensors : List Tensor) (maps : List AffineMap) (facts : List (Fact Info)),
      (∀ fact ∈ facts, fact.base.initialized = true) →
      FactsHold M ib (values.map RVal.scalar) (facts.map elementFact) →
      FactsHold M ib (tensors.map RVal.tensor) facts → MapsInBounds ranges maps (facts.map (·.base.ty)) →
      ∃ outputs, (values.zip (tensors.zip maps)).mapM write = .ok outputs ∧
        FactsHold M ib (outputs.map RVal.tensor) facts
  | [], [], [], [], _, _, _, _ => ⟨[], rfl, .nil⟩
  | value :: values, tensor :: tensors, map :: maps, fact :: facts, initialized,
      .cons head tail, .cons current rest, .cons mapped remaining => by
    obtain ⟨output, executed, holds⟩ := writeAt_refined elementwise bounds
      (initialized fact (List.mem_cons_self ..)) current mapped head
    obtain ⟨outputs, done, valid⟩ := mapM_write_refined elementwise bounds write same values tensors maps facts
      (fun fact member => initialized fact (List.mem_cons_of_mem _ member)) tail rest remaining
    exact ⟨output :: outputs, by simp only [List.zip_cons_cons, List.mapM_cons, Outcome.bindOp_eq,
      same, executed, done, Outcome.bind_ok, Outcome.pure_eq], .cons holds valid⟩
  | [], _, _, _ :: _, _, values, _, _ => by cases values
  | _ :: _, _, _, [], _, values, _, _ => by cases values
  | _, [], _, _ :: _, _, _, tensors, _ => by cases tensors
  | _, _ :: _, _, [], _, _, tensors, _ => by cases tensors
  | _, _, [], _ :: _, _, _, _, maps => by cases maps
  | _, _, _ :: _, [], _, _, _, maps => by cases maps

theorem canonize_refined (elementwise : Elementwise M) {ib : Nat} :
    ∀ {tensors : List Tensor} {facts : List (Fact Info)},
    FactsHold M ib (tensors.map RVal.tensor) facts →
    FactsHold M ib (tensors.map fun tensor => RVal.tensor tensor.canonize) facts
  | [], [], _ => .nil
  | tensor :: tensors, fact :: facts, .cons head tail => by
    refine .cons ⟨⟨hasTy_canonize head.1.1, ?_⟩, ?_⟩ (canonize_refined elementwise tail)
    · intro initialized point bounds
      change Idx.InBounds tensor.shape point at bounds
      simpa only [Tensor.canonize, Tensor.canon, bounds, ↓reduceIte] using head.1.2 initialized point bounds
    · apply (elementwise _ _).2
      intro point bounds
      change Idx.InBounds tensor.shape point at bounds
      simpa only [Tensor.canonize, Tensor.canon, bounds, ↓reduceIte] using
        (elementwise _ _).1 head.2 point bounds

#print axioms splitInsOuts_refined
#print axioms readAt_refined
#print axioms mapM_read_refined
#print axioms writeAt_refined
#print axioms mapM_write_refined
#print axioms canonize_refined

end MerlinVerify.Check.Definedness.Refined
