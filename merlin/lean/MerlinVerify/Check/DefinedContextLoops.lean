/- Structured initialization preserves proved elementwise refinements and explicit numerical
context/call premises. Loop coordinates change the context, so stability is proved separately. -/
import MerlinVerify.Check.DefinedElementwise

namespace MerlinVerify.Check.Definedness.Contextual

open MerlinVerify Refined MerlinVerify.Sem.Ops.Linalg

def LoopStable (allowed : Conditions) : Prop :=
  ∀ ctx call point, allowed ctx call → allowed {ctx with loopIndex := point} call

def BodyTotal {Info : Type} (M : Model Info) (ib : Nat) (allowed : Conditions) (call : CallSem)
    (body : Ctx → RegionSem) (inputs outputs : List (Refined.Fact Info)) : Prop :=
  ∀ ctx, ctx.indexBits = ib → allowed ctx call → ∀ args, Refined.FactsHold M ib args inputs →
    ∃ results, body ctx args = .ok results ∧ Refined.FactsHold M ib results outputs

/-- A structured body need only run at the actual bounded points of its enclosing loop.
The numerical/call premise is unchanged; coordinates are a separate checked premise. -/
def PointBodyTotal {Info : Type} (M : Model Info) (ib : Nat) (allowed : Conditions) (call : CallSem)
    (ranges : List Nat) (body : Ctx → RegionSem) (inputs outputs : List (Refined.Fact Info)) : Prop :=
  ∀ ctx, ctx.indexBits = ib → allowed ctx call → Idx.InBounds ranges ctx.loopIndex →
    ∀ args, Refined.FactsHold M ib args inputs →
      ∃ results, body ctx args = .ok results ∧ Refined.FactsHold M ib results outputs

variable {Info : Type} {M : Model Info} {ib : Nat} {allowed : Conditions} {call : CallSem}

theorem BodyTotal.atPoints {ranges : List Nat} {body : Ctx → RegionSem}
    {inputs outputs : List (Refined.Fact Info)} (total : BodyTotal M ib allowed call body inputs outputs) :
    PointBodyTotal M ib allowed call ranges body inputs outputs :=
  fun ctx width permitted _ => total ctx width permitted

theorem bodyAt_point_refined (elementwise : Elementwise M) (stable : LoopStable allowed)
    {ranges point : List Nat} (bounds : Idx.InBounds ranges point) {ctx : Ctx}
    (width : ctx.indexBits = ib) (permitted : allowed ctx call)
    {ins : List RVal} {inMaps outMaps : List AffineMap} {cur : List Tensor}
    {body : Ctx → RegionSem} {inFacts outFacts : List (Refined.Fact Info)}
    (inputs : Refined.FactsHold M ib ins inFacts)
    (current : Refined.FactsHold M ib (cur.map RVal.tensor) outFacts)
    (inputMaps : MapsInBounds ranges inMaps (inFacts.map (·.base.ty)))
    (outputMaps : MapsInBounds ranges outMaps (outFacts.map (·.base.ty)))
    (total : PointBodyTotal M ib allowed call ranges body ((inFacts ++ outFacts).map Refined.elementFact)
      (outFacts.map Refined.elementFact)) :
    ∃ values, bodyAt ctx ins inMaps outMaps cur body point = .ok values ∧
      Refined.FactsHold M ib (values.map RVal.scalar) (outFacts.map Refined.elementFact) := by
  obtain ⟨xs, readInputs, inputFacts⟩ := mapM_read_refined elementwise bounds id
    (fun (value, map) => readAt value map point) (by intro _ _; rfl) ins inMaps inFacts
      (by simpa using inputs) inputMaps
  obtain ⟨ys, readOutputs, outputFacts⟩ := mapM_read_refined elementwise bounds RVal.tensor
    (fun (value, map) => readAt (.tensor value) map point) (by intro _ _; rfl)
      cur outMaps outFacts current outputMaps
  have arguments : Refined.FactsHold M ib ((xs ++ ys).map RVal.scalar)
      ((inFacts ++ outFacts).map Refined.elementFact) := by
    rw [List.map_append, List.map_append]
    exact List.rel_append inputFacts outputFacts
  obtain ⟨results, executed, holds⟩ := total {ctx with loopIndex := point} width
    (stable ctx call point permitted) bounds _ arguments
  have typed : HasTys ib results ((outFacts.map (·.base.ty)).map VTy.eltScalar) := by
    simpa only [Refined.elementFact, Definedness.elementFact, List.map_map, Function.comp_def] using holds.typed
  have count : results.length = cur.length := by
    rw [HasTys.length typed, List.length_map, ← HasTys.length current.typed, List.length_map]
  obtain ⟨values, rfl, _⟩ := scalars_of_typed typed
  refine ⟨values, ?_, holds⟩
  simp only [bodyAt, Outcome.bindOp_eq, readInputs, readOutputs, executed, Outcome.bind_ok,
    count, ne_eq, not_true_eq_false, ↓reduceIte]
  exact mapM_scalars_total _ (by intro _; rfl) values

/-- Reduction steps maintain the checked numerical invariant in execution order. -/
theorem genericSem_point_refined (elementwise : Elementwise M) (stable : LoopStable allowed)
    {ctx : Ctx} (width : ctx.indexBits = ib) (permitted : allowed ctx call)
    {maps : List AffineMap} {iters : List IterKind} {ins outs : List RVal}
    {body : Ctx → RegionSem} {inFacts outFacts : List (Refined.Fact Info)} {ranges : List Nat}
    (structured : StructuredOk maps iters.length (inFacts.map (·.base.ty) ++ outFacts.map (·.base.ty)) ranges)
    (tensors : ∀ fact ∈ outFacts, fact.base.ty.isTensor = true)
    (initialized : ∀ fact ∈ outFacts, fact.base.initialized = true)
    (inputs : Refined.FactsHold M ib ins inFacts) (outputs : Refined.FactsHold M ib outs outFacts)
    (total : PointBodyTotal M ib allowed call ranges body ((inFacts ++ outFacts).map Refined.elementFact)
      (outFacts.map Refined.elementFact)) :
    ∃ results, genericSem ctx maps iters ins outs body = .ok results ∧
      Refined.FactsHold M ib results outFacts := by
  have outputTensors : ∀ ty ∈ outFacts.map (·.base.ty), ty.isTensor = true := by
    intro ty member
    obtain ⟨fact, factMember, rfl⟩ := List.mem_map.1 member
    exact tensors fact factMember
  obtain ⟨initial, same⟩ := tensors_of_typed outputs.typed outputTensors
  subst outs
  have allTyped := List.rel_append inputs.typed outputs.typed
  obtain ⟨mapCount, dimensions, rangesOk, loopRanges⟩ := structured_runtime structured allTyped
  have inputCount : ins.length = (inFacts.map (·.base.ty)).length := HasTys.length inputs.typed
  have inputMaps : MapsInBounds ranges (maps.take ins.length) (inFacts.map (·.base.ty)) := by
    have := List.forall₂_take (inFacts.map (·.base.ty)).length structured.inBounds
    rwa [List.take_left, ← inputCount] at this
  have outputMaps : MapsInBounds ranges (maps.drop ins.length) (outFacts.map (·.base.ty)) := by
    have := List.forall₂_drop (inFacts.map (·.base.ty)).length structured.inBounds
    rwa [List.drop_left, ← inputCount] at this
  have step : ∀ current point, Refined.FactsHold M ib (current.map RVal.tensor) outFacts →
      Idx.InBounds ranges point → ∃ results,
      loopStep ctx ins (maps.take ins.length) (maps.drop ins.length) body current point = .ok results ∧
      Refined.FactsHold M ib (results.map RVal.tensor) outFacts := by
    intro current point holds bounds
    obtain ⟨values, executed, valid⟩ := bodyAt_point_refined elementwise stable bounds width permitted
      inputs holds inputMaps outputMaps total
    obtain ⟨results, written, valid⟩ := mapM_write_refined elementwise bounds
      (fun (value, tensor, map) => writeAt tensor map point value) (by intro _ _ _; rfl)
      values current (maps.drop ins.length) outFacts initialized valid holds outputMaps
    exact ⟨results, by simp only [loopStep, Outcome.bindOp_eq, executed, Outcome.bind_ok, written], valid⟩
  obtain ⟨final, executed, valid⟩ := foldlM_total _ _ (Idx.InBounds ranges) step
    (allPoints ranges) initial (fun point member => (mem_allPoints _ _).1 member) outputs
  refine ⟨final.map fun tensor => RVal.tensor tensor.canonize, ?_, canonize_refined elementwise valid⟩
  simp only [genericSem, mapCount, ne_eq, not_true_eq_false, ↓reduceIte, dimensions,
    Bool.false_eq_true, rangesOk, Outcome.bindOp_eq]
  rw [mapM_tensors_total _ (by intro _; rfl) initial, Outcome.bind_ok, loopRanges,
    Outcome.bind_ok, executed, Outcome.bind_ok, Outcome.pure_eq]

/-- A full overwrite obtains its numerical postcondition from actual body execution, even
when the destination is poison. No numerical premise is invented for the old destination. -/
theorem overwriteSem_point_refined (elementwise : Elementwise M) (stable : LoopStable allowed)
    {ctx : Ctx} (width : ctx.indexBits = ib) (permitted : allowed ctx call)
    {ins : List RVal} {inFacts : List (Refined.Fact Info)} {maps : List AffineMap} {iters : List IterKind}
    {tensor : Tensor} {shape : List Nat} {elt : Ty} {info : Info} {body : Ctx → RegionSem}
    (structured : StructuredOk maps iters.length (inFacts.map (·.base.ty) ++ [.tensor shape elt]) shape)
    (outputMap : maps.drop inFacts.length = [identityMap shape.length])
    (inputs : Refined.FactsHold M ib ins inFacts) (typed : RVal.HasTy ib (.tensor tensor) (.tensor shape elt))
    (total : PointBodyTotal M ib allowed call shape body
      ((inFacts ++ [emptyFact M ⟨.tensor shape elt, false⟩]).map Refined.elementFact)
      [⟨⟨.scalar elt, true⟩, info⟩]) :
    ∃ results, genericSem ctx maps iters ins [.tensor tensor] body = .ok results ∧
      Refined.FactsHold M ib results [⟨⟨.tensor shape elt, true⟩, info⟩] := by
  classical
  have shapeEq := typed.1
  have allTyped : HasTys ib (ins ++ [.tensor tensor]) (inFacts.map (·.base.ty) ++ [.tensor shape elt]) :=
    List.rel_append inputs.typed (.cons typed .nil)
  obtain ⟨mapCount, dimensions, rangesOk, loopRanges⟩ := structured_runtime structured allTyped
  have inputCount : ins.length = inFacts.length := List.Forall₂.length_eq inputs
  have inputMaps : MapsInBounds shape (maps.take ins.length) (inFacts.map (·.base.ty)) := by
    have := List.forall₂_take (inFacts.map (·.base.ty)).length structured.inBounds
    rw [List.take_left] at this
    simpa only [List.length_map, ← inputCount] using this
  have pointResult : ∀ point, Idx.InBounds shape point → ∃ value,
      bodyAt ctx ins (maps.take ins.length) [identityMap shape.length] [tensor] body point = .ok [value] ∧
      (⟨⟨.scalar elt, true⟩, info⟩ : Refined.Fact Info).Holds M ib (.scalar value) := by
    intro point bounds
    obtain ⟨values, executed, holds⟩ := mapM_read_refined elementwise bounds id
      (fun (value, map) => readAt value map point) (by intro _ _; rfl)
      ins (maps.take ins.length) inFacts (by simpa using inputs) inputMaps
    have tensorBounds : Idx.InBounds tensor.shape point := shapeEq ▸ bounds
    have readOutput := readAt_identity tensorBounds
    rw [shapeEq] at readOutput
    have argsHold : Refined.FactsHold M ib ((values ++ [tensor.get point]).map RVal.scalar)
        ((inFacts ++ [emptyFact M ⟨.tensor shape elt, false⟩]).map Refined.elementFact) := by
      rw [List.map_append, List.map_append]
      exact List.rel_append holds
        (.cons ⟨⟨typed.2.2 point bounds, by simp [Refined.elementFact, Definedness.elementFact, emptyFact]⟩,
          M.empty_sound _⟩ .nil)
    obtain ⟨results, bodyExecuted, bodyHolds⟩ := total {ctx with loopIndex := point} width
      (stable ctx call point permitted) bounds _ argsHold
    obtain _ | ⟨head, tail⟩ := bodyHolds
    cases tail
    rename_i result
    cases result with
    | tensor _ => cases head.1.1
    | scalar value =>
      refine ⟨value, ?_, head⟩
      simp only [bodyAt, List.zip_cons_cons, List.zip_nil_left, List.mapM_cons, List.mapM_nil,
        Outcome.bindOp_eq, executed, readOutput, Outcome.bind_ok, Outcome.pure_eq, bodyExecuted,
        List.length_cons, List.length_nil, ne_eq, not_true_eq_false, ↓reduceIte]
  let result : Idx → Val := fun point =>
    if bounds : Idx.InBounds shape point then Classical.choose (pointResult point bounds) else .poison
  have resultHolds : ∀ point, Idx.InBounds shape point →
      bodyAt ctx ins (maps.take ins.length) [identityMap shape.length] [tensor] body point = .ok [result point] ∧
      (⟨⟨.scalar elt, true⟩, info⟩ : Refined.Fact Info).Holds M ib (.scalar (result point)) := by
    intro point bounds
    simpa only [result, dite_eq_left bounds] using Classical.choose_spec (pointResult point bounds)
  refine ⟨[.tensor (Tensor.canon shape tensor.elt result)], ?_, .cons ⟨⟨⟨rfl, typed.2.1, ?_⟩, ?_⟩, ?_⟩ .nil⟩
  · exact genericSem_parallel1 ctx maps iters ins tensor body shape result
      (by simpa only [List.length_append, List.length_cons, List.length_nil] using mapCount)
      dimensions rangesOk loopRanges shapeEq (inputCount ▸ outputMap) (fun point bounds => (resultHolds point bounds).1)
  · intro point bounds
    simpa only [Tensor.canon, bounds, ↓reduceIte, RVal.HasTy] using (resultHolds point bounds).2.1.1
  · intro _ point bounds
    change Idx.InBounds shape point at bounds
    simpa only [Tensor.canon, bounds, ↓reduceIte, RVal.Defined] using (resultHolds point bounds).2.1.2 rfl
  · apply (elementwise _ _).2
    intro point bounds
    change Idx.InBounds shape point at bounds
    simpa only [Tensor.canon, bounds, ↓reduceIte] using (resultHolds point bounds).2.2

/-- The original interfaces retain their stronger, coordinate-independent body obligation. -/
theorem bodyAt_refined (elementwise : Elementwise M) (stable : LoopStable allowed)
    {ranges point : List Nat} (bounds : Idx.InBounds ranges point) {ctx : Ctx}
    (width : ctx.indexBits = ib) (permitted : allowed ctx call)
    {ins : List RVal} {inMaps outMaps : List AffineMap} {cur : List Tensor}
    {body : Ctx → RegionSem} {inFacts outFacts : List (Refined.Fact Info)}
    (inputs : Refined.FactsHold M ib ins inFacts)
    (current : Refined.FactsHold M ib (cur.map RVal.tensor) outFacts)
    (inputMaps : MapsInBounds ranges inMaps (inFacts.map (·.base.ty)))
    (outputMaps : MapsInBounds ranges outMaps (outFacts.map (·.base.ty)))
    (total : BodyTotal M ib allowed call body ((inFacts ++ outFacts).map Refined.elementFact)
      (outFacts.map Refined.elementFact)) :
    ∃ values, bodyAt ctx ins inMaps outMaps cur body point = .ok values ∧
      Refined.FactsHold M ib (values.map RVal.scalar) (outFacts.map Refined.elementFact) :=
  bodyAt_point_refined elementwise stable bounds width permitted inputs current inputMaps outputMaps total.atPoints

theorem genericSem_refined (elementwise : Elementwise M) (stable : LoopStable allowed)
    {ctx : Ctx} (width : ctx.indexBits = ib) (permitted : allowed ctx call)
    {maps : List AffineMap} {iters : List IterKind} {ins outs : List RVal}
    {body : Ctx → RegionSem} {inFacts outFacts : List (Refined.Fact Info)} {ranges : List Nat}
    (structured : StructuredOk maps iters.length (inFacts.map (·.base.ty) ++ outFacts.map (·.base.ty)) ranges)
    (tensors : ∀ fact ∈ outFacts, fact.base.ty.isTensor = true)
    (initialized : ∀ fact ∈ outFacts, fact.base.initialized = true)
    (inputs : Refined.FactsHold M ib ins inFacts) (outputs : Refined.FactsHold M ib outs outFacts)
    (total : BodyTotal M ib allowed call body ((inFacts ++ outFacts).map Refined.elementFact)
      (outFacts.map Refined.elementFact)) :
    ∃ results, genericSem ctx maps iters ins outs body = .ok results ∧
      Refined.FactsHold M ib results outFacts :=
  genericSem_point_refined elementwise stable width permitted structured tensors initialized inputs outputs total.atPoints

theorem overwriteSem_refined (elementwise : Elementwise M) (stable : LoopStable allowed)
    {ctx : Ctx} (width : ctx.indexBits = ib) (permitted : allowed ctx call)
    {ins : List RVal} {inFacts : List (Refined.Fact Info)} {maps : List AffineMap} {iters : List IterKind}
    {tensor : Tensor} {shape : List Nat} {elt : Ty} {info : Info} {body : Ctx → RegionSem}
    (structured : StructuredOk maps iters.length (inFacts.map (·.base.ty) ++ [.tensor shape elt]) shape)
    (outputMap : maps.drop inFacts.length = [identityMap shape.length])
    (inputs : Refined.FactsHold M ib ins inFacts) (typed : RVal.HasTy ib (.tensor tensor) (.tensor shape elt))
    (total : BodyTotal M ib allowed call body
      ((inFacts ++ [emptyFact M ⟨.tensor shape elt, false⟩]).map Refined.elementFact)
      [⟨⟨.scalar elt, true⟩, info⟩]) :
    ∃ results, genericSem ctx maps iters ins [.tensor tensor] body = .ok results ∧
      Refined.FactsHold M ib results [⟨⟨.tensor shape elt, true⟩, info⟩] :=
  overwriteSem_point_refined elementwise stable width permitted structured outputMap inputs typed total.atPoints

#print axioms BodyTotal.atPoints
#print axioms bodyAt_point_refined
#print axioms genericSem_point_refined
#print axioms overwriteSem_point_refined
#print axioms bodyAt_refined
#print axioms genericSem_refined
#print axioms overwriteSem_refined

end MerlinVerify.Check.Definedness.Contextual
