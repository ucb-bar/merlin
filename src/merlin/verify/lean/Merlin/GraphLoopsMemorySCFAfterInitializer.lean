/- Exact syntax of the generated rank-two initializer, with actual add/min,
dynamic inner bound, scoped induction variables and both yields. UNCOMPILED.
The independent target term must equal this syntax; no copied string suffices. -/
import Merlin.GraphLoopsMemorySCFAfterChunks
import Merlin.GraphLoopsShapeEpochInitializer

set_option autoImplicit false

namespace Merlin.GraphLoopsMemorySCF.After

open GraphLoopsShape

structure TiledInitializer where
  uniform : Epoch.Uniform
  tile : Binding
  tileIV : Binding
  sum : Binding
  bound : Binding
  addKey : String
  minKey : String
  deriving DecidableEq, Repr

def TiledInitializer.inner (program : TiledInitializer) : Code :=
  .region program.uniform.loopKey [program.tileIV, program.bound, program.uniform.one]
    [program.uniform.outer] [] program.uniform.rowCode (.returned [])

def TiledInitializer.body (program : TiledInitializer) : Code :=
  .action program.addKey [program.tileIV, program.tile] [program.sum]
    (.action program.minKey [program.sum, program.uniform.rows] [program.bound] program.inner)

def TiledInitializer.code (program : TiledInitializer) : Code :=
  .region program.uniform.loopKey [program.uniform.zero, program.uniform.rows, program.tile]
    [program.tileIV] [] program.body (.returned [])

/-- An actual index operation performs one ordinary typed SSA assignment and
frames the entire world. This is a per-instruction lemma, not whole-loop equality. -/
theorem eval_index_action {Value World : Type} (meaning : GraphLoopsShape.Interpretation Value World)
    (bits : Nat) (opcode : IndexOpcode) (key : String)
    (selected : meaning.action key = IndexAction bits opcode)
    (left right result : Binding) (kind : result.kind = .index) (plain : result.known = none)
    (aliases : Aliases) (rest : Code) (env : GraphLoopsShape.Env Value) (a b : Nat)
    (leftValue : env left.name = .index a) (rightValue : env right.name = .index b)
    (leftFits : GraphLoopsSegment.FitsSigned bits a) (rightFits : GraphLoopsSegment.FitsSigned bits b)
    (resultFits : GraphLoopsSegment.FitsSigned bits (opcode.apply a b))
    (before : World) (outputs : List (Runtime Value)) (after : World) :
    Eval meaning false aliases (.action key [left, right] [result] rest) env before outputs after ↔
      Eval meaning false (extendAliases aliases [result]) rest
        (GraphLoopsShape.put env result.name (.index (opcode.apply a b))) before outputs after := by
  have inputs : readArguments false aliases env [left, right] = [.index a, .index b] := by
    simp only [readArguments, List.map_cons, List.map_nil, Bool.false_eq_true, ↓reduceIte, leftValue, rightValue]
  have typed := Epoch.index_parameter (Value := Value) result (opcode.apply a b) kind plain
  simp only [Eval, selected, inputs]
  constructor
  · rintro ⟨values, middle, action, _, suffix⟩
    obtain ⟨_, _, _, rfl, rfl⟩ := action
    simpa only [GraphLoopsShape.assign] using suffix
  · intro suffix
    refine ⟨[.index (opcode.apply a b)], before, ⟨leftFits, rightFits, resultFits, rfl, rfl⟩, typed, ?_⟩
    simpa only [GraphLoopsShape.assign] using suffix

/-- The two ordinary instructions compute the actual dynamic inner upper
bound in the scoped environment. The entire remaining inner loop stays in the
conclusion; neither its effects nor its heap are replaced by an equality premise. -/
theorem TiledInitializer.control_prefix {Value World : Type} (program : TiledInitializer)
    (meaning : GraphLoopsShape.Interpretation Value World) (bounds : GraphLoopsSegment.Bounds) (valid : bounds.Valid)
    (add : meaning.action program.addKey = IndexAction bounds.bits .add)
    (minimum : meaning.action program.minKey = IndexAction bounds.bits .minimum)
    (sumKind : program.sum.kind = .index) (sumPlain : program.sum.known = none)
    (boundKind : program.bound.kind = .index) (boundPlain : program.bound.known = none)
    (rowsFresh : program.uniform.rows.name ≠ program.sum.name)
    (aliases : Aliases) (env : GraphLoopsShape.Env Value) (outer : Nat) (live : outer < bounds.upper)
    (outerValue : env program.tileIV.name = .index outer)
    (tileValue : env program.tile.name = .index (bounds.step * bounds.tile))
    (rowsValue : env program.uniform.rows.name = .index bounds.upper)
    (before : World) (outputs : List (Runtime Value)) (after : World) :
    Eval meaning false aliases program.body env before outputs after ↔
      Eval meaning false (extendAliases (extendAliases aliases [program.sum]) [program.bound]) program.inner
        (GraphLoopsShape.put (GraphLoopsShape.put env program.sum.name (.index (outer + bounds.step * bounds.tile))) program.bound.name
          (.index (min (outer + bounds.step * bounds.tile) bounds.upper))) before outputs after := by
  have constants := GraphLoopsSegment.Bounds.constants valid
  have controls := GraphLoopsSegment.Bounds.outer_control valid live
  rw [TiledInitializer.body]
  rw [eval_index_action meaning bounds.bits .add program.addKey add program.tileIV program.tile program.sum
    sumKind sumPlain aliases _ env outer (bounds.step * bounds.tile) outerValue tileValue
    controls.1 constants.2.2.2 controls.2.1]
  have sumValue : GraphLoopsShape.put env program.sum.name (.index (outer + bounds.step * bounds.tile)) program.sum.name =
      .index (outer + bounds.step * bounds.tile) := by simp only [GraphLoopsShape.put, ↓reduceIte]
  have upperValue : GraphLoopsShape.put env program.sum.name (.index (outer + bounds.step * bounds.tile)) program.uniform.rows.name =
      .index bounds.upper := by simp only [GraphLoopsShape.put, rowsFresh, ↓reduceIte, rowsValue]
  exact eval_index_action meaning bounds.bits .minimum program.minKey minimum program.sum program.uniform.rows program.bound
    boundKind boundPlain _ program.inner _ (outer + bounds.step * bounds.tile) bounds.upper sumValue upperValue
    controls.2.1 constants.2.1 controls.2.2 before outputs after

end Merlin.GraphLoopsMemorySCF.After
