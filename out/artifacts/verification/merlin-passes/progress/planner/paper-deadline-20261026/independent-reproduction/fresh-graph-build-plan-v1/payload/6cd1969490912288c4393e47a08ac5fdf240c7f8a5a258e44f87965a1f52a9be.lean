/- Finite binding facts for the partial semantic adapter. These do not yet
constitute the surrounding epoch/public whole-program simulation. -/
import Merlin.GraphLoopsShapeEpochInitializer
import Merlin.GraphLoopsShapeEpochNormalize

set_option autoImplicit false

namespace Merlin.GraphLoopsShape.Epoch

def topRegions : Code → List Code
  | .returned _ => []
  | .cast _ _ _ _ rest => topRegions rest
  | .dimension _ _ _ _ _ _ rest => topRegions rest
  | .constant _ _ rest => topRegions rest
  | .action _ _ _ rest => topRegions rest
  | .region key operands parameters results body rest =>
      .region key operands parameters results body (.returned []) :: topRegions rest

def Uniform.Facts (program : Uniform) (rows columns : Nat) (element : String) : Prop :=
  program.seed.kind = .scalar element ∧ program.seed.known = none ∧
  program.buffer.kind = .view (type [rows, columns] element) ∧ program.buffer.known = none ∧
  program.zero.kind = .index ∧ program.zero.known = some 0 ∧
  program.rows.kind = .index ∧ program.rows.known = some rows ∧
  program.columns.kind = .index ∧ program.columns.known = some columns ∧
  program.one.kind = .index ∧ program.one.known = some 1 ∧
  program.outer.kind = .index ∧ program.outer.known = none ∧
  program.inner.kind = .index ∧ program.inner.known = none ∧
  program.seed.name ≠ program.outer.name ∧ program.seed.name ≠ program.inner.name ∧
  program.buffer.name ≠ program.outer.name ∧ program.buffer.name ≠ program.inner.name ∧
  program.outer.name ≠ program.inner.name ∧ program.zero.name ≠ program.outer.name ∧
  program.columns.name ≠ program.outer.name ∧ program.one.name ≠ program.outer.name

instance (program : Uniform) (rows columns : Nat) (element : String) : Decidable (program.Facts rows columns element) := by
  unfold Uniform.Facts
  infer_instance

/-- Derive an initializer execution equivalence from recomputed finite facts
and concrete values in its currently bound input environment. The latter facts
must be supplied by the whole-program SSA/public simulation, still separate. -/
theorem Uniform.checked_eval {Value Public : Type} (base : Interpretation Value (MemoryWorld Value Public))
    (classify : String → MemoryOpcode) (program : Uniform) (rows columns : Nat) (element : String)
    (checked : decide (program.Facts rows columns element) = true)
    (store : classify program.storeKey = .store) (loop : classify program.loopKey = .loop)
    (aliases : Aliases) (env : Env Value) (object tile : Nat) (value : Value)
    (seedValue : env program.seed.name = .scalar element value)
    (bufferValue : env program.buffer.name = view object [rows, columns] element)
    (zeroValue : env program.zero.name = .index 0) (rowsValue : env program.rows.name = .index rows)
    (columnsValue : env program.columns.name = .index columns) (oneValue : env program.one.name = .index 1)
    (meaning : GraphLoopsMemorySCF.PureMeaning Value) (before after : MemoryWorld Value Public)
    (outputs : List (Runtime Value)) :
    Eval (withMemory base classify) false aliases program.code env before outputs after ↔
      outputs = [] ∧ after.observed = before.observed ∧
        GraphLoopsMemorySCF.HeapRun meaning
          (GraphLoopsMemorySCF.Witness.uniform rows columns program.outer.name program.inner.name program.seed.name)
          tile object false (eraseEnv env) before.heap after.heap := by
  have facts := of_decide_eq_true checked
  rcases facts with ⟨_, _, _, _, _, _, _, _, _, _, _, _, outerKind, outerPlain, innerKind, innerPlain,
    seedOuter, seedInner, bufferOuter, bufferInner, separate, zeroOuter, columnsOuter, oneOuter⟩
  exact program.eval_iff base classify store loop aliases env object rows columns tile element value
    seedValue bufferValue zeroValue rowsValue columnsValue oneValue seedOuter seedInner bufferOuter bufferInner separate
    zeroOuter columnsOuter oneOuter outerKind outerPlain innerKind innerPlain meaning before after outputs

end Merlin.GraphLoopsShape.Epoch
