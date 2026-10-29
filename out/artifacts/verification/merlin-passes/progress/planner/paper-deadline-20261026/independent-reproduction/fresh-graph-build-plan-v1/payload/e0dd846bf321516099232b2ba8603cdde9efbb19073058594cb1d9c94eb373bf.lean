/- Complete cell coverage DERIVED from ordered partial loop/store executions.
No successful execution is assumed to exist; no tensor-sized proof is generated. -/
import Merlin.GraphLoopsMemorySCFLoops

set_option autoImplicit false

namespace Merlin.GraphLoopsMemorySCF

namespace Repeat

theorem monotone {Value : Type} {body : Nat → Log Value → Log Value → Prop}
    (preserve : ∀ index before after, body index before after → ∀ event ∈ before, event ∈ after)
    {count lower : Nat} {before after : Log Value} {trace : List Nat}
    (run : GraphLoopsUnroll.Repeat body 1 count lower before trace after) : ∀ event ∈ before, event ∈ after := by
  induction run with
  | nil => exact fun _ member => member
  | cons event remaining induction => exact fun write member => induction write (preserve _ _ _ event write member)

theorem covers {Value : Type} {body : Nat → Log Value → Log Value → Prop} (address : Nat → Nat)
    (preserve : ∀ index before after, body index before after → ∀ event ∈ before, event ∈ after)
    (emits : ∀ index before after, body index before after → ∃ value, (address index, value) ∈ after)
    {count lower : Nat} {before after : Log Value} {trace : List Nat}
    (run : GraphLoopsUnroll.Repeat body 1 count lower before trace after) :
    ∀ index, lower ≤ index → index < lower + count → ∃ value, (address index, value) ∈ after := by
  induction run with
  | nil => intro index start finish; omega
  | @cons count lower before middle after rest event remaining induction =>
    intro index start finish
    by_cases first : index = lower
    · subst index
      obtain ⟨value, member⟩ := emits lower before middle event
      exact ⟨value, monotone preserve remaining _ member⟩
    · exact induction index (by omega) (by omega)

end Repeat

theorem SCF_covers {Value : Type} {body : Nat → Log Value → Log Value → Prop} (address : Nat → Nat)
    (preserve : ∀ index before after, body index before after → ∀ event ∈ before, event ∈ after)
    (emits : ∀ index before after, body index before after → ∃ value, (address index, value) ∈ after)
    {extent : Nat} {before after : Log Value} {trace : List Nat}
    (run : GraphLoopsSegment.Execution body extent 1 0 before trace after) :
    ∀ index, index < extent → ∃ value, (address index, value) ∈ after := by
  have correct : GraphLoopsUnroll.CountCorrect 0 extent 1 extent := by
    simpa only [Nat.one_mul, Nat.zero_add] using GraphLoopsUnroll.CountCorrect.aligned 0 1 extent (by decide)
  have repeated := (GraphLoopsUnroll.repeat_iff_execution body extent 1 extent 0 before after trace correct).mpr run
  exact fun index inside => Repeat.covers address preserve emits repeated index (Nat.zero_le _) (by omega)

theorem LeafBody.emits_one {Value : Type} {meaning : PureMeaning Value} {leaf : Leaf} {extent iv index : Nat}
    {captures : Env Value} {before after : Log Value} (run : LeafBody meaning leaf [extent] iv captures index before after)
    (indices : leaf.indices = [iv]) (noShadow : ∀ name ∈ leaf.indices, name ∉ definitions leaf.ops) :
    ∃ value, (index, value) ∈ after := by
  obtain ⟨emitted, stored, rfl⟩ := run
  have binding : read (put captures iv (.index index)) leaf.indices = some ([index].map Datum.index) := by
    simp [indices, read, put]
  obtain ⟨value, _scope, equal⟩ := stored.address noShadow binding
  refine ⟨value, ?_⟩
  rw [equal]
  have member := List.mem_append_right before
    (List.mem_cons_self (a := (index, value)) (l := []))
  simpa only [GraphLoopsReuse.Geometry.flatten, GraphLoopsReuse.Geometry.volume, Nat.mul_one, Nat.add_zero] using member

theorem LeafBody.emits_two {Value : Type} {meaning : PureMeaning Value} {leaf : Leaf}
    {rows columns outerIV innerIV row column : Nat} {captures : Env Value} {before after : Log Value}
    (run : LeafBody meaning leaf [rows, columns] innerIV captures column before after)
    (indices : leaf.indices = [outerIV, innerIV]) (different : outerIV ≠ innerIV)
    (outerBinding : captures outerIV = some (.index row))
    (noShadow : ∀ name ∈ leaf.indices, name ∉ definitions leaf.ops) :
    ∃ value, (row * columns + column, value) ∈ after := by
  obtain ⟨emitted, stored, rfl⟩ := run
  have binding : read (put captures innerIV (.index column)) leaf.indices = some ([row, column].map Datum.index) := by
    simp [indices, read, put, different, outerBinding]
  obtain ⟨value, _scope, equal⟩ := stored.address noShadow binding
  refine ⟨value, ?_⟩
  rw [equal]
  exact List.mem_append_right _ (by
    simpa only [GraphLoopsReuse.Geometry.flatten, GraphLoopsReuse.Geometry.volume, Nat.mul_one, Nat.add_zero] using
      List.mem_cons_self (a := (row * columns + column, value)) (l := []))

theorem NestedBody.emits_row {Value : Type} {meaning : PureMeaning Value} {ops : List PureOp} {leaf : Leaf}
    {rows columns outerIV innerIV row : Nat} {captures : Env Value} {before after : Log Value}
    (run : NestedBody meaning ops leaf rows columns outerIV innerIV captures row before after)
    (indices : leaf.indices = [outerIV, innerIV]) (different : outerIV ≠ innerIV)
    (outerFresh : outerIV ∉ definitions ops) (leafFresh : ∀ name ∈ leaf.indices, name ∉ definitions leaf.ops) :
    ∀ column, column < columns → ∃ value, (row * columns + column, value) ∈ after := by
  obtain ⟨rowEnv, trace, evaluated, innerRun⟩ := run
  have rowBinding : rowEnv outerIV = some (.index row) :=
    (evaluated.frame outerFresh).trans (put_at captures outerIV (.index row))
  exact SCF_covers (fun column => row * columns + column)
    (fun _ _ _ leafRun => LeafBody.monotone leafRun)
    (fun _ _ _ leafRun => LeafBody.emits_two leafRun indices different rowBinding leafFresh) innerRun

end Merlin.GraphLoopsMemorySCF
