/- Operation-level heap semantics. Each store updates the actual heap; the
writer-log interpretation is derived by simulation, not a whole-run premise.
Pure SSA bindings are lexical and cannot inspect or modify the private heap. -/
import Merlin.GraphLoopsMemorySCFEffects

set_option autoImplicit false

namespace Merlin.GraphLoopsMemorySCF

/-- General relational simulation retains the ENTIRE ordered SCF trace. -/
theorem SCF_simulate {A B : Type} {source : Nat → A → A → Prop} {target : Nat → B → B → Prop}
    (related : A → B → Prop)
    (step : ∀ index before after other, related before other → source index before after →
      ∃ next, target index other next ∧ related after next)
    {upper stride lower : Nat} {before after : A} {other : B} {trace : List Nat}
    (same : related before other) (run : GraphLoopsSegment.Execution source upper stride lower before trace after) :
    ∃ next, GraphLoopsSegment.Execution target upper stride lower other trace next ∧ related after next := by
  induction run generalizing other with
  | done _ stopped => exact ⟨other, .done other stopped, same⟩
  | more inside transition remaining induction =>
    obtain ⟨middle, first, sameMiddle⟩ := step _ _ _ other same transition
    obtain ⟨next, suffix, sameNext⟩ := induction sameMiddle
    exact ⟨next, .more inside first suffix, sameNext⟩

/-- Evaluate the explicit SSA operands, check coordinates, then perform ONE
private store. The intermediate heap is the input of the following operation. -/
def StoreBody {Value : Type} (meaning : PureMeaning Value) (leaf : Leaf) (shape : List Nat)
    (iv object : Nat) (captures : Env Value) (index : Nat) (before after : Heap Value) : Prop :=
  ∃ coordinates value, LeafResult meaning leaf (put captures iv (.index index)) coordinates value ∧
    GraphLoopsReuse.Geometry.Coordinate shape coordinates ∧
    after = GraphLoopsReuse.replaceCell before object (GraphLoopsReuse.Geometry.flatten shape coordinates) value

def NestedStores {Value : Type} (meaning : PureMeaning Value) (ops : List PureOp) (leaf : Leaf)
    (rows columns outerIV innerIV object : Nat) (captures : Env Value) (row : Nat) (before after : Heap Value) : Prop :=
  ∃ rowEnv trace, EvalPure meaning ops (put captures outerIV (.index row)) rowEnv ∧
    GraphLoopsSegment.Execution (StoreBody meaning leaf [rows, columns] innerIV object rowEnv) columns 1 0 before trace after

def HeapBody {Value : Type} (meaning : PureMeaning Value) (program : Initializer) (object : Nat) (captures : Env Value) :
    Nat → Heap Value → Heap Value → Prop :=
  match program.shape, program.ivs with
  | [extent], [iv] => StoreBody meaning program.leaf [extent] iv object captures
  | [rows, columns], [outerIV, innerIV] => NestedStores meaning program.outerPrefix program.leaf rows columns outerIV innerIV object captures
  | _, _ => fun _ _ _ => False

def HeapRun {Value : Type} (meaning : PureMeaning Value) (program : Initializer) (tile object : Nat) (segmented : Bool)
    (captures : Env Value) (before after : Heap Value) : Prop :=
  ∃ trace, if segmented then GraphLoopsSegment.Segmented (HeapBody meaning program object captures) program.rows 1 tile 0
    before trace after else GraphLoopsSegment.Execution (HeapBody meaning program object captures) program.rows 1 0 before trace after

/-- Applying one emitted event equals the actual primitive store. -/
theorem applyHeapWrites_snoc {Value : Type} (object : Nat) (writes : Log Value) (heap : Heap Value) (event : Write Value) :
    applyHeapWrites object (writes ++ [event]) heap =
      GraphLoopsReuse.replaceCell (applyHeapWrites object writes heap) object event.1 event.2 := by
  rw [applyHeapWrites_append]
  rfl

theorem LeafBody.heap_forward {Value : Type} {meaning : PureMeaning Value} {leaf : Leaf} {shape : List Nat}
    {iv object index : Nat} {captures : Env Value} {before after : Log Value} (base : Heap Value)
    (run : LeafBody meaning leaf shape iv captures index before after) :
    StoreBody meaning leaf shape iv object captures index (applyHeapWrites object before base) (applyHeapWrites object after base) := by
  obtain ⟨emitted, stored, rfl⟩ := run
  obtain ⟨coordinates, value, computed, inside, rfl⟩ := stored
  exact ⟨coordinates, value, computed, inside, applyHeapWrites_snoc object before base (GraphLoopsReuse.Geometry.flatten shape coordinates, value)⟩

theorem StoreBody.log_backward {Value : Type} {meaning : PureMeaning Value} {leaf : Leaf} {shape : List Nat}
    {iv object index : Nat} {captures : Env Value} {before : Log Value} (base : Heap Value) {after : Heap Value}
    (run : StoreBody meaning leaf shape iv object captures index (applyHeapWrites object before base) after) :
    ∃ writes, LeafBody meaning leaf shape iv captures index before writes ∧ after = applyHeapWrites object writes base := by
  obtain ⟨coordinates, value, computed, inside, rfl⟩ := run
  refine ⟨before ++ [(GraphLoopsReuse.Geometry.flatten shape coordinates, value)], ?_, ?_⟩
  · exact ⟨⟨put captures iv (.index index), _⟩, ⟨coordinates, value, computed, inside, rfl⟩, rfl⟩
  · exact (applyHeapWrites_snoc object before base (GraphLoopsReuse.Geometry.flatten shape coordinates, value)).symm

theorem NestedBody.heap_forward {Value : Type} {meaning : PureMeaning Value} {ops : List PureOp} {leaf : Leaf}
    {rows columns outerIV innerIV object row : Nat} {captures : Env Value} {before after : Log Value} (base : Heap Value)
    (run : NestedBody meaning ops leaf rows columns outerIV innerIV captures row before after) :
    NestedStores meaning ops leaf rows columns outerIV innerIV object captures row
      (applyHeapWrites object before base) (applyHeapWrites object after base) := by
  obtain ⟨rowEnv, trace, computed, inner⟩ := run
  obtain ⟨next, translated, equal⟩ := SCF_simulate (fun log heap => heap = applyHeapWrites object log base)
    (fun _ _ _ _ same body => by subst_vars; exact ⟨_, LeafBody.heap_forward base body, rfl⟩) rfl inner
  exact ⟨rowEnv, trace, computed, equal ▸ translated⟩

theorem NestedStores.log_backward {Value : Type} {meaning : PureMeaning Value} {ops : List PureOp} {leaf : Leaf}
    {rows columns outerIV innerIV object row : Nat} {captures : Env Value} {before : Log Value} (base : Heap Value) {after : Heap Value}
    (run : NestedStores meaning ops leaf rows columns outerIV innerIV object captures row (applyHeapWrites object before base) after) :
    ∃ writes, NestedBody meaning ops leaf rows columns outerIV innerIV captures row before writes ∧ after = applyHeapWrites object writes base := by
  obtain ⟨rowEnv, trace, computed, inner⟩ := run
  obtain ⟨writes, translated, equal⟩ := SCF_simulate (fun heap log => heap = applyHeapWrites object log base)
    (fun _ _ _ _ same body => by rw [same] at body; exact StoreBody.log_backward base body) rfl inner
  exact ⟨writes, ⟨rowEnv, trace, computed, translated⟩, equal⟩

/-- Shape matching chooses the same lexical body on both sides. -/
theorem Body.heap_forward {Value : Type} {meaning : PureMeaning Value} {program : Initializer}
    {object index : Nat} {captures : Env Value} {before after : Log Value} (base : Heap Value) (valid : program.WellFormed)
    (run : Body meaning program captures index before after) :
    HeapBody meaning program object captures index (applyHeapWrites object before base) (applyHeapWrites object after base) := by
  rcases valid.1 with one | two
  · obtain ⟨shape, ivs, _positive, _empty⟩ := one
    simp only [Body, HeapBody, shape, ivs] at run ⊢
    exact LeafBody.heap_forward base run
  · obtain ⟨shape, ivs, _rows, _columns, _different⟩ := two
    simp only [Body, HeapBody, shape, ivs] at run ⊢
    exact NestedBody.heap_forward base run

theorem HeapBody.log_backward {Value : Type} {meaning : PureMeaning Value} {program : Initializer}
    {object index : Nat} {captures : Env Value} {before : Log Value} (base : Heap Value) (valid : program.WellFormed) {after : Heap Value}
    (run : HeapBody meaning program object captures index (applyHeapWrites object before base) after) :
    ∃ writes, Body meaning program captures index before writes ∧ after = applyHeapWrites object writes base := by
  rcases valid.1 with one | two
  · obtain ⟨shape, ivs, _positive, _empty⟩ := one
    simp only [Body, HeapBody, shape, ivs] at run ⊢
    exact StoreBody.log_backward base run
  · obtain ⟨shape, ivs, _rows, _columns, _different⟩ := two
    simp only [Body, HeapBody, shape, ivs] at run ⊢
    exact NestedStores.log_backward base run

/-- Actual operational SCF execution is equivalent to the DERIVED ordered log
interpretation, in BOTH directions. No scalar totality/source success is assumed. -/
theorem heap_run_iff_log {Value : Type} (meaning : PureMeaning Value) (program : Initializer) (tile object : Nat)
    (captures : Env Value) (before after : Heap Value) (valid : program.WellFormed) :
    HeapRun meaning program tile object false captures before after ↔
      ∃ writes, Run meaning program tile false captures writes ∧ after = applyHeapWrites object writes before := by
  constructor
  · rintro ⟨trace, run⟩
    have source : GraphLoopsSegment.Execution (HeapBody meaning program object captures) program.rows 1 0 before trace after := run
    obtain ⟨writes, translated, equal⟩ := SCF_simulate (fun heap log => heap = applyHeapWrites object log before)
      (fun _ _ _ _ same body => by rw [same] at body; exact HeapBody.log_backward before valid body) (show before = applyHeapWrites object [] before from rfl) source
    exact ⟨writes, ⟨trace, translated⟩, equal⟩
  · rintro ⟨writes, ⟨trace, run⟩, rfl⟩
    have source : GraphLoopsSegment.Execution (Body meaning program captures) program.rows 1 0 [] trace writes := run
    obtain ⟨next, translated, equal⟩ := SCF_simulate (fun log heap => heap = applyHeapWrites object log before)
      (fun _ _ _ _ same body => by subst_vars; exact ⟨_, Body.heap_forward before valid body, rfl⟩) rfl source
    exact ⟨trace, equal ▸ translated⟩

/-- Reuse the existing general ordered segmentation root over the ACTUAL heap
body, not a separately admitted atomic initializer meaning. -/
theorem segmentation_exact_heaps {Value : Type} (meaning : PureMeaning Value) (program : Initializer)
    (tile object : Nat) (positive : 0 < tile) (captures : Env Value) (before after : Heap Value) :
    HeapRun meaning program tile object false captures before after ↔ HeapRun meaning program tile object true captures before after := by
  unfold HeapRun
  simp only [Bool.false_eq_true, ↓reduceIte]
  apply exists_congr
  intro trace
  exact GraphLoopsSegment.complete_trace_iff _ _ 1 tile 0 (by decide) positive before after trace

end Merlin.GraphLoopsMemorySCF
