/- Actual nested initializer semantics in the old partial SCF relation.
No iter_args are admitted here: SSA captures are lexical inputs and temporary
pure results never escape a body. The carried state is the complete ordered
store log; its compositional heap interpretation is proved separately. -/
import Merlin.GraphLoopsMemorySCFScope

set_option autoImplicit false

namespace Merlin.GraphLoopsMemorySCF

def LeafBody {Value : Type} (meaning : PureMeaning Value) (leaf : Leaf) (shape : List Nat) (iv : Nat)
    (captures : Env Value) (index : Nat) (before after : Log Value) : Prop :=
  ∃ emitted, Emit meaning leaf shape ⟨put captures iv (.index index), before⟩ emitted ∧ after = emitted.writes

def NestedBody {Value : Type} (meaning : PureMeaning Value) (ops : List PureOp) (leaf : Leaf)
    (rows columns outerIV innerIV : Nat) (captures : Env Value) (row : Nat) (before after : Log Value) : Prop :=
  ∃ rowEnv trace, EvalPure meaning ops (put captures outerIV (.index row)) rowEnv ∧
    GraphLoopsSegment.Execution (LeafBody meaning leaf [rows, columns] innerIV rowEnv) columns 1 0 before trace after

def Body {Value : Type} (meaning : PureMeaning Value) (program : Initializer) (captures : Env Value) :
    Nat → Log Value → Log Value → Prop :=
  match program.shape, program.ivs with
  | [extent], [iv] => LeafBody meaning program.leaf [extent] iv captures
  | [rows, columns], [outerIV, innerIV] => NestedBody meaning program.outerPrefix program.leaf rows columns outerIV innerIV captures
  | _, _ => fun _ _ _ => False

/-- Mode false is source SCF; true is exactly the generated outer/chunk loops.
Every rank2 original body is itself a partial inner SCF execution. -/
def Run {Value : Type} (meaning : PureMeaning Value) (program : Initializer) (tile : Nat) (segmented : Bool)
    (captures : Env Value) (writes : Log Value) : Prop :=
  ∃ trace, if segmented then GraphLoopsSegment.Segmented (Body meaning program captures) (program.shape.headD 0) 1 tile 0
    [] trace writes else GraphLoopsSegment.Execution (Body meaning program captures) (program.shape.headD 0) 1 0 [] trace writes

theorem LeafBody.monotone {Value : Type} {meaning : PureMeaning Value} {leaf : Leaf} {shape : List Nat} {iv index : Nat}
    {captures : Env Value} {before after : Log Value} (run : LeafBody meaning leaf shape iv captures index before after) :
    ∀ event ∈ before, event ∈ after := by
  obtain ⟨emitted, stored, rfl⟩ := run
  exact stored.monotone

theorem SCF_monotone {Value : Type} {body : Nat → Log Value → Log Value → Prop} {upper step lower : Nat}
    (monotone : ∀ index before after, body index before after → ∀ event ∈ before, event ∈ after)
    {before after : Log Value} {trace : List Nat}
    (run : GraphLoopsSegment.Execution body upper step lower before trace after) :
    ∀ event ∈ before, event ∈ after := by
  induction run with
  | done => exact fun _ member => member
  | more _ event _ induction => exact fun write member => induction write (monotone _ _ _ event write member)

theorem NestedBody.monotone {Value : Type} {meaning : PureMeaning Value} {ops : List PureOp} {leaf : Leaf}
    {rows columns outerIV innerIV row : Nat} {captures : Env Value} {before after : Log Value}
    (run : NestedBody meaning ops leaf rows columns outerIV innerIV captures row before after) :
    ∀ event ∈ before, event ∈ after := by
  obtain ⟨_env, _trace, _prefixEval, innerRun⟩ := run
  exact SCF_monotone (fun _ _ _ leafRun => LeafBody.monotone leafRun) innerRun

/-- Exact possible complete logs, including nested pure choices and EVERY store.
The same lexical captures/partial operation meaning is used on both sides. -/
theorem segmentation_exact_logs {Value : Type} (meaning : PureMeaning Value) (program : Initializer)
    (tile : Nat) (positive : 0 < tile) (captures : Env Value) (writes : Log Value) :
    Run meaning program tile false captures writes ↔ Run meaning program tile true captures writes := by
  unfold Run
  simp only [Bool.false_eq_true, ↓reduceIte]
  apply exists_congr
  intro trace
  exact GraphLoopsSegment.complete_trace_iff (Body meaning program captures)
    (program.shape.headD 0) 1 tile 0 (by decide) positive [] writes trace

end Merlin.GraphLoopsMemorySCF
