/- Complete counted-call expansion under explicit total sequential semantics.
Extends the ordered-segment and resource-geometry boundary. Actual Kernel-to-model
projection remains separately bound and trusted Python; no C or hardware claim. -/
import Merlin.StateEffects
import Merlin.ResourceGeometry

set_option autoImplicit false

namespace Merlin.GraphLoops

inductive Expr where
  | constant (value : Int)
  | variable (name : String)
  | add (left right : Expr)
  | scaled (value : Expr) (factor : Int)
  | select (name : String) (equals : Int) (yes no : Expr)
  deriving DecidableEq, Repr

def Fits (bits : Nat) (value : Int) : Prop :=
  -(2 ^ (bits - 1) : Int) ≤ value ∧ value < (2 ^ (bits - 1) : Int)

instance (bits : Nat) (value : Int) : Decidable (Fits bits value) :=
  inferInstanceAs (Decidable (_ ∧ _))

/-- Check each executed arithmetic intermediate, including signed scale factors. -/
def evaluate (bits : Nat) (loopVar : String) (iteration : Nat) : Expr → Option Int
  | .constant value => if Fits bits value then some value else none
  | .variable name => if name = loopVar ∧ Fits bits iteration then some iteration else none
  | .add left right => do
    let a ← evaluate bits loopVar iteration left
    let b ← evaluate bits loopVar iteration right
    if Fits bits (a + b) then some (a + b) else none
  | .scaled value factor => do
    let a ← evaluate bits loopVar iteration value
    if Fits bits factor ∧ Fits bits (a * factor) then some (a * factor) else none
  | .select name equals yes no =>
    if name = loopVar ∧ Fits bits equals then
      if (iteration : Int) = equals then evaluate bits loopVar iteration yes
      else evaluate bits loopVar iteration no
    else none

inductive Operand where
  | integer (expression : Expr)
  | pointer (space : String) (offset : Expr)
  | null
  deriving DecidableEq, Repr

inductive ClosedOperand where
  | integer (value : Int)
  | pointer (space : String) (offset : Int)
  | null
  deriving DecidableEq, Repr

structure Call where
  instruction : String
  arguments : List (String × Operand)
  deriving DecidableEq, Repr

structure ClosedCall where
  instruction : String
  arguments : List (String × ClosedOperand)
  deriving DecidableEq, Repr

structure Counted where
  loopVar : String
  extent : Nat
  bits : Nat
  body : List Call
  deriving DecidableEq, Repr

def closeOperand (loop : Counted) (iteration : Nat) : Operand → Option ClosedOperand
  | .integer expression => ClosedOperand.integer <$> evaluate loop.bits loop.loopVar iteration expression
  | .pointer space offset => do
    if space = "" then none else
      return .pointer space (← evaluate loop.bits loop.loopVar iteration offset)
  | .null => some .null

def closeCall (loop : Counted) (iteration : Nat) (call : Call) : Option ClosedCall := do
  if call.instruction = "" ∨ ¬ (call.arguments.map Prod.fst).Nodup then none else
    return ⟨call.instruction, ← call.arguments.mapM fun (name, argument) => do
      if name = "" then none else return (name, ← closeOperand loop iteration argument)⟩

def expand (loop : Counted) : Option (List (List ClosedCall)) :=
  if (loop.bits = 32 ∨ loop.bits = 64) ∧ 0 < loop.extent ∧
      Fits loop.bits loop.extent ∧ loop.body ≠ [] ∧ loop.loopVar ≠ "" then
    (List.range loop.extent).mapM (fun iteration => loop.body.mapM (closeCall loop iteration))
  else none

def check (loop : Counted) (actual : List ClosedCall) : Bool :=
  match expand loop with
  | none => false
  | some blocks => decide (blocks.flatten = actual)

/-- A certificate supplies no iteration count or guessed trace: the checker
recomputes all substitutions and the complete ordered list from the counted IR. -/
theorem check_sound {loop : Counted} {actual : List ClosedCall} (accepted : check loop actual = true) :
    ∃ blocks, expand loop = some blocks ∧ blocks.flatten = actual := by
  unfold check at accepted
  cases expanded : expand loop with
  | none => simp [expanded] at accepted
  | some blocks =>
    simp only [expanded] at accepted
    exact ⟨blocks, rfl, of_decide_eq_true accepted⟩

/-- Ordered element-by-element application, including every list position. -/
inductive Instantiates {α β : Type} (f : α → Option β) : List α → List β → Prop where
  | nil : Instantiates f [] []
  | cons {before : α} {after : β} {source : List α} {result : List β}
      (head : f before = some after) (tail : Instantiates f source result) :
      Instantiates f (before :: source) (after :: result)

theorem Instantiates.length_eq {α β : Type} {f : α → Option β} {source : List α} {result : List β}
    (correspondence : Instantiates f source result) : source.length = result.length := by
  induction correspondence with
  | nil => rfl
  | cons _ _ induction => simpa using congrArg Nat.succ induction

private theorem mapM_corresponds {α β : Type} (f : α → Option β) :
    ∀ (source : List α) (result : List β), source.mapM f = some result →
      Instantiates f source result := by
  intro source
  induction source with
  | nil => intro result checked; cases checked; exact .nil
  | cons head tail induction =>
    intro result checked
    simp only [List.mapM_cons] at checked
    cases first : f head with
    | none => simp [first] at checked
    | some value =>
      cases rest : tail.mapM f with
      | none => simp [first, rest] at checked
      | some values =>
        simp only [first, rest, bind, Option.bind] at checked
        have same := Option.some.inj checked
        subst result
        exact .cons first (induction values rest)

/-- Every index in the entire declared range has its corresponding block, in
order. Each block is the checked substitution of the complete static body. -/
theorem complete_iterations {loop : Counted} {blocks : List (List ClosedCall)}
    (expanded : expand loop = some blocks) :
    Instantiates (fun iteration => loop.body.mapM (closeCall loop iteration))
      (List.range loop.extent) blocks := by
  unfold expand at expanded
  split at expanded
  · exact mapM_corresponds _ _ _ expanded
  · cases expanded

private theorem call_count (loop : Counted) {iterations : List Nat} {blocks : List (List ClosedCall)}
    (correspondence : Instantiates (fun iteration => loop.body.mapM (closeCall loop iteration)) iterations blocks) :
    blocks.flatten.length = iterations.length * loop.body.length := by
  induction correspondence with
  | nil => simp
  | cons first rest induction =>
    have count := Instantiates.length_eq (mapM_corresponds _ _ _ first)
    simp only [List.flatten_cons, List.length_append, List.length_cons, induction, Nat.succ_mul]
    omega

theorem complete_call_count {loop : Counted} {actual : List ClosedCall}
    (accepted : check loop actual = true) : actual.length = loop.extent * loop.body.length := by
  obtain ⟨blocks, expanded, rfl⟩ := check_sound accepted
  simpa only [List.length_range] using call_count loop (complete_iterations expanded)

/-- The same ordered-segment law used by Foundation, specialized to complete
closed calls. No arithmetic reassociation or independent-effect premise is used. -/
theorem ordered_execution {Value : Type} (blocks : List (List ClosedCall))
    (run : ClosedCall → StateEffects.State Value → StateEffects.State Value) (initial : StateEffects.State Value) :
    blocks.foldl (fun state block => StateEffects.execute (block.map run) state) initial =
      StateEffects.execute (blocks.flatten.map run) initial := by
  simp only [StateEffects.execute, List.foldl_map]
  exact List.foldl_flatten.symm

/-- Accepted actual expansion preserves full state for every explicit total
deterministic call interpretation, including its observed state components. -/
theorem checked_execution {loop : Counted} {actual : List ClosedCall}
    (accepted : check loop actual = true) {Value : Type}
    (run : ClosedCall → StateEffects.State Value → StateEffects.State Value) (initial : StateEffects.State Value) :
    ∃ blocks, expand loop = some blocks ∧
      blocks.foldl (fun state block => StateEffects.execute (block.map run) state) initial =
        StateEffects.execute (actual.map run) initial := by
  obtain ⟨blocks, expanded, same⟩ := check_sound accepted
  exact ⟨blocks, expanded, by rw [← same]; exact ordered_execution blocks run initial⟩

end Merlin.GraphLoops
