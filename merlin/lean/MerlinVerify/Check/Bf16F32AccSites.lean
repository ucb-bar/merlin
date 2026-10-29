/- Universal, initialized-input qualification of every rewritten contraction in a captured
boundary. Local programs contain the actual source operation and actual target replacement;
only SSA names and the surrounding function wrapper change. The complete-module equality check
is independent, so local success cannot excuse unrelated changes elsewhere in the capture. -/
import MerlinVerify.Check.Bf16F32Acc
import MerlinVerify.Check.Bf16F32AccExact
import MerlinVerify.Check.Bf16F32AccOperands

namespace MerlinVerify.Check.Bf16F32AccSites

open MerlinVerify Rules.Bf16F32AccReference

mutual
  def renameOps (rename : Nat → Nat) : List Op → List Op
    | [] => []
    | .mk name results operands attrs props regions successors :: rest =>
      .mk name (results.map fun value => { value with id := rename value.id }) (operands.map rename) attrs props
        (renameRegions rename regions) successors :: renameOps rename rest
  def renameRegions (rename : Nat → Nat) : List Region → List Region
    | [] => []
    | .mk blocks :: rest => .mk (renameBlocks rename blocks) :: renameRegions rename rest
  def renameBlocks (rename : Nat → Nat) : List Block → List Block
    | [] => []
    | .mk args ops :: rest =>
      .mk (args.map fun value => { value with id := rename value.id }) (renameOps rename ops) :: renameBlocks rename rest
end

def shiftOps (base : Nat) := renameOps (base + ·)
def shiftRegions (base : Nat) := renameRegions (base + ·)
def shiftBlocks (base : Nat) := renameBlocks (base + ·)

def wrapper (args : List Value) (ops : List Op) (result : Value) : Module :=
  ⟨[.mk "func.func" [] []
    [("sym_name", .str "subject"),
     ("function_type", .type (.function (args.map (·.ty)) [result.ty]))]
    [] [.mk [.mk args (ops ++ [.mk "func.return" [] [result.id] [] [] [] []])]] []], [], []⟩

/-- Copy the actual producers of the checked zero destination, rather than assuming its value.
Only the contract's two admitted initialization forms are supported. -/
def initialization? (seen : List Op) (destination : Nat) : Option (List Op) := do
  let producer ← owner? seen destination
  match producer with
  | .mk "tensor.splat" _ [literal] _ _ _ _ =>
    let zero ← owner? seen literal
    return [zero, producer]
  | .mk "linalg.fill" _ [literal, destination] _ _ _ _ =>
    let zero ← owner? seen literal
    let empty ← owner? seen destination
    if empty.name == "tensor.empty" && empty.operands.isEmpty then
      return [zero, empty, producer]
    else none
  | _ => none

structure Pair where
  function : String
  index : Nat
  source : Op
  target : List Op
  before : Module
  after : Module

def fragments? (args : List Value) (seen : List Op)
    (source : Op) (targets : List Op) : Option (Module × Module) := do
  let site ← site? args seen source
  let [_, _, destination] := source.operands | none
  let initialOps ← initialization? seen destination
  let leftTy ← type? args seen site.left
  let rightTy ← type? args seen site.right
  let [_, _, _, accumulated, _, final] := targets | none
  let [targetLeft, targetRight, _] := accumulated.operands | none
  let [result] := final.results | none
  -- Internal target definitions get fresh identifiers. Captured operands map
  -- back to their actual source identities, including values produced by the
  -- copied initialization. The alpha checker validates the entire renaming.
  let base := boundOps (source :: initialOps ++ targets) +
    args.foldl (fun n value => max n (value.id + 1)) 0
  let packedSource ← Bf16F32AccOperands.pack? ⟨site.left, leftTy⟩ ⟨site.right, rightTy⟩
  let targetArgs ← Bf16F32AccOperands.pack? ⟨targetLeft, leftTy⟩ ⟨targetRight, rightTy⟩
  if packedSource.length != targetArgs.length then none
  let sourceArgs := packedSource.filter fun value =>
    !(initialOps.any fun op => op.results.any fun result => result.id == value.id)
  let rename := fun id =>
    if id == targetLeft then site.left
    else if id == targetRight then site.right
    else base + id
  let renamed := renameOps rename targets
  if alphaRegion (targetArgs.map fun value => (value.id, rename value.id))
      (.mk [.mk [] (targets ++ [.mk "func.return" [] [result.id] [] [] [] []])])
      (.mk [.mk [] (renamed ++ [.mk "func.return" [] [rename result.id] [] [] [] []])]) then
    return ⟨wrapper sourceArgs (initialOps ++ [source]) site.result,
      wrapper sourceArgs (initialOps ++ renamed) { result with id := rename result.id }⟩
  else none


def pair? (fn : String) (index : Nat) (args : List Value) (seen : List Op)
    (source : Op) (targets : List Op) : Option Pair := do
  let (before, after) ← fragments? args seen source targets
  return ⟨fn, index, source, targets, before, after⟩

theorem pair_source {fn : String} {index : Nat} {args : List Value} {seen : List Op}
    {source : Op} {targets : List Op} {pair : Pair}
    (collected : pair? fn index args seen source targets = some pair) :
    pair.function = fn ∧ pair.index = index ∧ pair.source = source ∧ pair.target = targets := by
  unfold pair? at collected
  cases fragments : fragments? args seen source targets with
  | none => simp [fragments] at collected
  | some value =>
    obtain ⟨before, after⟩ := value
    simp only [fragments, bind, Option.bind, Option.pure_def, Option.some.injEq] at collected
    cases collected
    exact ⟨rfl, rfl, rfl, rfl⟩

def locations (index : Nat) : List Op → List (Nat × Op)
  | [] => []
  | op :: rest => (if candidate op then [(index, op)] else []) ++ locations (index + 1) rest

def collectOps (fn : String) (args : List Value) (index : Nat) (seen : List Op) :
    List Op → List Op → Option (List Pair)
  | [], [] => some []
  | [], _ :: _ => none
  | source :: rest, targets => do
    if candidate source then
      let pair ← pair? fn index args seen source (targets.take 6)
      let pairs ← collectOps fn args (index + 1) (source :: seen) rest (targets.drop 6)
      return pair :: pairs
    else
      let _ :: tail := targets | none
      collectOps fn args (index + 1) (source :: seen) rest tail


/-- No candidate may be omitted or duplicated: the reconstructed obligations have exactly the
original source operations, in their original order and at their original positions. -/
theorem collectOps_coverage {fn : String} {args : List Value} {sources targets : List Op}
    {index : Nat} {seen : List Op} {pairs : List Pair}
    (collected : collectOps fn args index seen sources targets = some pairs) :
    pairs.map (fun pair => (pair.index, pair.source)) = locations index sources := by
  induction sources generalizing targets index seen pairs with
  | nil =>
    cases targets <;> simp_all [collectOps, locations]
  | cons source rest ih =>
    by_cases selected : candidate source = true
    · simp only [collectOps, selected, ↓reduceIte] at collected
      cases first : pair? fn index args seen source (targets.take 6) with
      | none => simp [first] at collected
      | some pair =>
        cases remaining : collectOps fn args (index + 1) (source :: seen) rest (targets.drop 6) with
        | none => simp [first, remaining] at collected
        | some tail =>
          simp only [first, remaining, bind, Option.bind, Option.pure_def, Option.some.injEq] at collected
          cases collected
          have original := pair_source first
          simp only [List.map_cons, original.2.1, original.2.2.1, locations, selected, ↓reduceIte,
            List.singleton_append]
          exact congrArg (List.cons (index, source)) (ih remaining)
    · have skipped : candidate source = false := by simpa using selected
      cases targets with
      | nil => simp [collectOps, skipped] at collected
      | cons target tail =>
        simp only [collectOps, skipped] at collected
        simpa only [locations, skipped, Bool.false_eq_true, ↓reduceIte, List.nil_append] using ih collected

def collect (before after : Module) : Option (List Pair) := do
  if before.ops.length != after.ops.length then none
  let pairs ← (before.ops.zip after.ops).mapM fun (source, target) => do
    if source.name == "func.func" && !(source.asFunc?.map (·.isDecl)).getD true then
      let some fn := source.asFunc? | none
      let [.mk [.mk args ops]] := source.regions | none
      let [.mk [.mk _ targets]] := target.regions | none
      collectOps fn.name args 0 [] ops targets
    else some []
  return pairs.flatten

def pairCheck (ib : Nat) (pair : Pair) : Bool :=
  Bf16F32Acc.qualify ib pair.before pair.after

def accept (ib : Nat) (exactMatch : Bool) (pairs : Option (List Pair)) : Bool :=
  exactMatch && match pairs with
    | none => false
    | some pairs => pairs.all (pairCheck ib)

def check (ib : Nat) (before after : Module) : Bool :=
  accept ib (Bf16F32AccExact.check before after) (collect before after)

/-- Every local obligation comes from the exact bound before/after modules, and is a strict
policy refinement on all typed, initialized tensor operands. Whole-program source definedness
is deliberately absent from this pass-local theorem; it remains required for an end-to-end claim. -/
theorem check_sound {ib : Nat} {before after : Module}
    (accepted : check ib before after = true) :
    Bf16F32AccExact.check before after = true ∧
    ∃ pairs, collect before after = some pairs ∧
      ∀ pair ∈ pairs, Bf16F32Acc.Qualified ib pair.before pair.after := by
  simp only [check, accept, Bool.and_eq_true] at accepted
  refine ⟨accepted.1, ?_⟩
  cases collected : collect before after with
  | none => simp [collected] at accepted
  | some pairs =>
    refine ⟨pairs, rfl, ?_⟩
    intro pair member
    have checked := List.all_eq_true.1 (by simpa [collected] using accepted.2) pair member
    exact Bf16F32Acc.qualify_sound checked

#print axioms collectOps_coverage
#print axioms check_sound

end MerlinVerify.Check.Bf16F32AccSites
