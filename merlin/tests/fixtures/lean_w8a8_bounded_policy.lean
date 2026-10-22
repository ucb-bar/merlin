/- Exercise the bounded-domain checker against actual source/target exports. This test
driver does not approve the input policy or the separate primitive floating model laws. -/
import MerlinVerify.Core.Json
import MerlinVerify.Check.PolicyW8A8BoundedQualified
import MerlinVerify.Check.PolicyW8A8DynamicQualified

open Lean MerlinVerify MerlinVerify.Check

private theorem full_fact {ib : Nat} (shape : List Nat) (elt : Ty) (value : Val) (info : BoundedFloatFacts.Info)
    (typed : RVal.HasTy ib (.scalar value) (.scalar elt)) (defined : value ≠ .poison)
    (holds : BoundedFloatFacts.model.holds info (.scalar value)) :
    (⟨⟨.tensor shape elt, true⟩, info⟩ : Definedness.Refined.Fact _).Holds BoundedFloatFacts.model ib
      (.tensor (Tensor.full shape elt value)) := by
  refine ⟨⟨hasTy_full typed, ?_⟩, BoundedFloatFacts.full_holds holds shape elt⟩
  intro _ point bounds
  change Idx.InBounds shape point at bounds
  simpa only [Tensor.full, Tensor.canon, bounds, ↓reduceIte] using defined

/-- A kernel-checked witness for the concrete embedding test's complete input domain.
The floating context laws are still separate premises of the production theorem. -/
example : ∃ args, PolicyW8A8BoundedDefined.Domain 64
    [.tensor [2] (.int 64), .tensor [7, 3] (.float .f32), .tensor [3, 4] (.float .f32)]
    [some ⟨0, 7⟩, none, none] args := by
  let integer : Val := .some (.bits 64 0)
  let floating : Val := .some (.flt .f32 0)
  refine ⟨[.tensor (Tensor.full [2] (.int 64) integer),
    .tensor (Tensor.full [7, 3] (.float .f32) floating), .tensor (Tensor.full [3, 4] (.float .f32) floating)],
    _, rfl, .cons ?_ (.cons ?_ (.cons ?_ .nil))⟩
  · apply full_fact [2] (.int 64) integer _ (by simp [integer, RVal.HasTy, Val.HasTy, Scalar.conforms])
      (by simp [integer])
    exact ⟨⟨trivial, trivial⟩, ⟨0, rfl, by decide, by decide⟩⟩
  · apply full_fact [7, 3] (.float .f32) floating _
      (by simp [floating, RVal.HasTy, Val.HasTy, Scalar.conforms]) (by simp [floating])
    exact ⟨⟨by change FiniteFacts.pointCheck .finite floating = true; decide, trivial⟩, trivial⟩
  · apply full_fact [3, 4] (.float .f32) floating _
      (by simp [floating, RVal.HasTy, Val.HasTy, Scalar.conforms]) (by simp [floating])
    exact ⟨⟨by change FiniteFacts.pointCheck .finite floating = true; decide, trivial⟩, trivial⟩

def decodeInterval (value : Json) : Except String (Option Definedness.IndexRanges.Interval) := do
  if value == Json.null then return none
  return some ⟨← (← value.getObjVal? "lower").getNat?, ← (← value.getObjVal? "upper").getNat?⟩

def decodeContracts (value : Json) : Except String (List (String × List (Option Definedness.IndexRanges.Interval))) := do
  (← value.getArr?).toList.mapM fun entry => do
    let fn ← (← entry.getObjVal? "function").getStr?
    let intervals ← (← (← entry.getObjVal? "intervals").getArr?).toList.mapM decodeInterval
    return (fn, intervals)

def failure (ib : Nat) (Γ : Definedness.Refined.FEnv BoundedFloatFacts.Info) : List Op → Json
  | [] => Json.null
  | op :: rest =>
    if op.isTerminator then Json.null else
    match op.operands.mapM (fun id => Γ.lookup id) with
    | none => Json.mkObj [("operation", toJson op.name), ("reason", toJson "missing_binding")]
    | some inputs => match (PolicyW8A8BoundedDefined.book ib).rule Γ op inputs with
      | none => Json.mkObj [("operation", toJson op.name), ("reason", toJson "missing_proof"),
          ("results", toJson (op.results.map (·.id)))]
      | some outputs => failure ib ((op.results.map (·.id)).zip outputs ++ Γ) rest

def functionReport (ib : Nat) (intervals : PolicyW8A8BoundedQualified.InputIntervals) (program : MerlinVerify.Module) (fn : String) : Json :=
  match TypedModule.signature? ib program fn, program.func? fn with
  | some (inputs, outputs), some function =>
    let initialized := PolicyW8A8BoundedDefined.check ib program fn inputs outputs (intervals fn)
    let old := PolicyW8A8DynamicQualified.initializedCheck ib program fn inputs outputs
    let failed := match PolicyW8A8BoundedDefined.inputFacts inputs (intervals fn), function.body with
      | some facts, .mk [.mk args ops] => failure ib ((args.map (·.id)).zip facts) ops
      | _, _ => Json.null
    Json.mkObj [("initialized", toJson initialized), ("unbounded_initialized", toJson old), ("first_refusal", failed)]
  | _, _ => Json.null

def main (args : List String) : IO UInt32 := do
  let [beforePath, afterPath, contractsPath, width, reach] := args | return 2
  if reach != "all" && reach != "captured" then return 2
  let some ib := width.toNat? | return 2
  let .ok before := Decode.module (← IO.FS.readFile beforePath) | return 2
  let .ok after := Decode.module (← IO.FS.readFile afterPath) | return 2
  let .ok contracts := (Json.parse (← IO.FS.readFile contractsPath)).bind decodeContracts | return 2
  let intervals : PolicyW8A8BoundedQualified.InputIntervals := fun fn => (contracts.lookup fn).getD []
  let config : Rules.W8A8Reference.Config := ⟨if reach == "all" then .all else .captured⟩
  let reference := Rules.W8A8Reference.expand config before
  let checked := PolicyW8A8BoundedQualified.check config ib intervals before after
  let functions := (Module.definedNames before).map fun fn => Json.mkObj [
    ("function", toJson fn), ("source", functionReport ib intervals before fn),
    ("reference", (reference.map (fun m => functionReport ib intervals m fn)).getD Json.null)]
  IO.println (Json.mkObj [("conditional_initialized_policy_refinement", toJson checked),
    ("whole_pass_verified", toJson false), ("policy_reviewed", toJson false),
    ("primitive_model_laws_verified", toJson false),
    ("input_domain", toJson "finite_f32_and_declared_integer_intervals_per_function"),
    ("index_bits", toJson ib), ("policy_scope", toJson reach), ("functions", toJson functions),
    ("theorem", toJson "MerlinVerify.Check.PolicyW8A8BoundedQualified.check_sound")]).compress
  return if checked then 0 else 1
