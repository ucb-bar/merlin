/- Function inventory and body bindings for scalar-square program composition.
This supplements, and does not weaken, complete boundary and site validation. -/
import MerlinVerify.Rules.ScalarSquareComposition

namespace MerlinVerify.Check.ScalarSquareProgram

open MerlinVerify MerlinVerify.Check.ScalarSquare MerlinVerify.Rules

def body (source : Region) : Region := .mk (blocks [] [] 0 source.blocks).blocks

structure FunctionPair (source target : Func) : Prop where
  name : source.name = target.name
  declaration : source.isDecl = target.isDecl
  visibility : source.isPrivate = target.isPrivate
  region : target.body = body source.body

def functionCheck (source target : Func) : Bool :=
  source.name == target.name && source.isDecl == target.isDecl &&
    source.isPrivate == target.isPrivate && ExactSyntax.region (body source.body) target.body

def functionsCheck : List Func → List Func → Bool
  | [], [] => true
  | source :: ss, target :: ts => functionCheck source target && functionsCheck ss ts
  | _, _ => false

def check (source target : Module) : Bool :=
  ScalarSquare.check source target && functionsCheck source.funcs target.funcs

theorem function_sound {source target : Func} (checked : functionCheck source target = true) :
    FunctionPair source target := by
  simp only [functionCheck, Bool.and_eq_true, beq_iff_eq] at checked
  exact ⟨checked.1.1.1, checked.1.1.2, checked.1.2,
    (ExactSyntax.region_sound _ _ checked.2).symm⟩

theorem functions_sound {source target : List Func} (checked : functionsCheck source target = true) :
    List.Forall₂ FunctionPair source target := by
  induction source generalizing target with
  | nil => cases target <;> simp_all [functionsCheck]
  | cons source rest ih =>
    cases target with
    | nil => simp [functionsCheck] at checked
    | cons target tail =>
      simp only [functionsCheck, Bool.and_eq_true] at checked
      exact .cons (function_sound checked.1) (ih checked.2)

def LookupPair : Option Func → Option Func → Prop
  | none, none => True
  | some source, some target => FunctionPair source target
  | _, _ => False

theorem find_pair {source target : List Func} (pairs : List.Forall₂ FunctionPair source target)
    (name : String) : LookupPair (source.find? (·.name == name)) (target.find? (·.name == name)) := by
  induction pairs with
  | nil => trivial
  | @cons s t ss ts head tail ih =>
    simp only [List.find?_cons, ← head.name]
    by_cases hit : s.name == name
    · simpa [hit, LookupPair] using head
    · simpa [hit] using ih

theorem check_sound {source target : Module} (checked : check source target = true) :
    ScalarSquare.check source target = true ∧ source.funcs.length = target.funcs.length ∧
      ∀ name, LookupPair (source.func? name) (target.func? name) := by
  simp only [check, Bool.and_eq_true] at checked
  have pairs := functions_sound checked.2
  exact ⟨checked.1, List.Forall₂.length_eq pairs, fun name => find_pair pairs name⟩

#print axioms check_sound

end MerlinVerify.Check.ScalarSquareProgram
