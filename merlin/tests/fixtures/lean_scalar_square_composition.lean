/- Concrete negative obligations: lexical facts, runtime types, approval and definedness
must not be replaced by structural correspondence. All examples are kernel checked. -/
import MerlinVerify.Rules.ScalarSquareClaims

open MerlinVerify MerlinVerify.Check MerlinVerify.Check.ScalarSquare MerlinVerify.Rules
open ScalarSquareSSA ScalarSquareComposition ScalarSquareClaims

def squareConstant : Op := .mk "arith.constant" [⟨1, .float .f32⟩] []
  [("value", .float 1073741824 (.float .f32))] [] [] []

def squareOperation : Op := .mk "math.powf" [⟨2, .float .f32⟩] [0, 1] [] [] [] []

def squareScope : Scope := bindOp (bindArgs [] [⟨0, .float .f32⟩]) squareConstant

example : (site? squareScope [] squareOperation).isSome = true := by cbv

-- A block argument with the same id shadows the old literal producer.
example : site? (bindArgs squareScope [⟨1, .float .f32⟩]) [] squareOperation = none := by cbv

-- Correct operand arity does not establish runtime typing.
example (ib : Nat) : ¬ Arguments ib [⟨0, .float .f32⟩] [.scalar (.some (.bits 32 0))] := by
  simp [Arguments, HasType, vtyOf, Ty.bitWidth?, RVal.HasTy, Val.HasTy, Scalar.conforms]

-- A false approval decision remains false even for an otherwise valid reference match.
example (d : PolicyDecision) (pending : d.approved = false) (m : Module) (F : Ctx → Prop)
    (ib : Nat) (entry : String) : ¬ Authorizes d m F ib entry :=
  unapproved_rejected d pending m F ib entry

example (ib : Nat) : ¬ ContextDomain (fun _ => False) ib := by
  rintro ⟨⟨ctx, impossible⟩, _⟩
  exact impossible

-- Equality of error outcomes is not strict refinement or successful execution.
example : ¬ Refines ResultsRefine (.stuck "unsupported") (.stuck "unsupported") := by
  simp [Refines]

example (ctx : Ctx) (args : List RVal) (call : CallSem) :
    PrimitiveTyping (fun _ _ => True) (Family ctx.M ctx.indexBits) ctx.indexBits ctx
      squareConstant args call [] :=
  primitive_regionless _ _ _ _ _ _ rfl (by rfl) call

-- A concrete direct call consumes the callee's argument-domain obligation.
def squareCall : Op := .mk "func.call" [⟨4, .float .f32⟩] [3]
  [("callee", .symbolRef "leaf" [])] [] [] []

example (D : CallDomain) (F : Ctx → Prop) (ib : Nat) (ctx : Ctx) (args : List RVal)
    (typed : D "leaf" args) (call : CallSem) (rs : List (Ctx → RegionSem)) :
    PrimitiveTyping D F ib ctx squareCall args call rs :=
  primitive_call D F ib ctx squareCall args "leaf" rfl rfl rfl typed call rs

#print axioms ScalarSquareClaims.source_equal
#print axioms ScalarSquareClaims.approved_multiplication_policy
