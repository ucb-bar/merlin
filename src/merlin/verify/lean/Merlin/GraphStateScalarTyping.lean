/- Static SSA typing/dominance evidence. Every operand is resolved against the
actual ordered lexical stack. Loop-local bindings do not escape except through
the declared result/yield slots. This is not an operational poison-free proof. -/
import Merlin.GraphStateScalarNormalize

namespace Merlin.GraphState.Scalar
set_option autoImplicit false

def typeRead (environment : List Ty) : List Nat → Option (List Ty)
  | [] => some []
  | index::later => do
    let head ← environment[index]?
    let tail ← typeRead environment later
    return head::tail

inductive HasReads (environment : List Ty) : List Nat → List Ty → Prop where
  | nil : HasReads environment [] []
  | cons {index : Nat} {type : Ty} {indices : List Nat} {types : List Ty}
    (head : environment[index]? = some type) (tail : HasReads environment indices types) :
    HasReads environment (index::indices) (type::types)

def argsTyped (environment : List Ty) (indices : List Nat) (types : List Ty) : Bool :=
  typeRead environment indices == some types

inductive Typed (registry : List Signature) : List Ty → List Ty → Program → Prop where
  | ret {environment results : List Ty} {values : List Nat}
    (operands : HasReads environment values results) : Typed registry environment results (.ret values)
  | pure {environment results : List Ty} {op : Prim} {arguments : List Nat} {next : Program}
    (eligible : op.valid = true) (operands : HasReads environment arguments op.inputs)
    (later : Typed registry (op.output::environment) results next) :
    Typed registry environment results (.pure op arguments next)
  | effect {environment results outputs : List Ty} {identity : Nat} {arguments : List Nat}
    {signature : Signature} {next : Program}
    (known : registry[identity]? = some signature) (same : outputs = signature.outputs)
    (operands : HasReads environment arguments signature.inputs)
    (later : Typed registry (outputs++environment) results next) :
    Typed registry environment results (.opaque identity arguments outputs next)
  | loop {environment results carries : List Ty} {lower upper stride : Nat} {initial : List Nat} {body next : Program}
    (bounds : HasReads environment [lower,upper,stride] [.index,.index,.index])
    (values : HasReads environment initial carries)
    (inside : Typed registry (.index::carries++environment) carries body)
    (later : Typed registry (carries++environment) results next) :
    Typed registry environment results (.loop lower upper stride initial carries body next)

def checkTypes (registry : List Signature) : List Ty → List Ty → Program → Bool
  | environment,results,.ret values => argsTyped environment values results
  | environment,results,.pure op arguments next =>
    op.valid && argsTyped environment arguments op.inputs && checkTypes registry (op.output::environment) results next
  | environment,results,.opaque identity arguments outputs next =>
    match registry[identity]? with
    | none => false
    | some selected => (outputs == selected.outputs) && argsTyped environment arguments selected.inputs &&
      checkTypes registry (outputs++environment) results next
  | environment,results,.loop lower upper stride initial carries body next =>
    argsTyped environment [lower,upper,stride] [.index,.index,.index] && argsTyped environment initial carries &&
      checkTypes registry (.index::carries++environment) carries body &&
      checkTypes registry (carries++environment) results next

theorem typeRead_sound {environment : List Ty} {indices : List Nat} {types : List Ty}
    (checked : typeRead environment indices = some types) : HasReads environment indices types := by
  induction indices generalizing types with
  | nil => cases checked; exact .nil
  | cons index later ih =>
    cases head : environment[index]? with
    | none => simp [typeRead,head] at checked
    | some value =>
      cases tail : typeRead environment later with
      | none => simp [typeRead,head,tail] at checked
      | some values =>
        have same : value::values = types := by simpa [typeRead,head,tail] using checked
        subst types
        exact .cons head (ih tail)

theorem argsTyped_sound {environment types : List Ty} {indices : List Nat}
    (checked : argsTyped environment indices types = true) : HasReads environment indices types :=
  typeRead_sound (by simpa only [argsTyped,beq_iff_eq] using checked)

theorem checkTypes_sound {registry : List Signature} (program : Program) :
    ∀ environment results, checkTypes registry environment results program = true → Typed registry environment results program := by
  induction program with
  | ret values =>
    intro environment results checked
    exact .ret (argsTyped_sound checked)
  | pure op arguments next ih =>
    intro environment results checked
    have parts : (op.valid = true ∧ argsTyped environment arguments op.inputs = true) ∧
        checkTypes registry (op.output::environment) results next = true := by
      simpa only [checkTypes,Bool.and_eq_true] using checked
    exact .pure parts.1.1 (argsTyped_sound parts.1.2) (ih _ _ parts.2)
  | «opaque» identity arguments outputs next ih =>
    intro environment results checked
    cases known : registry[identity]? with
    | none => simp [checkTypes,known] at checked
    | some selected =>
      have parts : ((outputs = selected.outputs) ∧ argsTyped environment arguments selected.inputs = true) ∧
          checkTypes registry (outputs++environment) results next = true := by
        simpa only [checkTypes,known,Bool.and_eq_true,beq_iff_eq] using checked
      exact .effect known parts.1.1 (argsTyped_sound parts.1.2) (ih _ _ parts.2)
  | loop lower upper stride initial carries body next ihBody ihNext =>
    intro environment results checked
    have parts : ((argsTyped environment [lower,upper,stride] [.index,.index,.index] = true ∧
        argsTyped environment initial carries = true) ∧
        checkTypes registry (.index::carries++environment) carries body = true) ∧
        checkTypes registry (carries++environment) results next = true := by
      simpa only [checkTypes,Bool.and_eq_true] using checked
    exact .loop (argsTyped_sound parts.1.1.1) (argsTyped_sound parts.1.1.2)
      (ihBody _ _ parts.1.2) (ihNext _ _ parts.2)

theorem HasReads.length {environment types : List Ty} {indices : List Nat}
    (typed : HasReads environment indices types) : indices.length = types.length := by
  induction typed with
  | nil => rfl
  | cons _ _ ih => simpa only [List.length_cons] using congrArg Nat.succ ih

#print axioms checkTypes_sound
end Merlin.GraphState.Scalar
