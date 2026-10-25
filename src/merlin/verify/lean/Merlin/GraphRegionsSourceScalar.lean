/- Explicit initialized integer scalar interpretation of the frozen Core/Sem.
No source success is assumed. Unread destination poison remains unconstrained.
No float, flags capable of poison, division or effects. Captured integer
constants are explicit outer definitions, checked by the program layer. -/
import MerlinVerify.Rules.ParallelFusionScalar
import MerlinVerify.Check.ProgramLemmas
import MerlinVerify.Sem.Defined

namespace Merlin.GraphRegionsSource
open MerlinVerify MerlinVerify.Check MerlinVerify.Sem.Ops

inductive Binary where
  | add | sub | mul | band | bor | bxor | minsi | maxsi
  deriving DecidableEq, Repr

def Binary.name : Binary → String
  | .add => "arith.addi" | .sub => "arith.subi" | .mul => "arith.muli"
  | .band => "arith.andi" | .bor => "arith.ori" | .bxor => "arith.xori"
  | .minsi => "arith.minsi" | .maxsi => "arith.maxsi"

def Binary.bits (kind : Binary) {w : Nat} (x y : BitVec w) : BitVec w :=
  match kind with
  | .add => x + y | .sub => x - y | .mul => x * y
  | .band => x &&& y | .bor => x ||| y | .bxor => x ^^^ y
  | .minsi => if x.slt y then x else y
  | .maxsi => if x.slt y then y else x

def noOverflow : List (String × Attr) := [("overflowFlags", .enum "arith" "overflow" "none")]

def Binary.attrs : Binary → List (String × Attr)
  | .add | .sub | .mul => noOverflow
  | _ => []

inductive Cast where
  | trunc | signed | unsigned
  deriving DecidableEq, Repr

def Cast.name : Cast → String
  | .trunc => "arith.trunci" | .signed => "arith.extsi" | .unsigned => "arith.extui"

inductive Code where
  | constant (width : Nat) (literal : Int)
  | binary (width : Nat) (kind : Binary)
  | cast (input output : Nat) (kind : Cast)
  deriving DecidableEq, Repr

def Code.width : Code → Nat
  | .constant w _ | .binary w _ => w
  | .cast _ w _ => w

def Code.inputs : Code → List Nat
  | .constant .. => []
  | .binary w _ => [w, w]
  | .cast w _ _ => [w]

def Code.name : Code → String
  | .constant .. => "arith.constant"
  | .binary _ kind => kind.name
  | .cast _ _ kind => kind.name

def Code.attrs : Code → List (String × Attr)
  | .constant w n => [("value", .int n (.int w))]
  | .binary _ kind => kind.attrs
  | .cast _ _ .trunc => noOverflow
  | .cast _ _ _ => []

def Code.operation (code : Code) (result : Nat) (operands : List Nat) : Op :=
  .mk code.name [⟨result, .int code.width⟩] operands code.attrs [] [] []

def Code.Valid : Code → Prop
  | .constant w _ | .binary w _ => 0 < w
  | .cast src dst .trunc => 0 < dst ∧ dst < src
  | .cast src dst _ => 0 < src ∧ src < dst

instance (code : Code) : Decidable code.Valid := by
  cases code with
  | constant _ _ => unfold Code.Valid; infer_instance
  | binary _ _ => unfold Code.Valid; infer_instance
  | cast _ _ kind => cases kind <;> unfold Code.Valid <;> infer_instance

def Code.bits (code : Code) (args : List Nat) : BitVec code.width :=
  match code with
  | .constant w n => BitVec.ofInt w n
  | .binary w kind => (kind.bits (BitVec.ofNat w (args[0]?.getD 0))
      (BitVec.ofNat w (args[1]?.getD 0)))
  | .cast src dst .signed => (BitVec.ofNat src (args[0]?.getD 0)).signExtend dst
  | .cast src dst _ => (BitVec.ofNat src (args[0]?.getD 0)).setWidth dst

def Code.eval (code : Code) (args : List Nat) : Nat := (code.bits args).toNat

def Code.value (code : Code) (args : List Nat) : RVal :=
  .scalar (.some (.bits code.width (code.bits args)))

def scalar (w n : Nat) : RVal := .scalar (.some (.bits w (BitVec.ofNat w n)))

@[simp] theorem scalar_of_bits {w : Nat} (x : BitVec w) :
    scalar w x.toNat = .scalar (.some (.bits w x)) := by
  have h : BitVec.ofNat w x.toNat = x := by
    apply BitVec.eq_of_toNat_eq
    exact Nat.mod_eq_of_lt x.isLt
  unfold scalar
  rw [h]

def Code.arguments (code : Code) (args : List Nat) : List RVal :=
  (code.inputs.zip args).map fun (w, n) => scalar w n

theorem Code.dialect (code : Code) : dialectOf code.name = "arith" := by
  cases code with
  | constant _ _ => cbv
  | binary _ kind => cases kind <;> cbv
  | cast _ _ kind => cases kind <;> cbv

theorem Code.modelled (code : Code) (result : Nat) (operands : List Nat) :
    unmodelledAttr? (code.operation result operands).blind = none := by
  cases code with
  | constant _ _ => rfl
  | binary _ kind => cases kind <;> rfl
  | cast _ _ kind => cases kind <;> rfl

theorem Code.flags (code : Code) (result : Nat) (operands : List Nat) :
    overflowFlags (code.operation result operands).blind = [] := by
  have noneFlags : ((flagList "none").filter (· ≠ "none")) = [] := by cbv
  cases code with
  | constant _ _ => rfl
  | binary _ kind => cases kind <;> first | rfl | exact noneFlags
  | cast _ _ kind => cases kind <;> first | rfl | exact noneFlags

theorem code_arith (code : Code) (result : Nat) (operands : List Nat)
    (ctx : Ctx) (call : CallSem) (args : List RVal) :
    opSem ctx call (code.operation result operands) args [] =
      Arith.sem ctx call (code.operation result operands).blind args [] := by
  rw [opSem_modelled (code.modelled result operands)]
  have hd : dialectOf (code.operation result operands).blind.name = "arith" :=
    by change dialectOf code.name = "arith"; exact code.dialect
  simp only [dispatch, hd]
  rfl

/-- Exact frozen opSem execution, initialized result and correct width. The
total Nat evaluator is only a wrapper; this theorem supplies its semantic link. -/
theorem primitive_execution (code : Code) (valid : code.Valid) (args : List Nat)
    (arity : args.length = code.inputs.length) (result : Nat) (operands : List Nat)
    (ctx : Ctx) (call : CallSem) :
    opSem ctx call (code.operation result operands) (code.arguments args) [] =
      .ok [code.value args] := by
  rw [code_arith]
  have hf := code.flags result operands
  cases code with
  | constant w n =>
    have : args = [] := by simpa [Code.inputs] using arity
    subst args
    simp [Code.operation, Code.name, Code.attrs, Code.arguments, Code.inputs, Code.width,
      Code.value, Code.bits, scalar, Op.blind,
      Arith.sem, Op.name, Op.results, Op.attrs, Op.props, Op.successors, Op.attr?, scalarOfAttr]
  | binary w kind =>
    obtain ⟨a, b, rfl⟩ := List.length_eq_two.mp arity
    cases kind <;>
      simp only [Code.operation, Code.name, Binary.name, Code.attrs, Binary.attrs,
        noOverflow, Op.blind, Op.results, Op.name, Op.attrs, Op.props, Op.successors,
        List.map_cons, List.map_nil, Code.width] at hf
    all_goals
      simp [hf, Code.operation, Code.name, Code.attrs, Code.arguments, Code.inputs, Code.width,
        Code.value, Code.bits, scalar, Binary.name, Binary.attrs, Binary.bits, noOverflow,
        Op.blind, Arith.sem, Arith.binOp,
        Arith.pureN, Arith.resultElt, liftEager, Arith.intBin, Arith.arithWithFlags, Arith.bv,
        Op.name, Op.results, Op.attrs, Op.props, Op.successors] <;> rfl
  | cast src dst kind =>
    obtain ⟨a, rfl⟩ := List.length_eq_one_iff.mp arity
    cases kind <;>
      simp_all [Code.Valid, Code.operation, Code.name, Code.attrs, Code.arguments, Code.inputs,
        Code.width, Code.value, Code.bits, scalar, Cast.name, noOverflow, Op.blind,
        Arith.sem, Arith.unOp, Arith.pureN, Arith.widthOf, Arith.resultElt,
        liftEager, Arith.intUn, Arith.bv, Op.name, Op.results, Op.attrs, Op.props,
        Op.successors, Op.attr?, hasUnit, Ty.bitWidth?, Outcome.ofOption, Outcome.bind]

theorem scalar_typed (ib w n : Nat) : Check.RVal.HasTy ib (scalar w n) (.scalar (.int w)) := by
  simp [scalar, Check.RVal.HasTy, Check.Val.HasTy, Scalar.conforms]

theorem scalar_initialized (w n : Nat) : (scalar w n).Defined := by
  simp [scalar, RVal.Defined]

theorem Code.value_nat (code : Code) (args : List Nat) :
    code.value args = scalar code.width (code.eval args) :=
  (scalar_of_bits (code.bits args)).symm

theorem primitive_execution_nat (code : Code) (valid : code.Valid) (args : List Nat)
    (arity : args.length = code.inputs.length) (result : Nat) (operands : List Nat)
    (ctx : Ctx) (call : CallSem) :
    opSem ctx call (code.operation result operands) (code.arguments args) [] =
      .ok [scalar code.width (code.eval args)] := by
  rw [primitive_execution code valid args arity result operands ctx call, code.value_nat]

/-- A captured constant names an actual outer SSA definition. Its complete
width/literal record must be checked against that definition before use. -/
structure CapturedConstant where
  id : Nat
  width : Nat
  literal : Int
  deriving DecidableEq, Repr

def CapturedConstant.value (constant : CapturedConstant) : Nat :=
  (BitVec.ofInt constant.width constant.literal).toNat

def CapturedConstant.operation (constant : CapturedConstant) : Op :=
  (Code.constant constant.width constant.literal).operation constant.id []

def constantTypes (constants : List CapturedConstant) : List (Nat × Nat) :=
  constants.map fun constant => (constant.id, constant.width)

def constantState (constants : List CapturedConstant) (id : Nat) : Nat :=
  ((constants.map fun constant => (constant.id, constant.value)).lookup id).getD 0

end Merlin.GraphRegionsSource
