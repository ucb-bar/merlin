/- Additive captured-source adapter. Eligibility is recomputed from actual operation syntax
and the declared SSA types, independently of proposed lowering output and provenance hints.
This review candidate does not replace frozen quant_ext dispatch or approve its specification. -/
import MerlinVerify.Rules.W8A8DequantWidth
import MerlinVerify.Rules.W8A8SourceQuantize

namespace MerlinVerify.Rules.W8A8SourceContract

open MerlinVerify MerlinVerify.Sem.Ops

inductive Kind where
  | quantize | dequantize
  deriving DecidableEq, Repr

structure Config where
  kind : Kind
  shape : List Nat
  zeroWidth : Nat

def tensor (shape : List Nat) (elt : Ty) : Ty := .tensor (shape.map some) elt none

def operandTypes (c : Config) : List Ty :=
  [tensor c.shape (if c.kind = .quantize then .float .f32 else .int 8),
   tensor [] (.float .f32), tensor [] (.int c.zeroWidth)]

def resultType (c : Config) : Ty :=
  tensor c.shape (if c.kind = .quantize then .int 8 else .float .f32)

def name (kind : Kind) : String :=
  if kind = .quantize then "quant_ext.quantize_per_tensor" else "quant_ext.dequantize_per_tensor"

/-- Unknown semantic attributes and duplicate keys are refused. Only the already reviewed
annotation contract can declare an attribute irrelevant; provenance does not choose a recipe. -/
def metadataOK (kind : Kind) (op : Op) : Bool :=
  let attributes := (op.props ++ op.attrs).filter fun entry => !Spec.Annotations.semanticsFree entry.1
  decide ((attributes.map Prod.fst).Nodup) &&
  attributes.all (fun entry => ["quant_min", "quant_max",
    if kind = .quantize then "output_dtype" else "input_dtype"].contains entry.1) &&
  if kind = .quantize then
    op.attr? "quant_min" == some (.int (-128) (.int 64)) &&
    op.attr? "quant_max" == some (.int 127 (.int 64)) &&
    op.attr? "output_dtype" == some (.str "int8")
  else
    (match op.attr? "quant_min", op.attr? "quant_max" with
    | none, none => true
    | some (.int lo (.int 64)), some (.int hi (.int 64)) => (lo == -128 || lo == -127) && hi == 127
    | _, _ => false) &&
    (match op.attr? "input_dtype" with
    | none | some (.str "i8") | some (.str "int8") => true
    | _ => false)

/-- Static source contract, stated before its recognizer. Scalar scale/zero-point shapes and
the exact signed storage types are part of eligibility, even when parameter values are zero. -/
def Eligible (op : Op) (types : List Ty) (c : Config) : Prop :=
  op.name = name c.kind ∧ types = operandTypes c ∧ op.results.map (·.ty) = [resultType c] ∧
    op.operands.length = 3 ∧ op.regions = [] ∧ op.successors = [] ∧
    (c.zeroWidth = 32 ∨ c.zeroWidth = 64) ∧ metadataOK c.kind op = true

instance (op : Op) (types : List Ty) (c : Config) : Decidable (Eligible op types c) := by
  unfold Eligible
  infer_instance

def config? (op : Op) (types : List Ty) : Option Config := do
  let [result] := op.results | none
  let some (shape, _) := result.ty.staticTensor? | none
  let [_, _, .tensor [] (.int width) none] := types | none
  let kind ← if op.name == name .quantize then some Kind.quantize
    else if op.name == name .dequantize then some Kind.dequantize else none
  let c : Config := ⟨kind, shape, width⟩
  if Eligible op types c then some c else none

theorem config?_sound {op : Op} {types : List Ty} {c : Config}
    (accepted : config? op types = some c) : Eligible op types c := by
  unfold config? at accepted
  repeat' first
    | simp only [bind, Option.bind] at accepted
    | split at accepted
    | contradiction
  all_goals cases accepted
  all_goals assumption

/-- Type lookup uses the actual SSA environment, newest binding first. Unknown values refuse
rather than accepting an untrusted certificate's operand-type assertion. -/
def inspect (types : List (ValueId × Ty)) (op : Op) : Option Config := do
  config? op (← op.operands.mapM (fun id => types.lookup id))

theorem inspect_sound {types : List (ValueId × Ty)} {op : Op} {c : Config}
    (accepted : inspect types op = some c) :
    ∃ operands, op.operands.mapM (fun id => types.lookup id) = some operands ∧ Eligible op operands c := by
  unfold inspect at accepted
  cases found : op.operands.mapM (fun id => types.lookup id) with
  | none => simp [found] at accepted
  | some operands =>
    exact ⟨operands, rfl, config?_sound (by simpa [found] using accepted)⟩

def run (c : Config) (M : FloatModel) (args : List RVal) : Outcome (List RVal) :=
  match c.kind with
  | .quantize => W8A8SourceQuantize.run c.zeroWidth M c.shape args
  | .dequantize => W8A8DequantWidth.run c.zeroWidth ⟨false, 0, c.shape⟩ M args

/-- Runtime entry for the actual recognized source operation. Keeping the adapter uninstalled
prevents its candidate contracts from being mistaken for reviewed frozen semantics. -/
def eval (types : List (ValueId × Ty)) (op : Op) (M : FloatModel) (args : List RVal) : Outcome (List RVal) :=
  match inspect types op with
  | some c => run c M args
  | none => .stuck "source quantization contract: unsupported operation, metadata or SSA types"

theorem eval_recognized {types : List (ValueId × Ty)} {op : Op} {c : Config}
    (accepted : inspect types op = some c) (M : FloatModel) (args : List RVal) :
    eval types op M args = run c M args := by
  simp [eval, accepted]

#print axioms config?_sound
#print axioms inspect_sound
#print axioms eval_recognized

end MerlinVerify.Rules.W8A8SourceContract
