/- Complete selected raw call projection. Argument names/selectors/roles are
contract data, not target-specific constants. Unsupported expressions, control,
effects or asynchronous fields are refused rather than silently discarded. -/
import Merlin.GraphRegionsDeclaredMemory

namespace Merlin.GraphRegionsDeclared

structure RawCall where
  instruction : String
  arguments : List (String × Nat)
  assumes : List String
  awaits : List String
  produces : Option String
  sets : List String
  stages : List String
  unit : Option String
  deriving DecidableEq, Repr

structure CallSchema where
  instruction : String
  role : String
  argumentOrder : List String
  selectors : List (Nat × Nat)
  readFields : List (String × String)
  outputSelector : String
  outputOffset : String
  countField : String
  byteWidthField : String
  byteOrders : List String
  shape : List Nat
  deriving DecidableEq, Repr

def decodeWindow (schema : CallSchema) (raw : RawCall) (bytes : Nat) (fields : String × String) : Option Window := do
  let selector ← raw.arguments.lookup fields.1
  let backing ← schema.selectors.lookup selector
  let offset ← raw.arguments.lookup fields.2
  pure ⟨backing, offset, bytes⟩

def decodeCall (schema : CallSchema) (raw : RawCall) : Option BoundCall := do
  if raw.instruction ≠ schema.instruction ∨ raw.arguments.map Prod.fst ≠ schema.argumentOrder ∨
      ¬schema.argumentOrder.Nodup ∨ ¬(schema.selectors.map Prod.fst).Nodup ∨
      raw.assumes ≠ [] ∨ raw.awaits ≠ [] ∨ raw.produces ≠ none ∨ raw.sets ≠ [] ∨
      raw.stages ≠ [] ∨ raw.unit ≠ none then none else do
    let count ← raw.arguments.lookup schema.countField
    let width ← raw.arguments.lookup schema.byteWidthField
    let reads ← schema.readFields.mapM (decodeWindow schema raw (count * width))
    let output ← decodeWindow schema raw (count * width) (schema.outputSelector, schema.outputOffset)
    pure ⟨raw.instruction, schema.role, raw.arguments, reads, output, schema.byteOrders,
      schema.shape, width * 8, "synchronous", [], none, []⟩

/-- A complete raw Kernel call is accepted only when its contract-driven decoder
produces exactly the submitted bound call. Selected effect truth is independent. -/
def checkRawCall (schema : CallSchema) (raw : RawCall) (bound : BoundCall) : Bool :=
  decide (decodeCall schema raw = some bound)

theorem checkRawCall_sound {schema : CallSchema} {raw : RawCall} {bound : BoundCall}
    (accepted : checkRawCall schema raw bound = true) : decodeCall schema raw = some bound :=
  of_decide_eq_true accepted

structure TensorArgument where
  name : String
  shape : List Nat
  dtype : String
  access : String
  deriving DecidableEq, Repr

structure RawKernel where
  name : String
  arguments : List TensorArgument
  attrs : List (String × String)
  calls : List RawCall
  deriving DecidableEq, Repr

def decodeCalls : List CallSchema → List RawCall → Option (List BoundCall)
  | [], [] => some []
  | schema :: schemas, call :: calls => do
    let bound ← decodeCall schema call
    let rest ← decodeCalls schemas calls
    pure (bound :: rest)
  | _, _ => none

def checkKernel (actual : RawKernel) (name : String) (arguments : List TensorArgument)
    (schemas : List CallSchema) (bound : List BoundCall) : Bool :=
  decide (actual.name = name ∧ actual.arguments = arguments ∧ actual.attrs = [] ∧
    decodeCalls schemas actual.calls = some bound)

theorem checkKernel_sound {actual : RawKernel} {name : String} {arguments : List TensorArgument}
    {schemas : List CallSchema} {bound : List BoundCall}
    (accepted : checkKernel actual name arguments schemas bound = true) :
    actual.name = name ∧ actual.arguments = arguments ∧ actual.attrs = [] ∧
      decodeCalls schemas actual.calls = some bound := of_decide_eq_true accepted

instance sequenceValidDec (elements : Nat) (addresses : Nat → Window) :
    (known : List Nat) → (items : List CallStage) → Decidable (SequenceValid elements addresses known items)
  | _, [] => isTrue trivial
  | known, item :: rest =>
    letI := sequenceValidDec elements addresses (known ++ [item.stage.output]) rest
    inferInstanceAs (Decidable (
      callValid elements item.stage addresses item.call ∧
      (∀ slot ∈ item.stage.inputs, slot ∈ known) ∧ item.stage.output ∉ known ∧
      (∀ slot ∈ known, (addresses slot).Disjoint (addresses item.stage.output)) ∧
      SequenceValid elements addresses (known ++ [item.stage.output]) rest))

def checkSequence (elements : Nat) (addresses : Nat → Window) (known : List Nat) (items : List CallStage) : Bool :=
  decide (SequenceValid elements addresses known items)

theorem checkSequence_sound {elements : Nat} {addresses : Nat → Window} {known : List Nat} {items : List CallStage}
    (accepted : checkSequence elements addresses known items = true) : SequenceValid elements addresses known items :=
  of_decide_eq_true accepted

end Merlin.GraphRegionsDeclared
