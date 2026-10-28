/-
G0-FROZEN dispatch. The concrete op semantics `opSem` routes each op to its family by name.
Families live in `Sem/Ops/*.lean`, one owner each, so semantics work parallelizes without editing a
shared file. Adding a family requires an interface change request.
-/
import MerlinVerify.Sem.Ops.Arith
import MerlinVerify.Sem.Ops.Linalg
import MerlinVerify.Sem.Ops.LinalgNamed
import MerlinVerify.Sem.Ops.Tensor
import MerlinVerify.Sem.Ops.Func
import MerlinVerify.Sem.Ops.Cf
import MerlinVerify.Sem.Ops.Dequant
import MerlinVerify.Sem.Ops.Quantize
import MerlinVerify.Spec.Annotations

namespace MerlinVerify.Sem.Ops

/-- The dialect prefix of an op name (`"arith"` for `"arith.addi"`). -/
def dialectOf (name : String) : String := (name.splitOn ".").headD ""

/-- The op with its SSA identity erased: name, attributes, properties and result types only.
Operands and regions reach the semantics as values and region meanings, never as syntax. -/
def _root_.MerlinVerify.Op.blind (op : Op) : Op :=
  .mk op.name (op.results.map fun r => { r with id := 0 }) [] op.attrs op.props [] op.successors

/-- Dispatch on the blinded op, so `opSem` cannot depend on SSA identifiers by construction
(`opSem_blind`): renaming values never changes an op's meaning. -/
def dispatch : OpSem := fun ctx call op args regions =>
  match dialectOf op.name with
  | "arith" | "math" => Arith.sem ctx call op args regions
  | "linalg" =>
    if op.name == "linalg.generic" || op.name == "linalg.fill" || op.name == "linalg.index" then
      Linalg.sem ctx call op args regions
    else LinalgNamed.sem ctx call op args regions
  | "tensor" => Tensor.sem ctx call op args regions
  | "func" => Func.sem ctx call op args regions
  | "cf" => Cf.sem ctx call op args regions
  | "quant_ext" =>
    if op.name == "quant_ext.quantize_per_tensor" then Quantize.sem ctx call op args regions
    else Dequant.sem ctx call op args regions
  | _ => .stuck s!"no semantics for dialect of {op.name}"

/-- The flat `callee` symbol of an op, if it has one. -/
def calleeOf (op : Op) : Option String :=
  match op.attr? "callee" with
  | some (.symbolRef root []) => some root
  | _ => none

/-- `call` restricted to the op's own callee: an op can only invoke the function it names. -/
def restrictCall (callee : Option String) (call : CallSem) : CallSem := fun name args =>
  if callee = some name then call name args else .stuck s!"op may not call @{name}"

/-- Attributes (discardable or inherent) each op's semantics reads. Anything else on an op makes it
`stuck` unless `Spec.Annotations.semanticsFree` says it cannot affect the result: an attribute the
semantics does not read may change MLIR's meaning of the op, so ignoring it would be unsound. -/
def interpretedAttrs (name : String) : List String :=
  match name with
  | "arith.constant" => ["value"]
  | "arith.cmpi" | "arith.cmpf" => ["predicate", "fastmath"]
  | "arith.divsi" | "arith.divui" | "arith.shrsi" | "arith.shrui" => ["isExact"]
  | "arith.extui" | "arith.uitofp" | "arith.index_castui" => ["nonNeg"]
  | "arith.addi" | "arith.subi" | "arith.muli" | "arith.shli" | "arith.trunci" => ["overflowFlags"]
  | "arith.addf" | "arith.subf" | "arith.mulf" | "arith.divf" | "arith.truncf" =>
    ["fastmath", "roundingmode"]
  | "linalg.generic" => ["indexing_maps", "iterator_types", "operandSegmentSizes"]
  | "linalg.matmul" | "linalg.batch_matmul" => ["indexing_maps", "operandSegmentSizes", "cast"]
  | "linalg.quantized_matmul" | "linalg.copy" | "linalg.fill" => ["operandSegmentSizes"]
  | "linalg.index" => ["dim"]
  | "linalg.transpose" => ["permutation"]
  | "linalg.broadcast" | "linalg.reduce" => ["dimensions", "operandSegmentSizes"]
  | "tensor.extract_slice" | "tensor.insert_slice" =>
    ["static_offsets", "static_sizes", "static_strides", "operandSegmentSizes"]
  | "tensor.collapse_shape" => ["reassociation"]
  | "tensor.expand_shape" => ["reassociation", "static_output_shape"]
  | "tensor.concat" => ["dim"]
  | "quant_ext.dequantize_per_tensor" => ["quant_min", "quant_max", "input_dtype"]
  | "quant_ext.dequantize_per_channel" => ["axis", "quant_min", "quant_max", "input_dtype"]
  | "quant_ext.quantize_per_tensor" => ["quant_min", "quant_max", "output_dtype"]
  | "func.call" => ["callee"]
  | "cf.assert" => ["msg"]
  | n => if n.startsWith "math." || n.startsWith "arith." then ["fastmath"] else []

/-- The first attribute of `op` that its semantics neither reads nor may ignore, if any. -/
def unmodelledAttr? (op : Op) : Option String :=
  (op.props ++ op.attrs).findSome? fun (k, _) =>
    if (interpretedAttrs op.name).contains k || Spec.Annotations.semanticsFree k then none else some k

/-- Op meaning: dispatch on the blinded op with `call` restricted to its callee, after refusing any
attribute the semantics would otherwise ignore. By construction the meaning cannot depend on SSA
identifiers (`opSem_blind`) or on functions the op does not name (`opSem_call_local`). -/
def opSem : OpSem := fun ctx call op args regions =>
  match unmodelledAttr? op.blind with
  | some k => .stuck s!"{op.blind.name}: unmodelled attribute {k}"
  | none => dispatch ctx (restrictCall (calleeOf op.blind) call) op.blind args regions

theorem opSem_blind {ctx : Ctx} {call : CallSem} {op op' : Op} {args : List RVal}
    {regions : List (Ctx → RegionSem)} (h : op.blind = op'.blind) :
    opSem ctx call op args regions = opSem ctx call op' args regions := by
  simp only [opSem, h]

theorem opSem_call_local {ctx : Ctx} {call call' : CallSem} {op : Op} {args : List RVal}
    {regions : List (Ctx → RegionSem)}
    (h : ∀ c, calleeOf op.blind = some c → call c = call' c) :
    opSem ctx call op args regions = opSem ctx call' op args regions := by
  simp only [opSem]
  split
  · rfl
  · have : restrictCall (calleeOf op.blind) call = restrictCall (calleeOf op.blind) call' := by
      funext name xs
      simp only [restrictCall]
      split
      · next hc => rw [h name hc]
      · rfl
    rw [this]

/-- The meaning of function `name` in module `m` under Merlin's op semantics. -/
def denote (ctx : Ctx) (m : Module) (name : String) (args : List RVal) : Outcome (List RVal) :=
  MerlinVerify.denote opSem ctx m name args

end MerlinVerify.Sem.Ops
