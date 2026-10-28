/- Provisional source-semantics integration for the W8A8 draft. Only the two per-tensor
source operations use the additive contract. Other operations retain frozen dispatch, with
nested regions interpreted by the same extension. This does not approve the numerical recipe. -/
import MerlinVerify.Check.PolicyW8A8SourcePrefix

namespace MerlinVerify.Rules.W8A8SourceSemantics

open MerlinVerify MerlinVerify.Check MerlinVerify.Sem.Ops W8A8SourceContract

def isSource (name : String) : Bool :=
  name == W8A8SourceContract.name .quantize || name == W8A8SourceContract.name .dequantize

/-- Runtime tensor declarations are reconstructed from the actual values. Scalar arguments
cannot impersonate tensor calibration parameters. Element contents are checked by `run`. -/
def argumentTypes : List RVal → Option (List Ty) :=
  List.mapM fun value => match value with
    | .tensor t => some (tensor t.shape t.elt)
    | .scalar _ => none

def opSem : OpSem := fun ctx call op args regions =>
  if isSource op.name then
    if regions.isEmpty then
      match argumentTypes args >>= config? op with
      | some c => run c ctx.M args
      | none => .stuck "draft source quantization: unsupported operation or runtime tensor types"
    else .stuck "draft source quantization: regions are unsupported"
  else Sem.Ops.opSem ctx call op args regions

theorem unchanged {op : Op} (ordinary : isSource op.name = false)
    (ctx : Ctx) (call : CallSem) (args : List RVal) (regions : List (Ctx → RegionSem)) :
    opSem ctx call op args regions = Sem.Ops.opSem ctx call op args regions := by
  simp [opSem, ordinary]

/-- The program evaluator executes exactly the existing candidate adapter when actual operand
types agree with the checked SSA declarations. No certificate supplies runtime parameter values. -/
theorem recognized {types : List (ValueId × Ty)} {op : Op} {c : Config} {args : List RVal}
    (accepted : inspect types op = some c)
    (actualTypes : argumentTypes args = op.operands.mapM (fun id => types.lookup id))
    (ctx : Ctx) (call : CallSem) :
    opSem ctx call op args [] = W8A8SourceContract.eval types op ctx.M args := by
  obtain ⟨operands, lookup, eligible⟩ := inspect_sound accepted
  have selected : isSource op.name = true := by
    rw [eligible.1]
    cases c.kind <;> simp [isSource, W8A8SourceContract.name]
  have configured : config? op operands = some c := by
    simpa [inspect, lookup] using accepted
  rw [eval_recognized accepted]
  simp [opSem, selected, actualTypes, lookup, configured]

private theorem producer_regions {op : Op}
    (producer : op.name = "arith.constant" ∨ op.name = "tensor.splat")
    (ctx : Ctx) (call : CallSem) (args : List RVal) (left right : List (Ctx → RegionSem)) :
    Sem.Ops.opSem ctx call op args left = Sem.Ops.opSem ctx call op args right := by
  unfold Sem.Ops.opSem
  split
  · rfl
  · rcases producer with named | named
    · have dialect : dialectOf op.name = "arith" := by rw [named]; cbv
      simp only [dispatch, Op.blind_name]
      rw [dialect]
      simp only [Arith.sem, Op.blind_name, named]
    · have dialect : dialectOf op.name = "tensor" := by rw [named]; cbv
      simp only [dispatch, Op.blind_name]
      rw [dialect]
      simp only [Tensor.sem, Op.blind_name, named]

/-- Constant/splat calibration proofs apply to this program semantics even when intervening
regions are interpreted through the provisional source extension. -/
theorem producers : PolicyW8A8SourceFacts.ProducerAgreement opSem := by
  intro ctx call env op producer
  have ordinary : isSource op.name = false := by
    rcases producer with named | named <;> simp [isSource, named, W8A8SourceContract.name]
  unfold stepOp
  congr 1
  funext args
  rw [unchanged ordinary]
  rw [producer_regions producer ctx call args]

#print axioms recognized
#print axioms producers

end MerlinVerify.Rules.W8A8SourceSemantics
