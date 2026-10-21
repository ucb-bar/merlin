/- Source-level frontend repair interpretation. The user approved the documented f32 policy
and direct IEEE instance on 2026-10-21; see frontend/f32-policy-approval.json. This file does not alter frozen semantics. Eligible casts execute A2's primitive directly, without constructing comparison IR. -/
import MerlinVerify.Rules.FrontendF32Comparison
import MerlinVerify.Rules.FrontendBoolContext

namespace MerlinVerify.Rules.FrontendBoolPolicy
open MerlinVerify FrontendBoolFptosi

/-- Value/poison contract. Ill-typed runtime operands remain unsupported. The diagnostic strings
agree with the frozen scalar comparator so the simulation preserves even stuck outcomes. -/
def castValue : RVal → Outcome (List RVal)
  | .scalar .poison => .ok [.scalar (Spec.FrontendIntent.f32ToBool none)]
  | .scalar (.some (.flt .f32 x)) => .ok [.scalar (Spec.FrontendIntent.f32ToBool (some x))]
  | .scalar (.some (.flt _ _)) => .stuck "kind mismatch"
  | .scalar (.some (.bits ..)) => .stuck "cmpf on integer"
  | .tensor _ => .stuck "elementwise operands are neither all scalars nor all tensors"

/-- The source AST supplies static eligibility. Only an eligible f32-to-i1 cast changes meaning;
all surrounding operations use exactly the existing evaluator semantics. -/
def sem (types : List (ValueId × Ty)) : OpSem := fun ctx call op args regions =>
  match FrontendBoolProgram.legalSource types op with
  | none => Sem.Ops.opSem ctx call op args regions
  | some _ => match args with
    | [x] => castValue x
    | _ => .stuck "frontend f32 bool cast: expected one operand"

theorem cmpValue (ctx : Ctx) (call : CallSem) (op : Op)
    (hm : Sem.Ops.unmodelledAttr? op.blind = none)
    (hd : Sem.Ops.dialectOf op.name = "arith") (hn : op.name = "arith.cmpf")
    (hr : Sem.Ops.Arith.resultElt op.blind = some (.int 1))
    (hk : Sem.Ops.Arith.floatKey op.blind = uneKey)
    (hune : UneCorrect ctx.M) (x : RVal) :
    Sem.Ops.opSem ctx call op [x, .scalar (.some (.flt .f32 0))] [] = castValue x := by
  cases x with
  | tensor t =>
    rw [Check.opSem_modelled hm]
    simp only [Sem.Ops.dispatch, Check.Op.blind_name, hd]
    simp [Check.Op.blind_name, Sem.Ops.Arith.sem, hn,
      Sem.Ops.Arith.binOp, Sem.Ops.Arith.pureN, hr, liftEager, castValue]
  | scalar v =>
    cases v with
    | poison => exact FrontendBoolExecution.cmp_sem_checked_poison ctx call op hm hd hn hr
    | some a =>
      cases a with
      | bits w b =>
        rw [Check.opSem_modelled hm]
        simp only [Sem.Ops.dispatch, Check.Op.blind_name, hd]
        simp [Check.Op.blind_name, Sem.Ops.Arith.sem, hn,
          Sem.Ops.Arith.binOp, Sem.Ops.Arith.pureN, hr, liftEager, castValue]
      | flt k b =>
        cases k with
        | f32 => exact FrontendBoolExecution.cmp_sem_checked ctx call op hm hd hn hr hk hune b
        | f16 | bf16 | f64 =>
          rw [Check.opSem_modelled hm]
          simp only [Sem.Ops.dispatch, Check.Op.blind_name, hd]
          simp [Check.Op.blind_name, Sem.Ops.Arith.sem, hn,
            Sem.Ops.Arith.binOp, Sem.Ops.Arith.pureN, hr, liftEager, castValue]

#print axioms cmpValue
end MerlinVerify.Rules.FrontendBoolPolicy
