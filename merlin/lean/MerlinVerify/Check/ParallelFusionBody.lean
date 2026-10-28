/- Actual scalar-body execution for the initial bounded parallel-fusion family. Each primitive
reads its single tensor-input scalar twice; old destination values and captures are not inferred. -/
import MerlinVerify.Rules.ParallelFusionScalar
import MerlinVerify.Check.C1

namespace MerlinVerify.Check.ParallelFusionBody

open MerlinVerify MerlinVerify.Sem.Ops
open MerlinVerify.Rules.ParallelFusionScalar

def withOperands : Op → List ValueId → Op
  | .mk n results _ attrs props regions successors, operands =>
    .mk n results operands attrs props regions successors

theorem withOperands_of {op : Op} {operands : List ValueId} (h : op.operands = operands) :
    withOperands op operands = op := by cases op; cases h; rfl

@[simp] theorem withOperands_blind (op : Op) (operands : List ValueId) :
    (withOperands op operands).blind = op.blind := by cases op; rfl

@[simp] theorem withOperands_results (op : Op) (operands : List ValueId) :
    (withOperands op operands).results = op.results := by cases op; rfl

@[simp] theorem withOperands_operands (op : Op) (operands : List ValueId) :
    (withOperands op operands).operands = operands := by cases op; rfl

@[simp] theorem withOperands_regions (op : Op) (operands : List ValueId) :
    (withOperands op operands).regions = op.regions := by cases op; rfl

@[simp] theorem withOperands_name (op : Op) (operands : List ValueId) :
    (withOperands op operands).name = op.name := by cases op; rfl

@[simp] theorem withOperands_terminator (op : Op) (operands : List ValueId) :
    (withOperands op operands).isTerminator = op.isTerminator := by cases op; rfl

/-- A checked concrete primitive reads exactly its repeated actual input binding. Header and
result checks still refer to the original operation, including its attributes and width. -/
theorem repeated_step {ib w : Nat} {op : Op} {r : Value} {x : ValueId} {env : Env} {value : Val}
    (kind : Kind) (hd : dialectOf op.name = "arith") (hn : op.name = kind.name)
    (hm : unmodelledAttr? op.blind = none) (hr : op.results = [r]) (ht : r.ty = .int w)
    (hf : overflowFlags op = []) (hregs : op.regions = [])
    (hv : Val.HasTy ib (.int w) value) (hx : env.get? x = some (.scalar value))
    (ctx : Ctx) (call : CallSem) :
    stepOp opSem ctx call env (withOperands op [x, x]) =
      .ok ((r.id, .scalar (kind.value w value value)) :: env) := by
  simp only [stepOp, withOperands_operands, withOperands_regions, withOperands_results,
    hregs, evalRegions]
  have hargs : env.getAll [x, x] = .ok [.scalar value, .scalar value] := by
    simp [Env.getAll, Outcome.mapM, Outcome.ofOption, hx]
  rw [hargs]
  simp only [Outcome.bind_ok]
  rw [opSem_blind (withOperands_blind _ _), meaning kind hd hn hm hr ht hf hv hv ctx call]
  simp [Env.bindAll, hr, Outcome.bind_ok]

theorem terminator (kind : Kind) {op : Op} (hn : op.name = kind.name) :
    op.isTerminator = false := by
  cases kind <;> simp [Op.isTerminator, terminatorNames, hn, Kind.name]

/-- The one-primitive source body is total even when its input or unused old output is poison.
The closure's original environment remains present throughout actual block binding. -/
theorem repeated_body {ib w : Nat} {op term : Op} {input old r : Value} {value previous : Val}
    (kind : Kind) (hd : dialectOf op.name = "arith") (hn : op.name = kind.name)
    (hm : unmodelledAttr? op.blind = none) (hr : op.results = [r]) (ht : r.ty = .int w)
    (ho : op.operands = [input.id, input.id]) (hf : overflowFlags op = []) (hregs : op.regions = [])
    (htn : term.name = "linalg.yield") (hto : term.operands = [r.id])
    (hv : Val.HasTy ib (.int w) value) (ctx : Ctx) (call : CallSem) (env : Env) :
    evalRegion opSem ctx call env (.mk [.mk [input, old] [op, term]])
      [.scalar value, .scalar previous] = .ok [.scalar (kind.value w value value)] := by
  have heq := withOperands_of ho
  have hs := repeated_step kind hd hn hm hr ht hf hregs hv
    (env := (input.id, .scalar value) :: (old.id, .scalar previous) :: env)
    (x := input.id) (by simp [Env.get?, List.lookup]) ctx call
  rw [heq] at hs
  have hterm : term.isTerminator = true := by simp [Op.isTerminator, terminatorNames, htn]
  simp only [evalRegion, evalBlocks, evalBlock, Env.bindAll, List.length_cons,
    List.length_nil, ↓reduceIte, List.map_cons, List.map_nil, List.zip_cons_cons,
    List.zip_nil_right, List.cons_append, List.nil_append, Outcome.bind_ok]
  rw [evalOps_cons_step _ _ _ _ _ _ (terminator kind hn), hs]
  simp only [Outcome.bind_ok, evalOps_cons_term _ _ _ _ _ _ hterm, List.isEmpty_nil,
    ↓reduceIte, hto]
  simp [Env.getAll, Outcome.mapM, Outcome.ofOption, Env.get?, List.lookup]

/-- Data extracted from an actual scalar operation. Validity is recomputed, not supplied by a
certificate; separating these fields keeps later operand-selector extensions on the same kernel. -/
structure Primitive where
  kind : Kind
  operation : Op
  result : Value

def Primitive.Valid (p : Primitive) (w : Nat) : Prop :=
  dialectOf p.operation.name = "arith" ∧ p.operation.name = p.kind.name ∧
  unmodelledAttr? p.operation.blind = none ∧ p.operation.results = [p.result] ∧
  p.result.ty = .int w ∧ overflowFlags p.operation = [] ∧ p.operation.regions = []

theorem Primitive.step {ib w : Nat} {p : Primitive} (hp : p.Valid w)
    {x : ValueId} {env : Env} {value : Val} (hv : Val.HasTy ib (.int w) value)
    (hx : env.get? x = some (.scalar value)) (ctx : Ctx) (call : CallSem) :
    stepOp opSem ctx call env (withOperands p.operation [x, x]) =
      .ok ((p.result.id, .scalar (p.kind.value w value value)) :: env) := by
  obtain ⟨hd, hn, hm, hr, ht, hf, hregs⟩ := hp
  exact repeated_step p.kind hd hn hm hr ht hf hregs hv hx ctx call

/-- Actual scalar inlining evaluates the producer exactly once, then reuses that value for
both consumer operands. This is structural composition, not an integer algebra rewrite. -/
theorem fused_body {ib w : Nat} {p c : Primitive} {input old : Value} {term : Op}
    {value previous : Val} (hp : p.Valid w) (hc : c.Valid w)
    (ho : p.operation.operands = [input.id, input.id])
    (htn : term.name = "linalg.yield") (hto : term.operands = [c.result.id])
    (hv : Val.HasTy ib (.int w) value) (ctx : Ctx) (call : CallSem) (env : Env) :
    evalRegion opSem ctx call env
      (.mk [.mk [input, old]
        [p.operation, withOperands c.operation [p.result.id, p.result.id], term]])
      [.scalar value, .scalar previous] =
      .ok [.scalar (c.kind.value w (p.kind.value w value value) (p.kind.value w value value))] := by
  let scope := (input.id, .scalar value) :: (old.id, .scalar previous) :: env
  have hP := p.step hp hv (x := input.id) (env := scope)
    (by simp [scope, Env.get?, List.lookup]) ctx call
  have heq := withOperands_of ho
  rw [heq] at hP
  have hC := c.step hc (value_typed ib p.kind w value value)
    (x := p.result.id) (env := (p.result.id, .scalar (p.kind.value w value value)) :: scope)
    (by simp [Env.get?, List.lookup]) ctx call
  have htP := terminator p.kind hp.2.1
  have htC : (withOperands c.operation [p.result.id, p.result.id]).isTerminator = false := by
    simpa using terminator c.kind hc.2.1
  have htTerm : term.isTerminator = true := by simp [Op.isTerminator, terminatorNames, htn]
  simp only [evalRegion, evalBlocks, evalBlock, Env.bindAll, List.length_cons,
    List.length_nil, ↓reduceIte, List.map_cons, List.map_nil, List.zip_cons_cons,
    List.zip_nil_right, List.cons_append, List.nil_append, Outcome.bind_ok]
  change evalOps opSem ctx call scope _ = _
  rw [evalOps_cons_step _ _ _ _ _ _ htP, hP]
  simp only [Outcome.bind_ok]
  rw [evalOps_cons_step _ _ _ _ _ _ htC, hC]
  simp only [Outcome.bind_ok, evalOps_cons_term _ _ _ _ _ _ htTerm, List.isEmpty_nil,
    ↓reduceIte, hto]
  simp [Env.getAll, Outcome.mapM, Outcome.ofOption, Env.get?, List.lookup]

end MerlinVerify.Check.ParallelFusionBody
