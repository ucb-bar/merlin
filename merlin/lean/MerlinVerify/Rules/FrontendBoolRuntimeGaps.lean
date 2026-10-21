/- Explicit frozen-model refusals relevant to full SmolVLA execution.
These theorems do not change the model or assert that an inventoried operation is reached. -/
import MerlinVerify.Rules.FrontendBoolSimulation

namespace MerlinVerify.Rules.FrontendBoolRuntimeGaps
open MerlinVerify MerlinVerify.Check

inductive Gap where
  | dynamicEmpty | scfFor | scfIf
  deriving DecidableEq, Repr

def Gap.reason : Gap → String
  | .dynamicEmpty => "tensor.empty with dynamic sizes"
  | .scfFor => "no semantics for dialect of scf.for"
  | .scfIf => "no semantics for dialect of scf.if"

/-- Classify only explicit refusals proved below. Other model limitations are not inferred. -/
def classify (op : Op) : Option Gap :=
  if Sem.Ops.unmodelledAttr? op.blind = none then
    if op.name = "tensor.empty" ∧ Sem.Ops.Tensor.resultTensorTy op.blind = none then
      some .dynamicEmpty
    else if op.name = "scf.for" then some .scfFor
    else if op.name = "scf.if" then some .scfIf
    else none
  else none

/-- If this actual operation is stepped, the frozen semantics is stuck for every context,
operand list and region interpretation. This is not merely a conservative typing refusal. -/
theorem classify_stuck {op : Op} {gap : Gap} (h : classify op = some gap)
    (ctx : Ctx) (call : CallSem) (args : List RVal) (regions : List (Ctx → RegionSem)) :
    Sem.Ops.opSem ctx call op args regions = .stuck gap.reason := by
  unfold classify at h
  split at h
  · next hm =>
    rw [opSem_modelled hm]
    split at h
    · next hc =>
      cases h
      obtain ⟨hn, ht⟩ := hc
      have hd : Sem.Ops.dialectOf op.name = "tensor" := by rw [hn]; cbv
      simp only [Sem.Ops.dispatch, Op.blind_name, hd]
      simp only [Sem.Ops.Tensor.sem, Op.blind_name, hn, ht]
      rfl
    · split at h
      · next hn =>
        cases h
        have hd : Sem.Ops.dialectOf op.name = "scf" := by rw [hn]; cbv
        simp only [Sem.Ops.dispatch, Op.blind_name, hd]
        simp [hn, Gap.reason]
        decide
      · split at h
        · next hn =>
          cases h
          have hd : Sem.Ops.dialectOf op.name = "scf" := by rw [hn]; cbv
          simp only [Sem.Ops.dispatch, Op.blind_name, hd]
          simp [hn, Gap.reason]
          decide
        · cases h
  · cases h

theorem classify_not_supported {op : Op} {gap : Gap} (h : classify op = some gap)
    (ctx : Ctx) (call : CallSem) (args : List RVal) (regions : List (Ctx → RegionSem)) :
    ¬ (Sem.Ops.opSem ctx call op args regions).NotStuck := by
  rw [classify_stuck h ctx call args regions]
  simp [Outcome.NotStuck]

mutual
  def collectOps : List Op → List (Gap × Op)
    | [] => []
    | op@(.mk _ _ _ _ _ regions _) :: rest =>
      (match classify op with | some gap => [(gap, op)] | none => []) ++
        collectRegions regions ++ collectOps rest
  def collectRegions : List Region → List (Gap × Op)
    | [] => []
    | .mk blocks :: rest => collectBlocks blocks ++ collectRegions rest
  def collectBlocks : List Block → List (Gap × Op)
    | [] => []
    | .mk _ ops :: rest => collectOps ops ++ collectBlocks rest
end

mutual
  theorem collectOps_sound : ∀ ops pair, pair ∈ collectOps ops →
      classify pair.2 = some pair.1
    | [], _, h => by simp [collectOps] at h
    | .mk name results operands attrs props regions successors :: rest, pair, h => by
      let op : Op := .mk name results operands attrs props regions successors
      simp only [collectOps, List.mem_append] at h
      change ((pair ∈ (match classify op with | some gap => [(gap, op)] | none => [])) ∨
        pair ∈ collectRegions regions) ∨ pair ∈ collectOps rest at h
      rcases h with (h | h) | h
      · cases hc : classify op with
        | none => simp [hc] at h
        | some gap => simp only [hc, List.mem_singleton] at h; subst pair; exact hc
      · exact collectRegions_sound regions pair h
      · exact collectOps_sound rest pair h
  theorem collectRegions_sound : ∀ regions pair, pair ∈ collectRegions regions →
      classify pair.2 = some pair.1
    | [], _, h => by simp [collectRegions] at h
    | .mk blocks :: rest, pair, h => by
      simp only [collectRegions, List.mem_append] at h
      exact h.elim (collectBlocks_sound blocks pair) (collectRegions_sound rest pair)
  theorem collectBlocks_sound : ∀ blocks pair, pair ∈ collectBlocks blocks →
      classify pair.2 = some pair.1
    | [], _, h => by simp [collectBlocks] at h
    | .mk _ ops :: rest, pair, h => by
      simp only [collectBlocks, List.mem_append] at h
      exact h.elim (collectOps_sound ops pair) (collectBlocks_sound rest pair)
end

/-- Every reported witness is justified by the actual frozen evaluator. No claim of source
reachability, complete model support, or completeness of this finite gap vocabulary is made. -/
theorem collected_stuck {ops : List Op} {pair : Gap × Op} (h : pair ∈ collectOps ops)
    (ctx : Ctx) (call : CallSem) (args : List RVal) (regions : List (Ctx → RegionSem)) :
    Sem.Ops.opSem ctx call pair.2 args regions = .stuck pair.1.reason :=
  classify_stuck (collectOps_sound ops pair h) ctx call args regions

/-- Scan only unconditional block steps, stopping at a terminator. A reported gap in a
conditional child alone is insufficient to conclude that the whole function cannot succeed. -/
def refusesSuccess : List Op → Bool
  | [] => false
  | op :: rest => !op.isTerminator && ((classify op).isSome || refusesSuccess rest)

theorem bind_ne_ok {α β : Type} (o : Outcome α) (k : α → Outcome β) (result : β)
    (h : ∀ x, k x ≠ .ok result) : o.bind k ≠ .ok result := by
  cases o with
  | ub => simp [Outcome.bind]
  | stuck reason => simp [Outcome.bind]
  | ok x => exact h x

/-- No execution through an unconditional refused step can yield a successful result.
Earlier UB is permitted; no assertion of reachability or non-UB execution is smuggled in. -/
theorem refusesSuccess_sound : ∀ ops, refusesSuccess ops = true →
    ∀ ctx call env results, evalOps Sem.Ops.opSem ctx call env ops ≠ .ok results
  | [], h, _, _, _, _ => by simp [refusesSuccess] at h
  | .mk name res operands attrs props regions successors :: rest, h, ctx, call, env, results => by
    let op : Op := .mk name res operands attrs props regions successors
    change refusesSuccess (op :: rest) = true at h
    change evalOps Sem.Ops.opSem ctx call env (op :: rest) ≠ .ok results
    simp only [refusesSuccess, Bool.and_eq_true] at h
    have hterm : op.isTerminator = false := by simpa using h.1
    rw [evalOps_cons_step _ _ _ _ _ _ hterm]
    simp only [stepOp, Outcome.bind_assoc]
    apply bind_ne_ok
    intro args
    cases hg : classify op with
    | some gap =>
      rw [classify_stuck hg ctx call args (evalRegions Sem.Ops.opSem call env op.regions)]
      simp [Outcome.bind]
    | none =>
      have ht : refusesSuccess rest = true := by simpa [hg] using h.2
      apply bind_ne_ok
      intro values
      apply bind_ne_ok
      intro next
      exact refusesSuccess_sound rest ht ctx call next results

/-- The actual symbol lookup and body determine the claim. Declarations and multiblock bodies
are not classified by this finite checker. -/
def neverSucceeds (m : Module) (name : String) : Bool :=
  match m.func? name with
  | none => false
  | some f => if f.isDecl then false else
      match f.body with
      | .mk [.mk _ ops] => refusesSuccess ops
      | _ => false

theorem neverSucceeds_sound {m : Module} {name : String} (h : neverSucceeds m name = true)
    (ctx : Ctx) (args results : List RVal) :
    denote Sem.Ops.opSem ctx m name args ≠ .ok results := by
  unfold neverSucceeds at h
  split at h
  · cases h
  · next f hf =>
    split at h
    · cases h
    · next hd =>
      split at h
      · next formals ops hb =>
        simp only [denote, evalFunc, hf, ite_eq_right hd, evalRegion, hb, evalBlocks, evalBlock]
        apply bind_ne_ok
        intro env
        exact refusesSuccess_sound ops h ctx _ env results
      · cases h

#print axioms neverSucceeds_sound

#print axioms collected_stuck
end MerlinVerify.Rules.FrontendBoolRuntimeGaps
