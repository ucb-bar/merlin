/- Exact deletion uses only producer facts proved from execution of the original prefix.
Unknown operations reset those facts. Function arguments provide names for freshness, never
invented type or initialization facts; every removed operation must be proved total. -/
import MerlinVerify.Check.DefinedLinalg
import MerlinVerify.Check.TypedDead

namespace MerlinVerify.Check.InferredDead
open MerlinVerify Sem.Ops Definedness

def infer {ib : Nat} (book : Book opSem ib) (facts : FEnv) (op : Op) : Option (List Fact) := do
  let inputs ← op.operands.mapM (fun id => facts.lookup id)
  let results ← book.rule facts op inputs
  if results.length = op.results.length then some results else none

theorem infer_spec {ib : Nat} (book : Book opSem ib) {facts : FEnv} {op : Op} {results : List Fact}
    (selected : infer book facts op = some results) : ∃ inputs,
    op.operands.mapM (fun id => facts.lookup id) = some inputs ∧
    book.rule facts op inputs = some results ∧ results.length = op.results.length := by
  unfold infer at selected
  cases inputsFound : op.operands.mapM (fun id => facts.lookup id) with
  | none => simp [inputsFound] at selected
  | some inputs =>
    simp only [inputsFound, bind, Option.bind] at selected
    cases inferred : book.rule facts op inputs with
    | none => simp [inferred] at selected
    | some produced =>
      simp only [inferred] at selected
      split at selected
      · next arity => cases selected; exact ⟨inputs, rfl, inferred, arity⟩
      · cases selected

theorem infer_step {ib : Nat} (book : Book opSem ib) {facts : FEnv} {op : Op} {results : List Fact}
    (selected : infer book facts op = some results) {ctx : Ctx} {call : CallSem} {env : Env}
    (width : ctx.indexBits = ib) (valid : EnvHolds ib facts env) :
    ∃ values, stepOp opSem ctx call env op = .ok ((op.results.map (·.id)).zip values ++ env) ∧
      FactsHold ib values results ∧ EnvHolds ib ((op.results.map (·.id)).zip results ++ facts)
        ((op.results.map (·.id)).zip values ++ env) := by
  obtain ⟨inputs, inputsFound, inferred, arity⟩ := infer_spec book selected
  obtain ⟨args, foundArgs, holdsArgs⟩ := valid.getAll _ _ inputsFound
  obtain ⟨values, executed, holds⟩ := book.sound _ _ _ _ inferred ctx call env args width valid holdsArgs
  have count : op.results.length = values.length := by rw [← arity, List.Forall₂.length_eq holds]
  refine ⟨values, ?_, holds, valid.extend _ _ _ holds⟩
  simp only [stepOp, foundArgs, Outcome.bind_ok, executed]
  exact bindAll_ok count

def drop (names : TEnv) (op : Op) (tail : List Op) : Bool :=
  decide (op.results.length = 1) && TypedDead.unused names op tail

theorem drop_equal {ib : Nat} (book : Book opSem ib) {facts : FEnv} {op : Op} {results : List Fact}
    {tail : List Op} {names : TEnv} (selected : infer book facts op = some results)
    (unused : drop names op tail = true) (notTerm : op.isTerminator = false)
    {ctx : Ctx} {call : CallSem} {env : Env} (width : ctx.indexBits = ib) (valid : EnvHolds ib facts env) :
    evalOps opSem ctx call env tail = evalOps opSem ctx call env (op :: tail) := by
  simp only [drop, Bool.and_eq_true, decide_eq_true_eq] at unused
  obtain ⟨result, single⟩ := List.length_eq_one_iff.1 unused.1
  obtain ⟨values, executed, holds, _⟩ := infer_step book selected width valid
  obtain ⟨_, _, _, arity⟩ := infer_spec book selected
  have count : values.length = 1 := by rw [List.Forall₂.length_eq holds, arity, single]; rfl
  obtain ⟨value, rfl⟩ := List.length_eq_one_iff.1 count
  have total : ∃ r value, op.results = [r] ∧ stepOp opSem ctx call env op = .ok ((r.id, value) :: env) :=
    ⟨result, value, single, by simpa [single] using executed⟩
  obtain ⟨next, step, same⟩ := TypedDead.drop_sound unused.2 total
  rw [evalOps_cons_step _ _ _ _ _ _ notTerm, step, Outcome.bind_ok, same]

def normalize {ib : Nat} (book : Book opSem ib) (names : TEnv) (facts : FEnv) : List Op → List Op
  | [] => []
  | op :: rest =>
    if op.isTerminator then op :: rest else
      let moreNames := op.results.map (fun result => (result.id, VTy.scalar Ty.index)) ++ names
      match infer book facts op with
      | none => op :: normalize book moreNames [] rest
      | some results =>
        let tail := normalize book moreNames ((op.results.map (·.id)).zip results ++ facts) rest
        if drop names op tail then tail else op :: tail

theorem normalize_equal {ib : Nat} (book : Book opSem ib) (names : TEnv) (facts : FEnv)
    (ctx : Ctx) (call : CallSem) (env : Env) (ops : List Op) (width : ctx.indexBits = ib)
    (valid : EnvHolds ib facts env) :
    evalOps opSem ctx call env (normalize book names facts ops) = evalOps opSem ctx call env ops := by
  induction ops generalizing names facts env with
  | nil => rfl
  | cons op rest ih =>
    unfold normalize
    split
    · rfl
    · next notTerm =>
      have ht : op.isTerminator = false := by simpa using notTerm
      cases selected : infer book facts op with
      | none =>
        rw [evalOps_cons_step _ _ _ _ _ _ ht, evalOps_cons_step _ _ _ _ _ _ ht]
        cases stepOp opSem ctx call env op with
        | ub => rfl
        | stuck _ => rfl
        | ok next => exact ih _ [] next (by intro _ _ impossible; cases impossible)
      | some results =>
        let tail := normalize book
          (op.results.map (fun result => (result.id, VTy.scalar Ty.index)) ++ names)
          ((op.results.map (·.id)).zip results ++ facts) rest
        have finish : evalOps opSem ctx call env (if drop names op tail then tail else op :: tail) =
            evalOps opSem ctx call env (op :: tail) := by
          split
          · next unused => exact drop_equal book selected unused ht width valid
          · rfl
        rw [finish, evalOps_cons_step _ _ _ _ _ _ ht, evalOps_cons_step _ _ _ _ _ _ ht]
        obtain ⟨values, executed, _, nextValid⟩ := infer_step book selected width valid
        rw [executed, Outcome.bind_ok]
        exact ih _ _ _ nextValid

#print axioms infer_step
#print axioms drop_equal
#print axioms normalize_equal
end MerlinVerify.Check.InferredDead
