/- Exact operand ordering for unflagged commutative integer operations. Sorting identifiers is
only an untrusted alignment choice; the semantic equality holds for every runtime environment. -/
import MerlinVerify.Rules.Commutative

namespace MerlinVerify.Rules.IntegerOrder

open MerlinVerify MerlinVerify.Check MerlinVerify.Sem.Ops

def eligible (op : Op) : Prop :=
  dialectOf op.name = "arith" ∧ op.name ∈ Commutative.names ∧
  unmodelledAttr? op.blind = none ∧ overflowFlags op = []

instance (op : Op) : Decidable (eligible op) := by unfold eligible; infer_instance

/-- Missing SSA operands have the same failure reason in either order. Bound operands use the
existing universal integer commutativity theorem, including poison and malformed runtime values. -/
theorem swap_step {op : Op} {x y : ValueId} (ho : op.operands = [x, y]) (he : eligible op)
    (ctx : Ctx) (call : CallSem) (env : Env) :
    stepOp opSem ctx call env (Commutative.swap op) = stepOp opSem ctx call env op := by
  obtain ⟨hd, hn, hm, hf⟩ := he
  change ((env.getAll op.operands.reverse).bind fun args =>
    (opSem ctx call (Commutative.swap op) args (evalRegions opSem call env op.regions)).bind
      fun results => env.bindAll op.results results) = _
  cases hx : env.get? x with
  | none =>
    cases hy : env.get? y <;>
      simp [stepOp, ho, Env.getAll, Outcome.mapM, hx, hy, Outcome.ofOption]
  | some a =>
    cases hy : env.get? y with
    | none =>
      simp [stepOp, ho, Env.getAll, Outcome.mapM, hx, hy, Outcome.ofOption]
    | some b =>
      have hxy : env.getAll [x, y] = .ok [a, b] := by
        simp [Env.getAll, Outcome.mapM, hx, hy, Outcome.ofOption]
      have hyx : env.getAll [y, x] = .ok [b, a] := by
        simp [Env.getAll, Outcome.mapM, hx, hy, Outcome.ofOption]
      rw [ho]
      simp only [stepOp, ho, List.reverse_cons, List.reverse_nil, List.nil_append,
        List.singleton_append, hxy, hyx, Outcome.bind_ok]
      rw [opSem_blind (Commutative.swap_blind op), Commutative.meaning hd hn hm hf ctx call b a]

def operation (op : Op) : Op :=
  match op.operands with
  | [x, y] => if eligible op ∧ y < x then Commutative.swap op else op
  | _ => op

theorem operation_spec (op : Op) :
    (operation op).isTerminator = op.isTerminator ∧
    (operation op).results = op.results ∧
    ∀ ctx call env, stepOp opSem ctx call env (operation op) = stepOp opSem ctx call env op := by
  unfold operation
  split
  · next x y ho =>
    split
    · next he => exact ⟨rfl, rfl, fun ctx call env => swap_step ho he.1 ctx call env⟩
    · exact ⟨rfl, rfl, fun _ _ _ => rfl⟩
  · exact ⟨rfl, rfl, fun _ _ _ => rfl⟩

def operations (ops : List Op) : List Op := ops.map operation

/-- Ordering operands changes neither SSA results nor the execution of any continuation. Both
UB and stuck behavior are retained exactly; no typed or initialized-input premise is needed. -/
theorem operations_equal (ctx : Ctx) (call : CallSem) :
    ∀ ops env, evalOps opSem ctx call env (operations ops) = evalOps opSem ctx call env ops
  | [], _ => rfl
  | op :: ops, env => by
    by_cases hterm : op.isTerminator = true
    · have hn : operation op = op := by
        unfold operation
        split
        · next x y ho =>
          split
          · next he =>
            have hfalse : op.isTerminator = false := by
              have hn := he.1.2.1
              simp only [Commutative.names, List.mem_cons, List.not_mem_nil, or_false] at hn
              rcases hn with hn | hn | hn | hn | hn <;> simp [Op.isTerminator, terminatorNames, hn]
            rw [hfalse] at hterm; cases hterm
          · rfl
        · rfl
      simp only [operations, List.map_cons, hn, evalOps_cons_term _ _ _ _ _ _ hterm]
      cases ops <;> rfl
    · have ht : op.isTerminator = false := by simpa using hterm
      simp only [operations, List.map_cons]
      rw [evalOps_cons_step _ _ _ _ _ _ ((operation_spec op).1.trans ht),
        evalOps_cons_step _ _ _ _ _ _ ht, (operation_spec op).2.2]
      cases hs : stepOp opSem ctx call env op with
      | ub => rfl
      | stuck _ => rfl
      | ok next => exact operations_equal ctx call ops next

end MerlinVerify.Rules.IntegerOrder
