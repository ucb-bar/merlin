/- Conservativity of the inactive runtime interpretation on unchanged operation fragments.
The executable gate reads actual operation fields, including nested regions. -/
import MerlinVerify.Rules.FrontendMaskRuntime
import MerlinVerify.Check.Defined

namespace MerlinVerify.Rules.FrontendRuntimeBridge
open MerlinVerify MerlinVerify.Check

/-- The candidate delegates this operation to the frozen interpreter, on every input. -/
def delegates (op : Op) : Bool :=
  decide (op.name ≠ "scf.if" ∧ op.name ≠ "scf.for" ∧
    (op.name ≠ "tensor.empty" ∨ Sem.Ops.Tensor.resultTensorTy op.blind ≠ none))

theorem delegates_sound {op : Op} (h : delegates op = true) (limit : Nat)
    (ctx : Ctx) (call : CallSem) (args : List RVal) (regions : List (Ctx → RegionSem)) :
    FrontendMaskRuntime.sem limit ctx call op args regions = Sem.Ops.opSem ctx call op args regions := by
  simp only [delegates, decide_eq_true_eq] at h
  unfold FrontendMaskRuntime.sem
  split
  · split
    · split
      · next hn => exact (h.1 (Op.blind_name op ▸ hn)).elim
      · next hn => exact (h.2.1 (Op.blind_name op ▸ hn)).elim
      · rfl
    · split
      · next hn =>
        rcases h.2.2 with he | he
        · exact (he (Op.blind_name op ▸ hn.1)).elim
        · exact (he hn.2).elim
      · rfl
    · rfl
  · rfl

mutual
  def ops : List Op → Bool
    | [] => true
    | op@(.mk _ _ _ _ _ rs _) :: tail => delegates op && regions rs && ops tail
  def regions : List Region → Bool
    | [] => true
    | .mk blocks :: tail => blocksCheck blocks && regions tail
  def blocksCheck : List Block → Bool
    | [] => true
    | .mk _ body :: tail => ops body && blocksCheck tail
end

mutual
  theorem ops_sound (limit : Nat) (ctx : Ctx) (call : CallSem) (env : Env) :
      ∀ body, ops body = true →
      evalOps (FrontendMaskRuntime.sem limit) ctx call env body =
        evalOps Sem.Ops.opSem ctx call env body
    | [], _ => rfl
    | .mk name results operands attrs props rs successors :: tail, h => by
      simp only [ops, Bool.and_eq_true] at h
      rw [evalOps.eq_2, evalOps.eq_2]
      split
      · rfl
      · apply congrArg (Outcome.bind (env.getAll operands))
        funext args
        have hr := regions_sound limit call env rs h.1.2
        rw [hr, delegates_sound h.1.1]
        apply congrArg (Outcome.bind (Sem.Ops.opSem ctx call
          (.mk name results operands attrs props rs successors) args
          (evalRegions Sem.Ops.opSem call env rs)))
        funext values
        apply congrArg (Outcome.bind (env.bindAll results values))
        funext next
        exact ops_sound limit ctx call next tail h.2

  theorem regions_sound (limit : Nat) (call : CallSem) (env : Env) :
      ∀ rs, regions rs = true →
      evalRegions (FrontendMaskRuntime.sem limit) call env rs =
        evalRegions Sem.Ops.opSem call env rs
    | [], _ => rfl
    | .mk blocks :: tail, h => by
      simp only [regions, Bool.and_eq_true] at h
      simp only [evalRegions]
      congr 1
      · funext ctx args
        exact blocks_sound limit ctx call env blocks args h.1
      · exact regions_sound limit call env tail h.2

  theorem blocks_sound (limit : Nat) (ctx : Ctx) (call : CallSem) (env : Env) :
      ∀ blocks args, blocksCheck blocks = true →
      evalBlocks (FrontendMaskRuntime.sem limit) ctx call env blocks args =
        evalBlocks Sem.Ops.opSem ctx call env blocks args
    | [], _, _ => rfl
    | [.mk params body], args, h => by
      simp only [blocksCheck, Bool.and_eq_true] at h
      simp only [evalBlocks, evalBlock]
      apply congrArg (Outcome.bind (env.bindAll params args))
      funext next
      exact ops_sound limit ctx call next body h.1
    | _ :: _ :: _, _, _ => rfl
end

/-- Lift an existing definedness book only when the actual operation and all nested regions
retain the frozen interpretation. Dynamic/SCF cases must supply independent proofs. -/
def liftBook (limit ib : Nat) (old : Definedness.Book Sem.Ops.opSem ib) :
    Definedness.Book (FrontendMaskRuntime.sem limit) ib where
  rule Γ op fs := if delegates op && regions op.regions then old.rule Γ op fs else none
  sound Γ op fs rs h ctx call env args hib hΓ ha := by
    split at h
    · next hc =>
      simp only [Bool.and_eq_true] at hc
      rw [regions_sound limit call env op.regions hc.2, delegates_sound hc.1]
      exact old.sound Γ op fs rs h ctx call env args hib hΓ ha
    · cases h

#print axioms ops_sound
#print axioms liftBook
end MerlinVerify.Rules.FrontendRuntimeBridge
