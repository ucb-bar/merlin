/- Exact integer operand canonicalization. No floating-point commutativity is assumed. -/
import MerlinVerify.Rules.OneOp
import MerlinVerify.Check.RulesArith

namespace MerlinVerify.Rules.Commutative

open MerlinVerify MerlinVerify.Check MerlinVerify.Sem.Ops

theorem intBin_comm (f : (w : Nat) → BitVec w → BitVec w → Val)
    (hf : ∀ w x y, f w x y = f w y x) (a b : Val) :
    Arith.intBin f a b = Arith.intBin f b a := by
  cases a with
  | poison =>
    cases b with
    | poison => rfl
    | some s => cases s <;> rfl
  | some a =>
    cases b with
    | poison => cases a <;> rfl
    | some b =>
      cases a with
      | flt _ _ => cases b <;> rfl
      | bits w x =>
        cases b with
        | flt _ _ => rfl
        | bits w' y =>
          by_cases h : w' = w
          · subst h; simp only [Arith.intBin, ↓reduceDIte, hf]
          · simp only [Arith.intBin, h, Ne.symm h, ↓reduceDIte]

theorem lift_binary_comm (e : Ty) (f : Val → Val → Outcome Val)
    (hf : ∀ x y, f x y = f y x) (a b : RVal) :
    liftEager e (fun | [x, y] => f x y | _ => .stuck "arity") [a, b] =
      liftEager e (fun | [x, y] => f x y | _ => .stuck "arity") [b, a] := by
  cases a with
  | scalar x => cases b <;> simp [liftEager, hf]
  | tensor a =>
    cases b with
    | scalar _ => simp [liftEager]
    | tensor b =>
      by_cases hs : a.shape = b.shape
      · simp only [liftEager, List.mapM_cons, List.mapM_nil, bind, Option.bind, Option.pure_def]
        simp only [List.all_cons, List.all_nil, Bool.and_true, beq_iff_eq, hs, ↓reduceIte,
          List.map_cons, List.map_nil]
        have he : (fun i => f (a.get i) (b.get i)) = (fun i => f (b.get i) (a.get i)) :=
          funext fun i => hf _ _
        rw [he]
      · simp [liftEager, hs, Ne.symm hs]

theorem binOp_comm (op : Op) (f : Val → Val → Outcome Val) (hf : ∀ x y, f x y = f y x)
    (a b : RVal) : Arith.binOp op [a, b] f = Arith.binOp op [b, a] f := by
  simp only [Arith.binOp, Arith.pureN]
  cases Arith.resultElt op with
  | none => rfl
  | some e =>
    change (liftEager e _ [a, b]).bind _ = (liftEager e _ [b, a]).bind _
    exact congrArg (fun r : Outcome RVal => r.bind fun v => .ok [v]) (lift_binary_comm e f hf a b)

def names : List String := ["arith.addi", "arith.muli", "arith.andi", "arith.ori", "arith.xori"]

theorem meaning {op : Op} (hd : dialectOf op.name = "arith") (hn : op.name ∈ names)
    (hm : unmodelledAttr? op.blind = none) (hflags : overflowFlags op = [])
    (ctx : Ctx) (call : CallSem) (a b : RVal) (regions : List (Ctx → RegionSem)) :
    opSem ctx call op [a, b] regions = opSem ctx call op [b, a] regions := by
  rw [opSem_modelled hm, opSem_modelled hm]
  simp only [dispatch, Op.blind_name, hd]
  have hf : overflowFlags op.blind = [] := by simpa only [overflowFlags, attrEnum?, Op.blind_attr?] using hflags
  simp only [names, List.mem_cons, List.not_mem_nil, or_false] at hn
  rcases hn with hn | hn | hn | hn | hn
  all_goals
    simp only [Arith.sem, Op.blind_name, hn, hf]
    apply binOp_comm
    apply intBin_comm
    intro w x y
    simp [Arith.arithWithFlags, Arith.bv, BitVec.add_comm, BitVec.mul_comm,
      BitVec.and_comm, BitVec.or_comm, BitVec.xor_comm]

def swap (op : Op) : Op :=
  .mk op.name op.results op.operands.reverse op.attrs op.props op.regions op.successors

theorem swap_blind (op : Op) : (swap op).blind = op.blind := by cases op; rfl

theorem step_agrees {op : Op} {x y : ValueId} (ho : op.operands = [x, y])
    (hd : dialectOf op.name = "arith") (hn : op.name ∈ names)
    (hm : unmodelledAttr? op.blind = none) (hflags : overflowFlags op = [])
    (ctx : Ctx) (call : CallSem) (env : Env) :
    stepOp opSem ctx call env op = .ub ∨
    (∃ r, stepOp opSem ctx call env op = .stuck r) ∨
    stepOp opSem ctx call env (swap op) = stepOp opSem ctx call env op := by
  cases hx : env.get? x with
  | none =>
    exact Or.inr (Or.inl ⟨"unbound SSA value", by simp [stepOp, ho, Env.getAll, Outcome.mapM, hx, Outcome.ofOption]⟩)
  | some a =>
    cases hy : env.get? y with
    | none =>
      exact Or.inr (Or.inl ⟨"unbound SSA value", by simp [stepOp, ho, Env.getAll, Outcome.mapM, hx, hy, Outcome.ofOption]⟩)
    | some b =>
      have hxy : env.getAll [x, y] = .ok [a, b] := by
        simp [Env.getAll, Outcome.mapM, hx, hy, Outcome.ofOption]
      have hyx : env.getAll [y, x] = .ok [b, a] := by
        simp [Env.getAll, Outcome.mapM, hx, hy, Outcome.ofOption]
      refine Or.inr (Or.inr ?_)
      change ((env.getAll op.operands.reverse).bind fun args =>
        (opSem ctx call (swap op) args (evalRegions opSem call env op.regions)).bind fun results =>
        env.bindAll op.results results) = _
      rw [ho]
      simp only [stepOp, ho, List.reverse_cons, List.reverse_nil, List.nil_append, List.singleton_append,
        hxy, hyx, Outcome.bind_ok]
      rw [opSem_blind (swap_blind op), meaning hd hn hm hflags ctx call b a]

def check (ρ : Rho) : List Op → List Op → Option Rho
  | [s], [t] =>
    if dialectOf s.name = "arith" ∧ s.name ∈ names ∧ s.operands.length = 2 ∧
        s.regions = [] ∧ unmodelledAttr? s.blind = none ∧ overflowFlags s = [] ∧
        calleeOf s.blind = none then alphaOp ρ (swap s) t
    else none
  | _, _ => none

theorem check_sound : ∀ ρ ss ts ρ', check ρ ss ts = some ρ' →
    (∀ op ∈ ss, op.isTerminator = false) ∧ (∀ op ∈ ts, op.isTerminator = false) ∧
    ∀ ctx callS callT envS envT, Corr ρ envS envT →
      CallAgree callS callT (calleesOps ss ++ calleesOps ts) →
      SiteRel ρ' (execOps opSem ctx callT envT ts) (execOps opSem ctx callS envS ss)
  | ρ, [s], [t], ρ', h => by
    simp only [check] at h
    split at h
    · next hc =>
      obtain ⟨hd, hn, hlen, _, hm, hf, hcallee⟩ := hc
      have hst : s.isTerminator = false := by
        simp only [names, List.mem_cons, List.not_mem_nil, or_false] at hn
        rcases hn with hn | hn | hn | hn | hn <;> simp [Op.isTerminator, terminatorNames, hn]
      have hvt : (swap s).isTerminator = false := hst
      have htt : t.isTerminator = false := (isTerminator_of_opHeadEq (alphaOp_spec h).1) ▸ hvt
      refine ⟨by simpa using hst, by simpa using htt, fun ctx callS callT envS envT hc hcall => ?_⟩
      obtain ⟨x, y, ho⟩ : ∃ x y, s.operands = [x, y] := by
        cases he : s.operands with
        | nil => simp [he] at hlen
        | cons x rest =>
          cases rest with
          | nil => simp [he] at hlen
          | cons y rest =>
            have hr : rest = [] := by simpa [he] using hlen
            subst hr
            exact ⟨x, y, rfl⟩
      exact Expansion.site_of_step (v := swap s) hst htt rfl (by rw [swap_blind]; exact hcallee) h
        ctx callS callT envS envT hc hcall (step_agrees ho hd hn hm hf ctx callS envS)
    · cases h
  | _, [], _, _, h => by simp [check] at h
  | _, _ :: _ :: _, _, _, h => by simp [check] at h
  | _, [_], [], _, h => by simp [check] at h
  | _, [_], _ :: _ :: _, _, h => by simp [check] at h

def rule : Rule := ⟨"integer-commute", check, check_sound⟩

end MerlinVerify.Rules.Commutative
