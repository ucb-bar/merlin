/- Proved transfer rules for the straight-line definedness checker. -/
import MerlinVerify.Check.Defined
import MerlinVerify.Check.Rulebook

namespace MerlinVerify.Check.Definedness

open MerlinVerify MerlinVerify.Check MerlinVerify.Sem.Ops

abbrev Rule := Op → List Fact → Option (List Fact)

def RuleSound (ib : Nat) (r : Rule) : Prop :=
  ∀ op fs rs, unmodelledAttr? op.blind = none → r op fs = some rs → ∀ ctx call args,
    ctx.indexBits = ib → FactsHold ib args fs →
    ∃ vs, opSem ctx call op args [] = .ok vs ∧ FactsHold ib vs rs

def constantRule (ib : Nat) : Rule := fun op fs =>
  if dialectOf op.name = "arith" ∧ op.name = "arith.constant" ∧ fs = [] then
    match op.results, op.attr? "value" with
    | [r], some a =>
      match scalarOfAttr ib a with
      | some s => if s.conforms ib r.ty then some [⟨.scalar r.ty, true⟩] else none
      | none => none
    | _, _ => none
  else none

theorem constantRule_sound (ib : Nat) : RuleSound ib (constantRule ib) := by
  intro op fs rs hm h ctx call args hib hargs
  unfold constantRule at h
  split at h
  · next hc =>
    obtain ⟨hd, hn, rfl⟩ := hc
    cases hargs
    split at h
    · next r a hr ha =>
      split at h
      · next s hs =>
        split at h
        · next hty =>
          cases h
          refine ⟨[.scalar (.some s)], ?_, List.Forall₂.cons ⟨hty, by simp [RVal.Defined]⟩ .nil⟩
          rw [opSem_modelled hm]
          simp only [dispatch, Op.blind_name, hd]
          simp only [Arith.sem, Op.blind_name, hn, Op.blind_results, hr,
            List.map_cons, List.map_nil, Op.blind_attr?, ha, hib]
          cases a
          all_goals first
            | (simp [scalarOfAttr] at hs; done)
            | simp only [hs]
        · cases h
      · cases h
    · cases h
  · cases h

/-- Uninitialized storage is allowed as an intermediate, but its fact never certifies an output. -/
def emptyRule : Rule := fun op fs =>
  if dialectOf op.name = "tensor" ∧ op.name = "tensor.empty" ∧ fs = [] then
    (Sem.Ops.Tensor.resultTensorTy op).map fun (sh, e) => [⟨.tensor sh e, false⟩]
  else none

theorem emptyRule_sound (ib : Nat) : RuleSound ib emptyRule := by
  intro op fs rs hm h ctx call args _ hargs
  unfold emptyRule at h
  split at h
  · next hc =>
    obtain ⟨hd, hn, rfl⟩ := hc
    cases hargs
    cases hr : Sem.Ops.Tensor.resultTensorTy op with
    | none => simp [hr] at h
    | some p =>
      obtain ⟨sh, e⟩ := p
      simp only [hr, Option.map_some, Option.some.injEq] at h
      subst h
      refine ⟨[.tensor (MerlinVerify.Tensor.full sh e .poison)], ?_,
        List.Forall₂.cons ⟨hasTy_full (hasTy_poison _ _), by simp⟩ .nil⟩
      rw [tensor_sem_of hm hd]
      simp only [Sem.Ops.Tensor.sem, Op.blind_name, hn, resultTensorTy_blind, hr]
  · cases h

/-- `tensor.splat` fully initializes a static tensor from one initialized scalar. -/
def splatRule : Rule := fun op fs =>
  if dialectOf op.name = "tensor" ∧ op.name = "tensor.splat" then
    match fs, Sem.Ops.Tensor.resultTensorTy op with
    | [⟨.scalar t, true⟩], some (sh, e) =>
      if t = e then some [⟨.tensor sh e, true⟩] else none
    | _, _ => none
  else none

theorem splatRule_sound (ib : Nat) : RuleSound ib splatRule := by
  intro op fs rs hm h ctx call args _ hargs
  unfold splatRule at h
  split at h
  · next hc =>
    obtain ⟨hd, hn⟩ := hc
    split at h
    · next _ _ _ scalarTy shape _ hr =>
      split at h
      · next hte =>
        subst hte
        cases h
        obtain _ | ⟨hv, hnil⟩ := hargs
        cases hnil
        rename_i v
        cases v with
        | tensor t => cases hv.1
        | scalar x =>
          refine ⟨[.tensor (MerlinVerify.Tensor.full shape scalarTy x)], ?_, List.Forall₂.cons
            ⟨hasTy_full hv.1, ?_⟩ .nil⟩
          · rw [tensor_sem_of hm hd]
            simp only [Sem.Ops.Tensor.sem, Op.blind_name, hn, resultTensorTy_blind, hr]
          · intro _ i hi
            change Idx.InBounds shape i at hi
            simpa only [RVal.Defined, MerlinVerify.Tensor.full, MerlinVerify.Tensor.canon, hi,
              ↓reduceIte] using hv.2 rfl
      · cases h
    · cases h
  · cases h

/-- Integer operations that are total on initialized operands when overflow flags are absent.
Division, shifts, casts, and flagged arithmetic require separate obligations and are refused. -/
def intBinaryNames : List String :=
  ["arith.addi", "arith.subi", "arith.muli", "arith.andi", "arith.ori", "arith.xori",
   "arith.maxsi", "arith.minsi", "arith.maxui", "arith.minui"]

def intBinaryRule : Rule := fun op fs =>
  match fs, op.results with
  | [⟨.scalar (.int w), true⟩, ⟨.scalar (.int w'), true⟩], [r] =>
    if dialectOf op.name = "arith" ∧ op.name ∈ intBinaryNames ∧ w' = w ∧
        r.ty = .int w ∧ overflowFlags op = [] then some [⟨.scalar (.int w), true⟩]
    else none
  | _, _ => none

theorem initialized_int {ib w : Nat} {v : RVal}
    (h : Fact.Holds ib v ⟨.scalar (.int w), true⟩) :
    ∃ x : BitVec w, v = .scalar (.some (.bits w x)) := by
  cases v with
  | tensor t => cases h.1
  | scalar x =>
    rcases hasTy_intWidth (t := .int w) rfl h.1 with hp | ⟨a, ha⟩
    · exact (h.2 rfl hp).elim
    · exact ⟨a, congrArg RVal.scalar ha⟩

theorem intBinaryRule_sound (ib : Nat) : RuleSound ib intBinaryRule := by
  intro op fs rs hm h ctx call args _ hargs
  unfold intBinaryRule at h
  split at h
  · split at h
    · next hc =>
      obtain ⟨hd, hn, hw, hty, hflags⟩ := hc
      subst hw
      cases h
      obtain _ | ⟨ha, hb⟩ := hargs
      obtain _ | ⟨hb, hnil⟩ := hb
      cases hnil
      obtain ⟨x, rfl⟩ := initialized_int ha
      obtain ⟨y, rfl⟩ := initialized_int hb
      rename_i w r hr
      have hresult : Arith.resultElt op.blind = some (.int w) := by
        simp only [Arith.resultElt, Op.blind_results, hr, List.map_cons, List.map_nil, hty]
      rw [opSem_modelled hm]
      simp only [dispatch, Op.blind_name, hd]
      simp only [intBinaryNames, List.mem_cons, List.not_mem_nil, or_false] at hn
      rcases hn with hn | hn | hn | hn | hn | hn | hn | hn | hn | hn
      all_goals
        simp only [Arith.sem, Op.blind_name, hn, Arith.binOp, Arith.pureN, hresult,
          Outcome.bindOp_eq,
          show overflowFlags op.blind = [] from by simpa [overflowFlags, attrEnum?] using hflags]
        rw [show [.scalar (.some (.bits w x)), .scalar (.some (.bits w y))] =
          ([.some (.bits w x), .some (.bits w y)] : List Val).map RVal.scalar from rfl,
          liftEager_scalars]
        simp only [Arith.intBin, ↓reduceDIte, Arith.arithWithFlags, List.contains_nil,
          Bool.false_and, Bool.false_eq_true, ↓reduceIte, Outcome.bind_ok, Arith.bv]
        exact ⟨_, rfl, List.Forall₂.cons ⟨by simp [RVal.HasTy, Val.HasTy, Scalar.conforms],
          by simp [RVal.Defined]⟩ .nil⟩
    · cases h
  · cases h

def rules (ib : Nat) : Rule := fun op fs =>
  [constantRule ib, emptyRule, splatRule, intBinaryRule].findSome? fun r => r op fs

theorem rules_sound (ib : Nat) : RuleSound ib (rules ib) := by
  intro op fs rs hm hr
  obtain ⟨r, hmem, h⟩ := List.exists_of_findSome?_eq_some hr
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hmem
  rcases hmem with rfl | rfl | rfl | rfl
  · exact constantRule_sound ib op fs rs hm h
  · exact emptyRule_sound ib op fs rs hm h
  · exact splatRule_sound ib op fs rs hm h
  · exact intBinaryRule_sound ib op fs rs hm h

def book (ib : Nat) : Book opSem ib where
  rule op fs := if unmodelledAttr? op.blind = none then rules ib op fs else none
  sound op fs rs h := by
    split at h
    · next hm => exact rules_sound ib op fs rs hm h
    · cases h

/-- Acceptance proves successful, nonpoison results for every typed, nonpoison input. -/
theorem definedCheck_sound {ib : Nat} {m : Module} {f : String} {argTys resTys : List VTy}
    (h : check (book ib) m f argTys resTys = true) :
    FuncDefined opSem (fun ctx => ctx.indexBits = ib)
      (fun args => HasTys ib args argTys ∧ ResultsDefined args) m f := check_sound (book ib) h

end MerlinVerify.Check.Definedness
