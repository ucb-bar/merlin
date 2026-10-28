/-
Rule: dead-op elimination (`prune_dead_pure_tensor_ops` and friends).

Site: one source op, no target ops. Instance check: the op is not a terminator, it only computes
its results (`pureOp`), and none of its results is already mapped by `ρ`. "Dead" needs no separate
analysis: the removed results get no target counterpart, so any later use fails the lockstep alpha
check in C1.

Soundness: if the source op is UB or stuck the site relation holds trivially (a UB source permits
anything; this is why removing a possibly-UB op is a refinement). Otherwise the source binds fresh
ids the correspondence never mentions, so `Corr ρ` survives unchanged.
-/
import MerlinVerify.Check.C1

namespace MerlinVerify.Rules.Dce

open MerlinVerify MerlinVerify.Check MerlinVerify.Sem.Ops

/-- Dialects whose ops (on tensors) only compute their results. -/
def pureDialects : List String := ["arith", "math", "tensor", "linalg"]

/-- A value type an effect-free op may produce: a scalar or a tensor (never a memref, whose ops
write memory). -/
def valueTy : Ty → Bool
  | .int _ | .sint _ | .uint _ | .index | .float _ | .tensor .. => true
  | _ => false

mutual
  /-- `op` only computes its results: an op of a pure dialect with at least one result, all of
  value types, whose nested ops are pure (or the terminators that yield from their regions). An op
  with no results exists only for its effect, so it is never pure. -/
  def pureOp : Op → Bool
    | .mk name results _ _ _ regions _ =>
      pureDialects.contains (dialectOf name) && !results.isEmpty && results.all (valueTy ·.ty) &&
        pureRegions regions

  def pureRegions : List Region → Bool
    | [] => true
    | (.mk bs) :: rs => pureBlocks bs && pureRegions rs

  def pureBlocks : List Block → Bool
    | [] => true
    | (.mk _ ops) :: bs => pureNested ops && pureBlocks bs

  def pureNested : List Op → Bool
    | [] => true
    | op :: ops => (op.isTerminator || pureOp op) && pureNested ops
end

/-- The two `quant_ext` dequantizations `Sem/Ops/Dequant.lean` models: region-free ops that only
compute value results (ledger A9). Other `quant_ext` ops stay ineligible. -/
def dequantOp : Op → Bool
  | .mk name results _ _ _ regions _ =>
    (name == "quant_ext.dequantize_per_tensor" || name == "quant_ext.dequantize_per_channel") &&
      regions.isEmpty && !results.isEmpty && results.all (valueTy ·.ty)

/-- An op this rule may erase: it does nothing but compute its results. -/
def erasable (op : Op) : Bool := pureOp op || dequantOp op

/-- Erase one op. The model has no memory, trace or external state, so only an op that does
nothing but compute its results (`erasable`) may be erased: a call, an assertion or a store has an
effect the model cannot see, and the refinement theorem would say nothing about it. -/
def check (ρ : Rho) : List Op → List Op → Option Rho
  | [op], [] =>
    if op.isTerminator = false ∧ erasable op = true ∧ op.results.all (fun r => ρ.find r.id == none) then
      some ρ
    else none
  | _, _ => none

theorem lookup_zip_append_not_mem {α : Type} (x : ValueId) :
    ∀ (xs : List ValueId) (vs : List α) (e : List (ValueId × α)), x ∉ xs →
      List.lookup x (xs.zip vs ++ e) = List.lookup x e
  | [], _, _, _ => by simp
  | _ :: _, [], _, _ => by simp
  | y :: ys, v :: vs, e, h => by
    have hne : x ≠ y := fun he => h (he ▸ List.mem_cons_self ..)
    simp only [List.zip_cons_cons, List.cons_append, List.lookup]
    have : (x == y) = false := by simpa using hne
    rw [this]
    exact lookup_zip_append_not_mem x ys vs e (fun hm => h (List.mem_cons_of_mem _ hm))

theorem stepOp_ok_form {sem : OpSem} {ctx : Ctx} {call : CallSem} {env eS : Env} {op : Op}
    (h : stepOp sem ctx call env op = .ok eS) :
    ∃ vals, eS = (op.results.map (·.id)).zip vals ++ env := by
  unfold stepOp at h
  cases hg : env.getAll op.operands with
  | ub => rw [hg] at h; cases h
  | stuck r => rw [hg] at h; cases h
  | ok args =>
    rw [hg] at h
    simp only [Outcome.bind] at h
    cases hs : sem ctx call op args (evalRegions sem call env op.regions) with
    | ub => rw [hs] at h; cases h
    | stuck r => rw [hs] at h; cases h
    | ok results =>
      rw [hs] at h
      simp only [Env.bindAll] at h
      split at h
      · simp only [Outcome.ok.injEq] at h; exact ⟨results, h.symm⟩
      · cases h

theorem check_sound : ∀ ρ siteS siteT ρ', check ρ siteS siteT = some ρ' →
    (∀ op ∈ siteS, op.isTerminator = false) ∧ (∀ op ∈ siteT, op.isTerminator = false) ∧
    ∀ (ctx : Ctx) (callS callT : CallSem) (envS envT : Env), Corr ρ envS envT →
      CallAgree callS callT (calleesOps siteS ++ calleesOps siteT) →
      SiteRel ρ' (execOps opSem ctx callT envT siteT) (execOps opSem ctx callS envS siteS)
  | ρ, [op], [], ρ', h => by
    simp only [check] at h
    split at h
    · next hc =>
      simp only [Option.some.injEq] at h
      subst h
      obtain ⟨hterm, -, hfresh⟩ := hc
      refine ⟨by simpa using hterm, by simp, fun ctx callS callT envS envT hcorr _ => ?_⟩
      rw [execOps_cons_step _ _ _ _ _ _ hterm]
      cases hst : stepOp opSem ctx callS envS op with
      | ub => exact Or.inl rfl
      | stuck r => exact Or.inr (Or.inl ⟨r, rfl⟩)
      | ok eS =>
        obtain ⟨vals, rfl⟩ := stepOp_ok_form hst
        refine Or.inr (Or.inr ⟨_, envT, rfl, rfl, fun x y hxy => ?_⟩)
        have hx : x ∉ op.results.map (·.id) := by
          intro hm
          obtain ⟨r, hr, rfl⟩ := List.mem_map.1 hm
          have := List.all_eq_true.1 hfresh r hr
          simp [hxy] at this
        simp only [Env.get?]
        rw [lookup_zip_append_not_mem x _ vals envS hx]
        exact hcorr x y hxy
    · simp at h
  | _, [], _, _, h => by simp [check] at h
  | _, _ :: _ :: _, _, _, h => by simp [check] at h
  | _, [_], _ :: _, _, h => by simp [check] at h

def rule : Rule := { name := "dce", check := check, sound := check_sound }

end MerlinVerify.Rules.Dce
