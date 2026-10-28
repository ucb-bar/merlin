/- C1 registration: the proposed site is an untrusted hint and every AST field is rechecked. -/
import MerlinVerify.Rules.QuantizeSite

namespace MerlinVerify.Rules.QuantizeRule

open MerlinVerify MerlinVerify.Check MerlinVerify.Sem.Ops
open QuantizeSite

theorem run_witness {c : Quantize.Config} {M : FloatModel} {args results : List RVal}
    (h : Quantize.run c M args = .ok results) :
    ∃ w data scale zero, args = [.tensor data, .tensor scale, .tensor zero] ∧
      results = [.tensor (Quantize.result w c M data scale zero)] ∧
      Quantize.validInputs w c data scale zero = true := by
  unfold Quantize.run at h
  split at h
  · next data scale zero =>
    split at h
    · next valid => cases h; exact ⟨32, data, scale, zero, rfl, rfl, valid⟩
    · split at h
      · next valid => cases h; exact ⟨64, data, scale, zero, rfl, rfl, valid⟩
      · cases h
  · cases h

theorem site_sound {ρ : Rho} {s : Site} (checked : Checks ρ s)
    (ctx : Ctx) (callS callT : CallSem) (envS envT : Env) (corr : Corr ρ envS envT) :
    SiteRel ((s.rs.id, s.rg.id) :: ρ)
      (execOps opSem ctx callT envT (targets s)) (execOps opSem ctx callS envS [s.src]) := by
  have preserved := checked
  obtain ⟨frame, headers, configured, _⟩ := checked
  obtain ⟨rs, _, _, _, _, _, _, srcRegions, _⟩ := frameCheck_sound frame
  have sourceHeader := List.all_eq_true.1 headers s.src (List.mem_cons_self ..)
  simp only [Bool.and_eq_true, Bool.not_eq_eq_eq_not, Bool.not_true, decide_eq_true_eq] at sourceHeader
  rw [execOps_cons_step _ _ _ _ _ _ sourceHeader.1.1]
  unfold stepOp
  rw [srcRegions]
  simp only [evalRegions]
  simp_rw [QuantizeSource.source_sem ctx callS _ sourceHeader.1.2 configured]
  cases fetch : envS.getAll s.src.operands with
  | ub => exact Or.inl rfl
  | stuck reason => exact Or.inr (Or.inl ⟨reason, rfl⟩)
  | ok values =>
    simp only [Outcome.bind_ok]
    cases executed : Quantize.run s.c ctx.M values with
    | ub => exact Or.inl rfl
    | stuck reason => exact Or.inr (Or.inl ⟨reason, rfl⟩)
    | ok results =>
      obtain ⟨w, data, scale, zero, rfl, rfl, valid⟩ := run_witness executed
      obtain ⟨envT', target, related⟩ := target_sound preserved ctx callT envS envT corr w data scale zero fetch valid
      simp only [rs, Env.bindAll, List.length_singleton, ↓reduceIte, List.map_cons,
        List.map_nil, List.zip_cons_cons, List.zip_nil_right, List.cons_append,
        List.nil_append, Outcome.bind_ok, execOps]
      exact Or.inr (Or.inr ⟨_, envT', rfl, target, related⟩)

def check (s : Site) (ρ : Rho) (source target : List Op) : Option Rho :=
  if ExactSyntax.operations source [s.src] && ExactSyntax.operations target (targets s) && decide (Checks ρ s)
  then some ((s.rs.id, s.rg.id) :: ρ) else none

theorem check_sound (s : Site) : ∀ ρ source target ρ', check s ρ source target = some ρ' →
    (∀ op ∈ source, op.isTerminator = false) ∧ (∀ op ∈ target, op.isTerminator = false) ∧
    ∀ ctx callS callT envS envT, Corr ρ envS envT →
      CallAgree callS callT (calleesOps source ++ calleesOps target) →
      SiteRel ρ' (execOps opSem ctx callT envT target) (execOps opSem ctx callS envS source) := by
  intro ρ source target ρ' accepted
  unfold check at accepted
  split at accepted
  · next facts =>
    simp only [Bool.and_eq_true, decide_eq_true_eq] at facts
    obtain ⟨⟨hs, ht⟩, checked⟩ := facts
    have sourceEq := ExactSyntax.operations_sound _ _ hs
    have targetEq := ExactSyntax.operations_sound _ _ ht
    subst source target
    cases accepted
    have headers := checked.2.1
    have nonterminal (op : Op) (member : op ∈ s.src :: targets s) : op.isTerminator = false := by
      have h := List.all_eq_true.1 headers op member
      simp only [Bool.and_eq_true, Bool.not_eq_eq_eq_not, Bool.not_true, decide_eq_true_eq] at h
      exact h.1.1
    refine ⟨?_, ?_, ?_⟩
    · intro op member
      have same : op = s.src := by simpa using member
      subst op
      exact nonterminal _ (List.mem_cons_self ..)
    · intro op member
      exact nonterminal _ (List.mem_cons_of_mem _ member)
    · intro ctx callS callT envS envT corr _
      exact site_sound checked ctx callS callT envS envT corr
  · cases accepted

def rule (s : Site) : Rule :=
  { name := "quantize_per_tensor_i8_f32", check := check s, sound := check_sound s }

/-- Untrusted syntax proposal. `rule` checks the actual source and target lists again, so
neither this proposal nor a candidate compiler can assert its own correspondence facts. -/
def propose (ρ : Rho) (source target : List Op) : Option Site := do
  let src :: _ := source | none
  let invEmpty :: inverse :: empty :: lower :: upper :: generic :: _ := target | none
  let c ← Quantize.config? src
  let [rs] := src.results | none
  let [r0] := invEmpty.results | none
  let [r1] := inverse.results | none
  let [r2] := empty.results | none
  let [rl] := lower.results | none
  let [rh] := upper.results | none
  let [rg] := generic.results | none
  let [inverseRegion] := inverse.regions | none
  let [region] := generic.regions | none
  let [x, scale, zeroId] ← src.operands.mapM ρ.find | none
  let maps ← attrAffineMaps? generic "indexing_maps"
  let iters ← Sem.Ops.Linalg.iterKinds? generic
  return {
    src := src, invEmpty := invEmpty, inverse := inverse, empty := empty, lower := lower,
    upper := upper, generic := generic, rs := rs, r0 := r0, r1 := r1, r2 := r2, rl := rl,
    rh := rh, rg := rg, x := x, scale := scale, zero := zeroId, c := c,
    inverseRegion := inverseRegion, region := region, maps := maps, iters := iters }

#print axioms site_sound
#print axioms check_sound

end MerlinVerify.Rules.QuantizeRule
