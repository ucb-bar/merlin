/- Named transpose preserves numerical facts through its actual OpDSL maps and
actual scalar body. This interprets a source operation, not an LLVM pass. -/
import MerlinVerify.Check.PolicyW8A8RotaryCast
import MerlinVerify.Check.DefinedContextGeneric

namespace MerlinVerify.Check.PolicyW8A8RotaryTranspose

open MerlinVerify Rules Definedness Sem.Ops Sem.Ops.Linalg PolicyW8A8RotaryFacts

local notation "sourceSem" => W8A8SourceSemantics.opSem
local notation "allowed" => PolicyW8A8NumericalContext.allowed

variable {ib : Nat}

def inputMap (rank : Nat) (perm : List Nat) : AffineMap :=
  LinalgNamed.mapOf rank ((List.range rank).map fun p => (perm.findIdx? (· = p)).getD 0)

def rule (bodies : Contextual.Regions model sourceSem ib allowed) (Γ : Refined.FEnv Info)
    (op : Op) (facts : List (Refined.Fact Info)) : Option (List (Refined.Fact Info)) :=
  if unmodelledAttr? op.blind = none ∧ dialectOf op.name = "linalg" ∧ op.name = "linalg.transpose" then
    match facts, LinalgNamed.natArray? op "permutation", op.regions with
    | [input, ⟨⟨.tensor shape elt, _initialized⟩, _⟩], some perm, [body] =>
      let rank := shape.length
      if ¬ (perm.length ≠ rank ∨ (List.range rank).any (fun p => !(perm.contains p))) ∧
          op.results.length = 1 ∧ structuredRanges? [inputMap rank perm, identityMap rank] rank
            [input.base.ty, .tensor shape elt] = some shape then
        match bodies.infer Γ body [Refined.elementFact input, Refined.emptyFact model ⟨.scalar elt, false⟩] with
        | some [⟨⟨.scalar yielded, true⟩, info⟩] =>
          if yielded = elt then some [⟨⟨.tensor shape elt, true⟩, info⟩] else none
        | _ => none
      else none
    | _, _, _ => none
  else none

theorem rule_sound (bodies : Contextual.Regions model sourceSem ib allowed) : ∀ Γ op facts outputs,
    rule bodies Γ op facts = some outputs → ∀ ctx call env args,
    ctx.indexBits = ib → allowed ctx call → Refined.EnvHolds model ib Γ env →
    Refined.FactsHold model ib args facts → ∃ values,
      sourceSem ctx call op args (evalRegions sourceSem call env op.regions) = .ok values ∧
      Refined.FactsHold model ib values outputs := by
  intro Γ op facts outputs checked ctx call env args width permitted valid holds
  unfold rule at checked
  split at checked
  · next header =>
    obtain ⟨modelled, dialect, named⟩ := header
    split at checked
    · next input shape elt initialized oldInfo perm body permutation regions =>
      dsimp only at checked
      split at checked
      · next conditions =>
        obtain ⟨permutationValid, arity, geometry⟩ := conditions
        split at checked
        · next yielded info inferred =>
          split at checked
          · next same =>
            subst yielded
            cases checked
            obtain _ | ⟨inputHolds, tail⟩ := holds
            obtain _ | ⟨outputHolds, tail⟩ := tail
            cases tail
            rename_i inputValue outputValue
            cases outputValue with
            | scalar _ => cases outputHolds.1.1
            | tensor tensor =>
              have total : Contextual.BodyTotal model ib allowed call
                  (fun c => evalRegion sourceSem c call env body)
                  (([input] ++ [Refined.emptyFact model ⟨.tensor shape elt, false⟩]).map Refined.elementFact)
                  [⟨⟨.scalar elt, true⟩, info⟩] := by
                intro c sameWidth permitted arguments holds
                obtain ⟨values, executed, holds, _⟩ := bodies.sound Γ body _ _ inferred c call env arguments
                  sameWidth permitted valid holds
                exact ⟨values, executed, holds⟩
              obtain ⟨values, executed, valuesHold⟩ := Contextual.overwriteSem_refined elementwise
                PolicyW8A8NumericalContext.loopStable width permitted
                (maps := [inputMap shape.length perm, identityMap shape.length])
                (iters := LinalgNamed.parallel shape.length)
                (by simpa only [LinalgNamed.parallel, List.length_replicate, List.map_cons, List.map_nil,
                  List.cons_append, List.nil_append] using structuredRanges?_spec geometry)
                (by rfl) (.cons inputHolds .nil) outputHolds.1.1 total
              have ordinary : W8A8SourceSemantics.isSource op.name = false := by
                simp [W8A8SourceSemantics.isSource, named, W8A8SourceContract.name]
              refine ⟨values, ?_, valuesHold⟩
              rw [W8A8SourceSemantics.unchanged ordinary, named_sem_of modelled dialect (by simp [named])]
              have count : values.length = op.blind.results.length := by
                have count := List.Forall₂.length_eq valuesHold
                simpa only [List.length_cons, List.length_nil, Op.blind_results, List.length_map, arity] using count
              simp only [inputMap] at executed
              simp only [regions, evalRegions, LinalgNamed.sem, Op.blind_name, named, natArray?_blind,
                permutation, shapeOf_typed outputHolds.1.1, VTy.shape, permutationValid, ↓reduceIte,
                LinalgNamed.run, executed, Outcome.bind_ok, count]
          · cases checked
        · cases checked
      · cases checked
    · cases checked
  · cases checked

def book (bodies : Contextual.Regions model sourceSem ib allowed) : Contextual.Book model sourceSem ib allowed where
  rule := rule bodies
  sound := rule_sound bodies

#print axioms rule_sound

end MerlinVerify.Check.PolicyW8A8RotaryTranspose
