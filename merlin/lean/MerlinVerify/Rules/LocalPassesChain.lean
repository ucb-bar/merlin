/- Ordered pass-local pipeline composition. Repair matching is kept distinct from original
source equivalence, and the provisional TorchAO materialization claim is never promoted into
captured-call refinement. Every supplied suffix boundary is checked by Lean. -/
import MerlinVerify.Rules.BoolToFloatProgram
import MerlinVerify.Rules.TorchAOAffineProgram

namespace MerlinVerify.Rules.LocalPassesChain
open MerlinVerify

def suffix (first : Module) : List Module → Bool
  | [] => true
  | next :: rest => LocalProgramCorrespondence.check first next && suffix next rest

def final (first : Module) : List Module → Module
  | [] => first
  | next :: rest => final next rest

theorem suffix_sound (first : Module) : ∀ tail, suffix first tail = true →
    ∀ (ctx : Ctx) (name : String) (args : List RVal),
      denote Sem.Ops.opSem ctx first name args =
        denote Sem.Ops.opSem ctx (final first tail) name args
  | [], _, _, _, _ => rfl
  | next :: rest, h, ctx, name, args => by
    simp only [suffix, Bool.and_eq_true] at h
    exact (LocalProgramCorrespondence.check_sound h.1 ctx name args).trans
      (suffix_sound next rest h.2 ctx name args)

def boolCheck (src repaired : Module) (tail : List Module) : Bool :=
  BoolToFloatProgram.check src repaired && suffix repaired tail

/-- Composition preserves the declared f32 repair policy, including poison and failed outcomes.
It does not weaken the policy claim to ordinary equivalence with a signed-i1 conversion. -/
theorem bool_sound {src repaired : Module} {tail : List Module}
    (h : boolCheck src repaired tail = true) (base : FloatModel)
    (ctx : Ctx) (name : String) (args : List RVal) :
    BoolToFloatProgram.policyDenote src {ctx with M := BoolToFloat.model base} name args =
      denote Sem.Ops.opSem {ctx with M := BoolToFloat.model base} (final repaired tail) name args := by
  simp only [boolCheck, Bool.and_eq_true] at h
  exact (BoolToFloatProgram.check_sound_ieee h.1 base ctx name args).trans
    (suffix_sound repaired tail h.2 _ name args)

def torchaoCheck (src lowered : Module) (tail : List Module) : Bool :=
  TorchAOAffineProgram.check src lowered && suffix lowered tail

/-- A complete materialized-contract execution claim with an explicit source-derived witness.
The missing captured-call/tensor reference connection remains outside this statement. -/
theorem torchao_sound {src lowered : Module} {tail : List Module}
    (h : torchaoCheck src lowered tail = true) :
    ∃ intended, TorchAOAffineProgram.materialize? src = some intended ∧
      ∀ (ctx : Ctx) (name : String) (args : List RVal),
        denote Sem.Ops.opSem ctx intended name args =
          denote Sem.Ops.opSem ctx (final lowered tail) name args := by
  simp only [torchaoCheck, Bool.and_eq_true] at h
  cases hi : TorchAOAffineProgram.materialize? src with
  | none => simp [TorchAOAffineProgram.check, hi] at h
  | some intended =>
    refine ⟨intended, rfl, fun ctx name args => ?_⟩
    exact ((TorchAOAffineProgram.check_sound hi h.1).2 ctx name args).trans
      (suffix_sound lowered tail h.2 ctx name args)

#print axioms bool_sound
#print axioms torchao_sound
end MerlinVerify.Rules.LocalPassesChain
