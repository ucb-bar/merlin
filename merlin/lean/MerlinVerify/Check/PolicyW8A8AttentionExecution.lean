/- A checked complete source softmax segment executes to the exact same-row value.
Finite output bounds additionally require a proved finite member in every row. -/
import MerlinVerify.Check.PolicyW8A8AttentionTemplate
import MerlinVerify.Rules.W8A8AttentionExecution

namespace MerlinVerify.Check.PolicyW8A8AttentionExecution

open MerlinVerify Rules W8A8AttentionTemplateStages W8A8AttentionPointwiseRows W8A8AttentionRowValues

def Correct (shape : List Nat) (width input output : Nat) (actual : List Op) : Prop :=
  ∀ (cap : Nat) (ctx : Ctx) (call : CallSem) (env : Env) (fallback : FloatModel),
    ctx.M = W8A8RoundedReal.model fallback →
    ∀ (values : Idx → Val) (scores : Idx → FBits .f32),
    env.get? input = some (.tensor ⟨shape ++ [width], .float .f32, values⟩) →
    (∀ point, Idx.InBounds (shape ++ [width]) point → values point = W8A8OrderedBody.value (scores point)) →
    (∀ point, Idx.InBounds (shape ++ [width]) point → W8A8MaskedMaximum.Admissible cap (scores point)) →
    evalOps W8A8SourceSemantics.opSem ctx call env (actual ++ [W8A8AttentionTemplate.terminator output]) =
      .ok [.tensor (rowTensor (shape ++ [width]) (probability shape.length width scores))]

def check (shape : List Nat) (width input output : Nat) (actual : List Op) : Bool :=
  PolicyW8A8AttentionTemplate.check shape width input output actual && geometryCheck shape width

theorem check_sound {shape : List Nat} {width input output : Nat} {actual : List Op}
    (checked : check shape width input output actual = true) : Correct shape width input output actual := by
  simp only [check, Bool.and_eq_true] at checked
  intro cap ctx call env fallback interpretation values scores supplied initialized members
  have correspondence := PolicyW8A8AttentionTemplate.check_sound checked.1 ctx call env
    (.tensor ⟨shape ++ [width], .float .f32, values⟩) supplied
  have executed := W8A8AttentionExecution.template_execute ctx call
    [(0, .tensor ⟨shape ++ [width], .float .f32, values⟩)] fallback interpretation shape width
    (geometryCheck_sound checked.2) values scores (by rfl) initialized members
  exact correspondence.symm.trans executed

/-- Convert returned single-value success into actual straight-line execution and
an output binding. This cannot hide a failed prefix behind a successful return. -/
theorem exec_success {ctx : Ctx} {call : CallSem} {env : Env} {actual : List Op}
    {output : Nat} {value : RVal} (nonterm : ∀ op ∈ actual, op.isTerminator = false)
    (executed : evalOps W8A8SourceSemantics.opSem ctx call env
      (actual ++ [W8A8AttentionTemplate.terminator output]) = .ok [value]) :
    ∃ final, execOps W8A8SourceSemantics.opSem ctx call env actual = .ok final ∧ final.get? output = some value := by
  rw [evalOps_append _ _ _ _ _ _ nonterm] at executed
  cases prefixRun : execOps W8A8SourceSemantics.opSem ctx call env actual with
  | ub => simp [prefixRun] at executed
  | stuck reason => simp [prefixRun] at executed
  | ok final =>
    refine ⟨final, rfl, ?_⟩
    simp only [prefixRun, Outcome.bind_ok] at executed
    have returned : final.getAll [output] = .ok [value] := by
      simpa [evalOps, W8A8AttentionTemplate.terminator, Op.isTerminator, terminatorNames, Op.name, Op.operands]
        using executed
    cases supplied : final.get? output with
    | none => simp [Env.getAll, Outcome.mapM, supplied, Outcome.ofOption] at returned
    | some actualValue =>
      have same : actualValue = value := by
        simpa [Env.getAll, Outcome.mapM, supplied, Outcome.ofOption] using returned
      exact congrArg some same

theorem checked_exec {shape : List Nat} {width input output cap : Nat} {actual : List Op}
    (checked : check shape width input output actual = true)
    (ctx : Ctx) (call : CallSem) (env : Env) (fallback : FloatModel)
    (interpretation : ctx.M = W8A8RoundedReal.model fallback)
    (values : Idx → Val) (scores : Idx → FBits .f32)
    (supplied : env.get? input = some (.tensor ⟨shape ++ [width], .float .f32, values⟩))
    (initialized : ∀ point, Idx.InBounds (shape ++ [width]) point → values point = W8A8OrderedBody.value (scores point))
    (members : ∀ point, Idx.InBounds (shape ++ [width]) point → W8A8MaskedMaximum.Admissible cap (scores point)) :
    ∃ final, execOps W8A8SourceSemantics.opSem ctx call env actual = .ok final ∧
      final.get? output = some (.tensor (rowTensor (shape ++ [width]) (probability shape.length width scores))) := by
  have execution := check_sound checked cap ctx call env fallback interpretation values scores supplied initialized members
  have accepted := checked
  simp only [check, Bool.and_eq_true] at accepted
  have matched := accepted.1
  simp only [PolicyW8A8AttentionTemplate.check, Bool.and_eq_true] at matched
  have headers := (of_decide_eq_true matched.1).2.2
  exact exec_success (fun op member => (headers op member).2.1) execution

theorem output_bounded {shape : List Nat} {width cap : Nat} (scores : Idx → FBits .f32)
    (safe : cap ≤ 252)
    (members : ∀ point, Idx.InBounds (shape ++ [width]) point → W8A8MaskedMaximum.Admissible cap (scores point))
    (unmasked : ∀ cell, Idx.InBounds shape cell →
      ∃ column < width, W8A8IEEEBounds.ExponentLE cap (scores (cell ++ [column])))
    (count : width ≤ 2^22) (point : Idx) (bounds : Idx.InBounds (shape ++ [width]) point) :
    (rowTensor (shape ++ [width]) (probability shape.length width scores)).get point =
        W8A8OrderedBody.value (probability shape.length width scores point) ∧
      W8A8IEEEBounds.ExponentLE 128 (probability shape.length width scores point) ∧
      W8A8IEEE32.Nonnegative (probability shape.length width scores point) := by
  refine ⟨by simp only [rowTensor, Tensor.canon, bounds, ↓reduceIte], ?_⟩
  exact probability_bounded shape width scores safe members unmasked count point bounds

end MerlinVerify.Check.PolicyW8A8AttentionExecution
