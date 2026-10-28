/- Kernel-checked examples: typed internal calls and explicit external obligations.
Successful results may contain poison. A typing proof does not discharge the numerical law. -/
import MerlinVerify.Rules.ScalarSquareTypingClosed
open MerlinVerify MerlinVerify.Check MerlinVerify.Rules
open ScalarSquareTyping ScalarSquareClaims

def typingLeaf : Op := .mk "func.func" [] []
  [("sym_name", .str "leaf"), ("function_type", .type (.function [.float .f32] [.float .f32]))] []
  [.mk [.mk [⟨0, .float .f32⟩] [
    .mk "arith.constant" [⟨1, .float .f32⟩] [] [("value", .float 1073741824 (.float .f32))] [] [] [],
    .mk "math.powf" [⟨2, .float .f32⟩] [0, 1] [] [] [] [],
    .mk "func.return" [] [2] [] [] [] []]]] []

def typingCaller : Op := .mk "func.func" [] []
  [("sym_name", .str "entry"), ("function_type", .type (.function [.float .f32] [.float .f32]))] []
  [.mk [.mk [⟨3, .float .f32⟩] [
    .mk "func.call" [⟨4, .float .f32⟩] [3] [("callee", .symbolRef "leaf" [])] [] [] [],
    .mk "func.return" [] [4] [] [] [] []]]] []

def typingProgram : Module := ⟨[typingLeaf, typingCaller], [], []⟩
theorem typingProgram_checked : closedCheck 64 typingProgram = true := by cbv
example (M : FloatModel) : SourceTyping typingProgram (Family M 64) 64 :=
  closedCheck_sound typingProgram_checked M
example : RVal.HasTy 64 (.scalar .poison) (.scalar (.float .f32)) := trivial
example : (Outcome.ok [RVal.scalar .poison]).isOk = true := rfl
example : ¬ (Outcome.stuck "unsupported" : Outcome (List RVal)).NotStuck := by simp [Outcome.NotStuck]
#print axioms typingProgram_checked
#print axioms ScalarSquareTyping.closedCheck_sound
#print axioms ScalarSquareTyping.source_typing
