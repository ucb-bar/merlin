/- Closed, kernel-reduced geometry witnesses and full successful poison-free execution.
No numerical model is selected: the result is parametric in every FloatModel. -/
import MerlinVerify.Check.Overrank
open MerlinVerify Rules.OverrankIntent

def inhabited : Shape := ⟨[1,2],3,2⟩
theorem geometry : layoutCheck .f32 inhabited = true := by cbv
example : (allPoints inhabited.out).length = 4 := by cbv
example : layoutCheck .f64 ⟨[2,1],0,3⟩ = true := by cbv
example : layoutCheck .f32 ⟨[0,2],3,2⟩ = true := by cbv
example : layoutCheck .f64 ⟨[2,1],3,0⟩ = true := by cbv

/-- Arbitrary initialized inputs, not merely a diagnostic or a successful typing check. -/
example (ctx : Ctx) (A B : Idx → Option (FBits .f32))
    (h : Domain .f32 inhabited A B) (old : RVal) :
    ∃ results,
      denote Sem.Ops.opSem ctx (Rules.OverrankKernel.program .f32 inhabited) "subject"
        [.tensor (Rules.OverrankTensor.floats .f32 inhabited.left A),
         .tensor (Rules.OverrankTensor.floats .f32 inhabited.right B),old] = .ok results ∧
      ResultsDefined results := by
  exact ⟨_,Rules.OverrankKernel.program_defined ctx .f32 inhabited geometry A B h old⟩

/-- Concrete witness that the domain is inhabited and the complete program actually succeeds. -/
example (ctx : Ctx) : ∃ results,
    denote Sem.Ops.opSem ctx (Rules.OverrankKernel.program .f32 inhabited) "subject"
      [.tensor (Rules.OverrankTensor.floats .f32 inhabited.left (fun _ => some 0)),
       .tensor (Rules.OverrankTensor.floats .f32 inhabited.right (fun _ => some 0)),
       .tensor (Tensor.full inhabited.out (.float .f32) .poison)] = .ok results ∧
    ResultsDefined results := by
  exact ⟨_,Rules.OverrankKernel.program_defined ctx .f32 inhabited geometry
    (fun _ => some 0) (fun _ => some 0) (domain_inhabited .f32 inhabited) _⟩

#print axioms Rules.OverrankKernel.program_defined
#print axioms Check.Overrank.check_sound
