/- The selection theorem accepts ordinary typed initialized tensors, without canonicality hints. -/
import MerlinVerify.Rules.FrontendMaskWhole
import MerlinVerify.Check.DefinedLinalg

namespace MerlinVerify.Rules.FrontendMaskTyped
open MerlinVerify MerlinVerify.Check MerlinVerify.Check.Definedness
open FrontendMaskWhole FrontendMaskProgram

def word (value : Val) : BitVec 64 :=
  match value with
  | .some (.bits 64 x) => x
  | _ => 0

def boolean (value : Val) : Bool :=
  match value with
  | .some (.bits 1 x) => x == 1
  | _ => false

theorem initialized_bits {shape : List Nat} {t : Tensor} {width : Nat}
    (ht : RVal.HasTy 64 (.tensor t) (.tensor shape (.int width)))
    (hd : RVal.Defined (.tensor t)) {p : Idx} (hp : Idx.InBounds shape p) :
    ∃ x : BitVec width, t.get p = .some (.bits width x) := by
  obtain ⟨x, hx⟩ := initialized_int (show Fact.Holds 64 (.scalar (t.get p)) ⟨.scalar (.int width), true⟩ from
    ⟨ht.2.2 p hp, fun _ => hd p (ht.1.symm ▸ hp)⟩)
  exact ⟨x, RVal.scalar.inj hx⟩

theorem one_bit (x : BitVec 1) : x = 0 ∨ x = 1 := by
  have hbound := x.isLt
  by_cases h : x.toNat = 0
  · left; apply BitVec.eq_of_toNat_eq; change x.toNat = 0; exact h
  · right; apply BitVec.eq_of_toNat_eq; change x.toNat = 1; omega

def words (t : Tensor) (i : Nat) : BitVec 64 :=
  word (t.get (Sem.Ops.Tensor.delinearize t.shape i))

def bits (t : Tensor) (i : Nat) : Bool :=
  boolean (t.get (Sem.Ops.Tensor.delinearize t.shape i))

theorem word_represents {shape : List Nat} {t : Tensor}
    (ht : RVal.HasTy 64 (.tensor t) (.tensor shape (.int 64))) (hd : RVal.Defined (.tensor t)) :
    Represents t shape (.int 64) (fun i => .some (.bits 64 (words t i))) := by
  refine ⟨ht.1, ht.2.1, ?_⟩
  intro p hp
  obtain ⟨x, hx⟩ := initialized_bits ht hd hp
  simp only [words, ht.1, ReshapeIndex.delinearize_linear shape p hp, hx, word]

theorem mask_represents {t : Tensor}
    (ht : RVal.HasTy 64 (.tensor t) (.tensor [1,32,32] (.int 1))) (hd : RVal.Defined (.tensor t)) :
    Represents t [1,32,32] (.int 1) (fun i => Sem.Ops.Arith.boolVal (bits t i)) := by
  refine ⟨ht.1, ht.2.1, ?_⟩
  intro p hp
  obtain ⟨x, hx⟩ := initialized_bits ht hd hp
  simp only [bits, ht.1, ReshapeIndex.delinearize_linear [1,32,32] p hp, hx, boolean]
  rcases one_bit x with rfl | rfl <;> rfl

def inputTypes : List VTy :=
  [.tensor [1,32,32] (.int 1), .tensor [1,1024] (.int 64), .tensor [1,1024] (.int 64)]

def resultTypes : List VTy := [.tensor [1,1024] (.int 64), .scalar .index]

/-- This is a typed region theorem over arbitrary initialized tensor representations and arbitrary
call semantics. No input values or safety bits come from Python. -/
theorem body_exec (limit : Nat) (ctx : Ctx) (call : CallSem) (args : List RVal)
    (ha : HasTys 64 args inputTypes) (hd : ResultsDefined args) (hib : ctx.indexBits = 64)
    (hu : FrontendBoolFptosi.UneCorrect ctx.M) (hl : 1024 ≤ limit) :
    ∃ results, evalRegion (FrontendMaskRuntime.sem limit) ctx call [] FrontendMaskTemplate.body args = .ok results ∧
      HasTys 64 results resultTypes ∧ ResultsDefined results := by
  obtain _ | ⟨hm, rest⟩ := ha
  obtain _ | ⟨hv, rest⟩ := rest
  obtain _ | ⟨hs, rest⟩ := rest
  cases rest
  rename_i mask values seed
  have hdm := hd mask (by simp)
  have hdv := hd values (by simp)
  have hds := hd seed (by simp)
  cases mask with
  | scalar _ => cases hm
  | tensor mask =>
    cases values with
    | scalar _ => cases hv
    | tensor values =>
      cases seed with
      | scalar _ => cases hs
      | tensor seed =>
        obtain ⟨out, he, hshape, hdefined, htyped⟩ := body_inputs_exec limit ctx call
          (bits mask) (words values) (words seed) mask values seed
          (mask_represents hm hdm) (word_represents hv hdv) (word_represents hs hds) hib hu hl
        refine ⟨_, he, .cons htyped (.cons ?_ .nil), ?_⟩
        · simp [index, RVal.HasTy, Val.HasTy, Scalar.conforms]
        · intro value hvalue
          simp only [List.mem_cons, List.not_mem_nil, or_false] at hvalue
          rcases hvalue with rfl | rfl
          · exact hdefined
          · simp [index, RVal.Defined]

/-- Full actual-cut acceptance transports typed execution. The approved direct f32 model
supplies the generic comparison premise; the input domain no longer requires canonical tensors. -/
theorem check_sound {actual : Module} (h : FrontendMaskWhole.check actual = true)
    (limit : Nat) (base : FloatModel) (ctx : Ctx) (args : List RVal)
    (ha : HasTys 64 args inputTypes) (hd : ResultsDefined args) (hib : ctx.indexBits = 64)
    (hl : 1024 ≤ limit) :
    ∃ results, denote (FrontendMaskRuntime.sem limit) {ctx with M := FrontendF32Comparison.model base}
      actual "subject" args = .ok results ∧ HasTys 64 results resultTypes ∧ ResultsDefined results := by
  simp only [FrontendMaskWhole.check, Bool.and_eq_true] at h
  have he := FrontendBoolSimulationWith.evalFunc_sound (FrontendMaskRuntime.sem limit)
    (FrontendBoolSimulationWith.candidateLaws limit) h.2
    {ctx with M := FrontendF32Comparison.model base} (FrontendF32Comparison.model_correct base)
  rw [FrontendBoolSimulationWith.policy_nil] at he
  have hlen := FrontendBoolSimulation.functions_length FrontendMaskTemplate.subject.funcs actual.funcs h.2
  obtain ⟨results, hr, ht, hd⟩ := body_exec limit {ctx with M := FrontendF32Comparison.model base}
    (evalFunc (FrontendMaskRuntime.sem limit) {ctx with M := FrontendF32Comparison.model base}
      FrontendMaskTemplate.subject 1) args ha hd hib (FrontendF32Comparison.model_correct base) hl
  have hs : denote (FrontendMaskRuntime.sem limit) {ctx with M := FrontendF32Comparison.model base}
      FrontendMaskTemplate.subject "subject" args = .ok results := by
    rw [denote, show FrontendMaskTemplate.subject.funcs.length = 1 from by cbv, evalFunc, subject_find]
    exact hr
  refine ⟨results, ?_, ht, hd⟩
  simpa only [denote, he, hlen] using hs

#print axioms body_exec
#print axioms check_sound
end MerlinVerify.Rules.FrontendMaskTyped
