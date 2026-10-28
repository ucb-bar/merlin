/- Separate provisional claim forms. Strict source preservation requires the numerical law,
source operational typing and source definedness. Policy claims require an explicit external
approval decision and policy definedness, and never claim equality to the original source. -/
import MerlinVerify.Check.ScalarSquareProgram

namespace MerlinVerify.Rules.ScalarSquareClaims

open MerlinVerify MerlinVerify.Check MerlinVerify.Sem.Ops
open ScalarSquareSSA ScalarSquareComposition

def Family (M : FloatModel) (ib : Nat) (ctx : Ctx) : Prop := ctx.M = M ∧ ctx.indexBits = ib

/-- F may fix a model and impose explicit typed/defined external-call contracts. It must
be inhabited and pin the index width; choosing F=False cannot establish a paper claim. -/
def ContextDomain (F : Ctx → Prop) (ib : Nat) : Prop :=
  (∃ ctx, F ctx) ∧ ∀ ctx, F ctx → ctx.indexBits = ib

def Inputs (source : Module) (ib : Nat) (name : String) (values : List RVal) : Prop :=
  match source.func? name with
  | none => True
  | some f => if f.isDecl then True else BlockDomain ib f.body.blocks values

/-- A strict paper claim must name a defined entry and a nonempty typed input domain.
Unsupported input types must not turn a claim into an empty universal quantifier. -/
def EntryDomain (source : Module) (ib : Nat) (name : String) : Prop :=
  (∃ f, source.func? name = some f ∧ f.isDecl = false) ∧ ∃ values, Inputs source ib name values

/-- Explicit source-side obligations, for every call fuel and typed region invocation.
No source/target equality occurs in this definition. PrimitiveTyping and ResultFacts are
local semantic typing/constant-transfer obligations; they are not inferred from a receipt. -/
def SourceTyping (source : Module) (F : Ctx → Prop) (ib : Nat) : Prop :=
  ∀ ctx, F ctx → ∀ fuel name f, source.func? name = some f →
    ReadyBlocks (Inputs source ib) F ib (evalFunc opSem ctx source fuel)
      [] [] 0 [] f.body.blocks

theorem numerical_nonempty (ib : Nat) :
    Numerical (Family ScalarSquareContract.zeroModel ib) := by
  intro ctx family site bits
  rw [family.1]
  rfl

theorem calls_equal {source target : Module} (checked : ScalarSquareProgram.check source target = true)
    (F : Ctx → Prop) (ib : Nat) (numerical : Numerical F)
    (typing : SourceTyping source F ib) :
    ∀ fuel ctx, F ctx → ∀ name values, Inputs source ib name values →
      evalFunc opSem ctx source fuel name values = evalFunc opSem ctx target fuel name values := by
  have inventory := ScalarSquareProgram.check_sound checked
  intro fuel
  induction fuel with
  | zero => intros; rfl
  | succ fuel ih =>
    intro ctx family name values inputs
    have paired := inventory.2.2 name
    cases found : source.func? name with
    | none =>
      cases other : target.func? name with
      | none => simp [evalFunc, found, other]
      | some _ => simp [found, other, ScalarSquareProgram.LookupPair] at paired
    | some f =>
      cases other : target.func? name with
      | none => simp [found, other, ScalarSquareProgram.LookupPair] at paired
      | some t =>
        simp only [found, other, ScalarSquareProgram.LookupPair] at paired
        simp only [evalFunc, found, other, ← paired.declaration]
        by_cases declaration : f.isDecl = true
        · simp [declaration]
        · simp only [declaration, Bool.false_eq_true, ↓reduceIte]
          have args : BlockDomain ib f.body.blocks values := by
            simpa [Inputs, found, declaration] using inputs
          have calls : CallsAgree (Inputs source ib) (evalFunc opSem ctx source fuel)
              (evalFunc opSem ctx target fuel) := fun n vs typed => ih ctx family n vs typed
          rw [paired.region]
          rw [eval_region, eval_region]
          exact blocks_sound (Inputs source ib) F ib numerical _ _ calls
            [] [] 0 [] f.body.blocks ctx family (empty ib)
            (typing ctx family fuel name f found) values args

theorem source_equal {source target : Module} (checked : ScalarSquareProgram.check source target = true)
    (F : Ctx → Prop) (ib : Nat) (numerical : Numerical F)
    (typing : SourceTyping source F ib) (ctx : Ctx) (family : F ctx)
    (name : String) (values : List RVal) (inputs : Inputs source ib name values) :
    MerlinVerify.denote opSem ctx target name values = MerlinVerify.denote opSem ctx source name values := by
  have length := (ScalarSquareProgram.check_sound checked).2.1
  unfold MerlinVerify.denote
  rw [← length]
  exact (calls_equal checked F ib numerical typing _ ctx family name values inputs).symm

/-- This form excludes stuck executions. It retains the shared refinement convention for UB. -/
def StrictSourcePreservation (source target : Module) (F : Ctx → Prop) (ib : Nat) (name : String) : Prop :=
  ContextDomain F ib ∧ EntryDomain source ib name ∧ ∀ ctx, F ctx → ∀ values, Inputs source ib name values →
    Refines ResultsRefine (MerlinVerify.denote opSem ctx target name values)
      (MerlinVerify.denote opSem ctx source name values)

theorem strict_source_preservation {source target : Module}
    (checked : ScalarSquareProgram.check source target = true) (F : Ctx → Prop) (ib : Nat)
    (numerical : Numerical F) (typing : SourceTyping source F ib) (name : String)
    (contexts : ContextDomain F ib)
    (domain : EntryDomain source ib name)
    (defined : FuncNotStuck opSem F (Inputs source ib name) source name) :
    StrictSourcePreservation source target F ib name := by
  refine ⟨contexts, domain, ?_⟩
  intro ctx family values inputs
  rw [source_equal checked F ib numerical typing ctx family name values inputs]
  have h := defined ctx family values inputs
  cases result : MerlinVerify.denote opSem ctx source name values with
  | ub => trivial
  | stuck reason => simp [result, Outcome.NotStuck] at h
  | ok values => exact ResultsRefine.refl values

/-- Stronger successful-execution form: no UB or stuck case can satisfy this conclusion. -/
theorem source_success {source target : Module}
    (checked : ScalarSquareProgram.check source target = true) (F : Ctx → Prop) (ib : Nat)
    (numerical : Numerical F) (typing : SourceTyping source F ib)
    (ctx : Ctx) (family : F ctx) (name : String) (values results : List RVal)
    (inputs : Inputs source ib name values)
    (defined : MerlinVerify.denote opSem ctx source name values = .ok results) :
    MerlinVerify.denote opSem ctx target name values = .ok results := by
  rw [source_equal checked F ib numerical typing ctx family name values inputs, defined]

/-- An explicit multiplication-based policy is a different program specification.
This definition intentionally names the multiplication reference, not the original source. -/
def policyMeaning (ctx : Ctx) (source : Module) (name : String) (values : List RVal) : Outcome (List RVal) :=
  MerlinVerify.denote opSem ctx (ScalarSquare.reference source) name values

/-- Review evidence is data, not an axiom. The external review process establishes its
authority; the formal claim binds source, policy, context domain, width and entry. -/
structure PolicyDecision where
  source : Module
  multiplication : Module
  contexts : Ctx → Prop
  indexBits : Nat
  entry : String
  reviewer : String
  approved : Bool

def Authorizes (decision : PolicyDecision) (source : Module) (F : Ctx → Prop)
    (ib : Nat) (name : String) : Prop :=
  decision.approved = true ∧ decision.reviewer ≠ "" ∧ decision.source = source ∧
    decision.multiplication = ScalarSquare.reference source ∧ decision.contexts = F ∧
    decision.indexBits = ib ∧ decision.entry = name

def ApprovedMultiplicationPolicy (decision : PolicyDecision) (source target : Module)
    (F : Ctx → Prop) (ib : Nat) (name : String) : Prop :=
  Authorizes decision source F ib name ∧ ContextDomain F ib ∧ EntryDomain source ib name ∧
    ∀ ctx, F ctx → ∀ values, Inputs source ib name values →
    Refines ResultsRefine (MerlinVerify.denote opSem ctx target name values)
      (policyMeaning ctx source name values)

theorem approved_multiplication_policy {source target : Module}
    (checked : ScalarSquare.check source target = true) (F : Ctx → Prop) (ib : Nat)
    (name : String) (decision : PolicyDecision) (approval : Authorizes decision source F ib name)
    (contexts : ContextDomain F ib)
    (domain : EntryDomain source ib name)
    (defined : ∀ ctx, F ctx → ∀ values, Inputs source ib name values →
      (policyMeaning ctx source name values).NotStuck) :
    ApprovedMultiplicationPolicy decision source target F ib name := by
  refine ⟨approval, contexts, domain, ?_⟩
  intro ctx family values inputs
  rw [ScalarSquare.complete_output_sound checked]
  change Refines ResultsRefine (policyMeaning ctx source name values) (policyMeaning ctx source name values)
  have h := defined ctx family values inputs
  cases result : policyMeaning ctx source name values with
  | ub => trivial
  | stuck reason => simp [result, Outcome.NotStuck] at h
  | ok values => exact ResultsRefine.refl values

theorem unapproved_rejected (decision : PolicyDecision) (unreviewed : decision.approved = false)
    (source : Module) (F : Ctx → Prop) (ib : Nat) (name : String) :
    ¬ Authorizes decision source F ib name := by
  intro approval
  simp [Authorizes, unreviewed] at approval

/-- Successful policy execution is also available without any power/multiplication law. -/
theorem policy_success {source target : Module}
    (checked : ScalarSquare.check source target = true) (F : Ctx → Prop) (ib : Nat)
    (name : String) (decision : PolicyDecision) (approval : Authorizes decision source F ib name)
    (ctx : Ctx) (family : F ctx) (values results : List RVal)
    (inputs : Inputs source ib name values)
    (defined : policyMeaning ctx source name values = .ok results) :
    Authorizes decision source F ib name ∧ MerlinVerify.denote opSem ctx target name values = .ok results := by
  exact ⟨approval, (ScalarSquare.complete_output_sound checked opSem ctx name values).trans defined⟩

#print axioms calls_equal
#print axioms source_equal
#print axioms strict_source_preservation
#print axioms source_success
#print axioms approved_multiplication_policy
#print axioms unapproved_rejected
#print axioms policy_success
#print axioms numerical_nonempty

end MerlinVerify.Rules.ScalarSquareClaims
