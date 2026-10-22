/- Exact whole-program alpha correspondence, including internal calls, for local contracts.
No definedness obligation is imposed on unchanged computations. Unsupported operations retain
their exact stuck outcomes; this theorem does not claim they have been verified individually. -/
import MerlinVerify.Check.Module

namespace MerlinVerify.Rules.LocalProgramCorrespondence
open MerlinVerify MerlinVerify.Check

def function (s t : Func) : Bool :=
  s.name == t.name && s.isDecl == t.isDecl && alphaRegion [] s.body t.body

def functions : List Func → List Func → Bool
  | [], [] => true
  | s :: ss, t :: ts => function s t && functions ss ts
  | _, _ => false

theorem function_spec {s t : Func} (h : function s t = true) :
    s.name = t.name ∧ s.isDecl = t.isDecl ∧ alphaRegion [] s.body t.body = true := by
  simpa only [function, Bool.and_eq_true, beq_iff_eq, and_assoc] using h

def Found : Option Func → Option Func → Prop
  | none, none => True
  | some s, some t => function s t = true
  | _, _ => False

theorem functions_find : ∀ (ss ts : List Func),
    functions ss ts = true → ∀ name,
      Found (ss.find? (·.name == name)) (ts.find? (·.name == name))
  | [], [], _, _ => trivial
  | s :: ss, t :: ts, h, name => by
    simp only [functions, Bool.and_eq_true] at h
    have hn := (function_spec h.1).1
    simp only [List.find?_cons, hn]
    split
    · exact h.1
    · exact functions_find ss ts h.2 name
  | [], _ :: _, h, _ => by simp [functions] at h
  | _ :: _, [], h, _ => by simp [functions] at h

theorem functions_length : ∀ (ss ts : List Func),
    functions ss ts = true → ss.length = ts.length
  | [], [], _ => rfl
  | _ :: ss, _ :: ts, h => by
    simp only [functions, Bool.and_eq_true] at h
    simpa using functions_length ss ts h.2
  | [], _ :: _, h => by simp [functions] at h
  | _ :: _, [], h => by simp [functions] at h

/-- Fuel, symbol lookup, external declarations and calls are preserved. Region execution is
proved using the actual evaluator and the checked SSA correspondence, including tensor bodies. -/
theorem evalFunc_sound {src tgt : Module}
    (h : functions src.funcs tgt.funcs = true) (ctx : Ctx) :
    ∀ fuel, evalFunc Sem.Ops.opSem ctx src fuel =
      evalFunc Sem.Ops.opSem ctx tgt fuel := by
  intro fuel
  induction fuel with
  | zero => rfl
  | succ fuel ih =>
    funext name args
    have hf := functions_find src.funcs tgt.funcs h name
    change Found (src.func? name) (tgt.func? name) at hf
    cases hs : src.func? name with
    | none =>
      cases ht : tgt.func? name with
      | none => simp [evalFunc, hs, ht]
      | some t => simp [hs, ht, Found] at hf
    | some s =>
      cases ht : tgt.func? name with
      | none => simp [hs, ht, Found] at hf
      | some t =>
        simp only [hs, ht, Found] at hf
        obtain ⟨_, hd, hr⟩ := function_spec hf
        simp only [evalFunc, hs, ht, hd]
        split
        · rfl
        · rw [ih]
          exact congrFun (alphaRegion_sound
            (evalFunc Sem.Ops.opSem ctx tgt fuel) (evalFunc Sem.Ops.opSem ctx tgt fuel)
            [] s.body t.body ctx [] [] hr
            (by intro x y hxy; simp [Rho.find] at hxy) (fun _ _ => rfl)) args

/-- Full AST, metadata and function correspondence. No greedy rewrite synthesis is needed
for unchanged computations; repeated constants and internal calls remain ordinary alpha cases. -/
def check (src tgt : Module) : Bool :=
  (alphaOps [] src.ops tgt.ops && (outsideChanges src tgt).unverified.isEmpty &&
    decide (src.attrs = tgt.attrs ∧ src.props = tgt.props)) && functions src.funcs tgt.funcs

/-- The complete programs agree for every context, external interpretation, function and input.
The induction covers every call fuel rather than assuming an execution bound or acyclic calls. -/
theorem check_sound {src tgt : Module} (h : check src tgt = true)
    (ctx : Ctx) (name : String) (args : List RVal) :
    denote Sem.Ops.opSem ctx src name args = denote Sem.Ops.opSem ctx tgt name args := by
  simp only [check, Bool.and_eq_true] at h
  have he := evalFunc_sound h.2 ctx
  have hl := functions_length src.funcs tgt.funcs h.2
  simp only [denote]
  rw [he, hl]

#print axioms check_sound
end MerlinVerify.Rules.LocalProgramCorrespondence
