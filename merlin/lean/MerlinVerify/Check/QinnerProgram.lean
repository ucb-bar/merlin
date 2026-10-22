/- Whole-program correspondence with explicit source binding interpretation. Only @forward gains
arguments; calls to @forward are refused. Every other function and every nested region is checked.
The unchanged computations, float model, and external calls remain arbitrary. -/
import MerlinVerify.Rules.QinnerABI
import MerlinVerify.Check.ParametricModule
namespace MerlinVerify.Check.QinnerProgram
open MerlinVerify MerlinVerify.Check Rules.QinnerBinding Rules.QinnerABI

private def valueDecEq (a b : Value) : Decidable (a = b) :=
  decidable_of_iff (a.id = b.id ∧ a.ty = b.ty) (by cases a; cases b; simp)
local instance : DecidableEq Value := valueDecEq

abbrev rho (a b : List Value) : Rho := (a.map (·.id)).zip (b.map (·.id))

def entryCheck (slots : List Slot) : Region → Region → Bool
  | .mk [.mk a ops], .mk [.mk b tail] =>
    let old := b.take a.length
    decide (b = old ++ slots.map (·.arg)) &&
    decide (old.map (·.ty) = a.map (·.ty)) &&
    decide (b.map (·.id)).Nodup && protects slots (old.map (·.id)) &&
    QinnerSSA.operations slots (rho a old) ops tail
  | _, _ => false

theorem entryCheck_sound {slots : List Slot} {s t : Region}
    (checked : entryCheck slots s t = true) (binding : String → RVal)
    (ctx : Ctx) (callS callT : CallSem)
    (calls : CallAgree callS callT (calleesBlocks s.blocks))
    (args : List RVal) (arity : s.blocks.head!.args.length = args.length) :
    evalRegion (sem binding) ctx callS [] s args =
      evalRegion Sem.Ops.opSem ctx callT [] t (args ++ values slots binding) := by
  match s,t with
  | .mk [.mk a ops], .mk [.mk b tail] =>
    simp only [entryCheck, Bool.and_eq_true, decide_eq_true_eq] at checked
    obtain ⟨⟨⟨⟨shape,types⟩,distinct⟩,separate⟩,opsChecked⟩ := checked
    let old := b.take a.length
    change b = old ++ slots.map (·.arg) at shape
    change old.map (·.ty) = a.map (·.ty) at types
    have lengths : old.length = a.length := by
      simpa only [List.length_map] using congrArg List.length types
    have oldDistinct : (old.map (·.id)).Nodup := by
      rw [shape, List.map_append, List.nodup_append] at distinct
      exact distinct.1
    have extraDistinct : (slots.map (·.arg.id)).Nodup := by
      rw [shape, List.map_append, List.nodup_append] at distinct
      simpa only [List.map_map, Function.comp_def] using distinct.2.1
    have hargs : a.length = args.length := arity
    obtain ⟨corr,supplied⟩ := initial a old slots binding args hargs lengths oldDistinct extraDistinct separate
    have hb : b.length = (args ++ values slots binding).length := by
      rw [shape, List.length_append, List.length_append]
      simp only [List.length_map, values, lengths, hargs]
    simp only [evalRegion, evalBlocks, evalBlock]
    rw [bindAll_eq_ok hargs, bindAll_eq_ok hb]
    simp only [Outcome.bind_ok, List.append_nil]
    have envEq : (b.map (·.id)).zip (args ++ values slots binding) =
        (old.map (·.id)).zip args ++ environment slots binding := by
      rw [shape, List.map_append]
      rw [List.zip_append (by simpa only [List.length_map] using lengths.trans hargs)]
      simp only [List.map_map, Function.comp_def, environment_zip]
    rw [envEq]
    exact QinnerSSA.operations_sound binding slots callS callT _ ops tail ctx _ _ opsChecked corr supplied
      (fun name member => calls name (by simp only [Region.blocks, calleesBlocks]; exact List.mem_append_left _ member))
  | .mk [], _ | .mk (_ :: _ :: _), _ | _, .mk [] | _, .mk (_ :: _ :: _) =>
    simp [entryCheck] at checked

def functionCheck (slots : List Slot) (s t : Func) : Bool :=
  s.name == t.name && s.isPrivate == t.isPrivate && s.isDecl == t.isDecl &&
  (calleesBlocks s.body.blocks).all (· != "forward") &&
  (if s.name == "forward" then !s.isDecl && entryCheck slots s.body t.body
    else QinnerSSA.region [] [] s.body t.body)

def functionsCheck (slots : List Slot) : List Func → List Func → Bool
  | [], [] => true
  | a::rest,b::tail => functionCheck slots a b && functionsCheck slots rest tail
  | _,_ => false

def Matched (slots : List Slot) (s t : Func) : Prop := functionCheck slots s t = true

theorem functionsCheck_sound {slots : List Slot} {s t : List Func}
    (h : functionsCheck slots s t = true) : List.Forall₂ (Matched slots) s t := by
  induction s generalizing t with
  | nil => cases t <;> simp_all [functionsCheck]
  | cons a rest ih =>
    cases t with
    | nil => simp [functionsCheck] at h
    | cons b tail =>
      simp only [functionsCheck, Bool.and_eq_true] at h
      exact .cons h.1 (ih h.2)

theorem match_names {slots : List Slot} {a b : Func} (h : Matched slots a b) : a.name = b.name := by
  simp only [Matched, functionCheck, Bool.and_eq_true, beq_iff_eq] at h
  exact h.1.1.1.1

theorem lookup_match {slots : List Slot} {s t : List Func} (h : List.Forall₂ (Matched slots) s t)
    (name : String) :
    (s.find? (·.name == name) = none ∧ t.find? (·.name == name) = none) ∨
    ∃ a b, s.find? (·.name == name) = some a ∧ t.find? (·.name == name) = some b ∧ Matched slots a b := by
  induction h with
  | nil => exact .inl ⟨rfl,rfl⟩
  | @cons a b s t same rest ih =>
    by_cases named : a.name == name
    · have other : b.name == name := by simpa [match_names same] using named
      exact .inr ⟨a,b,by simp [List.find?,named],by simp [List.find?,other],same⟩
    · have other : (b.name == name) = false := by simpa [match_names same] using named
      simpa [List.find?,named,other] using ih

theorem nonentry_equal {slots : List Slot} {s t : Module}
    (matched : List.Forall₂ (Matched slots) s.funcs t.funcs) (binding : String → RVal) (ctx : Ctx) :
    ∀ fuel name args, name ≠ "forward" →
    evalFunc (sem binding) ctx s fuel name args = evalFunc Sem.Ops.opSem ctx t fuel name args := by
  intro fuel
  induction fuel with
  | zero => intros; rfl
  | succ fuel ih =>
    intro name args nonentry
    rcases lookup_match matched name with missing | ⟨a,b,foundS,foundT,same⟩
    · change s.func? name = none ∧ t.func? name = none at missing
      simp [evalFunc,missing.1,missing.2]
    · change s.func? name = some a at foundS
      change t.func? name = some b at foundT
      have named : a.name = name := by simpa using List.find?_some foundS
      simp only [Matched, functionCheck, Bool.and_eq_true, beq_iff_eq] at same
      obtain ⟨⟨⟨⟨hn,hpriv⟩,hdecl⟩,noCalls⟩,body⟩ := same
      have notForward : (a.name == "forward") = false := by simp [named,nonentry]
      simp only [show a.name ≠ "forward" by simpa [named] using nonentry, ↓reduceIte] at body
      simp only [evalFunc,foundS,foundT,hdecl]
      split
      · rfl
      · have calls : CallAgree (evalFunc (sem binding) ctx s fuel) (evalFunc Sem.Ops.opSem ctx t fuel)
            (calleesBlocks a.body.blocks) := fun c hc => funext fun vals => ih c vals
              (by simpa using List.all_eq_true.mp noCalls c hc)
        exact congrFun (QinnerSSA.region_sound binding [] _ _ [] a.body b.body ctx [] [] body
          Corr.nil (by intro x h; cases h) calls) args

theorem program_equal {slots : List Slot} {s t : Module}
    (checked : functionsCheck slots s.funcs t.funcs = true)
    (binding : String → RVal) (ctx : Ctx) {a : Func} (source : s.func? "forward" = some a)
    (args : List RVal) (arity : a.body.blocks.head!.args.length = args.length) :
    denote (sem binding) ctx s "forward" args =
    denote Sem.Ops.opSem ctx t "forward" (args ++ values slots binding) := by
  have matched := functionsCheck_sound checked
  have lengths := List.Forall₂.length_eq matched
  rcases lookup_match matched "forward" with missing | ⟨left,right,foundS,foundT,same⟩
  · change s.func? "forward" = none ∧ _ at missing
    rw [source] at missing; cases missing.1
  · change s.func? "forward" = some left at foundS
    rw [source] at foundS
    cases foundS
    change t.func? "forward" = some right at foundT
    have named : a.name = "forward" := by simpa using List.find?_some source
    simp only [Matched,functionCheck,Bool.and_eq_true,beq_iff_eq,named,beq_self_eq_true,↓reduceIte] at same
    obtain ⟨⟨⟨⟨hn,hpriv⟩,hdecl⟩,noCalls⟩,body⟩ := same
    try simp only [Bool.and_eq_true,Bool.not_eq_true] at body
    have nondecl : a.isDecl = false := by cases hd : a.isDecl <;> simp_all
    simp only [denote,lengths,evalFunc,source,foundT,← hdecl,nondecl,↓reduceIte, Bool.false_eq_true]
    apply entryCheck_sound body.2 binding ctx _ _ _ args arity
    intro name hc
    funext vals
    exact nonentry_equal matched binding ctx _ name vals
      (by simpa using List.all_eq_true.mp noCalls name hc)

#print axioms entryCheck_sound
#print axioms program_equal
end MerlinVerify.Check.QinnerProgram
