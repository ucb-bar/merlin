/-
The typing rule for Merlin's op semantics, assembled from the family rules, and the end-to-end
claim (design §4.5).

`merlinRule` tries the family rules in turn; `merlinRule_sound` is the conjunction of their
soundness theorems. `supportCheck` runs the progress checker over a function body, and
`supportCheck_sound` turns its acceptance into `FuncNotStuck` on typed inputs (with results of the
declared types, `supportCheck_safe`). `funcRefines_of_support` combines this with a chain of
conditional refinements: the final program strictly refines the capture on typed inputs.

An op no family rule accepts makes the check fail, so the claim is only made for captures inside
the proved fragment. `diagnoseOps` names the first refused op; it is a report, not part of any
proof.
-/
import MerlinVerify.Check.RulesLinalg
import MerlinVerify.Check.RulesTensor
import MerlinVerify.Check.RulesDequant
import MerlinVerify.Check.RulesQuantize
import MerlinVerify.Check.RulesBitcast

namespace MerlinVerify.Check

open MerlinVerify MerlinVerify.Sem.Ops

/-! ### The rulebook -/

/-- Merlin's typing rule at index width `ib`. -/
def merlinRule (ib : Nat) : OpRule :=
  firstRule [cfRule, arithRule ib, Bitcast.rule, linalgRule, tensorRule, dequantRule, quantizeRule]

theorem merlinRule_sound (ib : Nat) : RuleSound ib (merlinRule ib) := by
  apply firstRule_sound
  simp only [List.forall_mem_cons, List.not_mem_nil, IsEmpty.forall_iff, implies_true, and_true]
  exact ⟨cfRule_sound ib, arithRule_sound ib, Bitcast.rule_sound ib, linalgRule_sound ib,
    tensorRule_sound ib, dequantRule_sound ib, quantizeRule_sound ib⟩

def merlinBook (ib : Nat) : Rulebook opSem ib := (merlinRule_sound ib).book

/-! ### Whole functions -/

/-- Function `f` of `m` has a body, and the body passes the progress checker with arguments of
types `argTys` and results of types `resTys`, at loop depth 0 with nothing else in scope. -/
def supportCheck {sem : OpSem} {ib : Nat} (R : Rulebook sem ib) (m : Module) (f : String)
    (argTys resTys : List VTy) : Bool :=
  match m.func? f with
  | some fn => !fn.isDecl && checkRegion R [] fn.body ⟨argTys, resTys, 0⟩
  | none => false

theorem envTyped_nil (ib : Nat) : EnvTyped ib [] [] := fun _ _ h => by simp at h

/-- **Support.** A function accepted by `supportCheck` is never `stuck` on typed inputs, in any
context at index width `ib`, and its results have the declared types. -/
theorem supportCheck_safe {sem : OpSem} {ib : Nat} (R : Rulebook sem ib) {m : Module} {f : String}
    {argTys resTys : List VTy} (h : supportCheck R m f argTys resTys = true) (ctx : Ctx)
    (hib : ctx.indexBits = ib) (args : List RVal) (hargs : HasTys ib args argTys) :
    Safe (HasTys ib · resTys) (denote sem ctx m f args) := by
  unfold supportCheck at h
  split at h
  · next fn hfn =>
    simp only [Bool.and_eq_true, Bool.not_eq_eq_eq_not, Bool.not_true] at h
    obtain ⟨hnd, hc⟩ := h
    unfold denote
    simp only [evalFunc, hfn, hnd, Bool.false_eq_true, ↓reduceIte]
    exact checkRegion_safe R [] fn.body _ _ [] hc (envTyped_nil ib) ctx hib (Nat.zero_le _) args hargs
  · cases h

theorem supportCheck_sound {sem : OpSem} {ib : Nat} (R : Rulebook sem ib) {m : Module} {f : String}
    {argTys resTys : List VTy} (h : supportCheck R m f argTys resTys = true) :
    FuncNotStuck sem (fun ctx => ctx.indexBits = ib) (fun args => HasTys ib args argTys) m f :=
  fun ctx hib args hargs => (supportCheck_safe R h ctx hib args hargs).1

/-- **End to end.** A chain of conditional refinements from the capture `src` to `tgt` (what the
per-boundary checkers prove, composed with `FuncRefinesC.trans`), and a capture inside the proved
fragment, give strict refinement on typed inputs: `tgt` is defined wherever `src` is, and refines
it there. -/
theorem funcRefines_of_support {ib : Nat} {src tgt : Module} {f : String} {argTys resTys : List VTy}
    (hchain : FuncRefinesC opSem (fun _ => True) (fun _ => True) tgt f src f)
    (hsupp : supportCheck (merlinBook ib) src f argTys resTys = true) :
    FuncRefines opSem (fun ctx => ctx.indexBits = ib) (fun args => HasTys ib args argTys)
      tgt f src f :=
  FuncRefinesC.strict (fun ctx _ args _ => hchain ctx trivial args trivial)
    (supportCheck_sound (merlinBook ib) hsupp)

/-! ### Diagnostics -/

variable {sem : OpSem} {ib : Nat}

mutual
  /-- The first op `checkOps` refuses, with the reason; `none` when it accepts. -/
  def diagnoseOps (R : Rulebook sem ib) (d : Nat) (Γ : TEnv) : List Op → List VTy → Option String
    | [], _ => some "block without terminator"
    | (op@(.mk name results operands _ _ regions _)) :: rest, yields =>
      match operands.mapM (fun x => List.lookup x Γ) with
      | none => some s!"{name}: operand of unmodelled type"
      | some argTys =>
        if op.isTerminator then
          if rest.isEmpty && argTys == yields then none
          else some s!"{name}: yields other types than its region declares"
        else
          match results.mapM (fun (r : Value) => vtyOf ib r.ty) with
          | none => some s!"{name}: result of unmodelled type"
          | some resTys =>
            match R.rule d op argTys resTys with
            | none => some s!"{name}: no rule accepts this typing"
            | some sigs =>
              match diagnoseRegions R Γ regions sigs with
              | some e => some e
              | none => diagnoseOps R d ((results.map (fun (r : Value) => r.id)).zip resTys ++ Γ) rest yields

  def diagnoseRegions (R : Rulebook sem ib) (Γ : TEnv) : List Region → List RegionSig → Option String
    | [], [] => none
    | r :: rs, s :: ss =>
      match diagnoseRegion R Γ r s with
      | some e => some e
      | none => diagnoseRegions R Γ rs ss
    | _, _ => some "region count differs from the rule's"

  def diagnoseRegion (R : Rulebook sem ib) (Γ : TEnv) : Region → RegionSig → Option String
    | .mk [.mk args ops], sig =>
      if args.mapM (fun (a : Value) => vtyOf ib a.ty) == some sig.args then
        diagnoseOps R sig.depth ((args.map (fun (a : Value) => a.id)).zip sig.args ++ Γ) ops sig.yields
      else some "block arguments of other types than the rule's"
    | .mk _, _ => some "region without exactly one block"
end

/-- The first refusal in function `f`'s body; `none` when `supportCheck` accepts. -/
def diagnoseFunc (R : Rulebook sem ib) (m : Module) (f : String) (argTys resTys : List VTy) :
    Option String :=
  match m.func? f with
  | some fn => if fn.isDecl then some s!"@{f} is a declaration" else
    diagnoseRegion R [] fn.body ⟨argTys, resTys, 0⟩
  | none => some s!"no function @{f}"

end MerlinVerify.Check
