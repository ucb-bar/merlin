/- Provisional and frozen interpretations agree where the source extension is absent. The
check traverses every function body, including nested regions and internal callees. This bridge
permits reuse of existing proofs only after establishing that their interpretations coincide. -/
import MerlinVerify.Rules.W8A8SourceSemantics
import MerlinVerify.Sem.Defined

namespace MerlinVerify.Rules.W8A8SourceFree

open MerlinVerify

mutual
  def ops : List Op → Bool
    | [] => true
    | .mk name _ _ _ _ nested _ :: rest =>
      !W8A8SourceSemantics.isSource name && regions nested && ops rest
  def regions : List Region → Bool
    | [] => true
    | .mk body :: rest => blocks body && regions rest
  def blocks : List Block → Bool
    | [] => true
    | .mk _ body :: rest => ops body && blocks rest
end

def region : Region → Bool
  | .mk body => blocks body

def operation (op : Op) : Bool := !W8A8SourceSemantics.isSource op.name && regions op.regions

local notation "sourceSem" => W8A8SourceSemantics.opSem

mutual
  theorem ops_eq : ∀ body ctx call env, ops body = true →
      evalOps sourceSem ctx call env body = evalOps Sem.Ops.opSem ctx call env body
    | [], _, _, _, _ => rfl
    | .mk name results operands attrs props nested successors :: rest, ctx, call, env, checked => by
      simp only [ops, Bool.and_eq_true, Bool.not_eq_eq_eq_not, Bool.not_true] at checked
      obtain ⟨⟨ordinary, bodies⟩, remaining⟩ := checked
      rw [evalOps.eq_2, evalOps.eq_2]
      split
      · rfl
      · congr 1
        funext args
        rw [regions_eq nested call env bodies, W8A8SourceSemantics.unchanged
          (op := Op.mk name results operands attrs props nested successors) ordinary]
        congr 1
        funext values
        congr 1
        funext next
        exact ops_eq rest ctx call next remaining

  theorem regions_eq : ∀ body call env, regions body = true →
      evalRegions sourceSem call env body = evalRegions Sem.Ops.opSem call env body
    | [], _, _, _ => rfl
    | .mk body :: rest, call, env, checked => by
      simp only [regions, Bool.and_eq_true] at checked
      simp only [evalRegions]
      congr 1
      · funext ctx args
        exact blocks_eq body ctx call env args checked.1
      · exact regions_eq rest call env checked.2

  theorem blocks_eq : ∀ body ctx call env args, blocks body = true →
      evalBlocks sourceSem ctx call env body args = evalBlocks Sem.Ops.opSem ctx call env body args
    | [], _, _, _, _, _ => rfl
    | [.mk inputs body], ctx, call, env, args, checked => by
      simp only [blocks, Bool.and_true] at checked
      simp only [evalBlocks, evalBlock]
      congr 1
      funext next
      exact ops_eq body ctx call next checked
    | .mk _ _ :: _ :: _, _, _, _, _, _ => rfl
end

theorem region_eq {body : Region} (checked : region body = true)
    (ctx : Ctx) (call : CallSem) (env : Env) (args : List RVal) :
    evalRegion sourceSem ctx call env body args = evalRegion Sem.Ops.opSem ctx call env body args := by
  cases body with
  | mk body => exact blocks_eq body ctx call env args checked

def moduleCheck (m : Module) : Bool := m.funcs.all fun function => region function.body

/-- Internal calls use the same fuel and exact function lookup. Unselected callees cannot be
hidden from this check, and external declarations retain the same context interpretation. -/
theorem evalFunc_eq {m : Module} (checked : moduleCheck m = true) (ctx : Ctx) :
    ∀ fuel fn args, evalFunc sourceSem ctx m fuel fn args = evalFunc Sem.Ops.opSem ctx m fuel fn args := by
  intro fuel
  induction fuel with
  | zero => intros; rfl
  | succ fuel ih =>
    intro fn args
    simp only [evalFunc]
    cases found : m.func? fn with
    | none => rfl
    | some function =>
      dsimp only
      split
      · rfl
      · have calls : evalFunc sourceSem ctx m fuel = evalFunc Sem.Ops.opSem ctx m fuel := by
          funext name values
          exact ih name values
        rw [calls]
        exact region_eq (List.all_eq_true.1 checked function (List.mem_of_find?_eq_some found))
          ctx _ [] args

theorem denote_eq {m : Module} (checked : moduleCheck m = true) (ctx : Ctx)
    (fn : String) (args : List RVal) :
    denote sourceSem ctx m fn args = denote Sem.Ops.opSem ctx m fn args :=
  evalFunc_eq checked ctx _ fn args

theorem defined_iff {m : Module} (checked : moduleCheck m = true)
    {ctxOk : Ctx → Prop} {domain : List RVal → Prop} {fn : String} :
    FuncDefined sourceSem ctxOk domain m fn ↔ FuncDefined Sem.Ops.opSem ctxOk domain m fn := by
  simp only [FuncDefined, denote_eq checked]

theorem support_iff {m : Module} (checked : moduleCheck m = true)
    {ctxOk : Ctx → Prop} {domain : List RVal → Prop} {fn : String} :
    FuncNotStuck sourceSem ctxOk domain m fn ↔ FuncNotStuck Sem.Ops.opSem ctxOk domain m fn := by
  simp only [FuncNotStuck, denote_eq checked]

theorem refines_iff {before after : Module}
    (sourceChecked : moduleCheck before = true) (targetChecked : moduleCheck after = true)
    {ctxOk : Ctx → Prop} {domain : List RVal → Prop} {sourceFn targetFn : String} :
    FuncRefines sourceSem ctxOk domain after targetFn before sourceFn ↔
      FuncRefines Sem.Ops.opSem ctxOk domain after targetFn before sourceFn := by
  simp only [FuncRefines, denote_eq sourceChecked, denote_eq targetChecked]

#print axioms ops_eq
#print axioms regions_eq
#print axioms blocks_eq
#print axioms evalFunc_eq
#print axioms denote_eq
#print axioms defined_iff
#print axioms support_iff
#print axioms refines_iff

end MerlinVerify.Rules.W8A8SourceFree
