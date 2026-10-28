/- The typing checker also discharges the SSA literal-producer fact used by the rewrite.
A literal producer is evaluated from its exact attribute, independently of any float law. -/
import MerlinVerify.Rules.ScalarSquareTypingProgram

namespace MerlinVerify.Rules.ScalarSquareTyping
open MerlinVerify MerlinVerify.Check MerlinVerify.Sem.Ops ScalarSquareSSA
open MerlinVerify.Check.ScalarSquare

theorem literal_result {kind : FKind} {op : Op} {value : Value}
    (literal : literalTwo kind op value = true) (ctx : Ctx) (call : CallSem)
    (args : List RVal) (regions : List (Ctx → RegionSem)) :
    opSem ctx call op args regions =
      .ok [.scalar (.some (.flt kind (ScalarSquareContract.two kind)))] := by
  simp only [literalTwo, Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at literal
  have name : op.name = "arith.constant" := by tauto
  have attr : op.attr? "value" = some (.float (ScalarSquareContract.two kind).toNat (.float kind)) := by tauto
  have modelled : unmodelledAttr? op.blind = none := by tauto
  have single : ∃ result, op.results = [result] := by
    cases hr : op.results with
    | nil => simp [hr] at literal
    | cons result rest =>
      cases rest with
      | nil => exact ⟨result, rfl⟩
      | cons _ _ => simp [hr] at literal
  obtain ⟨result, hr⟩ := single
  rw [opSem_modelled modelled]
  simp only [dispatch, Op.blind_name, name, show dialectOf "arith.constant" = "arith" from by cbv,
    Arith.sem, Op.blind_attr?, attr, Op.blind_results, hr, List.map_cons, List.map_nil]
  cases kind <;> rfl

theorem result_facts {ib : Nat} {op : Op} {types : List VTy} {values : List RVal}
    (declared : op.results.mapM (fun r => vtyOf ib r.ty) = some types)
    (typed : HasTys ib values types) {ctx : Ctx} {call : CallSem}
    {args : List RVal} {regions : List (Ctx → RegionSem)}
    (executed : opSem ctx call op args regions = .ok values) : ResultFacts ib op values := by
  have arguments := (arguments_types op.results types declared values).mpr typed
  have constants : ∀ (result : Value) (v : RVal), v ∈ values → ∀ k,
      literalTwo k op ⟨result.id, .float k⟩ = true →
      v = .scalar (.some (.flt k (ScalarSquareContract.two k))) := by
    intro result v member k literal
    have same := (literal_result literal ctx call args regions).symm.trans executed
    have eq : values = [.scalar (.some (.flt k (ScalarSquareContract.two k)))] :=
      (Outcome.ok.inj same).symm
    simpa [eq] using member
  unfold ResultFacts
  have strengthen : ∀ (results : List Value) (vs : List RVal),
      Arguments ib results vs → (∀ v ∈ vs, v ∈ values) →
      List.Forall₂ (fun result v => Good ib result.id ⟨result, some op⟩ v) results vs := by
    intro results vs ha
    induction ha with
    | nil => intro _; exact .nil
    | @cons result v results vs head tail ih =>
      intro member
      exact .cons ⟨head, fun k producer hp hl => by
        cases hp
        exact constants result v (member v (by simp)) k hl⟩
        (ih fun v hv => member v (by simp [hv]))
  exact strengthen _ _ arguments (fun _ h => h)

#print axioms literal_result
#print axioms result_facts
end MerlinVerify.Rules.ScalarSquareTyping
