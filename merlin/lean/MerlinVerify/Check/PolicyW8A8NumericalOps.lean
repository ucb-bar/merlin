/- Actual scalar operations lift the proved numerical registry into SSA inference. Every
numeric key, operand/result type, region and unmodelled attribute is checked from the source. -/
import MerlinVerify.Check.PolicyW8A8NumericalScalar

namespace MerlinVerify.Check.PolicyW8A8NumericalOps

open MerlinVerify Rules Definedness Sem.Ops PolicyW8A8NumericalFacts PolicyW8A8NumericalScalar

local notation "sourceSem" => W8A8SourceSemantics.opSem
local notation "allowed" => PolicyW8A8NumericalContext.allowed

theorem binary_exec (op : Op) (kind : BinaryKind)
    (modelled : unmodelledAttr? op.blind = none) (dialect : dialectOf op.name = "arith")
    (named : op.name = kind.name) (regions : op.regions = [])
    (result : Arith.resultElt op.blind = some (.float .f32))
    (key : Arith.floatKey op.blind = Spec.Dequant.defaultKey op.name)
    (ctx : Ctx) (call : CallSem) (env : Env) (x y : FBits .f32) :
    sourceSem ctx call op [.scalar (.some (.flt .f32 x)), .scalar (.some (.flt .f32 y))]
      (evalRegions sourceSem call env op.regions) =
      .ok [.scalar (.some (.flt .f32 (ctx.M.op (Spec.Dequant.defaultKey kind.name) .f32 [x, y])))] := by
  have ordinary : W8A8SourceSemantics.isSource op.name = false := by
    cases kind <;> simp [BinaryKind.name] at named <;>
      simp [W8A8SourceSemantics.isSource, W8A8SourceContract.name, named]
  rw [regions, evalRegions, W8A8SourceSemantics.unchanged ordinary, opSem_modelled modelled]
  simp only [dispatch, Op.blind_name, dialect]
  cases kind <;> simp only [BinaryKind.name] at named ⊢
  all_goals simp only [Arith.sem, Op.blind_name, named, Arith.binOp, Arith.pureN, result, Outcome.bindOp_eq]
  all_goals
    rw [show [.scalar (.some (.flt .f32 x)), .scalar (.some (.flt .f32 y))] =
      ([.some (.flt .f32 x), .some (.flt .f32 y)] : List Val).map RVal.scalar from rfl,
      liftEager_scalars]
    simp only [Arith.fltBin, ↓reduceDIte, Outcome.bind_ok, key, named]
    rfl

theorem rsqrt_exec (op : Op) (modelled : unmodelledAttr? op.blind = none)
    (dialect : dialectOf op.name = "math") (named : op.name = "math.rsqrt") (regions : op.regions = [])
    (result : Arith.resultElt op.blind = some (.float .f32))
    (key : Arith.floatKey op.blind = Spec.Dequant.defaultKey op.name)
    (ctx : Ctx) (call : CallSem) (env : Env) (x : FBits .f32) :
    sourceSem ctx call op [.scalar (.some (.flt .f32 x))] (evalRegions sourceSem call env op.regions) =
      .ok [.scalar (.some (.flt .f32 (ctx.M.op (Spec.Dequant.defaultKey "math.rsqrt") .f32 [x])))] := by
  have ordinary : W8A8SourceSemantics.isSource op.name = false := by
    simp [W8A8SourceSemantics.isSource, W8A8SourceContract.name, named]
  rw [regions, evalRegions, W8A8SourceSemantics.unchanged ordinary, opSem_modelled modelled]
  simp only [dispatch, Op.blind_name, dialect]
  simp [Arith.sem, Op.blind_name, named, result, Arith.pureN, Outcome.bindOp_eq,
    Arith.fltN, liftEager, key]

def select (name : String) (left right : Info) : Option BinaryContract :=
  binaryContracts.find? fun contract => contract.kind.name == name &&
    requirements left contract.left && requirements right contract.right

theorem select_sound {name : String} {left right : Info} {contract : BinaryContract}
    (checked : select name left right = some contract) :
    contract.kind.name = name ∧ requirements left contract.left = true ∧ requirements right contract.right = true := by
  have found := List.find?_some checked
  simpa only [Bool.and_eq_true, beq_iff_eq, and_assoc] using found

def binaryRule (_ : Refined.FEnv Info) (op : Op) (facts : List (Refined.Fact Info)) :
    Option (List (Refined.Fact Info)) :=
  if unmodelledAttr? op.blind = none ∧ dialectOf op.name = "arith" ∧ op.regions = [] ∧
      Arith.floatKey op.blind = Spec.Dequant.defaultKey op.name then
    match facts, Arith.resultElt op.blind with
    | [⟨⟨.scalar (.float .f32), true⟩, left⟩, ⟨⟨.scalar (.float .f32), true⟩, right⟩], some (.float .f32) => do
      let contract ← select op.name left right
      some [fact contract.output]
    | _, _ => none
  else none

theorem binaryRule_sound (ib : Nat) : ∀ Γ op facts outputs,
    binaryRule Γ op facts = some outputs → ∀ ctx call env args,
    ctx.indexBits = ib → allowed ctx call → Refined.EnvHolds model ib Γ env →
    Refined.FactsHold model ib args facts → ∃ values,
      sourceSem ctx call op args (evalRegions sourceSem call env op.regions) = .ok values ∧
      Refined.FactsHold model ib values outputs := by
  intro Γ op facts outputs checked ctx call env args _width permitted _valid holds
  unfold binaryRule at checked
  split at checked
  · next header =>
    obtain ⟨modelled, dialect, regions, key⟩ := header
    split at checked
    · next left right resultType =>
      cases chosen : select op.name left right with
      | none => simp [chosen] at checked
      | some contract =>
        simp only [chosen, bind, Option.bind, Option.some.injEq] at checked
        subst outputs
        obtain ⟨named, leftRequired, rightRequired⟩ := select_sound chosen
        obtain _ | ⟨leftHead, tail⟩ := holds
        obtain _ | ⟨rightHead, tail⟩ := tail
        cases tail
        obtain ⟨x, rfl⟩ := initialized_float leftHead.1
        obtain ⟨y, rfl⟩ := initialized_float rightHead.1
        obtain ⟨base, _laws, agreement, interpretation⟩ := permitted
        have execution := binary_exec op contract.kind modelled dialect named.symm regions resultType key ctx call env x y
        have keyAgree : (W8A8ByteModel.extend base).op (Spec.Dequant.defaultKey contract.kind.name) .f32 [x, y] =
            base.op (Spec.Dequant.defaultKey contract.kind.name) .f32 [x, y] := by
          cases contract.kind <;> simp [BinaryKind.name, W8A8ByteModel.extend, Spec.Dequant.defaultKey]
        rw [interpretation, keyAgree] at execution
        exact ⟨_, execution, .cons (result (contract.sound base agreement x y
          (requirements_sound leftRequired leftHead) (requirements_sound rightRequired rightHead))) .nil⟩
    · cases checked
  · cases checked

def rsqrtRule (_ : Refined.FEnv Info) (op : Op) (facts : List (Refined.Fact Info)) :
    Option (List (Refined.Fact Info)) :=
  if unmodelledAttr? op.blind = none ∧ dialectOf op.name = "math" ∧ op.name = "math.rsqrt" ∧
      op.regions = [] ∧ Arith.floatKey op.blind = Spec.Dequant.defaultKey op.name then
    match facts, Arith.resultElt op.blind with
    | [⟨⟨.scalar (.float .f32), true⟩, input⟩], some (.float .f32) =>
      if accepts input .nonnegative then some [fact .nonnegative] else none
    | _, _ => none
  else none

theorem rsqrtRule_sound (ib : Nat) : ∀ Γ op facts outputs,
    rsqrtRule Γ op facts = some outputs → ∀ ctx call env args,
    ctx.indexBits = ib → allowed ctx call → Refined.EnvHolds model ib Γ env →
    Refined.FactsHold model ib args facts → ∃ values,
      sourceSem ctx call op args (evalRegions sourceSem call env op.regions) = .ok values ∧
      Refined.FactsHold model ib values outputs := by
  intro Γ op facts outputs checked ctx call env args _width permitted _valid holds
  unfold rsqrtRule at checked
  split at checked
  · next header =>
    obtain ⟨modelled, dialect, named, regions, key⟩ := header
    split at checked
    · next input resultType =>
      split at checked
      · next accepted =>
        cases checked
        obtain _ | ⟨head, tail⟩ := holds
        cases tail
        obtain ⟨x, rfl⟩ := initialized_float head.1
        obtain ⟨base, _laws, agreement, interpretation⟩ := permitted
        have inputNN := requirement accepted head
        have numeric : W8A8IEEE32.Nonnegative x := by
          simpa [bitsCheck] using inputNN
        have execution := rsqrt_exec op modelled dialect named regions resultType key ctx call env x
        have numerical := W8A8IEEE32.rsqrt_nonnegative numeric
        have keyAgree : (W8A8ByteModel.extend base).op (Spec.Dequant.defaultKey "math.rsqrt") .f32 [x] =
            base.op (Spec.Dequant.defaultKey "math.rsqrt") .f32 [x] := by
          simp [W8A8ByteModel.extend, Spec.Dequant.defaultKey]
        rw [interpretation, keyAgree, agreement.rsqrt32 _ numerical.1] at execution
        exact ⟨_, execution, .cons (result (by simpa [bitsCheck] using numerical)) .nil⟩
      · cases checked
    · cases checked
  · cases checked

def book (ib : Nat) : Contextual.Book model sourceSem ib allowed :=
  ({ rule := binaryRule, sound := binaryRule_sound ib } : Contextual.Book model sourceSem ib allowed).orElse
    { rule := rsqrtRule, sound := rsqrtRule_sound ib }

#print axioms binaryRule_sound
#print axioms rsqrtRule_sound
#print axioms book

end MerlinVerify.Check.PolicyW8A8NumericalOps
