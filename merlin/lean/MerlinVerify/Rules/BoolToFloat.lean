/- Additive f32 frontend-intent proposal. The primitive is A2's boolToF32; its approval
   is separate from the previously approved float-to-bool repair. No frozen file is changed. -/
import MerlinVerify.Rules.FrontendBoolSimulation
import Init.Data.Float.Model.Float32
import Lean.Elab.Tactic.Omega

namespace MerlinVerify.Rules.BoolToFloat
open MerlinVerify MerlinVerify.Check
open FrontendBoolProgram

abbrev Types := List (ValueId × Ty)

def unsignedKey : FloatOpKey :=
  Sem.Ops.Arith.floatKey (.mk "arith.uitofp" [] [] [] [] [] [])

/-- Precisely the two unsigned i1 conversions. Poison is handled outside FloatModel. -/
def UnsignedBoolCorrect (M : FloatModel) : Prop :=
  ∀ x : BitVec 1, M.ofInt unsignedKey 1 .f32 x =
    if x.toNat = 0 then 0 else 0x3f800000

/-- Use Lean's pre-existing IEEE ofNat, rather than a host cast or a postulated conversion law.
    Only this key, width and format are overridden; other conversions remain parametric. -/
def model (base : FloatModel) : FloatModel :=
  { base with ofInt := fun key w k x =>
      if key = unsignedKey ∧ w = 1 then
        match k with
        | .f32 => (Float32.Model.ofNat x.toNat).toBits.toBitVec
        | k => base.ofInt key w k x
      else base.ofInt key w k x }

theorem ofNat_zero : (Float32.Model.ofNat 0).toBits.toBitVec = (0 : BitVec 32) := by decide
theorem ofNat_one : (Float32.Model.ofNat 1).toBits.toBitVec = (0x3f800000 : BitVec 32) := by decide

theorem model_correct (base : FloatModel) : UnsignedBoolCorrect (model base) := by
  intro x
  have hx := x.isLt
  have hc : x.toNat = 0 ∨ x.toNat = 1 := by omega
  rcases hc with h | h
  · simp only [model, and_self, ↓reduceIte]
    change (Float32.Model.ofNat x.toNat).toBits.toBitVec = _
    rw [h, ofNat_zero]; rfl
  · simp only [model, and_self, ↓reduceIte]
    change (Float32.Model.ofNat x.toNat).toBits.toBitVec = _
    rw [h, ofNat_one]; rfl

/-- The independently stated primitive on well-typed inputs; an explicit off-domain extension
    uses the unchanged parametric unsigned conversion on integers of other widths. This extension
    permits exact stuck/UB correspondence without claiming totality of unchanged computations. -/
def scalar (M : FloatModel) : Val → Outcome Val
  | .poison => .ok (Spec.FrontendIntent.boolToF32 none)
  | .some (.bits 1 x) =>
      .ok (Spec.FrontendIntent.boolToF32 (some (x.toNat != 0)))
  | .some (.bits w x) => .ok (.some (.flt .f32 (M.ofInt unsignedKey w .f32 x)))
  | .some (.flt ..) => .stuck "*itofp of float"

theorem scalar_correct (M : FloatModel) (h : UnsignedBoolCorrect M) (v : Val) :
    (.ok (match v with
      | .poison => .poison
      | .some (.bits w x) => .some (.flt .f32 (M.ofInt unsignedKey w .f32 x))
      | .some (.flt ..) => .poison) : Outcome Val) =
    (match v with | .some (.flt ..) => .ok .poison | _ => scalar M v) := by
  cases v with
  | poison => rfl
  | some a =>
    cases a with
    | flt k x => rfl
    | bits w x =>
      cases w with
      | zero => rfl
      | succ w =>
        cases w with
        | zero => simp [scalar, Spec.FrontendIntent.boolToF32, h x]
        | succ w => rfl

def cast (M : FloatModel) (args : List RVal) : Outcome (List RVal) :=
  match args with
  | [a] => do
    let v ← liftEager (.float .f32) (fun | [x] => scalar M x | _ => .stuck "arity") [a]
    pure [v]
  | _ => .stuck "arith.uitofp: expected 1 operand"

/-- Static source eligibility is recomputed from the actual source inventory. Scalar i1-to-f32
    is supported; tensor casts and f16/bf16/f64 are refused by the complete-boundary inventory. -/
def legalSource (types : Types) (o : Op) : Option (ValueId × Value) := do
  if o.name != "arith.sitofp" then none else
  let [src] := o.operands | none
  let [result] := o.results | none
  if result.ty != .float .f32 || types.lookup src != some (.int 1) then none else
  if !o.regions.isEmpty || !o.successors.isEmpty || !o.props.isEmpty then none else
  if !o.attrs.all (fun (k, _) => k.startsWith "prov.") then none else
  pure (src, result)

def sem (types : Types) : OpSem := fun ctx call op args regions =>
  match legalSource types op with
  | none => Sem.Ops.opSem ctx call op args regions
  | some _ => cast ctx.M args

def CastFacts (op : Op) : Prop :=
  Sem.Ops.unmodelledAttr? op.blind = none ∧
  Sem.Ops.dialectOf op.name = "arith" ∧ op.name = "arith.uitofp" ∧
  Sem.Ops.Arith.resultElt op.blind = some (.float .f32) ∧
  Sem.Ops.Arith.floatKey op.blind = unsignedKey ∧ hasUnit op.blind "nonNeg" = false
instance (op : Op) : Decidable (CastFacts op) := inferInstanceAs (Decidable (_ ∧ _))

theorem cast_exec (ctx : Ctx) (call : CallSem) (op : Op) (hf : CastFacts op)
    (h : UnsignedBoolCorrect ctx.M) (args : List RVal) (regions : List (Ctx → RegionSem)) :
    Sem.Ops.opSem ctx call op args regions = cast ctx.M args := by
  obtain ⟨hm, hd, hn, hr, hk, hneg⟩ := hf
  rw [Check.opSem_modelled hm]
  simp only [Sem.Ops.dispatch, Check.Op.blind_name, hd]
  simp only [Sem.Ops.Arith.sem, Check.Op.blind_name, hn, hr, hk, hneg,
    Bool.and_false, Bool.false_and, Bool.false_eq_true, ↓reduceIte]
  have heq : (fun v : Val => match v with
      | .some (.bits w x) => (.ok (.some (.flt .f32 (ctx.M.ofInt unsignedKey w .f32 x))) : Outcome Val)
      | .poison => .ok .poison
      | _ => .stuck "*itofp of float") = scalar ctx.M := by
    funext v
    cases v with
    | poison => rfl
    | some a =>
      cases a with
      | flt k x => rfl
      | bits w x => exact scalar_correct ctx.M h (.some (.bits w x))
  trans Sem.Ops.Arith.unOp op.blind args (scalar ctx.M)
  · exact congrArg (Sem.Ops.Arith.unOp op.blind args) heq
  · cases args with
    | nil => simp [Sem.Ops.Arith.unOp, cast, Check.Op.blind_name, hn]; rfl
    | cons a rest => cases rest with
      | nil => simp only [Sem.Ops.Arith.unOp, Sem.Ops.Arith.pureN, hr, cast, Outcome.bind_ok]; rfl
      | cons b rest => simp [Sem.Ops.Arith.unOp, cast, Check.Op.blind_name, hn]; rfl
theorem poison (M : FloatModel) : scalar M .poison = .ok (Spec.FrontendIntent.boolToF32 none) := rfl
theorem false_positive_zero (M : FloatModel) :
    scalar M (.some (.bits 1 0)) = .ok (.some (.flt .f32 0)) := rfl
theorem true_positive_one (M : FloatModel) :
    scalar M (.some (.bits 1 1)) = .ok (.some (.flt .f32 0x3f800000)) := rfl

end MerlinVerify.Rules.BoolToFloat
