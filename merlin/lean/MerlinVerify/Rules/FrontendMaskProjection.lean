/- Checked projection of the actual capture; the cut location is an untrusted hint. -/
import MerlinVerify.Rules.FrontendMaskWhole

namespace MerlinVerify.Rules.FrontendMaskProjection
open MerlinVerify

/-- A projection carries proofs that its operations are exactly a contiguous slice of an
actual function block. It does not assert that the full model establishes the cut inputs. -/
structure Cut (m : Module) (name : String) (start : Nat) where
  source : Func
  found : m.func? name = some source
  block : Block
  region : source.body = .mk [block]
  args : List Value
  selected : List Op
  slice : selected = (block.ops.drop start).take 25
  outputs : List Value

def Cut.program {m : Module} {name : String} {start : Nat} (cut : Cut m name start) : Module :=
  {ops := [.mk "func.func" [] []
    [("function_type", .type (.function (cut.args.map (·.ty)) (cut.outputs.map (·.ty)))),
     ("sym_name", .str "subject")] []
    [.mk [.mk cut.args (cut.selected ++ [.mk "func.return" [] (cut.outputs.map (·.id)) [] [] [] []])]] []]}

/-- Roots, types, return values and the entire selected IR are derived from the authoritative
module. The whole-cut checker then checks every operation, edge, type and nested region. -/
def project? (m : Module) (name : String) (start : Nat) : Option (Cut m name start) :=
  match hf : m.func? name with
  | none => none
  | some f =>
    match hb : f.body with
    | .mk [.mk originalArgs originalOps] => do
      let selected := (originalOps.drop start).take 25
      if selected.length != 25 then none else do
        let [mask] := selected[0]!.operands | none
        let [values] := selected[2]!.operands | none
        let [seed] := selected[18]!.operands | none
        let [out] := selected[24]!.results | none
        let [_, count] := selected[23]!.results | none
        let defs ← FrontendBoolProgram.defsOps 64 m.ops
        let types := FrontendBoolProgram.sourceTypes defs
        let mt ← types.lookup mask
        let vt ← types.lookup values
        let st ← types.lookup seed
        pure {
          source := f
          found := hf
          block := .mk originalArgs originalOps
          region := hb
          args := [{id := mask, ty := mt}, {id := values, ty := vt}, {id := seed, ty := st}]
          selected := selected
          slice := rfl
          outputs := [out, count]}
    | _ => none

def check (m : Module) (name : String) (start : Nat) : Bool :=
  match project? m name start with
  | none => false
  | some cut => FrontendMaskWhole.check cut.program

/-- Accepted projections have an intrinsic source-slice correspondence and an acceptance-to-
execution theorem. The initialized cut contract is explicit; no full-function progress follows. -/
theorem check_sound {m : Module} {name : String} {start : Nat} (h : check m name start = true) :
    ∃ cut : Cut m name start, project? m name start = some cut ∧
      ∀ (limit : Nat) (base : FloatModel) (ctx : Ctx) (bits : Nat → Bool)
        (values seed : Nat → BitVec 64), ctx.indexBits = 64 → 1024 ≤ limit →
      ∃ out : Tensor,
        denote (FrontendMaskRuntime.sem limit) {ctx with M := FrontendF32Comparison.model base}
          cut.program "subject" (FrontendMaskWhole.inputs bits values seed) =
            .ok [.tensor out, FrontendMaskProgram.index (FrontendMaskBounds.countFrom bits 1024 0)] ∧
        out.shape = [1,1024] ∧ RVal.Defined (.tensor out) := by
  unfold check at h
  split at h
  · simp at h
  · next cut he =>
    exact ⟨cut, he, fun limit base ctx bits values seed hib hl =>
      FrontendMaskWhole.check_sound h limit base ctx bits values seed hib hl⟩

#print axioms check_sound
end MerlinVerify.Rules.FrontendMaskProjection
