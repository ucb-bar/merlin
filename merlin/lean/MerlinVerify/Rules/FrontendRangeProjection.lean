/- Bounds-aware execution of an actual contiguous source slice. The contract is
explicitly a cut-input contract; it never asserts execution of an omitted prefix. -/
import MerlinVerify.Rules.FrontendRangeDefined
import MerlinVerify.Rules.FrontendProgramSegments

namespace MerlinVerify.Rules.FrontendRangeProjection
open MerlinVerify MerlinVerify.Check

mutual
  def operands : List Op → List ValueId
    | [] => []
    | .mk _ _ xs _ _ rs _ :: rest => xs ++ regionOperands rs ++ operands rest
  def regionOperands : List Region → List ValueId
    | [] => []
    | .mk bs :: rest => blockOperands bs ++ regionOperands rest
  def blockOperands : List Block → List ValueId
    | [] => []
    | .mk _ ops :: rest => operands ops ++ blockOperands rest
end

structure Cut (m : Module) (name : String) (start count : Nat) where
  source : Func
  found : m.func? name = some source
  block : Block
  region : source.body = .mk [block]
  selected : List Op
  slice : selected = (block.ops.drop start).take count
  args : List Value
  outputs : List Value

def Cut.body {m : Module} {name : String} {start count : Nat} (cut : Cut m name start count) : Region :=
  .mk [.mk cut.args (cut.selected ++ [FrontendProgramSegments.returned (cut.outputs.map (·.id))])]

def project? (m : Module) (name : String) (start count : Nat) : Option (Cut m name start count) :=
  match hf : m.func? name with
  | none => none
  | some f =>
    match hb : f.body with
    | .mk [.mk originalArgs originalOps] => do
      let selected := (originalOps.drop start).take count
      if count = 0 ∨ selected.length ≠ count then none else do
        let localDefs ← FrontendBoolProgram.defsOps 64 selected
        let allDefs ← FrontendBoolProgram.defsOps 64 m.ops
        let types := FrontendBoolProgram.sourceTypes allDefs
        let free := ((operands selected).filter fun id => !(localDefs.any (fun v => v.id == id))).eraseDups
        let args ← free.mapM fun id => do
          let ty ← types.lookup id
          pure ({id := id, ty := ty} : Value)
        let last ← selected.getLast?
        pure {
          source := f
          found := hf
          block := .mk originalArgs originalOps
          region := hb
          selected := selected
          slice := rfl
          args := args
          outputs := last.results}
    | _ => none

def Cut.check {m : Module} {name : String} {start count : Nat} (cut : Cut m name start count)
    (limit : Nat) : Bool :=
  match cut.args.mapM (fun v => vtyOf 64 v.ty), cut.outputs.mapM (fun v => vtyOf 64 v.ty) with
  | some a, some r => cut.selected.all (fun op => !op.isTerminator) &&
      Definedness.Refined.checkRegion (FrontendRangeDefined.candidateBook limit 64) [] cut.body
        (a.map fun ty => Definedness.Refined.emptyFact FrontendTensorRanges.model ⟨ty, true⟩) r
  | _, _ => false

def check (m : Module) (name : String) (start count limit : Nat) : Bool :=
  match project? m name start count with | none => false | some cut => cut.check limit

/-- All selected operations are the actual source operations, in order. Typed initialized
cut inputs suffice; internal element bounds are derived by their checked producers. -/
theorem check_sound {m : Module} {name : String} {start count limit : Nat}
    (h : check m name start count limit = true) :
    ∃ cut : Cut m name start count, project? m name start count = some cut ∧
      ∃ a r, cut.args.mapM (fun v => vtyOf 64 v.ty) = some a ∧
        cut.outputs.mapM (fun v => vtyOf 64 v.ty) = some r ∧
        ∀ (ctx : Ctx) (call : CallSem) (args : List RVal), ctx.indexBits = 64 →
          HasTys 64 args a → ResultsDefined args →
          ∃ results, evalRegion (FrontendMaskRuntime.sem limit) ctx call [] cut.body args = .ok results ∧
            HasTys 64 results r ∧ ResultsDefined results := by
  unfold check at h
  split at h
  · cases h
  · next cut hc =>
    unfold Cut.check at h
    split at h
    · next a r ha hr =>
      simp only [Bool.and_eq_true] at h
      refine ⟨cut, hc, a, r, ha, hr, ?_⟩
      intro ctx call args hib ht hd
      exact Definedness.Refined.checkRegion_sound (FrontendRangeDefined.candidateBook limit 64) h.2 hib
        (by intro _ _ h; cases h) (Definedness.Refined.inputFacts FrontendTensorRanges.model ht hd)
    · cases h

#print axioms check_sound
end MerlinVerify.Rules.FrontendRangeProjection
