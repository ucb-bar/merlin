/- A checked continuation after actual mask selection. Every slice, observation,
extra capture and type is recomputed from the original exported function. -/
import MerlinVerify.Rules.FrontendRuntimeCheckpoint
import MerlinVerify.Rules.FrontendPrefixRanges
import MerlinVerify.Rules.FrontendProgramFrames
import MerlinVerify.Rules.FrontendRangeProjection

namespace MerlinVerify.Rules.FrontendPositionScope
open MerlinVerify MerlinVerify.Check
open FrontendTensorRanges FrontendProgramSegments

structure Scope (m : Module) (name : String) (stop count : Nat) where
  selection : FrontendRuntimeCheckpoint.Scope m name stop
  extras : List Value

def Scope.tail {m : Module} {name : String} {stop count : Nat} (s : Scope m name stop count) : List Op :=
  (s.selection.p.cut.block.ops.drop (stop + 25)).take count

def Scope.outputs {m : Module} {name : String} {stop count : Nat} (s : Scope m name stop count) : List Value :=
  s.selection.p.cut.outputs ++ ((s.tail.getLast?).map (·.results)).getD []

def Scope.observed {m : Module} {name : String} {stop count : Nat} (s : Scope m name stop count) : List Value :=
  s.selection.p.cut.args ++ s.extras

def Scope.preBody {m : Module} {name : String} {stop count : Nat} (s : Scope m name stop count) : Region :=
  .mk [.mk s.selection.p.cut.block.args ((s.selection.p.cut.block.ops.take stop) ++
    [returned (s.observed.map (·.id))])]

def Scope.tailBody {m : Module} {name : String} {stop count : Nat} (s : Scope m name stop count) : Region :=
  .mk [.mk [] (s.tail ++ [returned (s.outputs.map (·.id))])]

def Scope.body {m : Module} {name : String} {stop count : Nat} (s : Scope m name stop count) : Region :=
  .mk [.mk s.selection.p.cut.block.args ((s.selection.p.cut.block.ops.take (stop + 25 + count)) ++
    [returned (s.outputs.map (·.id))])]

/-- The execution proof composes exactly these source operations; no embedding operation is skipped. -/
theorem Scope.body_segments {m : Module} {name : String} {stop count : Nat} (s : Scope m name stop count) :
    s.body = .mk [.mk s.selection.p.cut.block.args ((s.selection.p.cut.block.ops.take stop) ++
      s.selection.p.cut.selected ++ s.tail ++ [returned (s.outputs.map (·.id))])] := by
  simp only [Scope.body, Scope.tail, s.selection.p.cut.slice, ← List.take_add]


def selectedFacts : List Fact :=
  [⟨⟨.tensor [1,1024] (.int 64), true⟩, some ⟨0,1024⟩⟩, ⟨⟨.scalar .index, true⟩, none⟩]

def Scope.preFacts? {m : Module} {name : String} {stop count : Nat} (s : Scope m name stop count) :
    Option (List Fact) :=
  (s.extras.mapM fun v => vtyOf 64 v.ty).map fun ts =>
    FrontendPrefixRanges.expected 1024 ++ ts.map (fun ty => Definedness.Refined.emptyFact model ⟨ty,true⟩)

def Scope.resultFacts? {m : Module} {name : String} {stop count : Nat} (s : Scope m name stop count) :
    Option (List Fact) :=
  (((s.tail.getLast?).map (·.results)).getD []).mapM (fun v => vtyOf 64 v.ty) |>.map fun ts =>
    selectedFacts ++ ts.map (fun ty => Definedness.Refined.emptyFact model ⟨ty,true⟩)

def Scope.preEnv {m : Module} {name : String} {stop count : Nat} (s : Scope m name stop count) (fs : List Fact) : FEnv :=
  (s.observed.map (·.id)).zip fs

def Scope.postEnv {m : Module} {name : String} {stop count : Nat} (s : Scope m name stop count) (fs : List Fact) : FEnv :=
  (s.selection.p.cut.outputs.map (·.id)).zip selectedFacts ++ s.preEnv fs


def project? (m : Module) (name : String) (stop count : Nat) : Option (Scope m name stop count) := do
  let selection ← FrontendRuntimeCheckpoint.project? m name stop
  let tail := (selection.p.cut.block.ops.drop (stop + 25)).take count
  if count = 0 ∨ tail.length ≠ count then none else do
    let localDefs ← FrontendBoolProgram.defsOps 64 tail
    let allDefs ← FrontendBoolProgram.defsOps 64 m.ops
    let types := FrontendBoolProgram.sourceTypes allDefs
    let outputs := selection.p.cut.outputs ++ ((tail.getLast?).map (·.results)).getD []
    let uses := FrontendRangeProjection.operands (tail ++ [returned (outputs.map (·.id))])
    let free := (uses.filter fun id => !(localDefs.any (fun v => v.id == id))).eraseDups
    let old := (selection.p.cut.args ++ selection.p.cut.outputs).map (·.id)
    let extras ← (free.filter fun id => !old.contains id).mapM fun id => do
      let ty ← types.lookup id
      pure ({id := id, ty := ty} : Value)
    pure ⟨selection, extras⟩

def Scope.Executes {m : Module} {name : String} {stop count : Nat} (s : Scope m name stop count) : Prop :=
  ∃ a rs, s.selection.p.cut.block.args.mapM (fun v => vtyOf 64 v.ty) = some a ∧
    s.resultFacts? = some rs ∧
    ∀ (limit : Nat) (base : FloatModel) (ctx : Ctx) (call : CallSem) (args : List RVal),
      ctx.indexBits = 64 → 1024 ≤ limit → HasTys 64 args a → ResultsDefined args →
      ∃ results, evalRegion (FrontendMaskRuntime.sem limit) {ctx with M := FrontendF32Comparison.model base}
        call [] s.body args = .ok results ∧ Definedness.Refined.FactsHold model 64 results rs

/-- The prefix establishes every fact used by the continuation; selected operations
must preserve those exact captured SSA bindings. No range is assumed at a cut input. -/
def Scope.check {m : Module} {name : String} {stop count : Nat} (s : Scope m name stop count) : Bool :=
  FrontendMaskWhole.check s.selection.p.cut.program &&
  FrontendBoolSimulation.region [] [(50,s.selection.mask.id),(51,s.selection.values.id),(52,s.selection.seed.id)]
    FrontendRuntimeCheckpoint.bareSource s.selection.bare &&
  (s.selection.p.cut.block.ops.take stop).all (fun op => !op.isTerminator) &&
  s.selection.p.cut.selected.all (fun op => !op.isTerminator) &&
  s.tail.all (fun op => !op.isTerminator) &&
  FrontendRuntimeBridge.regions [s.preBody, s.tailBody] &&
  match s.selection.p.cut.block.args.mapM (fun v => vtyOf 64 v.ty), s.preFacts?, s.resultFacts? with
  | some a, some fs, some rs =>
    match FrontendRangeFrame.inferRegion (FrontendRangeDefined.frameBook 64 []) [] s.preBody
        (a.map fun ty => Definedness.Refined.emptyFact model ⟨ty,true⟩),
      FrontendRangeFrame.inferRegion (FrontendRangeDefined.frameBook 64 []) (s.postEnv fs) s.tailBody [] with
    | some prefixFacts, some resultFacts =>
      FrontendRangeFacts.fits prefixFacts fs &&
      s.selection.p.cut.selected.all (FrontendProgramFrames.preserves (s.preEnv fs)) &&
      FrontendRangeFacts.fits resultFacts rs
    | _, _ => false
  | _, _, _ => false

def check (m : Module) (name : String) (stop count : Nat) : Bool :=
  match project? m name stop count with | some s => s.check | none => false
end MerlinVerify.Rules.FrontendPositionScope
