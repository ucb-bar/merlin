import MerlinVerify.Check.Module
import MerlinVerify.Spec.FrontendIntent
import MerlinVerify.Rules.FrontendBoolExecution

/-!
An independently reconstructed frontend-intent program for the supported f32 -> i1 repair.
The input is the decoded authoritative MLIR AST, never a Python-produced candidate record.
Other float formats are refused until A2 supplies their bit-level intent.
-/
namespace MerlinVerify.Rules.FrontendBoolProgram

open MerlinVerify MerlinVerify.Check

/-- Refuse nesting deeper than the explicit checker limit; never truncate the source inventory. -/
def defsOps : Nat → List Op → Option (List Value)
  | 0, _ => none
  | fuel + 1, ops => do
    let groups ← ops.mapM fun o => do
      let regions ← o.regions.mapM fun r => do
        let blocks ← r.blocks.mapM fun b => do
          let children ← defsOps fuel b.ops
          pure (b.args ++ children)
        pure blocks.flatten
      pure (o.results ++ regions.flatten)
    pure groups.flatten

def sourceTypes (defs : List Value) : List (ValueId × Ty) :=
  defs.map fun v => (v.id, v.ty)

def freshStart (defs : List Value) : Nat :=
  defs.foldl (fun n v => max n v.id) 0 + 1

def provenance (o : Op) : List (String × Attr) := Id.run do
  let attributes := o.attrs.filter fun (k, _) => k.startsWith "prov."
  let previous := match attributes.lookup "prov.transforms" with
    | some (Attr.str s) => s.splitOn "," | _ => []
  let chain := (previous.filter (· != "")) ++ ["bool_fptosi_cmpf_une"]
  let kept := attributes.filter fun (k, _) => k != "prov.transforms"
  return (kept ++ [("prov.transforms", Attr.str (String.intercalate "," chain))]).mergeSort
    (fun a b => a.1 ≤ b.1)

def legalSource (types : List (ValueId × Ty)) (o : Op) : Option (ValueId × Value) := do
  if o.name != "arith.fptosi" && o.name != "arith.fptoui" then none else
  let [src] := o.operands | none
  let [result] := o.results | none
  if result.ty != .int 1 || types.lookup src != some (.float .f32) then none else
  if !o.regions.isEmpty || !o.successors.isEmpty || !o.props.isEmpty then none else
  if !o.attrs.all (fun (k, _) => k.startsWith "prov.") then none else
  pure (src, result)

def replacementCmp (o : Op) (src : ValueId) (result : Value) (zeroId : ValueId) : Op :=
  .mk "arith.cmpf" [result] [src, zeroId]
      ([("fastmath", .enum "arith" "fastmath" "none"),
        ("predicate", .int 13 (.int 64))] ++ provenance o) [] [] []

def replacement (o : Op) (src : ValueId) (result : Value) (zeroId : ValueId) : List Op :=
  [FrontendBoolExecution.zero zeroId, replacementCmp o src result zeroId]

/-- Check the interpreted key of every marked replacement in the reconstructed program.
This condition is evaluated on Lean's decoded IR, including provenance-bearing comparisons. -/
def repairMarked (o : Op) : Bool :=
  match o.attr? "prov.transforms" with
  | some (.str s) => (s.splitOn ",").contains "bool_fptosi_cmpf_une"
  | _ => false

def repairReady (o : Op) : Bool :=
  if repairMarked o then
    decide (o.name = "arith.cmpf" ∧
      Sem.Ops.dialectOf o.name = "arith" ∧
      Sem.Ops.unmodelledAttr? o.blind = none ∧
      Sem.Ops.Arith.resultElt o.blind = some (.int 1) ∧
      Sem.Ops.Arith.floatKey o.blind = FrontendBoolFptosi.uneKey)
  else true

def repairsReady : Nat → List Op → Bool
  | 0, _ => false
  | fuel + 1, ops => ops.all fun o =>
      repairReady o && o.regions.all fun r =>
        r.blocks.all fun b => repairsReady fuel b.ops

theorem ready_of_mem {fuel : Nat} {ops : List Op} {o : Op}
    (h : repairsReady (fuel + 1) ops = true) (ho : o ∈ ops) : repairReady o = true := by
  simp only [repairsReady, List.all_eq_true] at h
  have hs := h o ho
  simp only [Bool.and_eq_true] at hs
  exact hs.1

inductive Occurs (o : Op) : List Op → Prop where
  | here {ops : List Op} : o ∈ ops → Occurs o ops
  | child {ops : List Op} {parent : Op} {r : Region} {b : Block} :
      parent ∈ ops → r ∈ parent.regions → b ∈ r.blocks →
      Occurs o b.ops → Occurs o ops

theorem ready_of_occurs (o : Op) : ∀ fuel ops,
    repairsReady fuel ops = true → Occurs o ops → repairReady o = true
  | 0, _, h, _ => by simp [repairsReady] at h
  | fuel + 1, ops, h, hocc => by
    cases hocc with
    | here ho => exact ready_of_mem h ho
    | child hp hr hb hc =>
      rename_i parent r b
      simp only [repairsReady, List.all_eq_true] at h
      have hparent := h parent hp
      simp only [Bool.and_eq_true, List.all_eq_true] at hparent
      exact ready_of_occurs o fuel _ ((hparent.2 r hr) b hb) hc

theorem marked_ready_facts {o : Op} (hm : repairMarked o = true)
    (hr : repairReady o = true) :
    o.name = "arith.cmpf" ∧ Sem.Ops.dialectOf o.name = "arith" ∧
    Sem.Ops.unmodelledAttr? o.blind = none ∧
    Sem.Ops.Arith.resultElt o.blind = some (.int 1) ∧
    Sem.Ops.Arith.floatKey o.blind = FrontendBoolFptosi.uneKey := by
  simpa only [repairReady, hm, ↓reduceIte, decide_eq_true_eq] using hr

/-- One generated site executes A2's scalar result before any following operations, including
inside a tensor body's region. `repairMarked` and `repairReady` are checked in `rewriteOps` and
again across the reconstructed module by `check`; the comparison law remains explicit. -/
theorem replacement_exec (ctx : Ctx) (call : CallSem) (env : Env) (tail : List Op)
    (source z : Nat) (result : Value) (o : Op) (x : Option (FBits .f32))
    (hty : result.ty = .int 1)
    (hmark : repairMarked (replacementCmp o source result z) = true)
    (hready : repairReady (replacementCmp o source result z) = true)
    (hune : FrontendBoolFptosi.UneCorrect ctx.M)
    (hsource : env.get? source = some (.scalar (FrontendBoolExecution.sourceVal x)))
    (hfresh : source ≠ z) :
    evalOps Sem.Ops.opSem ctx call env (replacement o source result z ++ tail) =
      evalOps Sem.Ops.opSem ctx call
        ((result.id, .scalar (Spec.FrontendIntent.f32ToBool x)) ::
         (z, .scalar (.some (.flt .f32 0))) :: env) tail := by
  obtain ⟨hn, hd, hm, hr, hk⟩ := marked_ready_facts hmark hready
  have hresult : (replacementCmp o source result z).results =
      [{ id := result.id, ty := .int 1 }] := by
    cases result with
    | mk id ty =>
      change ty = .int 1 at hty
      subst ty
      rfl
  simpa [replacement] using FrontendBoolExecution.checked_site_then_tail
    ctx call env tail (replacementCmp o source result z) source z result.id x
    hm hd hn hr hk (by rfl) hresult (by rfl) hune hsource hfresh

def rewriteOps (types : List (ValueId × Ty)) : Nat → List Op → StateT Nat Option (List Op)
  | 0, _ => failure
  | fuel + 1, ops => do
    let groups ← ops.mapM fun o => do
      if o.name == "arith.fptosi" || o.name == "arith.fptoui" then
        if o.results.any (fun r => r.ty == .int 1) then
          let some (src, result) := legalSource types o | failure
          let zeroId ← get
          if !(repairMarked (replacementCmp o src result zeroId) &&
              repairReady (replacementCmp o src result zeroId)) then failure
          modify (· + 1)
          return replacement o src result zeroId
      let regions ← o.regions.mapM fun r => do
        let blocks ← r.blocks.mapM fun b => do
          let children ← rewriteOps types fuel b.ops
          pure (.mk b.args children : Block)
        pure (.mk blocks : Region)
      match o with
      | .mk name results operands attrs props _ successors =>
        return [.mk name results operands attrs props regions successors]
    pure groups.flatten

/-- A2 intent materialization. Return `none` for unsupported format/attributes, duplicate source
definitions, or a boundary without a repaired site. The new zero IDs are globally fresh. -/
def intent? (src : Module) : Option Module := do
  let defs ← defsOps 64 src.ops
  if !decide (defs.map (·.id)).Nodup then none else
  let start := freshStart defs
  let (ops, finish) ← (rewriteOps (sourceTypes defs) 64 src.ops).run start
  if finish == start then none else
  pure { src with ops }

/-- Exact structural correspondence is required for the entire module, including every nested
region and every result use. `moduleCheck` supplies the existing evaluator theorem on every
defined function; `outsideChanges` accounts for top-level attributes. -/
def check (src tgt : Module) : Bool :=
  match intent? src with
  | none => false
  | some intended => repairsReady 64 intended.ops && alphaOps [] intended.ops tgt.ops &&
      moduleCheck intended tgt && (outsideChanges intended tgt).unverified.isEmpty

/-- Acceptance is a whole-program execution claim for every defined function. It deliberately
compares the target with `Intent(src)`, not the poison-producing captured cast. The additional
`alphaOps` conjunct proves exact operation, operand, type, attribute, and SSA correspondence
through nested regions, so the target cannot contain an unrelated change. -/
theorem check_sound {src tgt intended : Module} (hi : intent? src = some intended)
    (h : check src tgt = true) :
    repairsReady 64 intended.ops = true ∧ alphaOps [] intended.ops tgt.ops = true ∧
      ∀ f ∈ Module.definedNames intended,
        FuncRefinesC Sem.Ops.opSem (fun _ => True) (fun _ => True) tgt f intended f := by
  simp only [check, hi, Bool.and_eq_true] at h
  exact ⟨h.1.1.1, h.1.1.2, moduleCheck_sound h.1.2⟩

/-- At any marked generated site in the accepted program, the actual evaluator binds A2's
poison-aware f32 bool result before executing the rest of the surrounding block. The occurrence
premise identifies the site in the independently reconstructed intent module, including nested
regions. -/
theorem accepted_site_exec {src tgt intended : Module}
    (hi : intent? src = some intended) (h : check src tgt = true)
    (ctx : Ctx) (call : CallSem) (env : Env) (tail : List Op)
    (o : Op) (source z : Nat) (result : Value) (x : Option (FBits .f32))
    (hocc : Occurs (replacementCmp o source result z) intended.ops)
    (hmark : repairMarked (replacementCmp o source result z) = true)
    (hty : result.ty = .int 1)
    (hune : FrontendBoolFptosi.UneCorrect ctx.M)
    (hsource : env.get? source = some (.scalar (FrontendBoolExecution.sourceVal x)))
    (hfresh : source ≠ z) :
    evalOps Sem.Ops.opSem ctx call env (replacement o source result z ++ tail) =
      evalOps Sem.Ops.opSem ctx call
        ((result.id, .scalar (Spec.FrontendIntent.f32ToBool x)) ::
         (z, .scalar (.some (.flt .f32 0))) :: env) tail := by
  have hr := (check_sound hi h).1
  exact replacement_exec ctx call env tail source z result o x hty hmark
    (ready_of_occurs _ 64 intended.ops hr hocc) hune hsource hfresh

#print axioms check_sound
#print axioms accepted_site_exec

end MerlinVerify.Rules.FrontendBoolProgram
