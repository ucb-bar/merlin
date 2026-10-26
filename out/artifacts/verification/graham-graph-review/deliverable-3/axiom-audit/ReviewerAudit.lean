/- Graham D3 reviewer-authored environment audit (independent of GraphStateAudit.lean).
For every constant defined in a module whose name starts with `Merlin`, `MerlinVerify`,
`GrahamD3` or one of the given root modules, report: axiom declarations, unsafe/partial
definitions, opaque constants, implemented_by/extern attributes, and the union of
axioms used by every theorem (kernel `collectAxioms`). Usage:
  LEAN_PATH=... lean --run ReviewerAudit.lean <Module> [<Module> ...] -/
import Lean
open Lean

def inScope (env : Environment) (roots : List Name) (name : Name) : Bool :=
  match env.getModuleIdxFor? name with
  | none => false
  | some idx =>
    match env.header.moduleNames[idx.toNat]? with
    | none => false
    | some m => (`Merlin).isPrefixOf m || (`MerlinVerify).isPrefixOf m || (`GrahamD3).isPrefixOf m || roots.contains m

def axiomsOf (env : Environment) (name : Name) : IO (Array Name) := do
  let (result, _) ← (collectAxioms name : CoreM (Array Name)).toIO
    {fileName := "<reviewer-audit>", fileMap := default, maxHeartbeats := 0} {env}
  return result

-- `unsafe` only because loading attribute extensions requires enabling initializers;
-- this is the audit driver, not a proof term.
unsafe def main (args : List String) : IO UInt32 := do
  initSearchPath (← findSysroot)
  enableInitializersExecution
  let roots := args.map String.toName
  let env ← importModules (roots.toArray.map fun m => {module := m}) {} (loadExts := true)
  let mut theorems := 0
  let mut defs := 0
  let mut findings : Array String := #[]
  let mut compilerAux := 0
  let mut used : Std.HashMap Name Nat := {}
  for (name, info) in env.constants.toList do
    if !inScope env roots name then continue
    -- Compiler-generated `_unsafe_rec` twins of SAFE recursive definitions are code-generation
    -- helpers; the kernel never unfolds them. Count them separately after checking the parent.
    let auxParentSafe : Bool := match name with
      | .str parent "_unsafe_rec" => match env.find? parent with
        | some (.defnInfo d) => d.safety == .safe
        | _ => false
      | _ => false
    if auxParentSafe then
      compilerAux := compilerAux + 1
      continue
    if info.isUnsafe then findings := findings.push s!"unsafe {name}"
    if info.isPartial then findings := findings.push s!"partial {name}"
    if (Compiler.implementedByAttr.getParam? env name).isSome then findings := findings.push s!"implemented_by {name}"
    if isExtern env name then findings := findings.push s!"extern {name}"
    match info with
    | .axiomInfo _ => findings := findings.push s!"axiom {name}"
    | .opaqueInfo _ => findings := findings.push s!"opaque {name}"
    | .thmInfo _ =>
      theorems := theorems + 1
      for ax in ← axiomsOf env name do
        used := used.insert ax (used.getD ax 0 + 1)
    | .defnInfo _ => defs := defs + 1
    | _ => pure ()
  let allowed := [``propext, ``Classical.choice, ``Quot.sound]
  let extra := used.toList.filter (fun (ax, _) => !allowed.contains ax)
  IO.println s!"roots: {args}"
  IO.println s!"modules_in_environment: {env.header.moduleNames.size}"
  IO.println s!"theorems_in_scope: {theorems}"
  IO.println s!"definitions_in_scope: {defs}"
  IO.println s!"axiom_use_counts: {used.toList.map (fun (a, n) => s!"{a}={n}")}"
  IO.println s!"non_std3_axioms: {extra.map (·.1)}"
  IO.println s!"compiler_unsafe_rec_aux_of_safe_defs: {compilerAux}"
  IO.println s!"findings: {findings.toList}"
  return if extra.isEmpty && findings.isEmpty then 0 else 1
