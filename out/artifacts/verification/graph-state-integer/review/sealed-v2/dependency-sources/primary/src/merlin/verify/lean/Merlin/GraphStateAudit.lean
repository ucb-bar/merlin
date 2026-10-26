/- Standalone transitive claim audit. This inspects kernel proof objects; it does
not use native reflection to discharge a theorem or require Lean-MLIR. -/
import Lean

open Lean

def graphAllowedAxioms : List Name := [``propext,``Classical.choice,``Quot.sound]

def graphCompilerAux (env : Environment) (name : Name) : Bool :=
  match name with
  | .str parent "_unsafe_rec" => match env.find? parent with
    | some (.defnInfo definition) => definition.safety == .safe
    | _ => false
  | _ => false

def graphProject (env : Environment) (roots : List Name) (name : Name) : Bool :=
  (`Merlin).isPrefixOf name ||
    (((env.getModuleIdxFor? name).bind (env.header.modules[·]?)).map (·.module)).any
      (fun moduleName => (`Merlin).isPrefixOf moduleName || roots.contains moduleName)

def graphAxioms (env : Environment) (name : Name) : IO (Array Name) := do
  let (axioms,_) ← (collectAxioms name : CoreM (Array Name)).toIO
    {fileName := "<graph-state-audit>",fileMap := default,maxHeartbeats := 0} {env}
  return axioms

def main (args : List String) : IO UInt32 := do
  initSearchPath (←findSysroot)
  let (statementNames,modules) := match args with
    | "--statements"::names::modules => (names.splitOn ",",modules)
    | modules => ([],modules)
  if modules.isEmpty then
    IO.eprintln "expected selected compiled graph claim modules"
    return 2
  let roots := modules.map String.toName
  let env ← importModules (roots.toArray.map fun name => {module := name}) {} (loadExts := false)
  if !statementNames.isEmpty then
    let mut statements : List (String × Json) := []
    for name in statementNames do
      let some declaration := env.find? name.toName | throw (IO.userError s!"missing claim {name}")
      statements := statements ++ [(name,toJson (toString declaration.type))]
    IO.println (Json.mkObj [("statements",Json.mkObj statements)]).compress
    return 0
  let mut violations : Array String := #[]
  let mut theorems := 0
  let mut used : Array String := #[]
  for (name,declaration) in env.constants.toList do
    if !graphProject env roots name then continue
    if declaration.isUnsafe && !graphCompilerAux env name then
      violations := violations.push s!"unsafe:{name}"
    if declaration.isPartial && !graphCompilerAux env name then
      violations := violations.push s!"partial:{name}"
    if (Compiler.implementedByAttr.getParam? env name).isSome then
      violations := violations.push s!"implemented_by:{name}"
    if isExtern env name then violations := violations.push s!"extern:{name}"
    match declaration with
    | .axiomInfo _ => violations := violations.push s!"project axiom:{name}"
    | .thmInfo _ =>
      theorems := theorems+1
      for axiomName in ←graphAxioms env name do
        if !used.contains axiomName.toString then used := used.push axiomName.toString
        if !graphAllowedAxioms.contains axiomName then
          violations := violations.push s!"unapproved axiom:{name}:{axiomName}"
    | _ => pure ()
  let imports ← env.header.modules.toList.mapM fun entry => do
    return Json.mkObj [("module",toJson entry.module.toString),("olean",toJson (←findOLean entry.module).toString)]
  IO.println (Json.mkObj [("theorems_audited",toJson theorems),("axioms_used",toJson used),
    ("violations",toJson violations),("imports",toJson imports),
    ("assurance",toJson "lean_kernel_std3_no_native_reflection")]).compress
  return if violations.isEmpty then 0 else 1
