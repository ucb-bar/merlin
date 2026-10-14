import Lean

/-!
`merlin-lean-audit`: guardrail audit of the compiled `MerlinVerify` environment (design §1.2, §8.6).

`merlin-lean-audit <module>...` imports the given modules and inspects every constant named
`MerlinVerify.*`:
* every theorem's axioms (`collectAxioms`) must be ⊆ {propext, Classical.choice, Quot.sound}, plus
  axioms declared in `MerlinVerify.Assumptions.*`; anything else (`sorryAx`, native-decide/
  bv_decide axioms, `Lean.ofReduceBool`, `Lean.trustCompiler`) is a violation;
* no `axiom` declarations outside `MerlinVerify.Assumptions`;
* no `unsafe` or `partial` constants, and none carrying `@[implemented_by]` or `@[extern]`, so the
  compiled checker is the function the theorems are about (design D11).
`merlin-lean-audit --axioms <decl> <module>...` prints one declaration's axioms as JSON.
Exit status: 0 clean, 1 violations, 2 usage/import error.
-/

open Lean

def allowedAxioms : List Name := [``propext, ``Classical.choice, ``Quot.sound]

def inMerlin (n : Name) : Bool := (`MerlinVerify).isPrefixOf n
def inAssumptions (n : Name) : Bool := (`MerlinVerify.Assumptions).isPrefixOf n

/-- `f._unsafe_rec` is the executable twin Lean's compiler generates for a recursive definition `f`.
When `f` itself is a safe definition (structural or well-founded recursion, so the kernel checked
its equations) the twin is a compiler artifact, and trusting it is trusting the Lean compiler
(design D11). A user-written `partial def f` instead makes `f` an `opaque` constant (no body the
kernel can see), which the audit flags directly. -/
def isCompilerAux (env : Environment) (n : Name) : Bool :=
  match n with
  | .str base "_unsafe_rec" =>
    match env.find? base with
    | some (.defnInfo d) => d.safety == .safe
    | _ => false
  | _ => false

def axiomsOf (env : Environment) (n : Name) : IO (Array Name) := do
  let (r, _) ← (collectAxioms n : CoreM (Array Name)).toIO
    { fileName := "<audit>", fileMap := default, maxHeartbeats := 0 } { env }
  return r

def jsonStrList (xs : List String) : String :=
  "[" ++ ", ".intercalate (xs.map fun x => "\"" ++ x ++ "\"") ++ "]"

def main (args : List String) : IO UInt32 := do
  initSearchPath (← findSysroot)
  let (axiomTarget, mods) := match args with
    | "--axioms" :: d :: ms => (some d, ms)
    | ms => (none, ms)
  if mods.isEmpty then
    IO.eprintln "usage: merlin-lean-audit [--axioms <decl>] <module>..."
    return 2
  let env ← importModules (mods.toArray.map fun m => { module := m.toName }) {} (loadExts := false)
  if let some d := axiomTarget then
    let axs ← axiomsOf env d.toName
    IO.println s!"\{\"decl\": \"{d}\", \"axioms\": {jsonStrList (axs.toList.map toString)}}"
    return 0
  let mut violations : Array String := #[]
  let mut theorems := 0
  for (n, ci) in env.constants.toList do
    if !inMerlin n || isCompilerAux env n then continue
    if ci.isUnsafe then violations := violations.push s!"unsafe: {n}"
    if ci.isPartial then violations := violations.push s!"partial: {n}"
    if (Compiler.implementedByAttr.getParam? env n).isSome then
      violations := violations.push s!"implemented_by: {n}"
    if isExtern env n then violations := violations.push s!"extern: {n}"
    match ci with
    | .opaqueInfo _ => violations := violations.push s!"opaque (e.g. partial def): {n}"
    | .axiomInfo _ =>
      if !inAssumptions n then violations := violations.push s!"axiom outside Assumptions: {n}"
    | .thmInfo _ =>
      theorems := theorems + 1
      for a in (← axiomsOf env n) do
        if !(allowedAxioms.contains a || inAssumptions a) then
          violations := violations.push s!"theorem {n} depends on axiom {a}"
    | _ => pure ()
  IO.println s!"\{\"theorems_audited\": {theorems}, \"violations\": {jsonStrList (violations.toList.map toString)}}"
  return (if violations.isEmpty then 0 else 1)
