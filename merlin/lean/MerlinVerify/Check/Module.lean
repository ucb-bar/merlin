/-
Module-level boundary check: C1 on every function the source module defines, plus an accounting of
every difference outside function bodies (design §5.3).

* `moduleCheck_sound`: every function defined in the source is refined by the same-named function
  of the target (`FuncRefinesC`), for every float model, external environment and input.
* `outsideChanges` classifies what the semantic claim does not cover, against the reviewed
  `Spec.Annotations`: ABI-only attributes (honored by the trusted lowering, design §10 T4) are
  reported separately; any other difference is an unverified change and blocks a `verified` verdict.
-/
import MerlinVerify.Check.Synth
import MerlinVerify.Spec.Annotations

namespace MerlinVerify.Check

open MerlinVerify

/-- Names of the functions a module defines (has a body for), in order. -/
def Module.definedNames (m : Module) : List String :=
  (m.funcs.filter fun f => !f.isDecl).map (·.name)

def moduleCheck (src tgt : Module) : Bool :=
  (Module.definedNames src).all fun f => c1Check (synthC1 src tgt f)

theorem moduleCheck_sound {src tgt : Module} (h : moduleCheck src tgt = true) :
    ∀ f ∈ Module.definedNames src,
      FuncRefinesC Sem.Ops.opSem (fun _ => True) (fun _ => True) tgt f src f := by
  intro f hf
  have hc := List.all_eq_true.1 h f hf
  have := c1Check_sound (synthC1 src tgt f) hc
  simpa [synthC1] using this

/-! ### Differences outside function bodies -/

def isAbiOnly (k : String) : Bool := Spec.Annotations.abiOnly k
def isIgnorable (k : String) : Bool := Spec.Annotations.semanticsFree k || Spec.Annotations.abiOnly k

/-- Attributes (discardable and inherent) that matter for the claim. -/
def claimAttrs (as ps : List (String × Attr)) : List (String × Attr) :=
  ((as ++ ps).filter fun (k, _) => !isIgnorable k).mergeSort fun a b => a.1 ≤ b.1

def abiAttrs (as ps : List (String × Attr)) : List (String × Attr) :=
  ((as ++ ps).filter fun (k, _) => isAbiOnly k).mergeSort fun a b => a.1 ≤ b.1

def symName (op : Op) : String :=
  match op.attr? "sym_name" with
  | some (.str s) => s
  | _ => ""

def topKey (op : Op) : String := op.name ++ " @" ++ symName op

structure OutsideChanges where
  /-- ABI-only attribute changes: outside the claim, honored by the trusted lowering. -/
  abi : List String
  /-- Anything else: unverified. -/
  unverified : List String

def outsideChanges (src tgt : Module) : OutsideChanges := Id.run do
  let mut abi : List String := []
  let mut unv : List String := []
  -- Pairing top-level ops by key is only sound when keys are unique on both sides.
  if !(src.ops.map topKey).Nodup || !(tgt.ops.map topKey).Nodup then
    return { abi, unverified := ["ambiguous top-level ops (duplicate name/symbol keys)"] }
  if claimAttrs src.attrs src.props ≠ claimAttrs tgt.attrs tgt.props then
    unv := unv ++ ["module attributes"]
  if abiAttrs src.attrs src.props ≠ abiAttrs tgt.attrs tgt.props then
    abi := abi ++ ["module ABI attributes"]
  for o in src.ops do
    match tgt.ops.find? (fun o' => topKey o' == topKey o) with
    | none => unv := unv ++ [s!"removed {topKey o}"]
    | some o' =>
      if claimAttrs o.attrs o.props ≠ claimAttrs o'.attrs o'.props then
        unv := unv ++ [s!"{topKey o}: attributes"]
      if abiAttrs o.attrs o.props ≠ abiAttrs o'.attrs o'.props then
        abi := abi ++ [s!"{topKey o}: ABI attributes"]
      if o.name ≠ "func.func" && !alphaOps [] [o] [o'] then
        unv := unv ++ [s!"{topKey o}: changed"]
  for o' in tgt.ops do
    if (src.ops.find? fun o => topKey o == topKey o').isNone then
      unv := unv ++ [s!"added {topKey o'}"]
  return { abi, unverified := unv }

end MerlinVerify.Check
