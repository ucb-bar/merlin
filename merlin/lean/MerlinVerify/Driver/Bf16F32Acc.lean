/- Reporting adapter for the f32-accumulation contract checker. Diagnostics below are untrusted
explanations of a refusal; only `Check.Bf16F32Acc.check` and `qualify` carry theorems. The
contract is a review candidate, so no result is reported as a reviewed policy verdict. -/
import MerlinVerify.Core.Json
import MerlinVerify.Check.Bf16F32AccChain
import MerlinVerify.Check.Bf16F32AccSites

namespace MerlinVerify.Driver.Bf16F32Acc

open MerlinVerify MerlinVerify.Check MerlinVerify.Rules.Bf16F32AccReference Lean

def flagJson (f : Rules.Bf16F32AccContract.Flag) : Json := toJson f.text

/-- The first candidate the reference transformer cannot accept, with the condition it fails. -/
def ineligible? (source : Module) : Option String := Id.run do
  for op in source.ops do
    let some fn := op.asFunc? | continue
    if fn.isDecl then continue
    let [.mk [.mk args ops]] := op.regions | return some s!"@{fn.name}: not a single-block body"
    let mut seen : List Op := []
    for inner in ops do
      if candidate inner then
        if let .error why := siteE args seen inner then
          let id := (inner.results.head?.map (·.id)).getD 0
          return some s!"@{fn.name}: {inner.name} defining %{id}: {why}"
      seen := inner :: seen
  return none

def emitted : List String :=
  ["f32 tensor.empty", "f32 arith.constant", "linalg.fill", "accumulating linalg.generic",
   "result tensor.empty", "final-conversion linalg.generic"]

def flagText? (op : Op) : String :=
  match op.attrs with
  | [("fastmath", .enum "arith" "fastmath" text)] => s!"fastmath<{text}>"
  | attrs => s!"attributes {attrs.map (·.1)}"

/-- Why per-site hints cannot be read from the output: the first site without six replacement
operations, or whose accumulating region carries a flag outside the permitted license. -/
def hintRefusal? (sources targets : List Op) (index : Nat) : Option String :=
  match sources with
  | [] => none
  | source :: rest =>
    if candidate source then
      match targets.take 6 with
      | [_, _, _, acc, _, _] =>
        match licenseOf? acc with
        | some _ => hintRefusal? rest (targets.drop 6) (index + 1)
        | none =>
          match acc.regions with
          | [.mk [.mk _ [_, _, mul, add, _]]] =>
            some s!"site {index}: multiply {flagText? mul} / add {flagText? add}; permitted flags are none and contract"
          | _ => some s!"site {index}: operation 4 of the replacement is {acc.name}, not the accumulating linalg.generic"
      | _ => some s!"site {index}: fewer than six operations replace the contraction"
    else hintRefusal? rest (targets.drop 1) index

/-- Why an admissible-looking hint list is still refused: an emitted operation whose attributes
differ from the contract's canonical ones other than by semantics-free annotations. -/
def emitRefusal? (site : Site) (hint : SiteHint) (index : Nat) : Option String :=
  ((emitted.zip ((canonical site).zip hint.attrs))).findSome? fun (name, canon, hinted) =>
    let actual := semantic hinted
    if decide (actual = canon) then none else
      let keys := (actual.map (·.1) ++ canon.map (·.1)).eraseDups.filter fun key =>
        !decide (actual.lookup key = canon.lookup key)
      some s!"site {index}: {name}: attribute(s) {keys} differ from the contract"

def refusal? (before after : Module) (hints : Option (List SiteHint)) : Option String := Id.run do
  let mut index := 0
  let mut remaining := hints.getD []
  for (source, target) in before.ops.zip after.ops do
    let some fn := source.asFunc? | continue
    if fn.isDecl then continue
    let [.mk [.mk args ops]] := source.regions | return some s!"@{fn.name}: not a single-block body"
    if hints.isNone then
      let [.mk [.mk _ targetOps]] := target.regions | return some s!"@{fn.name}: output body is not a single block"
      if let some why := hintRefusal? ops targetOps index then return some s!"@{fn.name}: {why}"
      continue
    let mut seen : List Op := []
    for inner in ops do
      if candidate inner then
        match siteE args seen inner, remaining with
        | .error why, _ => return some s!"@{fn.name}: site {index}: source outside the contract: {why}"
        | .ok site, hint :: rest =>
          if let some why := emitRefusal? site hint index then return some s!"@{fn.name}: {why}"
          remaining := rest
        | .ok _, [] => return some s!"@{fn.name}: site {index}: no replacement in the output"
        index := index + 1
      seen := inner :: seen
  return none

/-- The first operation where the output stops being alpha-equivalent to the contract reference,
named by its position in the function body. -/
def firstMismatch? (ρ : Rho) : List Op → List Op → Nat → Option String
  | [], [], _ => none
  | source :: sources, target :: targets, index =>
    match alphaOp ρ source target with
    | some ρ' => firstMismatch? ρ' sources targets (index + 1)
    | none =>
      let part := if opHeadEq source target then "operands or region" else "name, attributes or result types"
      some s!"operation {index}: output {target.name} differs from the contract reference {source.name} in its {part}"
  | _, _, index => some s!"operation {index}: the output has a different number of operations"

def bodyMismatch? (reference after : Module) (fn : String) : Option String := do
  let source ← reference.func? fn
  let target ← after.func? fn
  let [.mk sourceArgs sourceOps] := source.body.blocks | none
  let [.mk targetArgs targetOps] := target.body.blocks | none
  firstMismatch? ((sourceArgs.map (·.id)).zip (targetArgs.map (·.id))) sourceOps targetOps 0

def licensed (op : Op) : Nat :=
  if (op.name == "arith.mulf" || op.name == "arith.addf") &&
      (match op.attr? "fastmath" with
        | some (.enum "arith" "fastmath" text) => (flagList text).any (fun f => f == "contract" || f == "fast")
        | _ => false) then 1 else 0

-- `contract`-licensed float multiplies and adds anywhere in a program, including nested regions.
-- `Rules.Bf16F32AccFusion.fused_step` covers fused execution only if the source has none, so
-- every licensed operation of the output is a contract step.
mutual
  def licensedOps : List Op → Nat
    | [] => 0
    | (op@(.mk _ _ _ _ _ regions _)) :: rest => licensed op + licensedRegions regions + licensedOps rest
  def licensedRegions : List Region → Nat
    | [] => 0
    | .mk blocks :: rest => licensedBlocks blocks + licensedRegions rest
  def licensedBlocks : List Block → Nat
    | [] => 0
    | .mk _ ops :: rest => licensedOps ops + licensedBlocks rest
end

def licensedIn (m : Module) : Nat := licensedOps m.ops

/-- Operand formats of the source sites, so a report states whether one format is used. -/
def siteFormats (source : Module) : List String := Id.run do
  let mut formats : List String := []
  for op in source.ops do
    let [.mk [.mk args ops]] := op.regions | continue
    if op.name != "func.func" then continue
    let mut seen : List Op := []
    for inner in ops do
      if candidate inner then
        if let some site := site? args seen inner then
          let name := match site.k with | .bf16 => "bf16" | .f16 => "f16" | .f32 => "f32" | .f64 => "f64"
          if !formats.contains name then formats := formats ++ [name]
      seen := inner :: seen
  return formats

/-- Count candidates per function, so a report states the denominator it was checked against. -/
def candidates (source : Module) : Nat :=
  source.ops.foldl (fun n op => n + match op.regions with
    | [.mk [.mk _ ops]] => if op.name == "func.func" then (ops.filter candidate).length else 0
    | _ => 0) 0

/-- Pass-local qualification is separate from whole-capture definedness. Every operand-general
site obligation is reconstructed inside Lean from the bound source and target operations. -/
def siteReport (ib : Nat) (before after : Module) : Json := Id.run do
  let exactMatch := Bf16F32AccExact.check before after
  let pairs := Bf16F32AccSites.collect before after
  let mut details : List Json := []
  let mut allQualified := pairs.isSome
  for pair in pairs.getD [] do
    let qualified := Bf16F32AccSites.pairCheck ib pair
    allQualified := allQualified && qualified
    details := details ++ [Json.mkObj [
      ("function", toJson pair.function), ("source_operation", toJson pair.index),
      ("source_results", toJson (pair.source.results.map (·.id))),
      ("qualification_passed", toJson qualified)]]
  return Json.mkObj [
    ("checked", toJson (Bf16F32AccSites.accept ib exactMatch pairs)),
    ("semantic_match", toJson exactMatch),
    ("semantic_match_theorem", toJson "MerlinVerify.Check.Bf16F32AccExact.check_sound"),
    ("site_qualification_passed", toJson allQualified),
    ("site_qualification_theorem", toJson "MerlinVerify.Check.Bf16F32AccSites.check_sound"),
    ("site_obligations", toJson details),
    ("expected_sites", toJson (candidates before)),
    ("covered_sites", toJson details.length),
    ("scope", toJson "bf16 pass boundary; universal local operand domain; complete output equality under SSA-invariant semantics"),
    ("whole_capture_claim", toJson "separate; see function_obligations")]

/-- Each obligation is computed once. `qualify` is evaluated only when every component it
conjoins has already succeeded; otherwise it is false by its definition. -/
def report (ib : Nat) (before after : Module) : Json := Id.run do
  let mut reasons := ["f32-accumulation contract is a review candidate: independent specification review is required"]
  let mut details : List Json := []
  let mut sites : List Json := []
  let mut domains := false
  let matched := Bf16F32Acc.check before after
  let recovered := hints? before after
  match recovered with
  | none =>
    reasons := reasons ++ [match ineligible? before, refusal? before after none with
      | some why, _ => s!"source outside the contract: {why}"
      | none, some why => why
      | none, none => "output functions do not pair with the source functions"]
  | some hints =>
    sites := hints.map fun hint => Json.mkObj
      [("mul", flagJson hint.license.mul), ("add", flagJson hint.license.add)]
    match expand hints before with
    | none =>
      reasons := reasons ++ [(refusal? before after recovered).getD
        "the number of replaced sites differs from the number of source candidates"]
    | some reference =>
      if !matched then
        reasons := reasons ++ (outsideChanges reference after).unverified
      domains := true
      for fn in Module.definedNames reference do
        let functionMatch := matched || c1Check (synthC1 reference after fn)
        if !functionMatch then reasons := reasons ++
          [s!"@{fn}: " ++ ((bodyMismatch? reference after fn).orElse fun _ => explainC1 (synthC1 reference after fn)).getD
            "output does not match the contract reference"]
        let (sourceDefined, referenceSupported, why) :=
          match TypedModule.signature? ib before fn, TypedModule.signature? ib reference fn with
          | some (a, r), some (a', r') =>
            if a = a' ∧ r = r' then
              let supported := supportCheck (merlinBook ib) reference fn a r
              (DefinedExtended.check ib before fn a r, supported,
                if supported then "" else (diagnoseFunc (merlinBook ib) reference fn a r).getD "reference support refused")
            else (false, false, "contract reference changes the declared input/result domain")
          | _, _ => (false, false, "missing or unsupported function signature")
        domains := domains && sourceDefined && referenceSupported
        if !sourceDefined then reasons := reasons ++ [s!"@{fn}: source definedness has not been established"]
        if !referenceSupported then reasons := reasons ++ [s!"@{fn}: {why}"]
        details := details ++ [Json.mkObj [("function", toJson fn), ("conditional_match", toJson functionMatch),
          ("source_defined", toJson sourceDefined), ("reference_supported", toJson referenceSupported),
          ("first_refusal", toJson why)]]
  let qualified := matched && domains && Bf16F32Acc.qualify ib before after
  return Json.mkObj [("checker", toJson "MerlinVerify.Check.Bf16F32Acc.check"),
    ("theorem", toJson "MerlinVerify.Check.Bf16F32Acc.check_sound"),
    ("qualification_theorem", toJson "MerlinVerify.Check.Bf16F32Acc.qualify_sound"),
    ("contract_theorems", toJson ["MerlinVerify.Rules.Bf16F32AccTensor.accumulate_sound",
      "MerlinVerify.Rules.Bf16F32AccTensor.truncate_sound",
      "MerlinVerify.Rules.Bf16F32AccTensor.zero_fill_sound",
      "MerlinVerify.Rules.Bf16F32AccContract.accumulate_cell"]),
    ("claim_type", toJson "policy"), ("policy", toJson "bf16-f32acc-contract"),
    ("index_bits", toJson ib), ("candidates", toJson (candidates before)),
    ("fusion_premises", Json.mkObj [("source_licensed_operations", toJson (licensedIn before)),
      ("site_formats", toJson (siteFormats before)),
      ("theorem", toJson "MerlinVerify.Rules.Bf16F32AccFusion.fused_step")]),
    ("sites", toJson sites), ("conditional_match", toJson matched),
    ("qualification_passed", toJson qualified), ("policy_reviewed", toJson false),
    ("pass_boundary", siteReport ib before after),
    ("function_obligations", toJson details), ("unverified_changes", toJson reasons),
    ("reason", toJson ("; ".intercalate reasons)), ("verdict", toJson "abstracted")]

/-- A policy boundary followed by checked refinement boundaries, ending before the next
numerics-changing boundary. Each suffix step is diagnosed separately; only the two segment
checks carry theorems. -/
def chainReport (ib : Nat) (before after : Module) (rest : List Module) : Json := Id.run do
  let boundary := report ib before after
  let segment := Bf16F32AccChain.check ib before after rest
  let mut steps : List Json := []
  let mut stepsPass := true
  match hints? before after with
  | none => stepsPass := false
  | some hints =>
    match expand hints before with
    | none => stepsPass := false
    | some reference =>
      let programs := after :: rest
      for (index, previous, next) in (List.range rest.length).zip (programs.zip rest) do
        -- An accepted segment already checked every step; diagnose steps only after a refusal.
        if segment then
          steps := steps ++ [Json.mkObj [("index", toJson index), ("conditional_match", toJson true),
            ("reason", toJson "")]]
          continue
        let mut ok := true
        let mut why := ""
        for fn in Module.definedNames reference do
          match TypedModule.signature? ib reference fn with
          | some (args, results) =>
            if !(PolicyW8A8Chain.stepCheck ib args results fn previous next) then
              ok := false
              why := s!"@{fn}: " ++ ((bodyMismatch? previous next fn).orElse
                fun _ => explainC1 (synthC1 previous next fn)).getD "refinement refused"
          | none => ok := false; why := s!"@{fn}: missing signature"
        let outside := (outsideChanges previous next).unverified
        if !outside.isEmpty then
          ok := false
          why := why ++ "; " ++ "; ".intercalate outside
        stepsPass := stepsPass && ok
        steps := steps ++ [Json.mkObj [("index", toJson index), ("conditional_match", toJson ok),
          ("reason", toJson why)]]
  let qualified := segment && stepsPass &&
    (boundary.getObjValAs? Bool "qualification_passed").toOption == some true &&
    Bf16F32AccChain.qualify ib before after rest
  return Json.mkObj [("checker", toJson "MerlinVerify.Check.Bf16F32AccChain.check"),
    ("theorem", toJson "MerlinVerify.Check.Bf16F32AccChain.check_sound"),
    ("qualification_theorem", toJson "MerlinVerify.Check.Bf16F32AccChain.qualify_sound"),
    ("claim_type", toJson "policy"), ("policy", toJson "bf16-f32acc-contract"), ("index_bits", toJson ib),
    ("policy_boundary", boundary), ("suffix_boundaries", toJson steps),
    ("segment_match", toJson segment), ("segment_qualified", toJson qualified),
    ("policy_reviewed", toJson false),
    ("reason", toJson ((boundary.getObjValAs? String "reason").toOption.getD "" ++
      (if segment then "" else "; downstream suffix does not refine the contract reference"))),
    ("verdict", toJson "abstracted")]

def runChain (beforePath afterPath width : String) (suffix : List String) : IO UInt32 := do
  let some ib := width.toNat?.filter (· > 0) | do
    IO.eprintln "index_bits must be one positive integer"
    return 2
  let decodeAll (paths : List String) : IO (Except String (List Module)) := do
    let mut out := []
    for path in paths do
      match Decode.module (← IO.FS.readFile path) with
      | .ok m => out := out ++ [m]
      | .error e => return .error e
    return .ok out
  match ← decodeAll (beforePath :: afterPath :: suffix) with
  | .ok (before :: after :: rest) =>
    IO.println (chainReport ib before after rest).compress
    return 1
  | .ok _ => return 2
  | .error error =>
    IO.eprintln s!"decode error: {error}"
    return 2

/-- Exit 1 for every report, since an unreviewed contract is never `verified` (exit 0 is reserved
for that verdict); `conditional_match` carries the checked outcome. Exit 2 on bad input. -/
def run (beforePath afterPath width : String) : IO UInt32 := do
  let some ib := width.toNat?.filter (· > 0) | do
    IO.eprintln "index_bits must be one positive integer"
    return 2
  match Decode.module (← IO.FS.readFile beforePath), Decode.module (← IO.FS.readFile afterPath) with
  | .ok before, .ok after =>
    IO.println (report ib before after).compress
    return 1
  | .error error, _ | _, .error error =>
    IO.eprintln s!"decode error: {error}"
    return 2

end MerlinVerify.Driver.Bf16F32Acc
