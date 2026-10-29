# ruff: noqa: E501
# Literal Lean equations are kept intact for readable, byte-bound proof evidence.
"""Kernel proof generation from the complete five-program scalar/SCF boundary.

Finite syntax/lane expansion only. Trip counts enter generic W/fold theorems;
no dynamic iteration trace is generated to certify a huge loop.
"""

from __future__ import annotations

from .lean_graph_state_pipeline import SIMPS, controls_literal, lean_list, prefix_environment
from .lean_graph_state_scalar import budget, integer, primitive, program, types


def unpack(name, size, count, prefix, pad="  ", offset=0):
    # Iterative construction keeps Python's call-stack depth out of the entry
    # and carry domain. Lean still checks each actual list-length elimination.
    if count == 0:
        return f"{pad}have {prefix}empty : {name} = [] := List.length_eq_zero_iff.mp {size}\n{pad}subst {name}\n"
    pieces = []
    for remaining in range(count, 1, -1):
        tail = f"{prefix}tail{offset}"
        later_size = f"{prefix}size{offset}"
        pieces.append(f"{pad}obtain ⟨{prefix}{offset},{tail},rfl⟩ := List.exists_cons_of_length_eq_add_one {size}\n")
        pieces.append(
            f"{pad}have {later_size} : {tail}.length = {remaining - 1} := by\n{pad}  simp only [List.length_cons] at {size}\n{pad}  omega\n"
        )
        name, size, offset = tail, later_size, offset + 1
    pieces.append(f"{pad}obtain ⟨{prefix}{offset},rfl⟩ := List.length_eq_one_iff.mp {size}\n")
    return "".join(pieces)


def entry_arguments(input_count, lemma, inputs="inputs"):
    """Eliminate a typed entry list without assuming a particular ABI arity."""
    args = " ".join(f"x{i}" for i in range(input_count))
    text = f"  have size : arguments.length = {input_count} := by\n"
    text += f"    simpa only [{inputs},List.length_nil,List.length_cons] using arity\n"
    text += unpack("arguments", "size", input_count, "x")
    return text + f"  exact {lemma} primitive external {args} world\n"


def static_controls(records, capture, *, input_count=2):
    env = [None] * input_count
    controls = {r["identity"]: r for r in capture["index_operations"]}
    for record in records:
        if "loop" in record:
            values = [env[i] for i in record["loop"]["bounds"]]
            if any(type(value) is not int or value < 0 for value in values):
                raise ValueError("actual static nonnegative controls required")
            return values
        if "pure" in record:
            value = None
        else:
            op = controls.get(record["effect"])
            if op is None:
                raise ValueError("unsupported partial prefix")
            args = [env[i] for i in record["arguments"]]
            if op["kind"] == "index-constant":
                value = op["value"]
            elif all(type(v) is int for v in args) and len(args) == 2:
                value = sum(args) if op["kind"] == "index-add" else min(args)
            else:
                value = None
        env.insert(0, value)
    raise ValueError("missing loop")


def symbolic_body(records, environment, control_map, result_prefix, world):
    env = list(environment)
    calls = []
    for row in records:
        if "ret" in row:
            return [env[i] for i in row["ret"]], calls
        if "loop" in row:
            raise ValueError("region-free original body required in this submode")
        args = [env[i] for i in row["arguments"]]
        if "pure" in row:
            env.insert(0, "(primitive " + primitive(row["pure"]) + " " + lean_list(args) + ")")
        else:
            control = control_map.get(row["effect"])
            if control:
                if control["kind"] == "index-constant":
                    value = f"({control['value']} : Int)"
                elif control["kind"] == "index-add" and len(args) == 2:
                    value = f"({args[0]} + {args[1]})"
                elif control["kind"] == "index-minui" and len(args) == 2:
                    value = f"(min {args[0]} {args[1]})"
                else:
                    raise ValueError("malformed original body control")
                env.insert(0, value)
            else:
                index = result_prefix + len(calls)
                results = [f"v{index}_{r}" for r in range(len(row["outputs"]))]
                calls.append(
                    {
                        "index": index,
                        "identity": row["effect"],
                        "arguments": args,
                        "world": world,
                        "outputs": len(results),
                    }
                )
                world = "w" + str(index)
                env = results + env
    raise ValueError("missing complete body return")


def header(capture):
    registry = capture["stages"][0]["registry"]
    value = (
        "["
        + ",".join("{inputs := " + types(r["inputs"]) + ",outputs := " + types(r["outputs"]) + "}" for r in registry)
        + "]"
    )
    return f"""import Merlin.GraphStatePipelineIndex
import Merlin.GraphStatePipelineLiterals
import Merlin.GraphStatePipelineScope
import Merlin.GraphStatePipelineLoops
import Merlin.GraphStatePipelineProgram
import Merlin.GraphStateScalar
set_option maxRecDepth 4096
set_option maxHeartbeats 2000000
set_option linter.unusedSimpArgs false
namespace Merlin.GraphStatePipelineInstance
open Merlin.GraphState.Scalar Merlin.GraphState.Pipeline
def registry : List Signature := {value}
def controls : Controls := {controls_literal(capture)}
theorem controlAccepted : checkControl registry controls = true := by decide
def profileBits : Nat := {capture["index_bits"]}
theorem literalsAccepted : checkLiterals profileBits controls = true := by decide
theorem literalsFit (identity : Nat) (value : Int) (found : Merlin.GraphState.Pipeline.controlAt controls identity = some (.constant value)) :
    signedRepresentative profileBits value := checkLiterals_sound literalsAccepted found
def M {{World Error : Type}} (primitive : Prim → List Int → Int)
    (external : External World Error registry) : Model Int World Error registry :=
  indexModel (checkControl_sound controlAccepted) primitive external
theorem naturalRange (lower upper step : Nat) (positive : 0 < step) :
  indexIterations (Int.ofNat lower) (Int.ofNat upper) (Int.ofNat step) =
    (scfRange lower upper step).map Int.ofNat := by
  change indexIterations (lower : Int) (upper : Int) (step : Int) = _
  have pos : (0 : Int) < (step : Int) := by omega
  simp only [indexIterations,Int.natCast_nonneg,pos,and_self,↓reduceIte,Int.toNat_natCast]
"""


def body_group_text(capture, loops, envs, count, factor, step, args, *, max_calls=16):
    # Exact actual Program suffixes, not a replacement body/evaluator.
    control_map = {r["identity"]: r for r in capture["index_operations"]}
    result, groups = "", {}
    for stage in (2, 3):
        records = loops[stage]["body"]
        # Validate full terms through the unchanged serializer first.
        program(records)
        pieces, pending = [], []
        for i, row in enumerate(records):
            name = f"Piece{stage}_{i}"
            pending.append(name)
            external = "effect" in row and row["effect"] not in control_map
            if external or "ret" in row:
                pieces.append(pending)
                pending = []
        if pending or len(pieces) < 1:
            raise ValueError("complete actual suffix/call partition required")
        groups[stage] = pieces
        for i in reversed(range(len(records))):
            row = records[i]
            if "ret" in row:
                term = program([row])
            else:
                if "loop" in row:
                    raise ValueError("region-free actual unrolled body required")
                term = program([row, {"ret": []}])
                if term.count("(.ret [])") != 1:
                    raise ValueError("unique structural suffix continuation required")
                term = term.replace("(.ret [])", f"Piece{stage}_{i + 1}", 1)
            result += f"@[irreducible] def Piece{stage}_{i} : Program := {term}\n"
        result += f"theorem B{stage}_factored : B{stage} = Piece{stage}_0 := by\n"
        result += (
            "  unfold B" + str(stage) + " " + " ".join(f"Piece{stage}_{i}" for i in range(len(records))) + "\n  rfl\n"
        )
    source_calls = len(groups[2]) - 1
    if len(groups[3]) - 1 != factor * source_calls:
        raise ValueError("actual target external prefix count differs from source lanes")
    result += f"""theorem body_group (base : Int) (carry : List Int) (world : World) (arity : carry.length = {count}) :
  evalProgram (M primitive external) B3 (base::carry++E3 primitive {args}) world =
  loopEval {count} (fun i c w => evalProgram (M primitive external) B2 (i::c++E2 primitive {args}) w)
    {lean_list(["base" if lane == 0 else f"(base+{step * lane})" for lane in range(factor)])} carry world := by
"""
    result += unpack("carry", "arity", count, "c")
    initial = ",".join(groups[2][0] + groups[3][0])
    result += f"  simp only [B2_factored,B3_factored,{initial},E2,E3,{SIMPS},loopEval]\n"
    values = [f"c{i}" for i in range(count)]
    calls = []
    for lane in range(factor):
        iv = "base" if lane == 0 else f"(base+{step * lane})"
        values, more = symbolic_body(
            loops[2]["body"],
            [iv, *values, *envs[2]],
            control_map,
            len(calls),
            "world" if not calls else f"w{len(calls) - 1}",
        )
        calls += more
    if max_calls is not None and len(calls) > max_calls:
        raise ValueError("bounded lane proof supports at most sixteen retained calls")
    for depth, call in enumerate(calls):
        pad = "  " + "  " * depth
        index, identity = call["index"], call["identity"]
        args, world = lean_list(call["arguments"]), call["world"]
        next_defs = groups[3][index + 1] + groups[2][(index + 1) % source_calls]
        if (index + 1) % source_calls == 0:
            next_defs += groups[2][-1]
        definitions = ",".join(next_defs)
        result += f"{pad}cases step{index} : external.run {identity} {args} {world} with\n"
        result += f"{pad}| error failure =>\n{pad}  rcases failure with ⟨error,terminal⟩\n"
        result += f"{pad}  simp only [step{index},{definitions},{SIMPS},loopEval]\n"
        result += f"{pad}| ok result =>\n{pad}  rcases result with ⟨values,w{index}⟩\n"
        result += f"{pad}  have size : values.length = {call['outputs']} := by\n"
        result += f"{pad}    have localArity := external.arity {identity} {args} {world} values w{index} step{index}\n"
        result += f"{pad}    simpa only [registry,signature,List.getElem?_cons_succ,List.getElem?_cons_zero,Option.getD_some,List.length_cons,List.length_nil] using localArity\n"
        result += unpack("values", "size", call["outputs"], f"v{index}_", pad + "  ")
        result += f"{pad}  simp only [step{index},{definitions},{SIMPS},loopEval]\n"
    return result


def scope_text(capture, loops, envs, count, args):
    carry = [f"c{i}" for i in range(count)]
    outer_carry = [f"outerCarry{i}" for i in range(count)]
    outer_env = prefix_environment(loops[4]["body"], ["outer", *outer_carry, *envs[4]], capture)
    inner_loop = next(row["loop"] for row in loops[4]["body"] if "loop" in row)
    target_frame = ["iv", *carry, *outer_env]
    logical = list(dict.fromkeys(target_frame))
    source = ["iv", *carry, *envs[3]]
    selectors = [logical.index(value) for value in source]

    def symbols(values):
        return lean_list([f"(.var {i})" for i in values])

    result = "def I4 : Program := " + program(inner_loop["body"]) + "\n"
    result += "def LS : List Expr := " + symbols(selectors) + "\n"
    result += "def RS : List Expr := " + symbols([logical.index(value) for value in target_frame]) + "\n"
    result += "theorem innerAccepted : Scope.check registry LS RS B3 I4 = true := by decide\n"
    result += f"theorem inner_same (iv outer {' '.join(outer_carry)} : Int) (carry : List Int) (world : World) (arity : carry.length = {count}) :\n"
    result += "  evalProgram (M primitive external) I4 (iv::carry++" + lean_list(outer_env) + ") world =\n"
    result += f"  evalProgram (M primitive external) B3 (iv::carry++E3 primitive {args}) world := by\n"
    result += unpack("carry", "arity", count, "c")
    result += (
        "  have checked := Scope.check_sound innerAccepted (M primitive external) " + lean_list(logical) + " world\n"
    )
    result += "  simpa only [LS,RS,E3,evalExpr,List.cons_append,List.nil_append,List.map_cons,List.map_nil,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD_some] using checked.symm\n"
    for left, right in ((2, 3), (3, 4)):
        target_frame = [*carry, *envs[right]]
        logical = list(dict.fromkeys(target_frame))
        selected = [logical.index(value) for value in [*carry, *envs[left]]]
        result += f"def CL{left}{right} : List Expr := " + symbols(selected) + "\n"
        result += (
            f"def CR{left}{right} : List Expr := " + symbols([logical.index(value) for value in target_frame]) + "\n"
        )
        result += f"theorem suffixAccepted{left}{right} : Scope.check registry CL{left}{right} CR{left}{right} N{left} N{right} = true := by decide\n"
        result += (
            f"theorem suffix_same{left}{right} (carry : List Int) (world : World) (arity : carry.length = {count}) :\n"
        )
        result += f"  evalProgram (M primitive external) N{left} (carry++E{left} primitive {args}) world =\n"
        result += f"  evalProgram (M primitive external) N{right} (carry++E{right} primitive {args}) world := by\n"
        result += unpack("carry", "arity", count, "c")
        result += (
            f"  have checked := Scope.check_sound suffixAccepted{left}{right} (M primitive external) "
            + lean_list(logical)
            + " world\n"
        )
        result += f"  simpa only [CL{left}{right},CR{left}{right},E{left},E{right},evalExpr,List.cons_append,List.nil_append,List.map_cons,List.map_nil,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD_some] using checked\n"
    return result


def outer_text(count, upper, stride, tile, args):
    block = stride * tile
    names = [f"c{i}" for i in range(count)]
    values = lean_list(names)
    frame = "[min (base+" + str(block) + ") " + str(upper) + ",base+" + str(block) + ",base," + ",".join(names) + "]"
    natframe = frame.replace("base", "Int.ofNat base")
    rets = lean_list([str(i) for i in range(count)])
    result = f"""theorem outer_form (base {" ".join(names)} : Int) (world : World) :
  evalProgram (M primitive external) B4 (base::{values}++E4 primitive {args}) world =
  (loopEval {count} (fun i cs w => evalProgram (M primitive external) I4
    (i::cs++{frame}++E4 primitive {args}) w)
      (indexIterations base (min (base+{block}) {upper}) {stride}) {values} world).bind
        (fun result => evalProgram (M primitive external) (.ret {rets})
          (result.1++{frame}++E4 primitive {args}) result.2) := by
  simp only [List.append_assoc]
  rfl
theorem outer_same (base : Nat) (carry : List Int) (world : World) (arity : carry.length = {count}) :
  evalProgram (M primitive external) B4 (Int.ofNat base::carry++E4 primitive {args}) world =
  loopEval {count} (fun i cs w => evalProgram (M primitive external) B3 (i::cs++E3 primitive {args}) w)
    ((scfRange base (min (base+{stride}*{tile}) {upper}) {stride}).map Int.ofNat) carry world := by
"""
    result += unpack("carry", "arity", count, "c")
    result += f"""  rw [outer_form]
  have bodies : ∀ i cs w, cs.length = {count} →
      evalProgram (M primitive external) I4 (i::cs++{natframe}++E4 primitive {args}) w =
      evalProgram (M primitive external) B3 (i::cs++E3 primitive {args}) w := by
    intro i cs w size
    simpa only [E4,List.append_assoc,List.cons_append,List.nil_append] using
      inner_same primitive external {args} i (Int.ofNat base) {" ".join(names)} cs w size
  rw [loopEval_congr {count} _ _ bodies]
  have minimum : min (Int.ofNat base+{block}) {upper} = Int.ofNat (min (base+{block}) {upper}) := minui_natural (base+{block}) {upper}
  have ranges : indexIterations (Int.ofNat base) (Int.ofNat (min (base+{block}) {upper})) {stride} =
      (scfRange base (min (base+{block}) {upper}) {stride}).map Int.ofNat := naturalRange base (min (base+{block}) {upper}) {stride} (by decide)
  rw [minimum,ranges]
  cases run : loopEval {count} (fun i cs w => evalProgram (M primitive external) B3 (i::cs++E3 primitive {args}) w)
      ((scfRange base (min (base+{block}) {upper}) {stride}).map Int.ofNat) {values} world with
  | error failure => rfl
  | ok result =>
    have size := loopEval_arity {count} _ _ _ _ run
    rcases result with ⟨values,final⟩
"""
    result += unpack("values", "size", count, "returned", "    ")
    return result + "    rfl\n"


def compose_text(
    count, bounds, bulk_bounds, factor, groups, tile, bits, initial, input_names, *, identity_unroll=False
):
    lower, upper, step = bounds
    bulk_lower, bulk_upper, stride = bulk_bounds
    outer_stride = stride * tile
    args, vector = " ".join(input_names), lean_list(input_names)
    text = f"""def unrollBounds : Merlin.GraphLoopsUnroll.Bounds :=
  {{ bits := {bits}, lower := {lower}, upper := {upper}, step := {step}, factor := {factor}, count := {groups * factor}, groups := {groups}, remainder := 0 }}
def segmentBounds : Merlin.GraphLoopsSegment.Bounds :=
  {{ bits := {bits}, lower := {bulk_lower}, upper := {bulk_upper}, step := {stride}, tile := {tile} }}
theorem unrollBoundsChecked : unrollBounds.Valid := by decide
theorem segmentBoundsChecked : segmentBounds.Valid := by decide
theorem unroll_actual (world : World) :
    evalProgram (M primitive external) S2 {vector} world = evalProgram (M primitive external) S3 {vector} world := by
  have bodies : ∀ index values w, values.length = {count} →
      evalProgram (M primitive external) B3 (Int.ofNat index::values++E3 primitive {args}) w =
      loopEval {count} (fun i c current => evalProgram (M primitive external) B2 (i::c++E2 primitive {args}) current)
        ((progression index {step} {factor}).map Int.ofNat) values w := by
    intro index values w size
    simpa [progression,Int.ofNat_eq_natCast,Int.add_assoc] using body_group primitive external {args} (Int.ofNat index) values w size
  have loops : loopEval {count} (fun i c w => evalProgram (M primitive external) B2 (i::c++E2 primitive {args}) w)
      (indexIterations {lower} {upper} {step}) {initial} world =
    loopEval {count} (fun i c w => evalProgram (M primitive external) B3 (i::c++E3 primitive {args}) w)
      (indexIterations {bulk_lower} {bulk_upper} {stride}) {initial} world :=
    unroll_loop {count} _ _ {lower} {upper} {step} {factor} {groups} (by decide) (by decide) (by decide) bodies {initial} world
  rw [form2,form3,loops]
  cases run : loopEval {count} (fun i c w => evalProgram (M primitive external) B3 (i::c++E3 primitive {args}) w)
      (indexIterations {bulk_lower} {bulk_upper} {stride}) {initial} world with
  | error failure => rfl
  | ok result => exact suffix_same23 primitive external {args} result.1 result.2 (loopEval_arity {count} _ _ _ _ run)
theorem segment_actual (world : World) :
    evalProgram (M primitive external) S3 {vector} world = evalProgram (M primitive external) S4 {vector} world := by
  have r2 : indexIterations {bulk_lower} {bulk_upper} {stride} = (scfRange {bulk_lower} {bulk_upper} {stride}).map Int.ofNat := naturalRange {bulk_lower} {bulk_upper} {stride} (by decide)
  have r6 : indexIterations {bulk_lower} {bulk_upper} {outer_stride} = (scfRange {bulk_lower} {bulk_upper} {outer_stride}).map Int.ofNat := naturalRange {bulk_lower} {bulk_upper} {outer_stride} (by decide)
  have loops : loopEval {count} (fun i c w => evalProgram (M primitive external) B3 (i::c++E3 primitive {args}) w)
      (indexIterations {bulk_lower} {bulk_upper} {stride}) {initial} world =
    loopEval {count} (fun i c w => evalProgram (M primitive external) B4 (i::c++E4 primitive {args}) w)
      (indexIterations {bulk_lower} {bulk_upper} {outer_stride}) {initial} world := by
    rw [r2,r6]
    exact segment_loop {count} _ _ {bulk_lower} {bulk_upper} {stride} {tile} (by decide) (by decide) (outer_same primitive external {args}) {initial} world
  rw [form3,form4,loops]
  cases run : loopEval {count} (fun i c w => evalProgram (M primitive external) B4 (i::c++E4 primitive {args}) w)
      (indexIterations {bulk_lower} {bulk_upper} {outer_stride}) {initial} world with
  | error failure => rfl
  | ok result => exact suffix_same34 primitive external {args} result.1 result.2 (loopEval_arity {count} _ _ _ _ run)
theorem actual_inputs (world : World) :
    evalProgram (M primitive external) S0 {vector} world = evalProgram (M primitive external) S4 {vector} world :=
  (firstTwo Int World Error (M primitive external) {vector} world rfl).trans
    ((unroll_actual primitive external {args} world).trans (segment_actual primitive external {args} world))
theorem actual (arguments : List Int) (world : World) (arity : arguments.length = inputs.length) :
    evalProgram (M primitive external) S0 arguments world = evalProgram (M primitive external) S4 arguments world := by
"""
    text += entry_arguments(len(input_names), "actual_inputs")
    if identity_unroll:
        # A standalone segmentation boundary has identical S2/S3 syntax. Do not
        # claim that factor one satisfies the separate unrolling contract.
        start = text.index("theorem unrollBoundsChecked")
        end = text.index("theorem segment_actual")
        text = (
            text[:start]
            + "theorem segmentBoundsChecked : segmentBounds.Valid := by decide\n"
            + f"""theorem unroll_actual (world : World) :
    evalProgram (M primitive external) S2 {vector} world = evalProgram (M primitive external) S3 {vector} world := rfl
"""
            + text[end:]
        )
    return text


def emit(capture, *, identity_unroll=False, flexible_parameters=False):
    stages = capture["stages"]
    if len(stages) != 4 or any(
        s["registry"] != stages[0]["registry"] or s["types"] != stages[0]["types"] for s in stages
    ):
        raise ValueError("one exact common five-stage registry/type table required")
    text = header(capture)
    factor, tile, bits = capture["factor"], capture["tile"], capture["index_bits"]
    for value in (factor, tile, bits):
        integer(value, natural=True)
    valid_factor = factor == 1 if identity_unroll else factor >= 2
    if not valid_factor or tile < 2 or bits not in (32, 64):
        raise ValueError("integer factor/tile at least two and signed32-or64 profile required")
    if not flexible_parameters and (factor > 4 or tile > 16):
        raise ValueError("historical factor2..4/tile2..16 profile required")
    for index, function in enumerate(stages[0]["functions"]):
        if function["declaration"]:
            continue
        records = [function["before"], *[s["functions"][index]["after"] for s in stages]]
        controls = {r["identity"]: r for r in capture["index_operations"]}

        def original_domain(rows):
            for row in rows:
                if "loop" in row:
                    original_domain(row["loop"]["body"])
                elif (
                    "effect" in row
                    and row["effect"] in controls
                    and controls[row["effect"]]["kind"] != "index-constant"
                ):
                    raise ValueError("source index arithmetic outside generated control bridges is unsupported")

        for raw in records:
            budget(raw, len(function["inputs"]), [0])
        original_domain(records[0])
        if records[2] == records[4] and any("loop" in row for row in records[2]):
            raise ValueError("unchanged SCF entries require their own explicit control bridge")
        text += (
            f"namespace F{index}\ndef inputs : List Ty := "
            + types(function["inputs"])
            + "\ndef outputs : List Ty := "
            + types(function["outputs"])
            + "\n"
        )
        for stage, raw in enumerate(records):
            text += (
                f"def S{stage} : Program := "
                + program(raw)
                + f"\ntheorem typed{stage} : checkTypes registry inputs outputs S{stage} = true := by decide\n"
            )
        text += "theorem firstAccepted : check registry inputs outputs S0 S1 = true := by decide\ntheorem secondAccepted : check registry inputs outputs S1 S2 = true := by decide\ntheorem firstTwo : Preserves registry inputs.length S0 S2 := check_composes firstAccepted secondAccepted\n"
        if records[2] == records[4]:
            text += "theorem actual {World Error : Type} (primitive : Prim → List Int → Int) (external : External World Error registry)\n    (arguments : List Int) (world : World) (arity : arguments.length = inputs.length) :\n    evalProgram (M primitive external) S0 arguments world = evalProgram (M primitive external) S4 arguments world :=\n  firstTwo Int World Error (M primitive external) arguments world arity\n"
            text += f"end F{index}\n"
            continue
        input_count = len(function["inputs"])
        input_names = [f"x{i}" for i in range(input_count)]
        args, vector = " ".join(input_names), lean_list(input_names)
        binders = f"({args} : Int)" if args else ""
        loops, envs = {}, {}
        for stage in (2, 3, 4):
            raw = records[stage]
            positions = [i for i, r in enumerate(raw) if "loop" in r]
            if len(positions) != 1:
                raise ValueError("exactly one top-level changed loop required")
            pos = positions[0]
            loops[stage] = raw[pos]["loop"]
            envs[stage] = prefix_environment(raw, input_names, capture)
            text += (
                f"def E{stage} (primitive : Prim → List Int → Int) {binders} : List Int := "
                + lean_list(envs[stage])
                + f"\ndef B{stage} : Program := "
                + program(loops[stage]["body"])
                + f"\ndef N{stage} : Program := "
                + program(raw[pos + 1 :])
                + "\n"
            )
        bounds = static_controls(records[2], capture, input_count=input_count)
        bulk_bounds = static_controls(records[3], capture, input_count=input_count)
        lower, upper, step = bounds
        if step <= 0 or upper < lower:
            raise ValueError("positive static source stride and ordered bounds required")
        trips = (upper - lower + step - 1) // step
        groups, remainder = divmod(trips, factor)
        if trips < 1 if identity_unroll else trips <= factor or remainder:
            raise ValueError("inline/empty/tail explicitly outside supported no-tail bulk mode")
        expected_bulk = bounds if identity_unroll else [lower, lower + step * trips, step * factor]
        if identity_unroll and records[2] != records[3]:
            raise ValueError("standalone segmentation requires identical source/unroll stages")
        if bulk_bounds != expected_bulk or static_controls(records[4], capture, input_count=input_count) != [
            lower,
            expected_bulk[1],
            expected_bulk[2] * tile,
        ]:
            raise ValueError("actual source/bulk/segmented control joins differ")
        initial_vectors = [[envs[stage][i] for i in loops[stage]["initial"]] for stage in (2, 3, 4)]
        if initial_vectors[1:] != [initial_vectors[0], initial_vectors[0]]:
            raise ValueError("ordered initial carry bindings differ across complete stages")
        if any(loops[stage]["carries"] != loops[2]["carries"] for stage in (3, 4)):
            raise ValueError("declared carried result type/order differs across complete stages")
        count = len(loops[2]["carries"])
        text += f"variable {{World Error : Type}} (primitive : Prim → List Int → Int)\n    (external : External World Error registry) {binders}\n"
        for stage in (2, 3, 4):
            loop, env = loops[stage], envs[stage]
            control_args = " ".join(env[i] for i in loop["bounds"])
            initial = lean_list([env[i] for i in loop["initial"]])
            text += f"theorem form{stage} (world : World) : evalProgram (M primitive external) S{stage} {vector} world =\n  (loopEval {len(loop['carries'])} (fun i carry w => evalProgram (M primitive external) B{stage} (i::carry++E{stage} primitive {args}) w)\n    (indexIterations {control_args}) {initial} world).bind\n      (fun result => evalProgram (M primitive external) N{stage} (result.1++E{stage} primitive {args}) result.2) := by rfl\n"
        if not identity_unroll:
            if flexible_parameters and any("loop" in row for row in loops[2]["body"]):
                from .compiler_loop_nested import body_group_text as nested_body_group

                text += nested_body_group(capture, loops, envs, count, factor, step, args)
            else:
                text += body_group_text(
                    capture, loops, envs, count, factor, step, args, max_calls=None if flexible_parameters else 16
                )
        text += scope_text(capture, loops, envs, count, args)
        text += outer_text(count, bulk_bounds[1], bulk_bounds[2], tile, args)
        initial = lean_list([envs[2][i] for i in loops[2]["initial"]])
        text += compose_text(
            count,
            bounds,
            bulk_bounds,
            factor,
            groups,
            tile,
            bits,
            initial,
            input_names,
            identity_unroll=identity_unroll,
        )
        text += f"end F{index}\n"
    defined = [i for i, f in enumerate(stages[0]["functions"]) if not f["declaration"]]

    def entry(i, stage):
        return f"{{inputs := F{i}.inputs, outputs := F{i}.outputs, body := F{i}.S{stage}}}"

    text += "def originalEntries : List Entry := " + lean_list([entry(i, 0) for i in defined]) + "\n"
    text += "def finalEntries : List Entry := " + lean_list([entry(i, 4) for i in defined]) + "\n"
    text += "theorem originalEntriesChecked : checkEntries registry originalEntries = true := by decide\n"
    text += "theorem finalEntriesChecked : checkEntries registry finalEntries = true := by decide\n"
    proof = "entries_nil (M primitive external)"
    for i in reversed(defined):
        proof = f"entries_cons (M primitive external) rfl rfl (F{i}.actual primitive external) ({proof})"
    text += "theorem complete {World Error : Type} (primitive : Prim → List Int → Int)\n"
    text += (
        "    (external : External World Error registry) : PreservesEntries (M primitive external) originalEntries finalEntries :=\n  "
        + proof
        + "\n"
    )
    return text + "end Merlin.GraphStatePipelineInstance\n"
