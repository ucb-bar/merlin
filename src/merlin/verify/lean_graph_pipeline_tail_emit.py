# ruff: noqa: E501
# Keep literal Lean equations readable and byte-bound.
"""Source-only complete bulk/remainder query; no SDK or evaluator execution.

Local cloned-body, lexical frame and suffix obligations are closed declarations.
The final root quantifies over the same primitive and partial external meaning.
All output is UNCOMPILED until a separately authorized kernel context checks it.
"""

from .lean_graph_state_pipeline import lean_list, prefix_environment
from .lean_graph_state_pipeline_emit import body_group_text, entry_arguments, header, static_controls, unpack
from .lean_graph_state_scalar import budget, program, types


def _symbols(indices):
    return lean_list([f"(.var {i})" for i in indices])


def _scope(name, left, right, logical, left_frame, right_frame, statement, introductions):
    selected = [logical.index(value) for value in left_frame]
    other = [logical.index(value) for value in right_frame]
    return (
        f"def {name}L : List Expr := {_symbols(selected)}\n"
        f"def {name}R : List Expr := {_symbols(other)}\n"
        f"theorem {name}Checked : Scope.check registry {name}L {name}R {left} {right} = true := by decide\n"
        + statement
        + " := by\n"
        + introductions
        + f"  have checked := Scope.check_sound {name}Checked (M primitive external) {lean_list(logical)} world\n"
        + f"  simpa only [{name}L,{name}R,E2,E3,evalExpr,List.cons_append,List.nil_append,List.map_cons,List.map_nil,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD_some] using checked\n"
    )


def emit(capture, *, flexible_parameters=False):
    pair = capture["stages"][0]
    factor, bits = capture["factor"], capture["index_bits"]
    text = header(capture).replace(
        "namespace Merlin.GraphStatePipelineInstance", "namespace Merlin.GraphStatePipelineTailInstance"
    )
    text = text.replace("set_option linter.unusedSimpArgs false\n", "")
    text = "import Merlin.GraphStatePipelineTail\n" + text
    for fi, function in enumerate(pair["functions"]):
        if function["declaration"]:
            continue
        before, after = function["before"], function["after"]
        for raw in (before, after):
            budget(raw, len(function["inputs"]), [0])
        text += f"namespace F{fi}\ndef inputs : List Ty := {types(function['inputs'])}\n"
        text += f"def outputs : List Ty := {types(function['outputs'])}\n"
        text += f"def S2 : Program := {program(before)}\ndef S3 : Program := {program(after)}\n"
        for stage in (2, 3):
            text += f"theorem typed{stage} : checkTypes registry inputs outputs S{stage} = true := by decide\n"
        if before == after:
            text += "theorem actual {World Error : Type} (primitive : Prim → List Int → Int) (external : External World Error registry)\n"
            text += "    (arguments : List Int) (world : World) (_arity : arguments.length = inputs.length) :\n"
            text += "    evalProgram (M primitive external) S2 arguments world = evalProgram (M primitive external) S3 arguments world := rfl\n"
            text += f"end F{fi}\n"
            continue
        input_count = len(function["inputs"])
        args = [f"x{i}" for i in range(input_count)]
        arg_text = " ".join(args)
        binder = f"({' '.join(args)} : Int)" if args else ""
        start = len(text)
        if not flexible_parameters and input_count != 2:
            raise ValueError("changed entry requires two scalar arguments")
        pos = [[i for i, row in enumerate(raw) if "loop" in row] for raw in (before, after)]
        if len(pos[0]) != 1 or len(pos[1]) != 2 or pos[1][1] != pos[1][0] + 1:
            raise ValueError("one original loop and adjacent bulk/remainder loops required")
        p, q = pos[0][0], pos[1][0]
        loops = {2: before[p]["loop"], 3: after[q]["loop"]}
        tail = after[q + 1]["loop"]
        if any("loop" in row for loop in (*loops.values(), tail) for row in loop["body"]):
            raise ValueError("nested original/tail bodies require separate bridges")
        count = len(loops[2]["carries"])
        if (not flexible_parameters and not 1 <= count <= 2) or any(
            loop["carries"] != loops[2]["carries"] for loop in (loops[3], tail)
        ):
            raise ValueError("one/two identical ordered carry types required")
        envs = {2: prefix_environment(before, args, capture), 3: prefix_environment(after, args, capture)}
        lower, upper, step = static_controls(before, capture, input_count=input_count)
        if step <= 0 or upper < lower:
            raise ValueError("nonnegative static bounds and positive stride required")
        trips = (upper - lower + step - 1) // step
        groups, remainder = divmod(trips, factor)
        cutoff = lower + groups * factor * step
        if trips <= factor or remainder == 0:
            raise ValueError("this profile requires changed nonempty bulk AND nonempty remainder")
        if upper + step * factor >= 2 ** (bits - 1):
            raise ValueError("signed bound proof outside selected width")
        if static_controls(after, capture, input_count=input_count) != [lower, cutoff, factor * step]:
            raise ValueError("actual bulk cutoff/stride differ")
        prior = [f"p{i}" for i in range(count)]
        carry = [f"c{i}" for i in range(count)]
        tail_env = prior + envs[3]
        actual_tail_bounds = [tail_env[i] for i in tail["bounds"]]
        expected = [f"({cutoff} : Int)", f"({upper} : Int)", f"({step} : Int)"]
        if actual_tail_bounds != expected or tail["initial"] != list(range(count)):
            raise ValueError("tail controls or forwarded bulk carry are not exact")
        initial = [[envs[s][i] for i in loops[s]["initial"]] for s in (2, 3)]
        if initial[0] != initial[1]:
            raise ValueError("bulk initial bindings differ")
        for s in (2, 3):
            text += f"def E{s} (primitive : Prim → List Int → Int) (x0 x1 : Int) : List Int := {lean_list(envs[s])}\n"
            text += f"def B{s} : Program := {program(loops[s]['body'])}\n"
        text += f"def BT : Program := {program(tail['body'])}\n"
        text += f"def N2 : Program := {program(before[p + 1 :])}\n"
        text += f"def N3 : Program := {program(after[q + 1 :])}\n"
        text += f"def NT : Program := {program(after[q + 2 :])}\n"
        text += "variable {World Error : Type} (primitive : Prim → List Int → Int)\n    (external : External World Error registry) (x0 x1 : Int)\n"
        for s, bounds in ((2, (lower, upper, step)), (3, (lower, cutoff, step * factor))):
            text += f"theorem form{s} (world : World) : evalProgram (M primitive external) S{s} [x0,x1] world =\n"
            text += f"  (loopEval {count} (fun i c w => evalProgram (M primitive external) B{s} (i::c++E{s} primitive x0 x1) w)\n"
            text += f"    (indexIterations {' '.join(map(str, bounds))}) {lean_list(initial[0])} world).bind\n"
            text += (
                f"    (fun r => evalProgram (M primitive external) N{s} (r.1++E{s} primitive x0 x1) r.2) := by rfl\n"
            )
        text += body_group_text(
            capture, loops, envs, count, factor, step, "x0 x1", max_calls=None if flexible_parameters else 16
        )
        intro = unpack("previous", "previousSize", count, "p") + unpack("carry", "arity", count, "c")
        statement = f"theorem tail_body (iv : Int) (previous carry : List Int) (world : World) (previousSize : previous.length = {count}) (arity : carry.length = {count}) :\n"
        statement += "  evalProgram (M primitive external) B2 (iv::carry++E2 primitive x0 x1) world =\n"
        statement += "  evalProgram (M primitive external) BT (iv::carry++(previous++E3 primitive x0 x1)) world"
        text += _scope(
            "tailScope",
            "B2",
            "BT",
            ["iv", *carry, *prior, *envs[3]],
            ["iv", *carry, *envs[2]],
            ["iv", *carry, *prior, *envs[3]],
            statement,
            intro,
        )
        statement = f"theorem suffix_same (previous carry : List Int) (world : World) (previousSize : previous.length = {count}) (arity : carry.length = {count}) :\n"
        statement += "  evalProgram (M primitive external) N2 (carry++E2 primitive x0 x1) world =\n"
        statement += "  evalProgram (M primitive external) NT (carry++(previous++E3 primitive x0 x1)) world"
        text += _scope(
            "suffixScope",
            "N2",
            "NT",
            [*carry, *prior, *envs[3]],
            [*carry, *envs[2]],
            [*carry, *prior, *envs[3]],
            statement,
            intro,
        )
        text += (
            f"theorem after_bulk (previous : List Int) (world : World) (previousSize : previous.length = {count}) :\n"
        )
        text += "  evalProgram (M primitive external) N3 (previous++E3 primitive x0 x1) world =\n"
        text += f"  (loopEval {count} (fun i c w => evalProgram (M primitive external) BT (i::c++(previous++E3 primitive x0 x1)) w)\n"
        text += f"    (indexIterations {cutoff} {upper} {step}) previous world).bind\n"
        text += "    (fun r => evalProgram (M primitive external) NT (r.1++(previous++E3 primitive x0 x1)) r.2) := by\n"
        text += unpack("previous", "previousSize", count, "p") + "  rfl\n"
        text += f"def bounds : Merlin.GraphLoopsUnroll.Bounds := {{bits := {bits},lower := {lower},upper := {upper},step := {step},factor := {factor},count := {trips},groups := {groups},remainder := {remainder}}}\n"
        text += "theorem boundsChecked : bounds.Valid := by decide\n"
        text += """theorem constantsFit :
    Merlin.GraphLoopsSegment.FitsSigned bounds.bits bounds.lower ∧
    Merlin.GraphLoopsSegment.FitsSigned bounds.bits bounds.upper ∧
    Merlin.GraphLoopsSegment.FitsSigned bounds.bits bounds.step ∧
    Merlin.GraphLoopsSegment.FitsSigned bounds.bits (bounds.step*bounds.factor) ∧
    Merlin.GraphLoopsSegment.FitsSigned bounds.bits (bounds.lower+bounds.step*(bounds.groups*bounds.factor)) :=
  Merlin.GraphLoopsUnroll.Bounds.constants boundsChecked
theorem lanesFit (group lane : Nat) (inGroup : group < bounds.groups) (inLane : lane < bounds.factor) :
    bounds.lower+(bounds.step*bounds.factor)*group+bounds.step*lane =
      bounds.lower+bounds.step*(group*bounds.factor+lane) ∧
    Merlin.GraphLoopsSegment.FitsSigned bounds.bits (bounds.step*lane) ∧
    Merlin.GraphLoopsSegment.FitsSigned bounds.bits (bounds.lower+(bounds.step*bounds.factor)*group+bounds.step*lane) :=
  Merlin.GraphLoopsUnroll.Bounds.lane_control boundsChecked inGroup inLane
theorem tailIndicesFit (ordinal : Nat) (inside : ordinal < bounds.remainder) :
    Merlin.GraphLoopsSegment.FitsSigned bounds.bits
      (bounds.lower+bounds.step*(bounds.groups*bounds.factor+ordinal)) ∧
    Merlin.GraphLoopsSegment.FitsSigned bounds.bits
      (bounds.lower+bounds.step*(bounds.groups*bounds.factor+ordinal)+bounds.step) := by
  apply Merlin.GraphLoopsUnroll.Bounds.inline_control boundsChecked
  have partition := boundsChecked.2.2.2.2.2.1
  omega
"""
        b2 = "(fun i c w => evalProgram (M primitive external) B2 (i::c++E2 primitive x0 x1) w)"
        b3 = "(fun i c w => evalProgram (M primitive external) B3 (i::c++E3 primitive x0 x1) w)"
        bt = "(fun i c w => evalProgram (M primitive external) BT (i::c++(previous++E3 primitive x0 x1)) w)"
        text += "theorem actual_inputs (world : World) : evalProgram (M primitive external) S2 [x0,x1] world = evalProgram (M primitive external) S3 [x0,x1] world := by\n"
        text += f"  have bodies : ∀ index values w, values.length = {count} →\n"
        text += "      evalProgram (M primitive external) B3 (Int.ofNat index::values++E3 primitive x0 x1) w =\n"
        text += f"      loopEval {count} {b2} ((progression index {step} {factor}).map Int.ofNat) values w := by\n"
        text += "    intro index values w size\n    simpa [progression,Int.ofNat_eq_natCast,Int.add_assoc] using body_group primitive external x0 x1 (Int.ofNat index) values w size\n"
        text += f"  have loops : loopEval {count} {b2} (indexIterations {lower} {upper} {step}) {lean_list(initial[0])} world =\n"
        text += f"      (loopEval {count} {b3} (indexIterations {lower} {cutoff} {step * factor}) {lean_list(initial[0])} world).bind\n"
        text += f"        (fun r => loopEval {count} {b2} (indexIterations {cutoff} {upper} {step}) r.1 r.2) :=\n"
        text += f"    Tail.ordered_bulk_tail {count} {b2} {b3} {lower} {upper} {step} {factor} {groups} {remainder}\n"
        text += f"    (by decide) (by decide) (by decide) bodies {lean_list(initial[0])} world\n"
        text += "  rw [form2,form3,loops]\n"
        text += f"  cases bulkRun : loopEval {count} {b3} (indexIterations {lower} {cutoff} {step * factor}) {lean_list(initial[0])} world with\n"
        text += "  | error failure => rfl\n  | ok result =>\n    rcases result with ⟨previous,afterBulk⟩\n"
        text += f"    have previousSize := loopEval_arity {count} _ _ _ _ bulkRun\n"
        text += "    simp only [Except.bind]\n    rw [after_bulk primitive external x0 x1 previous afterBulk previousSize]\n"
        text += f"    have tails := loopEval_congr {count} {b2} {bt}\n"
        text += "      (fun i c w size => tail_body primitive external x0 x1 i previous c w previousSize size)\n"
        text += f"      (indexIterations {cutoff} {upper} {step}) previous afterBulk\n    rw [tails]\n"
        text += f"    cases tailRun : loopEval {count} {bt} (indexIterations {cutoff} {upper} {step}) previous afterBulk with\n"
        text += "    | error failure => rfl\n    | ok result =>\n"
        text += f"      exact suffix_same primitive external x0 x1 previous result.1 result.2 previousSize (loopEval_arity {count} _ _ _ _ tailRun)\n"
        text += "theorem actual (arguments : List Int) (world : World) (arity : arguments.length = inputs.length) :\n"
        text += "    evalProgram (M primitive external) S2 arguments world = evalProgram (M primitive external) S3 arguments world := by\n"
        text = text[:start] + text[start:].replace("(x0 x1 : Int)", binder).replace(
            "primitive x0 x1", f"primitive {arg_text}"
        ).replace("external x0 x1", f"external {arg_text}").replace("S2 [x0,x1]", f"S2 {lean_list(args)}").replace(
            "S3 [x0,x1]", f"S3 {lean_list(args)}"
        )
        text += entry_arguments(input_count, "actual_inputs")
        text += f"end F{fi}\n"
    defined = [i for i, f in enumerate(pair["functions"]) if not f["declaration"]]
    for name, stage in (("originalEntries", 2), ("finalEntries", 3)):
        text += (
            f"def {name} : List Entry := "
            + lean_list([f"{{inputs := F{i}.inputs,outputs := F{i}.outputs,body := F{i}.S{stage}}}" for i in defined])
            + "\n"
        )
        text += f"theorem {name}Checked : checkEntries registry {name} = true := by decide\n"
    proof = "entries_nil (M primitive external)"
    for i in reversed(defined):
        proof = f"entries_cons (M primitive external) rfl rfl (F{i}.actual primitive external) ({proof})"
    text += "theorem complete {World Error : Type} (primitive : Prim → List Int → Int) (external : External World Error registry) :\n"
    text += "    PreservesEntries (M primitive external) originalEntries finalEntries :=\n  " + proof + "\n"
    text += "end Merlin.GraphStatePipelineTailInstance\n"
    text += "theorem merlin_unroll_tail_transform {World Error : Type}\n"
    text += "    (primitive : Merlin.GraphState.Scalar.Prim → List Int → Int)\n"
    text += "    (external : Merlin.GraphState.Pipeline.External World Error Merlin.GraphStatePipelineTailInstance.registry) :\n"
    text += (
        "    Merlin.GraphState.Pipeline.PreservesEntries (Merlin.GraphStatePipelineTailInstance.M primitive external)\n"
    )
    text += "      Merlin.GraphStatePipelineTailInstance.originalEntries Merlin.GraphStatePipelineTailInstance.finalEntries :=\n"
    text += "  Merlin.GraphStatePipelineTailInstance.complete primitive external\n"
    return text
