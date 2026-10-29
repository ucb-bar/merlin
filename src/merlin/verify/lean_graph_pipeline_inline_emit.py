# ruff: noqa: E501
# Keep literal Lean equations readable and byte-bound.
"""Pure empty/full-inline complete-entry query generation, never execution.

Only the bounded selected lane syntax is expanded. Every partial call and final
observer is split using the same external meaning and local result arity law.
"""

from .lean_graph_state_pipeline import SIMPS, lean_list, prefix_environment
from .lean_graph_state_pipeline_emit import entry_arguments, header, static_controls, symbolic_body, unpack
from .lean_graph_state_scalar import budget, program, types


def _calls(calls, *, max_calls=16):
    if max_calls is not None and len(calls) > max_calls:
        raise ValueError("inline source proof admits at most sixteen retained call occurrences")
    text = ""
    for depth, call in enumerate(calls):
        # Earlier simplification can already close identical continuations.
        # Case splits are needed only for remaining goals, never assumptions.
        pad = "  " + "    " * depth
        text += pad + "all_goals\n"
        pad += "  "
        index, identity = call["index"], call["identity"]
        args, world = lean_list(call["arguments"]), call["world"]
        text += f"{pad}cases step{index} : external.run {identity} {args} {world} with\n"
        text += f"{pad}| error failure =>\n{pad}  rcases failure with ⟨error,terminal⟩\n"
        text += f"{pad}  simp only [step{index},{SIMPS},loopEval]\n"
        text += f"{pad}| ok result =>\n{pad}  rcases result with ⟨values,w{index}⟩\n"
        text += f"{pad}  have size : values.length = {call['outputs']} := by\n"
        text += f"{pad}    have localArity := external.arity {identity} {args} {world} values w{index} step{index}\n"
        text += f"{pad}    simpa only [registry,signature,List.getElem?_cons_succ,List.getElem?_cons_zero,Option.getD_some,List.length_cons,List.length_nil] using localArity\n"
        text += unpack("values", "size", call["outputs"], f"v{index}_", pad + "  ")
        text += f"{pad}  simp only [step{index},{SIMPS},loopEval]\n"
    return text


def emit(capture, *, flexible_parameters=False):
    pair = capture["stages"][0]
    factor, bits = capture["factor"], capture["index_bits"]
    text = header(capture).replace(
        "namespace Merlin.GraphStatePipelineInstance", "namespace Merlin.GraphStatePipelineInlineInstance"
    )
    text = "import Merlin.GraphStatePipelineInline\n" + text.replace("set_option linter.unusedSimpArgs false\n", "")
    text = text.replace("import Merlin.GraphStatePipelineLoops\n", "")
    if not flexible_parameters:
        text = text.replace("import Merlin.GraphStatePipelineScope\n", "")
    for fi, function in enumerate(pair["functions"]):
        if function["declaration"]:
            continue
        before, after = function["before"], function["after"]
        for raw in (before, after):
            budget(raw, len(function["inputs"]), [0])
        text += f"namespace F{fi}\ndef inputs : List Ty := {types(function['inputs'])}\n"
        text += f"def outputs : List Ty := {types(function['outputs'])}\n"
        text += f"def S0 : Program := {program(before)}\ndef S1 : Program := {program(after)}\n"
        for stage in (0, 1):
            text += f"theorem typed{stage} : checkTypes registry inputs outputs S{stage} = true := by decide\n"
        if before == after:
            text += "theorem actual {World Error : Type} (primitive : Prim → List Int → Int) (external : External World Error registry)\n"
            text += "    (arguments : List Int) (world : World) (_arity : arguments.length = inputs.length) :\n"
            text += "    evalProgram (M primitive external) S0 arguments world = evalProgram (M primitive external) S1 arguments world := rfl\n"
            text += f"end F{fi}\n"
            continue
        positions = [i for i, row in enumerate(before) if "loop" in row]
        input_count = len(function["inputs"])
        args = [f"x{i}" for i in range(input_count)]
        arg_text = " ".join(args)
        binder = f"({' '.join(args)} : Int)" if args else ""
        start = len(text)
        if (
            (not flexible_parameters and input_count != 2)
            or len(positions) != 1
            or (not flexible_parameters and any("loop" in row for row in after))
        ):
            raise ValueError("one changed top-level source loop/two arguments/no target loop required")
        pos = positions[0]
        loop = before[pos]["loop"]
        count = len(loop["carries"])
        if (not flexible_parameters and not 1 <= count <= 2) or (
            not flexible_parameters and any("loop" in row for row in loop["body"])
        ):
            raise ValueError("region-free inline body with one/two carries required")
        env = prefix_environment(before, args, capture)
        lower, upper, step = static_controls(before, capture, input_count=input_count)
        if step <= 0 or upper < lower:
            raise ValueError("static nonnegative ordered controls required")
        trips = (upper - lower + step - 1) // step
        if trips > factor:
            raise ValueError("bulk/tail form outside empty/full-inline profile")
        if upper + step * factor >= 2 ** (bits - 1):
            raise ValueError("selected signed control bound exceeded")
        initial = [env[i] for i in loop["initial"]]
        primitive_name = "primitive" if any("primitive" in v for v in env) else "_primitive"
        text += f"def E ({primitive_name} : Prim → List Int → Int) (x0 x1 : Int) : List Int := {lean_list(env)}\n"
        text += f"def B : Program := {program(loop['body'])}\ndef N : Program := {program(before[pos + 1 :])}\n"
        text += f"def bounds : Merlin.GraphLoopsUnroll.Bounds := {{bits := {bits},lower := {lower},upper := {upper},step := {step},factor := {factor},count := {trips},groups := {trips // factor},remainder := {trips % factor}}}\n"
        text += "theorem boundsChecked : bounds.Valid := by decide\n"
        text += "theorem isInline : bounds.count ≤ bounds.factor := by decide\n"
        text += """theorem inlineIndicesFit (ordinal : Nat) (inside : ordinal < bounds.count) :
    Merlin.GraphLoopsSegment.FitsSigned bounds.bits (bounds.lower+bounds.step*ordinal) ∧
    Merlin.GraphLoopsSegment.FitsSigned bounds.bits (bounds.lower+bounds.step*ordinal+bounds.step) :=
  Merlin.GraphLoopsUnroll.Bounds.inline_control boundsChecked inside
"""
        text += "variable {World Error : Type} (primitive : Prim → List Int → Int)\n    (external : External World Error registry) (x0 x1 : Int)\n"
        body = "(fun i c w => evalProgram (M primitive external) B (i::c++E primitive x0 x1) w)"
        nxt = "(fun c w => evalProgram (M primitive external) N (c++E primitive x0 x1) w)"
        text += "theorem source_form (world : World) : evalProgram (M primitive external) S0 [x0,x1] world =\n"
        text += (
            f"  (loopEval {count} {body} (indexIterations {lower} {upper} {step}) {lean_list(initial)} world).bind\n"
        )
        text += "    (fun r => evalProgram (M primitive external) N (r.1++E primitive x0 x1) r.2) := rfl\n"
        if trips == 0:
            text += "theorem empty_body_never_runs (world : World) :\n"
            text += f"    loopEval {count} {body} (indexIterations {lower} {upper} {step}) {lean_list(initial)} world = .ok ({lean_list(initial)},world) :=\n"
            text += f"  Inline.empty_no_body {count} _ {lower} {upper} {step} (by decide) (by decide) _ world rfl\n"
        if flexible_parameters and any("loop" in row for row in loop["body"]):
            from .compiler_unroll_inline_nested import emit as nested_inline

            text += nested_inline(capture, before, after, loop, env, initial, trips, lower, upper, step, args)
        else:
            text += "theorem actual_inputs (world : World) : evalProgram (M primitive external) S0 [x0,x1] world =\n    evalProgram (M primitive external) S1 [x0,x1] world := by\n"
            text += "  rw [source_form]\n"
            if trips == 0:
                text += "  rw [empty_body_never_runs]\n"
            elif flexible_parameters:
                text += f"  have indices : indexIterations ({lower} : Int) {upper} {step} = (progression {lower} {step} {trips}).map Int.ofNat :=\n"
                text += f"    indexIterations_count (lower := {lower}) (upper := {upper}) (step := {step}) (count := {trips}) (by decide) (by decide)\n"
                text += "  rw [indices]\n"
            else:
                text += f"  rw [Inline.counted_continuation {count} {body} {nxt} {lower} {upper} {step} {trips} (by decide) (by decide)]\n"
            text += f"  simp only [B,N,E,S1,progression,List.map_cons,List.map_nil,{SIMPS},loopEval]\n"
            values = list(initial)
            calls = []
            control_map = {r["identity"]: r for r in capture["index_operations"]}
            for lane in range(trips):
                values, more = symbolic_body(
                    loop["body"],
                    [f"({lower + lane * step} : Int)", *values, *env],
                    control_map,
                    len(calls),
                    "world" if not calls else f"w{len(calls) - 1}",
                )
                calls += more
            _, more = symbolic_body(
                before[pos + 1 :],
                [*values, *env],
                control_map,
                len(calls),
                "world" if not calls else f"w{len(calls) - 1}",
            )
            calls += more
            text += _calls(calls, max_calls=None if flexible_parameters else 16)
        text += "theorem actual (arguments : List Int) (world : World) (arity : arguments.length = inputs.length) :\n"
        text += "    evalProgram (M primitive external) S0 arguments world = evalProgram (M primitive external) S1 arguments world := by\n"
        text = text[:start] + text[start:].replace("(x0 x1 : Int)", binder).replace(
            "primitive x0 x1", f"primitive {arg_text}"
        ).replace("S0 [x0,x1]", f"S0 {lean_list(args)}").replace("S1 [x0,x1]", f"S1 {lean_list(args)}")
        text += entry_arguments(input_count, "actual_inputs")
        text += f"end F{fi}\n"
    defined = [i for i, f in enumerate(pair["functions"]) if not f["declaration"]]
    for name, stage in (("originalEntries", 0), ("finalEntries", 1)):
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
    text += "end Merlin.GraphStatePipelineInlineInstance\n"
    text += "theorem merlin_unroll_inline_transform {World Error : Type}\n"
    text += "    (primitive : Merlin.GraphState.Scalar.Prim → List Int → Int)\n"
    text += "    (external : Merlin.GraphState.Pipeline.External World Error Merlin.GraphStatePipelineInlineInstance.registry) :\n"
    text += "    Merlin.GraphState.Pipeline.PreservesEntries (Merlin.GraphStatePipelineInlineInstance.M primitive external)\n"
    text += "      Merlin.GraphStatePipelineInlineInstance.originalEntries Merlin.GraphStatePipelineInlineInstance.finalEntries :=\n"
    text += "  Merlin.GraphStatePipelineInlineInstance.complete primitive external\n"
    return text
