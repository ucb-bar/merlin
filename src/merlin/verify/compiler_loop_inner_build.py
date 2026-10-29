# ruff: noqa: E501
"""Scratch assembly of inherited region facts into an unconditional actual root."""


def build(stages, before, after, scope, emitter, parent_emitter, *, middle=None, transform="unroll"):
    from merlin.transforms.certificates import parse_ir
    from merlin.verify import compiler_loop_atomic
    from merlin.verify.compiler_loop_context import reproject_stages
    from merlin.verify.lean_graph_pipeline_tail import query_roots
    from merlin.verify.lean_graph_state_pipeline import lean_list
    from merlin.verify.lean_graph_state_scalar import program, types
    from merlin.verify.lean_graph_state_scalar_capture import project

    leaves, universe = reproject_stages([stage["leaf"] for stage in stages], before, after)
    controls = leaves[0]["local"]["capture"]["index_operations"]
    texts = []
    proofs = {}
    bases = []
    boundary_proofs = {2: {}, 3: {}}
    for i, stage in enumerate(stages):
        row = stage["leaf"]
        if row["local"]["capture"]["index_operations"] != controls:
            raise ValueError("complete common control model required")
        row["projection"] = scope.project_scope(row, universe)
        query, info = emitter.emit_stage(row, universe, compiler_loop_atomic)
        whole = "Merlin.LocalWholeBridge"
        child = row
        renames = []
        for level, parent in reversed(list(enumerate(stage["ancestors"]))):
            parent["local"] = {**row["local"], "frame": parent["frame"]}
            shift = 1 + parent["carry_count"]
            parent["required_facts"] = {slot - shift: value for slot, value in child["inherited"].items()}
            if any(slot < 0 for slot in parent["required_facts"]):
                raise ValueError("inherited fact cannot concern an unknown IV/carry")
            parent["projection"] = scope.project_scope(parent, universe)
            rest, prefix, newwhole = (f"Merlin.InnerLevel{level}{suffix}" for suffix in ("Rest", "Prefix", "Whole"))
            added, _ = parent_emitter.emit(
                parent,
                universe,
                compiler_loop_atomic,
                emitter,
                child={"whole": whole, "inherited": child["inherited"]},
                rest_name=rest,
                prefix_name=prefix,
                whole_name=newwhole,
            )
            query += added
            renames.extend(
                [(rest, f"Layer{level}.Rest"), (prefix, f"Layer{level}.Prefix"), (newwhole, f"Layer{level}.Whole")]
            )
            whole = newwhole
            child = parent
        if child["inherited"]:
            raise ValueError("actual complete entry retained unproved inherited facts")
        namespace = f"Merlin.CompilerLoopInnerStep{i}"
        oldbase = info["base"]
        newbase = namespace + ".Model"
        query = query.replace(oldbase, newbase)
        bases.append(newbase)
        for old, new in [
            ("Merlin.LocalSourceAtomicBridge", "SourceAtomic"),
            ("Merlin.LocalTargetAtomicBridge", "TargetAtomic"),
            ("Merlin.LocalUnrollSuffixBridge", "Suffix"),
            ("Merlin.LocalRestBridge", "Rest"),
            ("Merlin.LocalPrefixFacts", "Prefix"),
            ("Merlin.LocalWholeBridge", "Whole"),
            *renames,
        ]:
            query = query.replace(old, namespace + "." + new)
        query = query.replace("merlin_unroll_inline_transform", f"merlin_inner_inline_fragment_{i}").replace(
            "merlin_unroll_tail_transform", f"merlin_inner_tail_fragment_{i}"
        )
        finalwhole = namespace + "." + dict(renames).get(whole, "Whole")
        entry = namespace + ".Entry"
        parent = stage["ancestors"][0] if stage["ancestors"] else row
        function = next(f for f in parent["projection"]["functions"] if f["name"] == stage["entry"])
        query += f"namespace {entry}\nopen Merlin.GraphState.Scalar Merlin.GraphState.Pipeline\n"
        query += f"abbrev inputs := {finalwhole}.inputs\n"
        query += (
            f"def S0 : Program := {program(function['before'])}\ndef S1 : Program := {program(function['after'])}\n"
        )
        query += f"theorem actual {{World Error : Type}} (primitive : Prim → List Int → Int) (external : External World Error {newbase}.registry) (arguments : List Int) (world : World) (arity : arguments.length = inputs.length) :\n"
        query += f"    evalProgram ({newbase}.M primitive external) S0 arguments world = evalProgram ({newbase}.M primitive external) S1 arguments world := by\n"
        query += f"  have same := {finalwhole}.actual primitive external arguments world arity\n"
        query += f"  simpa only [S0,S1,{finalwhole}.S0,{finalwhole}.S1] using same\nend {entry}\n"
        finalwhole = entry
        texts.append(query)
        proofs.setdefault(stage["entry"], []).append(finalwhole + ".actual")
        boundary = stage.get("boundary", 3 if transform == "tile" else 2)
        boundary_proofs[boundary].setdefault(stage["entry"], []).append(finalwhole + ".actual")
    imports = sorted(
        {
            word
            for text in texts
            for line in text.splitlines()
            if line.startswith("import ")
            for word in line.split()[1:]
        }
    )
    query = (
        "".join(f"import {name}\n" for name in imports)
        + "\n".join("\n".join(line for line in text.splitlines() if not line.startswith("import ")) for text in texts)
        + "\n"
    )
    actual = project(parse_ir(before), parse_ir(after), context=universe)
    query += "namespace Merlin.GraphStatePipelineInstance\nopen Merlin.GraphState.Scalar Merlin.GraphState.Pipeline\n"
    query += f"abbrev registry := {bases[0]}.registry\n"
    query += (
        "def M {World Error : Type} (primitive : Prim → List Int → Int) (external : External World Error registry) : Model Int World Error registry := "
        + bases[0]
        + ".M primitive external\n"
    )
    defined = []
    for i, function in enumerate(actual["functions"]):
        if function["declaration"]:
            continue
        defined.append(i)
        query += f"namespace F{i}\ndef inputs : List Ty := {types(function['inputs'])}\ndef outputs : List Ty := {types(function['outputs'])}\n"
        query += (
            f"def S0 : Program := {program(function['before'])}\ndef S1 : Program := {program(function['after'])}\n"
        )
        query += "theorem typed0 : checkTypes registry inputs outputs S0 = true := by decide\ntheorem typed1 : checkTypes registry inputs outputs S1 = true := by decide\n"
        query += "theorem actual {World Error : Type} (primitive : Prim → List Int → Int) (external : External World Error registry)\n"
        query += "    (arguments : List Int) (world : World) (arity : arguments.length = inputs.length) :\n"
        query += "    evalProgram (M primitive external) S0 arguments world = evalProgram (M primitive external) S1 arguments world := "
        steps = proofs.get(function["name"], [])
        if not steps:
            if function["before"] != function["after"]:
                raise ValueError("unproved actual entry changed")
            query += "rfl\n"
        else:
            proof = f"({steps[-1]} primitive external arguments world arity)"
            for step in reversed(steps[:-1]):
                proof = f"({step} primitive external arguments world arity).trans ({proof})"
            query += proof + "\n"
        query += f"end F{i}\n"
    for side, n in (("original", 0), ("final", 1)):
        query += (
            f"def {side}Entries : List Entry := "
            + lean_list([f"{{inputs := F{i}.inputs,outputs := F{i}.outputs,body := F{i}.S{n}}}" for i in defined])
            + "\n"
        )
        query += f"theorem {side}EntriesChecked : checkEntries registry {side}Entries = true := by decide\n"
    proof = "entries_nil (M primitive external)"
    for i in reversed(defined):
        proof = f"entries_cons (M primitive external) rfl rfl (F{i}.actual primitive external) ({proof})"
    query += "theorem complete {World Error : Type} (primitive : Prim → List Int → Int) (external : External World Error registry) :\n"
    query += (
        "    PreservesEntries (M primitive external) originalEntries finalEntries :=\n  "
        + proof
        + "\nend Merlin.GraphStatePipelineInstance\n"
    )
    if middle is not None:
        from merlin.verify.compiler_loop_context_chain import append_boundaries

        query = append_boundaries(
            query, actual, project(parse_ir(middle), parse_ir(middle), context=universe), boundary_proofs
        )
    query += "namespace Merlin.GraphGateProfileAdjacent\n"
    for boundary, first, last in (
        ((2, "original", "middle"), (3, "middle", "final"))
        if middle is not None
        else ((3 if transform == "tile" else 2, "original", "final"),)
    ):
        query += f"theorem complete{boundary} {{World Error : Type}} (primitive : Merlin.GraphState.Scalar.Prim → List Int → Int) (external : Merlin.GraphState.Pipeline.External World Error Merlin.GraphStatePipelineInstance.registry) :\n"
        query += f"    Merlin.GraphState.Pipeline.PreservesEntries (Merlin.GraphStatePipelineInstance.M primitive external) Merlin.GraphStatePipelineInstance.{first}Entries Merlin.GraphStatePipelineInstance.{last}Entries :=\n"
        proof = f"boundary{boundary}" if middle is not None else "complete"
        query += f"  Merlin.GraphStatePipelineInstance.{proof} primitive external\n"
    query += "end Merlin.GraphGateProfileAdjacent\n"
    return {"query": query, "roots": query_roots(query), "stages": stages, "projection": actual, "universe": universe}
