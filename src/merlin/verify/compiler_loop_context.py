# ruff: noqa: E501
# Keep literal Lean equations readable and byte-bound.
"""Checked local-loop composition over complete actual compiler module buffers.

Wrappers expose the entire lexical frame and all carries. Actual source and
actual target segments are joined independently to those wrappers, then suffix
capture and prefix factorization lift each local theorem to the complete entry.
No wrapper, candidate stage, producer replay or saved verdict is authority.
"""

from pathlib import Path


def lexical_frame(module, selected):
    """Recover the exact enclosing SSA frame without executing a region."""
    from xdsl.dialects import func, scf

    path = [selected]
    owner = selected.parent_op()
    while type(owner) is scf.ForOp:
        path.append(owner)
        owner = owner.parent_op()
    if type(owner) is not func.FuncOp or owner.parent_op() is not module:
        raise ValueError("selected loop is outside the actual function roster")
    frame = list(owner.body.block.args)
    for operation in reversed(path):
        for earlier in operation.parent_block().ops:
            if earlier is operation:
                break
            frame = [*earlier.results, *frame]
        if operation is not selected:
            frame = [*operation.body.block.args, *frame]
    return frame


def prepare_local(module, selected, *, factor, index_bits=64, context=(), transform="unroll", tile=3):
    """Expose source loop carries and its full original lexical capture."""
    from xdsl.dialects import arith, builtin, func, scf
    from xdsl.ir import Block, Region

    from merlin.transforms.certificates import ir_text
    from merlin.transforms.normalization import constant_integer

    if type(selected) is not scf.ForOp or type(factor) is not int or factor < 2 or index_bits != 64:
        raise ValueError("selected actual scf loop, factor and index64 required")
    frame = lexical_frame(module, selected)
    lower, upper, step = (constant_integer(v) for v in (selected.lb, selected.ub, selected.step))
    if any(type(v) is not int for v in (lower, upper, step)) or not 0 <= lower <= upper or step < 1:
        raise ValueError("actual local static nonnegative controls required")
    block = Block(arg_types=[v.type for v in frame])
    mapping = dict(zip(frame, block.args, strict=True))
    constants = [arith.ConstantOp.from_int_and_width(v, builtin.IndexType()) for v in (lower, upper, step)]
    clone = selected.clone(mapping)
    clone.operands = (*[op.result for op in constants], *clone.iter_args)
    block.add_ops([*constants, clone, func.ReturnOp(*clone.results, *block.args)])
    names = {op.sym_name.data for op in module.body.block.ops if type(op) is func.FuncOp}
    name = "__merlin_local_unroll"
    while name in names:
        name += "_"
    local = func.FuncOp(
        name, ([v.type for v in frame], [v.type for v in selected.results] + [v.type for v in frame]), Region(block)
    )
    declarations = [op.clone() for op in module.body.block.ops if type(op) is func.FuncOp and not op.body.blocks]
    wrapper = builtin.ModuleOp([*declarations, local])
    wrapper.verify()
    trips = (upper - lower + step - 1) // step
    mode = "tile" if transform == "tile" else "inline" if trips <= factor else "tail" if trips % factor else "bulk"
    return {
        "before": ir_text(wrapper),
        "frame": frame,
        "frame_types": [str(v.type) for v in frame],
        "controls": [lower, upper, step],
        "count": trips,
        "mode": mode,
        "wrapper_name": name,
        "capture": {"factor": factor, "index_bits": index_bits, "tile": tile},
    }


def split_sibling_stages(before, after, *, factor, index_bits=64, transform="unroll", tile=3):
    """Transplant actual target chunks as candidates; no stage is proof authority."""
    from xdsl.dialects import builtin, func, scf
    from xdsl.ir import SSAValue
    from xdsl.rewriter import Rewriter

    from merlin.transforms.certificates import ir_text, parse_ir
    from merlin.transforms.normalization import constant_integer

    current, target = parse_ir(before), parse_ir(after)
    current.verify()
    target.verify()
    stages = []
    stage_bytes = 0
    targets = {op.sym_name.data: op for op in target.body.block.ops if type(op) is func.FuncOp}
    for function in current.body.block.ops:
        if type(function) is not func.FuncOp or not function.body.blocks:
            continue
        counterpart = targets.get(function.sym_name.data)
        if counterpart is None or not counterpart.body.blocks:
            raise ValueError("actual complete function roster differs")
        selected = [op for op in function.body.block.ops if type(op) is scf.ForOp]
        for loop in selected:
            lb, ub, step = (constant_integer(value) for value in (loop.lb, loop.ub, loop.step))
            if any(type(value) is not int for value in (lb, ub, step)) or not 0 <= lb <= ub or step < 1:
                continue
            if any(type(value.type) is not builtin.IntegerType for value in loop.iter_args):
                raise ValueError("selected loop requires scalar word carried state")
            trips = (ub - lb + step - 1) // step
            source_ops = list(function.body.block.ops)
            target_ops = list(counterpart.body.block.ops)
            position = source_ops.index(loop)
            prefix_context = {function.body.block: counterpart.body.block}
            prefix_context.update(zip(function.body.block.args, counterpart.body.block.args, strict=True))
            for left, right in zip(source_ops[:position], target_ops[:position], strict=True):
                if not left.is_structurally_equivalent(right, prefix_context):
                    raise ValueError("actual prefix differs before selected rewrite")
            mapping = {right: left for left, right in prefix_context.items() if isinstance(left, SSAValue)}
            inverse = {right: left for left, right in mapping.items()}
            if min(trips, factor) > 65536:
                from .loop_transform_intake import Unavailable

                raise Unavailable("local lane syntax work resource budget exceeded")
            width = (
                2
                if transform == "tile"
                else (trips * len(tuple(loop.body.block.ops)))
                if trips <= factor
                else 3 + (trips % factor != 0)
            )
            chunk = target_ops[position : position + width]
            if len(chunk) != width:
                raise ValueError("actual selected target segment truncated")
            if transform == "tile":
                terminal = chunk[-1]
                if type(terminal) is not scf.ForOp or len(terminal.results) != len(loop.results):
                    raise ValueError("actual segmented result geometry differs")
                outputs = list(terminal.results)
            elif trips == 0:
                outputs = [inverse[value] for value in loop.iter_args]
            elif trips <= factor:
                state = [inverse[value] for value in loop.iter_args]
                body = list(loop.body.block.ops)
                stride = len(body)
                for lane in range(trips):
                    row = chunk[lane * stride : (lane + 1) * stride]
                    if len(row) != stride or len(row[0].results) != 1:
                        raise ValueError("actual inline lane geometry differs")
                    local = {**inverse, loop.body.block.args[0]: row[0].results[0]}
                    local.update(zip(loop.body.block.args[1:], state, strict=True))
                    for original, actual in zip(body[:-1], row[1:], strict=True):
                        if tuple(v.type for v in original.results) != tuple(v.type for v in actual.results):
                            raise ValueError("actual result slot geometry differs")
                        local.update(zip(original.results, actual.results, strict=True))
                    state = [local[value] for value in body[-1].operands]
                outputs = state
            else:
                terminal = chunk[-1]
                if type(terminal) is not scf.ForOp or len(terminal.results) != len(loop.results):
                    raise ValueError("actual bulk/tail result geometry differs")
                outputs = list(terminal.results)
            stage_before = ir_text(current)
            frame = lexical_frame(current, loop)
            local = prepare_local(
                current, loop, factor=factor, index_bits=index_bits, context=(target,), transform=transform, tile=tile
            )
            local = prepare_actual_target_wrapper(current, loop, local, target, chunk, outputs, inverse)
            clones = [operation.clone(mapping) for operation in chunk]
            replacements = [mapping[value] for value in outputs]
            Rewriter.replace_op(loop, clones, replacements)
            current.verify()
            final_frame = list(function.body.block.args)
            for operation in list(function.body.block.ops)[: position + width]:
                final_frame = list(operation.results) + final_frame
            slots = [final_frame.index(value) for value in [*replacements, *frame]]
            stage_after = ir_text(current)
            stage_bytes += len(stage_before.encode()) + len(stage_after.encode())
            if stage_bytes > 32 * 1024 * 1024:
                from .loop_transform_intake import Unavailable

                raise Unavailable("complete intermediate capture byte resource budget exceeded")
            stages.append(
                {
                    "before": stage_before,
                    "after": stage_after,
                    "entry": function.sym_name.data,
                    "position": position,
                    "return_slots": slots,
                    "target_width": width,
                    "source_frame_types": [str(value.type) for value in frame],
                    "local": local,
                    "authority": False,
                }
            )
    if ir_text(current) != ir_text(target):
        raise ValueError("complete actual final module does not join candidate local segments")
    return stages


def prepare_actual_target_wrapper(module, selected, local, target, chunk, outputs, inverse):
    from xdsl.dialects import arith, builtin, func, scf
    from xdsl.ir import Block, Region

    from merlin.transforms.certificates import ir_text, parse_ir
    from merlin.verify.lean_graph_pipeline_gate_projection import canonical
    from merlin.verify.lean_graph_pipeline_inline_emit import emit as inline_emit
    from merlin.verify.lean_graph_pipeline_tail_emit import emit as tail_emit
    from merlin.verify.lean_graph_state_pipeline_capture import index_operation
    from merlin.verify.lean_graph_state_scalar_capture import metadata, project

    frame = local["frame"]
    block = Block(arg_types=[value.type for value in frame])
    mapping = {inverse[value]: arg for value, arg in zip(frame, block.args, strict=True)}
    constants = [arith.ConstantOp.from_int_and_width(value, builtin.IndexType()) for value in local["controls"]]
    cloned = [operation.clone(mapping) for operation in chunk]
    if local["mode"] == "tile":
        outer = cloned[-1]
        if type(outer) is not scf.ForOp:
            raise ValueError("actual segmented outer loop location differs")
        outer.operands = (constants[0].result, constants[1].result, outer.step, *outer.iter_args)
        body = list(outer.body.block.ops)
        if len(body) != 4 or type(body[1]) is not arith.MinUIOp or type(body[2]) is not scf.ForOp:
            raise ValueError("actual segmented inner control geometry differs")
        body[1].operands = (body[1].lhs, constants[1].result)
        inner = body[2]
        inner.operands = (inner.lb, inner.ub, constants[2].result, *inner.iter_args)
    elif local["count"] > local["capture"]["factor"]:
        bulk = cloned[2]
        if type(bulk) is not scf.ForOp:
            raise ValueError("actual bulk loop location differs")
        bulk.operands = (constants[0].result, bulk.ub, bulk.step, *bulk.iter_args)
        if local["count"] % local["capture"]["factor"]:
            tail = cloned[-1]
            if type(tail) is not scf.ForOp:
                raise ValueError("actual tail loop location differs")
            tail.operands = (tail.lb, constants[1].result, constants[2].result, *tail.iter_args)
    block.add_ops([*constants, *cloned, func.ReturnOp(*(mapping[value] for value in outputs), *block.args)])
    local_function = func.FuncOp(
        local["wrapper_name"],
        ([value.type for value in frame], [value.type for value in selected.results] + [value.type for value in frame]),
        Region(block),
    )
    declarations = [
        operation.clone()
        for operation in module.body.block.ops
        if type(operation) is func.FuncOp and not operation.body.blocks
    ]
    wrapper = builtin.ModuleOp([*declarations, local_function])
    wrapper.verify()
    before = parse_ir(local["before"])
    universe = (module, target, before, wrapper)
    pair = project(before, wrapper, context=universe, _allow_captured_stride=True)
    descriptors = {}
    for current in universe:
        for operation in current.walk():
            control = index_operation(operation)
            if control is not None:
                descriptors[canonical(metadata(operation))] = control
    controls = [
        {"identity": i, **descriptors[canonical(row["descriptor"])]}
        for i, row in enumerate(pair["registry"])
        if canonical(row["descriptor"]) in descriptors
    ]
    capture = {**local["capture"], "stages": [pair], "index_operations": controls}
    if local["mode"] == "inline":
        query = inline_emit(capture, flexible_parameters=True)
    elif local["mode"] == "tail":
        query = tail_emit(capture, flexible_parameters=True)
    else:
        query = None
    return {
        **local,
        "after": ir_text(wrapper),
        "capture": capture,
        "query": query,
        "actual_target_segment": True,
        "obligation": "controlled wrapper target must join actual segment under explicit original control-slot equations; complete context lift required",
    }


def reproject_stages(stages, before, after, *, extra=()):
    from merlin.transforms.certificates import parse_ir
    from merlin.verify.compiler_loop_scaffold import auxiliary
    from merlin.verify.lean_graph_pipeline_gate_projection import canonical
    from merlin.verify.lean_graph_pipeline_inline_emit import emit as inline_emit
    from merlin.verify.lean_graph_pipeline_tail_emit import emit as tail_emit
    from merlin.verify.lean_graph_state_pipeline_capture import index_operation
    from merlin.verify.lean_graph_state_scalar_capture import metadata, project

    for row in stages:
        if row["local"]["mode"] == "bulk":
            row["local"]["auxiliary"] = auxiliary(row["local"], tile=3)
    texts = [before, after, *extra, *[value for row in stages for value in (row["before"], row["after"])]]
    texts += [row["local"]["auxiliary"] for row in stages if "auxiliary" in row["local"]]
    universe = tuple(parse_ir(raw) for raw in texts)
    registry = project(universe[0], universe[1], context=universe)["registry"]
    for row in stages:
        local = row["local"]
        wrappers = (parse_ir(local["before"]), parse_ir(local["after"]))
        pair = project(*wrappers, context=universe, _allow_captured_stride=True)
        if pair["registry"] != registry:
            raise ValueError("local wrapper introduced descriptors outside complete actual stage universe")
        descriptors = {}
        for module in (*universe, *wrappers):
            for op in module.walk():
                control = index_operation(op)
                if control is not None:
                    descriptors[canonical(metadata(op))] = control
        controls = [
            {"identity": i, **descriptors[canonical(item["descriptor"])]}
            for i, item in enumerate(registry)
            if canonical(item["descriptor"]) in descriptors
        ]
        capture = {**local["capture"], "stages": [pair], "index_operations": controls}
        query = (
            (inline_emit if local["mode"] == "inline" else tail_emit)(capture, flexible_parameters=True)
            if local["mode"] not in ("bulk", "tile")
            else None
        )
        local = {**local, "capture": capture, "query": query}
        if local["mode"] == "bulk":
            from merlin.verify.compiler_loop_scaffold import emit as emit_scaffold

            local = emit_scaffold(local, universe, controls)
        if local["mode"] == "tile":
            from merlin.verify.compiler_loop_tile_emit import emit as emit_tile

            local = emit_tile(local, universe, controls, tile=local["capture"]["tile"])
        row["local"] = local
    return stages, universe


def _prepare_query(before, after, *, factor, index_bits=64, middle=None, tile=3, transform="unroll"):
    from merlin.transforms.certificates import parse_ir
    from merlin.verify.lean_graph_pipeline_tail import query_roots
    from merlin.verify.lean_graph_state_pipeline import lean_list
    from merlin.verify.lean_graph_state_scalar import program, types
    from merlin.verify.lean_graph_state_scalar_capture import project

    from . import compiler_loop_atomic as atomic_module
    from .compiler_loop_context_emit import emit_stage
    from .lean_graph_input_admission import parse_admitted
    from .lean_graph_pipeline_gate_profile_prepare import _admit
    from .lean_graph_pipeline_gate_projection import project_lean_sources
    from .lean_graph_state_pipeline_capture import index_operation
    from .loop_transform_intake import ROOT, Unavailable, file_ref

    if type(tile) is not int or tile < 2 or transform not in ("unroll", "tile"):
        raise ValueError("explicit compiler loop transform and integer tile at least two required")
    if type(factor) is not int or factor < 2 or type(index_bits) is not int or index_bits != 64:
        raise ValueError("explicit compiler index64 and factor at least2 required")
    if any(type(raw) is not str for raw in (before, after)):
        raise ValueError("complete actual compiler module text required")
    if max(len(raw.encode()) for raw in (before, after)) > 4 * 1024 * 1024:
        raise Unavailable("actual compiler buffer byte resource budget exceeded")
    admitted = [
        parse_admitted(raw, parse_ir, calls="external-only")
        for raw in (before, after, *((middle,) if middle is not None else ()))
    ]
    for module in admitted:
        if (
            sum(
                1 + len(op.operands) + len(op.results) + sum(len(b.args) for r in op.regions for b in r.blocks)
                for op in module.walk()
            )
            > 2048
        ):
            raise Unavailable("complete scalar/SCF syntax resource budget exceeded")
    _admit(admitted)
    for op in admitted[0].walk():
        control = index_operation(op)
        if control is not None and control["kind"] != "index-constant":
            raise ValueError("source index arithmetic needs separate runtime range evidence")
    if before == after:
        raise ValueError("actual changing compiler boundary required")
    if middle is None:
        stages = split_sibling_stages(
            before, after, factor=factor, index_bits=index_bits, transform=transform, tile=tile
        )
        for row in stages:
            row["boundary"] = 3 if transform == "tile" else 2
    else:
        stages = split_sibling_stages(before, middle, factor=factor, index_bits=index_bits)
        for row in stages:
            row["boundary"] = 2
        later = split_sibling_stages(middle, after, factor=factor, index_bits=index_bits, transform="tile", tile=tile)
        for row in later:
            row["boundary"] = 3
        stages += later
    stages, universe = reproject_stages(stages, before, after, extra=(middle,) if middle is not None else ())
    if not stages:
        raise ValueError("at least one selected source loop required")
    projection = project(parse_ir(before), parse_ir(after), context=universe)
    texts = []
    proofs = {}
    model_bases = []
    boundary_proofs = {2: {}, 3: {}}
    first_controls = stages[0]["local"]["capture"]["index_operations"]
    for index, row in enumerate(stages):
        if row["local"]["capture"]["index_operations"] != first_controls:
            raise ValueError("one exact common control model required")
        query, info = emit_stage(row, universe, atomic_module)
        namespace = f"Merlin.CompilerLoopContextStep{index}"
        oldbase = info["base"]
        newbase = namespace + ".Model"
        query = query.replace(oldbase, newbase)
        for old, new in (
            ("Merlin.LocalSourceAtomicBridge", namespace + ".SourceAtomic"),
            ("Merlin.LocalTargetAtomicBridge", namespace + ".TargetAtomic"),
            ("Merlin.LocalUnrollSuffixBridge", namespace + ".Suffix"),
            ("Merlin.LocalRestBridge", namespace + ".Rest"),
            ("Merlin.LocalPrefixFacts", namespace + ".Prefix"),
            ("Merlin.LocalWholeBridge", namespace + ".Whole"),
            ("merlin_unroll_inline_transform", f"merlin_context_inline_fragment_{index}"),
            ("merlin_unroll_tail_transform", f"merlin_context_tail_fragment_{index}"),
        ):
            query = query.replace(old, new)
        texts.append(query)
        model_bases.append(newbase)
        proofs.setdefault(row["entry"], []).append(namespace + ".Whole.actual")
        boundary_proofs[row["boundary"]].setdefault(row["entry"], []).append(namespace + ".Whole.actual")
    imports = sorted(
        {
            word
            for query in texts
            for line in query.splitlines()
            if line.startswith("import ")
            for word in line.split()[1:]
        }
    )
    query = "".join(f"import {name}\n" for name in imports)
    query += (
        "\n".join("\n".join(line for line in text.splitlines() if not line.startswith("import ")) for text in texts)
        + "\n"
    )
    first = model_bases[0]
    query += "namespace Merlin.GraphStatePipelineInstance\nopen Merlin.GraphState.Scalar Merlin.GraphState.Pipeline\n"
    query += f"abbrev registry := {first}.registry\n"
    query += (
        "def M {World Error : Type} (primitive : Prim → List Int → Int) (external : External World Error registry) : Model Int World Error registry := "
        + first
        + ".M primitive external\n"
    )
    defined = []
    for index, function in enumerate(projection["functions"]):
        if function["declaration"]:
            continue
        defined.append(index)
        query += f"namespace F{index}\n"
        query += f"def inputs : List Ty := {types(function['inputs'])}\ndef outputs : List Ty := {types(function['outputs'])}\n"
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
                raise ValueError("unselected actual definition changed")
            query += "rfl\n"
        else:
            proof = f"({steps[-1]} primitive external arguments world arity)"
            for step in reversed(steps[:-1]):
                proof = f"({step} primitive external arguments world arity).trans ({proof})"
            query += proof + "\n"
        query += f"end F{index}\n"
    for side, stage in (("original", 0), ("final", 1)):
        entries = [f"{{inputs := F{i}.inputs,outputs := F{i}.outputs,body := F{i}.S{stage}}}" for i in defined]
        query += f"def {side}Entries : List Entry := " + lean_list(entries) + "\n"
        query += f"theorem {side}EntriesChecked : checkEntries registry {side}Entries = true := by decide\n"
    proof = "entries_nil (M primitive external)"
    for index in reversed(defined):
        proof = f"entries_cons (M primitive external) rfl rfl (F{index}.actual primitive external) ({proof})"
    query += "theorem complete {World Error : Type} (primitive : Prim → List Int → Int) (external : External World Error registry) :\n"
    query += "    PreservesEntries (M primitive external) originalEntries finalEntries :=\n  " + proof + "\n"
    query += "end Merlin.GraphStatePipelineInstance\n"
    if middle is not None:
        from merlin.verify.compiler_loop_context_chain import append_boundaries

        query = append_boundaries(
            query, projection, project(parse_ir(middle), parse_ir(middle), context=universe), boundary_proofs
        )
    query += "namespace Merlin.GraphGateProfileAdjacent\n"
    query += "theorem complete2 {World Error : Type} (primitive : Merlin.GraphState.Scalar.Prim → List Int → Int)\n"
    query += "    (external : Merlin.GraphState.Pipeline.External World Error Merlin.GraphStatePipelineInstance.registry) :\n"
    query += (
        "    Merlin.GraphState.Pipeline.PreservesEntries (Merlin.GraphStatePipelineInstance.M primitive external)\n"
    )
    query += (
        "      Merlin.GraphStatePipelineInstance.originalEntries Merlin.GraphStatePipelineInstance.finalEntries :=\n"
    )
    query += "  Merlin.GraphStatePipelineInstance.complete primitive external\nend Merlin.GraphGateProfileAdjacent\n"
    if middle is not None:
        query = query.replace(
            "namespace Merlin.GraphGateProfileAdjacent\ntheorem complete2",
            "namespace Merlin.GraphGateProfileAdjacent\ntheorem combined",
        )
        query += "namespace Merlin.GraphGateProfileAdjacent\n"
        for boundary, first, last in ((2, "original", "middle"), (3, "middle", "final")):
            query += f"theorem complete{boundary} {{World Error : Type}} (primitive : Merlin.GraphState.Scalar.Prim → List Int → Int) (external : Merlin.GraphState.Pipeline.External World Error Merlin.GraphStatePipelineInstance.registry) :\n"
            query += f"    Merlin.GraphState.Pipeline.PreservesEntries (Merlin.GraphStatePipelineInstance.M primitive external) Merlin.GraphStatePipelineInstance.{first}Entries Merlin.GraphStatePipelineInstance.{last}Entries :=\n"
            query += f"  Merlin.GraphStatePipelineInstance.boundary{boundary} primitive external\n"
        query += "end Merlin.GraphGateProfileAdjacent\n"
    elif transform == "tile":
        query = query.replace("theorem complete2 ", "theorem complete3 ")
    if len(query.encode()) > 4 * 1024 * 1024:
        raise Unavailable("operational proof query byte resource budget exceeded")
    parent = Path(__file__).parent
    paths = {
        Path(__file__),
        parent / "compiler_loop_context_emit.py",
        parent / "compiler_loop_atomic.py",
        parent / "compiler_loop_scaffold.py",
        parent / "compiler_loop_tile_emit.py",
        parent / "compiler_loop_context_chain.py",
        parent / "compiler_loop_nested.py",
        parent / "compiler_loop_resources.py",
        parent / "compiler_unroll_inline_nested.py",
        parent / "lean_graph_pipeline_inline_emit.py",
        parent / "lean_graph_pipeline_tail_emit.py",
        parent / "lean_graph_pipeline_tail.py",
        parent / "lean_graph_input_admission.py",
        parent / "lean_graph_pipeline_gate_profile_prepare.py",
        parent / "lean_graph_pipeline_gate_projection.py",
    }
    paths.update(parent.glob("lean_graph_state*.py"))
    sources = [file_ref(path) for path in sorted(paths)]
    sources.extend(file_ref(ref.path) for ref in project_lean_sources(imports))
    captured = []
    for row in stages:
        local = row["local"]
        captured.append(
            {
                **{key: value for key, value in row.items() if key != "local"},
                "local": {key: value for key, value in local.items() if key not in ("frame", "query")},
            }
        )
    return {
        "query": query,
        "roots": query_roots(query),
        "sources": sources,
        "final_root": ROOT,
        "capture": {"mode": "actual-loop-context", "stages": captured, "projection": projection},
    }


def prepare_query(before, after, *, factor, index_bits=64):
    return _prepare_query(before, after, factor=factor, index_bits=index_bits)


def prepare_chain(before, middle, after, *, factor, tile, index_bits=64):
    if type(middle) is not str or len(middle.encode()) > 4 * 1024 * 1024:
        from .loop_transform_intake import Unavailable

        raise Unavailable("complete intermediate compiler buffer resource budget exceeded")
    if before == middle or middle == after:
        raise ValueError("both actual composed compiler boundaries must change")
    return _prepare_query(before, after, factor=factor, tile=tile, index_bits=index_bits, middle=middle)


def prepare_tile_query(before, after, *, tile, index_bits=64):
    return _prepare_query(before, after, factor=2, tile=tile, index_bits=index_bits, transform="tile")
