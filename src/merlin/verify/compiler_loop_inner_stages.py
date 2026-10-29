"""Scratch complete-buffer staging at one actual lexical loop depth."""


def code_type(value):
    from xdsl.dialects import builtin

    if type(value) is builtin.IndexType:
        return {"index": True}
    if type(value) is builtin.IntegerType:
        return {"bits": value.width.data}
    raise ValueError("scalar/index body input required")


def inherited(values):
    from xdsl.dialects import builtin

    from merlin.transforms.normalization import constant_integer

    return {
        i: constant_integer(value)
        for i, value in enumerate(values)
        if type(value.type) is builtin.IndexType and type(constant_integer(value)) is int
    }


def ancestry(loop):
    from xdsl.dialects import func, scf

    result = [loop]
    owner = loop.parent_op()
    while type(owner) is scf.ForOp:
        result.append(owner)
        owner = owner.parent_op()
    if type(owner) is not func.FuncOp:
        raise ValueError("actual structured function ancestry required")
    return owner, list(reversed(result))


def match_actual_path(module, target, selected):
    from xdsl.dialects import func, scf

    from merlin.verify.lean_graph_state_scalar_capture import metadata

    function, nodes = ancestry(selected)
    other = next(
        (op for op in target.body.block.ops if type(op) is func.FuncOp and op.sym_name.data == function.sym_name.data),
        None,
    )
    if other is None or not other.body.blocks:
        raise ValueError("actual whole function roster differs")
    context = {function.body.block: other.body.block}
    context.update(zip(function.body.block.args, other.body.block.args, strict=True))
    block = other.body.block
    positions = []
    for node in nodes:
        source = list(node.parent_block().ops)
        position = source.index(node)
        actual = list(block.ops)
        positions.append(position)
        if len(actual) < position:
            raise ValueError("actual lexical prefix truncated")
        for op, row in zip(source[:position], actual[:position], strict=True):
            if not op.is_structurally_equivalent(row, context):
                raise ValueError("actual lexical prefix differs")
        if node is selected:
            return function, nodes, block, context, positions
        if position >= len(actual):
            raise ValueError("actual ancestor missing")
        retained = actual[position]
        if type(retained) is not scf.ForOp or metadata(node) != metadata(retained):
            raise ValueError("actual ancestor metadata changed")
        if len(node.operands) != len(retained.operands) or any(
            context.get(value) != other for value, other in zip(node.operands, retained.operands, strict=True)
        ):
            raise ValueError("actual ancestor operands changed")
        context.update(zip(node.results, retained.results, strict=True))
        context[node.body.block] = retained.body.block
        context.update(zip(node.body.block.args, retained.body.block.args, strict=True))
        block = retained.body.block
    raise ValueError("actual selected loop missing")


def split(before, after, *, factor, depth, transform="unroll", tile=3):
    from xdsl.dialects import builtin, scf
    from xdsl.ir import SSAValue
    from xdsl.rewriter import Rewriter

    from merlin.transforms.certificates import ir_text, parse_ir
    from merlin.transforms.normalization import constant_integer
    from merlin.verify.compiler_loop_context import lexical_frame, prepare_local
    from merlin.verify.loop_transform_intake import Unavailable

    current, target = parse_ir(before), parse_ir(after)
    stages = []
    used = 0
    selected = [op for op in current.walk() if type(op) is scf.ForOp and len(ancestry(op)[1]) - 1 == depth]
    for loop in selected:
        controls = [constant_integer(value) for value in (loop.lb, loop.ub, loop.step)]
        if any(type(value) is not int for value in controls) or not 0 <= controls[0] <= controls[1] or controls[2] < 1:
            continue
        if any(type(value.type) is not builtin.IntegerType for value in loop.iter_args):
            raise ValueError("scalar word carries required")
        lower, upper, step = controls
        count = (upper - lower + step - 1) // step
        function, nodes, target_block, forward, path = match_actual_path(current, target, loop)
        mapping = {other: value for value, other in forward.items() if isinstance(value, SSAValue)}
        inverse = {value: other for other, value in mapping.items()}
        position = path[-1]
        width = (
            2
            if transform == "tile"
            else count * len(tuple(loop.body.block.ops))
            if count <= factor
            else 3 + (count % factor != 0)
        )
        chunk = list(target_block.ops)[position : position + width]
        if len(chunk) != width:
            raise ValueError("actual selected target chunk truncated")
        if transform == "tile" or count > factor:
            if not chunk or type(chunk[-1]) is not scf.ForOp or len(chunk[-1].results) != len(loop.results):
                raise ValueError("actual selected result geometry differs")
            outputs = list(chunk[-1].results)
        elif count == 0:
            outputs = [inverse[value] for value in loop.iter_args]
        else:
            state = [inverse[value] for value in loop.iter_args]
            body = list(loop.body.block.ops)
            stride = len(body)
            if count > 65536:
                raise Unavailable("selected inline lane work budget exceeded")
            for lane in range(count):
                items = chunk[lane * stride : (lane + 1) * stride]
                if len(items) != stride or len(items[0].results) != 1:
                    raise ValueError("actual inline lane geometry differs")
                local = {**inverse, body[0].parent_block().args[0]: items[0].results[0]}
                local.update(zip(loop.body.block.args[1:], state, strict=True))
                for op, row in zip(body[:-1], items[1:], strict=True):
                    if tuple(value.type for value in op.results) != tuple(value.type for value in row.results):
                        raise ValueError("actual inline result types differ")
                    local.update(zip(op.results, row.results, strict=True))
                state = [local[value] for value in body[-1].operands]
            outputs = state
        original = ir_text(current)
        frame = lexical_frame(current, loop)
        local = prepare_local(current, loop, factor=factor, index_bits=64, transform=transform, tile=tile)
        local = actual_wrapper(current, loop, local, chunk, outputs, inverse)
        rows = []
        for level, node in enumerate(nodes):
            if level == 0:
                initial = list(function.body.block.args)
                result_types = [code_type(t) for t in function.function_type.outputs]
            else:
                parent = nodes[level - 1]
                initial = [*parent.body.block.args, *lexical_frame(current, parent)]
                result_types = [code_type(value.type) for value in parent.results]
            rows.append(
                {
                    "entry": function.sym_name.data,
                    "position": path[level],
                    "scope_path": path[:level],
                    "inputs": [code_type(value.type) for value in initial],
                    "outputs": result_types,
                    "inherited": inherited(initial),
                    "frame": lexical_frame(current, node),
                    "carry_count": len(node.iter_args),
                }
            )
        clones = [op.clone(mapping) for op in chunk]
        replacements = [mapping[value] for value in outputs]
        parent_block = loop.parent_block()
        initial_frame = (
            [*nodes[-2].body.block.args, *lexical_frame(current, nodes[-2])]
            if len(nodes) > 1
            else list(function.body.block.args)
        )
        Rewriter.replace_op(loop, clones, replacements)
        current.verify()
        final = list(initial_frame)
        for op in list(parent_block.ops)[: position + width]:
            final = list(op.results) + final
        row = rows[-1]
        row.update(
            {
                "target_width": width,
                "return_slots": [final.index(value) for value in [*replacements, *frame]],
                "local": local,
            }
        )
        actual_after = ir_text(current)
        used += len(original.encode()) + len(actual_after.encode())
        if used > 32 * 1024 * 1024:
            raise Unavailable("complete actual context capture budget exceeded")
        for context in rows:
            context.update(before=original, after=actual_after)
        stages.append(
            {
                "before": original,
                "after": actual_after,
                "entry": function.sym_name.data,
                "path": path,
                "depth": depth,
                "leaf": row,
                "ancestors": rows[:-1],
            }
        )
    if ir_text(current) != ir_text(target):
        raise ValueError("complete actual target does not join this selected depth")
    if not stages:
        raise ValueError("actual selected-depth boundary did not change")
    return stages


def project_scope(row, universe):
    from merlin.transforms.certificates import parse_ir
    from merlin.verify.lean_graph_state_scalar_capture import project

    actual = project(parse_ir(row["before"]), parse_ir(row["after"]), context=universe)
    original = next(f for f in actual["functions"] if f["name"] == row["entry"])
    function = {**original, "inputs": row["inputs"], "outputs": row["outputs"]}
    for side in ("before", "after"):
        records = original[side]
        for position in row["scope_path"]:
            records = records[position]["loop"]["body"]
        function[side] = records
    return {**actual, "functions": [function]}


def actual_wrapper(module, selected, local, chunk, outputs, inverse):
    """Copy actual target syntax; proofs are emitted only after whole-module join."""
    from xdsl.dialects import arith, builtin, func, scf
    from xdsl.ir import Block, Region

    from merlin.transforms.certificates import ir_text

    frame = local["frame"]
    block = Block(arg_types=[value.type for value in frame])
    mapping = {inverse[value]: argument for value, argument in zip(frame, block.args, strict=True)}
    constants = [arith.ConstantOp.from_int_and_width(value, builtin.IndexType()) for value in local["controls"]]
    cloned = [op.clone(mapping) for op in chunk]
    if local["mode"] == "tile":
        outer = cloned[-1]
        if type(outer) is not scf.ForOp:
            raise ValueError("actual segmented outer loop missing")
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
            raise ValueError("actual bulk loop missing")
        bulk.operands = (constants[0].result, bulk.ub, bulk.step, *bulk.iter_args)
        if local["count"] % local["capture"]["factor"]:
            tail = cloned[-1]
            if type(tail) is not scf.ForOp:
                raise ValueError("actual remainder loop missing")
            tail.operands = (tail.lb, constants[1].result, constants[2].result, *tail.iter_args)
    block.add_ops([*constants, *cloned, func.ReturnOp(*(mapping[value] for value in outputs), *block.args)])
    function = func.FuncOp(
        local["wrapper_name"],
        ([value.type for value in frame], [value.type for value in selected.results] + [value.type for value in frame]),
        Region(block),
    )
    declarations = [op.clone() for op in module.body.block.ops if type(op) is func.FuncOp and not op.body.blocks]
    result = builtin.ModuleOp([*declarations, function])
    result.verify()
    return {**local, "after": ir_text(result)}
