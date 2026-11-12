"""Replace copies of a private, uniformly initialized buffer by destination fills.

This optional post-bufferization rewrite requires one direct fresh allocation,
one full identity-map uniform writer, and only same-block copies/deallocation
as remaining uses. No alias-producing operation, external call, partial writer,
reordered lifetime, or source destination-copy use is admitted. Each copy becomes
a clone of the original store-only scalar body at the original copy position.
The destination may be a strided subview; its placement and shape are unchanged.
"""

RUNNER_PRELUDE = r"""
def _fold_uniform_fill_copies(ctx, module):
    from torch_mlir import ir

    allocations = []
    def walk(op):
        for region in op.regions:
            for block in region.blocks:
                for child in list(block.operations):
                    walk(child.operation)
                    if child.operation.name == "memref.alloc":
                        allocations.append(child.operation)
    walk(module.operation)
    plans = []
    for alloc in allocations:
        if len(alloc.results) != 1 or len(alloc.operands):
            continue
        source = alloc.results[0]
        try:
            ty = ir.MemRefType(source.type)
        except (ValueError, TypeError):
            continue
        if ty.rank < 1 or any(n <= 0 for n in ty.shape):
            continue
        block = alloc.block
        positions = {op.operation: i for i, op in enumerate(block.operations)}
        users = list(source.uses)
        writers, copies, frees = [], [], []
        valid = True
        for use in users:
            op = use.owner.operation
            if op.block != block:
                valid = False
                break
            if op.name == "linalg.generic" and len(op.operands) == 1 and use.operand_number == 0:
                writers.append(op)
            elif op.name == "memref.copy" and use.operand_number == 0:
                copies.append(op)
            elif op.name == "memref.dealloc" and use.operand_number == 0:
                frees.append(op)
            else:
                valid = False
                break
        if not valid or len(writers) != 1 or not copies or len(frees) > 1:
            continue
        writer = writers[0]
        if len(writer.results) or len(writer.regions) != 1 or len(writer.regions[0].blocks) != 1:
            continue
        body = writer.regions[0].blocks[0]
        ops = list(body.operations)
        if len(body.arguments) != 1 or list(body.arguments[0].uses) or len(ops) != 1:
            continue
        term = ops[0].operation
        if term.name != "linalg.yield" or len(term.operands) != 1:
            continue
        maps = ir.ArrayAttr(writer.attributes["indexing_maps"])
        kinds = ir.ArrayAttr(writer.attributes["iterator_types"])
        if len(maps) != 1 or ir.AffineMapAttr(maps[0]).value != ir.AffineMap.get_identity(ty.rank, context=ctx):
            continue
        if len(kinds) != ty.rank or any(str(kind) != '#linalg.iterator_type<parallel>' for kind in kinds):
            continue
        if any(positions[copy] <= positions[writer] for copy in copies):
            continue
        if frees and any(positions[copy] >= positions[frees[0]] for copy in copies):
            continue
        for copy in copies:
            try:
                dest = ir.MemRefType(copy.operands[1].type)
            except (ValueError, TypeError):
                valid = False
                break
            if list(dest.shape) != list(ty.shape) or dest.element_type != ty.element_type or dest.memory_space != ty.memory_space:
                valid = False
                break
        if valid:
            plans.append((alloc, writer, copies, frees))
    count = 0
    with ctx:
        for alloc, writer, copies, frees in plans:
            for copy in copies:
                with ir.InsertionPoint(copy):
                    cloned = writer.clone()
                cloned.operands[0] = copy.operands[1]
                copy.erase()
                count += 1
            writer.erase()
            for free in frees:
                free.erase()
            alloc.erase()
    return count
"""

FEATURE = "fold_uniform_fill_copy"

MID_STAGE_SRC = r"""
if len(sys.argv) > 17 and sys.argv[17] == "1":
    _MID_STAGES.insert(0, ("fold_uniform_fill_copy", _fold_uniform_fill_copies))
"""


def require_report(stdout, work):
    """Refuse a selected feature silently bypassed by a lowering runner."""
    import json
    from pathlib import Path

    prefix = "OK fold_uniform_fill_copy "
    reports = [line.removeprefix(prefix) for line in stdout.splitlines() if line.startswith(prefix)]
    if len(reports) != 1 or not reports[0].isdigit():
        raise ValueError("selected uniform-fill-copy rewrite did not report exactly one nonnegative count")
    result = {"schema": "merlin.uniform_fill_copy.v1", "copies_folded": int(reports[0])}
    (Path(work) / "uniform_fill_copy.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
