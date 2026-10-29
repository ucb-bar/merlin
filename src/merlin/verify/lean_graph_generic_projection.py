"""SOURCE CANDIDATE ONLY: complete pair projection, never acceptance authority.

Structured non-SCF operations are exact partial opaque closures. SCF loops keep
their actual operational lower/upper/stride/carry/body representation. This
candidate does not call a compiler gate or claim its Int model covers tensors.
"""

from xdsl.dialects import arith, builtin, func, scf

from . import lean_graph_generic_admission as admission
from .lean_graph_generic_exact import canonical, exact, metadata, selected_primitive
from .lean_graph_generic_regions import capture_regions

MAX_WEIGHTED_SYNTAX = 65536
MAX_RAW_BYTES = 4 * 1024 * 1024
MAX_BLOCK_RECORDS = 4096
MAX_REGION_DEPTH = 32


def project_pair(before, after):
    modules = (before, after)
    for module in modules:
        admission.admit_module(module)
        module.verify()
        if (
            sum(
                1
                + len(op.operands)
                + len(op.results)
                + sum(len(block.args) for region in op.regions for block in region.blocks)
                for op in module.walk()
            )
            > MAX_WEIGHTED_SYNTAX
        ):
            raise ValueError("complete module weighted syntax resource budget exceeded")
    if metadata(before) != metadata(after) or exact(before.location) != exact(after.location):
        raise ValueError("actual module metadata changed")
    functions = [tuple(module.body.block.ops) for module in modules]
    if len(functions[0]) != len(functions[1]) or any(type(op) is not func.FuncOp for rows in functions for op in rows):
        raise ValueError("complete exact function roster required")
    for left, right in zip(*functions, strict=True):
        if metadata(left) != metadata(right) or exact(left.location) != exact(right.location):
            raise ValueError("actual function interface/frame changed")
    tokens = sorted(
        {
            canonical(exact(value.type))
            for module in modules
            for op in module.walk()
            for value in (
                *op.operands,
                *op.results,
                *(arg for region in op.regions for block in region.blocks for arg in block.args),
            )
        }
    )

    def ty(value):
        if type(value) is builtin.IndexType:
            return {"index": True}
        if type(value) is builtin.IntegerType and value.signedness.data == builtin.Signedness.SIGNLESS:
            return {"bits": value.width.data}
        return {"opaque": tokens.index(canonical(exact(value)))}

    # Gather lexical environments once; nested structured operations included in
    # a non-SCF closure are represented inside that exact descriptor, not also
    # exposed as independent execution events.
    requests_by_module = []

    def gather(block, captured, requests):
        environment = [*block.args, *captured]
        for op in block.ops:
            if type(op) is scf.ForOp:
                if len(op.body.blocks) != 1:
                    raise ValueError("operational SCF loop needs one actual body block")
                gather(op.body.block, environment, requests)
            elif op.regions:
                requests.append((op, tuple(environment)))
            environment = [*op.results, *environment]

    for module, rows in zip(modules, functions, strict=True):
        requests = []
        for function in rows:
            if len(function.body.blocks) > 1:
                raise ValueError("whole function CFG is unsupported")
            if function.body.blocks:
                gather(function.body.block, [], requests)
        requests_by_module.append(requests)
    closures = {}
    for module, requests in zip(modules, requests_by_module, strict=True):
        for (op, _), value in zip(requests, capture_regions(module, requests), strict=True):
            closures[op] = value

    registry = {}

    def block_code(block, captured, terminator, depth=0):
        if depth > MAX_REGION_DEPTH or len(tuple(block.ops)) > MAX_BLOCK_RECORDS:
            raise ValueError("Program block/depth resource budget exceeded")
        environment = [*block.args, *captured]
        operations = tuple(block.ops)
        if not operations or type(operations[-1]) is not terminator:
            raise ValueError("actual exact terminator required")
        records = []
        for op in operations:
            if op.successors:
                raise ValueError("operational CFG unsupported")

            def uses(values):
                if any(value not in environment for value in values):
                    raise ValueError(
                        "actual SSA use lacks lexical availability at "
                        + op.name
                        + ": "
                        + repr([value for value in values if value not in environment])
                        + " op="
                        + str(op)[:1000]
                        + " env="
                        + str(len(environment))
                    )
                return [environment.index(value) for value in values]

            if type(op) in (func.ReturnOp, scf.YieldOp):
                if op is not operations[-1] or type(op) is not terminator or op.attributes or op.properties:
                    raise ValueError("changed exact terminator")
                records.append({"ret": uses(op.operands)})
                continue
            primitive = selected_primitive(op)
            if primitive is not None:
                records.append({"pure": primitive, "arguments": uses(op.operands)})
            elif type(op) is scf.ForOp:
                owner = op.step.owner
                if (
                    op.attributes
                    or op.properties
                    or type(owner) is not arith.ConstantOp
                    or type(owner.result.type) is not builtin.IndexType
                    or not isinstance(owner.value, builtin.IntegerAttr)
                    or owner.value.value.data <= 0
                ):
                    raise ValueError("actual operational loop requires existing positive constant stride profile")
                carries = [ty(value.type) for value in op.iter_args]
                if [ty(value.type) for value in op.body.block.args] != [{"index": True}, *carries]:
                    raise ValueError("exact operational IV/carry slots required")
                records.append(
                    {
                        "loop": {
                            "bounds": uses((op.lb, op.ub, op.step)),
                            "initial": uses(op.iter_args),
                            "carries": carries,
                            "body": block_code(op.body.block, environment, scf.YieldOp, depth + 1),
                        }
                    }
                )
            else:
                closure = closures.get(op)
                descriptor = (
                    closure.descriptor if closure is not None else {**metadata(op), "location": exact(op.location)}
                )
                inputs = closure.captures if closure is not None else tuple(op.operands)
                token = canonical(descriptor)
                row = {
                    "descriptor": descriptor,
                    "inputs": [ty(value.type) for value in inputs],
                    "outputs": [ty(value.type) for value in op.results],
                }
                if token in registry and registry[token] != row:
                    raise ValueError("same exact opaque key has inconsistent actual ABI")
                registry[token] = row
                records.append({"effect": token, "arguments": uses(inputs), "outputs": row["outputs"]})
            environment = [*op.results, *environment]
        return records

    projected = []
    for stage, rows in enumerate(functions):
        try:
            projected.append(
                [
                    {
                        "name": function.sym_name.data,
                        "header": metadata(function),
                        "inputs": [ty(value) for value in function.function_type.inputs],
                        "outputs": [ty(value) for value in function.function_type.outputs],
                        "declaration": not bool(function.body.blocks),
                        "body": block_code(function.body.block, [], func.ReturnOp) if function.body.blocks else None,
                    }
                    for function in rows
                ]
            )
        except ValueError as error:
            raise ValueError(f"actual stage{stage}: {error}") from error
    ordered = sorted(registry)
    identities = {token: i for i, token in enumerate(ordered)}

    def number(records):
        for row in records:
            if "effect" in row:
                row["effect"] = identities[row["effect"]]
            elif "loop" in row:
                number(row["loop"]["body"])

    for rows in projected:
        for function in rows:
            if not function["declaration"]:
                number(function["body"])
    return {
        "authority": False,
        "status": "SOURCE_CANDIDATE_NO_GATE",
        "registry": [registry[token] for token in ordered],
        "functions": projected,
        "opaque_region_closures": len(closures),
        "type_tokens": tokens,
    }
