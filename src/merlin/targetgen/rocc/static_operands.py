"""Bounded scalar observations for compact emitted LLVM command schedules.

These are static observations, not an expanded execution trace or a hazard
proof. Only a proved single-iteration loop body can bind an induction value to
one constant. Varying induction, exit/header values and arbitrary phi arguments
remain unknown. No work is proportional to a loop's trip count.
"""

from xdsl.dialects.builtin import IntegerAttr, IntegerType
from xdsl.ir import Operation


def fold_integer(value, bindings):
    """Fold closed fixed-width integer expressions, refusing poison flags/shifts."""
    cache = {}

    def visit(v, depth=0):
        if v in cache:
            return cache[v]
        if depth > 64 or len(cache) >= 4096 or not isinstance(v.type, IntegerType):
            return None
        width = v.type.width.data
        if not 0 < width <= 256:
            return None
        cache[v] = None
        if v in bindings:
            result = bindings[v]
        else:
            op = v.owner
            if not isinstance(op, Operation):
                return None
            flags = op.properties.get("overflowFlags")
            if flags is not None and (not isinstance(flags, IntegerAttr) or flags.value.data != 0):
                return None
            if op.name == "llvm.mlir.constant":
                attr = op.properties.get("value")
                if not isinstance(attr, IntegerAttr):
                    return None
                result = attr.value.data
            else:
                args = [visit(arg, depth + 1) for arg in op.operands]
                if not args or any(arg is None for arg in args):
                    return None
                a = args[0]
                if op.name in {"llvm.trunc", "llvm.sext", "llvm.zext"} and len(args) == 1:
                    result = a
                    if op.name == "llvm.zext":
                        result &= (1 << op.operands[0].type.width.data) - 1
                elif len(args) == 2:
                    b = args[1]
                    if op.name == "llvm.add":
                        result = a + b
                    elif op.name == "llvm.sub":
                        result = a - b
                    elif op.name == "llvm.mul":
                        result = a * b
                    elif op.name == "llvm.and":
                        result = a & b
                    elif op.name == "llvm.or":
                        result = a | b
                    elif op.name == "llvm.xor":
                        result = a ^ b
                    elif op.name in {"llvm.shl", "llvm.lshr", "llvm.ashr"} and 0 <= b < width:
                        result = (a << b if op.name == "llvm.shl" else
                                  a >> b if op.name == "llvm.ashr" else
                                  (a & ((1 << width) - 1)) >> b)
                    else:
                        return None
                else:
                    return None
        result &= (1 << width) - 1
        if result >= 1 << (width - 1):
            result -= 1 << width
        cache[v] = result
        return result

    return visit(value)


def counted_loop_observations(module):
    """Reuse the independent constant-loop proof, with no dynamic enumeration."""
    from merlin.perf.host_cfg_activity import _constant_loop
    from merlin.perf.host_cfg_index import prepare_host_cfg

    functions, bindings = [], {}
    for function in module.body.block.ops:
        if function.name != "llvm.func" or not function.body.blocks:
            continue
        blocks = tuple(function.body.blocks)
        if len(blocks) == 1:
            continue
        record = {"function": function.sym_name.data, "loops": [], "unavailable": []}
        functions.append(record)
        if len(blocks) > 4096:
            record["unavailable"].append("CFG exceeds bounded static observation")
            continue
        try:
            cfg = prepare_host_cfg(function)
        except (ValueError, KeyError, TypeError, AttributeError) as error:
            record["unavailable"].append(str(error))
            continue
        positions = {block: i for i, block in enumerate(blocks)}
        for header in blocks:
            for latch in cfg.predecessors[header]:
                if not cfg.dominance.dominates(header, latch):
                    continue
                nodes, pending = {header, latch}, [latch] if header is not latch else []
                while pending:
                    for pred in cfg.predecessors[pending.pop()]:
                        if pred not in nodes:
                            nodes.add(pred)
                            pending.append(pred)
                try:
                    row, nodes = _constant_loop(header, latch, nodes, cfg.predecessors, positions)
                except (ValueError, KeyError, TypeError, AttributeError) as error:
                    record["unavailable"].append(f"loop at block {positions[header]}: {error}")
                    continue
                record["loops"].append(row)
                if row["trip_count"] == 1:
                    induction = header.args[row["induction_argument_index"]]
                    # A pretested header observes both initial and final IV. A
                    # posttested singleton header executes its body once.
                    body = nodes if row.get("post_tested") else nodes - {header}
                    for block in body:
                        bindings.setdefault(block, {})[induction] = row["initial"]
    return {"basis": "original_llvm_cfg_constant_induction_v1", "functions": functions,
            "instruction_multiplicity": "static_sites_only",
            "varying_operands": "unknown; loop bounds are not operand or hazard proof"}, bindings
