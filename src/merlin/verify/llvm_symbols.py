"""Resolve a defined LLVM function through structurally proved symbol aliases."""

from xdsl.dialects.builtin import SymbolRefAttr

from merlin.transforms.analysis_types import operation_name


def defined_function(module, symbol):
    """Return the original body; opaque, cyclic or type-changing aliases refuse.

    A supported alias consists solely of taking one flat symbol's address and
    returning that exact SSA value. This resolves names without rewriting the
    artifact, trusting an arbitrary pointer expression or skipping helper audits.
    """
    symbols = {}
    for op in module.body.block.ops:
        name = op.properties.get("sym_name") or op.attributes.get("sym_name")
        if isinstance(getattr(name, "data", None), str):
            symbols.setdefault(name.data, []).append(op)
    visited, required_types = set(), []
    while symbol not in visited:
        visited.add(symbol)
        matches = symbols.get(symbol, [])
        if len(matches) != 1:
            break
        op = matches[0]
        if operation_name(op) == "llvm.func":
            typ = op.properties.get("function_type")
            if op.body.blocks and all(typ == expected for expected in required_types):
                return op
            break
        if operation_name(op) != "llvm.mlir.alias" or len(op.regions) != 1:
            break
        region = op.regions[0]
        if len(region.blocks) != 1 or region.block.args:
            break
        body = list(region.block.ops)
        if (len(body) != 2 or operation_name(body[0]) != "llvm.mlir.addressof"
                or operation_name(body[1]) != "llvm.return" or len(body[0].results) != 1
                or list(body[1].operands) != [body[0].results[0]]):
            break
        target = body[0].properties.get("global_name")
        if not isinstance(target, SymbolRefAttr) or target.nested_references:
            break
        required_types.append(op.properties.get("alias_type"))
        symbol = target.root_reference.data
    raise ValueError("LLVM symbol has no uniquely defined, type-preserving function body")
