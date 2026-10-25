"""Independent original-member and borrowed-call joins for sibling composition.

These checks add obligations to the ordinary owned-model audit. They do not
grant source eligibility, arithmetic equivalence, numerics or target effects.
"""

from pathlib import Path


def composition_members(composition, *, assigned):
    if (not isinstance(composition, dict)
            or composition.get("schema") != "merlin.original_sibling_source_composition.v1"
            or composition.get("original_source_denominator_unchanged") is not True
            or len(assigned) != 1
            or assigned[0].get("symbol") != composition.get("physical_symbol")
            or assigned[0].get("group") != composition.get("physical_group")):
        raise ValueError("sibling dispatch lacks its independently rederived source composition")
    members = composition.get("members")
    if (not isinstance(members, list) or len(members) < 2
            or len({row["source_group"] for row in members}) != len(members)
            or len({row["source_region_id"] for row in members}) != len(members)
            or set(composition["required_source_groups"]) - {row["source_group"] for row in members}):
        raise ValueError("sibling dispatch omits or duplicates original source members")
    return members


def attributed_commands(commands, members):
    """Read exact original IDs on real work operations, including optional roots."""
    from xdsl.dialects.builtin import ArrayAttr, StringAttr

    expected = {row["source_region_id"]: tuple(row["source_node_ids"]) for row in members}
    by_region = {region: {} for region in expected}
    for family, operations in commands.items():
        for operation in operations:
            region = operation.attributes.get("prov.region_id")
            nodes = operation.attributes.get("prov.source_node_ids")
            if (type(region) is not StringAttr or region.data not in expected
                    or type(nodes) is not ArrayAttr or any(type(node) is not StringAttr for node in nodes)
                    or tuple(node.data for node in nodes) != expected[region.data]):
                raise ValueError("actual sibling work lacks its complete original member identity")
            by_region[region.data].setdefault(family, set()).add(operation)
    return by_region


def mandatory_member_commands(function, commands, obligations, members, *, mandatory):
    """Keep every original operation's unavoidable-CFG obligation separately."""
    # Shared loads/stores are transport of all members, not distinct original
    # arithmetic roots. Only the source raw contraction family is attributed;
    # a required supporting/movement operation still refuses below.
    by_region = attributed_commands({"contraction": commands.get("contraction", set())}, members)
    by_group = {row["source_group"]: row for row in members}
    completed = []
    for obligation in obligations:
        member = by_group.get(obligation["source_group"])
        if (member is None or member["source_region_id"] != obligation["region_id"]
                or obligation["semantic_family"] != "contraction"):
            raise ValueError("sibling raw core cannot cover another original supporting source operation")
        # One member is still a separate original obligation. It may not use
        # work attributed to another member, even if only one root is required.
        completed.extend(mandatory(function, by_region[member["source_region_id"]], [obligation]))
    return completed


def borrowed_host_chain(module, function, composition, *, mandatory):
    """Prove the actual defined-wrapper chain reaches one unavoidable ABI leaf."""
    from merlin.verify.llvm_symbols import defined_function

    leaf = "_mlir_ciface_" + composition["borrowed_entry_symbol"]
    arity = 2 * len(composition["members"]) + 1

    def symbol(operation):
        return getattr(getattr(operation.properties.get("callee"), "root_reference", None), "data", None)

    definitions = {}
    for operation in module.body.block.ops:
        if operation.name == "llvm.func":
            name = getattr(operation.properties.get("sym_name"), "data", None)
            if name in definitions:
                raise ValueError("borrowed host chain contains duplicate LLVM symbols")
            definitions[name] = operation
    declared = definitions.get(leaf)
    if declared is None or any(block.ops for region in declared.regions for block in region.blocks):
        raise ValueError("borrowed host leaf is not the original external descriptor ABI")

    def visit(current, stack):
        name = current.properties["sym_name"].data
        if name in stack:
            raise ValueError("borrowed host chain contains recursion")
        paths = []
        for call in current.walk():
            if call.name != "llvm.call":
                continue
            callee = symbol(call)
            if callee == leaf:
                if (len(call.operands) != arity or call.results
                        or any(str(v.type) != "!llvm.ptr" for v in call.operands)):
                    raise ValueError("borrowed host call lost the complete pointer-only multi-output ABI")
                branches = [[name, leaf]]
            elif callee in definitions and any(block.ops for region in definitions[callee].regions
                                               for block in region.blocks):
                branches = visit(defined_function(module, callee), (*stack, name))
                branches = [[name, *path] for path in branches]
            else:
                continue
            if branches and not mandatory(current, {call}):
                raise ValueError("borrowed host call can be bypassed on a returning CFG path")
            if branches:
                from merlin.perf.host_cfg_index import prepare_host_cfg

                cfg = prepare_host_cfg(current)
                owner = next(block for block in cfg.blocks if call in block.ops)
                pending, reached = list(cfg.successors[owner]), set()
                while pending:
                    block = pending.pop()
                    if block is owner:
                        raise ValueError("borrowed host call can repeat in a counted or dynamic loop")
                    if block not in reached:
                        reached.add(block)
                        pending.extend(cfg.successors[block])
            paths.extend(branches)
        return paths

    paths = visit(function, ())
    if len(paths) != 1:
        raise ValueError("actual borrowed host callchain omits or repeats the original shared call")
    path = paths[0]
    required = [composition["host_entry_symbol"], composition["borrowed_entry_symbol"]]
    if [name for name in path if name in required] != required:
        raise ValueError("actual host bypasses the original source-owned sibling wrappers")
    return {"chain": path, "leaf": leaf, "descriptor_arguments": arity,
            "mandatory_call_roster": {composition["physical_symbol"]: 1}}


def verify_descriptor_bridge(entry, composition, *, built, target):
    """Reproduce the stock C bridge and join its actual compile and final link."""
    import json

    from merlin.common import invocation_record
    from merlin.llvmlower import toolchain
    from merlin.llvmlower.borrowed_effects import BorrowedBufferEffects
    from merlin.llvmlower.compilation_recipe import verify_completed_recipe
    from merlin.llvmlower.device_shim import emit_logical_translation_unit, logical_buffers
    from merlin.llvmlower.sibling_source_route import borrowed_bridge_source

    from .native_model_execution import _digest
    from .owned_host_model_execution import _pinned

    custody = (entry.get("custody") or {}).get("sibling_source")
    if (not isinstance(custody, dict)
            or custody.get("source_ir_sha256") != composition["source_ir_sha256"]
            or custody.get("interface_sha256") != composition["interface_sha256"]
            or custody.get("original_epilogues_and_all_live_outputs_preserved") is not True):
        raise ValueError("sibling codegen custody lost the original owner source and interface")
    BorrowedBufferEffects(**custody["borrowed_effects"])
    bridge = custody.get("descriptor_bridge")
    if not isinstance(bridge, dict) or set(bridge) != {"source", "object", "compilation"}:
        raise ValueError("sibling descriptor bridge has no actual source/object compilation custody")
    c_source, obj = _pinned(bridge, "source"), _pinned(bridge, "object")
    # Ordinary build sidecars use the path+SHA identity schema. The emitted
    # SHA remains authoritative; the native payload reader additionally wants
    # a size, as at dispatch's existing final-ELF boundary.
    sidecar_pin = built["device_sidecar"]
    sidecar = json.loads(_pinned({"sidecar": {
        **sidecar_pin, "size_bytes": Path(sidecar_pin["path"]).stat().st_size,
    }}, "sidecar").read_text())
    symbol = composition["physical_symbol"]
    interface = sidecar["expected_interfaces"][symbol]
    from hashlib import sha256

    if sha256(interface["mlir"].encode()).hexdigest() != composition["interface_sha256"]:
        raise ValueError("sibling descriptor bridge interface changed after source join")
    buffers, reason = logical_buffers(interface["mlir"])
    if buffers is None:
        raise ValueError("sibling descriptor bridge lost logical buffer ABI: " + reason)
    call = sidecar["call_buffers"][symbol]
    unit = emit_logical_translation_unit(target, {symbol: (buffers, call)},
                                         kernel_symbol_for=lambda _: entry["kernel_symbol"])
    output_positions = tuple(i for i, row in enumerate(call) if row["role"].startswith("out_"))
    expected = unit.text + "\n" + borrowed_bridge_source(
        symbol, composition["borrowed_entry_symbol"], call, output_positions)
    if unit.symbols != (symbol,) or c_source.read_text() != expected:
        raise ValueError("actual sibling descriptor bridge differs from stock complete ABI transport")
    observed = invocation_record.verify(_pinned(bridge, "compilation"))
    def pin(path):
        return {key: _digest(path)[key] for key in ("path", "sha256")}
    if (observed.get("stage") != "device_shim_compilation"
            or observed.get("inputs") != [pin(c_source)] or observed.get("outputs") != [pin(obj)]
            or Path(observed["executable"]["path"]) != toolchain.clang().resolve()
            or observed["argv"][-4:] != ["-c", str(c_source), "-o", str(obj)]):
        raise ValueError("sibling descriptor bridge did not use the original observed C compiler boundary")
    recipe = built["compilation_recipe"]
    compilation = verify_completed_recipe(Path(recipe["path"]), executable=Path(built["elf"]),
                                          expected_recipe_sha256=recipe["sha256"])
    linked = {(row["path"], row["sha256"]) for row in compilation["commands"][-1]["inputs"]}
    if (str(obj), _digest(obj)["sha256"]) not in linked:
        raise ValueError("sibling descriptor bridge object is not an actual final ELF link input")
    return {"source": _digest(c_source), "object": _digest(obj),
            "compilation": _digest(_pinned(bridge, "compilation")),
            "scope": "stock pointer/identity-descriptor transport, observed compiler, actual final link input"}
