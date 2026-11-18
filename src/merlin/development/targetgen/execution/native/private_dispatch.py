"""Independent head invocation and actual stock descriptor-shim custody checks."""

from __future__ import annotations

import json
from pathlib import Path

# ---- private_head_dispatch ---------------------------------------------------------------------


def verify_bridge(entry, component, *, built, target, collection=None):
    from merlin.common.process import invocation_records as invocation_record
    from merlin.compiler.lower.device.device_shim import emit_logical_translation_unit, logical_buffers
    from merlin.compiler.lower.device.head import validate_head_member_ranges
    from merlin.compiler.lower.model import toolchain
    from merlin.compiler.lower.model.compilation_recipe import verify_completed_recipe
    from merlin.development.targetgen.execution.host.model_execution import _pinned
    from merlin.development.targetgen.execution.native.native_model_execution import _digest

    whole = component.get("whole_batch", False)
    if type(whole) is not bool:
        raise ValueError("head bridge has no explicit original batch ownership")
    kind = "whole_logical_batch" if whole else "stock_logical_batch_slices"
    products = validate_head_member_ranges(component)
    custody = (entry.get("custody") or {}).get("head_source")
    if (
        not isinstance(custody, dict)
        or custody.get("source_ir_sha256") != component["source_ir_sha256"]
        or custody.get("interface_sha256") != component["interface_sha256"]
        or custody.get("matrix_invocations") != component["matrix_invocations"]
        or custody.get("bridge_kind") != kind
        or whole
        and (custody.get("kernel_invocations") != 1 or custody.get("mathematical_products") != products)
    ):
        raise ValueError("head bridge lacks exact original source/interface/invocation custody")
    bridge = custody.get("descriptor_bridge")
    if not isinstance(bridge, dict) or set(bridge) != {"source", "object", "compilation"}:
        raise ValueError("head bridge lacks actual source/object/compiler custody")
    source, obj = _pinned(bridge, "source"), _pinned(bridge, "object")
    sidecar_pin = built["device_sidecar"]
    sidecar = json.loads(
        _pinned(
            {"sidecar": {**sidecar_pin, "size_bytes": Path(sidecar_pin["path"]).stat().st_size}}, "sidecar"
        ).read_text()
    )
    kernels = built["device_artifacts"]["kernels"]
    symbols = {}
    for row in sidecar["routed"]:
        name = row["symbol"]
        chosen = sidecar["expected_interfaces"][name]
        from hashlib import sha256

        if sha256(chosen["mlir"].encode()).hexdigest() != chosen["sha256"]:
            raise ValueError("head matrix interface changed after source admission")
        buffers, reason = logical_buffers(chosen["mlir"])
        if buffers is None:
            raise ValueError("head matrix interface refused: " + reason)
        symbols[name] = buffers, sidecar["call_buffers"][name]
    unit = emit_logical_translation_unit(target, symbols, kernel_symbol_for=lambda name: kernels[name]["kernel_symbol"])
    expected = unit.text
    if collection is not None:
        from merlin.compiler.lower.device.sibling_source_route import borrowed_bridge_source

        for sibling in collection.get("components", []):
            if sibling.get("schema") != "merlin.original_sibling_source_composition.v1":
                continue
            call = sidecar["call_buffers"][sibling["physical_symbol"]]
            outputs = tuple(i for i, row in enumerate(call) if row["role"].startswith("out_"))
            expected += "\n" + borrowed_bridge_source(
                sibling["physical_symbol"], sibling["borrowed_entry_symbol"], call, outputs
            )
    if set(unit.symbols) != set(symbols) or source.read_text() != expected or unit.skipped:
        raise ValueError("head descriptor bridge differs from exact stock batch-slice transport")
    call = sidecar["call_buffers"][component["physical_symbol"]]
    h = component["matrix_invocations"]
    m, k, n = component["geometry_m_k_n"]
    if [r["shape"] for r in call] != [[h, m, k], [h, k, n], [h, m, n]] or [r["dtype"] for r in call] != [
        "i8",
        "i8",
        "i32",
    ]:
        raise ValueError("head slice bridge no longer covers every original product/view")
    buffers, reason = logical_buffers(sidecar["expected_interfaces"][component["physical_symbol"]]["mlir"])
    expected_shapes = [[h, m, k], [h, k, n], [h, m, n]] if whole else [[m, k], [k, n], [m, n]]
    if buffers is None or [list(b.shape) for b in buffers] != expected_shapes:
        raise ValueError("head interface lost original whole/sliced shape ownership: " + (reason or "shape drift"))
    observed = invocation_record.verify(_pinned(bridge, "compilation"))

    def pin(path):
        return {key: _digest(path)[key] for key in ("path", "sha256")}

    if (
        observed.get("stage") != "device_shim_compilation"
        or observed.get("inputs") != [pin(source)]
        or observed.get("outputs") != [pin(obj)]
        or Path(observed["executable"]["path"]) != toolchain.clang().resolve()
        or observed["argv"][-4:] != ["-c", str(source), "-o", str(obj)]
    ):
        raise ValueError("head bridge lost the original observed C compiler boundary")
    recipe = built["compilation_recipe"]
    compilation = verify_completed_recipe(
        Path(recipe["path"]), executable=Path(built["elf"]), expected_recipe_sha256=recipe["sha256"]
    )
    linked = {(row["path"], row["sha256"]) for row in compilation["commands"][-1]["inputs"]}
    if (str(obj), _digest(obj)["sha256"]) not in linked:
        raise ValueError("head descriptor bridge object is not an actual final ELF input")
    result = {
        "source": _digest(source),
        "object": _digest(obj),
        "compilation": _digest(_pinned(bridge, "compilation")),
        "matrix_invocations": h,
        "scope": "exact stock positive-count loop over disjoint original head slices; "
        "one synchronous matrix artifact invocation per original product",
    }
    if products != h:
        result["scope"] = (
            "exact stock positive-count loop over source-proven grouped row slices; "
            "every original product retains its independently rederived row range"
        )
    if whole:
        result.update(
            kernel_invocations=1,
            mathematical_products=products,
            scope="exact ranked whole-batch interface; one kernel call covers every original product",
        )
    return result


def mandatory_heads(function, commands, obligations, component, *, mandatory):
    """Keep every source obligation; whole-batch object budgets are checked separately."""
    from merlin.compiler.lower.device.head import validate_head_member_ranges

    members = component["members"]
    products = validate_head_member_ranges(component)
    h = component["matrix_invocations"]
    if type(h) is not int or h < 2:
        raise ValueError("head dispatch requires every original positive-count slice")
    by_group = {r["source_group"]: r for r in members}
    if len(by_group) != products:
        raise ValueError("head dispatch repeats an original product")
    whole = component.get("whole_batch", False)
    if whole and (
        {r["source_group"] for r in obligations} != set(by_group)
        or component.get("mathematical_products") != products
        or component.get("kernel_invocations") != 1
    ):
        raise ValueError("whole head dispatch lacks its complete independently admitted product roster")
    result = []
    for obligation in obligations:
        member = by_group.get(obligation["source_group"])
        if (
            member is None
            or obligation["region_id"] != member["source_region_id"]
            or obligation["semantic_family"] != "contraction"
        ):
            raise ValueError("raw head invocation cannot cover original supporting source work")
        checked = mandatory(function, commands, [obligation])
        ranges = (
            {key: member[key] for key in ("projected_head", "row_start", "row_count")}
            if "projected_head" in member
            else {}
        )
        result.extend(
            {
                **row,
                **ranges,
                "head_slice": member["head"],
                "invocation_basis": (
                    "independently verified complete whole-batch interface and object budget"
                    if whole
                    else "independently verified stock batch-slice descriptor loop"
                ),
            }
            for row in checked
        )
    return result


# ---- private_sibling_dispatch ------------------------------------------------------------------
# Independent original-member and borrowed-call joins for sibling composition.
#
# These checks add obligations to the ordinary owned-model audit. They do not
# grant source eligibility, arithmetic equivalence, numerics or target effects.


def composition_members(composition, *, assigned):
    if isinstance(composition, dict) and composition.get("schema") == "merlin.original_head_source_composition.v1":
        members, physical = [], set()
        for component in composition["components"]:
            rows = [
                r
                for r in assigned
                if r["symbol"] == component["physical_symbol"] and r["group"] == component["physical_group"]
            ]
            if component.get("schema") == "merlin.original_sibling_source_composition.v1":
                members.extend(composition_members(component, assigned=rows))
                if component["physical_group"] in physical:
                    raise ValueError("joint dispatch repeats physical work")
                physical.add(component["physical_group"])
                continue
            h = component["matrix_invocations"]
            if (
                len(rows) != 1
                or rows[0]["group"] != component["physical_group"]
                or type(h) is not int
                or h < 2
                or len(component["members"]) != h
                or [m["head"] for m in component["members"]] != list(range(h))
                or component["physical_group"] in physical
            ):
                raise ValueError("head dispatch lost original per-slice invocation custody")
            members.extend(component["members"])
            physical.add(component["physical_group"])
        ordinary = composition["ordinary_joined_groups"]
        indices = [r["source_group"] for r in members]
        if (
            len(set(indices)) != len(indices)
            or set(indices).intersection(ordinary.values())
            or set(physical).intersection(ordinary)
            or set(physical) | set(ordinary) != {r["group"] for r in assigned}
            or not set(composition["required_source_groups"]) <= set(indices) | set(ordinary.values())
        ):
            raise ValueError("head dispatch omits or duplicates original source work")
        return members
    if isinstance(composition, dict) and composition.get("schema") == "merlin.original_sibling_source_components.v1":
        members, physical = [], set()
        for component in composition["components"]:
            rows = [
                r
                for r in assigned
                if r["symbol"] == component["physical_symbol"] and r["group"] == component["physical_group"]
            ]
            members.extend(composition_members(component, assigned=rows))
            if component["physical_group"] in physical:
                raise ValueError("component dispatch repeats physical work")
            physical.add(component["physical_group"])
        ordinary = composition["ordinary_joined_groups"]
        indices = [r["source_group"] for r in members]
        if (
            len(set(indices)) != len(indices)
            or set(indices).intersection(ordinary.values())
            or set(physical).intersection(ordinary)
            or set(physical) | set(ordinary) != {r["group"] for r in assigned}
            or not set(composition["required_source_groups"]) <= set(indices) | set(ordinary.values())
        ):
            raise ValueError("component dispatch omits or duplicates original source work")
        return members
    if (
        not isinstance(composition, dict)
        or composition.get("schema") != "merlin.original_sibling_source_composition.v1"
        or composition.get("original_source_denominator_unchanged") is not True
        or len(assigned) != 1
        or assigned[0].get("symbol") != composition.get("physical_symbol")
        or assigned[0].get("group") != composition.get("physical_group")
    ):
        raise ValueError("sibling dispatch lacks its independently rederived source composition")
    members = composition.get("members")
    if (
        not isinstance(members, list)
        or len(members) < 2
        or len({row["source_group"] for row in members}) != len(members)
        or len({row["source_region_id"] for row in members}) != len(members)
        or set(composition["required_source_groups"]) - {row["source_group"] for row in members}
    ):
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
            if (
                type(region) is not StringAttr
                or region.data not in expected
                or type(nodes) is not ArrayAttr
                or any(type(node) is not StringAttr for node in nodes)
                or tuple(node.data for node in nodes) != expected[region.data]
            ):
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
        if (
            member is None
            or member["source_region_id"] != obligation["region_id"]
            or obligation["semantic_family"] != "contraction"
        ):
            raise ValueError("sibling raw core cannot cover another original supporting source operation")
        # One member is still a separate original obligation. It may not use
        # work attributed to another member, even if only one root is required.
        completed.extend(mandatory(function, by_region[member["source_region_id"]], [obligation]))
    return completed


def borrowed_host_chain(module, function, composition, *, mandatory):
    """Prove the actual defined-wrapper chain reaches one unavoidable ABI leaf."""
    from merlin.verification.host.llvm_symbols import defined_function

    leaf = "_mlir_ciface_" + composition["borrowed_entry_symbol"]
    members = len(composition["members"])
    # A source-proved INT8 readout may add one immutable folded-bias input per
    # member. The independent source join binds its complete logical arity;
    # historical raw-only joins retain the original fixed ABI.
    arity = composition.get("descriptor_arguments", 2 * members + 1)
    if type(arity) is not int or not 2 * members + 1 <= arity <= 3 * members + 1:
        raise ValueError("borrowed host chain lost its source-proved logical arity")

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
                if (
                    len(call.operands) != arity
                    or call.results
                    or any(str(v.type) != "!llvm.ptr" for v in call.operands)
                ):
                    raise ValueError("borrowed host call lost the complete pointer-only multi-output ABI")
                branches = [[name, leaf]]
            elif callee in definitions and any(
                block.ops for region in definitions[callee].regions for block in region.blocks
            ):
                branches = visit(defined_function(module, callee), (*stack, name))
                branches = [[name, *path] for path in branches]
            else:
                continue
            if branches and not mandatory(current, {call}):
                raise ValueError("borrowed host call can be bypassed on a returning CFG path")
            if branches:
                from merlin.analysis.programs.host_cfg_index import prepare_host_cfg

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
    return {
        "chain": path,
        "leaf": leaf,
        "descriptor_arguments": arity,
        "mandatory_call_roster": {composition["physical_symbol"]: 1},
    }


def verify_descriptor_bridge(entry, composition, *, built, target, collection=None):
    """Reproduce the stock C bridge and join its actual compile and final link."""
    import json

    from merlin.common.process import invocation_records as invocation_record
    from merlin.compiler.lower.device.device_shim import emit_logical_translation_unit, logical_buffers
    from merlin.compiler.lower.device.sibling_source_route import borrowed_bridge_source
    from merlin.compiler.lower.host.borrowed_effects import BorrowedBufferEffects
    from merlin.compiler.lower.model import toolchain
    from merlin.compiler.lower.model.compilation_recipe import verify_completed_recipe
    from merlin.development.targetgen.execution.host.model_execution import _pinned
    from merlin.development.targetgen.execution.native.native_model_execution import _digest

    custody = (entry.get("custody") or {}).get("sibling_source")
    if (
        not isinstance(custody, dict)
        or custody.get("source_ir_sha256") != composition["source_ir_sha256"]
        or custody.get("interface_sha256") != composition["interface_sha256"]
        or custody.get("original_epilogues_and_all_live_outputs_preserved") is not True
    ):
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
    sidecar = json.loads(
        _pinned(
            {
                "sidecar": {
                    **sidecar_pin,
                    "size_bytes": Path(sidecar_pin["path"]).stat().st_size,
                }
            },
            "sidecar",
        ).read_text()
    )
    symbol = composition["physical_symbol"]
    interface = sidecar["expected_interfaces"][symbol]
    from hashlib import sha256

    if sha256(interface["mlir"].encode()).hexdigest() != composition["interface_sha256"]:
        raise ValueError("sibling descriptor bridge interface changed after source join")
    buffers, reason = logical_buffers(interface["mlir"])
    if buffers is None:
        raise ValueError("sibling descriptor bridge lost logical buffer ABI: " + reason)
    call = sidecar["call_buffers"][symbol]
    if collection is not None and collection.get("schema") in {
        "merlin.original_sibling_source_components.v1",
        "merlin.original_head_source_composition.v1",
    }:
        artifacts = built["device_artifacts"]
        if artifacts.get("target") != target or not isinstance(artifacts.get("kernels"), dict):
            raise ValueError("component actual kernel roster changed")
        kernels = artifacts["kernels"]
        symbols = {}
        for row in sidecar["routed"]:
            name = row["symbol"]
            iface = Path(built["elf"]).parent / "device" / (name + ".iface.mlir")
            actual_buffers, reason = logical_buffers(iface.read_text())
            if actual_buffers is None:
                raise ValueError("component shim lost actual ordinary interface: " + reason)
            if (
                name in sidecar["expected_interfaces"]
                and iface.read_text() != sidecar["expected_interfaces"][name]["mlir"]
            ):
                raise ValueError("component actual interface differs from source-bound interface")
            symbols[name] = actual_buffers, sidecar["call_buffers"][name]
        unit = emit_logical_translation_unit(
            target, symbols, kernel_symbol_for=lambda name: kernels[name]["kernel_symbol"]
        )
        expected = unit.text
        for c in collection["components"]:
            if c.get("schema") != "merlin.original_sibling_source_composition.v1":
                continue
            member_call = sidecar["call_buffers"][c["physical_symbol"]]
            outputs = tuple(i for i, r in enumerate(member_call) if r["role"].startswith("out_"))
            expected += "\n" + borrowed_bridge_source(
                c["physical_symbol"], c["borrowed_entry_symbol"], member_call, outputs
            )
        correct_symbols = set(symbols) == set(unit.symbols)
    else:
        unit = emit_logical_translation_unit(
            target, {symbol: (buffers, call)}, kernel_symbol_for=lambda _: entry["kernel_symbol"]
        )
        output_positions = tuple(i for i, row in enumerate(call) if row["role"].startswith("out_"))
        expected = (
            unit.text
            + "\n"
            + borrowed_bridge_source(symbol, composition["borrowed_entry_symbol"], call, output_positions)
        )
        correct_symbols = unit.symbols == (symbol,)
    if not correct_symbols or c_source.read_text() != expected:
        raise ValueError("actual sibling descriptor bridge differs from stock complete ABI transport")
    observed = invocation_record.verify(_pinned(bridge, "compilation"))

    def pin(path):
        return {key: _digest(path)[key] for key in ("path", "sha256")}

    if (
        observed.get("stage") != "device_shim_compilation"
        or observed.get("inputs") != [pin(c_source)]
        or observed.get("outputs") != [pin(obj)]
        or Path(observed["executable"]["path"]) != toolchain.clang().resolve()
        or observed["argv"][-4:] != ["-c", str(c_source), "-o", str(obj)]
    ):
        raise ValueError("sibling descriptor bridge did not use the original observed C compiler boundary")
    recipe = built["compilation_recipe"]
    compilation = verify_completed_recipe(
        Path(recipe["path"]), executable=Path(built["elf"]), expected_recipe_sha256=recipe["sha256"]
    )
    linked = {(row["path"], row["sha256"]) for row in compilation["commands"][-1]["inputs"]}
    if (str(obj), _digest(obj)["sha256"]) not in linked:
        raise ValueError("sibling descriptor bridge object is not an actual final ELF link input")
    return {
        "source": _digest(c_source),
        "object": _digest(obj),
        "compilation": _digest(_pinned(bridge, "compilation")),
        "scope": "stock pointer/identity-descriptor transport, observed compiler, actual final link input",
    }


def component_for_symbol(composition, symbol, *, group=None):
    if composition is None:
        return None
    if composition.get("schema") in {
        "merlin.original_sibling_source_components.v1",
        "merlin.original_head_source_composition.v1",
    }:
        components = composition["components"]
    else:
        components = [composition]
    matches = [
        c for c in components if c["physical_symbol"] == symbol and (group is None or c["physical_group"] == group)
    ]
    if len(matches) > 1:
        raise ValueError("shared physical symbol needs an unambiguous original call group")
    return matches[0] if matches else None
