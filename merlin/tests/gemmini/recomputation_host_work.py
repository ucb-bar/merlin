"""A scoped actual submitted-LLVM producer-work witness for the public case.

Opaque target calls retain their original semantics. This counts neither their
work nor post-optimization machine instructions, cache traffic or bus transfers.
"""

import json
from collections import Counter
from pathlib import Path

from xdsl.dialects import llvm
from xdsl.dialects.builtin import IntegerAttr, IntegerType, i64
from xdsl.ir import Block

from merlin.frontends import linalg_mlir
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.perf import address_loop_control, constant_control_cfg, host_cfg_activity, host_cfg_index
from merlin.targetgen.oot_starterkit import llvm_context
from merlin.verify import host_call_alignment
from merlin.xdsl_dialects.lowering import integer_constant_eval


def implementation_paths():
    return (
        Path(__file__).resolve(),
        *tuple(
            Path(module.__file__).resolve()
            for module in (
                host_cfg_activity,
                host_cfg_index,
                constant_control_cfg,
                address_loop_control,
                host_call_alignment,
                llvm_context,
                integer_constant_eval,
                host_call_alignment.toolchain,
                linalg_mlir,
            )
        ),
    )


def observer_paths():
    translator = host_call_alignment.toolchain.mlir_translate().resolve(strict=True)
    return (
        translator,
        translator.with_name("llvm-extract").resolve(strict=True),
        translator.with_name("mlir-opt").resolve(strict=True),
    )


def functions():
    return (
        functions,
        implementation_paths,
        observer_paths,
        argument_slots,
        _plain,
        _ordinary_memory,
        _closed_generations,
        _readonly_region,
        verify_work,
        inspect_work,
        parse_mlir_text,
        host_cfg_activity.analyze_host_cfg_activity,
        host_cfg_activity._constant_loop,
        host_cfg_activity._posttested_single_block_loop,
        host_cfg_activity._integer_attr,
        host_cfg_activity._payload_bytes,
        host_cfg_activity._real_name,
        host_cfg_activity.constant_integer,
        host_cfg_index.prepare_host_cfg,
        host_cfg_index.require_prepared_host_cfg,
        host_cfg_index._IntervalDominanceInfo.__init__,
        host_cfg_index._IntervalDominanceInfo.dominates,
        host_call_alignment.import_host_entry,
        host_call_alignment.revalidate_host_entry,
        integer_constant_eval.constant_integer,
        host_call_alignment.toolchain.mlir_translate,
    )


def argument_slots(prepared, arguments, manifest, input_order):
    """Join original ordered inputs to actual current runtime descriptor slots."""
    entries = [op for op in prepared.body.block.ops if op.name == "func.func" and op.sym_name.data == "forward"]
    assert len(entries) == 1
    entry = entries[0]
    inputs, outputs = tuple(entry.function_type.inputs), tuple(entry.function_type.outputs)
    assert len(inputs) == 6 and len(outputs) == 9
    assert set(manifest) == {str(i) for i in range(6)} and set(input_order.values()) == set(range(6))
    assert len(arguments) == 6 and {row["original_index"] for row in arguments} == set(range(6))
    assert {row["runtime_index"] for row in arguments} == set(range(6))
    fields, starts = [], []
    for ty in (*inputs, *outputs):
        assert len(ty.get_shape()) == 2 and all(type(n) is int and n > 0 for n in ty.get_shape())
        starts.append(len(fields))
        fields.extend((llvm.LLVMPointerType(), llvm.LLVMPointerType(), *([i64] * 5)))
    roles = {}
    for row in arguments:
        original, current = row["original_index"], row["runtime_index"]
        declared = manifest[str(original)]
        name = declared["name"]
        assert declared["kind"] == "input" and row["kind"] == "MERLIN_INPUT" and input_order[name] == original
        ty = inputs[current]
        assert tuple(row["shape"]) == ty.get_shape() and row["dtype"] == str(ty.element_type)
        roles[name] = dict(runtime_index=current, aligned=starts[current] + 1, bits=ty.element_type.width.data)
    assert set(roles) == {"x", "bias0", "bias1", "w0", "w1", "state"}
    for name, bits in (("x", 8), ("bias0", 8), ("bias1", 8), ("state", 32)):
        assert roles[name]["bits"] == bits and inputs[roles[name]["runtime_index"]].get_shape() == (2, 5)
    return roles, tuple(fields)


def _plain(op):
    assert not op.attributes
    assert set(op.properties) <= {"overflowFlags"}
    flags = op.properties.get("overflowFlags")
    assert (
        flags is None
        or type(flags) is IntegerAttr
        and flags.value.data == 0
        or type(flags) is llvm.OverflowAttr
        and not flags.data
    )


def _ordinary_memory(op):
    assert not op.attributes and set(op.properties) <= {"alignment", "ordering"}
    assert op.properties["ordering"].value.data == 0, "atomic/volatile memory is outside this witness"


def verify_work(function, roles, fields, callees, *, candidate):
    """Reconstruct exact state32->i8/add-x DAGs and proven per-point counts."""
    function.verify()
    cfg = host_cfg_index.prepare_host_cfg(function)
    activity = host_cfg_activity.analyze_host_cfg_activity(function, prepared_cfg=cfg)
    # Descriptor aggregate stores can leave total memory payload UNKNOWN without
    # affecting the independently proven loop execution counts used here.
    assert not activity["problems"] and activity["control_count_basis"] == "constant_induction_v1"
    assert all(type(row["execution_count"]) is int and row["execution_count"] >= 0 for row in activity["blocks"])
    entry = cfg.blocks[0]
    assert tuple(value.type for value in entry.args) == fields
    counts = {cfg.blocks[row["block"]]: row["execution_count"] for row in activity["blocks"]}
    block_indices = {block: i for i, block in enumerate(cfg.blocks)}
    axes = {}
    for row in activity["loops"]:
        if (row["initial"], row["step"], row["predicate"]) == (0, 1, "slt") and row["bound"] in (2, 5):
            value = cfg.blocks[row["header_block"]].args[row["induction_argument_index"]]
            axes[value] = (0 if row["bound"] == 2 else 1, {row["header_block"], *row["body_blocks"]})

    def integer(value, block, depth=0):
        assert depth < 32
        constant = integer_constant_eval.constant_integer(value)
        if constant is not None:
            return constant
        if value in axes:
            axis, nodes = axes[value]
            assert block_indices[block] in nodes
            return ("axis", axis)
        op = value.owner
        assert not isinstance(op, Block) and op.name in ("llvm.add", "llvm.mul")
        left, right = (integer(v, block, depth + 1) for v in op.operands)
        if isinstance(left, int) and isinstance(right, int):
            return left + right if op.name == "llvm.add" else left * right
        if op.name == "llvm.add" and right == 0 or op.name == "llvm.mul" and right == 1:
            return left
        return (op.name, left, right)

    coordinate = ("llvm.add", ("llvm.mul", ("axis", 0), 5), ("axis", 1))

    allowed = {
        "llvm.func",
        "llvm.mlir.constant",
        "llvm.mlir.undef",
        "llvm.mlir.poison",
        "llvm.br",
        "llvm.cond_br",
        "llvm.return",
        "llvm.icmp",
        "llvm.call",
        "llvm.insertvalue",
        "llvm.extractvalue",
        "llvm.alloca",
        "llvm.load",
        "llvm.store",
        "llvm.getelementptr",
        "llvm.ptrtoint",
        "llvm.inttoptr",
        "llvm.bitcast",
        "llvm.add",
        "llvm.mul",
        "llvm.sub",
        "llvm.urem",
        "llvm.and",
        "llvm.or",
        "llvm.trunc",
        "llvm.sext",
        "llvm.zext",
        "llvm.intr.memcpy",
        "llvm.intr.memset",
        "llvm.intr.stacksave",
        "llvm.intr.stackrestore",
    }
    assert len(callees) == len(set(callees)) == 2
    calls, call_counts = {}, Counter()
    for op in function.walk():
        assert host_cfg_activity._real_name(op) in allowed, "unknown host instruction/effect"
        if op.name in ("llvm.load", "llvm.store"):
            _ordinary_memory(op)
        if op.name != "llvm.call":
            continue
        reference = op.properties["callee"]
        assert not reference.nested_references.data
        symbol = reference.root_reference.data
        if symbol in callees:
            assert symbol not in calls and counts[op.parent] == 1
            calls[symbol] = op
            call_counts[symbol] += counts[op.parent]
        else:
            # Standard runtime ABI helpers are not assumed pure. They must be
            # outside all producer/consumer work blocks below; their payload
            # and observability remain outside this work-only witness.
            assert symbol in {"malloc", "free", "memrefCopy", "llvm.stacksave.p0", "llvm.stackrestore.p0"}, (
                "unknown host call"
            )
    assert call_counts == Counter(callees), "actual contraction call roster differs"
    first, second = (calls[name] for name in callees)
    if first.parent is second.parent:
        sequence = tuple(first.parent.ops)
        assert sequence.index(first) < sequence.index(second), "actual opaque contraction order changed"
    else:
        assert cfg.dominance.dominates(first.parent, second.parent), "actual opaque contraction order changed"
    for call in calls.values():
        assert len(call.operands) == 21
        assert tuple(integer_constant_eval.constant_integer(v) for v in call.operands[2:7]) == (0, 2, 5, 5, 1)

    def address(value, block, bits):
        gep = value.owner
        assert not isinstance(gep, Block) and gep.name == "llvm.getelementptr" and len(gep.operands) == 2
        assert gep.properties["elem_type"] == IntegerType(bits)
        assert tuple(gep.properties["rawConstantIndices"].get_values()) == (llvm.GEP_USE_SSA_VAL,)
        assert integer(gep.operands[1], block) == coordinate
        return gep.operands[0]

    input_reads = set()

    def load(value, role, block):
        op = value.owner
        assert not isinstance(op, Block) and op.name == "llvm.load" and op.parent is block
        assert value.type == IntegerType(roles[role]["bits"]) and not op.attributes
        assert set(op.properties) <= {"alignment", "ordering"}
        assert op.properties["ordering"].value.data == 0
        assert address(op.operands[0], block, roles[role]["bits"]) is entry.args[roles[role]["aligned"]]
        input_reads.add(op)

    def completed_before(block, later):
        outer = [row for row in activity["loops"] if row["bound"] == 2 and block_indices[block] in row["body_blocks"]]
        assert len(outer) == 1
        header = cfg.blocks[outer[0]["header_block"]]
        nodes = {outer[0]["header_block"], *outer[0]["body_blocks"]}
        exits = [b for b in header.last_op.successors if block_indices[b] not in nodes]
        assert len(exits) == 1 and cfg.dominance.dominates(exits[0], later)

    writes, reads, consumers, epilogue_stores = {}, {}, {}, {}

    def watch(root, *, writer=None, reader=None, consumer=None):
        if writer is not None:
            writes.setdefault(root, set()).add(writer)
        if reader is not None:
            reads.setdefault(root, set()).add(reader)
        if consumer is not None:
            consumers.setdefault(root, []).append(consumer)

    def store_to_call(user, role, block):
        assert user.parent is block and len(tuple(user.results[0].uses)) == 1
        store = next(iter(user.results[0].uses)).operation
        assert store.name == "llvm.store" and store.parent is block and store.operands[0] is user.results[0]
        call = calls[callees[0 if role == "bias0" else 1]]
        assert address(store.operands[1], block, 8) is call.operands[1], (
            "epilogue store does not feed actual contraction LHS"
        )
        completed_before(block, call.parent)
        watch(call.operands[1], writer=store, consumer=call)
        epilogue_stores[role] = store
        return store

    def no_opaque_work(block):
        assert all(
            op.name != "llvm.call" and not host_cfg_activity._real_name(op).startswith("llvm.intr.") for op in block.ops
        ), "opaque effect inside producer/epilogue work"

    totals, sites, producer_stores = Counter(), [], []
    for ordinal, trunc in enumerate(function.walk()):
        if (
            trunc.name != "llvm.trunc"
            or tuple(v.type for v in trunc.operands) != (IntegerType(32),)
            or tuple(v.type for v in trunc.results) != (IntegerType(8),)
        ):
            continue
        _plain(trunc)
        block = trunc.parent
        no_opaque_work(block)
        load(trunc.operands[0], "state", block)
        uses = tuple(trunc.results[0].uses)
        assert len(uses) == 1
        producer = uses[0].operation
        assert producer.name == "llvm.add" and producer.parent is block
        _plain(producer)
        assert producer.operands[1] is trunc.results[0]
        load(producer.operands[0], "x", block)
        assert counts[block] == 10
        uses = tuple(producer.results[0].uses)
        assert len(uses) == 1
        user = uses[0].operation
        if candidate:
            assert user.name in ("llvm.add", "llvm.mul") and user.parent is block
            _plain(user)
            assert user.operands[0] is producer.results[0]
            role = "bias0" if user.name == "llvm.add" else "bias1"
            load(user.operands[1], role, block)
            store_to_call(user, role, block)
            totals[role] += counts[block]
        else:
            assert user.name == "llvm.store" and user.operands[0] is producer.results[0]
            assert user.parent is block
            producer_stores.append((address(user.operands[1], block, 8), block))
            watch(producer_stores[-1][0], writer=user)
            totals["producer_store_payload_bytes"] += counts[block]
        totals["producer_evaluations"] += counts[block]
        sites.append(dict(operation_index=ordinal, block=block_indices[block], evaluations=counts[block]))
    assert totals["producer_evaluations"] == (20 if candidate else 10), "actual host producer work differs"
    assert not candidate or totals["bias0"] == totals["bias1"] == 10
    if not candidate:
        assert len(producer_stores) == 1
        base, producer_block = producer_stores[0]
        assert base is not calls[callees[0]].operands[1], "first epilogue overwrites P before the second reads it"
        for role, opcode in (("bias0", "llvm.add"), ("bias1", "llvm.mul")):
            epilogues = []
            for op in function.walk():
                if op.name != opcode or tuple(v.type for v in op.results) != (IntegerType(8),):
                    continue
                left = op.operands[0].owner
                if isinstance(left, Block) or left.name != "llvm.load" or left.parent is not op.parent:
                    continue
                pointer = left.operands[0].owner
                if (
                    isinstance(pointer, Block)
                    or pointer.name != "llvm.getelementptr"
                    or pointer.operands[0] is not base
                ):
                    continue
                no_opaque_work(op.parent)
                _plain(op)
                assert address(left.operands[0], op.parent, 8) is base
                load(op.operands[1], role, op.parent)
                assert counts[op.parent] == 10
                completed_before(producer_block, op.parent)
                watch(base, reader=left, consumer=op.parent)
                store_to_call(op, role, op.parent)
                epilogues.append(op)
            assert len(epilogues) == 1, "stored producer is not consumed by both actual epilogues"
            totals[role] = 10
    assert calls[callees[0]].operands[1] is not calls[callees[1]].operands[1], "epilogues share a destination"
    completed_before(epilogue_stores["bias0"].parent, epilogue_stores["bias1"].parent)
    completed_before(epilogue_stores["bias1"].parent, calls[callees[0]].parent)
    _closed_generations(function, writes, reads, consumers, calls, cfg, completed_before)
    last = epilogue_stores["bias1"].parent
    outer = [row for row in activity["loops"] if row["bound"] == 2 and block_indices[last] in row["body_blocks"]]
    header = cfg.blocks[outer[0]["header_block"]]
    nodes = {outer[0]["header_block"], *outer[0]["body_blocks"]}
    end = next(block for block in header.last_op.successors if block_indices[block] not in nodes)
    _readonly_region(entry, roles, input_reads, cfg, end)
    return dict(
        counts=dict(totals),
        sites=sites,
        argument_roles=roles,
        activity=activity,
        scope="actual submitted LLVM exact same-coordinate producer DAGs and constant-loop counts; "
        "opaque calls excluded, not final machine work, total memory traffic or hardware performance",
    )


def _closed_generations(function, writes, reads, consumers, calls, cfg, completed_before):
    """Close exact watched SSA use cones; no address evaluator or alias guesses."""
    owners, paths = {}, {}

    def ancestors(value, depth=0):
        assert depth < 32
        if integer_constant_eval.constant_integer(value) is not None:
            return set(), set()
        op = value.owner
        if isinstance(op, Block):
            assert op is cfg.blocks[0] and value.type == llvm.LLVMPointerType(), "unknown pointer owner"
            return {value}, {value}
        if op.name == "llvm.alloca" or op.name == "llvm.call" and op.callee.root_reference.data == "malloc":
            return {value}, {value}
        assert op.name in {
            "llvm.getelementptr",
            "llvm.bitcast",
            "llvm.ptrtoint",
            "llvm.inttoptr",
            "llvm.add",
            "llvm.sub",
            "llvm.urem",
        }, "unknown watched-pointer provenance"
        leaves, chain = set(), {value}
        for operand in op.operands:
            roots, prefix = ancestors(operand, depth + 1)
            leaves.update(roots)
            chain.update(prefix)
        return leaves, chain

    for root in writes:
        leaves, chain = ancestors(root)
        assert len(leaves) == 1, "ambiguous watched allocation owner"
        owners[root], paths[root] = next(iter(leaves)), chain
    assert len(set(owners.values())) == len(owners), "ambiguous overlapping watched views"
    for call in calls.values():
        raw, _ = ancestors(call.operands[0])
        assert raw == {owners[call.operands[1]]}, "raw/aligned LHS ownership is unknown"
    pending = [(value, root) for root, chain in paths.items() for value in chain]
    seen = set()
    while pending:
        value, root = pending.pop()
        if (value, root) in seen:
            continue
        seen.add((value, root))
        for use in tuple(value.uses):
            op = use.operation
            if any(result in paths[root] for result in op.results):
                continue  # The exact retained allocation/alignment chain.
            if op.name in ("llvm.getelementptr", "llvm.bitcast") and use.index == 0:
                pending.extend((result, root) for result in op.results)
            elif op.name == "llvm.store":
                assert use.index == 1 and any(op in rows for rows in writes.values()), (
                    "intervening watched-buffer write/escape"
                )
            elif op.name == "llvm.load":
                assert any(op in rows for rows in reads.values()), "extra watched-buffer read/observer"
            elif op.name == "llvm.call" and op in calls.values():
                assert use.index in (0, 1) and any(op in rows for rows in consumers.values()), (
                    "unexpected watched-pointer call escape"
                )
            elif op.name == "llvm.call" and op.callee.root_reference.data == "free":
                # Raw allocation bases and aligned/GEP aliases share this check.
                related = [r for r in writes if owners[r] is owners[root]]
                for other in related:
                    for consumer in consumers.get(other, ()):
                        if isinstance(consumer, Block):
                            completed_before(consumer, op.parent)
                        elif consumer.parent is op.parent:
                            sequence = tuple(op.parent.ops)
                            assert sequence.index(consumer) < sequence.index(op), (
                                "watched buffer freed before consumption"
                            )
                        else:
                            assert cfg.dominance.dominates(consumer.parent, op.parent), (
                                "watched buffer freed before consumption"
                            )
            else:
                raise AssertionError("unknown watched-pointer alias/escape/effect")


def _readonly_region(entry, roles, permitted_loads, cfg, end):
    """No incoming argument version change/escape before all P/E work ends.

    Direct stack descriptor assembly is followed structurally; it cannot issue
    an early opaque call. No native call effects or general address semantics
    are inferred from this transport.
    """
    pending = [entry.args[row["aligned"] - delta] for row in roles.values() for delta in (0, 1)]
    seen, boxes = set(), set()
    while pending:
        value = pending.pop()
        if value in seen:
            continue
        seen.add(value)
        for use in tuple(value.uses):
            op = use.operation
            if cfg.dominance.dominates(end, op.parent):
                continue
            if op.name in (
                "llvm.getelementptr",
                "llvm.bitcast",
                "llvm.ptrtoint",
                "llvm.inttoptr",
                "llvm.insertvalue",
                "llvm.extractvalue",
                "llvm.add",
                "llvm.sub",
                "llvm.urem",
            ):
                pending.extend(op.results)
            elif op.name == "llvm.load":
                if value in boxes:
                    pending.extend(op.results)
                else:
                    assert op in permitted_loads, "extra early incoming-buffer observation"
            elif op.name == "llvm.store":
                if use.index == 0 and op.operands[1].owner.name == "llvm.alloca":
                    boxes.add(op.operands[1])
                    pending.append(op.operands[1])
                else:
                    assert use.index == 1 and value in boxes, "incoming buffer changed before recomputation completed"
            else:
                raise AssertionError("early incoming-pointer escape/lifetime/effect")


def inspect_work(host_artifacts, prepared_path, capture, readonly_manifest, callees, *, candidate, diagnostics):
    prepared = parse_mlir_text(Path(prepared_path).read_text())
    prepared.verify()
    forward = next(op for op in prepared.body.block.ops if op.name == "func.func" and op.sym_name.data == "forward")
    source_calls = [op for op in forward.walk() if op.name == "func.call"]
    assert tuple(op.callee.root_reference.data for op in source_calls) == tuple(callees)
    assert all(
        tuple(op.arguments[0].type.get_shape()) == (2, 5) and str(op.arguments[0].type.element_type) == "i8"
        for op in source_calls
    )
    roles, fields = argument_slots(
        prepared,
        readonly_manifest["arguments"],
        json.loads((capture / "weights.safetensors.manifest.json").read_bytes()),
        json.loads((capture / "input_order.json").read_bytes()),
    )
    module, observation = host_call_alignment.import_host_entry(
        host_artifacts, diagnostics=diagnostics, max_ir_bytes=1 << 20
    )
    functions = [op for op in module.body.block.ops if op.name == "llvm.func" and op.sym_name.data == "forward"]
    assert len(functions) == 1
    witness = verify_work(functions[0], roles, fields, callees, candidate=candidate)
    host_call_alignment.revalidate_host_entry(observation)
    return dict(**witness, import_observation=observation)
