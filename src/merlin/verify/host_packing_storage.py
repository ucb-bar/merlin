"""Conditional packing ownership and footprint in the actual compiled host LLVM.

The stock importer and existing CFG/SCC analyses own syntax and control flow.
This closed checker follows allocation-relative address expressions, original
ranked call descriptors and complete borrowed-call premises. It neither moves
storage nor qualifies a runtime, provider implementation or native compiler.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from math import prod
from pathlib import Path

from xdsl.dialects import builtin, llvm
from xdsl.ir import Operation
from xdsl.parser import Parser

from merlin.llvmlower.borrowed_buffer_calls import validate_calls
from merlin.perf.host_cfg_index import prepare_host_cfg
from merlin.targetgen.oot_starterkit.llvm_context import make_llvm_context
from merlin.xdsl_dialects.lowering.integer_constant_eval import constant_integer


@dataclass(frozen=True)
class PackingAllocationEffects:
    """Explicit original runtime premises, never inferred from symbol spellings.

    Fresh private allocation means a disjoint contiguous object of the complete
    requested byte extent. Success includes non-null, representable storage.
    The separately selected pointer width must match every address conversion.
    Releases are possible only through the selected deallocator. These are
    original qualification premises; this record proves none of them.
    """

    allocator_symbol: str
    deallocator_symbol: str
    qualification_identity: str
    pointer_bits: int
    fresh_private_allocations: bool = field(kw_only=True)
    successful_allocations: bool = field(kw_only=True)
    releases_only_via_deallocator: bool = field(kw_only=True)

    def __post_init__(self):
        if (
            any(
                type(s) is not str or not s or len(s) > 4096
                for s in (self.allocator_symbol, self.deallocator_symbol, self.qualification_identity)
            )
            or self.allocator_symbol == self.deallocator_symbol
            or type(self.pointer_bits) is not int
            or self.pointer_bits not in (32, 64)
            or any(
                v is not True
                for v in (
                    self.fresh_private_allocations,
                    self.successful_allocations,
                    self.releases_only_via_deallocator,
                )
            )
        ):
            raise ValueError("packing storage needs original complete allocator effects")


@dataclass(frozen=True)
class PackingStorageUse:
    wrapper_symbol: str
    argument: int

    def __post_init__(self):
        if (
            type(self.wrapper_symbol) is not str
            or not self.wrapper_symbol
            or type(self.argument) is not int
            or self.argument < 0
        ):
            raise ValueError("packing storage requires an original positional call argument")


def implementation_paths():
    from merlin.llvmlower import arena_bind
    from merlin.perf import host_cfg_index
    from merlin.targetgen.oot_starterkit import llvm_context
    from merlin.xdsl_dialects.lowering import integer_constant_eval

    from . import host_call_alignment

    return tuple(
        Path(p).resolve()
        for p in (
            __file__,
            host_call_alignment.__file__,
            host_cfg_index.__file__,
            arena_bind.__file__,
            integer_constant_eval.__file__,
            llvm_context.__file__,
        )
    )


def _require(claim, reason):
    if not claim:
        raise ValueError("final packing storage: " + reason)


def _callee(op):
    reference = op.properties.get("callee") if op.name == "llvm.call" else None
    if reference is None or reference.nested_references.data:
        return None
    return reference.root_reference.data


def _ccc(op):
    convention = op.properties.get("CConv")
    return getattr(getattr(convention, "convention", None), "data", None) == "ccc"


def _types(call):
    result = []
    for text in call.argument_types:
        typ = Parser(make_llvm_context(), text).parse_type()
        _require(
            type(typ) is builtin.MemRefType
            and type(typ.layout) is builtin.NoneAttr
            and type(typ.memory_space) is builtin.NoneAttr
            and type(typ.element_type) is builtin.IntegerType
            and typ.element_type.width.data in (8, 16, 32)
            and all(d > 0 for d in typ.get_shape()),
            "original ABI requires static dense numeric memory arguments",
        )
        result.append((tuple(typ.get_shape()), typ.element_type.width.data // 8))
    return tuple(result)


def _forwarder(functions, call, shapes):
    wrapper = functions.get(call.wrapper_symbol)
    _require(wrapper is not None and len(wrapper.body.blocks) == 1, "original fresh writer definition is missing")
    arguments, operations = tuple(wrapper.body.block.args), tuple(wrapper.body.block.ops)
    counts = tuple(3 + 2 * len(shape) for shape, _ in shapes)
    total, output_count = sum(counts), counts[call.result_argument]
    _require(len(arguments) == total + output_count, "fresh writer flattened ABI differs from original")
    index_type = arguments[2].type
    _require(
        type(index_type) is builtin.IntegerType and index_type.width.data in (32, 64), "unknown descriptor index ABI"
    )
    expected_types = tuple(
        t
        for count in (*counts, output_count)
        for t in (llvm.LLVMPointerType(), llvm.LLVMPointerType(), *([index_type] * (count - 2)))
    )
    _require(tuple(a.type for a in arguments) == expected_types, "fresh writer descriptor types differ")
    _require(
        type(wrapper.function_type.output) is llvm.LLVMVoidType and _ccc(wrapper), "fresh writer must use void C ABI"
    )
    _require(
        len(operations) == 2
        and _callee(operations[0]) == call.borrowed_symbol
        and not operations[0].results
        and _ccc(operations[0])
        and operations[1].name == "llvm.return"
        and not operations[1].operands,
        "fresh writer does not forward exactly to its original borrowed implementation",
    )
    starts, cursor, expected = [], 0, []
    for index, count in enumerate(counts):
        starts.append(cursor)
        start = total if index == call.result_argument else cursor
        expected.extend(arguments[start : start + count])
        cursor += count
    _require(tuple(operations[0].operands) == tuple(expected), "fresh writer changes original argument ownership")
    borrowed = functions.get(call.borrowed_symbol)
    _require(
        borrowed is not None
        and tuple(borrowed.function_type.inputs) == expected_types[:total]
        and type(borrowed.function_type.output) is llvm.LLVMVoidType
        and _ccc(borrowed),
        "borrowed declaration differs from the original positional ABI",
    )
    return counts, tuple(starts), expected_types


def _allocation(value, effects):
    owner = value.owner
    _require(
        isinstance(owner, Operation)
        and _callee(owner) == effects.allocator_symbol
        and len(owner.results) == 1
        and owner.results[0] is value
        and len(owner.operands) == 1
        and value.type == llvm.LLVMPointerType()
        and _ccc(owner),
        "packing pointer is not an original fresh allocation root",
    )
    request = constant_integer(owner.operands[0])
    _require(type(request) is int and 0 < request < 1 << 63, "allocator request is not a finite positive constant")
    return owner, request


def _alignment(raw, aligned, pointer_bits):
    """Match complete round-up from this exact raw pointer, not just low bits."""
    if aligned is raw:
        return 1, frozenset()
    convert = aligned.owner
    _require(
        isinstance(convert, Operation) and convert.name == "llvm.inttoptr" and len(convert.operands) == 1,
        "unknown allocation-relative aligned pointer",
    )
    subtract = convert.operands[0].owner
    _require(
        isinstance(subtract, Operation) and subtract.name == "llvm.sub" and len(subtract.operands) == 2,
        "aligned pointer does not subtract its own remainder",
    )
    x, remainder_value = subtract.operands
    remainder = remainder_value.owner
    _require(
        isinstance(remainder, Operation)
        and remainder.name == "llvm.urem"
        and len(remainder.operands) == 2
        and remainder.operands[0] is x,
        "alignment remainder changes its numerator",
    )
    alignment = constant_integer(remainder.operands[1])
    _require(
        type(alignment) is int and 0 < alignment < 1 << 63 and not alignment & (alignment - 1),
        "alignment requires an exact positive power of two",
    )
    addition = x.owner
    _require(
        isinstance(addition, Operation) and addition.name == "llvm.add" and len(addition.operands) == 2,
        "alignment does not round up the allocation base",
    )
    cast = addition.operands[0].owner
    _require(
        isinstance(cast, Operation)
        and cast.name == "llvm.ptrtoint"
        and tuple(cast.operands) == (raw,)
        and constant_integer(addition.operands[1]) == alignment - 1,
        "alignment changed the original allocation base",
    )
    index_type = cast.results[0].type
    chain = frozenset((cast, addition, remainder, subtract, convert))
    for operation in (addition, subtract):
        flags = operation.properties.get("overflowFlags")
        _require(
            flags is None
            or (type(flags) is builtin.IntegerAttr and flags.value.data == 0)
            or (isinstance(flags, llvm.OverflowAttr) and not flags.data),
            "unknown poison-producing pointer arithmetic flags",
        )
    _require(
        type(index_type) is builtin.IntegerType
        and index_type.width.data == pointer_bits
        and index_type == raw.owner.operands[0].type
        and all(v.type == index_type for op in (addition, remainder, subtract) for v in (*op.operands, *op.results))
        and convert.operands[0].type == index_type
        and aligned.type == raw.type,
        "alignment changes pointer/index width",
    )
    return alignment, chain


def verify_packing_storage(
    module,
    *,
    borrowed_calls,
    uses,
    allocation_effects,
    max_payload_bytes,
    entry,
    max_operations=100000,
    max_alias_values=4096,
):
    """Prove finite distinct packing ownership under original runtime/ABI effects.

    The budget is logical activation payload, not total host RAM. Gather-value
    correctness and all unrelated memory-access bounds remain separate source,
    compiler and execution obligations.
    """
    from merlin.llvmlower.arena_bind import _cyclic_blocks

    _require(type(allocation_effects) is PackingAllocationEffects, "original allocator effects are unavailable")
    allocation_effects.__post_init__()
    validate_calls(borrowed_calls)
    _require(
        type(uses) is tuple and uses and all(type(use) is PackingStorageUse for use in uses),
        "an immutable original packing-use roster is required",
    )
    for use in uses:
        use.__post_init__()
    _require(len({(u.wrapper_symbol, u.argument) for u in uses}) == len(uses), "duplicate packing-use declarations")
    _require(
        type(entry) is str
        and entry
        and all(type(v) is int and v > 0 for v in (max_payload_bytes, max_operations, max_alias_values)),
        "explicit finite packing budgets and entry are required",
    )
    operations = []
    for operation in module.walk():
        _require(len(operations) < max_operations, "final LLVM operation budget exceeded")
        operations.append(operation)
    operations = tuple(operations)
    operation_ids = {operation: i for i, operation in enumerate(operations)}
    module.verify()
    functions = {}
    for op in module.body.block.ops:
        if op.name == "llvm.func":
            name = op.sym_name.data
            _require(name not in functions, "duplicate current LLVM function symbol")
            functions[name] = op
    function = functions.get(entry)
    _require(function is not None and function.body.blocks, "original model entry is missing")
    allocator = functions.get(allocation_effects.allocator_symbol)
    deallocator = functions.get(allocation_effects.deallocator_symbol)
    _require(
        allocator is not None and not allocator.body.blocks and _ccc(allocator),
        "allocator implementation shadows the original runtime selection",
    )
    _require(
        len(allocator.function_type.inputs) == 1
        and type(tuple(allocator.function_type.inputs)[0]) is builtin.IntegerType
        and tuple(allocator.function_type.inputs)[0].width.data == allocation_effects.pointer_bits
        and allocator.function_type.output == llvm.LLVMPointerType(),
        "unknown original allocator size/pointer ABI",
    )
    _require(
        deallocator is None or (not deallocator.body.blocks and _ccc(deallocator)),
        "deallocator implementation shadows the original runtime selection",
    )
    cfg = prepare_host_cfg(function)
    _require(
        all(
            block.last_op is not None and block.last_op.name in ("llvm.br", "llvm.cond_br", "llvm.return")
            for block in cfg.blocks
        ),
        "unsupported actual LLVM control flow",
    )
    _require(
        all(target in cfg.block_set for block in cfg.blocks for target in cfg.successors[block]),
        "actual CFG leaves the original function",
    )
    reached, pending = set(), [cfg.blocks[0]]
    while pending:
        block = pending.pop()
        if block not in reached:
            reached.add(block)
            pending.extend(cfg.successors[block])
    _require(reached == cfg.block_set and cfg.dominance is not None, "unreachable or unproved actual CFG")
    indexes = {block: i for i, block in enumerate(cfg.blocks)}
    positions = {op: (block, i) for block in cfg.blocks for i, op in enumerate(block.ops)}
    cyclic = _cyclic_blocks([[indexes[b] for b in cfg.successors[block]] for block in cfg.blocks])
    records = {call.wrapper_symbol: call for call in borrowed_calls}
    selected = {(use.wrapper_symbol, use.argument) for use in uses}
    _require(
        all(
            symbol in records
            and argument < len(records[symbol].argument_types)
            and argument != records[symbol].result_argument
            for symbol, argument in selected
        ),
        "packing use is not an original borrowed input",
    )
    layouts = {symbol: _types(records[symbol]) for symbol, _ in selected}
    abis = {symbol: _forwarder(functions, records[symbol], layouts[symbol]) for symbol in layouts}
    calls = [op for block in cfg.blocks for op in block.ops if _callee(op) in layouts]
    _require(set(_callee(op) for op in calls) == set(layouts), "actual call roster misses original packing uses")
    buffers, observed, allowed_calls = {}, [], {}
    for call in calls:
        symbol = _callee(call)
        counts, starts, expected_types = abis[symbol]
        _require(
            not call.results and _ccc(call) and tuple(v.type for v in call.operands) == expected_types,
            "actual caller ABI differs from original ranked descriptors",
        )
        for argument, (shape, element_bytes) in enumerate(layouts[symbol]):
            if (symbol, argument) not in selected:
                continue
            start, rank = starts[argument], len(shape)
            raw, aligned, offset, *dimensions = call.operands[start : start + counts[argument]]
            strides, running = [], 1
            for dimension in reversed(shape):
                strides.insert(0, running)
                running *= dimension
            _require(
                constant_integer(offset) == 0
                and tuple(constant_integer(v) for v in dimensions[:rank]) == shape
                and tuple(constant_integer(v) for v in dimensions[rank:]) == tuple(strides),
                "actual packing descriptor changes source extent, offset or strides",
            )
            allocation, request = _allocation(raw, allocation_effects)
            _require(
                offset.type == allocation.operands[0].type,
                "packing descriptor index width differs from original allocator pointer ABI",
            )
            alignment, chain = _alignment(raw, aligned, allocation_effects.pointer_bits)
            payload = prod(shape) * element_bytes
            _require(
                alignment >= element_bytes and payload + alignment - 1 <= request,
                "allocated backing cannot contain the complete aligned source payload",
            )
            block, ordinal = positions.get(allocation, (None, None))
            _require(
                block in cfg.block_set and indexes[block] not in cyclic,
                "packing allocation executes inside a CFG cycle",
            )
            call_block, call_ordinal = positions[call]
            _require(
                cfg.dominance.dominates(block, call_block) and (block is not call_block or ordinal < call_ordinal),
                "allocation does not dominate its uses",
            )
            row = buffers.setdefault(
                raw, dict(allocation=allocation, requested=request, payload=0, chains=set(), alignments=set(), uses=[])
            )
            row["payload"] = max(row["payload"], payload)
            row["chains"].update(chain)
            row["alignments"].add(alignment)
            row["uses"].append(call)
            allowed_calls.setdefault(call, set()).update((start, start + 1))
            observed.append(
                dict(
                    symbol=symbol,
                    argument=argument,
                    shape=list(shape),
                    element_bytes=element_bytes,
                    offset=0,
                    element_strides=strides,
                    logical_payload_bytes=payload,
                    alignment=alignment,
                    allocation_operation=operation_ids[allocation],
                    call_operation=operation_ids[call],
                )
            )
    _require(
        {(row["symbol"], row["argument"]) for row in observed} == selected,
        "actual packing coverage differs from original uses",
    )
    buffer_records = []
    for raw, row in buffers.items():
        aliases, frontier, memory_uses, releases = set(), [raw], set(), set()
        while frontier:
            value = frontier.pop()
            if value in aliases:
                continue
            aliases.add(value)
            _require(len(aliases) <= max_alias_values, "packing alias analysis budget exceeded")
            for use in value.uses:
                user = use.operation
                _require(user in positions, "packing storage escapes the original model entry")
                allocation_block, allocation_position = positions[row["allocation"]]
                user_block, user_position = positions[user]
                _require(
                    cfg.dominance.dominates(allocation_block, user_block)
                    and (allocation_block is not user_block or allocation_position < user_position),
                    "allocation does not dominate a packing alias use",
                )
                if user in row["chains"]:
                    frontier.extend(user.results)
                elif user.name in ("llvm.getelementptr", "llvm.bitcast") and use.index == 0:
                    _require(
                        len(user.results) == 1 and user.results[0].type == llvm.LLVMPointerType(),
                        "unknown pointer alias representation",
                    )
                    frontier.extend(user.results)
                elif user.name == "llvm.load" and use.index == 0:
                    _require(
                        len(user.results) == 1 and type(user.results[0].type) is builtin.IntegerType,
                        "packing buffer contains a retained pointer or aggregate",
                    )
                elif user.name == "llvm.store" and use.index == 1:
                    _require(
                        type(user.operands[0].type) is builtin.IntegerType,
                        "packing buffer stores a pointer or aggregate",
                    )
                elif user in allowed_calls and use.index in allowed_calls[user]:
                    pass
                elif _callee(user) == allocation_effects.deallocator_symbol:
                    _require(
                        value is raw and tuple(user.operands) == (raw,) and not user.results and _ccc(user),
                        "deallocator does not release the exact raw allocation",
                    )
                    releases.add(user)
                    continue
                else:
                    _require(False, "unknown packing pointer use, escape or unsupported alias")
                memory_uses.add(user)
        _require(len(releases) <= 1, "packing allocation can be released more than once")
        for release in releases:
            block, position = positions[release]
            _require(indexes[block] not in cyclic, "packing release executes inside a CFG cycle")
            later, todo = set(), list(cfg.successors[block])
            while todo:
                next_block = todo.pop()
                if next_block not in later:
                    later.add(next_block)
                    todo.extend(cfg.successors[next_block])
            _require(
                all(
                    (positions[op][0] is block and positions[op][1] < position)
                    or (positions[op][0] is not block and positions[op][0] not in later)
                    for op in memory_uses
                ),
                "packing storage is used after release",
            )
        buffer_records.append(
            dict(
                allocation_operation=operation_ids[row["allocation"]],
                logical_payload_bytes=row["payload"],
                allocator_requested_bytes=row["requested"],
                alignment_overhead_bytes=row["requested"] - row["payload"],
                proven_alignments=sorted(row["alignments"]),
                alias_values=len(aliases),
                release_operations=sorted(operation_ids[op] for op in releases),
                released_once=bool(releases),
                allocation_outside_all_cfg_cycles=True,
            )
        )
    payload = sum(row["logical_payload_bytes"] for row in buffer_records)
    requested = sum(row["allocator_requested_bytes"] for row in buffer_records)
    _require(payload <= max_payload_bytes, "actual distinct packing payload exceeds original budget")
    return dict(
        schema="merlin.final_packing_storage.v1",
        entry=entry,
        logical_payload_bytes=payload,
        allocator_requested_bytes=requested,
        alignment_overhead_bytes=requested - payload,
        max_payload_bytes=max_payload_bytes,
        buffers=buffer_records,
        uses=observed,
        original_borrowed_calls=[asdict(call) for call in borrowed_calls],
        allocation_effects=asdict(allocation_effects),
        scope="conditional per-entry packing ownership and payload; excludes other buffers, ABI scratch and total RAM",
        unproved=[
            "runtime/provider implementation qualification",
            "all host gather access bounds and values",
            "native machine-code equivalence",
            "performance and physical traffic",
        ],
    )


def inspect_packing_storage(
    host_artifacts,
    *,
    borrowed_calls,
    uses,
    allocation_effects,
    max_payload_bytes,
    entry,
    diagnostics,
    max_operations=100000,
    max_alias_values=4096,
    max_ir_bytes=64 << 20,
):
    from .host_call_alignment import import_host_entry, revalidate_host_entry

    _require(diagnostics is not None, "final packing proof requires retained import products")
    _require(type(allocation_effects) is PackingAllocationEffects, "original allocator effects are unavailable")
    validate_calls(borrowed_calls)
    _require(
        type(uses) is tuple and uses and all(type(use) is PackingStorageUse for use in uses),
        "an immutable original packing-use roster is required",
    )
    originals = json.dumps(
        dict(
            calls=[asdict(call) for call in borrowed_calls],
            uses=[asdict(use) for use in uses],
            effects=asdict(allocation_effects),
        ),
        sort_keys=True,
    )
    owners = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in implementation_paths()}
    module, observation = import_host_entry(
        host_artifacts,
        diagnostics=diagnostics,
        entry=entry,
        include_functions=tuple(
            dict.fromkeys(
                (
                    *(use.wrapper_symbol for use in uses),
                    allocation_effects.allocator_symbol,
                    allocation_effects.deallocator_symbol,
                )
            )
        ),
        max_ir_bytes=max_ir_bytes,
    )
    result = verify_packing_storage(
        module,
        borrowed_calls=borrowed_calls,
        uses=uses,
        allocation_effects=allocation_effects,
        max_payload_bytes=max_payload_bytes,
        entry=entry,
        max_operations=max_operations,
        max_alias_values=max_alias_values,
    )
    revalidate_host_entry(observation)
    _require(
        originals
        == json.dumps(
            dict(
                calls=[asdict(call) for call in borrowed_calls],
                uses=[asdict(use) for use in uses],
                effects=asdict(allocation_effects),
            ),
            sort_keys=True,
        ),
        "original packing contracts changed during inspection",
    )
    _require(
        owners == {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in implementation_paths()},
        "packing checker implementation changed during inspection",
    )
    result["observation"] = observation
    pins = {**owners, **observation["inputs"], **observation["tools"], **observation["products"]}
    payload = (json.dumps(result, sort_keys=True, indent=2) + "\n").encode()
    digest = hashlib.sha256(payload).hexdigest()
    report = Path(diagnostics) / ("packing-storage-" + digest + ".json")
    if report.exists():
        _require(not report.is_symlink() and report.read_bytes() == payload, "retained packing report changed")
    else:
        report.write_bytes(payload)
    result.update(pins={**pins, str(report.resolve()): digest}, report=dict(path=str(report.resolve()), sha256=digest))
    return result
