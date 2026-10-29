"""Original-source grammar for the bounded uniform private-buffer model.

Read-only classification, never a producer, target matcher or proof. Runtime
controllers own UNVERIFIED coverage for explicit source exclusions.
"""

from __future__ import annotations

from xdsl.dialects import arith, builtin, func, memref, scf

from .compiler_profile import SourceOutsideProfile
from .lean_graph_input_admission import InputAdmissionError, admit_module, admit_text
from .memory_scalar_nan_admission import admit_memory_scalar_attributes, parse_memory_module

PROFILE_ID = "buffer-uniform-private-rank1-rank2-index64-v1"
MODEL = {
    "id": PROFILE_ID,
    "gate": "buffer_reuse",
    "claim": "equal_possible_successful_public_observations",
    "root": "GraphMemoryAfterEntries.every_definition",
    "bounds": {
        "source_bytes": 256 * 1024,
        "syntax_nodes": 5000,
        "rank": [1, 2],
        "element_bits": [8, 16, 32, 64],
        "bits": 64,
        "tile": [2, 65536],
        "IR_boundaries": [3, 4],
    },
    "source": "static private integer allocations, captured uniform fills, explicit entry-block lifetimes",
    "excluded": ["computed initializer", "rank3+", "dynamic geometry", "escape", "completion/control", "NaN metadata"],
    "premises": ["typed inputs", "total/private/unobservable allocation", "shared partial primitives"],
    "not_proved": ["categorized errors", "native allocator", "LLVM or hardware fidelity"],
}


def _outside(code, reason):
    raise SourceOutsideProfile(PROFILE_ID, code, reason)


def _kind(kind):
    if isinstance(kind, builtin.IndexType):
        return
    if isinstance(kind, builtin.IntegerType):
        if kind.width.data in (8, 16, 32, 64) and kind.signedness.data == builtin.Signedness.SIGNLESS:
            return
        _outside("scalar-type", "BUFFER model admits declared signless i8/i16/i32/i64 scalar types")
    if isinstance(kind, builtin.MemRefType):
        if not isinstance(kind.layout, builtin.NoneAttr) or not isinstance(kind.memory_space, builtin.NoneAttr):
            _outside("view-layout", "BUFFER model requires static default-space contiguous views")
        if any(type(n) is not int or not 0 < n < (1 << 63) for n in kind.get_shape()):
            _outside("view-shape", "BUFFER model requires positive fitting static view extents")
        if not isinstance(kind.element_type, (builtin.IntegerType, builtin.IndexType)):
            _outside("view-element", "BUFFER views require scalar integer/index element representations")
        _kind(kind.element_type)
        span = kind.element_type.compile_time_size
        for extent in kind.get_shape():
            if span > ((1 << 63) - 1) // extent:
                _outside("view-span", "BUFFER static view span exceeds its signed-index64 domain")
            span *= extent
        return
    _outside("value-type", "BUFFER source has a value type outside its scalar/index/ranked-view model")


def _known(value):
    dimensions = []
    while type(value.owner) is memref.DimOp:
        dimensions.append(value.owner)
        value = value.owner.operands[1]
    owner = value.owner
    if type(owner) is arith.ConstantOp and isinstance(owner.value, builtin.IntegerAttr):
        result = owner.value.value.data
    else:
        return None
    for dimension in reversed(dimensions):
        kind = dimension.operands[0].type
        if not isinstance(kind, builtin.MemRefType) or not 0 <= result < len(kind.get_shape()):
            return None
        result = kind.get_shape()[result]
    return result


def _top(operation, block):
    while operation.parent_block() is not block:
        operation = operation.parent_op()
        if operation is None:
            return None
    return operation


def _uniform(loop):
    loops = []
    current = loop
    while type(current) is scf.ForOp:
        if current.attributes or current.properties or len(current.operands) != 3 or current.results:
            _outside("initializer-carries", "BUFFER initializer is a plain no-carry counted loop")
        body = current.body.block
        operations = tuple(body.ops)
        if len(body.args) != 1 or len(operations) != 2 or type(operations[-1]) is not scf.YieldOp:
            _outside("computed-initializer", "BUFFER initializer contains an unsupported prefix or control body")
        if operations[-1].operands:
            _outside("initializer-yield", "BUFFER initializer does not carry yielded values")
        loops.append(current)
        current = operations[0]
    if type(current) is not memref.StoreOp:
        _outside("initializer-store", "BUFFER initializer must end in its one actual store")
    return loops, current


def admit_source(raw, *, options):
    """Return the fixed model id; explicit source exclusions are the only bypass.

    Invalid selections, parse/verification errors, resource limits and unexpected
    implementation failures propagate. This function reads no target and calls
    neither normalization, reuse, tiling nor a proof/SDK runner.
    """
    from .compiler_transforms import ALLOCATOR, Options

    if type(options) is not Options or options.buffer_allocator != ALLOCATOR:
        raise ValueError("original typed BUFFER options and allocator selection required")
    options.__post_init__()
    if type(raw) is not str:
        raise TypeError("original complete BUFFER text required")
    if len(raw.encode()) > 256 * 1024:
        _outside("source-bytes", "BUFFER source exceeds the declared raw-byte model bound")
    try:
        admit_text(raw)
        module = parse_memory_module(raw)
        admit_module(module, calls="shape-direct")
        admit_memory_scalar_attributes(module)
    except InputAdmissionError as error:
        _outside(error.code, str(error))
    operations = tuple(module.walk())
    if len(operations) > 5000:
        _outside("source-nodes", "BUFFER source exceeds its declared syntax-node bound")
    if module.attributes or module.properties:
        _outside("module-metadata", "BUFFER model requires a plain complete module header")
    functions = tuple(module.body.block.ops)
    if any(type(operation) is not func.FuncOp or len(operation.body.blocks) != 1 for operation in functions):
        _outside("entry-shape", "BUFFER basic all-entry model requires defined single-block functions")
    for operation in operations:
        if operation.successors:
            _outside("CFG", "BUFFER source is outside the single-block structured-control model")
        for value in (*operation.operands, *operation.results):
            _kind(value.type)
        for region in operation.regions:
            for block in region.blocks:
                for value in block.args:
                    _kind(value.type)
        if operation.regions and type(operation) not in (builtin.ModuleOp, func.FuncOp, scf.ForOp):
            _outside("region", "BUFFER model does not interpret this structured region")
        if type(operation) is memref.CastOp:
            if operation.attributes or operation.properties or operation.operands[0].type != operation.results[0].type:
                _outside("cast", "BUFFER normalization admits only exact same-type casts")
        if type(operation) is memref.DimOp and (
            operation.attributes or operation.properties or _known(operation.results[0]) is None
        ):
            _outside("dimension", "BUFFER normalization requires exact static dimension metadata")
        if operation.name in ("memref.subview", "memref.reinterpret_cast", "memref.view", "memref.reshape"):
            _outside("descriptor", "BUFFER model excludes unsupported descriptor construction")
    allocated_functions = []
    for function in functions:
        block = function.body.block
        positions = {operation: index for index, operation in enumerate(block.ops)}
        allocations = [operation for operation in block.ops if type(operation) is memref.AllocOp]
        if not allocations:
            if any(type(operation) is scf.ForOp for operation in function.walk()):
                _outside("retained-loop", "BUFFER profile excludes non-initializer structured loops")
            continue
        allocated_functions.append(function.sym_name.data)
        if any(type(operation) is memref.AllocOp and operation not in positions for operation in function.walk()):
            _outside("nested-allocation", "BUFFER allocations must have explicit entry-block lifetimes")
        aliases = {}
        for allocation in allocations:
            kind = allocation.memref.type
            if not isinstance(kind.element_type, builtin.IntegerType):
                _outside("allocation-element", "BUFFER private allocations require integer elements")
            if len(kind.get_shape()) not in (1, 2):
                _outside("initializer-rank", "BUFFER initializer model supports rank1/rank2 only")
            if allocation.dynamic_sizes or allocation.symbol_operands:
                _outside("dynamic-allocation", "BUFFER model requires static concrete private allocations")
            aliases[allocation.memref] = allocation
        for operation in function.walk():
            if type(operation) is memref.CastOp and operation.operands[0] in aliases:
                aliases[operation.results[0]] = aliases[operation.operands[0]]
        fills = {}
        for loop in (operation for operation in block.ops if type(operation) is scf.ForOp):
            nest, store = _uniform(loop)
            allocation = aliases.get(store.memref)
            if allocation is None or allocation in fills:
                _outside("initializer-object", "BUFFER needs one private full initializer per logical epoch")
            shape = allocation.memref.type.get_shape()
            if len(nest) != len(shape) or tuple(store.indices) != tuple(item.body.block.args[0] for item in nest):
                _outside("initializer-coordinate", "BUFFER initializer coordinates must be its exact lexical IVs")
            for nested, extent in zip(nest, shape, strict=True):
                if (_known(nested.lb), _known(nested.ub), _known(nested.step)) != (0, extent, 1):
                    _outside("initializer-bounds", "BUFFER initializer requires exact zero/extent/unit controls")
            fills[allocation] = loop
        for allocation in allocations:
            if allocation not in fills:
                _outside("initializer-missing", "BUFFER private epoch lacks a supported complete uniform initializer")
            uses = [use.operation for value, owner in aliases.items() if owner is allocation for use in value.uses]
            frees = [operation for operation in uses if type(operation) is memref.DeallocOp]
            if len(frees) != 1 or frees[0] not in positions:
                _outside("free", "BUFFER private epoch requires one explicit entry-block free")
            begin, fill, end = positions[allocation], positions[fills[allocation]], positions[frees[0]]
            if not begin < fill < end:
                _outside("lifetime", "BUFFER initializer/free must occur after allocation in its source lifetime")
            for operation in uses:
                if type(operation) not in (
                    memref.CastOp,
                    memref.DimOp,
                    memref.LoadOp,
                    memref.StoreOp,
                    memref.DeallocOp,
                ):
                    _outside("private-escape", "BUFFER private views may not escape through unsupported uses")
                if type(operation) in (memref.LoadOp, memref.StoreOp):
                    owner = _top(operation, block)
                    if owner not in positions or not fill <= positions[owner] < end:
                        _outside(
                            "memory-lifetime", "BUFFER memory access lies outside its initialized private lifetime"
                        )
    if len(allocated_functions) != 1:
        _outside("changed-entry", "BUFFER basic profile requires one private-allocation owner entry")
    owner = allocated_functions[0]
    graph = {function.sym_name.data: set() for function in functions}
    for function in functions:
        graph[function.sym_name.data].update(
            operation.callee.root_reference.data for operation in function.walk() if type(operation) is func.CallOp
        )
    for operation in operations:
        if type(operation) is func.CallOp and operation.callee.root_reference.data == owner:
            _outside("called-private-entry", "BUFFER callers reaching the changed entry need a separate context model")
    remaining = {name: len(targets) for name, targets in graph.items()}
    callers = {name: [] for name in graph}
    for name, targets in graph.items():
        for target in targets:
            callers[target].append(name)
    ready = [name for name, count in remaining.items() if not count]
    visited = 0
    while ready:
        visited += 1
        for name in callers[ready.pop()]:
            remaining[name] -= 1
            if remaining[name] == 0:
                ready.append(name)
    if visited != len(graph):
        _outside("recursive-calls", "BUFFER basic source profile excludes recursive helper graphs")
    return PROFILE_ID
