"""Memoize adjacent bit-identical exp arguments inside a closed tensor loop.

Adjacent cache state is an invocation-local scalar loop value. Its first/new
argument invokes the original exp. An alternative bounded private cache also
reuses exact earlier results for finite arguments in [-64,64], where exp is
finite and normal without range errno. Every miss/outside-domain argument keeps
the original exp. No approximated result, global state, changed arithmetic
order or foreign-call purity is introduced.
"""

from __future__ import annotations

FEATURE = "memoize_adjacent_exp_arguments"
MARKER = "__merlin_memoize_adjacent_exp_arguments__"
BOUNDED_FEATURE = "memoize_bounded_exp_arguments"
BOUNDED_MARKER = "__merlin_memoize_bounded_exp_arguments__"
LARGE_FEATURE = "memoize_bounded_exp_arguments_128"
LARGE_MARKER = "__merlin_memoize_bounded_exp_arguments_128__"
ERRNO_FEATURE = "memoize_errno_checked_exp_arguments"
ERRNO_MARKER = "__merlin_memoize_errno_checked_exp_arguments__"
UNDERFLOW_FEATURE = "memoize_selected_expf_underflow_class"
UNDERFLOW_MARKER = "__merlin_memoize_selected_expf_underflow_class__"
INVOCATION_FEATURE = "memoize_invocation_selected_expf"
INVOCATION_MARKER = "__merlin_memoize_invocation_selected_expf__"
INPUT_FEATURE = "memoize_invocation_exp_input_words"
INPUT_MARKER = "__merlin_memoize_invocation_exp_input_words__"
INPUT_UNROLL_FEATURE = "unroll_exp_input_cache_by_2"
INPUT_UNROLL_MARKER = "__merlin_unroll_exp_input_cache_2__"
INPUT_UNROLL4_FEATURE = "unroll_exp_input_cache_by_4"
INPUT_UNROLL4_MARKER = "__merlin_unroll_exp_input_cache_4__"
RUNTIME_SYMBOL = "__merlin_exp_errno_location"
RUNTIME_C = r"""
#include <errno.h>
#include <math.h>
int *__merlin_exp_errno_location(void) { return &errno; }
float __merlin_exp_observe_errno(float x) { return expf(x); }
"""


def runtime_c_for(llvm_source):
    if not any(line.startswith("declare ") and "@" + RUNTIME_SYMBOL + "(" in line for line in llvm_source.splitlines()):
        return ""
    if "@__merlin_exp_environment_capture(" in llvm_source:
        from .exp_cache_environment import RUNTIME_C as environment_runtime

        return RUNTIME_C + environment_runtime
    return RUNTIME_C


def _edit_pipeline(passes: list[str], marker: str = MARKER) -> list[str]:
    hits = [i for i, p in enumerate(passes) if "one-shot-bufferize" in p]
    if len(hits) != 1 or any(
        p in (MARKER, BOUNDED_MARKER, LARGE_MARKER, ERRNO_MARKER, UNDERFLOW_MARKER, INVOCATION_MARKER, INPUT_MARKER)
        for p in passes
    ):
        raise ValueError("adjacent exp memoization requires one tensor bufferization boundary")
    if marker in (UNDERFLOW_MARKER, INVOCATION_MARKER, INPUT_MARKER):
        from .selected_expf_equivalence import current_expf_underflow_equivalence

        if current_expf_underflow_equivalence() is None:
            raise ValueError("selected expf underflow class requires an explicit live callee grant")
    i = hits[0]
    return [*passes[:i], marker, *passes[i:]]


def ensure_registered() -> str:
    from .impr_features import ImprFeature, known, register

    if FEATURE not in known():
        register(
            ImprFeature(
                name=FEATURE,
                action_class="PASS",
                description=(
                    "Reuse only adjacent bit-identical exp arguments within a closed tensor loop; "
                    "preserve the first original call and invocation-local state."
                ),
                edit_pipeline=_edit_pipeline,
                alternative_group="exact_exp_memoization",
            )
        )
    if BOUNDED_FEATURE not in known():
        register(
            ImprFeature(
                name=BOUNDED_FEATURE,
                action_class="PASS",
                description=(
                    "Extend adjacent exact exp reuse with a bounded private cache for finite [-64,64] arguments; "
                    "retain the original exp on every miss/outside-domain argument."
                ),
                edit_pipeline=lambda passes: _edit_pipeline(passes, BOUNDED_MARKER),
                alternative_group="exact_exp_memoization",
            )
        )
    if LARGE_FEATURE not in known():
        register(
            ImprFeature(
                name=LARGE_FEATURE,
                action_class="PASS",
                description=(
                    "Use 128 bounded private exact-exp slots with unchanged domain, original-call fallback "
                    "and total module storage cap."
                ),
                edit_pipeline=lambda passes: _edit_pipeline(passes, LARGE_MARKER),
                alternative_group="exact_exp_memoization",
            )
        )
    if ERRNO_FEATURE not in known():
        register(
            ImprFeature(
                name=ERRNO_FEATURE,
                action_class="PASS",
                description=(
                    "Cache exact non-NaN exp results privately only while the recorded post-call errno matches; "
                    "preserve original calls for misses and environment mismatches."
                ),
                edit_pipeline=lambda passes: _edit_pipeline(passes, ERRNO_MARKER),
                alternative_group="exact_exp_memoization",
            )
        )
    if UNDERFLOW_FEATURE not in known():
        register(
            ImprFeature(
                name=UNDERFLOW_FEATURE,
                action_class="PASS",
                description=(
                    "Reuse a supplied exact selected-expf finite-underflow input class under original first-call "
                    "and errno/sticky-environment guards; explicit live callee binding required."
                ),
                edit_pipeline=lambda passes: _edit_pipeline(passes, UNDERFLOW_MARKER),
                alternative_group="exact_exp_memoization",
            )
        )
    if INVOCATION_FEATURE not in known():
        register(
            ImprFeature(
                name=INVOCATION_FEATURE,
                action_class="PASS",
                description=(
                    "Share a bounded private selected-expf cache across sequential tensor loops only while "
                    "the complete portable floating environment matches; preserve per-entry errno guards."
                ),
                edit_pipeline=lambda passes: _edit_pipeline(passes, INVOCATION_MARKER),
                alternative_group="exact_exp_memoization",
            )
        )
    if INPUT_FEATURE not in known():
        register(
            ImprFeature(
                name=INPUT_FEATURE,
                action_class="PASS",
                description=(
                    "Memoize a closed scalar exp body for repeated exact input words "
                    "only while every other input stays invariant."
                ),
                edit_pipeline=lambda passes: _edit_pipeline(passes, INPUT_MARKER),
                alternative_group="exact_exp_memoization",
            )
        )
    for unroll_feature, unroll_marker, unroll_count in (
        (INPUT_UNROLL_FEATURE, INPUT_UNROLL_MARKER, 2),
        (INPUT_UNROLL4_FEATURE, INPUT_UNROLL4_MARKER, 4),
    ):
        if unroll_feature in known():
            continue

        def edit_unroll(passes, marker=unroll_marker):
            if passes.count(INPUT_MARKER) != 1 or any(
                hint in passes for hint in (INPUT_UNROLL_MARKER, INPUT_UNROLL4_MARKER)
            ):
                raise ValueError("exact exp input unrolling requires one cache stage")
            i = passes.index(INPUT_MARKER)
            return [*passes[:i], marker, *passes[i:]]

        register(
            ImprFeature(
                name=unroll_feature,
                action_class="PASS",
                description=(
                    f"Request stock LLVM unrolling by {unroll_count} for the exact input cache's innermost loops."
                ),
                edit_pipeline=edit_unroll,
                implies=frozenset({INPUT_FEATURE}),
            )
        )
    return FEATURE


def bind_runner(source, *, features):
    if not features & {UNDERFLOW_FEATURE, INVOCATION_FEATURE, INPUT_FEATURE}:
        return source
    from .selected_expf_equivalence import current_expf_underflow_equivalence

    grant = current_expf_underflow_equivalence()
    if grant is None:
        raise ValueError("selected expf underflow class requires an explicit live callee grant")
    anchor = "_AE_ORIGINAL_RUN_STAGES = _run_stages"
    if source.count(anchor) != 1:
        raise ValueError("selected expf class requires the original exact-exp runner")
    facts = (grant.minimum_input_bits, grant.maximum_input_bits)
    return source.replace(anchor, "_AE_SELECTED_EXPF_CLASS = " + repr(facts) + "\n" + anchor)


RUNNER_PRELUDE = r"""
def _ae_spec(operation):
    from torch_mlir import ir

    if (
        operation.name != "linalg.generic"
        or len(operation.results) != 1
        or len(operation.regions) != 1
        or len(operation.regions[0].blocks) != 1
        or len(operation.operands) < 2
    ):
        return None
    ancestor = operation
    while ancestor is not None:
        if any("strictfp" in name or "strictfp" in str(ancestor.attributes[name]) for name in ancestor.attributes):
            return None
        ancestor = ancestor.parent
    allowed = {"indexing_maps", "iterator_types", "operandSegmentSizes"}
    if any(name not in allowed and not name.startswith("prov.") for name in operation.attributes):
        return None
    types = [value.type for value in operation.operands]
    if any(
        not isinstance(typ, ir.RankedTensorType) or any(x < 0 for x in ir.RankedTensorType(typ).shape) for typ in types
    ):
        return None
    output = ir.RankedTensorType(types[-1])
    rank = len(output.shape)
    if not rank or str(output.element_type) != "f32" or operation.results[0].type != types[-1]:
        return None
    if [str(x) for x in operation.attributes["iterator_types"]] != ["#linalg.iterator_type<parallel>"] * rank:
        return None
    maps = [ir.AffineMapAttr(x).value for x in operation.attributes["indexing_maps"]]
    if len(maps) != len(types) or any(m.n_dims != rank or m.n_symbols for m in maps):
        return None
    output_map = maps[-1]
    if (
        len(output_map.results) != rank
        or any(not isinstance(x, ir.AffineDimExpr) for x in output_map.results)
        or sorted(ir.AffineDimExpr(x).position for x in output_map.results) != list(range(rank))
    ):
        return None
    extents = [None] * rank
    for expr, extent in zip(output_map.results, output.shape):
        extents[ir.AffineDimExpr(expr).position] = extent
    positions = []
    for affine, typ in zip(maps, types):
        shape = ir.RankedTensorType(typ).shape
        if len(affine.results) != len(shape):
            return None
        indices = []
        for expr, extent in zip(affine.results, shape):
            if isinstance(expr, ir.AffineDimExpr):
                axis = ir.AffineDimExpr(expr).position
                if extent != extents[axis]:
                    return None
                indices.append(axis)
            elif isinstance(expr, ir.AffineConstantExpr) and ir.AffineConstantExpr(expr).value == 0 and extent == 1:
                indices.append(None)
            else:
                return None
        positions.append(indices)
    block = operation.regions[0].blocks[0]
    body = list(block.operations)
    exps = [op for op in body if op.operation.name == "math.exp"]
    arithmetic = {
        "arith.constant",
        "arith.sitofp",
        "arith.uitofp",
        "arith.extf",
        "arith.truncf",
        "arith.addf",
        "arith.subf",
        "arith.mulf",
        "arith.divf",
        "arith.cmpf",
        "arith.cmpi",
        "arith.select",
        "arith.bitcast",
        "arith.extsi",
        "arith.extui",
        "arith.trunci",
        "arith.addi",
        "arith.subi",
        "arith.muli",
        "arith.andi",
        "arith.ori",
        "arith.xori",
    }
    if (
        len(exps) != 1
        or len(exps[0].operands) != 1
        or str(exps[0].operands[0].type) != "f32"
        or str(exps[0].results[0].type) != "f32"
        or body[-1].operation.name != "linalg.yield"
        or list(body[-1].operands) != [exps[0].results[0]]
        or list(block.arguments[-1].uses)
        or any(op.regions or op.operation.name not in arithmetic | {"math.exp"} for op in body[:-1])
        or any("fastmath" in op.attributes and str(op.attributes["fastmath"]) != "#arith.fastmath<none>" for op in body)
    ):
        return None
    return positions, extents


def _memoize_adjacent_tensor_exp(
    ctx, module, cache_entries=0, errno_checked=False, underflow_class=None,
    invocation_shared=False, memoize_inputs=False, inner_unroll=0
):
    from torch_mlir import ir

    plans = []
    groups = []

    def walk(parent):
        for region in parent.regions:
            for block in region.blocks:
                group = None
                if invocation_shared and parent.name == "func.func" and len(parent.regions[0].blocks) == 1:
                    group = len(groups)
                    groups.append(block)
                for child in list(block.operations):
                    spec = _ae_spec(child.operation)
                    if spec is not None:
                        if invocation_shared and group is None:
                            continue
                        plans.append((child.operation, spec, group))
                    else:
                        walk(child.operation)

    walk(module.operation)
    with ctx, ir.Location.unknown():
        index, i1, i32, f32 = (
            ir.IndexType.get(),
            ir.IntegerType.get_signless(1),
            ir.IntegerType.get_signless(32),
            ir.F32Type.get(),
        )
        if plans and errno_checked:
            functions = {
                "__merlin_exp_errno_location": ([], [ir.Type.parse("!llvm.ptr")]),
                "__merlin_exp_observe_errno": ([f32], [f32]),
            }
            if invocation_shared:
                for name in ("__merlin_exp_environment_matches", "__merlin_exp_environment_capture"):
                    functions[name] = ([ir.MemRefType.get([4096], ir.IntegerType.get_signless(8))], [i32])
            if any(
                "sym_name" in op.attributes and ir.StringAttr(op.attributes["sym_name"]).value in functions
                for op in module.body.operations
            ):
                raise ValueError("exp errno helper symbol collision")
            with ir.InsertionPoint(module.body):
                for symbol, (arguments, results) in functions.items():
                    ir.Operation.create(
                        "func.func",
                        attributes={
                            "sym_name": ir.StringAttr.get(symbol),
                            "sym_visibility": ir.StringAttr.get("private"),
                            "function_type": ir.TypeAttr.get(ir.FunctionType.get(arguments, results)),
                        },
                        regions=1,
                    )
        cache_budget = 8192
        shared = {}
        for old, (positions, extents), group in plans:
            previous = shared.get(group) if group is not None else None
            entries = previous[0] if previous else cache_entries if cache_budget >= cache_entries else 0
            if invocation_shared and group is None:
                entries = 0
            if previous is None:
                cache_budget -= entries
            input_key = None
            if memoize_inputs:
                varying = [i for i, axes in enumerate(positions[:-1]) if len(extents) - 1 in axes]
                if len(varying) == 1 and str(old.regions[0].blocks[0].arguments[varying[0]].type) == "f32":
                    input_key = varying[0]
            with old.location, ir.InsertionPoint(old):

                def create(name, operands=(), results=(), attributes=None, regions=0):
                    return ir.Operation.create(
                        name,
                        operands=list(operands),
                        results=list(results),
                        attributes=attributes or {},
                        regions=regions,
                    )

                def constant(value, typ=index):
                    return create(
                        "arith.constant",
                        results=[typ],
                        attributes={"value": ir.IntegerAttr.get(typ, value)},
                    ).results[0]

                zero, one = constant(0), constant(1)
                scalar_zero = create(
                    "arith.constant",
                    results=[f32],
                    attributes={"value": ir.FloatAttr.get(f32, 0.0)},
                ).results[0]
                state = [
                    old.operands[-1],
                    constant(0, i1),
                    constant(0, i32),
                    scalar_zero,
                ]
                state_types = [old.results[0].type, i1, i32, f32]
                original_block = old.regions[0].blocks[0]
                keys = values = errnos = errno_pointer = environment = None
                if previous is not None:
                    _, keys, values, errnos, errno_pointer, environment, environment_valid = previous
                if entries and errno_checked and previous is None:
                    if not invocation_shared:
                        errnos = create("memref.alloca", results=[ir.MemRefType.get([entries], i32)]).results[0]
                    errno_pointer = create(
                        "func.call",
                        results=[ir.Type.parse("!llvm.ptr")],
                        attributes={
                            "callee": ir.FlatSymbolRefAttr.get("__merlin_exp_errno_location"),
                        },
                    ).results[0]
                if entries:
                    if previous is None:
                        if invocation_shared:
                            storage = create(
                                "memref.alloca",
                                results=[ir.MemRefType.get([entries * 16], ir.IntegerType.get_signless(8))],
                                attributes={"alignment": ir.IntegerAttr.get(ir.IntegerType.get_signless(64), 16)},
                            ).results[0]
                            keys = create(
                                "memref.view", [storage, zero], [ir.MemRefType.get([entries, 4], i32)]
                            ).results[0]
                            values = create(
                                "memref.view", [storage, zero], [ir.MemRefType.get([entries, 4], f32)]
                            ).results[0]
                            errnos = keys
                        else:
                            keys = create("memref.alloca", results=[ir.MemRefType.get([entries], i32)]).results[0]
                            values = create("memref.alloca", results=[ir.MemRefType.get([entries], f32)]).results[0]
                        if group is not None:
                            environment = create(
                                "memref.alloca", results=[ir.MemRefType.get([4096], ir.IntegerType.get_signless(8))]
                            ).results[0]
                    def cache_indices(slot, field):
                        return [slot, constant(field)] if invocation_shared else [slot]
                    def initialize_keys():
                        initialize = create("scf.for", [zero, constant(entries), one], regions=1)
                        initialize_block = ir.Block.create_at_start(initialize.regions[0], [index])
                        with ir.InsertionPoint(initialize_block):
                            create(
                                "memref.store",
                                [constant(0x7FC00000, i32), keys, *cache_indices(initialize_block.arguments[0], 0)],
                            )
                            create("scf.yield")
                    if previous is None:
                        initialize_keys()
                    else:
                        checked = create("scf.if", [environment_valid], [i1], regions=2)
                        for branch in range(2):
                            block = ir.Block.create_at_start(checked.regions[branch], [])
                            with ir.InsertionPoint(block):
                                valid = constant(0, i1)
                                if branch == 0:
                                    status = create(
                                        "func.call", [environment], [i32],
                                        {"callee": ir.FlatSymbolRefAttr.get("__merlin_exp_environment_matches")},
                                    ).results[0]
                                    valid = create(
                                        "arith.cmpi", [status, constant(0, i32)], [i1],
                                        {"predicate": ir.IntegerAttr.get(ir.IntegerType.get_signless(64), 1)},
                                    ).results[0]
                                create("scf.yield", [valid])
                        clear = create("scf.if", [checked.results[0]], regions=2)
                        for branch in range(2):
                            block = ir.Block.create_at_start(clear.regions[branch], [])
                            with ir.InsertionPoint(block):
                                if branch:
                                    initialize_keys()
                                create("scf.yield")

                def original_exp(operands, attributes):
                    if entries and errno_checked:
                        return create(
                            "func.call",
                            operands,
                            [f32],
                            {**attributes, "callee": ir.FlatSymbolRefAttr.get("__merlin_exp_observe_errno")},
                        ).results[0]
                    return create("math.exp", operands, [f32], attributes).results[0]

                def lookup_exp(word, operands, attributes):
                    if not entries:
                        return original_exp(operands, attributes)
                    key_word = word
                    if underflow_class is not None:
                        minimum, maximum = underflow_class
                        lower = create(
                            "arith.cmpi",
                            [word, constant(minimum, i32)],
                            [i1],
                            {"predicate": ir.IntegerAttr.get(ir.IntegerType.get_signless(64), 9)},
                        ).results[0]
                        upper = create(
                            "arith.cmpi",
                            [word, constant(maximum, i32)],
                            [i1],
                            {"predicate": ir.IntegerAttr.get(ir.IntegerType.get_signless(64), 7)},
                        ).results[0]
                        member = create("arith.andi", [lower, upper], [i1]).results[0]
                        key_word = create("arith.select", [member, constant(minimum, i32), word], [i32]).results[0]
                    absolute = create("arith.andi", [word, constant(0x7FFFFFFF, i32)], [i32]).results[0]
                    bounded = create(
                        "arith.cmpi",
                        [absolute, constant(0x7F800000 if errno_checked else 0x42800000, i32)],
                        [i1],
                        {"predicate": ir.IntegerAttr.get(ir.IntegerType.get_signless(64), 7)},
                    ).results[0]
                    selected = create("scf.if", [bounded], [f32], regions=2)
                    for branch in range(2):
                        block = ir.Block.create_at_start(selected.regions[branch], [])
                        with ir.InsertionPoint(block):
                            if branch:
                                result = original_exp(operands, attributes)
                            else:
                                # Fold upper binary32 fraction bits with sign/exponent for the shared table.
                                fraction_shift = 23 - (entries.bit_length() - 1) if invocation_shared else 8
                                low = create(
                                    "arith.shrui", [key_word, constant(fraction_shift, i32)], [i32]
                                ).results[0]
                                high = create(
                                    "arith.shrui", [key_word, constant(23 if invocation_shared else 16, i32)], [i32]
                                ).results[0]
                                mixed = create("arith.xori", [low, high], [i32]).results[0]
                                slot_word = create(
                                    "arith.andi",
                                    [mixed, constant(entries - 1, i32)],
                                    [i32],
                                ).results[0]
                                slot = create("arith.index_cast", [slot_word], [index]).results[0]
                                stored = create("memref.load", [keys, *cache_indices(slot, 0)], [i32]).results[0]
                                hit = create(
                                    "arith.cmpi",
                                    [key_word, stored],
                                    [i1],
                                    {"predicate": ir.IntegerAttr.get(ir.IntegerType.get_signless(64), 0)},
                                ).results[0]
                                if errno_checked:
                                    matched = create("scf.if", [hit], [i1], regions=2)
                                    for effect_branch in range(2):
                                        effect_block = ir.Block.create_at_start(matched.regions[effect_branch], [])
                                        with ir.InsertionPoint(effect_block):
                                            effect = constant(0, i1)
                                            if effect_branch == 0:
                                                previous_errno = create(
                                                    "memref.load", [errnos, *cache_indices(slot, 2)], [i32]
                                                ).results[0]
                                                current_errno = create("llvm.load", [errno_pointer], [i32]).results[0]
                                                effect = create(
                                                    "arith.cmpi",
                                                    [previous_errno, current_errno],
                                                    [i1],
                                                    {
                                                        "predicate": ir.IntegerAttr.get(
                                                            ir.IntegerType.get_signless(64), 0
                                                        )
                                                    },
                                                ).results[0]
                                            create("scf.yield", [effect])
                                    hit = matched.results[0]
                                cached_exp = create("scf.if", [hit], [f32], regions=2)
                                for cache_branch in range(2):
                                    cache_block = ir.Block.create_at_start(cached_exp.regions[cache_branch], [])
                                    with ir.InsertionPoint(cache_block):
                                        if cache_branch:
                                            value = original_exp(operands, attributes)
                                            create("memref.store", [value, values, *cache_indices(slot, 1)])
                                            if errno_checked:
                                                error = create("llvm.load", [errno_pointer], [i32]).results[0]
                                                create("memref.store", [error, errnos, *cache_indices(slot, 2)])
                                            create("memref.store", [key_word, keys, *cache_indices(slot, 0)])
                                        else:
                                            value = create(
                                                "memref.load", [values, *cache_indices(slot, 1)], [f32]
                                            ).results[0]
                                        create("scf.yield", [value])
                                result = cached_exp.results[0]
                            create("scf.yield", [result])
                    return selected.results[0]

                def loop(depth, coordinates, state):
                    if depth < len(extents):
                        if input_key is not None and depth == len(extents) - 1:
                            state = [state[0], constant(0, i1), state[2], state[3]]
                        body = create(
                            "scf.for",
                            [zero, constant(extents[depth]), one, *state],
                            state_types,
                            attributes=({"loop_annotation": ir.Attribute.parse(
                                "#llvm.loop_annotation<unroll = <count = " + str(inner_unroll) + " : i32>>")}
                                if inner_unroll and depth == len(extents) - 1 else {}),
                            regions=1,
                        )
                        block = ir.Block.create_at_start(body.regions[0], [index, *state_types])
                        with ir.InsertionPoint(block):
                            create(
                                "scf.yield",
                                loop(
                                    depth + 1,
                                    [*coordinates, block.arguments[0]],
                                    list(block.arguments[1:]),
                                ),
                            )
                        return list(body.results)
                    mapping = {}
                    for argument, value, axes in zip(original_block.arguments[:-1], old.operands[:-1], positions[:-1]):
                        mapping[argument] = create(
                            "tensor.extract",
                            [
                                value,
                                *[zero if axis is None else coordinates[axis] for axis in axes],
                            ],
                            [argument.type],
                        ).results[0]
                    destination, valid, key, cached = state
                    if input_key is not None:
                        word = create("arith.bitcast", [mapping[original_block.arguments[input_key]]], [i32]).results[0]
                        same = create("arith.cmpi", [word, key], [i1],
                            {"predicate": ir.IntegerAttr.get(ir.IntegerType.get_signless(64), 0)}).results[0]
                        hit = create("arith.andi", [valid, same], [i1]).results[0]
                        repeated = create("scf.if", [hit], [f32], regions=2)
                        for branch in range(2):
                            block = ir.Block.create_at_start(repeated.regions[branch], [])
                            with ir.InsertionPoint(block):
                                value = cached
                                if branch:
                                    body_mapping = dict(mapping)
                                    for scalar in list(original_block.operations)[:-1]:
                                        operands = [body_mapping.get(v, v) for v in scalar.operands]
                                        attributes = {name: scalar.attributes[name] for name in scalar.attributes}
                                        if scalar.operation.name == "math.exp":
                                            exp_word = create("arith.bitcast", operands, [i32]).results[0]
                                            value = lookup_exp(exp_word, operands, attributes)
                                            body_mapping[scalar.results[0]] = value
                                        else:
                                            clone = create(scalar.operation.name, operands,
                                                [v.type for v in scalar.results], attributes)
                                            body_mapping.update(zip(scalar.results, clone.results))
                                    value = body_mapping[list(original_block.operations)[-1].operands[0]]
                                create("scf.yield", [value])
                        result = repeated.results[0]
                        destination = create("tensor.insert",
                            [result, destination, *[coordinates[x] for x in positions[-1]]],
                            [state_types[0]]).results[0]
                        return [destination, constant(1, i1), word, result]
                    for scalar in list(original_block.operations)[:-1]:
                        operands = [mapping.get(value, value) for value in scalar.operands]
                        attributes = {name: scalar.attributes[name] for name in scalar.attributes}
                        if scalar.operation.name != "math.exp":
                            clone = create(
                                scalar.operation.name,
                                operands,
                                [v.type for v in scalar.results],
                                attributes,
                            )
                        else:
                            word = create("arith.bitcast", operands, [i32]).results[0]
                            same = create(
                                "arith.cmpi",
                                [word, key],
                                [i1],
                                {"predicate": ir.IntegerAttr.get(ir.IntegerType.get_signless(64), 0)},
                            ).results[0]
                            hit = create("arith.andi", [valid, same], [i1]).results[0]
                            clone = create("scf.if", [hit], [f32], regions=2)
                            for branch in range(2):
                                block = ir.Block.create_at_start(clone.regions[branch], [])
                                with ir.InsertionPoint(block):
                                    result = cached if branch == 0 else lookup_exp(word, operands, attributes)
                                    create("scf.yield", [result])
                            key, cached, valid = word, clone.results[0], constant(1, i1)
                        mapping.update(zip(scalar.results, clone.results))
                    value = mapping[list(original_block.operations)[-1].operands[0]]
                    destination = create(
                        "tensor.insert",
                        [value, destination, *[coordinates[x] for x in positions[-1]]],
                        [state_types[0]],
                    ).results[0]
                    return [destination, valid, key, cached]

                result = loop(0, [], state)[0]
                for name in old.attributes:
                    if name.startswith("prov."):
                        result.owner.attributes[name] = old.attributes[name]
                previous = (
                    ir.StringAttr(old.attributes["prov.transforms"]).value + ","
                    if "prov.transforms" in old.attributes
                    else ""
                )
                result.owner.attributes["prov.transforms"] = ir.StringAttr.get(
                    previous
                    + (
                        "memoize_invocation_exp_input_words"
                        if input_key is not None
                        else "memoize_invocation_selected_expf"
                        if entries and group is not None
                        else "memoize_selected_expf_underflow_class"
                        if entries and underflow_class is not None
                        else "memoize_errno_checked_exp_arguments"
                        if entries and errno_checked
                        else "memoize_bounded_exp_arguments_128"
                        if entries == 128
                        else "memoize_bounded_exp_arguments"
                        if entries
                        else "memoize_adjacent_exp_arguments"
                    )
                )
                if entries and group is not None:
                    captured = create(
                        "func.call", [environment], [i32],
                        {"callee": ir.FlatSymbolRefAttr.get("__merlin_exp_environment_capture")},
                    ).results[0]
                    valid = create(
                        "arith.cmpi", [captured, constant(0, i32)], [i1],
                        {"predicate": ir.IntegerAttr.get(ir.IntegerType.get_signless(64), 1)},
                    ).results[0]
                    shared[group] = (entries, keys, values, errnos, errno_pointer, environment, valid)
                old.results[0].replace_all_uses_with(result)
                old.erase()
    module.operation.verify()
    return len(plans)


_AE_ORIGINAL_RUN_STAGES = _run_stages


def _run_stages(ctx, module, pipeline, erase, mid=(), late=(), post_openmp=(), pre_generalize=()):
    passes = [p for p in pipeline.split(",") if p]
    markers = {
        "__merlin_memoize_adjacent_exp_arguments__": (0, False),
        "__merlin_memoize_bounded_exp_arguments__": (64, False),
        "__merlin_memoize_bounded_exp_arguments_128__": (128, False),
        "__merlin_memoize_errno_checked_exp_arguments__": (128, True),
        "__merlin_memoize_selected_expf_underflow_class__": (128, True),
        "__merlin_memoize_invocation_selected_expf__": (1024, True),
        "__merlin_memoize_invocation_exp_input_words__": (1024, True),
    }
    unroll_counts = {"__merlin_unroll_exp_input_cache_2__": 2, "__merlin_unroll_exp_input_cache_4__": 4}
    unroll_hints = [p for p in passes if p in unroll_counts]
    if len(unroll_hints) > 1:
        raise ValueError("exact exp input unrolling must be unique")
    passes = [p for p in passes if p not in unroll_counts]
    selected = [p for p in passes if p in markers]
    if unroll_hints and selected != ["__merlin_memoize_invocation_exp_input_words__"]:
        raise ValueError("exact exp unrolling requires its input-word cache stage")
    if not selected:
        return _AE_ORIGINAL_RUN_STAGES(ctx, module, pipeline, erase, mid, late, post_openmp, pre_generalize)
    if len(selected) != 1:
        raise ValueError("exp memoization stage must be unique")
    marker = selected[0]
    entries, errno_checked = markers[marker]
    class_bounds = None
    if marker in ("__merlin_memoize_selected_expf_underflow_class__",
                  "__merlin_memoize_invocation_selected_expf__",
                  "__merlin_memoize_invocation_exp_input_words__"):
        if "_AE_SELECTED_EXPF_CLASS" not in globals():
            raise ValueError("missing compiler-bound selected expf equivalence")
        class_bounds = _AE_SELECTED_EXPF_CLASS
    i = passes.index(marker)
    _AE_ORIGINAL_RUN_STAGES(ctx, module, ",".join(passes[:i]), 0, (), (), (), pre_generalize)
    print("OK exact_exp_memoization", _memoize_adjacent_tensor_exp(
        ctx, module, entries, errno_checked, class_bounds,
        marker in ("__merlin_memoize_invocation_selected_expf__", "__merlin_memoize_invocation_exp_input_words__"),
        marker == "__merlin_memoize_invocation_exp_input_words__", unroll_counts[unroll_hints[0]] if unroll_hints else 0
    ))
    return _AE_ORIGINAL_RUN_STAGES(ctx, module, ",".join(passes[i + 1 :]), erase, mid, late, post_openmp, ())
"""
