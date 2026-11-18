"""Typed scalar MLIR correspondence for exact elementwise composition.

This is an immutable verification DAG, not another arithmetic language. The
operation meanings and widths remain those of the actual scalar MLIR. Only
total integer operations are admitted; no reassociation, approximation or
floating-environment policy is inferred from a target capability.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import lru_cache

from merlin.compiler.kernel_ir.ir import Call, Kernel
from merlin.compiler.schedule.storage.temporaries import (
    StorageAdapter,
    StorageEffects,
    TemporaryObject,
    _overlap,
    _problem,
    _project,
)
from merlin.verification.references.epilogue_applicability import ReadoutCapability, selectors_applying

# ---- fusion_semantics --------------------------------------------------------------------------


class UnsupportedElementwise(ValueError):
    """Keep the original operation: its scalar semantics are outside this family."""


class ScalarBudgetExceeded(UnsupportedElementwise):
    """A candidate exceeds a selected compile-time bound; retain its source."""


@dataclass(frozen=True)
class ElementwiseSemantics:
    """Qualified contiguous elementwise execution with ordinary byte memory.

    Every read access is one function argument, in access order; the sole final
    write is the scalar return. Every access has exactly `elements` values.
    All arguments of an element are read before writing that element. There
    are no hidden effects, volatile accesses, state, traps or cross-element
    dependencies. Exact in-place aliases are allowed; shifted aliases are not.
    These execution properties require independent adapter qualification.
    """

    function_ir: bytes
    elements: int
    stages: tuple[str, ...]
    byte_orders: tuple[str, ...]
    readout: str | None = None

    def __post_init__(self):
        if (
            type(self.function_ir) is not bytes
            or not self.function_ir
            or type(self.elements) is not int
            or self.elements <= 0
            or type(self.stages) is not tuple
            or not self.stages
            or any(not isinstance(stage, str) or not stage for stage in self.stages)
            or type(self.byte_orders) is not tuple
            or not self.byte_orders
            or any(not isinstance(order, str) or order not in {"little", "big"} for order in self.byte_orders)
            or self.readout is not None
            and (not isinstance(self.readout, str) or not self.readout)
        ):
            raise ValueError(
                "elementwise semantics require typed scalar IR, extent, ordered stages and explicit byte layouts"
            )


@dataclass(frozen=True)
class ScalarGraph:
    input_bits: tuple[int, ...]
    output_bits: int
    # Typed MLIR applications with SSA operands numbered after the arguments.
    operations: tuple[tuple[str, str, tuple, tuple[int, ...]], ...]
    returned: int

    def expression(self, arguments):
        if len(arguments) != len(self.input_bits):
            raise ValueError("scalar expression argument count differs from its actual typed function")
        values = list(arguments)
        for name, dtype, properties, operands in self.operations:
            values.append(term((name, dtype, properties, tuple(values[index] for index in operands))))
        return values[self.returned]


def term(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


@lru_cache(maxsize=128)
def scalar_graph(source: bytes, max_operations: int, max_ir_bytes: int) -> ScalarGraph:
    """Reopen actual scalar IR; cache only an immutable, byte-bound typed DAG."""
    if type(max_operations) is not int or max_operations <= 0 or type(max_ir_bytes) is not int or max_ir_bytes <= 0:
        raise ValueError("scalar fusion requires positive explicit operation and source-byte budgets")
    if len(source) > max_ir_bytes:
        raise ScalarBudgetExceeded("scalar function exceeds its selected source-byte budget")
    from xdsl.dialects import arith, builtin, func

    from merlin.compiler.transforms.certificates import parse_ir
    from merlin.compiler.transforms.structural_rewrites import eligible_scalar_operation

    module = parse_ir(source.decode("utf-8"))
    functions = tuple(module.body.block.ops)
    if module.attributes or module.properties or len(functions) != 1 or type(functions[0]) is not func.FuncOp:
        raise UnsupportedElementwise("elementwise computation requires exactly one plain scalar function")
    function = functions[0]
    if (
        function.attributes
        or set(function.properties) - {"sym_name", "function_type", "sym_visibility"}
        or len(function.body.blocks) != 1
        or len(function.function_type.outputs) != 1
    ):
        raise UnsupportedElementwise("elementwise function requires a plain single-block single-result definition")
    block = function.body.block

    def bits(typ, *, memory=False):
        if (
            not isinstance(typ, builtin.IntegerType)
            or typ.signedness.data != builtin.Signedness.SIGNLESS
            or not 1 <= typ.width.data <= 256
            or memory
            and typ.width.data % 8
        ):
            raise UnsupportedElementwise(
                "scalar fusion supports total fixed-width integers with byte-sized memory elements"
            )
        return int(typ.width.data)

    inputs = tuple(bits(arg.type, memory=True) for arg in block.args)
    output = bits(function.function_type.outputs.data[0], memory=True)
    ops = tuple(block.ops)
    if len(ops) > max_operations + 1:
        raise ScalarBudgetExceeded("scalar function exceeds its selected operation budget")
    if not ops or type(ops[-1]) is not func.ReturnOp or len(ops[-1].operands) != 1:
        raise UnsupportedElementwise("elementwise scalar function needs one explicit return")
    if ops[-1].attributes or ops[-1].properties:
        raise UnsupportedElementwise("scalar return has unmodeled metadata")
    values = {value: index for index, value in enumerate(block.args)}
    applications = []
    extra = (
        arith.CmpiOp,
        arith.AndIOp,
        arith.OrIOp,
        arith.ShRSIOp,
        arith.SelectOp,
        arith.MinSIOp,
        arith.MaxSIOp,
        arith.MinUIOp,
        arith.MaxUIOp,
        arith.ExtSIOp,
        arith.ExtUIOp,
        arith.TruncIOp,
    )
    # Some xDSL versions omit effect traits on these standard scalar casts.
    # Admit only these exact, total builtin classes, never an operation merely
    # sharing their namespace. This does not broaden the frozen scalar passes.
    for op in ops[:-1]:
        admitted = eligible_scalar_operation(op)
        if type(op) in extra:
            admitted = (
                not op.attributes
                and not op.regions
                and not op.successors
                and len(op.results) == 1
                and set(op.properties) <= ({"predicate"} if type(op) is arith.CmpiOp else set())
            )
            if type(op) is arith.CmpiOp:
                admitted = (
                    admitted and isinstance(op.predicate, builtin.IntegerAttr) and 0 <= op.predicate.value.data <= 9
                )
        if type(op) is arith.ShRSIOp:
            # A shift is total only when its actual SSA amount is a constant
            # within the operand width; dynamic/oversized shifts can be poison.
            owner = op.operands[1].owner
            admitted = (
                admitted
                and type(owner) is arith.ConstantOp
                and eligible_scalar_operation(owner)
                and isinstance(owner.value, builtin.IntegerAttr)
                and 0 <= owner.value.value.data < bits(op.operands[0].type)
            )
        if not admitted:
            raise UnsupportedElementwise(f"unmodeled or non-total scalar operation {op.name}")
        for value in (*op.operands, *op.results):
            bits(value.type)
        if any(value not in values for value in op.operands):
            raise UnsupportedElementwise("scalar function reads an external or undefined SSA value")
        properties = tuple(sorted((name, str(value)) for name, value in op.properties.items()))
        applications.append(
            (op.name, str(op.results[0].type), properties, tuple(values[value] for value in op.operands))
        )
        values[op.results[0]] = len(values)
    returned = ops[-1].operands[0]
    if returned not in values or bits(returned.type, memory=True) != output:
        raise UnsupportedElementwise("scalar return differs from its declared result")
    return ScalarGraph(inputs, output, tuple(applications), values[returned])


# ---- fusion ------------------------------------------------------------------------------------
# Consume one qualified physical composition and verify its actual arithmetic.
#
# Only adjacent, synchronous elementwise operations with a private, unobserved
# intermediate are composed. Single-readout admission reuses the existing public
# capability vocabulary. Instruction encoding and physical view selection remain
# target-owned; Merlin reconstructs the returned call's numerical and memory
# effects before allocation, then the ordinary pipeline checks the final bytes.


@dataclass(frozen=True)
class FusionCapability:
    readout: ReadoutCapability
    emit: Callable[[Call, Call], Call | None]
    configuration_identity: str
    numerical_policy_identity: str
    qualification_identity: str

    def __post_init__(self):
        if (
            type(self.readout) is not ReadoutCapability
            or not isinstance(self.readout.selector, str)
            or not self.readout.selector
            or not isinstance(self.readout.evidence, str)
            or type(self.readout.applies) is not frozenset
            or any(not isinstance(stage, str) or not stage for stage in self.readout.applies)
            or not callable(self.emit)
            or any(
                not isinstance(value, str) or not value
                for value in (
                    self.configuration_identity,
                    self.numerical_policy_identity,
                    self.qualification_identity,
                )
            )
        ):
            raise ValueError("fusion requires one explicit readout, physical emitter and qualified contract identities")

    def record(self):
        return {
            "readout": self.readout.to_dict(),
            "configuration_identity": self.configuration_identity,
            "numerical_policy_identity": self.numerical_policy_identity,
            "qualification_identity": self.qualification_identity,
        }


@dataclass(frozen=True)
class FusionCandidate:
    kernel: Kernel
    objects: tuple[TemporaryObject, ...]
    checks: tuple[dict, ...]


def validate_fusion_capabilities(capabilities, adapter):
    if type(adapter) is not StorageAdapter:
        raise ValueError("fusion requires a selected typed storage and numerical adapter")
    if type(capabilities) is not tuple or any(type(capability) is not FusionCapability for capability in capabilities):
        raise ValueError("fusion capabilities must be explicit immutable selections")
    if any(
        capability.configuration_identity != adapter.configuration_identity
        or capability.numerical_policy_identity != adapter.numerical_policy_identity
        for capability in capabilities
    ):
        raise ValueError("fusion capabilities do not match selected hardware and numerical policy")


def _elementwise(effect, banks, max_scalar_operations, max_scalar_ir_bytes):
    semantics = effect.elementwise
    if semantics is None or effect.exact_copy or effect.completion != "synchronous":
        raise UnsupportedElementwise("no qualified synchronous elementwise arithmetic")
    graph = scalar_graph(semantics.function_ir, max_scalar_operations, max_scalar_ir_bytes)
    if (
        len(effect.accesses) != len(graph.input_bits) + 1
        or len(semantics.byte_orders) != len(effect.accesses)
        or not effect.accesses[-1].writes
    ):
        raise UnsupportedElementwise("elementwise effects do not match scalar arguments and sole final result")
    reads, write = effect.accesses[:-1], effect.accesses[-1]
    if any(access.writes for access in reads) or any(
        access.size != semantics.elements * (bits // 8)
        for access, bits in zip(effect.accesses, (*graph.input_bits, graph.output_bits), strict=True)
    ):
        raise UnsupportedElementwise("elementwise accesses do not cover every typed element exactly")
    if any(
        _overlap(access, write, banks)
        and (access.start != write.start or access.size != write.size or bits != graph.output_bits)
        for access, bits in zip(reads, graph.input_bits, strict=True)
    ):
        raise UnsupportedElementwise("shifted or differently typed elementwise aliases have no in-place contract")
    return graph, reads, write


def _physical(access, banks):
    backing = ("private", banks[access.space].backing) if access.space in banks else ("external", access.space)
    return backing, access.start, access.size


def _pair(index, effects, bindings, intervals, banks, max_scalar_operations, max_scalar_ir_bytes):
    producer, consumer = effects[index : index + 2]
    p_graph, p_reads, intermediate = _elementwise(producer, banks, max_scalar_operations, max_scalar_ir_bytes)
    c_graph, c_reads, output = _elementwise(consumer, banks, max_scalar_operations, max_scalar_ir_bytes)
    if producer.elementwise.elements != consumer.elementwise.elements:
        raise UnsupportedElementwise("composition would change element counts or introduce a broadcast")
    binding = bindings[index][-1]
    if binding is None or intervals[binding[0]][0] != index:
        raise UnsupportedElementwise("fusion intermediate is not newly initialized private storage")
    name = binding[0]
    if bindings[index + 1][-1] is not None and bindings[index + 1][-1][0] == name:
        raise UnsupportedElementwise("this composition requires a distinct final output backing")
    intermediates = []
    for operand, (read, bits) in enumerate(zip(c_reads, c_graph.input_bits, strict=True)):
        if (
            _physical(read, banks) == _physical(intermediate, banks)
            and bits == p_graph.output_bits
            and consumer.elementwise.byte_orders[operand] == producer.elementwise.byte_orders[-1]
        ):
            intermediates.append(operand)
    if not intermediates:
        raise UnsupportedElementwise("consumer does not read the complete typed producer result")
    merged = (
        *zip(p_reads, p_graph.input_bits, strict=True),
        *(
            (read, bits)
            for operand, (read, bits) in enumerate(zip(c_reads, c_graph.input_bits, strict=True))
            if operand not in intermediates
        ),
    )
    if any(
        _overlap(read, output, banks)
        and (read.start != output.start or read.size != output.size or bits != c_graph.output_bits)
        for read, bits in merged
    ):
        raise UnsupportedElementwise("composition would introduce an unsupported streaming alias")
    # Reuse the actual completion-aware lifetime. The initializer is the
    # producer, and adjacent calls leave no intervening place for an observer.
    if intervals[name][1] != index + 1 or any(
        item is not None and item[0] == name and operand not in intermediates
        for operand, item in enumerate(bindings[index + 1])
    ):
        raise UnsupportedElementwise("private intermediate has another view, writer or observer")
    return name, p_graph, c_graph, p_reads, c_reads, output, tuple(intermediates)


def verify_fusion_step(
    original,
    realized,
    objects,
    banks,
    adapter,
    capability,
    *,
    external_windows=(),
    private_ownership=False,
    max_calls=4096,
    max_objects=256,
    max_scalar_operations=128,
    max_scalar_ir_bytes=65536,
):
    """Find the realized collapsed pair; accept no selected pair or arithmetic plan."""
    validate_fusion_capabilities((capability,), adapter)
    bank_map, source, bindings, intervals, *_ = _problem(
        original,
        objects,
        banks,
        adapter,
        external_windows,
        private_ownership,
        max_calls,
        max_objects,
    )
    if (original.name, original.args, original.attrs) != (realized.name, realized.args, realized.attrs):
        raise ValueError("fusion changed the original invocation boundary")
    actual = _project(realized, adapter, max_calls)
    if len(actual) != len(source) - 1:
        raise ValueError("a physical fusion step must remove exactly one actual instruction")
    index = 0
    while index < len(actual) and actual[index] == source[index]:
        index += 1
    if index + 1 >= len(source) or source[index + 2 :] != actual[index + 1 :]:
        raise ValueError("fusion changed operations outside one adjacent pair")
    name, p_graph, c_graph, p_reads, c_reads, output, intermediates = _pair(
        index,
        source,
        bindings,
        intervals,
        bank_map,
        max_scalar_operations,
        max_scalar_ir_bytes,
    )
    fused = actual[index]
    try:
        graph, reads, write = _elementwise(fused, bank_map, max_scalar_operations, max_scalar_ir_bytes)
    except UnsupportedElementwise as error:
        raise ValueError("physical fusion has unsupported actual arithmetic: " + str(error)) from error
    stages = (*source[index].elementwise.stages, *source[index + 1].elementwise.stages)
    if (
        not selectors_applying((capability.readout,), stages)
        or fused.elementwise.stages != stages
        or fused.elementwise.readout != capability.readout.selector
    ):
        raise ValueError("no single selected readout applies the realized ordered composition")
    expected_reads = (*p_reads, *(read for operand, read in enumerate(c_reads) if operand not in intermediates))
    expected_bits = (
        *p_graph.input_bits,
        *(bits for operand, bits in enumerate(c_graph.input_bits) if operand not in intermediates),
    )
    expected_orders = (
        *source[index].elementwise.byte_orders[:-1],
        *(
            order
            for operand, order in enumerate(source[index + 1].elementwise.byte_orders[:-1])
            if operand not in intermediates
        ),
        source[index + 1].elementwise.byte_orders[-1],
    )
    if (
        fused.elementwise.elements != source[index].elementwise.elements
        or graph.input_bits != expected_bits
        or graph.output_bits != c_graph.output_bits
        or fused.elementwise.byte_orders != expected_orders
        or tuple(_physical(read, bank_map) for read in reads)
        != tuple(_physical(read, bank_map) for read in expected_reads)
        or _physical(write, bank_map) != _physical(output, bank_map)
    ):
        raise ValueError("physical fusion changed actual operands, layouts, extents or numerical boundaries")
    arguments = tuple(
        term(("value", bits, order, _physical(read, bank_map)))
        for read, bits, order in zip(expected_reads, expected_bits, expected_orders[:-1], strict=True)
    )
    producer = p_graph.expression(arguments[: len(p_reads)])
    remaining = iter(arguments[len(p_reads) :])
    consumer = c_graph.expression(
        tuple(producer if operand in intermediates else next(remaining) for operand in range(len(c_reads)))
    )
    if graph.expression(arguments) != consumer:
        raise ValueError("actual fused arithmetic changes operation order, conversion, rounding or saturation")
    return name, {
        "schema": "merlin.physical_fusion_step.v1",
        "source_sha256": original.digest(),
        "realized_sha256": realized.digest(),
        "capability": capability.record(),
        "elements": fused.elementwise.elements,
        "ordered_stages": list(stages),
        "scalar_expression_sha256": consumer,
        "original_commands": len(source),
        "actual_commands": len(actual),
        "removed_private_object": name,
        "original_requested_read_bytes": sum(
            access.size for effect in source for access in effect.accesses if not access.writes
        ),
        "original_requested_write_bytes": sum(
            access.size for effect in source for access in effect.accesses if access.writes
        ),
        "actual_requested_read_bytes": sum(
            access.size for effect in actual for access in effect.accesses if not access.writes
        ),
        "actual_requested_write_bytes": sum(
            access.size for effect in actual for access in effect.accesses if access.writes
        ),
        "machine_emission": "not_established",
        "execution_qualification": "not_established",
        "timing": "UNKNOWN",
    }


def plan_fusion(
    kernel,
    objects,
    banks,
    adapter,
    capabilities,
    *,
    external_windows=(),
    private_ownership=False,
    max_calls=4096,
    max_objects=256,
    max_fusions=8,
    max_scalar_operations=128,
    max_scalar_ir_bytes=65536,
) -> FusionCandidate | None:
    """Build and check actual target calls; allocation can still select fallback."""
    validate_fusion_capabilities(capabilities, adapter)
    if type(max_fusions) is not int or max_fusions < 0:
        raise ValueError("fusion requires an explicit nonnegative rewrite budget")
    checks = []
    for _attempt in range(max_fusions):
        bank_map, effects, bindings, intervals, *_ = _problem(
            kernel,
            objects,
            banks,
            adapter,
            external_windows,
            private_ownership,
            max_calls,
            max_objects,
        )
        selected = None
        for index in range(len(effects) - 1):
            try:
                _pair(index, effects, bindings, intervals, bank_map, max_scalar_operations, max_scalar_ir_bytes)
            except UnsupportedElementwise:
                continue
            stages = (*effects[index].elementwise.stages, *effects[index + 1].elementwise.stages)
            for capability in capabilities:
                if not selectors_applying((capability.readout,), stages):
                    continue
                fused = capability.emit(kernel.body[index], kernel.body[index + 1])
                if fused is None:
                    continue
                if type(fused) is not Call or type(adapter.project(fused)) is not StorageEffects:
                    raise ValueError("physical fusion emitter did not produce a decodable actual call")
                try:
                    _elementwise(adapter.project(fused), bank_map, max_scalar_operations, max_scalar_ir_bytes)
                except ScalarBudgetExceeded:
                    continue
                candidate = replace(kernel, body=(*kernel.body[:index], fused, *kernel.body[index + 2 :]))
                removed, proof = verify_fusion_step(
                    kernel,
                    candidate,
                    objects,
                    banks,
                    adapter,
                    capability,
                    external_windows=external_windows,
                    private_ownership=private_ownership,
                    max_calls=max_calls,
                    max_objects=max_objects,
                    max_scalar_operations=max_scalar_operations,
                    max_scalar_ir_bytes=max_scalar_ir_bytes,
                )
                selected = candidate, tuple(obj for obj in objects if obj.name != removed), proof
                break
            if selected is not None:
                break
        if selected is None:
            break
        kernel, objects, proof = selected
        checks.append(proof)
    return FusionCandidate(kernel, objects, tuple(checks)) if checks else None
