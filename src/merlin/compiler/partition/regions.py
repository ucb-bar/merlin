"""Outline a selected execution plan in one transaction on a private source clone.

CPU source remains inline. Selected provider regions become private definitions
and calls with complete ordered live boundaries. Lowering their implementations
and constructing a target image are subsequent, distinct operations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from xdsl.dialects import builtin, func

from merlin.compiler.dialects.lowering.dispatch_program import build_dispatch_program
from merlin.compiler.dialects.lowering.global_plan_emission import (
    BoundaryMapping,
    EmittedComponent,
    GlobalPlanEmission,
    dispatch_digest,
    verify_global_plan_emission,
)
from merlin.compiler.dialects.lowering.outline import DispatchInfo, OutlineResult
from merlin.compiler.ir.boundary import live_values
from merlin.compiler.ir.region import preview_region
from merlin.compiler.planning.execution import SelectedExecution
from merlin.compiler.transforms.certificates import ir_digest
from merlin.verification.partition import verify_partition


def _source_members():
    import sys

    from merlin.compiler.ir import region

    owner = sys.modules[__name__]
    return (
        *(
            (owner, name)
            for name in (
                "_source_members",
                "partition_execution",
                "preview_region",
                "verify_partition",
                "_region_record",
                "_emission_record",
                "ir_digest",
                "live_values",
                "verify_global_plan_emission",
                "build_dispatch_program",
            )
        ),
        (PartitionedExecution, "revalidate"),
        (PartitionedExecution, "source_observation"),
        (PartitionRegion, "request_source"),
        *region._source_members(),
    )


@dataclass(frozen=True)
class PartitionRegion:
    alternative: str
    provider: str
    symbol: str
    source_module: builtin.ModuleOp
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    source_ordinals: tuple[int, ...]

    def request_source(self, entry: str):
        """Serialize this exact region at the explicitly selected provider entry."""
        from merlin.compiler.request import Source
        from merlin.compiler.transforms.certificates import ir_text

        if type(entry) is not str or not entry:
            raise ValueError("region source needs an explicit provider entry")
        module = self.source_module.clone()
        definition = module.body.block.first_op
        definition.properties["sym_name"] = builtin.StringAttr(entry)
        definition.properties.pop("sym_visibility", None)
        module.verify()
        return Source(ir_text(module), origin=self.symbol)


def _region_record(regions):
    return tuple(
        (r.alternative, r.provider, r.symbol, r.inputs, r.outputs, r.source_ordinals, ir_digest(r.source_module))
        for r in regions
    )


def _emission_record(emission):
    return json.dumps(emission.receipt(), sort_keys=True, allow_nan=False)


@dataclass(frozen=True)
class PartitionedExecution:
    selection: SelectedExecution
    module: builtin.ModuleOp
    regions: tuple[PartitionRegion, ...]
    emission: GlobalPlanEmission
    verification: object
    _digest: str
    _region_record: tuple
    _emission_record: str
    _owners: tuple
    _sources: object

    def revalidate(self):
        if any(
            a is not b
            for a, b in zip(
                (self.selection, self.module, self.regions, self.emission, self._sources), self._owners, strict=True
            )
        ):
            raise ValueError("partition ownership changed after materialization")
        self._sources.revalidate()
        self.selection.revalidate()
        self._sources.revalidate()
        if ir_digest(self.module) != self._digest or _region_record(self.regions) != self._region_record:
            raise ValueError("partitioned source changed after materialization")
        if _emission_record(self.emission) != self._emission_record:
            raise ValueError("partition emission changed after materialization")
        receipt = verify_partition(
            self.selection.source.module,
            self.module,
            entry=self.selection.source.entry,
            symbols=tuple(r.symbol for r in self.regions),
        )
        if receipt != self.verification:
            raise ValueError("partition verification changed")
        problems = verify_global_plan_emission(self.selection.source.program, self.selection.plan, self.emission)
        self._sources.revalidate()
        if problems:
            raise ValueError("partition emission: " + "; ".join(problems))

    def source_observation(self):
        return self._sources.record()


def _values(source):
    entry = next(
        op for op in source.module.body.block.ops if type(op) is func.FuncOp and op.sym_name.data == source.entry
    )
    result = {}
    for name, buffer in source.program.buffers.items():
        if buffer.kind == "arg":
            result[entry.body.block.args[buffer.arg_index]] = name
    for operation, node in zip(source.operations, source.program.nodes, strict=True):
        if len(operation.results) != len(node.outputs):
            raise ValueError("source graph lost operation result correspondence")
        result.update(zip(operation.results, node.outputs, strict=True))
    return result


def partition_execution(selection: SelectedExecution) -> PartitionedExecution:
    """Validate the entire roster before mutation; never replan after one provider."""
    from merlin.compiler.sourceguard import observe

    sources = observe("partition", _source_members)
    if type(selection) is not SelectedExecution:
        raise ValueError("partitioning requires an original selected execution")
    selection.revalidate()
    sources.revalidate()
    source = selection.source
    if source.graph_mode != "ssa":
        raise ValueError("source-preserving partitioning requires the complete SSA graph contract")
    module = source.module.clone()
    original_walk, cloned_walk = tuple(source.module.walk()), tuple(module.walk())
    operation_map = dict(zip(original_walk, cloned_walk, strict=True))
    value_map = {}
    for original, copied in zip(original_walk, cloned_walk, strict=True):
        value_map.update(zip(original.results, copied.results, strict=True))
        for left, right in zip(original.regions, copied.regions, strict=True):
            for left_block, right_block in zip(left.blocks, right.blocks, strict=True):
                value_map.update(zip(left_block.args, right_block.args, strict=True))
    logical = _values(source)
    symbols = {op.sym_name.data for op in module.body.block.ops if type(op) is func.FuncOp}
    descriptions = []
    # Preflight every region's complete original boundary and symbol ownership.
    for index, item in enumerate(selection.selected):
        if item.proposal.abi == "inlined":
            continue
        operations = tuple(source.operations[n] for n in item.alternative.nodes)
        inputs, outputs = live_values(operations)
        wanted_in = tuple(rep.buffer for rep in item.alternative.inputs)
        wanted_out = tuple(rep.buffer for rep in item.alternative.outputs)
        if set(logical[v] for v in inputs) != set(wanted_in) or set(logical[v] for v in outputs) != set(wanted_out):
            raise ValueError("selected region differs from its complete live SSA boundary")
        inputs = tuple(sorted(inputs, key=lambda v: wanted_in.index(logical[v])))
        outputs = tuple(sorted(outputs, key=lambda v: wanted_out.index(logical[v])))
        if any(type(v.type) is not builtin.TensorType for v in (*inputs, *outputs)):
            raise ValueError("logical tensor region ABI does not admit raw scalar boundaries")
        symbol = f"{source.entry}$kernel_execution_{index}"
        if symbol in symbols:
            raise ValueError("partition symbol conflicts with an original function")
        symbols.add(symbol)
        preview = preview_region(source.module, source.entry, operations, symbol=symbol, inputs=inputs, outputs=outputs)
        sources.revalidate()
        preview.source_observation.revalidate()
        if preview.source_ordinals != item.proposal.operation_ordinals:
            raise ValueError("selected region differs from the actual original source operation ordinals")
        definition = preview.module.body.block.first_op.clone()
        region = PartitionRegion(
            item.alternative.id,
            item.proposal.provider,
            symbol,
            builtin.ModuleOp([definition.clone()]),
            wanted_in,
            wanted_out,
            item.proposal.operation_ordinals,
        )
        region.source_module.verify()
        descriptions.append((item, operations, inputs, outputs, definition, region))
    selection.revalidate()
    sources.revalidate()
    dispatches = []
    owned_operations = {}
    for item, operations, inputs, outputs, definition, region in descriptions:
        originals = [operation_map[op] for op in operations]
        call = func.CallOp(region.symbol, [value_map[v] for v in inputs], [v.type for v in outputs])
        originals[0].parent.insert_op_before(call, originals[0])
        for old, new in zip(outputs, call.results, strict=True):
            value_map[old].replace_all_uses_with(new)
            value_map[old] = new
        for operation in reversed(originals):
            operation.parent.erase_op(operation)
        module.body.block.add_op(definition)
        owned_operations[item.alternative.id] = (call,)
        dispatches.append(
            DispatchInfo(len(dispatches), region.symbol, "selected-region", len(inputs), [str(v.type) for v in outputs])
        )
    for item in selection.selected:
        if item.proposal.abi == "inlined":
            owned_operations[item.alternative.id] = tuple(
                operation_map[source.operations[n]] for n in item.alternative.nodes
            )
    module.verify()
    verification = verify_partition(
        source.module, module, entry=source.entry, symbols=tuple(row[-1].symbol for row in descriptions)
    )
    sources.revalidate()
    # This graph is the actual partitioned source graph, before backend lowering.
    emitted = build_dispatch_program(OutlineResult(module, dispatches), entry=source.entry)
    driver = next(op for op in module.body.block.ops if type(op) is func.FuncOp and op.sym_name.data == source.entry)
    positions = {op: index for index, op in enumerate(tuple(driver.body.block.ops)[:-1])}
    emitted_values = {}
    for name, buffer in emitted.buffers.items():
        if buffer.kind == "arg":
            emitted_values[driver.body.block.args[buffer.arg_index]] = name
    for operation, node in zip(tuple(driver.body.block.ops)[:-1], emitted.nodes, strict=True):
        emitted_values.update(zip(operation.results, node.outputs, strict=True))
    names = {
        name: emitted_values[value_map[value]] for value, name in logical.items() if value_map[value] in emitted_values
    }
    components = tuple(
        EmittedComponent(
            item.alternative.id,
            tuple(positions[op] for op in owned_operations[item.alternative.id]),
            tuple((rep.buffer, names[rep.buffer]) for rep in item.alternative.inputs),
            tuple((rep.buffer, names[rep.buffer]) for rep in item.alternative.outputs),
        )
        for item in selection.selected
    )
    boundaries = tuple(
        BoundaryMapping("input", name, names[name])
        for name, buffer in source.program.buffers.items()
        if buffer.kind == "arg"
    ) + tuple(BoundaryMapping("output", name, names[name]) for name in source.program.results)
    emission = GlobalPlanEmission(
        emitted,
        selection.plan.digest,
        dispatch_digest(source.program),
        components,
        (),
        boundaries,
        ("source-preserving region partition; no backend execution claim",),
    )
    regions = tuple(row[-1] for row in descriptions)
    result = PartitionedExecution(
        selection,
        module,
        regions,
        emission,
        verification,
        ir_digest(module),
        _region_record(regions),
        _emission_record(emission),
        (selection, module, regions, emission, sources),
        sources,
    )
    result.revalidate()
    return result
