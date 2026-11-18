"""Outline a selected execution plan in one transaction on a private source clone.

CPU source remains inline. Selected provider regions become private definitions
and calls with complete ordered live boundaries. Lowering their implementations
and constructing a target image are subsequent, distinct operations.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from math import prod
from pathlib import Path

from xdsl.dialects import builtin, func

from merlin.compiler.ir.boundary import live_values
from merlin.compiler.ir.region import preview_region
from merlin.compiler.partition.dispatch_program import build_dispatch_program
from merlin.compiler.partition.global_plan_emission import (
    BoundaryMapping,
    EmittedComponent,
    GlobalPlanEmission,
    dispatch_digest,
    verify_global_plan_emission,
)
from merlin.compiler.partition.outline import DispatchInfo, OutlineResult
from merlin.compiler.planning.cover_selection import SelectedExecution
from merlin.compiler.transforms.certificates import ir_digest
from merlin.verification.translation.pass_validation import verify_partition

# ---- regions -----------------------------------------------------------------------------------


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
        from merlin.compiler.compilation import Source
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
    from merlin.compiler.compilation import observe

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


# ---- connected ---------------------------------------------------------------------------------
# Plan connected SSA regions, complete live boundaries and source-owned lifetimes.


@dataclass(frozen=True)
class RegionStage:
    operation: object
    ordinal: int
    kind: str
    inputs: tuple[int, ...]
    output: int


@dataclass(frozen=True)
class ConnectedRegion:
    source: object
    source_sha256: str
    stages: tuple[RegionStage, ...]
    values: tuple
    inputs: tuple[int, ...]
    outputs: tuple[int, ...]
    lifetimes: tuple[tuple[int, int], ...]

    def revalidate(self):
        if ir_digest(self.source) != self.source_sha256:
            raise ValueError("connected region source changed; replan")
        current = tuple(self.source.walk())
        if any(stage.ordinal >= len(current) or current[stage.ordinal] is not stage.operation for stage in self.stages):
            raise ValueError("connected region operation ownership changed; replan")
        rebuilt = plan_connected_region(self.source, tuple(stage.operation for stage in self.stages))
        if rebuilt != self:
            raise ValueError("connected region dataflow or lifetimes changed; replan")

    def record(self):
        return {
            "source_ir_sha256": self.source_sha256,
            "stages": [
                {"ordinal": s.ordinal, "kind": s.kind, "inputs": s.inputs, "output": s.output} for s in self.stages
            ],
            "values": [
                {"type": str(value.type), "lifetime": life}
                for value, life in zip(self.values, self.lifetimes, strict=True)
            ],
            "inputs": self.inputs,
            "outputs": self.outputs,
            "maturity": "typed_source_plan_not_physical_residency",
        }


def plan_connected_region(module, operations):
    """Plan one ordered same-block region, retaining every external consumer.

    Source operand identity, never equal shape, connects stages. Initializers
    are excluded only after native pointwise admission proves they are unread.
    Unknown intervening operations refuse rather than disappearing. Inputs stay
    borrowed/read-only for the full region; externally observed outputs stay live.
    """
    from xdsl.dialects import arith, tensor
    from xdsl.dialects.builtin import IntegerType, TensorType

    from merlin.compiler.planning.frontiers import _structural_family
    from merlin.compiler.schedule.elementwise.pointwise_source import read_pointwise_source

    operations = tuple(operations)
    members = tuple(module.walk())
    if not 2 <= len(operations) <= 64 or len(members) > 10000 or len(set(operations)) != len(operations):
        raise ValueError("connected region requires two to sixty-four distinct bounded source operations")
    ordinals = {op: i for i, op in enumerate(members)}
    block = operations[0].parent
    if block is None or any(op not in ordinals or op.parent is not block for op in operations):
        raise ValueError("connected region requires current operations from one source block")
    body = tuple(block.ops)
    positions = tuple(body.index(op) for op in operations)
    if positions != tuple(sorted(positions)):
        raise ValueError("connected region must retain original program order")
    # Admit only source-proven pure zero fills used by selected contractions.
    # This permits dense -> pointwise -> dense regions without treating an
    # arbitrary intervening operation as an ignorable initializer.
    initializers = set()
    from merlin.compiler.schedule.dense.source import read_zero_seeded_integer_matmul

    for op in operations:
        if _structural_family(op) == "contraction":
            dtypes = tuple(str(value.type.element_type) for value in (*op.operands[:2], op.results[0]))
            read_zero_seeded_integer_matmul(op, dtypes=dtypes)
            seed = op.operands[2].owner
            if getattr(seed, "name", None) == "linalg.fill" and all(
                use.operation in operations for value in seed.results for use in value.uses
            ):
                initializers.add(seed)
    for op in body[positions[0] : positions[-1] + 1]:
        if (
            op not in operations
            and op not in initializers
            and not (isinstance(op, (arith.ConstantOp, tensor.EmptyOp)) or op.name == "tensor.splat")
        ):
            raise ValueError("connected region crosses an unmodeled operation or observer")
    values, inputs, stages = [], [], []
    created, last = [], []
    for index, op in enumerate(operations):
        kind = _structural_family(op)
        if kind == "elementwise_map":
            read_pointwise_source(op, max_elements=64 << 20, max_operations=128, max_ir_bytes=65536)
            operands = tuple(op.inputs)
        elif kind == "contraction":
            operands = tuple(op.operands[:2])
        else:
            raise ValueError("connected region needs native pointwise or declared contraction source")
        if len(op.results) != 1:
            raise ValueError("connected stage needs one explicit tensor result")
        used = []
        for value in (*operands, op.results[0]):
            typ = value.type
            if (
                not isinstance(typ, TensorType)
                or not isinstance(typ.element_type, IntegerType)
                or typ.element_type.width.data % 8
                or any(n <= 0 for n in typ.get_shape())
                or prod(typ.get_shape()) > 64 << 20
            ):
                raise ValueError("connected tensor requires bounded static byte-addressable integer storage")
            if value not in values:
                if value.owner in operations and value is not op.results[0]:
                    raise ValueError("connected input is not available in source program order")
                values.append(value)
                external = value is not op.results[0]
                created.append(-1 if external else index)
                last.append(len(operations) if external else index)
                if external:
                    inputs.append(len(values) - 1)
            ident = values.index(value)
            last[ident] = max(last[ident], index)
            used.append(ident)
        stages.append(RegionStage(op, ordinals[op], kind, tuple(used[:-1]), used[-1]))
    outputs = tuple(
        stage.output for stage in stages if any(use.operation not in operations for use in values[stage.output].uses)
    )
    for ident in outputs:
        last[ident] = len(operations)
    connected = {0}
    for _ in stages:
        for index, stage in enumerate(stages):
            for value in stage.inputs:
                producer = created[value]
                if producer >= 0 and (producer in connected or index in connected):
                    connected.update((producer, index))
    if not outputs or len(connected) != len(stages):
        raise ValueError("connected region needs one producer-consumer component and an externally visible output")
    return ConnectedRegion(
        module,
        ir_digest(module),
        tuple(stages),
        tuple(values),
        tuple(inputs),
        outputs,
        tuple(zip(created, last, strict=True)),
    )


# ---- model -------------------------------------------------------------------------------------
# Export the ordinary whole-model host/device boundary without building an OOT backend.
#
# Inputs are an explicitly selected capture, target and corpus binder. The same
# group rewrite used by whole-model compilation owns layout, fusion, preparation
# and call signatures. This endpoint neither changes quantization nor runs a
# simulator; emitted interfaces are requests to a backend, not execution evidence.


def export_rewrite(
    module, rewrite, *, binder, output: str | Path, source_reference: str, connect_spatial: bool = False
) -> dict:
    """Export a completed ordinary rewrite and its deduplicated backend requests.

    Accepting the rewrite itself lets a caller retain its earlier work. Every
    actual call must have a matching statement; a partial interface export is
    reported as incomplete, never as a completed model compilation.
    """
    from merlin.compiler.lower.device.whole_program import group_interface
    from merlin.target.contract.interfaces.grammar import parse_interface_mlir

    connected = None
    if connect_spatial:
        from merlin.compiler.lower.device.spatial import group_connected_spatial

        selection = group_connected_spatial(module, rewrite)
        module, rewrite, connected = selection.module, selection.rewrite, selection.receipt

    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    module.verify()
    host_text = str(module) + "\n"
    (root / "host.mlir").write_text(host_text)
    sidecar = rewrite.to_sidecar()
    (root / "device_signatures.json").write_text(json.dumps(sidecar, indent=2) + "\n")
    requests = []
    for index, (symbol, entry) in enumerate(sorted(rewrite.entries.items())):
        row = {"symbol": symbol, "entry": entry}
        try:
            text = group_interface(entry, binder, source_reference=source_reference)
            parse_interface_mlir(text)
            filename = f"interface-{index}.mlir"
            (root / filename).write_text(text)
            row.update(status="emitted", file=filename, sha256=hashlib.sha256(text.encode()).hexdigest())
        except Exception as error:
            row.update(status="refused", error=type(error).__name__, reason=str(error))
        requests.append(row)
    emitted = {row["symbol"] for row in requests if row["status"] == "emitted"}
    missing = sorted({row.symbol for row in rewrite.routed} - emitted)
    report = {
        "schema": "model_interface_export_v1",
        "status": "incomplete" if missing or rewrite.skipped else "interface_exported",
        "source_reference": source_reference,
        "host_ir_sha256": hashlib.sha256(host_text.encode()).hexdigest(),
        "device_calls": len(rewrite.routed),
        "calls_with_interface": sum(row.symbol in emitted for row in rewrite.routed),
        "unique_interfaces": len(emitted),
        "missing_call_interfaces": missing,
        "skipped": list(rewrite.skipped),
        "requests": requests,
        "scope": "host/device boundary only; host operations remain in host.mlir",
        "target_execution": False,
        "numerical_evaluation": False,
    }
    if connected is not None:
        report["connected_spatial_selection"] = connected
    (root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def export_capture(
    capture: str | Path, *, target: str, binder, output: str | Path, connect_spatial: bool = False
) -> dict:
    """Run ordinary complete-group lowering up to the authored backend boundary.

    The caller selects the target contract/facts with the normal target registry
    context and supplies its corpus binder. No target or model defaults are
    discovered here. Source captures are read only.
    """
    from merlin.compiler.ir import mlir_query as mq
    from merlin.compiler.lower.device.device_offload import rewrite_groups_to_device

    source = Path(capture)
    source = source / "model.mlir" if source.is_dir() else source
    module = mq.parse(source.read_text())
    rewrite = rewrite_groups_to_device(
        module,
        target,
        select=lambda _: True,
        capture=source,
        require_complete_groups=True,
        blob_weights=True,
    )
    return export_rewrite(
        module, rewrite, binder=binder, output=output, source_reference=str(source), connect_spatial=connect_spatial
    )
