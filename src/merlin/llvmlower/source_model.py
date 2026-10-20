"""One model preparation/link transaction for selected shared source families.

Discovery is fixed shared code. Target callbacks receive checked computations
for implementation, never an arbitrary source-selection hook. Families retain
their existing typed IR and pass pipelines; this module only joins their fresh
writer ABI, immutable inputs and ordinary provider/link lifecycle.
"""

from __future__ import annotations

import importlib
import json
from dataclasses import asdict, dataclass, fields, is_dataclass, replace
from pathlib import Path

from merlin.common.digest import is_sha256, sha256_file

_SOURCE_LABELS = {"row": "source_rows", "pointwise": "source_pointwise", "dense": "source_dense"}


@dataclass(frozen=True)
class SourceRegion:
    """Internal SSA replacement, not another computation or target IR."""

    compilation: object
    operations: tuple
    inputs: tuple
    shape: tuple[int, ...]


class _ConvolutionFallback(ValueError):
    """Private bounded repair before publishing any provider selection."""

    def __init__(self, reports):
        self.reports = reports
        super().__init__("not every bounded convolution arm acquired a dense provider")


@dataclass(frozen=True)
class _Family:
    name: str
    selection: object
    discover: object
    binding: object
    source_file: Path


def _values(value):
    if is_dataclass(value) and not isinstance(value, type):
        # IR owners are dataclasses too. Snapshot only the finite revalidated
        # selection, never recurse through a module's cyclic SSA ownership graph.
        from merlin.perf.offload_profitability import OffloadCostDecision

        from .offload_plan import OffloadPlanSelection

        if type(value) is OffloadCostDecision:
            value.__post_init__()
            if type(value._record) is not str:
                raise ValueError("shared source cost selection requires its original immutable report")
            # Keep the owner alive and compare identity before callable equality:
            # a replacement can define an equality operator that always succeeds.
            # This snapshot grants no freshness. revalidate() separately calls
            # the live owner once, including its own before/after checks.
            return type(value), id(value._check), value._check, value._construction, value._record
        if type(value) is OffloadPlanSelection:
            return type(value), _values(value.record())
        return type(value), tuple((f.name, _values(getattr(value, f.name))) for f in fields(value))
    if isinstance(value, (tuple, list)):
        return type(value), tuple(_values(v) for v in value)
    if isinstance(value, dict):
        return type(value), tuple((_values(k), _values(v)) for k, v in value.items())
    return value


class _SourceModelBuild:
    def __init__(self, families, device, device_admission):
        from . import borrowed_buffer_calls, fresh_tensor_writer

        self.families = families
        self.device, self.device_admission = device, device_admission
        self.bindings = None
        self.packing_plans = self._packing_stamp = ()
        self.selection_values = tuple(_values(f.selection) for f in families)
        self.components = tuple(
            (value, _values(value))
            for f in families
            for name in ("adapter", "encoding", "limits", "budgets")
            if (value := getattr(f.selection, name, None)) is not None
        )
        owners = (
            Path(__file__),
            Path(fresh_tensor_writer.__file__),
            Path(borrowed_buffer_calls.__file__),
            *(Path(__file__).with_name(name) for name in ("dealloc_lifetime.py", "pipeline.py", "lower.py")),
            *(f.source_file for f in families),
        )
        if any(getattr(f.selection, "logical_abi", None) is not None for f in families):
            from .logical_borrowed_abi import implementation_paths

            owners = (*owners, *implementation_paths())
        if any(getattr(f.selection, "offload_plan", None) is not None for f in families):
            from .source_dense import offload_implementation_paths

            owners = (*owners, *offload_implementation_paths())
        if any(getattr(f.selection, "max_convolution_packing_bytes", None) is not None for f in families):
            from merlin.verify.host_packing_storage import implementation_paths as packing_paths

            from .convolution_routes import implementation_paths

            owners = (*owners, *implementation_paths(), *packing_paths())
        self.pins = tuple((path.resolve(), sha256_file(path)) for path in dict.fromkeys(owners))
        if any(getattr(f.selection, "offload_plan", None) is not None for f in families):
            self.revalidate()

    def revalidate(self):
        checks_cost = any(getattr(f.selection, "offload_comparison", None) is not None for f in self.families)
        if self.packing_plans != self._packing_stamp:
            raise ValueError("original convolution packing coverage changed; replan")
        if tuple(_values(f.selection) for f in self.families) != self.selection_values or any(
            _values(value) != before for value, before in self.components
        ):
            raise ValueError("shared source capabilities or implementation selection changed; replan")
        for family in self.families:
            family.selection.selected_contracts.revalidate()
            if family.name == "dense":
                from .source_dense import validate_offload_selection

                validate_offload_selection(family.selection)
        # Live evidence owners are callbacks too. Recheck the inexpensive
        # selection snapshot after them without repeating their evidence I/O.
        if checks_cost and (
            tuple(_values(f.selection) for f in self.families) != self.selection_values
            or any(_values(value) != before for value, before in self.components)
        ):
            raise ValueError("shared source capabilities or implementation selection changed; replan")
        for path, digest in self.pins:
            if not path.is_file() or path.is_symlink() or sha256_file(path) != digest:
                raise ValueError("shared source preparation inputs or selected output changed; replan")
        if self.packing_plans or any(
            getattr(f.selection, "max_convolution_packing_bytes", None) is not None for f in self.families
        ):
            # Derive the receipt role from original family selection and pinned
            # files. A mutable optional receipt pointer cannot disable custody.
            receipts = {path for path, _ in self.pins if path.name == self._label() + ".json"}
            if hasattr(self, "selected_source") and len(receipts) != 1:
                raise ValueError("original convolution packing coverage lacks its pinned source receipt")
            for path in receipts:
                record = json.loads(path.read_text())
                if record.get("packing_plans", []) != json.loads(json.dumps(self.packing_plans)):
                    raise ValueError("original convolution packing coverage differs from its pinned source receipt")
        for group in self.bindings or ():
            for binding in group:
                binding.compilation.revalidate()
        self.require_device_admission()

    def require_device_admission(self):
        if self.bindings is not None and any(
            group and family.selection.execution_domain == "device"
            for family, group in zip(self.families, self.bindings, strict=True)
        ):
            from .device_build import require_device_elf_admission

            require_device_elf_admission(self.device, self.device_admission)

    def _label(self):
        return "source_schedules" if len(self.families) != 1 else _SOURCE_LABELS[self.families[0].name]

    def observe_native(self, paths, remember):
        """Retain actual checked lifetime files through provider/link validation."""
        self.revalidate()
        known = dict(self.pins)
        for path in paths:
            if path.is_symlink() or not path.is_file():
                raise ValueError("native lifetime observation requires unchanged regular files")
            digest = sha256_file(path)
            if not remember and known.get(path) != digest:
                raise ValueError("native lifetime observation is not an original retained checkpoint")
            if path in known and known[path] != digest:
                raise ValueError("native lifetime observation changed; replan")
            known[path] = digest
        if remember:
            self.pins = tuple(known.items())
        self.revalidate()

    def prepare(self, source_path, output):
        discarded = None
        try:
            return self._prepare(source_path, output)
        except _ConvolutionFallback as repair:
            self.revalidate()
            discarded = repair.reports
            # Release the discarded IR and compiled candidates held by the
            # failed frame before constructing a second full-model candidate.
            repair.__traceback__ = None
        return self._prepare(source_path, output, discarded_normalizations=discarded)

    def _prepare(self, source_path, output, *, discarded_normalizations=None):
        from xdsl.dialects import builtin, func, tensor
        from xdsl.rewriter import InsertPoint, Rewriter

        from merlin.common import mlir_query as mq
        from merlin.common.jsonio import write_pretty_json
        from merlin.transforms.certificates import ir_text

        from .fresh_tensor_writer import FreshTensorWriterContract, rewrite_fresh_tensor_writers

        if self.bindings is not None:
            raise ValueError("shared source preparation belongs to one model invocation")
        self.revalidate()
        original = mq.parse(source_path).clone()
        original.verify()
        normalizations = [] if discarded_normalizations is None else discarded_normalizations
        for family in self.families:
            if (
                discarded_normalizations is None
                and family.name == "dense"
                and family.selection.max_convolution_packing_bytes is not None
            ):
                from .convolution_routes import normalize_convolution_source

                original, report = normalize_convolution_source(original, family.selection, self.revalidate)
                if report["rewritten"]:
                    checkpoint = output / "convolution-normalized.mlir"
                    checkpoint.write_text(ir_text(original))
                    report["checkpoint"] = {"path": str(checkpoint.resolve()), "sha256": sha256_file(checkpoint)}
                    self.pins = (*self.pins, (checkpoint.resolve(), sha256_file(checkpoint)))
                normalizations.append(report)
        positions = {op: i for i, op in enumerate(original.walk())}
        reserved = {
            value.data
            for op in original.body.block.ops
            if isinstance(value := op.properties.get("sym_name", op.attributes.get("sym_name")), builtin.StringAttr)
        }
        groups, replacements, records, members, refusals, placements = [], [], [], set(), [], []
        fused_readouts = set()
        for family in self.families:
            regions, declined = family.discover(
                original,
                family.selection,
                self.revalidate,
                **({"excluded_operations": frozenset(fused_readouts)} if family.name == "pointwise" else {}),
            )
            self.revalidate()
            refusals.extend(declined)
            if family.name == "dense" and family.selection.offload_plan is not None:
                regions, placement = self._place_dense(original, family.selection, regions)
                placements.append(placement)
            group = []
            for region in regions:
                if members.intersection(region.operations):
                    # Dense discovery runs first. An exact readout already
                    # absorbed into its checked contraction is no longer a
                    # separate pointwise dispatch. All other overlaps refuse.
                    if (
                        family.name == "pointwise"
                        and len(region.operations) == 1
                        and region.operations[0] in fused_readouts
                    ):
                        continue
                    raise ValueError("selected shared source regions overlap; select an unambiguous capability set")
                members.update(region.operations)
                if (
                    family.name == "dense"
                    and (readout := getattr(region.compilation, "output_operation", None)) is not None
                ):
                    fused_readouts.add(readout)
                index = len(group)
                while True:
                    symbol = f"__merlin_source_{family.name}_{index}"
                    borrowed = symbol + "_borrowed"
                    if symbol not in reserved and borrowed not in reserved:
                        break
                    index += 1
                reserved.update((symbol, borrowed))
                result_index = len(region.inputs)
                writer = FreshTensorWriterContract(
                    symbol,
                    result_index,
                    (result_index,),
                    borrowed,
                    family.selection.allocation_alignment,
                    borrowed_effects=family.selection.borrowed_effects,
                )
                binding = family.binding(region.compilation, writer)
                if family.name == "dense" and family.selection.logical_abi is not None:
                    names = (symbol + "_logical_kernel", symbol + "_abi_observe")
                    if any(name in reserved for name in names):
                        raise ValueError(
                            "shared logical ABI primitive or observer symbol collides with original source"
                        )
                    reserved.update(names)
                group.append(binding)
                replacements.append((region, binding))
                record = dict(
                    family=family.name,
                    writer=asdict(writer),
                    shape=list(region.shape),
                    source_passes=[asdict(p) for p in region.compilation.source_passes],
                    schedule=(
                        region.compilation.compiled.realization
                        if family.name == "dense"
                        else region.compilation.compiled.report
                    ),
                )
                if hasattr(region.compilation, "conformance"):
                    record["source_conformance"] = region.compilation.conformance
                records.append(record)
            groups.append(tuple(group))
            family.selection.selected_contracts.revalidate()
        actual_dense = {
            binding.compilation.operation_ordinal
            for family, group in zip(self.families, groups, strict=True)
            if family.name == "dense"
            for binding in group
        }
        incomplete = any(
            any(ordinal not in actual_dense for ordinal in route["selected_contraction_ordinals"])
            for report in normalizations
            for route in report["routes"]
        )
        if incomplete or (not replacements and any(row["rewritten"] for row in normalizations)):
            # Discard the entire private candidate and try the original legal
            # full route once. A single admitted arm cannot qualify its sibling.
            for report in normalizations:
                if report["rewritten"]:
                    report["discarded_candidate"] = dict(report)
                    report.update(
                        rewritten=0,
                        routes=[],
                        pipeline=[],
                        selected_sha256=report["source_sha256"],
                        declined=[
                            *report["declined"],
                            "no completed dense binding for every panel arm; original full source replanned",
                        ],
                    )
                    report.pop("checkpoint", None)
                    report["outcomes"] = [
                        {
                            **outcome,
                            "selected_route": "full",
                            "selected_packing_payload_bytes": outcome["full_packing_payload_bytes"],
                            "budget_status": (
                                "unknown"
                                if outcome["full_packing_payload_bytes"] is None
                                else "unbounded"
                                if outcome["requested_packing_budget_bytes"] is None
                                else "met"
                                if outcome["full_packing_payload_bytes"] <= outcome["requested_packing_budget_bytes"]
                                else "exceeded"
                            ),
                            "reason": "no_completed_dense_provider",
                        }
                        for outcome in report.get("outcomes", ())
                    ]
            raise _ConvolutionFallback(normalizations)
        # Generate ABI artifacts only after coverage commits to this source.
        # A discarded panel arm must not occupy a final full-route bridge path.
        updated = {}
        by_symbol = {record["writer"]["symbol"]: record for record in records}
        for family, group in zip(self.families, groups, strict=True):
            if family.name != "dense" or family.selection.logical_abi is None:
                continue
            from .logical_borrowed_abi import prepare_binding

            for binding in group:
                result = prepare_binding(
                    binding,
                    family.selection.logical_abi,
                    output / "logical_abi",
                    borrow_contiguous=family.selection.borrow_contiguous,
                )
                updated[result.writer.symbol] = result
                self.pins = (*self.pins, (result.logical_abi_source.resolve(), sha256_file(result.logical_abi_source)))
                by_symbol[result.writer.symbol]["logical_abi"] = dict(
                    capability=asdict(family.selection.logical_abi),
                    borrow_contiguous=family.selection.borrow_contiguous,
                    primitive_symbol=result.logical_kernel_symbol,
                    observer_symbol=result.logical_observer_symbol,
                    source=dict(
                        path=str(result.logical_abi_source.resolve()), sha256=sha256_file(result.logical_abi_source)
                    ),
                    scope="original conditional format/effects; actual descriptor guards and packing fallback",
                )
        groups = [tuple(updated.get(binding.writer.symbol, binding) for binding in group) for group in groups]
        replacements = [(region, updated.get(binding.writer.symbol, binding)) for region, binding in replacements]
        self._bind_packing_plans(original, normalizations, groups)
        self.bindings = tuple(groups)
        self.revalidate()

        values = {}
        replacement = original.clone(value_mapper=values) if replacements else original
        clones = dict(zip(original.walk(), replacement.walk(), strict=True))
        # Reverse source order also updates dependent replacement call operands
        # when an earlier producer publishes its fresh result.
        for region, binding in sorted(replacements, key=lambda item: positions[item[0].operations[-1]], reverse=True):
            target, writer = clones[region.operations[-1]], binding.writer
            inputs = [values[value] for value in region.inputs]
            empty = tensor.EmptyOp([], target.results[0].type)
            call = func.CallOp(writer.symbol, [*inputs, empty.tensor], [target.results[0].type])
            Rewriter.insert_op([empty, call], InsertPoint.before(target))
            target.results[0].replace_all_uses_with(call.results[0])
            for operation in reversed(region.operations):
                Rewriter.erase_op(clones[operation])
            declaration = func.FuncOp.external(
                writer.symbol,
                [value.type for value in (*inputs, empty.tensor)],
                [call.results[0].type],
            )
            declaration.attributes["llvm.emit_c_interface"] = builtin.UnitAttr()
            declaration.properties["arg_attrs"] = builtin.ArrayAttr(
                [
                    builtin.DictionaryAttr({"bufferization.access": builtin.StringAttr(access)})
                    for access in (*("read",) * len(inputs), "write")
                ]
            )
            replacement.body.block.add_op(declaration)
        rewrite_fresh_tensor_writers(replacement, [binding.writer for _, binding in replacements])
        replacement.verify()
        label = self._label()
        selected_path, receipt = output / (label + ".mlir"), output / (label + ".json")
        if replacements or any(row["rewritten"] for row in normalizations):
            selected_path.write_text(ir_text(replacement))
        else:
            selected_path = Path(source_path)
        selections = [
            dict(
                family=family.name,
                execution_domain=family.selection.execution_domain,
                premises={
                    name: getattr(family.selection, name)
                    for name in (
                        "assume_finite_inputs",
                        "private_ownership",
                        "disjoint_arguments",
                        "external_buffers_disjoint",
                    )
                    if hasattr(family.selection, name)
                },
            )
            for family in self.families
        ]
        write_pretty_json(
            receipt,
            {
                "schema": "merlin.source_model_lowering.v1"
                if len(self.families) != 1
                else {
                    "row": "merlin.source_row_lowering.v1",
                    "pointwise": "merlin.source_pointwise_lowering.v1",
                    "dense": "merlin.source_dense_lowering.v1",
                }[self.families[0].name],
                "source_sha256": sha256_file(source_path),
                "selected_sha256": sha256_file(selected_path),
                "execution_domain": selections[0]["execution_domain"] if len(selections) == 1 else "per_selection",
                "premises": selections[0]["premises"] if len(selections) == 1 else {},
                "selections": selections,
                "bindings": records,
                **({"offload_placement": placements} if placements else {}),
                **({"packing_plans": self.packing_plans} if normalizations else {}),
                "declined": refusals,
                **({"source_normalization": normalizations} if normalizations else {}),
                "scope": "checked source selection; domain/premises require independent implementation qualification",
            },
        )
        self.revalidate()
        paths = (Path(source_path), selected_path, receipt)
        self.selected_source = selected_path.resolve()
        self.pins = (*self.pins, *((path.resolve(), sha256_file(path)) for path in paths))
        self.revalidate()
        for family, group in zip(self.families, self.bindings, strict=True):
            if group:
                implementation = output / (family.name + "_implementation")
                implementation.mkdir()
                family.selection.prepare(group, implementation)
                self.revalidate()
        return selected_path

    def _place_dense(self, module, selection, regions):
        """Narrow already-admitted dense work using current whole-program intent.

        A plan cannot grant computation, ABI, storage or execution support. The
        normal dense discovery has already checked those independently; all
        retained CPU operations preserve their original typed source bodies.
        """
        from .source_dense import offload_context, validate_offload_selection

        cost = validate_offload_selection(selection)
        plan = selection.offload_plan
        context = offload_context(selection)
        selected = frozenset(plan.select_for(module, context=context))
        operations = tuple(module.walk())
        covered = frozenset(operations[index] for index in plan.covered_ordinals)
        if cost is not None and not cost["choose_candidate"]:
            return regions, {
                "candidate_plan": plan.record(),
                "comparison": cost,
                "actual_route": "original",
                "selected_dense_ordinals": [region.compilation.operation_ordinal for region in regions],
                "retained_cpu_dense_ordinals": [],
                "scope": "original route retained; rejected candidate needs no implementation admission",
            }
        admitted = {op for region in regions for op in region.operations}
        if not selected <= admitted:
            raise ValueError("whole-program placement requests dense work without a legal current provider")
        kept, cpu = [], []
        for region in regions:
            members = frozenset(region.operations)
            if not members.intersection(covered):
                kept.append(region)
                continue
            if not members <= covered or members.intersection(selected) and not members <= selected:
                raise ValueError("whole-program placement splits an admitted dense numerical region")
            if members <= selected:
                kept.append(region)
            else:
                cpu.append(region.compilation.operation_ordinal)
        return tuple(kept), {
            "candidate_plan": plan.record(),
            "comparison": cost,
            "actual_route": "candidate",
            "selected_dense_ordinals": [region.compilation.operation_ordinal for region in kept],
            "retained_cpu_dense_ordinals": cpu,
            "scope": "source operation selection under original dense admission; final execution remains separate",
        }

    def _bind_packing_plans(self, original, normalizations, groups):
        """Join checked source arms to exact original writer declarations."""
        from xdsl.dialects import func

        from merlin.transforms.certificates import ir_digest

        plans = []
        source_identity = None
        for index, (family, group) in enumerate(zip(self.families, groups, strict=True)):
            if family.name != "dense":
                continue
            bindings = {binding.compilation.operation_ordinal: binding for binding in group}
            for report in normalizations:
                for route in report["routes"]:
                    if source_identity is None:
                        source_identity = ir_digest(original)
                    if route["selected_source_sha256"] != source_identity:
                        raise ValueError("convolution packing coverage refers to stale source")
                    uses = []
                    for ordinal in route["selected_contraction_ordinals"]:
                        binding = bindings[ordinal]
                        operation = binding.compilation.operation
                        entry = operation.parent_op()
                        while entry is not None and type(entry) is not func.FuncOp:
                            entry = entry.parent_op()
                        if entry is None or entry.sym_name.data != route["entry_function"]:
                            raise ValueError("convolution packing arm changed its source entry")
                        # This is the checked dense computation's RHS position,
                        # independent of the flattened native descriptor width.
                        if len(operation.inputs) != 2:
                            raise ValueError("packing coverage requires the checked two-input dense form")
                        uses.append((binding.writer.wrapper_symbol or binding.writer.symbol, 1))
                    if not uses or len(uses) != len(set(uses)):
                        raise ValueError("convolution packing arms lack distinct original bindings")
                    plans.append(
                        (
                            index,
                            route["entry_function"],
                            tuple(uses),
                            route["realized_distinct_panel_payload_bytes"],
                            route["source_identity"],
                        )
                    )
        self.packing_plans = self._packing_stamp = tuple(plans)

    def _check_packing(self, context):
        if not self.packing_plans:
            return
        from merlin.common.jsonio import write_pretty_json
        from merlin.verify.host_packing_storage import (
            PackingStorageRequest,
            PackingStorageUse,
            inspect_packing_storage_batch,
        )

        roster = source_borrowed_calls(self.build, context.prepared_path)
        if roster is None:
            raise ValueError("bounded convolution storage requires original synchronous borrowed effects")
        artifacts = {
            role: {"path": str(path.resolve()), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
            for role, path in (("llvm_ir", context.model_llvm_path), ("object", context.model_object_path))
        }
        groups = {}
        for index, (family_index, entry, uses, payload, identity) in enumerate(self.packing_plans):
            selected = self.families[family_index].selection
            if selected.packing_allocation_effects is None:
                raise ValueError("bounded convolution storage lacks original qualified allocation semantics")
            request = PackingStorageRequest(
                tuple(PackingStorageUse(symbol, argument) for symbol, argument in uses),
                selected.max_convolution_packing_bytes,
            )
            groups.setdefault((entry, selected.packing_allocation_effects), []).append(
                (index, request, payload, identity)
            )
        retained = {}
        for group_index, ((entry, effects), islands) in enumerate(groups.items()):
            batch = inspect_packing_storage_batch(
                artifacts,
                borrowed_calls=roster.calls,
                requests=tuple(island[1] for island in islands),
                allocation_effects=effects,
                entry=entry,
                diagnostics=context.workdir / f"packing_storage_batch_{group_index}",
            )
            for path, digest in batch["pins"].items():
                if path in retained and retained[path] != digest:
                    raise ValueError("final native packing observations disagree about retained bytes")
                retained[path] = digest
            for request_index, ((index, _, payload, identity), result) in enumerate(
                zip(islands, batch["results"], strict=True)
            ):
                # Native coalescing may share backing between statically distinct
                # source panels. Exact descriptor/use coverage was checked above;
                # sharing can reduce their conservative source payload sum.
                if result["logical_payload_bytes"] > payload:
                    raise ValueError("final native packing coverage exceeds the checked source panels")
                report = {
                    **batch["combined"],
                    **result,
                    **{key: batch[key] for key in ("observation", "pins", "report")},
                    "batch_request_index": request_index,
                }
                directory = context.workdir / f"packing_storage_{index}"
                directory.mkdir()
                link = directory / "source_packing.json"
                write_pretty_json(
                    link,
                    {
                        "schema": "merlin.source_packing_storage.v1",
                        "source_identity": identity,
                        "source_payload_bytes": payload,
                        "native": report,
                        "scope": (
                            "per-island activation payload and requested backing; excludes padding/output/ABI storage"
                        ),
                    },
                )
                retained[str(link.resolve())] = sha256_file(link)
        if any(sha256_file(Path(path)) != digest for path, digest in retained.items()):
            raise ValueError("final native packing observation changed before adoption")
        self.observe_native(tuple(Path(path) for path in retained), True)

    def build(self, context):
        from merlin.compile.scheduling.library_sources import LIBRARY_MODULES
        from merlin.runtime.host_provider import HostProviderBuild, HostProviderObject

        self.revalidate()
        active = [(f, g) for f, g in zip(self.families, self.bindings or (), strict=True) if g]
        if not active:
            raise ValueError("shared source provider requires a completed nonempty preparation")
        self._check_packing(context)
        self.revalidate()
        originals = tuple(
            sha256_file(p)
            for p in (
                context.prepared_path,
                context.model_llvm_path,
                context.model_object_path,
            )
        )
        results, witnesses = [], []
        for family, group in active:
            child = context
            if len(active) > 1:
                directory = context.workdir / family.name
                directory.mkdir()
                child = replace(context, workdir=directory)
            result = family.selection.build(group, child)
            self.revalidate()
            if (
                type(result) is not HostProviderBuild
                or not result.objects
                or any(type(artifact) is not HostProviderObject for artifact in result.objects)
                or (result.prepared_sha256, result.model_llvm_sha256, result.model_object_sha256) != originals
                or any(
                    not is_sha256(value)
                    for value in (
                        result.numerical_witness_sha256,
                        result.effect_witness_sha256,
                    )
                )
                or tuple(
                    sha256_file(p)
                    for p in (
                        context.prepared_path,
                        context.model_llvm_path,
                        context.model_object_path,
                    )
                )
                != originals
            ):
                raise ValueError("shared source child provider lost current model identity or pinned object/witnesses")
            if family.name == "dense" and family.selection.logical_abi is not None:
                from .logical_borrowed_abi import compile_binding

                bridges = tuple(compile_binding(binding, child) for binding in group)
                self.revalidate()
                result = replace(result, objects=(*result.objects, *bridges))
            results.append(result)
            witnesses.append(
                dict(
                    family=family.name,
                    numerical=result.numerical_witness_sha256,
                    effects=result.effect_witness_sha256,
                )
            )
        paths = (
            *(
                path
                for f in self.families
                for path in (
                    f.selection.selected_contracts.facts_path,
                    f.selection.selected_contracts.software_path,
                )
            ),
            *(Path(importlib.import_module(name).__file__) for name in LIBRARY_MODULES),
        )
        pins = (*self.pins, *((path.resolve(), sha256_file(path)) for path in paths))
        numerical, effects = results[0].numerical_witness_sha256, results[0].effect_witness_sha256
        if len(results) > 1:
            from merlin.common.jsonio import write_pretty_json

            manifest = context.workdir / "source_lowerings.json"
            write_pretty_json(
                manifest,
                dict(
                    schema="merlin.source_provider_bundle.v1",
                    providers=witnesses,
                    scope="individual witness custody only; no aggregate execution or numerical qualification",
                ),
            )
            numerical = effects = sha256_file(manifest)
            pins = (*pins, (manifest.resolve(), numerical))
        return HostProviderBuild(
            *originals,
            tuple(
                replace(artifact, compilation_pins=(*artifact.compilation_pins, *pins))
                for result in results
                for artifact in result.objects
            ),
            numerical,
            effects,
        )


def bind_source_lowerings(selections, row_selection, transform, builder, *, device=None, device_admission=None):
    """Select only known source families, through the ordinary existing hooks."""
    if type(selections) is not tuple:
        raise ValueError("shared_source_lowerings requires an immutable tuple of typed selections")
    if row_selection is not None:
        selections = (row_selection, *selections)
    if not selections:
        return transform, builder
    if transform is not None or builder is not None:
        raise ValueError("shared source lowering cannot replace an explicitly selected transform or provider")
    from .source_dense import SourceDenseBinding, SourceDenseLowering, discover_dense
    from .source_pointwise import SourcePointwiseBinding, SourcePointwiseLowering, discover_pointwise
    from .source_rows import SourceRowBinding, SourceRowLowering, discover_rows

    families = []
    for selection in selections:
        if type(selection) is SourceRowLowering:
            families.append(
                _Family(
                    "row",
                    selection,
                    discover_rows,
                    SourceRowBinding,
                    Path(importlib.import_module(SourceRowLowering.__module__).__file__),
                )
            )
        elif type(selection) is SourcePointwiseLowering:
            families.append(
                _Family(
                    "pointwise",
                    selection,
                    discover_pointwise,
                    SourcePointwiseBinding,
                    Path(importlib.import_module(SourcePointwiseLowering.__module__).__file__),
                )
            )
        elif type(selection) is SourceDenseLowering:
            families.append(
                _Family(
                    "dense",
                    selection,
                    discover_dense,
                    SourceDenseBinding,
                    Path(importlib.import_module(SourceDenseLowering.__module__).__file__),
                )
            )
        else:
            raise ValueError("shared source lowering requires an explicit row, pointwise or dense selection")
    if len({f.name for f in families}) != len(families):
        raise ValueError("duplicate shared source family selections are ambiguous")
    if len(families) != 1 and any(getattr(f.selection, "offload_plan", None) is not None for f in families):
        raise ValueError(
            "whole-program dense placement requires one source family; other families need explicit planning"
        )
    invocation = _SourceModelBuild(tuple(sorted(families, key=lambda f: f.name)), device, device_admission)
    return invocation.prepare, invocation.build


def source_provider_builder(builder):
    owner = getattr(builder, "__self__", None)
    if type(owner) is _SourceModelBuild:
        owner.revalidate()
        if owner.bindings is None:
            raise ValueError("shared source provider used before ordinary preparation")
        if not any(owner.bindings):
            return None
    return builder


def source_borrowed_calls(builder, prepared_path):
    """Original selected writer effects; no grants from a declaration or target."""
    owner = getattr(builder, "__self__", None)
    from .reviewed_source_routing import SelectedSourceRouting
    from .shared_source_routing import PointwiseSourceRouting, SharedSourceRouting

    if type(owner) in (SharedSourceRouting, PointwiseSourceRouting):
        if getattr(builder, "__func__", None) is not SharedSourceRouting.build:
            return None
        return owner.borrowed_calls(prepared_path)
    if type(owner) is SelectedSourceRouting:
        if getattr(builder, "__func__", None) is not SelectedSourceRouting.build:
            return None
        owner._revalidate()
        selected = owner._owner
        if selected is None:
            return None
        calls = selected.borrowed_calls(prepared_path)
        owner._revalidate()
        if calls is not None:
            from dataclasses import replace

            return replace(calls, pins=(*calls.pins, *owner._sources), validate_original=owner._revalidate)
        return None
    if type(owner) is not _SourceModelBuild or getattr(builder, "__func__", None) is not _SourceModelBuild.build:
        return None
    owner.revalidate()
    if owner.bindings is None:
        raise ValueError("shared source effects used before ordinary preparation")
    writers = tuple(binding.writer for group in owner.bindings for binding in group)
    if not any(writer.borrowed_effects is not None for writer in writers):
        return None
    if Path(prepared_path).resolve() != owner.selected_source:
        raise ValueError("borrowed source effects require the original selected prepared source")
    from .borrowed_buffer_calls import from_writers

    pins = (
        *owner.pins,
        *(
            (path.resolve(), sha256_file(path))
            for family in owner.families
            for path in (
                family.selection.selected_contracts.facts_path,
                family.selection.selected_contracts.software_path,
            )
        ),
    )
    return from_writers(owner.selected_source, writers, pins, owner.revalidate, owner.observe_native)


def source_device_work(builder):
    owner = getattr(builder, "__self__", None)
    if type(owner) is not _SourceModelBuild:
        return False
    owner.revalidate()
    if owner.bindings is None:
        raise ValueError("shared source provider used before ordinary preparation")
    return any(
        group and family.selection.execution_domain == "device"
        for family, group in zip(owner.families, owner.bindings or (), strict=True)
    )
