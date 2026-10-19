"""One model preparation/link transaction for selected shared source families.

Discovery is fixed shared code. Target callbacks receive checked computations
for implementation, never an arbitrary source-selection hook. Families retain
their existing typed IR and pass pipelines; this module only joins their fresh
writer ABI, immutable inputs and ordinary provider/link lifecycle.
"""

from __future__ import annotations

import importlib
from dataclasses import asdict, dataclass, fields, is_dataclass, replace
from pathlib import Path

from merlin.common.digest import is_sha256, sha256_file


@dataclass(frozen=True)
class SourceRegion:
    """Internal SSA replacement, not another computation or target IR."""

    compilation: object
    operations: tuple
    inputs: tuple
    shape: tuple[int, ...]


@dataclass(frozen=True)
class _Family:
    name: str
    selection: object
    discover: object
    binding: object
    source_file: Path


def _values(value):
    if is_dataclass(value) and not isinstance(value, type):
        return type(value), tuple((f.name, _values(getattr(value, f.name))) for f in fields(value))
    if isinstance(value, (tuple, list)):
        return type(value), tuple(_values(v) for v in value)
    if isinstance(value, dict):
        return type(value), tuple((_values(k), _values(v)) for k, v in value.items())
    return value


class _SourceModelBuild:
    def __init__(self, families, device, device_admission):
        from . import fresh_tensor_writer

        self.families = families
        self.device, self.device_admission = device, device_admission
        self.bindings = None
        self.selection_values = tuple(_values(f.selection) for f in families)
        self.components = tuple(
            (value, _values(value)) for f in families
            for name in ("adapter", "encoding", "limits", "budgets")
            if (value := getattr(f.selection, name, None)) is not None
        )
        owners = (Path(__file__), Path(fresh_tensor_writer.__file__), *(f.source_file for f in families))
        self.pins = tuple((path.resolve(), sha256_file(path)) for path in dict.fromkeys(owners))

    def revalidate(self):
        if (
            tuple(_values(f.selection) for f in self.families) != self.selection_values
            or any(_values(value) != before for value, before in self.components)
        ):
            raise ValueError("shared source capabilities or implementation selection changed; replan")
        for family in self.families:
            family.selection.selected_contracts.revalidate()
        for path, digest in self.pins:
            if not path.is_file() or path.is_symlink() or sha256_file(path) != digest:
                raise ValueError("shared source preparation inputs or selected output changed; replan")
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

    def prepare(self, source_path, output):
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
        positions = {op: i for i, op in enumerate(original.walk())}
        reserved = {
            value.data for op in original.body.block.ops
            if isinstance(value := op.properties.get("sym_name", op.attributes.get("sym_name")), builtin.StringAttr)
        }
        groups, replacements, records, members, refusals = [], [], [], set(), []
        for family in self.families:
            regions, declined = family.discover(original, family.selection, self.revalidate)
            self.revalidate()
            refusals.extend(declined)
            group = []
            for region in regions:
                if members.intersection(region.operations):
                    raise ValueError("selected shared source regions overlap; select an unambiguous capability set")
                members.update(region.operations)
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
                    symbol, result_index, (result_index,), borrowed, family.selection.allocation_alignment,
                )
                binding = family.binding(region.compilation, writer)
                group.append(binding)
                replacements.append((region, binding))
                record = dict(
                    family=family.name, writer=asdict(writer), shape=list(region.shape),
                    source_passes=[asdict(p) for p in region.compilation.source_passes],
                    schedule=region.compilation.compiled.report,
                )
                if hasattr(region.compilation, "conformance"):
                    record["source_conformance"] = region.compilation.conformance
                records.append(record)
            groups.append(tuple(group))
            family.selection.selected_contracts.revalidate()
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
                writer.symbol, [value.type for value in (*inputs, empty.tensor)], [call.results[0].type],
            )
            declaration.attributes["llvm.emit_c_interface"] = builtin.UnitAttr()
            declaration.properties["arg_attrs"] = builtin.ArrayAttr([
                builtin.DictionaryAttr({"bufferization.access": builtin.StringAttr(access)})
                for access in (*("read",) * len(inputs), "write")
            ])
            replacement.body.block.add_op(declaration)
        rewrite_fresh_tensor_writers(replacement, [binding.writer for _, binding in replacements])
        replacement.verify()
        label = "source_schedules" if len(self.families) != 1 else {
            "row": "source_rows", "pointwise": "source_pointwise",
        }[self.families[0].name]
        selected_path, receipt = output / (label + ".mlir"), output / (label + ".json")
        if replacements:
            selected_path.write_text(ir_text(replacement))
        else:
            selected_path = Path(source_path)
        selections = [dict(
            family=family.name, execution_domain=family.selection.execution_domain,
            premises={name: getattr(family.selection, name) for name in (
                "assume_finite_inputs", "private_ownership", "disjoint_arguments", "external_buffers_disjoint",
            ) if hasattr(family.selection, name)},
        ) for family in self.families]
        write_pretty_json(receipt, {
            "schema": "merlin.source_model_lowering.v1" if len(self.families) != 1 else {
                "row": "merlin.source_row_lowering.v1", "pointwise": "merlin.source_pointwise_lowering.v1",
            }[self.families[0].name],
            "source_sha256": sha256_file(source_path), "selected_sha256": sha256_file(selected_path),
            "execution_domain": selections[0]["execution_domain"] if len(selections) == 1 else "per_selection",
            "premises": selections[0]["premises"] if len(selections) == 1 else {},
            "selections": selections, "bindings": records, "declined": refusals,
            "scope": "checked source selection; domain/premises require independent implementation qualification",
        })
        self.revalidate()
        paths = (Path(source_path), selected_path, receipt)
        self.pins = (*self.pins, *((path.resolve(), sha256_file(path)) for path in paths))
        self.revalidate()
        for family, group in zip(self.families, self.bindings, strict=True):
            if group:
                implementation = output / (family.name + "_implementation")
                implementation.mkdir()
                family.selection.prepare(group, implementation)
                self.revalidate()
        return selected_path

    def build(self, context):
        from merlin.compile.scheduling.library_sources import LIBRARY_MODULES
        from merlin.runtime.host_provider import HostProviderBuild, HostProviderObject

        self.revalidate()
        active = [(f, g) for f, g in zip(self.families, self.bindings or (), strict=True) if g]
        if not active:
            raise ValueError("shared source provider requires a completed nonempty preparation")
        originals = tuple(sha256_file(p) for p in (
            context.prepared_path, context.model_llvm_path, context.model_object_path,
        ))
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
                or any(not is_sha256(value) for value in (
                    result.numerical_witness_sha256, result.effect_witness_sha256,
                ))
                or tuple(sha256_file(p) for p in (
                    context.prepared_path, context.model_llvm_path, context.model_object_path,
                )) != originals
            ):
                raise ValueError("shared source child provider lost current model identity or pinned object/witnesses")
            results.append(result)
            witnesses.append(dict(
                family=family.name, numerical=result.numerical_witness_sha256, effects=result.effect_witness_sha256,
            ))
        paths = (
            *(path for f in self.families for path in (
                f.selection.selected_contracts.facts_path, f.selection.selected_contracts.software_path,
            )),
            *(Path(importlib.import_module(name).__file__) for name in LIBRARY_MODULES),
        )
        pins = (*self.pins, *((path.resolve(), sha256_file(path)) for path in paths))
        numerical, effects = results[0].numerical_witness_sha256, results[0].effect_witness_sha256
        if len(results) > 1:
            from merlin.common.jsonio import write_pretty_json

            manifest = context.workdir / "source_lowerings.json"
            write_pretty_json(manifest, dict(
                schema="merlin.source_provider_bundle.v1", providers=witnesses,
                scope="individual witness custody only; no aggregate execution or numerical qualification",
            ))
            numerical = effects = sha256_file(manifest)
            pins = (*pins, (manifest.resolve(), numerical))
        return HostProviderBuild(
            *originals,
            tuple(replace(artifact, compilation_pins=(*artifact.compilation_pins, *pins))
                  for result in results for artifact in result.objects),
            numerical, effects,
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
    from .source_pointwise import SourcePointwiseBinding, SourcePointwiseLowering, discover_pointwise
    from .source_rows import SourceRowBinding, SourceRowLowering, discover_rows

    families = []
    for selection in selections:
        if type(selection) is SourceRowLowering:
            families.append(_Family("row", selection, discover_rows, SourceRowBinding, Path(importlib.import_module(
                SourceRowLowering.__module__).__file__)))
        elif type(selection) is SourcePointwiseLowering:
            families.append(_Family("pointwise", selection, discover_pointwise, SourcePointwiseBinding,
                                    Path(importlib.import_module(SourcePointwiseLowering.__module__).__file__)))
        else:
            raise ValueError("shared source lowering requires an explicit SourceRowLowering or SourcePointwiseLowering")
    if len({f.name for f in families}) != len(families):
        raise ValueError("duplicate shared source family selections are ambiguous")
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


def source_device_work(builder):
    owner = getattr(builder, "__self__", None)
    if type(owner) is not _SourceModelBuild:
        return False
    owner.revalidate()
    if owner.bindings is None:
        raise ValueError("shared source provider used before ordinary preparation")
    return any(group and family.selection.execution_domain == "device"
               for family, group in zip(owner.families, owner.bindings or (), strict=True))
