"""Select checked source rows during ordinary model preparation.

The shared compiler discovers and rewrites supported source islands. Explicit
backend bindings implement their primitives and ranked borrowed ABI, including
the declared input, workspace and completion premises. This transport neither
qualifies those bindings nor interprets opaque backend output.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import TYPE_CHECKING

from merlin.common.digest import sha256_file

if TYPE_CHECKING:
    from merlin.compile.scheduling.row_softmax import RowSoftmaxAdapter, RowSoftmaxSourceCompilation
    from merlin.compile.scheduling.selected_contracts import SelectedSchedulingContracts
    from merlin.runtime.host_provider import HostProviderBuild, HostProviderContext

    from .fresh_tensor_writer import FreshTensorWriterContract


@dataclass(frozen=True)
class SourceRowLowering:
    """Selected capabilities and implementation bindings, never source selection.

    ``prepare`` receives all checked bindings and an invocation-private directory;
    ``build`` receives those bindings and the existing HostProviderContext. The
    latter returns a HostProviderBuild subject to the ordinary object/link gates.
    Finite inputs must be established at each island's actual input, for example
    by a qualified runtime guard. Finite model inputs alone are insufficient.
    """

    selected_contracts: SelectedSchedulingContracts
    adapter: RowSoftmaxAdapter
    numerical_policy: str
    execution_domain: str
    allocation_alignment: int
    prepare: Callable[[tuple[SourceRowBinding, ...], Path], None]
    build: Callable[[tuple[SourceRowBinding, ...], HostProviderContext], HostProviderBuild]
    assume_finite_inputs: bool = False
    private_ownership: bool = False
    disjoint_arguments: bool = False
    chunk_elements: int | None = None
    max_source_elements: int = 64 << 20
    max_islands: int = 64
    timeout_ms: int = 2000

    def __post_init__(self):
        from merlin.compile.scheduling.row_softmax import RowSoftmaxAdapter
        from merlin.compile.scheduling.selected_contracts import SelectedSchedulingContracts

        alignment = self.allocation_alignment
        if (
            type(self.selected_contracts) is not SelectedSchedulingContracts
            or type(self.adapter) is not RowSoftmaxAdapter
            or not isinstance(self.numerical_policy, str)
            or not self.numerical_policy
            or self.execution_domain not in ("host", "device")
            or type(alignment) is not int
            or not 0 < alignment < 1 << 63
            or alignment & (alignment - 1)
            or not callable(self.prepare)
            or not callable(self.build)
            or any(type(value) is not bool for value in (
                self.assume_finite_inputs, self.private_ownership, self.disjoint_arguments,
            ))
            or any(type(value) is not int or value <= 0 for value in (
                self.max_source_elements, self.max_islands, self.timeout_ms,
            ))
            or self.chunk_elements is not None and (
                type(self.chunk_elements) is not int or self.chunk_elements <= 0
            )
        ):
            raise ValueError("source row lowering requires typed capabilities, explicit premises and bounded controls")
        self.selected_contracts.revalidate()


@dataclass(frozen=True)
class SourceRowBinding:
    """One checked source computation and its fresh-result borrowed ABI."""

    compilation: RowSoftmaxSourceCompilation
    writer: FreshTensorWriterContract


class _SourceRowBuild:
    def __init__(self, selection, device, device_admission):
        from . import fresh_tensor_writer

        self.selection = selection
        self.selection_values = tuple(getattr(selection, f.name) for f in fields(selection))
        self.adapter_values = tuple(getattr(selection.adapter, f.name) for f in fields(selection.adapter))
        self.device, self.device_admission = device, device_admission
        self.bindings = None
        self.pins = tuple(
            (path.resolve(), sha256_file(path)) for path in (Path(__file__), Path(fresh_tensor_writer.__file__))
        )

    def revalidate(self):
        if (
            tuple(getattr(self.selection, f.name) for f in fields(self.selection)) != self.selection_values
            or tuple(getattr(self.selection.adapter, f.name) for f in fields(self.selection.adapter))
            != self.adapter_values
        ):
            raise ValueError("source row capabilities or implementation selection changed; replan")
        self.selection.selected_contracts.revalidate()
        for path, digest in self.pins:
            if not path.is_file() or path.is_symlink() or sha256_file(path) != digest:
                raise ValueError("source row preparation inputs or selected output changed; replan")
        for binding in self.bindings or ():
            binding.compilation.revalidate()
        if self.bindings:
            self.require_device_admission()

    def prepare(self, source_path, output):
        from dataclasses import asdict

        from xdsl.dialects import builtin, func, tensor
        from xdsl.rewriter import InsertPoint, Rewriter

        from merlin.common import mlir_query as mq
        from merlin.common.jsonio import write_pretty_json
        from merlin.compile.scheduling.row_softmax import (
            RowSoftmaxUnavailable,
            compile_source_rows,
            read_source_rows,
        )
        from merlin.compile.scheduling.temporary_storage import StorageOptimizationUnavailable
        from merlin.transforms.certificates import ir_text

        from . import fresh_tensor_writer

        if self.bindings is not None:
            raise ValueError("source row preparation belongs to one model invocation")
        self.revalidate()
        selection = self.selection
        original = mq.parse(source_path).clone()
        original.verify()
        bindings, declined = [], []
        reserved = {
            value.data for op in original.body.block.ops
            if isinstance(value := op.properties.get("sym_name", op.attributes.get("sym_name")), builtin.StringAttr)
        }
        operations = original.walk() if selection.selected_contracts.row_reduction is not None else ()
        if selection.selected_contracts.row_reduction is None:
            declined.append("selected software specification has no complete-row capability")
        members = set()
        for operation in operations:
            if operation.name != "linalg.generic":
                continue
            try:
                read_source_rows(
                    operation, numerical_policy=selection.numerical_policy,
                    assume_finite_inputs=selection.assume_finite_inputs,
                    max_source_elements=selection.max_source_elements,
                )
            except RowSoftmaxUnavailable:
                continue
            if len(bindings) + len(declined) >= selection.max_islands:
                raise ValueError("source row discovery exceeds the selected island budget")
            try:
                compiled = compile_source_rows(
                    original, operation, selected_contracts=selection.selected_contracts,
                    adapter=selection.adapter, numerical_policy=selection.numerical_policy,
                    assume_finite_inputs=selection.assume_finite_inputs,
                    private_ownership=selection.private_ownership, disjoint_arguments=selection.disjoint_arguments,
                    chunk_elements=selection.chunk_elements, timeout_ms=selection.timeout_ms,
                    max_source_elements=selection.max_source_elements,
                )
            except (RowSoftmaxUnavailable, StorageOptimizationUnavailable) as error:
                declined.append(str(error))
                continue
            if members.intersection(compiled.source.operations):
                raise ValueError("source row islands overlap; retain the original source route")
            members.update(compiled.source.operations)
            index = len(bindings)
            while True:
                symbol = f"__merlin_source_row_{index}"
                borrowed = symbol + "_borrowed"
                if symbol not in reserved and borrowed not in reserved:
                    break
                index += 1
            reserved.update((symbol, borrowed))
            writer = fresh_tensor_writer.FreshTensorWriterContract(
                symbol, 1, (1,), borrowed, selection.allocation_alignment,
            )
            bindings.append(SourceRowBinding(compiled, writer))

        self.bindings = tuple(bindings)
        values = {}
        replacement = original.clone(value_mapper=values) if bindings else original
        clones = dict(zip(original.walk(), replacement.walk(), strict=True))
        # Later consumers are rewritten first so replace_all_uses also updates
        # their input when an earlier selected island publishes its fresh result.
        for binding in reversed(bindings):
            source, writer = binding.compilation.source, binding.writer
            target = clones[source.operations[-1]]
            scores = values[source.source]
            empty = tensor.EmptyOp([], target.results[0].type)
            call = func.CallOp(writer.symbol, [scores, empty.tensor], [target.results[0].type])
            Rewriter.insert_op([empty, call], InsertPoint.before(target))
            target.results[0].replace_all_uses_with(call.results[0])
            for operation in reversed(source.operations):
                Rewriter.erase_op(clones[operation])
            declaration = func.FuncOp.external(
                writer.symbol, [scores.type, empty.tensor.type], [call.results[0].type],
            )
            declaration.attributes["llvm.emit_c_interface"] = builtin.UnitAttr()
            declaration.properties["arg_attrs"] = builtin.ArrayAttr([
                builtin.DictionaryAttr({"bufferization.access": builtin.StringAttr(access)})
                for access in ("read", "write")
            ])
            replacement.body.block.add_op(declaration)
        fresh_tensor_writer.rewrite_fresh_tensor_writers(replacement, [binding.writer for binding in bindings])
        replacement.verify()
        selected_path, receipt = output / "source_rows.mlir", output / "source_rows.json"
        if bindings:
            selected_path.write_text(ir_text(replacement))
        else:
            selected_path = Path(source_path)
        write_pretty_json(receipt, {
            "schema": "merlin.source_row_lowering.v1",
            "source_sha256": sha256_file(source_path), "selected_sha256": sha256_file(selected_path),
            "execution_domain": selection.execution_domain,
            "premises": dict(
                assume_finite_inputs=selection.assume_finite_inputs,
                private_ownership=selection.private_ownership,
                disjoint_arguments=selection.disjoint_arguments,
            ),
            "bindings": [dict(
                writer=asdict(binding.writer), shape=list(binding.compilation.source.shape),
                source_passes=[asdict(p) for p in binding.compilation.source_passes],
                schedule=binding.compilation.compiled.report,
            ) for binding in bindings],
            "declined": declined,
            "scope": "checked source selection; domain/premises require independent implementation qualification",
        })
        self.revalidate()
        paths = (Path(source_path), selected_path, receipt)
        self.pins = (*self.pins, *((path.resolve(), sha256_file(path)) for path in paths))
        self.revalidate()
        if bindings:
            self.require_device_admission()
            implementation = output / "implementation"
            implementation.mkdir()
            selection.prepare(self.bindings, implementation)
            self.revalidate()
        return selected_path

    def build(self, context):
        import importlib

        from merlin.compile.scheduling.library_sources import LIBRARY_MODULES
        from merlin.runtime.host_provider import HostProviderBuild

        self.revalidate()
        if not self.bindings:
            raise ValueError("source row provider requires a completed nonempty preparation")
        self.require_device_admission()
        result = self.selection.build(self.bindings, context)
        self.revalidate()
        if type(result) is not HostProviderBuild:
            raise ValueError("source row implementation must return the ordinary pinned provider build")
        paths = (
            self.selection.selected_contracts.facts_path, self.selection.selected_contracts.software_path,
            *(Path(importlib.import_module(name).__file__) for name in LIBRARY_MODULES),
        )
        pins = (*self.pins, *((path.resolve(), sha256_file(path)) for path in paths))
        return replace(result, objects=tuple(
            replace(artifact, compilation_pins=(*artifact.compilation_pins, *pins)) for artifact in result.objects
        ))

    def require_device_admission(self):
        if self.selection.execution_domain == "device":
            from .device_build import require_device_elf_admission

            require_device_elf_admission(self.device, self.device_admission)


def bind_source_row_lowering(selection, transform, builder, *, device=None, device_admission=None):
    """Connect selected capabilities to existing preparation and link hooks."""
    if selection is None:
        return transform, builder
    if type(selection) is not SourceRowLowering:
        raise ValueError("shared_row_lowering requires an explicit SourceRowLowering selection")
    if transform is not None or builder is not None:
        raise ValueError("shared row lowering cannot replace an explicitly selected transform or provider")
    invocation = _SourceRowBuild(selection, device, device_admission)
    return invocation.prepare, invocation.build


def source_row_provider_builder(builder):
    """A model with no supported islands needs no provider object or symbols."""
    owner = getattr(builder, "__self__", None)
    if type(owner) is _SourceRowBuild:
        owner.revalidate()
        if owner.bindings is None:
            raise ValueError("source row provider used before ordinary preparation")
        if not owner.bindings:
            return None
    return builder


def source_row_device_work(builder):
    """Declared device implementations retain the original final ELF gate."""
    owner = getattr(builder, "__self__", None)
    if type(owner) is not _SourceRowBuild:
        return False
    owner.revalidate()
    active = bool(owner.bindings) and owner.selection.execution_domain == "device"
    if active:
        owner.require_device_admission()
    return active
