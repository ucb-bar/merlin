"""Compile all selected regions through the public provider contract.

The default product is a checked partition and its region artifacts. Explicit
C control emission can consume admitted synchronous wrappers; an optional
caller-selected image linker adds startup and a deployment-bound ELF. None of
these production stages establishes numerical or device-execution correctness.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from merlin.compiler.artifact import Artifact, Compilation
from merlin.compiler.backend.selected import LoweringTicket
from merlin.compiler.backend.service import resolve
from merlin.compiler.build.control import RegionProduct, control_entry, emit_selected_c
from merlin.compiler.build.image import CImageLink, link_selected_image
from merlin.compiler.invocation import invoke, request_identity
from merlin.compiler.partition.regions import partition_execution
from merlin.compiler.planning.execution import SelectedExecution
from merlin.compiler.request import CompileOptions, CompileRequest, Source
from merlin.transforms.certificates import ir_text


@dataclass(frozen=True)
class RegionBackend:
    """Bind one proposal provider to its selected compiler and input entry ABI."""

    backend: Any
    entry: str
    options: CompileOptions = field(default_factory=CompileOptions)

    def __post_init__(self):
        if type(self.entry) is not str or not self.entry.strip():
            raise ValueError("region backend requires an explicit source entry name")
        if type(self.options) is not CompileOptions:
            raise ValueError("region backend requires typed compile options")


class PlannedCompiler:
    """An explicit backend object for ``merlin.api.compile`` with a selected plan.

    Provider keys must exactly match selected proposal provider identities.
    CPU source remains inline. Every region is validated before any backend
    compiles; a failure stops the build without replanning or claiming fallback.
    """

    name = "merlin.planned"

    def __init__(self, selection: SelectedExecution, backends: dict[str, RegionBackend], *,
                 image: CImageLink | None = None):
        if type(selection) is not SelectedExecution or type(backends) is not dict:
            raise ValueError("planned compilation requires a selection and explicit backend bindings")
        if any(type(key) is not str or type(value) is not RegionBackend for key, value in backends.items()):
            raise ValueError("planned backend bindings must name typed RegionBackend records")
        if image is not None and type(image) is not CImageLink:
            raise ValueError("planned image requires a typed CImageLink selection")
        self.selection = selection
        self.backends = dict(backends)
        self._selection = selection
        self._bindings = tuple(self.backends.items())
        self._options = self._option_identity()
        self.image = image
        self._image = image

    def _option_identity(self):
        return json.dumps({key: {"entry": value.entry, "options": asdict(value.options)}
                           for key, value in self.backends.items()}, sort_keys=True, allow_nan=False)

    def _revalidate(self):
        if self.image is not self._image:
            raise ValueError("planned image binding changed")
        if self.image is not None:
            self.image.revalidate()
        if self.selection is not self._selection or len(self.backends) != len(self._bindings):
            raise ValueError("planned compiler binding changed")
        if any(self.backends.get(key) is not value for key, value in self._bindings):
            raise ValueError("planned backend binding changed")
        if self._option_identity() != self._options:
            raise ValueError("planned backend options changed")
        self.selection.revalidate()

    def validate(self, request):
        from merlin.frontends.linalg_mlir import parse_mlir_text

        self._revalidate()
        if request.source.kind != "mlir":
            raise ValueError("planned compilation currently requires imported MLIR")
        control = control_entry(request.options, self.selection.source.entry,
                                self.image.deployment if self.image is not None else None)
        if control is not None and any(item.proposal.abi == "inlined" for item in self.selection.selected):
            raise ValueError("C control needs a compiled wrapper for every operation; inline CPU cannot be omitted")
        actual = parse_mlir_text(request.source.text)
        if not actual.is_structurally_equivalent(self.selection.source.module):
            raise ValueError("compilation source differs from the original selected program")
        expected = {item.proposal.provider for item in self.selection.selected if item.proposal.abi != "inlined"}
        if set(self.backends) != expected:
            raise ValueError("backend bindings must exactly cover selected non-inline providers")

    def compile(self, request):
        self.validate(request)
        control = control_entry(request.options, self.selection.source.entry,
                                self.image.deployment if self.image is not None else None)
        if request.output.exists():
            raise FileExistsError(f"compilation requires fresh output: {request.output}")
        partition = partition_execution(self.selection)
        services = {}
        for key, binding in self.backends.items():
            name, provider = resolve(binding.backend, group="merlin.backends", methods=("validate", "compile"))
            if name != key:
                raise ValueError("resolved backend identity differs from its selected proposal provider")
            services[key] = (name, provider)
        work = []
        for index, region in enumerate(partition.regions):
            binding = self.backends[region.provider]
            serialized = region.request_source(binding.entry)
            source = Source(serialized.text, origin=f"{request.source.origin or '<input>'}#{region.symbol}")
            child = CompileRequest(source, request.output / "regions" / str(index), binding.options)
            name, provider = services[region.provider]
            item = next(item for item in self.selection.selected if item.alternative.id == region.alternative)
            ticket = None
            if item.proposal.lowering is not None:
                if any(not callable(getattr(provider, method, None))
                       for method in ("validate_selected", "compile_selected")):
                    raise ValueError("selected offer requires explicit validate_selected/compile_selected support")
                ticket = LoweringTicket.bind(partition, region, binding.entry, child)
            before = request_identity(child)
            if ticket is None:
                provider.validate(child)
            else:
                provider.validate_selected(child, ticket)
                ticket.revalidate(child)
            if request_identity(child) != before:
                raise ValueError("region validation changed its original request")
            self._revalidate()
            partition.revalidate()
            work.append((region, binding.entry, child, name, provider, ticket))
        # Publish stage products only after the full roster validates.
        request.output.mkdir(parents=True)
        if work:
            (request.output / "regions").mkdir()
        products, records, region_products = [], [], []
        for region, entry, child, name, provider, ticket in work:
            self._revalidate()
            partition.revalidate()
            compiled = invoke(child, name, provider, selected=ticket)
            self._revalidate()
            partition.revalidate()
            # A region ELF/binary is not the whole-program executable. Retain its
            # original role in the nested manifest and qualify the aggregate role.
            products.extend(Artifact(a.path, a.kind, "region." + a.role, a.sha256) for a in compiled.artifacts)
            products.append(Artifact.from_file(compiled.manifest, kind="json", role="region.manifest"))
            region_products.append(RegionProduct(entry, compiled))
            records.append({"alternative": region.alternative, "provider": region.provider,
                            "partition_entry": region.symbol, "backend_entry": entry,
                            "source_ordinals": list(region.source_ordinals),
                            "inputs": list(region.inputs), "outputs": list(region.outputs),
                            "lowering": ticket.record() if ticket is not None else None,
                            "lowering_binding": "selected_offer" if ticket is not None else "legacy_source_and_options",
                            "compilation": compiled.manifest.relative_to(request.output).as_posix()})
        stage = "compiled_regions"
        remaining = ["control_lowering", "region_abi_adaptation", "link", "deployment_admission",
                     "backend_correctness_verification", "execution_evidence"]
        if control is not None:
            description, files = emit_selected_c(
                partition, tuple(region_products), name=control, directory=request.output)
            control_directory = request.output / "control"
            control_directory.mkdir()
            for member, text in files.items():
                path = control_directory / member
                path.write_text(text, encoding="utf-8")
                products.append(Artifact.from_file(path, kind="c-source" if path.suffix == ".c" else "c-header",
                                                   role="control"))
            description["link_inputs"] = [f"control/{control}.c", *description["link_inputs"]]
            path = control_directory / "control.json"
            path.write_text(json.dumps(description, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            products.append(Artifact.from_file(path, kind="json", role="control_plan"))
            stage = "c_dispatch"
            remaining = ["link", "deployment_admission", "backend_correctness_verification", "execution_evidence"]
            if self.image is not None:
                products.extend(link_selected_image(self.image, partition, tuple(region_products), description,
                                                    directory=request.output))
                self._revalidate()
                stage = "linked_image"
                remaining = ["platform_startup_qualification", "backend_correctness_verification", "execution_evidence"]
        source_path = request.output / "partition.mlir"
        source_path.write_text(ir_text(partition.module), encoding="utf-8")
        report_path = request.output / "plan.json"
        report = {"selection": self.selection.record(), "partition": asdict(partition.verification),
                  "source_observation": partition.source_observation(),
                  "emission": partition.emission.receipt(), "regions": records,
                  "stage": stage, "execution_verified": False, "remaining": remaining}
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        products.extend((Artifact.from_file(source_path, kind="mlir", role="partitioned_source"),
                         Artifact.from_file(report_path, kind="json", role="execution_plan")))
        partition.revalidate()
        self._revalidate()
        result = Compilation(request.output, self.name, request.source.sha256, tuple(products),
                             deployment=self.image.deployment if self.image is not None else None,
                             metadata={"stage": stage, "execution_verified": False})
        result.verify()
        return result
