"""Original-owned economics for a source-bound ordinary package route.

Only an original selector's accepted contractions may be retained or removed.
Neither package hashes, public eligibility nor cost records qualify numerical,
ABI, completion or timing semantics. The original caller/measurement verifier
owns those premises and host build inputs outside DeviceRouting. Selected
packages must already have a tool and remain byte-immutable through the build.
"""

from __future__ import annotations

import importlib
import inspect
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from merlin.common.digest import sha256_file
from merlin.common.jsonio import canonical_json, canonical_sha256
from merlin.compile.scheduling.library_sources import shared_library_sources
from merlin.perf.offload_profitability import OffloadCostDecision, compare_complete_offload_cost
from merlin.targetgen.contract.execution_service import _callback_selection, _same_callback
from merlin.transforms.certificates import ir_digest
from merlin.transforms.pipeline_state import PipelineContext

from .device_build import DeviceRouting
from .offload_plan import OffloadPlanSelection, _read, bind_source


def _require(value, message):
    if not value:
        raise ValueError("device offload selection: " + message)


def _plain(path, *, directory=False):
    path = Path(path)
    _require(
        path.is_absolute()
        and path.resolve() == path
        and not any(p.is_symlink() for p in (path, *path.parents))
        and (path.is_dir() if directory else path.is_file()),
        "canonical unlinked original files required",
    )
    return path


def _file(path):
    path = _plain(path)
    return dict(path=str(path), sha256=sha256_file(path), bytes=path.stat().st_size)


def _data(value):
    if type(value) in (str, int, bool, type(None)):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if isinstance(value, Path):
        return str(value)
    if type(value) in (tuple, list):
        return [_data(item) for item in value]
    if type(value) is dict and all(type(key) is str for key in value):
        return {key: _data(item) for key, item in value.items()}
    raise ValueError("device offload selection: unsupported original configuration value")


def implementation_paths():
    """Checker owners, not an author import grant or transitive dependency proof."""
    return tuple(
        Path(importlib.import_module(name).__file__).resolve()
        for name in (
            __name__,
            "merlin.llvmlower.device_build",
            "merlin.llvmlower.device_offload",
            "merlin.llvmlower.device_preparation",
            "merlin.llvmlower.device_shim",
            "merlin.llvmlower.offload_plan",
            "merlin.system.offload",
            "merlin.system.derive",
            "merlin.targetgen.package_runtime",
            "merlin.targetgen.package_records",
            "merlin.targetgen.capability_manifest",
            "merlin.targetgen.rtl.facts",
            "merlin.targetgen.contract.execution_service",
            "merlin.targetgen.contract.elf_admission",
            "merlin.perf.offload_profitability",
            "merlin.perf.component_cost",
            "merlin.perf.term",
        )
    )


def _package(routing, max_files, max_bytes):
    from merlin.targetgen.package_records import payload_inventory
    from merlin.targetgen.package_runtime import load_package

    root = _plain(routing.package_dir, directory=True)
    total, count = 0, 0
    for path in root.rglob("*"):
        count += 1
        _require(count <= max_files, "package member budget exceeded")
        _require(not path.is_symlink(), "package symlink is unsupported")
        if path.is_file():
            total += path.stat().st_size
            _require(total <= max_bytes, "package byte budget exceeded")
        else:
            _require(path.is_dir(), "nonregular package member")
    package = load_package(root)
    _require(package.target == routing.device, "package endpoint differs from original routing")
    return dict(root=str(root), payload=payload_inventory(root), tool=_file(package.tool))


def _metadata(routing):
    from merlin.system.derive import link_for
    from merlin.system.offload import device_contraction_ranks, device_dtype_triples
    from merlin.targetgen.rtl.facts import body_if_present
    from merlin.targetgen.target_experiment import load_capability_manifest

    from .device_shim import kernel_abi_for

    manifest = load_capability_manifest(routing.device)
    abi = kernel_abi_for(routing.device)
    _require(abi is not None and abi.version == 2, "only the declared logical host-buffer ABI is supported")
    return dict(
        manifest=asdict(manifest),
        facts=body_if_present(routing.device),
        abi=asdict(abi),
        link=asdict(link_for(routing.device, manifest.endpoint_kind)),
        dtypes=device_dtype_triples(routing.device),
        ranks=device_contraction_ranks(routing.device),
    )


def _route(routing):
    from merlin.compile.scheduling.configuration_ir import ConfigurationIRSelection
    from merlin.targetgen.contract.elf_admission import LinkedElfAdmissionService

    from .device_ir_scalability import DeviceIRBudget

    _require(type(routing) is DeviceRouting, "original typed DeviceRouting required")
    _require(
        routing.granularity == "contraction" and not routing.require_complete_groups,
        "only singleton contraction routing is supported",
    )
    _require(
        all(
            getattr(routing, name) is None
            for name in ("exact_selection", "catalog_manifest", "catalog_object", "catalog_builder")
        ),
        "catalog or exact operation selection cannot mix with economics",
    )
    _require(routing.select is not None, "an original selector is required")
    callbacks, record, callables = [], {}, []
    for field in fields(routing):
        name, value = field.name, getattr(routing, field.name)
        if name == "offload_selection":
            continue
        if name == "linked_elf_admission":
            _require(type(value) is LinkedElfAdmissionService, "original typed linked ELF admission is required")
            record[name] = value.verify(routing.device)
            service, depth = value, 0
            # verify already bounded and checked the complete prerequisite chain.
            while service is not None:
                for role in ("evaluator", "result_validator"):
                    callback = getattr(service, role)
                    if callback is not None:
                        callables.append((f"{name}.{depth}.{role}", callback))
                service, depth = service.prerequisite, depth + 1
        elif name == "target_configuration":
            _require(
                value is None or type(value) is ConfigurationIRSelection, "original typed target configuration required"
            )
            record[name] = None if value is None else value.record()
        elif name == "device_ir_budget":
            _require(value is None or type(value) is DeviceIRBudget, "original typed device IR budget required")
            record[name] = None if value is None else value.record()
        elif callable(value):
            callables.append((name, value))
        else:
            record[name] = _data(value)
    for role, callback in callables:
        selected = _callback_selection(callback)
        _require(selected is not None, "original callback must be an inspectable function or bound method")
        path = inspect.getsourcefile(selected[1])
        _require(path is not None, "original callback has no source owner")
        callbacks.append((role, selected))
        record[role] = dict(owner=_file(Path(path).resolve()), function=selected[1].__qualname__)
    return record, tuple(callbacks)


def _mask(module, routing):
    from merlin.system.offload import offloadable_contractions

    before = ir_digest(module)
    ordinals = {op: index for index, op in enumerate(module.walk())}
    selected = []
    for operation, shape in offloadable_contractions(module, routing.device):
        accepted = routing.select(shape)
        _require(type(accepted) is bool, "original selector must return an exact Boolean")
        if accepted:
            selected.append(ordinals[operation])
        _require(ir_digest(module) == before, "source changed during original selector")
    return tuple(selected)


def _context(record, mask):
    route = record["routing"]
    return PipelineContext(
        hardware=canonical_sha256(record["metadata"]),
        numerical_policy=canonical_sha256(route["numeric_policy"]),
        resources=canonical_sha256(dict(configuration=record["configuration"], budget=route["device_ir_budget"])),
        execution=canonical_sha256(dict(route=record, original_ordinals=mask)),
        library=canonical_sha256(record["library"]),
    )


def _candidate(owner, plan):
    _require(type(plan) is OffloadPlanSelection, "original current source plan required")
    _require(
        plan.entry == owner.source.entry and plan.covered_ordinals == owner.source.operation_ordinals,
        "candidate covers another original entry or graph",
    )
    # select_for checks the complete plan and current source itself. No
    # callback separates those checks from the candidate constraints below.
    plan.select_for(owner.source.module, context=owner.context)
    _require(
        set(plan.selected_ordinals).issubset(owner.original_ordinals),
        "candidate expands the original selector route",
    )
    _require(
        all(
            alt.placement == owner.routing.device
            and alt.implementation == owner.implementation_identity
            and not alt.required_capabilities
            and not alt.metadata
            for alt in plan.device_alternatives
        ),
        "candidate implementation differs from the original package route",
    )


@dataclass(frozen=True)
class DeviceOffloadSource:
    source: object
    routing: DeviceRouting
    configuration: tuple
    source_pins: tuple
    max_package_files: int
    max_package_bytes: int
    original_ordinals: tuple[int, ...]
    implementation_identity: str
    _record: str
    _callbacks: tuple

    @property
    def context(self):
        return self.source.stamp.context

    @property
    def stamp(self):
        return self.source.stamp

    @property
    def original_selection_sha256(self):
        return canonical_sha256(
            dict(
                schema="merlin.original_device_route.v1",
                stamp=asdict(self.stamp),
                original_ordinals=self.original_ordinals,
            )
        )

    def _current(self, routing):
        record, callbacks = _route(routing)
        _require(
            len(callbacks) == len(self._callbacks)
            and all(
                name == old_name and _same_callback(actual, original)
                for (name, actual), (old_name, original) in zip(callbacks, self._callbacks, strict=True)
            ),
            "original callback selection changed",
        )
        package = _package(routing, self.max_package_files, self.max_package_bytes)
        pins = []
        for path, expected in self.source_pins:
            identity = _file(path)
            _require(identity["sha256"] == expected, "original tool/runtime source pin changed")
            pins.append(identity)
        return dict(
            routing=record,
            package=package,
            metadata=_metadata(routing),
            configuration=_data(self.configuration),
            source_pins=pins,
            library=dict(shared=shared_library_sources(), owners=[_file(path) for path in implementation_paths()]),
            limits=[self.max_package_files, self.max_package_bytes],
        )

    def revalidate_context(self, *, routing=None):
        """Reopen original dependencies without selector, cost or policy callbacks."""
        self.source.revalidate()
        before = self._current(self.routing if routing is None else routing)
        _require(canonical_json(before).decode() == self._record, "original route context changed; replan")
        _require(
            self.implementation_identity == canonical_sha256(before["package"])
            and self.context == _context(before, self.original_ordinals),
            "original source/context/package binding changed; replan",
        )
        self.source.revalidate()

    def revalidate(self, *, routing=None):
        self.revalidate_context(routing=routing)
        _require(
            _mask(self.source.module, self.routing if routing is None else routing) == self.original_ordinals,
            "original selector outcomes changed; replan",
        )
        self.revalidate_context(routing=routing)

    def select(self, plan, comparison=None, *, explicit=False):
        self.revalidate()
        _candidate(self, plan)
        _require(
            type(explicit) is bool and not (explicit and comparison is not None),
            "explicit unpriced placement cannot mix with complete-cost selection",
        )
        _require(
            comparison is None or type(comparison) is OffloadCostDecision,
            "original live complete-cost decision required",
        )
        callback = None if comparison is None else _callback_selection(comparison._check)
        _require(comparison is None or callback is not None, "cost verifier implementation is uninspectable")
        result = DeviceOffloadSelection(
            self, plan, comparison, explicit, callback, (self, plan, comparison, explicit, callback)
        )
        result.revalidate()
        return result


def bind_device_offload_source(
    module,
    *,
    routing,
    entry="forward",
    configuration=(),
    source_pins=(),
    max_operations=4096,
    max_package_files=4096,
    max_package_bytes=64 << 20,
):
    """Freeze source/route identity; hashes do not qualify host closure or timing."""
    _require(
        type(configuration) is tuple
        and all(
            type(pair) is tuple
            and len(pair) == 2
            and type(pair[0]) is str
            and type(pair[1]) in (str, int, bool, type(None))
            for pair in configuration
        )
        and len(dict(configuration)) == len(configuration)
        and len(configuration) <= 256,
        "configuration requires bounded unique original scalar fields",
    )
    _require(
        type(source_pins) is tuple
        and len(source_pins) <= 4096
        and all(
            type(pair) is tuple and len(pair) == 2 and type(pair[0]) is str and type(pair[1]) is str
            for pair in source_pins
        )
        and len(dict(source_pins)) == len(source_pins),
        "original source pins require a bounded unique tuple",
    )
    _require(
        type(max_package_files) is int
        and 0 < max_package_files <= 100_000
        and type(max_package_bytes) is int
        and 0 < max_package_bytes <= 1 << 30,
        "positive bounded package budgets required",
    )
    _read(module, entry, max_operations)  # Bound and admit the source before any selector invocation.
    # Take all route/file snapshots before invoking the original selector.
    route, callbacks = _route(routing)
    temporary = DeviceOffloadSource(
        None, routing, configuration, source_pins, max_package_files, max_package_bytes, (), "", "", callbacks
    )
    before = temporary._current(routing)
    source_sha = ir_digest(module)
    mask = _mask(module, routing)
    _require(
        before == temporary._current(routing) and ir_digest(module) == source_sha,
        "original source or route changed during construction",
    )
    context = _context(before, mask)
    source = bind_source(module, entry=entry, context=context, max_operations=max_operations)
    _require(set(mask).issubset(source.operation_ordinals), "original selector extends outside the bound flat entry")
    result = DeviceOffloadSource(
        source,
        routing,
        configuration,
        source_pins,
        max_package_files,
        max_package_bytes,
        mask,
        canonical_sha256(before["package"]),
        canonical_json(before).decode(),
        callbacks,
    )
    result.revalidate()
    return result


@dataclass(frozen=True)
class DeviceOffloadSelection:
    owner: DeviceOffloadSource
    plan: OffloadPlanSelection
    comparison: OffloadCostDecision | None
    explicit: bool
    _comparison_callback: object
    _frozen: tuple

    def _binding(self):
        current = (self.owner, self.plan, self.comparison, self.explicit, self._comparison_callback)
        _require(
            len(self._frozen) == len(current) and all(a is b for a, b in zip(current, self._frozen, strict=True)),
            "original selection fields changed",
        )
        _require(
            self.comparison is None
            or type(self.comparison) is OffloadCostDecision
            and _same_callback(_callback_selection(self.comparison._check), self._comparison_callback),
            "original cost verifier changed",
        )

    def revalidate_context(self, *, routing=None):
        """Check source, route and plan freshness without calling original evaluators."""
        self._binding()
        self.owner.revalidate_context(routing=routing)
        _candidate(self.owner, self.plan)
        self._binding()

    def revalidate(self, *, routing=None):
        self._binding()
        self.owner.revalidate(routing=routing)
        _candidate(self.owner, self.plan)
        self._binding()
        _require(
            type(self.explicit) is bool and not (self.explicit and self.comparison is not None),
            "explicit placement and economics conflict",
        )
        record = None
        if self.comparison is not None:
            record = self.comparison.record()
            self._binding()
            _require(
                record["stamp"] == asdict(self.owner.stamp)
                and record["original_selection_sha256"] == self.owner.original_selection_sha256
                and record["candidate_selection_sha256"] == self.plan.plan.digest,
                "complete costs differ from the exact original/candidate route selections",
            )
            self.owner.revalidate(routing=routing)
            _candidate(self.owner, self.plan)
            self._binding()
        return record

    def select_for(self, module, *, routing=None):
        record = self.revalidate(routing=routing)
        candidate = self.plan.select_for(module, context=self.owner.context)
        current = self.owner.routing if routing is None else routing
        _require(_mask(module, current) == self.owner.original_ordinals, "current selector route changed")
        self._binding()
        chosen = self.explicit or record is not None and record["choose_candidate"]
        if chosen:
            result = candidate
        else:
            ordinals = set(self.owner.original_ordinals)
            result = tuple(op for index, op in enumerate(module.walk()) if index in ordinals)
        _require(self.revalidate(routing=routing) == record, "comparison changed during current source selection")
        self.plan.select_for(module, context=self.owner.context)
        return result

    def with_complete_cost(self, **query):
        """Ask the original complete-cost owner for this exact package comparison."""
        self.revalidate()
        _require(not self.explicit, "explicit unpriced selection must be removed before comparing costs")
        decision = compare_complete_offload_cost(
            **query,
            stamp=self.owner.stamp,
            original_selection_sha256=self.owner.original_selection_sha256,
            candidate_selection_sha256=self.plan.plan.digest,
        )
        self.revalidate()
        return self.owner.select(self.plan, decision)

    def record(self, *, routing=None):
        comparison = self.revalidate(routing=routing)
        chosen = self.explicit or comparison is not None and comparison["choose_candidate"]
        return dict(
            schema="merlin.device_offload_selection.v1",
            stamp=asdict(self.owner.stamp),
            entry=self.plan.entry,
            original_selection_sha256=self.owner.original_selection_sha256,
            candidate_selection_sha256=self.plan.plan.digest,
            original_ordinals=list(self.owner.original_ordinals),
            selected_ordinals=list(self.plan.selected_ordinals if chosen else self.owner.original_ordinals),
            route="explicit_unpriced" if self.explicit else "candidate" if chosen else "original",
            comparison=comparison,
            timing="UNKNOWN" if comparison is None else comparison["status"],
            scope="original-selector narrowing only; runtime, numerical and timing qualification independent",
        )
