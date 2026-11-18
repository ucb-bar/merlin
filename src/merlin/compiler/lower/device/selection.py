"""Original-owned economics for a source-bound ordinary package route.

Only an original selector's accepted contractions may be retained or removed.
Neither package hashes, public eligibility nor cost records qualify numerical,
ABI, completion or timing semantics. The original caller/measurement verifier
owns those premises and host build inputs outside DeviceRouting. Selected
packages must already have a tool and remain byte-immutable through the build.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import math
import os
import stat
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from merlin.analysis.optimization.evaluation.offload_profitability import (
    OffloadCostDecision,
    compare_complete_offload_cost,
)
from merlin.common.digest import canonical_json, canonical_sha256, sha256_file
from merlin.compiler.lower.device.device_build import DeviceRouting
from merlin.compiler.lower.device.plan import OffloadPlanSelection, _read, bind_source
from merlin.compiler.schedule.library_sources import shared_library_sources
from merlin.compiler.transforms.certificates import ir_digest
from merlin.compiler.transforms.pipeline_state import PipelineContext
from merlin.target.contract.build.execution_service import _callback_selection, _same_callback

# ---- device_offload_selection ------------------------------------------------------------------


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
            "merlin.compiler.lower.device.build",
            "merlin.compiler.lower.device.region",
            "merlin.compiler.lower.device.native",
            "merlin.compiler.lower.device.offload",
            "merlin.compiler.lower.device.preparation",
            "merlin.compiler.lower.device.target",
            "merlin.compiler.lower.device.shim",
            "merlin.compiler.lower.device.plan",
            "merlin.target.system",
            "merlin.compiler.planning.source_coverage",
            "merlin.development.targetgen.packaging.loading",
            "merlin.development.targetgen.packaging.records",
            "merlin.target.capabilities.manifest",
            "merlin.target.rtl.facts",
            "merlin.target.contract.build.execution_service",
            "merlin.target.contract.build.object_admission",
            "merlin.analysis.optimization.evaluation.offload_profitability",
            "merlin.analysis.optimization.evaluation.component_cost",
            "merlin.analysis.perf.contract.terms",
        )
    )


def _package(routing, max_files, max_bytes):
    from merlin.development.targetgen.packaging.loading import load_package
    from merlin.development.targetgen.packaging.records import payload_inventory

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
    from merlin.compiler.lower.device.shim import kernel_abi_for
    from merlin.target.registry.experiment_descriptor import load_capability_manifest
    from merlin.target.rtl.facts import body_if_present
    from merlin.target.system import device_contraction_ranks, device_dtype_triples, link_for

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
    from merlin.compiler.schedule.loops.configuration_ir import ConfigurationIRSelection
    from merlin.target.contract.build.object_admission import LinkedElfAdmissionService

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
        if name == "offload_selection" or name == "target_schedule" and value is None:
            continue
        if name == "target_schedule":
            raise ValueError(
                "device offload selection: target scheduling requires separate economic cost qualification"
            )
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
    from merlin.target.system import offloadable_contractions

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


# ---- device_ir_scalability ---------------------------------------------------------------------
# Cheap static device-build cost observations and explicitly selected budgets.
#
# Counts describe canonical textual LLVM, not instruction effects, hardware limits
# or numerical correctness. The original build caller selects and freezes budgets;
# emitted programs and environment variables never select or relax them. Counted
# device loops should keep these static costs bounded across larger logical shapes.

SCHEMA = "merlin.device_ir_scalability.v1"
_IDENTIFIER_BYTES = frozenset(b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.$-")
_LABEL_BYTES = _IDENTIFIER_BYTES | frozenset(b" \t:")
_BEFORE_WORD = _IDENTIFIER_BYTES | frozenset(b"%@")
_AFTER_WORD = _IDENTIFIER_BYTES | frozenset(b":")
_LIMITS = {
    "bytes": "max_bytes",
    "function_code_bytes": "max_code_bytes",
    "lines": "max_lines",
    "inline_asm_sites": "max_inline_asm_sites",
    "call_sites": "max_call_sites",
    "functions": "max_functions",
    "basic_blocks": "max_basic_blocks",
}


def _word_spans(data):
    """Observe exact unquoted keywords, retaining LLVM identifier boundaries."""
    for prefix in (b"call", b"invoke", b"asm"):
        pos = 0
        while (start := data.find(prefix, pos)) >= 0:
            word = b"callbr" if prefix == b"call" and data.startswith(b"callbr", start) else prefix
            end = start + len(word)
            if (start == 0 or data[start - 1] not in _BEFORE_WORD) and (
                end == len(data) or data[end] not in _AFTER_WORD
            ):
                yield start, end, word
            pos = end


def _define_prefix(prefix):
    return prefix.startswith(b"define") and len(prefix) > 6 and prefix[6] in b" \t\r\f\v\n"


def _next_special(chunk, positions, start, selected):
    """Advance each fixed-character search monotonically within this chunk."""
    result = len(chunk)
    for char in selected:
        found = positions[char]
        if 0 <= found < start:
            found = positions[char] = chunk.find(bytes((char,)), start)
        if found >= 0:
            result = min(result, found)
    return result


@dataclass(frozen=True)
class DeviceIRBudget:
    mode: str = "report_only"
    max_bytes: int | None = None
    max_code_bytes: int | None = None
    max_lines: int | None = None
    max_inline_asm_sites: int | None = None
    max_call_sites: int | None = None
    max_functions: int | None = None
    max_basic_blocks: int | None = None

    def __post_init__(self):
        if self.mode not in {"report_only", "enforce"}:
            raise ValueError("device IR budget mode must be report_only or enforce")
        values = [getattr(self, key) for key in _LIMITS.values()]
        if any(value is not None and (type(value) is not int or value <= 0) for value in values):
            raise ValueError("device IR limits must be explicit positive integers")
        if self.mode == "enforce" and all(value is None for value in values):
            raise ValueError("enforced device IR budget requires at least one explicit limit")

    def record(self):
        return asdict(self)

    def evaluate(self, analysis):
        if type(analysis) is not dict or analysis.get("schema") != SCHEMA:
            raise ValueError("device IR budget requires its actual source analysis")
        exceeded = [
            {"metric": key, "observed": analysis[key], "limit": getattr(self, field)}
            for key, field in _LIMITS.items()
            if getattr(self, field) is not None
            and analysis.get(key) is not None
            and analysis[key] > getattr(self, field)
        ]
        accepted = self.mode == "report_only" or (analysis["lexical_complete"] and not exceeded)
        reason = "device_ir_static_expansion_budget" if exceeded else "device_ir_lexical_observation_incomplete"
        return {
            "status": "accepted" if accepted else "refused",
            "policy": self.record(),
            "exceeded": exceeded,
            "reason": None if accepted else reason,
            "action": None if accepted else "Retain counted loops or reduce static expansion; raw LLVM is retained.",
            "scope": "operator-selected build resource budget; not hardware legality or correctness",
        }


class _Counts:
    def __init__(self):
        self.functions = self.basic_blocks = self.call_sites = self.inline_asm_sites = 0
        self.function_code_bytes = self.module_bytes = self.global_definition_bytes = 0
        self.in_function = self.pending = self.first_block = False
        self.reset()

    def reset(self):
        self.prefix = self.suffix = self.tail = b""
        self.prior = b" "
        self.calls = self.asms = self.colons = 0
        self.raw_bytes = 0
        self.label_chars = True

    def feed(self, data, final=False):
        self.prefix = (self.prefix + data)[:512]
        self.suffix = (self.suffix + data)[-512:]
        self.colons += data.count(b":")
        if self.label_chars:
            self.label_chars = all(char in _LABEL_BYTES for char in data)
        joined = self.prior + self.tail + data
        cut = len(joined) if final else max(1, len(joined) - 32)
        matches = list(_word_spans(joined))
        for start, end, _word in matches:
            if start >= 1 and start < cut < end:
                cut = start
        for start, end, word in matches:
            if 1 <= start < cut and end <= cut:
                if word == b"asm":
                    self.asms += 1
                else:
                    self.calls += 1
        self.prior, self.tail = joined[cut - 1 : cut], joined[cut:]
        if not final:
            return
        prefix, suffix = self.prefix.strip(), self.suffix.strip()
        if self.in_function or self.pending or _define_prefix(prefix):
            self.function_code_bytes += self.raw_bytes
        else:
            self.module_bytes += self.raw_bytes
            if prefix.startswith(b"@") and b"=" in prefix:
                self.global_definition_bytes += self.raw_bytes
        label = self.label_chars and self.colons == 1 and suffix.endswith(b":")
        if not label:
            self.call_sites += self.calls
            if self.calls:
                self.inline_asm_sites += self.asms
        if _define_prefix(prefix):
            self.functions += 1
            self.pending = True
        if self.pending and suffix.endswith(b"{"):
            self.in_function, self.pending, self.first_block = True, False, True
            self.basic_blocks += 1  # explicit or implicit function entry
        elif self.in_function and label:
            if not self.first_block:
                self.basic_blocks += 1
            self.first_block = False
        elif self.in_function and prefix and prefix != b"}":
            self.first_block = False
        if prefix == b"}":
            self.in_function = False
        self.reset()


def analyze_device_ir(path):
    """One bounded-buffer byte/lexical pass; quotes/comments do not contribute ops.

    The digest, physical lines and keyword counts are gathered from the same
    actual read. No LLVM parser, model data, subprocess or dependency scan is used.
    This analysis can be shared with an already deduplicated identical artifact;
    admission decisions are freshly bound to caller shape/source/tool/policy.
    """
    started = time.monotonic()
    path = Path(path)
    digest, counts = hashlib.sha256(), _Counts()
    total = newlines = quoted_bytes = 0
    state, escaped, last = "normal", False, None
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns", "st_mode")
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("device IR is not an ordinary file")
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
            total += len(chunk)
            newlines += chunk.count(b"\n")
            last = chunk[-1]
            pos = 0
            positions = {char: chunk.find(bytes((char,))) for char in (10, 34, 59, 92)}
            while pos < len(chunk):
                if state == "comment":
                    end = _next_special(chunk, positions, pos, (10,))
                    if end == len(chunk):
                        counts.raw_bytes += len(chunk) - pos
                        break
                    counts.raw_bytes += end + 1 - pos
                    counts.feed(b"", final=True)
                    state, pos = "normal", end + 1
                elif state == "string":
                    if escaped:
                        counts.raw_bytes += 1
                        quoted_bytes += 1
                        escaped = False
                        pos += 1
                        continue
                    end = _next_special(chunk, positions, pos, (10, 34, 92))
                    if end == len(chunk):
                        counts.raw_bytes += len(chunk) - pos
                        quoted_bytes += len(chunk) - pos
                        break
                    delta = end + 1 - pos
                    counts.raw_bytes += delta
                    quoted_bytes += delta
                    pos = end
                    char = chunk[pos]
                    pos += 1
                    if char == 34:
                        quoted_bytes -= 1  # contents only; delimiters are not literal payload
                    if char == 92:
                        escaped = True
                    elif char == 34:
                        state = "normal"
                    else:
                        counts.feed(b"", final=True)
                else:
                    end = _next_special(chunk, positions, pos, (10, 34, 59))
                    counts.feed(chunk[pos:end])
                    counts.raw_bytes += end - pos
                    if end == len(chunk):
                        break
                    char, pos = chunk[end], end + 1
                    counts.raw_bytes += 1
                    if char == 10:
                        counts.feed(b"", final=True)
                    elif char == 59:
                        state = "comment"
                    else:
                        counts.feed(b"quoted")  # preserve quoted identifiers for block-label observation
                        state = "string"
        if total and last != 10:
            counts.feed(b"", final=True)
        after = os.fstat(stream.fileno())
    current = path.stat()
    if any(
        getattr(before, key) != getattr(after, key) or getattr(after, key) != getattr(current, key) for key in fields
    ):
        raise ValueError("device IR changed during static build analysis")
    return {
        "schema": SCHEMA,
        "sha256": digest.hexdigest(),
        "bytes": total,
        "lines": newlines + int(bool(total and last != 10)),
        "functions": counts.functions,
        "basic_blocks": counts.basic_blocks,
        "call_sites": counts.call_sites,
        "inline_asm_sites": counts.inline_asm_sites,
        "lexical_complete": state != "string",
        "function_code_bytes": counts.function_code_bytes,
        "module_bytes": counts.module_bytes,
        "global_definition_bytes": counts.global_definition_bytes,
        "quoted_bytes": quoted_bytes,
        "elapsed_s": time.monotonic() - started,
        "scope": "canonical textual LLVM static cost observations",
    }


def inspect_device_ir(path, budget):
    """Refuse an explicit total-byte cap before any scan; never cache that as PASS."""
    if type(budget) is not DeviceIRBudget:
        raise ValueError("device build requires an explicitly selected DeviceIRBudget")
    size = Path(path).stat().st_size
    if budget.mode == "enforce" and budget.max_bytes is not None and size > budget.max_bytes:
        analysis = {
            "schema": SCHEMA,
            "sha256": None,
            "bytes": size,
            "lexical_complete": False,
            "elapsed_s": 0.0,
            "scope": "early total-file resource refusal; contents were not scanned",
        }
    else:
        analysis = analyze_device_ir(path)
    return {"analysis": analysis, "decision": budget.evaluate(analysis)}
