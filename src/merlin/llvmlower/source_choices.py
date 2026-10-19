"""Original-caller bounded choices through ordinary source/provider compilation.

Requests select a frozen choice, never an implementation or qualification. This
host-owned seam grants no author imports, target support, numerical admission or
execution authority. The existing ordinary compiler retains those obligations.
"""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from threading import Lock
from weakref import WeakKeyDictionary, ref

from merlin.common.digest import sha256_bytes, sha256_file
from merlin.runtime.host_provider import _symbols, read_host_function_abi, revalidate_host_provider
from merlin.targetgen.contract.execution_service import _callback_selection, _same_callback

from .source_dense import SourceDenseLowering, offload_context
from .source_model import _values, bind_source_lowerings

REQUEST_SCHEMA = "merlin.source_dense_choice_request.v1"
_RESULTS = WeakKeyDictionary()
_CONTROLS = frozenset(
    {
        "outer_order",
        "borrow_contiguous",
        "fuse_output_readout",
        "convolution_route",
        "max_convolution_packing_bytes",
        "max_convolution_candidates",
        "max_convolution_added_operations",
    }
)


def _plain(path, *, directory=False):
    path = Path(path)
    if (
        not path.is_absolute()
        or path.resolve() != path
        or any(part.is_symlink() for part in (path, *path.parents))
        or (not path.is_dir() if directory else not path.is_file())
    ):
        raise ValueError("source choices require original canonical unlinked paths")
    return path


def _file(path):
    path = _plain(path)
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


@dataclass(frozen=True)
class SourceDenseChoice:
    """One original finite control point; no callback or capability fields."""

    choice_id: str
    controls: tuple[tuple[str, object], ...] = ()
    offload_plan: object = None
    offload_comparison: object = None

    def __post_init__(self):
        if (
            type(self.choice_id) is not str
            or not 0 < len(self.choice_id) <= 128
            or not all(c.isascii() and (c.isalnum() or c in "_-") for c in self.choice_id)
            or type(self.controls) is not tuple
            or any(type(pair) is not tuple or len(pair) != 2 for pair in self.controls)
        ):
            raise ValueError("source choices require bounded IDs and explicit control tuples")
        names = tuple(pair[0] for pair in self.controls)
        if any(type(name) is not str or name not in _CONTROLS for name in names) or len(set(names)) != len(names):
            raise ValueError("source choice contains unknown, repeated or non-tuning controls")
        from merlin.perf.offload_profitability import OffloadCostDecision

        from .offload_plan import OffloadPlanSelection

        if (
            self.offload_plan is not None
            and type(self.offload_plan) is not OffloadPlanSelection
            or self.offload_comparison is not None
            and type(self.offload_comparison) is not OffloadCostDecision
            or self.offload_comparison is not None
            and self.offload_plan is None
        ):
            raise ValueError("source choices require original typed current plans and comparisons")
        for name, value in self.controls:
            if name == "outer_order":
                valid = type(value) is tuple and value in (("m", "n"), ("n", "m"))
            elif name in ("borrow_contiguous", "fuse_output_readout"):
                valid = type(value) is bool
            elif name == "convolution_route":
                valid = type(value) is str and value in ("auto", "full", "panel")
            else:
                valid = (
                    value is None
                    and name == "max_convolution_packing_bytes"
                    or (type(value) is int and 0 < value < 1 << 63)
                )
            if not valid:
                raise ValueError("source choice control is outside its typed bounded domain")


@dataclass(frozen=True, eq=False)
class SourceChoiceResult:
    """Original private artifacts; record() exposes only bounded identities."""

    lowered: object
    receipt_path: Path
    _owner: SourceDenseChoices = field(repr=False)
    _record: str = field(repr=False)
    _pins: tuple[dict, ...] = field(repr=False)
    _stats: str = field(repr=False)
    _fields: tuple = field(repr=False)

    def _check_binding(self):
        original = _RESULTS.get(self)
        if (
            original is None
            or original[0]() is not self._owner
            or original[1]() is not self.lowered
            or original[2:] != (self.receipt_path, self._record, json.dumps(self._pins), self._stats, self._fields)
        ):
            raise ValueError("source choice result lost its original compilation binding")

    def revalidate(self):
        from .compilation_recipe import verify_completed_recipe

        self._check_binding()
        self._owner.revalidate()
        if self._lowered_fields() != self._fields or json.dumps(self.lowered.stats) != self._stats:
            raise ValueError("source choice original compiler result changed")
        for identity in self._pins:
            if _file(identity["path"]) != identity:
                raise ValueError("source choice compiled artifact changed")
        native = json.loads(self._stats)["shared_source_lowering"]
        revalidate_host_provider(native["host_provider"])
        verify_completed_recipe(
            Path(native["compilation_recipe"]["path"]),
            executable=self.lowered.host_so,
            expected_recipe_sha256=native["compilation_recipe"]["sha256"],
        )
        if (
            self.receipt_path.read_text() != self._record
            or self._lowered_fields() != self._fields
            or json.dumps(self.lowered.stats) != self._stats
            or _file(self.lowered.ll_path) != self._pins[0]
            or _file(self.lowered.host_so) != self._pins[1]
        ):
            raise ValueError("source choice result or original receipt changed")
        self._check_binding()

    def _lowered_fields(self):
        return tuple(
            getattr(self.lowered, name) for name in ("workdir", "ll_path", "host_so", "riscv_obj", "audit_index")
        )

    def record(self):
        self.revalidate()
        return json.loads(self._record)


class SourceDenseChoices:
    """Freeze original semantics and finite controls before exposing requests.

    The caller owns private output placement and excludes concurrent writes.
    Callback state and reachable imports need independent original qualification;
    inspected callback identity/source pins establish only freshness. This first
    transport calls lower_model_file, preserving its existing device-domain
    unsupported diagnostic rather than inventing a backend continuation.
    """

    def __init__(
        self,
        source,
        *,
        entry,
        baseline,
        choices,
        output_root,
        max_choices=64,
        max_source_bytes=64 << 20,
        max_request_bytes=4096,
    ):
        from xdsl.dialects.func import FuncOp

        from merlin.common import mlir_query
        from merlin.runtime import host_provider

        from . import codegen, compilation_recipe, lower, native_source_model

        if any(type(v) is not int or v <= 0 for v in (max_choices, max_source_bytes, max_request_bytes)):
            raise ValueError("source choices require positive original resource bounds")
        if (
            type(baseline) is not SourceDenseLowering
            or type(choices) is not tuple
            or not 0 < len(choices) <= max_choices
        ):
            raise ValueError("source choices require an original dense selection and bounded nonempty roster")
        if any(type(choice) is not SourceDenseChoice for choice in choices):
            raise ValueError("source choices require exact typed original control points")
        if len({choice.choice_id for choice in choices}) != len(choices):
            raise ValueError("source choice IDs must be unique")
        self.source = _plain(source)
        if self.source.stat().st_size > max_source_bytes:
            raise ValueError("source choice input exceeds its original byte budget")
        raw = self.source.read_bytes()
        if len(raw) > max_source_bytes:
            raise ValueError("source choice input exceeds its original byte budget")
        if (
            type(entry) is not str
            or not 0 < len(entry) <= 256
            or not (entry[0].isascii() and (entry[0].isalpha() or entry[0] == "_"))
            or not all(c.isascii() and (c.isalnum() or c == "_") for c in entry)
        ):
            raise ValueError("source choices require a supported explicit C-interface entry name")
        module = mlir_query.parse(raw.decode("utf-8"))
        module.verify()
        entries = [op for op in module.body.block.ops if type(op) is FuncOp and op.sym_name.data == entry]
        if len(entries) != 1 or not entries[0].body.blocks:
            raise ValueError("source choice entry is absent, external or ambiguous")
        self.entry, self.baseline, self.choices = entry, baseline, choices
        self.max_choices = max_choices
        self.max_source_bytes, self.max_request_bytes = max_source_bytes, max_request_bytes
        root = Path(output_root)
        if not root.is_absolute() or root.resolve() != root or any(p.is_symlink() for p in (root, *root.parents)):
            raise ValueError("source choices require an original private output root")
        if self.source.is_relative_to(root):
            raise ValueError("source choice outputs must be disjoint from the original source")
        root.mkdir(parents=True, exist_ok=True)
        self.output_root = _plain(root, directory=True)
        self.source_sha256 = sha256_bytes(raw)
        self._source_identity = _file(self.source)
        if self._source_identity["sha256"] != self.source_sha256:
            raise ValueError("source choice original source changed during construction")
        original_values = _values((baseline, choices))
        original_context = offload_context(baseline)
        for choice in choices:
            choice.__post_init__()
            controls = dict(choice.controls)
            for name in ("max_convolution_candidates", "max_convolution_added_operations"):
                if name in controls and controls[name] > getattr(baseline, name):
                    raise ValueError("source choice exceeds its original search or growth budget")
            cap = baseline.max_convolution_packing_bytes
            if cap is not None and "max_convolution_packing_bytes" in controls:
                selected = controls["max_convolution_packing_bytes"]
                if selected is None or selected > cap:
                    raise ValueError("source choice cannot raise or remove its original packing cap")
        self._selections, self._guards, self._contexts = [], [], []
        callbacks = [baseline.prepare, baseline.build, baseline.adapter.encode, baseline.adapter.project]
        callbacks.extend(c.offload_comparison._check for c in choices if c.offload_comparison is not None)
        self._callbacks = tuple((callback, _callback_selection(callback)) for callback in callbacks)
        paths = [
            Path(__file__),
            Path(inspect.getsourcefile(_callback_selection)),
            *(
                Path(module.__file__)
                for module in (host_provider, codegen, compilation_recipe, lower, native_source_model)
            ),
        ]
        for callback, selected in self._callbacks:
            if selected is None:
                raise ValueError("source choice callbacks need original inspectable functions or bound methods")
            owner = inspect.getsourcefile(selected[1])
            if owner is None:
                raise ValueError("source choice callback lacks its original source owner")
            paths.append(Path(owner).resolve())
        self._pins = tuple(_file(path) for path in dict.fromkeys(paths))
        for choice in choices:
            choice.__post_init__()
            selection = replace(
                baseline,
                **dict(choice.controls),
                offload_plan=choice.offload_plan,
                offload_comparison=choice.offload_comparison,
            )
            context = offload_context(selection)
            if choice.offload_plan is not None:
                if choice.offload_plan.entry != entry:
                    raise ValueError("source choice offload plan names another entry")
                choice.offload_plan.select_for(module, context=context)
            transform, _ = bind_source_lowerings((selection,), None, None, None)
            self._selections.append(selection)
            self._guards.append(transform.__self__)
            self._contexts.append(context)
        self._selections, self._guards, self._contexts = map(tuple, (self._selections, self._guards, self._contexts))
        if _values((baseline, choices)) != original_values or offload_context(baseline) != original_context:
            raise ValueError("source choice original roster changed during construction")
        for callback, selected in self._callbacks:
            if not _same_callback(_callback_selection(callback), selected):
                raise ValueError("source choice original callback changed during construction")
        if any(_file(pin["path"]) != pin for pin in self._pins):
            raise ValueError("source choice original implementation changed during construction")
        self._results = ()
        self._attempts = 0
        self._lock = Lock()
        self._configuration = self._snapshot()
        self.revalidate()

    @property
    def results(self):
        return self._results

    def _snapshot(self):
        return _values(
            (
                self.source,
                self.entry,
                self.baseline,
                self.choices,
                self.output_root,
                self.source_sha256,
                self.max_choices,
                self.max_source_bytes,
                self.max_request_bytes,
                self._selections,
                self._contexts,
                self._callbacks,
                self._pins,
            )
        )

    def revalidate(self):
        if self._snapshot() != self._configuration or _file(self.source) != self._source_identity:
            raise ValueError("source choice original source or configuration changed")
        _plain(self.output_root, directory=True)
        for callback, selected in self._callbacks:
            if not _same_callback(_callback_selection(callback), selected):
                raise ValueError("source choice original callback changed")
        for pin in self._pins:
            if _file(pin["path"]) != pin:
                raise ValueError("source choice original implementation changed")
        for selection, guard, context in zip(self._selections, self._guards, self._contexts, strict=True):
            guard.revalidate()
            if offload_context(selection) != context:
                raise ValueError("source choice original library or context changed")
        if self._snapshot() != self._configuration or _file(self.source) != self._source_identity:
            raise ValueError("source choice original selection changed during verification")

    def advertisement(self):
        self.revalidate()
        return {
            "schema": "merlin.source_dense_choices.v1",
            "source_sha256": self.source_sha256,
            "entry": self.entry,
            "choices": [{"choice_id": c.choice_id, "controls": dict(c.controls)} for c in self.choices],
            "scope": "original bounded compiler choices; no target, execution or timing qualification",
        }

    def compile(self, request, output_dir=None):
        if not self._lock.acquire(blocking=False):
            raise ValueError("source choice owner cannot compile concurrently or recursively")
        try:
            return self._compile(request, output_dir)
        finally:
            self._lock.release()

    def _compile(self, request, output_dir):
        from . import lower, toolchain

        original_results = self._results
        if type(request) is not dict or set(request) != {"schema", "source_sha256", "choice_id"}:
            raise ValueError("source choice request has unknown or missing fields")
        if any(type(value) is not str for value in request.values()):
            raise ValueError("source choice request fields must be strings")
        if sum(len(value) for value in request.values()) > self.max_request_bytes:
            raise ValueError("source choice request exceeds its original byte budget")
        request_bytes = json.dumps(request)
        original_request, request = request, json.loads(request_bytes)
        self.revalidate()
        if len(request_bytes.encode()) > self.max_request_bytes:
            raise ValueError("source choice request exceeds its original byte budget")
        if request["schema"] != REQUEST_SCHEMA or request["source_sha256"] != self.source_sha256:
            raise ValueError("source choice request schema or original source identity differs")
        names = tuple(c.choice_id for c in self.choices)
        if type(request["choice_id"]) is not str or request["choice_id"] not in names:
            raise ValueError("source choice request names an unavailable original choice")
        selection = self._selections[names.index(request["choice_id"])]
        self._attempts += 1
        destination = self.output_root / f"compile-{self._attempts}" if output_dir is None else Path(output_dir)
        if (
            not destination.is_absolute()
            or destination.resolve() != destination
            or not destination.is_relative_to(self.output_root)
            or destination == self.output_root
            or destination.exists()
            or any(p.is_symlink() for p in (destination, *destination.parents))
        ):
            raise ValueError("source choice compilation needs a fresh private output descendant")
        destination.mkdir(parents=True)
        try:
            lowered = lower.lower_model_file(self.source, destination, shared_source_lowerings=(selection,))
        finally:
            self.revalidate()
            if json.dumps(original_request) != request_bytes or self._results is not original_results:
                raise ValueError("source choice original request or result history changed during compilation")
        if type(lowered) is not lower.LowerResult or lowered.host_so is None or lowered.workdir != destination:
            raise ValueError("source choice lost its ordinary completed native compilation")
        native = lowered.stats["shared_source_lowering"]
        for name in ("prepared_source", "preparation", "compilation_recipe"):
            identity = native[name]
            if {key: value for key, value in _file(identity["path"]).items() if key != "bytes"} != identity:
                raise ValueError("source choice native artifact identity differs from its original record")
        preparation = json.loads(Path(native["preparation"]["path"]).read_text())
        if preparation["selected_sha256"] != native["prepared_source"]["sha256"]:
            raise ValueError("source choice prepared-source custody differs")
        provider = native["host_provider"]
        image = {key: value for key, value in _file(lowered.host_so).items() if key != "bytes"}
        if provider is None:
            if preparation["selected_sha256"] != preparation["input_snapshot_sha256"]:
                raise ValueError("source choice changed its prepared source without a provider")
        elif (
            provider["executable"] != image
            or provider["source_model"][:2]
            != [
                native["prepared_source"],
                {key: value for key, value in _file(lowered.ll_path).items() if key != "bytes"},
            ]
            or provider["source_model"][2] != provider["ordered_link_objects"][0]
        ):
            raise ValueError("source choice provider belongs to another model or image")
        read_host_function_abi(lowered.ll_path, "_mlir_ciface_" + self.entry)
        defined, _ = _symbols(lowered.host_so, toolchain.llvm_nm())
        if defined.get("_mlir_ciface_" + self.entry) != ["T"]:
            raise ValueError("source choice final image lacks its exact strong entry function")
        paths = [lowered.ll_path, lowered.host_so]
        paths.extend(Path(native[name]["path"]) for name in ("prepared_source", "preparation", "compilation_recipe"))
        if native["host_provider"] is not None:
            paths.append(Path(native["host_provider"]["receipt_path"]))
        pins = tuple(_file(path) for path in dict.fromkeys(paths))
        record = (
            json.dumps(
                {
                    "schema": "merlin.source_dense_choice_result.v1",
                    "status": "compiled",
                    "source_sha256": self.source_sha256,
                    "entry": self.entry,
                    "choice_id": request["choice_id"],
                    "artifacts": [
                        {
                            "role": "llvm" if i == 0 else "image" if i == 1 else "compilation_evidence",
                            "sha256": pin["sha256"],
                            "bytes": pin["bytes"],
                        }
                        for i, pin in enumerate(pins)
                    ],
                    "scope": "ordinary host compilation and custody; no numerical, runtime or target qualification",
                },
                sort_keys=True,
            )
            + "\n"
        )
        receipt = destination / "source_choice.json"
        receipt.write_text(record)
        result = SourceChoiceResult(
            lowered,
            receipt,
            self,
            record,
            pins,
            json.dumps(lowered.stats),
            tuple(getattr(lowered, name) for name in ("workdir", "ll_path", "host_so", "riscv_obj", "audit_index")),
        )
        _RESULTS[result] = (ref(self), ref(lowered), receipt, record, json.dumps(pins), result._stats, result._fields)
        try:
            result.revalidate()
            if json.dumps(original_request) != request_bytes or self._results is not original_results:
                raise ValueError("source choice original request or result history changed during verification")
        except BaseException:
            _RESULTS.pop(result, None)
            receipt.unlink(missing_ok=True)
            raise
        self._results = (*original_results, result)
        return result
