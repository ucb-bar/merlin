"""Original host permissions and bounded author choices for ordinary compilation.

Selections are caller-owned data, not proof or runtime authority. This bridge
reuses existing graph checks, upstream lowering and arena allocation. It never
derives permissions from an identity, target name or candidate declaration.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

REQUEST_SCHEMA = "merlin.host_optimization_request.v1"
REPORT_SCHEMA = "merlin.host_optimization_selection.v1"
_CHOICES = ("propagate_layout", "fusion", "hoist_static_allocs", "sink_deallocs", "static_arena")
_STAGES = ("logical_host", "structured_host", "llvm")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _identity(value):
    if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("host optimization requires an explicit SHA256 identity")


def _members(values, allowed):
    if type(values) is not tuple or len(set(values)) != len(values) or any(v not in allowed for v in values):
        raise ValueError("host optimization requires an immutable admitted selection")


@dataclass(frozen=True)
class HostOptimizationContext:
    """Original reviewed semantic/runtime selection; never read from a candidate."""

    entry: str
    semantic_sha256: str
    permitted_stages: tuple[str, ...] = ()
    permitted_optimizations: tuple[str, ...] = ()
    index_bits: int | None = None
    data_layout: str | None = None
    independent_element_order_unobserved: bool = False
    floating_environment_unobserved_nontrapping: bool = False
    sequential_execution: bool = False
    successful_allocations: bool = False
    non_reentrant_entry: bool = False

    def __post_init__(self):
        _identity(self.semantic_sha256)
        if type(self.entry) is not str or not self.entry or len(self.entry.encode()) > 1024:
            raise ValueError("host optimization requires an original entry symbol")
        _members(self.permitted_stages, _STAGES)
        _members(self.permitted_optimizations, _CHOICES)
        for field in (
            "independent_element_order_unobserved",
            "floating_environment_unobserved_nontrapping",
            "sequential_execution",
            "successful_allocations",
            "non_reentrant_entry",
        ):
            if type(getattr(self, field)) is not bool:
                raise ValueError("host optimization permissions require explicit booleans")
        if self.index_bits is not None and (type(self.index_bits) is not int or self.index_bits <= 0):
            raise ValueError("host optimization requires an explicit positive index width")
        if self.data_layout is not None and (type(self.data_layout) is not str or not self.data_layout):
            raise ValueError("host optimization requires an explicit compiler data layout")


@dataclass(frozen=True)
class HostOptimizationRequest:
    """Safe policy choices only; source identity is the current actual emitted IR."""

    source_sha256: str
    propagate_layout: bool = False
    fusion_groups: tuple[tuple[int, ...], ...] = ()
    hoist_static_allocs: bool | None = None
    sink_deallocs: bool | None = None
    static_arena: bool = False
    minimum_transpose_bytes_saved: int = 1
    minimum_dispatches_saved: int = 1
    maximum_added_operations: int = 8192

    def __post_init__(self):
        _identity(self.source_sha256)
        for field in ("propagate_layout", "static_arena"):
            if type(getattr(self, field)) is not bool:
                raise ValueError("host optimization choices require exact booleans")
        for field in ("hoist_static_allocs", "sink_deallocs"):
            if getattr(self, field) is not None and type(getattr(self, field)) is not bool:
                raise ValueError("host optimization optional choices require booleans or null")
        for field in ("minimum_transpose_bytes_saved", "minimum_dispatches_saved", "maximum_added_operations"):
            value = getattr(self, field)
            if type(value) is not int or not 1 <= value <= 1048576:
                raise ValueError("host optimization structural limits require bounded positive integers")
        if type(self.fusion_groups) is not tuple or len(self.fusion_groups) > 64:
            raise ValueError("host fusion requires bounded immutable groups")
        for group in self.fusion_groups:
            if (
                type(group) is not tuple
                or not 2 <= len(group) <= 256
                or len(set(group)) != len(group)
                or any(type(v) is not int or not 0 <= v <= 1048576 for v in group)
            ):
                raise ValueError("host fusion requires bounded unique node ordinals")

    def record(self):
        result = {"schema": REQUEST_SCHEMA, **asdict(self)}
        result["fusion_groups"] = [list(group) for group in self.fusion_groups]
        return result


def request_from_record(record) -> HostOptimizationRequest:
    """Decode the exact public choice schema, refusing semantic/pipeline selectors."""
    keys = set(HostOptimizationRequest.__dataclass_fields__)
    if type(record) is not dict or set(record) != keys | {"schema"} or record["schema"] != REQUEST_SCHEMA:
        raise ValueError("host optimization request has unknown or missing choice fields")
    values = {key: record[key] for key in keys}
    groups = values["fusion_groups"]
    if type(groups) is not list or any(type(group) is not list for group in groups):
        raise ValueError("serialized host fusion choices require arrays")
    values["fusion_groups"] = tuple(tuple(group) for group in groups)
    return HostOptimizationRequest(**values)


@dataclass(frozen=True)
class HostOptimizationSelection:
    context: HostOptimizationContext
    request: HostOptimizationRequest

    def __post_init__(self):
        if type(self.context) is not HostOptimizationContext or type(self.request) is not HostOptimizationRequest:
            raise TypeError("host optimization requires exact original context and choice types")

    def record(self):
        return {"context": asdict(self.context), "request": self.request.record()}

    def identity(self):
        return _sha(json.dumps(self.record(), sort_keys=True, allow_nan=False))


def _inventory(source):
    from xdsl.context import Context
    from xdsl.dialects import affine, arith, builtin, cf, func, linalg, llvm, math, memref, scf, tensor
    from xdsl.parser import Parser

    context = Context(allow_unregistered=True)
    for dialect in (
        builtin.Builtin,
        llvm.LLVM,
        func.Func,
        arith.Arith,
        memref.MemRef,
        scf.Scf,
        tensor.Tensor,
        linalg.Linalg,
        math.Math,
        cf.Cf,
        affine.Affine,
    ):
        context.load_dialect(dialect)
    module = Parser(context, source).parse_module()
    operations = tuple(module.walk())
    names = tuple(op.op_name.data if op.name == "builtin.unregistered" else op.name for op in operations)
    complete_llvm = all(name == "builtin.module" or name.startswith("llvm.") for name in names)
    entries = tuple(
        op.sym_name.data for op in module.body.block.ops if type(op) in (func.FuncOp, llvm.FuncOp) and op.body.blocks
    )
    return module, names, entries, complete_llvm


def _scalar_pipeline(*, hoist: bool, sink: bool):
    """Choose from the existing upstream registry without process env mutation."""
    from .pipeline import _UPSTREAM_PASSES, _sink_dealloc_passes

    dealloc = [
        "ownership-based-buffer-deallocation",
        "buffer-deallocation-simplification",
        "bufferization-lower-deallocations",
    ]
    if sink:
        dealloc += _sink_dealloc_passes()
    result = []
    for item in _UPSTREAM_PASSES:
        if item == "__DEALLOC__":
            result.extend(dealloc)
        elif item == "func.func(buffer-hoisting,buffer-loop-hoisting)" and not hoist:
            continue
        elif item == "buffer-results-to-out-params{modify-public-functions hoist-static-allocs}" and not hoist:
            result.append("buffer-results-to-out-params{modify-public-functions}")
        else:
            result.append(item)
    return ",".join(result)


@dataclass(frozen=True)
class PreparedHostOptimization:
    selection: HostOptimizationSelection
    selected_mlir: str
    options_json: str
    report_json: str

    def lowering_options(self):
        return json.loads(self.options_json)

    def report(self):
        return json.loads(self.report_json)


def prepare_host_optimization(source, selection, *, finished_llvm_only=False) -> PreparedHostOptimization:
    """Check actual stage and original permissions before invoking existing passes."""
    if type(selection) is not HostOptimizationSelection:
        raise TypeError("host compilation requires the exact original optimization selection")
    context, request = selection.context, selection.request
    if request.source_sha256 != _sha(source):
        raise ValueError("host optimization request does not bind the actual compiler source")
    module, names, entries, complete_llvm = _inventory(source)
    if context.entry not in entries:
        raise ValueError("original host entry is not defined by the actual submitted module")
    selected = []
    for name, enabled in (
        ("propagate_layout", request.propagate_layout),
        ("fusion", bool(request.fusion_groups)),
        ("hoist_static_allocs", request.hoist_static_allocs is not None),
        ("sink_deallocs", request.sink_deallocs is not None),
        ("static_arena", request.static_arena),
    ):
        if enabled:
            if name not in context.permitted_optimizations:
                raise ValueError(f"original host context has not admitted optimization {name}")
            selected.append(name)
    graph = request.propagate_layout or bool(request.fusion_groups)
    structured = request.hoist_static_allocs is not None or request.sink_deallocs is not None
    if (complete_llvm or finished_llvm_only) and (graph or structured):
        raise ValueError("finished LLVM cannot select logical graph or structured-buffer optimizations")
    stage = "llvm" if complete_llvm else "logical_host" if graph else "structured_host"
    if selected and stage not in context.permitted_stages:
        raise ValueError(f"original host context has not admitted stage {stage}")
    if graph or structured:
        permitted = ("func.", "arith.", "math.", "memref.", "scf.", "tensor.", "linalg.", "cf.", "affine.")
        if any(name != "builtin.module" and not name.startswith(permitted) for name in names):
            raise ValueError("host optimization refuses unmodeled or device operations")
        module.verify()
        if context.index_bits is None or context.data_layout is None:
            raise ValueError("structured host lowering requires original index width and compiler data layout")
    if structured or request.static_arena:
        if not context.sequential_execution or not context.successful_allocations:
            raise ValueError(
                "host storage optimization requires original sequential and successful-allocation premises"
            )
    if request.static_arena and ("llvm" not in context.permitted_stages or not context.non_reentrant_entry):
        raise ValueError("static host arena requires original LLVM stage and non-reentrant entry admission")
    before = source
    graph_report = None
    if graph:
        from xdsl.dialects import builtin

        from .graph_optimizations import GraphOptimizationContract, optimize_host_graph

        if not context.independent_element_order_unobserved:
            raise ValueError("host graph choices require original independent-element order permission")
        floating = any(
            isinstance(value.type, builtin.TensorType)
            and not isinstance(value.type.get_element_type(), builtin.IntegerType)
            for operation in module.walk()
            for value in (*operation.operands, *operation.results)
        )
        if floating and not context.floating_environment_unobserved_nontrapping:
            raise ValueError("host graph choices require original nontrapping unobserved floating environment")
        source, graph_report = optimize_host_graph(
            source,
            GraphOptimizationContract(
                source_sha256=request.source_sha256,
                entry=context.entry,
                propagate_layout=request.propagate_layout,
                fusion_groups=request.fusion_groups,
                independent_element_order_unobserved=context.independent_element_order_unobserved,
                floating_environment_unobserved_nontrapping=context.floating_environment_unobserved_nontrapping,
                minimum_transpose_bytes_saved=request.minimum_transpose_bytes_saved,
                minimum_dispatches_saved=request.minimum_dispatches_saved,
                maximum_added_operations=request.maximum_added_operations,
            ),
        )
    options = {}
    if graph or structured:
        from .optional_passes import active

        if active():
            raise ValueError("selected host request cannot mix unbound ambient optional-pass choices")
        options = {
            "pipeline": _scalar_pipeline(
                hoist=request.hoist_static_allocs if request.hoist_static_allocs is not None else True,
                sink=request.sink_deallocs if request.sink_deallocs is not None else False,
            ),
            "index_bits": context.index_bits,
            "data_layout": context.data_layout,
        }
    report = {
        "schema": REPORT_SCHEMA,
        "selection": selection.record(),
        "selection_sha256": selection.identity(),
        "source_sha256": _sha(before),
        "selected_ir_sha256": _sha(source),
        "source_stage": stage,
        "selected_optimizations": selected,
        "graph": graph_report,
        "lowering_options": options,
        "status": "selected" if selected else "unchanged",
        "semantics": "conditional_independent_structural_checks",
        "formal_proof": "not_issued",
        "numerics": "not_established",
        "machine_semantics": "not_established",
        "runtime": "not_established",
        "timing": "UNKNOWN",
    }
    return PreparedHostOptimization(
        selection, source, json.dumps(options, sort_keys=True), json.dumps(report, sort_keys=True)
    )


def optimize_host_llvm(llvm_ir, prepared):
    """Reuse the existing checked arena binder at its actual post-LLVM stage."""
    if type(prepared) is not PreparedHostOptimization:
        raise TypeError("original prepared host optimization required")
    report = prepared.report()
    report["llvm_before_sha256"] = _sha(llvm_ir)
    if prepared.selection.request.static_arena:
        from .arena_bind import bind_arena

        llvm_ir, arena = bind_arena(llvm_ir, entry=prepared.selection.context.entry)
        report["arena"] = arena.to_dict()
    report["llvm_after_sha256"] = _sha(llvm_ir)
    report["status"] = (
        "applied"
        if (
            report["source_sha256"] != report["selected_ir_sha256"]
            or report["llvm_before_sha256"] != report["llvm_after_sha256"]
        )
        else "selected_lowering"
        if report["lowering_options"]
        else "unchanged"
    )
    return llvm_ir, report


def write_host_optimization_report(
    report, *, selection, source_path, selected_source_path, llvm_path, object_path, workdir
):
    """Reopen actual final bytes, including objects selected after stack repair."""
    if type(selection) is not HostOptimizationSelection or report["selection_sha256"] != selection.identity():
        raise ValueError("original host optimization selection changed during compilation")
    if _sha(Path(source_path).read_text(encoding="utf-8")) != selection.request.source_sha256:
        raise ValueError("host optimization original compiler source changed during compilation")
    if _sha(Path(selected_source_path).read_text(encoding="utf-8")) != report["selected_ir_sha256"]:
        raise ValueError("host optimization selected IR changed during compilation")
    report = dict(report)
    report["products"] = {
        role: {"path": str(Path(path).absolute()), "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
        for role, path in (
            ("source", source_path),
            ("selected_ir", selected_source_path),
            ("llvm", llvm_path),
            ("object", object_path),
        )
    }
    report["final_object_transform"] = "unchanged"
    if report["products"]["llvm"]["sha256"] != report["llvm_after_sha256"]:
        repair = json.loads((Path(workdir) / "kernel.stack_frame.json").read_text(encoding="utf-8"))
        if (
            repair.get("status") != "passed"
            or repair.get("repair", {}).get("transform") != "stack_arena_bind"
            or repair.get("llvm_ir_sha256") != report["products"]["llvm"]["sha256"]
            or repair.get("object_sha256") != report["products"]["object"]["sha256"]
            or repair["repair"].get("emitted_llvm_ir_sha256") != report["llvm_after_sha256"]
        ):
            raise ValueError("host optimization final LLVM differs without the original stack-repair product")
        report["final_object_transform"] = "later_stack_repair"
    path = Path(workdir) / "kernel.host_optimization.json"
    path.write_text(json.dumps(report, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return path
