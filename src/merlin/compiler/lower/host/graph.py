"""Explicit logical graph optimization on the ordinary host compilation route.

Contracts select semantics and structural cost proxies, never hardware schedules.
Typed ABI, scalar bodies and the fusion emitter's expanded driver correspondence
are checked independently of pass reports. This supplies no Lean or timing proof.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from math import prod
from pathlib import Path
from typing import TYPE_CHECKING

import yaml
from xdsl.dialects import builtin, func

from merlin.common.paths import module_source_path
from merlin.compiler.dialects.xdsl_context import text
from merlin.compiler.frontend.linalg_mlir import parse_mlir_text
from merlin.compiler.lower.host.host_optimization_request import (
    HostOptimizationContext,
    PreparedHostOptimization,
    context_from_software,
    optimize_host_llvm,
    prepare_host_optimization,
    write_host_optimization_report,
)

# ---- graph_optimizations -----------------------------------------------------------------------

if TYPE_CHECKING:
    pass


class GraphOptimizationUnavailable(ValueError):
    """A supported optional choice lacks permission or its required savings."""


@dataclass(frozen=True)
class GraphOptimizationContract:
    source_sha256: str
    entry: str
    propagate_layout: bool = False
    independent_element_order_unobserved: bool = False
    floating_environment_unobserved_nontrapping: bool = False
    minimum_transpose_bytes_saved: int = 1
    fusion_groups: tuple[tuple[int, ...], ...] = ()
    minimum_dispatches_saved: int = 1
    maximum_added_operations: int = 8192
    physical_layout: bool = False
    maximum_layout_assignments: int = 100_000
    maximum_layout_permutations: int = 16
    connected_layout: bool = False
    column_reduction_layout: bool = False

    def validate(self, source: str):
        if self.source_sha256 != hashlib.sha256(source.encode()).hexdigest():
            raise ValueError("graph optimization contract does not bind the actual input source")
        if not isinstance(self.entry, str) or not self.entry:
            raise ValueError("graph optimization requires its explicit logical entry")
        for field in (
            "propagate_layout",
            "independent_element_order_unobserved",
            "floating_environment_unobserved_nontrapping",
            "physical_layout",
            "connected_layout",
            "column_reduction_layout",
        ):
            if type(getattr(self, field)) is not bool:
                raise ValueError("graph permissions must be explicit booleans")
        if self.connected_layout and not self.physical_layout:
            raise ValueError("connected layouts require the existing physical layout planner")
        for field in (
            "minimum_transpose_bytes_saved",
            "minimum_dispatches_saved",
            "maximum_added_operations",
            "maximum_layout_assignments",
            "maximum_layout_permutations",
        ):
            if type(getattr(self, field)) is not int or getattr(self, field) <= 0:
                raise ValueError("graph cost constraints must be explicit positive integers")
        if type(self.fusion_groups) is not tuple or any(type(group) is not tuple for group in self.fusion_groups):
            raise ValueError("fusion groups require immutable node selections")


def _functions(module):
    rows = [op for op in module.body.block.ops if type(op) is func.FuncOp]
    result = {op.sym_name.data: op for op in rows}
    if len(result) != len(rows):
        raise ValueError("graph optimization refuses duplicate function symbols")
    return result


def _signature(module, entry):
    functions = _functions(module)
    if entry not in functions or len(functions[entry].body.blocks) != 1:
        raise ValueError("graph optimization requires its defined single-block entry")
    return functions[entry].function_type


def _layout_bodies(module):
    from merlin.compiler.lower.host.propagation import scalar_bodies

    return scalar_bodies(module)


def _transpose_bytes(module):
    from merlin.compiler.lower.host.propagation import _permutation, pointwise_kind

    total = 0
    for op in module.walk():
        if pointwise_kind(op) not in {"transpose", "generic"}:
            continue
        permutation = _permutation(op)
        if permutation is None or permutation == tuple(range(len(permutation))) or not op.results[0].uses:
            continue
        dtype = op.results[0].type
        if not isinstance(dtype, builtin.TensorType) or any(d <= 0 for d in dtype.get_shape()):
            raise ValueError("layout materialization cost requires positive static tensor extents")
        element = dtype.get_element_type()
        bits = element.width.data if isinstance(element, builtin.IntegerType) else getattr(element, "bitwidth", None)
        if type(bits) is not int or bits <= 0 or bits % 8:
            raise ValueError("layout materialization cost has unknown byte-addressable storage")
        total += prod(dtype.get_shape()) * (bits // 8)
    return total


def _check_layout_output(original, modified, bodies, *, connected=False):
    before = _functions(original)
    after = _functions(modified)
    if set(before) != set(after) or any(before[name].function_type != after[name].function_type for name in before):
        raise ValueError("layout propagation changed complete function interfaces")
    if any(
        before[name].attributes != after[name].attributes or before[name].properties != after[name].properties
        for name in before
    ):
        raise ValueError("layout propagation changed function declarations or contracts")
    if any(body not in bodies for body in _layout_bodies(modified)):
        raise ValueError("layout propagation introduced unaccounted scalar arithmetic or evaluation order")
    modified.verify()
    from merlin.compiler.lower.host.propagation import layout_expression

    before, before_work = layout_expression(original, include_work=True, connected=connected)
    after, after_work = layout_expression(modified, include_work=True, connected=connected)
    if before != after:
        raise ValueError("layout propagation changed complete logical output coordinates or scalar expressions")
    if any(count > before_work[key] for key, count in after_work.items()):
        raise ValueError("layout propagation increases pointwise arithmetic in a control scope")
    return {
        "before_scalar_evaluations": sum(before_work.values()),
        "after_scalar_evaluations": sum(after_work.values()),
        "scope": "static parallel tensor domains by typed scalar operation and lexical control scope; "
        "not dynamic invocation counts, machine instructions or cycles",
    }


def _layout_expression(module, *, connected=False):
    from merlin.compiler.lower.host.propagation import layout_expression

    return layout_expression(module, connected=connected)


def _check_fusion_growth(outlined, plan, entry, original_count, maximum_added):
    """Count bounded call expansion before the emitter clones fused bodies."""
    functions = _functions(outlined.module)
    driver = functions[entry]
    # The selected plan is built from this entry; its implementation symbols
    # cannot add new functions before emission. Do not infer callees from names.
    budget = original_count + maximum_added
    total = sum(1 for _ in outlined.module.walk()) - sum(1 for _ in driver.walk()) + 2
    source_ops = [op for op in driver.body.block.ops if op.name != "func.return"]

    def expansion(operation, stack=()):
        if len(stack) >= 256:
            raise ValueError("fusion exceeds its bounded call nesting budget")
        if operation.name == "func.call":
            symbol = operation.callee.string_value()
            if symbol in stack or symbol not in functions:
                raise ValueError("fusion refuses recursive or undefined call expansion")
            function = functions[symbol]
            if not function.body.blocks and symbol in outlined.external_symbols:
                return 1
            if len(function.body.blocks) != 1:
                raise ValueError("fusion growth checking requires single-block functions")
            count = 0
            for child in function.body.block.ops:
                if child.name == "func.return":
                    break
                count += expansion(child, (*stack, symbol))
                if count > budget:
                    raise ValueError("fusion exceeds its explicit compilation-space cost contract")
            return count
        if any(child.name == "func.call" for region in operation.regions for child in region.walk()):
            raise ValueError("fusion refuses opaque nested-call expansion")
        return sum(1 for _ in operation.walk())

    expanded_count = 0
    for operation in source_ops:
        expanded_count += expansion(operation)
        if expanded_count > budget:
            raise ValueError("fusion exceeds its explicit compilation-space cost contract")
    for item in plan.selected:
        if len(item.nodes) == 1:
            total += sum(1 for _ in source_ops[item.nodes[0]].walk())
        else:
            total += 3  # Fused definition, return, and one driver call.
            for index in item.nodes:
                total += expansion(source_ops[index])
                if total > budget:
                    raise ValueError("fusion exceeds its explicit compilation-space cost contract")
        if total > budget:
            raise ValueError("fusion exceeds its explicit compilation-space cost contract")


def _implementation_sources():
    from merlin.analysis.programs import physical_layout
    from merlin.compiler.ir import xdsl_io as dense_float_ir
    from merlin.compiler.partition import memory_plan as arena_plan
    from merlin.compiler.transforms import pipeline_state
    from merlin.compiler.transforms.dense import ir as dense_ir
    from merlin.compiler.transforms.dense import row_batching as dense_reduction_layout

    # Trusted host observations, not an SDK/export grant or imported dependency lock.
    return {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in (
            ("graph_optimizations", Path(__file__)),
            *(
                (
                    name,
                    module_source_path(
                        {
                            "arena_bind.py": "merlin.compiler.lower.host.arena",
                            "constant_panel_layout.py": "merlin.compiler.lower.host.layout",
                            "graph_optimizations.py": "merlin.compiler.lower.host.graph",
                            "host_graph_policy.py": "merlin.compiler.lower.host.graph",
                            "host_layout_planning.py": "merlin.compiler.lower.host.layout",
                            "host_layout_realization.py": "merlin.compiler.lower.host.layout",
                            "host_optimization_request.py": "merlin.compiler.lower.host.host_optimization_request",
                            "host_storage_policy.py": "merlin.compiler.lower.host.host_storage_policy",
                            "layout_conformance.py": "merlin.compiler.lower.host.propagation",
                            "layout_propagation.py": "merlin.compiler.lower.host.propagation",
                            "quant_round_check.py": "merlin.compiler.lower.quant.readout",
                        }[name + ".py"]
                    ),
                )
                for name in (
                    "layout_propagation",
                    "layout_conformance",
                    "quant_round_check",
                    "host_graph_policy",
                    "constant_panel_layout",
                    "host_storage_policy",
                    "arena_bind",
                    "host_optimization_request",
                    "host_layout_planning",
                    "host_layout_realization",
                )
            ),
            ("physical_layout", Path(physical_layout.__file__)),
            ("pipeline_state", Path(pipeline_state.__file__)),
            ("dense_ir", Path(dense_ir.__file__)),
            ("dense_reduction_layout", Path(dense_reduction_layout.__file__)),
            ("dense_float_ir", Path(dense_float_ir.__file__)),
            ("arena_plan", Path(arena_plan.__file__)),
        )
    }


def recheck_host_graph_artifacts(directory: Path, contract: GraphOptimizationContract | HostGraphPolicy) -> dict:
    """Reopen actual source/report bytes; this grants no machine-code qualification."""
    paths = {
        role: Path(directory) / name
        for role, name in (
            ("input", "host-graph-input.mlir"),
            ("selected", "host-graph-optimized.mlir"),
            ("report", "host-graph-optimization.json"),
        )
    }
    if any(path.is_symlink() or not path.is_file() for path in paths.values()):
        raise ValueError("host graph optimization artifacts are absent or indirect")

    if type(contract) is HostGraphPolicy:
        return contract.recheck(directory, paths)
    if type(contract) is not GraphOptimizationContract:
        raise ValueError("host graph artifacts require the original typed contract")
    contract.validate(paths["input"].read_text())
    report = json.loads(paths["report"].read_text())
    if (
        report.get("schema") != "merlin.host_graph_optimization.v1"
        or report.get("source_sha256") != contract.source_sha256
        or report.get("contract") != json.loads(json.dumps(asdict(contract)))
        or report.get("transformed_sha256") != hashlib.sha256(paths["selected"].read_bytes()).hexdigest()
        or report.get("implementation_sources_sha256") != _implementation_sources()
    ):
        raise ValueError("host graph contract, selected source or implementation bytes changed")
    return {
        role: {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for role, path in paths.items()
    }


def optimize_host_graph(source: str, contract: GraphOptimizationContract | HostGraphPolicy) -> tuple[str, dict]:
    """Select the existing layout/fusion mechanisms before host preprocessing.

    The default source-check domain is static pointwise tensors/permutations.
    Connected layout selection adds strict ordered generic reductions while
    preserving initialized destinations, output layouts and iteration order.
    Fusion concerns logical host tensor SSA only. Physical representations,
    device residency and target conversion emitters require separate OOT APIs.
    Lowering/LLVM conformance and independent numerical execution remain required.
    """

    if type(contract) is HostGraphPolicy:
        return contract.optimize(source)
    if type(contract) is not GraphOptimizationContract:
        raise ValueError("graph optimization needs the explicit supported contract")
    contract.validate(source)
    module = parse_mlir_text(source)
    module.verify()
    signature = _signature(module, contract.entry)
    report = {
        "schema": "merlin.host_graph_optimization.v1",
        "contract": asdict(contract),
        "source_sha256": contract.source_sha256,
        "semantics": "conditional_structural_checks",
        "timing": "UNKNOWN",
        "hardware_conformance": "not_established",
        "implementation_sources_sha256": _implementation_sources(),
    }
    if contract.propagate_layout:
        from merlin.compiler.lower.host.propagation import rewrite_module

        if not contract.independent_element_order_unobserved:
            raise ValueError("layout propagation requires explicit independent-element order permission")
        floating = any(
            isinstance(value.type, builtin.TensorType)
            and not isinstance(value.type.get_element_type(), builtin.IntegerType)
            for op in module.walk()
            for value in (*op.operands, *op.results)
        )
        if floating and not contract.floating_environment_unobserved_nontrapping:
            raise GraphOptimizationUnavailable(
                "floating layout propagation requires its nontrapping/unobserved environment contract"
            )
        original = module.clone()
        bodies = _layout_bodies(original)
        before_bytes = _transpose_bytes(original)
        from merlin.compiler.transforms.pipeline import PassSpec, PipelineContext, PipelineSession, validate_pass_order

        sources = report["implementation_sources_sha256"]
        session = PipelineSession(
            module,
            ir_level="host_tensor",
            context=PipelineContext(
                numerical_policy=hashlib.sha256(json.dumps(asdict(contract), sort_keys=True).encode()).hexdigest(),
                resources=f"added_operations={contract.maximum_added_operations}",
                library=hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest(),
            ),
        )
        specs = (
            PassSpec("layout-source", 0, produces=("coordinates",), mutates_ir=False, ir_level="host_tensor"),
            PassSpec(
                "layout-propagation",
                1,
                requires=("coordinates",),
                produces=("rewrite",),
                invalidates=("*",),
                ir_level="host_tensor",
            ),
            PassSpec(
                "layout-conformance",
                2,
                requires=("rewrite",),
                produces=("conformance",),
                mutates_ir=False,
                ir_level="host_tensor",
            ),
        )
        if contract.physical_layout:
            from merlin.compiler.lower.host.layout import plan_islands
            from merlin.compiler.lower.host.layout import rewrite_module as realize_layouts

            specs = (
                specs[0],
                PassSpec(
                    "layout-planning",
                    1,
                    requires=("coordinates",),
                    produces=("layout_plans",),
                    mutates_ir=False,
                    ir_level="host_tensor",
                ),
                PassSpec(
                    "layout-propagation",
                    2,
                    requires=("coordinates", "layout_plans"),
                    produces=("rewrite",),
                    invalidates=("*",),
                    ir_level="host_tensor",
                ),
                PassSpec(
                    "layout-conformance",
                    3,
                    requires=("rewrite",),
                    produces=("conformance",),
                    mutates_ir=False,
                    ir_level="host_tensor",
                ),
            )
        validate_pass_order(specs)
        session.run(specs[0], lambda: {"coordinates": _layout_expression(module, connected=contract.connected_layout)})
        if contract.physical_layout:
            session.run(
                specs[1],
                lambda: {
                    "layout_plans": plan_islands(
                        module,
                        max_assignments=contract.maximum_layout_assignments,
                        max_permutations=contract.maximum_layout_permutations,
                        connected=contract.connected_layout,
                    )
                },
            )
            session.run(
                specs[2],
                lambda: {
                    "rewrite": realize_layouts(
                        module,
                        islands=session.analysis("layout_plans"),
                        maximum_added_operations=contract.maximum_added_operations,
                    )
                },
            )
        else:
            session.run(
                specs[1],
                lambda: {
                    "rewrite": rewrite_module(
                        module,
                        reduction_channel_block=0,
                        pointwise_only=True,
                        maximum_added_operations=contract.maximum_added_operations,
                    )
                },
            )
        layout = session.analysis("rewrite")

        def conformance():
            checked = _check_layout_output(original, module, bodies, connected=contract.connected_layout)
            if contract.physical_layout:
                checked["physical_layout"] = layout.verify(module)
            return {"conformance": checked}

        session.run(specs[-1], conformance)
        after_bytes = _transpose_bytes(module)
        if before_bytes - after_bytes < contract.minimum_transpose_bytes_saved:
            reasons = []
            if contract.physical_layout:
                for island in layout.to_dict()["islands"]:
                    reasons.extend(row["code"] for row in island["selection"]["refusals"])
                    if island.get("reason"):
                        reasons.append(island["reason"])
            detail = "; " + ", ".join(sorted(set(reasons))) if reasons else ""
            raise GraphOptimizationUnavailable(
                "layout rewrite does not satisfy its materialization-byte cost contract" + detail
            )
        report["layout"] = {
            "rewrite": layout.to_dict(),
            "before_transpose_bytes": before_bytes,
            "after_transpose_bytes": after_bytes,
            "cost_kind": "structural_byte_proxy",
            "library_sources_sha256": sources,
            "passes": [asdict(record) for record in session.records],
            "pointwise_work": session.analysis("conformance"),
        }
    if contract.column_reduction_layout:
        from merlin.compiler.transforms.dense.row_batching import rewrite_module as orient_reductions

        if not contract.independent_element_order_unobserved:
            raise ValueError("contraction layout requires explicit independent-element order permission")
        if not contract.floating_environment_unobserved_nontrapping and any(
            isinstance(v.type, builtin.TensorType) and not isinstance(v.type.element_type, builtin.IntegerType)
            for op in module.walk()
            for v in (*op.operands, *op.results)
        ):
            raise GraphOptimizationUnavailable("contraction layout requires its unobserved floating environment")
        before_operations = sum(1 for _ in module.walk())
        reduction = orient_reductions(module)
        if sum(1 for _ in module.walk()) - before_operations > contract.maximum_added_operations:
            raise ValueError("contraction layout exceeds its compilation-space cost contract")
        report["column_reduction_layout"] = {
            "rewrite": reduction.to_dict(),
            "semantics": "checked_signed_modular_recurrence_and_original_reduction_order",
            "hardware_conformance": "not_established",
        }
    if contract.fusion_groups:
        from merlin.compiler.partition.dispatch_program import build_dispatch_program
        from merlin.compiler.partition.global_plan import ValueRepresentation
        from merlin.compiler.partition.global_plan_emission import emit_global_plan
        from merlin.compiler.partition.outline import outline_dispatches
        from merlin.compiler.partition.outlined_plan_emission import (
            OutlinedGlobalPlanEmitter,
            _expanded_driver,
            plan_dispatch_fusion,
        )

        original_count = sum(1 for _ in module.walk())
        outlined = outline_dispatches(module, forward=contract.entry)
        program = build_dispatch_program(outlined, entry=contract.entry)
        plan = plan_dispatch_fusion(
            program,
            contract.fusion_groups,
            placement="host",
            representation=lambda name: ValueRepresentation("tensor_ssa", "logical", program.buffers[name].dtype),
        )
        _check_fusion_growth(outlined, plan, contract.entry, original_count, contract.maximum_added_operations)
        actual_source = _expanded_driver(module, contract.entry)
        _layout_expression(actual_source)
        actual_bodies = _layout_bodies(actual_source)
        emitter = OutlinedGlobalPlanEmitter(outlined)
        emitted = emit_global_plan(program, plan, emitter)
        if emitter.module is None or emitter.proof is None or emitter.emitted_outline is None:
            raise ValueError("fusion emitter produced no checked executable graph")
        _check_layout_output(actual_source, _expanded_driver(emitter.module, contract.entry), actual_bodies)
        module = emitter.module
        if program.n_dispatches - emitted.dispatch.n_dispatches < contract.minimum_dispatches_saved:
            raise ValueError("fusion does not satisfy its dispatch-count cost contract")
        if sum(1 for _ in module.walk()) - original_count > contract.maximum_added_operations:
            raise ValueError("fusion exceeds its explicit compilation-space cost contract")
        report["fusion"] = {
            "proof": emitter.proof,
            "plan_digest": plan.digest,
            "before_dispatches": program.n_dispatches,
            "after_dispatches": emitted.dispatch.n_dispatches,
            "cost_kind": "structural_dispatch_proxy",
            "source_correspondence": "static_pointwise_coordinates_and_scalar_regions",
        }
    if _signature(module, contract.entry) != signature:
        raise ValueError("graph optimization changed the complete public entry ABI")
    output = text(module, generic=True)
    report["transformed_sha256"] = hashlib.sha256(output.encode()).hexdigest()
    return output, report


# ---- host_graph_policy -------------------------------------------------------------------------
# Select checked host layouts from immutable original software permissions.
#
# The operator supplies the software file, never a candidate permission override.
# Prepared source is bound at the existing logical-host hook. Missing permission
# or insufficient structural savings retains exact input bytes. Invalid inputs,
# stale selections and conformance failures remain errors, not fallback reasons.

SCHEMA = "merlin.host_graph_policy.v1"
_BUDGET = 1 << 20


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


def _read(path):
    with path.open("rb") as stream:
        payload = stream.read(_BUDGET + 1)
    if len(payload) > _BUDGET:
        raise ValueError("host software specification exceeds its byte budget")
    return payload


@dataclass(frozen=True)
class HostGraphPolicy:
    software_path: Path
    software_bytes: bytes
    target: str
    entry: str
    enabled: bool = True
    minimum_transpose_bytes_saved: int = 1
    maximum_added_operations: int = 8192
    physical_layout: bool = True
    maximum_layout_assignments: int = 100_000
    maximum_layout_permutations: int = 16
    connected_layout: bool = False
    column_reduction_layout: bool = False
    constant_panel_columns: int | None = None
    constant_panel_minimum_input_channels: int = 1
    context: HostOptimizationContext = field(init=False)
    sources: tuple[tuple[str, str], ...] = field(init=False)

    def __post_init__(self):

        if not isinstance(self.software_path, Path) or type(self.software_bytes) is not bytes:
            raise ValueError("host graph policy requires original software bytes and path")
        if len(self.software_bytes) > _BUDGET or any(
            type(v) is not bool
            for v in (self.enabled, self.physical_layout, self.connected_layout, self.column_reduction_layout)
        ):
            raise ValueError("host graph policy requires bounded bytes and an explicit boolean choice")
        if self.constant_panel_columns is not None and (
            type(self.constant_panel_columns) is not int or not 1 <= self.constant_panel_columns <= _BUDGET
        ):
            raise ValueError("constant panel width must be an explicit bounded positive integer")
        if (
            type(self.constant_panel_minimum_input_channels) is not int
            or not 1 <= self.constant_panel_minimum_input_channels <= _BUDGET
        ):
            raise ValueError("constant panel input-channel capability must be a bounded positive integer")
        for value in (
            self.minimum_transpose_bytes_saved,
            self.maximum_added_operations,
            self.maximum_layout_assignments,
            self.maximum_layout_permutations,
        ):
            if type(value) is not int or not 1 <= value <= _BUDGET:
                raise ValueError("host graph policy requires bounded positive structural budgets")
        context = context_from_software(
            yaml.safe_load(self.software_bytes),
            semantic_sha256=_sha(self.software_bytes),
            target=self.target,
            entry=self.entry,
            allow_missing=True,
        )
        object.__setattr__(self, "context", context)
        object.__setattr__(self, "sources", tuple(sorted(_implementation_sources().items())))

    def revalidate(self):

        if _read(self.software_path) != self.software_bytes:
            raise ValueError("original host software specification changed; reselect and replan")
        if tuple(sorted(_implementation_sources().items())) != self.sources:
            raise ValueError("host graph policy implementation changed; reselect and replan")

    def record(self):
        record = {
            "schema": SCHEMA,
            "software_spec": {
                "path": str(self.software_path),
                "sha256": _sha(self.software_bytes),
            },
            "target": self.target,
            "context": asdict(self.context),
            "enabled": self.enabled,
            "minimum_transpose_bytes_saved": self.minimum_transpose_bytes_saved,
            "maximum_added_operations": self.maximum_added_operations,
            "physical_layout": self.physical_layout,
            "maximum_layout_assignments": self.maximum_layout_assignments,
            "maximum_layout_permutations": self.maximum_layout_permutations,
            "connected_layout": self.connected_layout,
        }
        if self.column_reduction_layout:
            record["column_reduction_layout"] = True
        if self.constant_panel_columns is not None:
            record["constant_panel_columns"] = self.constant_panel_columns
            record["constant_panel_minimum_input_channels"] = self.constant_panel_minimum_input_channels
        return record

    def retain(self, directory):
        self.revalidate()
        (Path(directory) / "host-graph-software.yaml").write_bytes(self.software_bytes)
        (Path(directory) / "host-graph-policy.json").write_text(json.dumps(self.record(), sort_keys=True))

    def optimize(self, source):

        self.revalidate()
        allowed = (
            "logical_host" in self.context.permitted_stages
            and "propagate_layout" in self.context.permitted_optimizations
            and self.context.independent_element_order_unobserved
        )
        contract = GraphOptimizationContract(
            _sha(source.encode()),
            self.entry,
            propagate_layout=self.enabled and allowed,
            independent_element_order_unobserved=self.context.independent_element_order_unobserved,
            floating_environment_unobserved_nontrapping=self.context.floating_environment_unobserved_nontrapping,
            minimum_transpose_bytes_saved=self.minimum_transpose_bytes_saved,
            maximum_added_operations=self.maximum_added_operations,
            physical_layout=self.physical_layout,
            maximum_layout_assignments=self.maximum_layout_assignments,
            maximum_layout_permutations=self.maximum_layout_permutations,
            connected_layout=self.connected_layout,
            column_reduction_layout=self.enabled and allowed and self.column_reduction_layout,
        )
        reason = None if contract.propagate_layout else "disabled" if not self.enabled else "permission_not_admitted"
        try:
            selected, report = optimize_host_graph(source, contract)
        except GraphOptimizationUnavailable as exc:
            from dataclasses import replace

            reason = str(exc)
            selected, report = optimize_host_graph(source, replace(contract, propagate_layout=False))
        column_applied = bool(report.get("column_reduction_layout", {}).get("rewrite", {}).get("rewritten"))
        if reason is not None and not column_applied:
            selected = source
            report["transformed_sha256"] = _sha(source.encode())
        if self.constant_panel_columns is not None and self.enabled and allowed:
            from merlin.compiler.dialects.xdsl_context import text
            from merlin.compiler.frontend.linalg_mlir import parse_mlir_text
            from merlin.compiler.lower.host.layout import request_constant_panels

            module = parse_mlir_text(selected)
            request_constant_panels(
                module, self.constant_panel_columns, minimum_input_channels=self.constant_panel_minimum_input_channels
            )
            selected = text(module, generic=True)
            report["transformed_sha256"] = _sha(selected.encode())
            report["constant_panel_request"] = self.constant_panel_columns
        report["automatic_selection"] = {
            "policy": self.record(),
            "status": "applied" if reason is None or column_applied else "unchanged",
            "reason": reason,
        }
        self.revalidate()
        return selected, report

    def recheck(self, directory, paths):
        self.revalidate()
        directory = Path(directory)
        paths = {
            **paths,
            "software_spec": directory / "host-graph-software.yaml",
            "policy": directory / "host-graph-policy.json",
        }
        binding = directory / "host-graph-source-binding.json"
        if binding.exists() or binding.is_symlink():
            paths["source_binding"] = binding
        if any(path.is_symlink() or not path.is_file() for path in paths.values()):
            raise ValueError("host graph policy artifacts are absent or indirect")
        if paths["software_spec"].read_bytes() != self.software_bytes or json.loads(
            paths["policy"].read_text()
        ) != json.loads(json.dumps(self.record())):
            raise ValueError("host graph original software or policy artifacts changed")
        if "source_binding" in paths:
            _require_projected_source(json.loads(binding.read_text()), paths["input"].read_text())
        selected, expected = self.optimize(paths["input"].read_text())
        if paths["selected"].read_text() != selected or json.loads(paths["report"].read_text()) != json.loads(
            json.dumps(expected)
        ):
            raise ValueError("host graph policy, prepared source or realized selection changed")
        return {role: {"path": str(path.resolve()), "sha256": _sha(path.read_bytes())} for role, path in paths.items()}


def read_host_graph_policy(software_path, *, target, entry, **choices):
    """Read explicitly selected original semantics; absent permissions stay false."""
    path = Path(software_path).resolve(strict=True)
    return HostGraphPolicy(path, _read(path), target, entry, **choices)


def _require_projected_source(binding, source):
    """Reopen the exact ordinary projection product; saved metadata grants nothing."""
    if (
        type(binding) is not dict
        or set(binding) != {"schema", "stage", "source"}
        or binding["schema"] != "merlin.host_layout_source_binding.v1"
        or binding["stage"] != "after_source_projection"
        or type(binding["source"]) is not dict
        or set(binding["source"]) != {"path", "sha256"}
    ):
        raise ValueError("host layout source binding is malformed")
    pin = binding["source"]
    path = Path(pin["path"])
    if not path.is_absolute() or any(part.is_symlink() for part in (path, *path.parents)) or not path.is_file():
        raise ValueError("host layout needs its original direct projected source")
    raw = path.read_bytes()
    # Ordinary projection writers already emit this exact generic IR, with
    # an optional final newline. Compare their complete bytes at every use;
    # reparsing a large unchanged model adds no correspondence information.
    canonical = source.encode()
    if _sha(raw) != pin["sha256"] or raw not in (canonical, canonical + b"\n"):
        raise ValueError("host layout projected source or exact input correspondence changed")


def prepare_group_layout(module, policy, directory, *, projected_source=None):
    """Apply the existing checked policy before source groups become device calls.

    Unknown operations remain coordinate/effect boundaries under the existing
    conformance checker. All original function signatures and observations stay
    intact; source group, object and numerical owners must still rejoin this IR.
    """
    from merlin.compiler.dialects.xdsl_context import text
    from merlin.compiler.frontend.linalg_mlir import parse_mlir_text

    if type(policy) is not HostGraphPolicy:
        raise TypeError("pre-group layout requires original software-owned HostGraphPolicy")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    policy.retain(directory)
    source = text(module, generic=True)
    if projected_source is not None:
        path = Path(projected_source).absolute()
        binding = {
            "schema": "merlin.host_layout_source_binding.v1",
            "stage": "after_source_projection",
            "source": {"path": str(path), "sha256": _sha(path.read_bytes())},
        }
        _require_projected_source(binding, source)
        (directory / "host-graph-source-binding.json").write_text(json.dumps(binding, sort_keys=True))
    selected, report = policy.optimize(source)
    (directory / "host-graph-input.mlir").write_text(source)
    (directory / "host-graph-optimized.mlir").write_text(selected)
    (directory / "host-graph-optimization.json").write_text(json.dumps(report, sort_keys=True))
    result = module if selected == source else parse_mlir_text(selected)
    result.verify()
    return result


def bind_group_layout_artifacts(device, work, compilation):
    """Retain the actual early policy consumption in the ordinary object recipe."""
    policy = getattr(device, "pre_group_layout", None)
    if policy is None:
        return

    directory = Path(work) / "pre_group_layout"
    artifacts = recheck_host_graph_artifacts(directory, policy)
    if getattr(device, "sibling_source", None) is not None and "source_binding" not in artifacts:
        raise ValueError("composed host layout lost its original projected source binding")
    for role, pin in artifacts.items():
        compilation.bind_preparation("pre_group_layout_" + role, Path(pin["path"]))


# ---- host_optimization_compile -----------------------------------------------------------------
# Ordinary compile transport for original host choices and distinct products.
#
# This private bridge is called by the ordinary object producer. The public request
# is data only; the original scoped executor owns its reviewed semantic selection.
# No reports or callback results issue formal or physical execution authority.


@dataclass(frozen=True)
class HostCompilationOptimization:
    prepared: PreparedHostOptimization
    executor: object
    explicit_selection: bool
    source_path: Path
    selected_source_path: Path

    @classmethod
    def select(cls, source, workdir, *, explicit=None, finished_llvm_only=False):
        from merlin.development.targetgen.execution.host.admission import active_selection
        from merlin.development.targetgen.packaging.loading import active_package_executor

        selected = getattr(active_package_executor(), "host_optimization_selection", None)
        original_buffer = active_selection()
        if original_buffer is not None:
            if selected is not None and selected is not original_buffer:
                raise ValueError("host buffer choices differ from the original execution owner")
            selected = original_buffer
        if explicit is not None:
            if selected is not None and selected is not explicit:
                raise ValueError("host optimization override differs from the original execution owner")
            selected = explicit
        if selected is None:
            return None
        prepared = prepare_host_optimization(source, selected, finished_llvm_only=finished_llvm_only)
        selected_source = Path(workdir) / "kernel.host.mlir"
        selected_source.write_text(prepared.selected_mlir, encoding="utf-8")
        return cls(
            prepared,
            active_package_executor(),
            explicit is not None,
            Path(workdir) / "kernel.llvm.mlir",
            selected_source,
        )

    def lowering_options(self):
        options = self.prepared.lowering_options()
        observation = {} if options.get("index_bits") is not None else None
        if observation is not None:
            options["lowering_selection"] = observation
        return options, observation

    def translation_arguments(self):
        return {
            "workdir": str(self.source_path.parent.resolve()),
            "host_optimization": self.prepared.selection.record(),
        }

    def validate_lowering(self, observation):
        if observation is not None and (
            observation.get("index_bits") != self.prepared.selection.context.index_bits
            or not observation.get("effective_pipeline")
        ):
            raise ValueError("selected host lowering lost its actual index-bound pass pipeline")

    def optimize_translated(self, llvm_path, *, observation):
        self.validate_lowering(observation)
        llvm_path = Path(llvm_path)
        llvm, report = optimize_host_llvm(llvm_path.read_text(encoding="utf-8"), self.prepared)
        report["observed_lowering"] = observation
        # Completed translation observations continue to bind their exact raw
        # product. A later selected host transform emits a distinct LLVM file.
        if report["llvm_before_sha256"] != report["llvm_after_sha256"]:
            llvm_path = llvm_path.parent / "kernel.host.ll"
            llvm_path.write_text(llvm, encoding="utf-8")
        return llvm_path, report

    def complete(self, obj, llvm, *, report):
        from merlin.development.targetgen.execution.host.admission import active_selection

        if (
            not self.explicit_selection
            and getattr(self.executor, "host_optimization_selection", None) is not self.prepared.selection
            and active_selection() is not self.prepared.selection
        ):
            raise ValueError("original execution owner changed its host optimization selection")
        write_host_optimization_report(
            report,
            selection=self.prepared.selection,
            source_path=self.source_path,
            selected_source_path=self.selected_source_path,
            llvm_path=llvm,
            object_path=obj,
            workdir=self.source_path.parent,
        )
        return obj
