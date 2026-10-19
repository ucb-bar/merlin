"""Explicit logical graph optimization on the ordinary host compilation route.

Contracts select semantics and structural cost proxies, never hardware schedules.
Typed ABI, scalar bodies and the fusion emitter's expanded driver correspondence
are checked independently of pass reports. This supplies no Lean or timing proof.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from math import prod
from pathlib import Path
from typing import TYPE_CHECKING

from xdsl.dialects import builtin, func

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.xdsl_dialects._common import text

if TYPE_CHECKING:
    from .host_graph_policy import HostGraphPolicy


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

    def validate(self, source: str):
        if self.source_sha256 != hashlib.sha256(source.encode()).hexdigest():
            raise ValueError("graph optimization contract does not bind the actual input source")
        if not isinstance(self.entry, str) or not self.entry:
            raise ValueError("graph optimization requires its explicit logical entry")
        for field in (
            "propagate_layout",
            "independent_element_order_unobserved",
            "floating_environment_unobserved_nontrapping",
        ):
            if type(getattr(self, field)) is not bool:
                raise ValueError("graph permissions must be explicit booleans")
        for field in ("minimum_transpose_bytes_saved", "minimum_dispatches_saved", "maximum_added_operations"):
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
    from .layout_conformance import scalar_bodies

    return scalar_bodies(module)


def _transpose_bytes(module):
    from .layout_conformance import pointwise_kind
    from .layout_propagation import _permutation

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


def _check_layout_output(original, modified, bodies):
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
    from .layout_conformance import layout_expression

    before, before_work = layout_expression(original, include_work=True)
    after, after_work = layout_expression(modified, include_work=True)
    if before != after:
        raise ValueError("layout propagation changed complete logical output coordinates or scalar expressions")
    if any(count > before_work[key] for key, count in after_work.items()):
        raise ValueError("layout propagation increases pointwise arithmetic in a control scope")
    return {"before_scalar_evaluations": sum(before_work.values()),
            "after_scalar_evaluations": sum(after_work.values()),
            "scope": "static parallel tensor domains by typed scalar operation and lexical control scope; "
                     "not dynamic invocation counts, machine instructions or cycles"}


def _layout_expression(module):
    from .layout_conformance import layout_expression

    return layout_expression(module)


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
    from merlin.common import dense_float_ir
    from merlin.transforms import pipeline_state

    # Trusted host observations, not an SDK/export grant or imported dependency lock.
    return {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in (
            ("graph_optimizations", Path(__file__)),
            *((name, Path(__file__).with_name(name + ".py")) for name in (
                "layout_propagation", "layout_conformance", "quant_round_check",
                "host_graph_policy", "host_optimization_request")),
            ("pipeline_state", Path(pipeline_state.__file__)),
            ("dense_float_ir", Path(dense_float_ir.__file__)),
        )
    }


def recheck_host_graph_artifacts(directory: Path, contract: GraphOptimizationContract | HostGraphPolicy) -> dict:
    """Reopen actual source/report bytes; this grants no machine-code qualification."""
    paths = {role: Path(directory) / name for role, name in (
        ("input", "host-graph-input.mlir"), ("selected", "host-graph-optimized.mlir"),
        ("report", "host-graph-optimization.json"),
    )}
    if any(path.is_symlink() or not path.is_file() for path in paths.values()):
        raise ValueError("host graph optimization artifacts are absent or indirect")
    from .host_graph_policy import HostGraphPolicy

    if type(contract) is HostGraphPolicy:
        return contract.recheck(directory, paths)
    if type(contract) is not GraphOptimizationContract:
        raise ValueError("host graph artifacts require the original typed contract")
    contract.validate(paths["input"].read_text())
    report = json.loads(paths["report"].read_text())
    if (report.get("schema") != "merlin.host_graph_optimization.v1"
            or report.get("source_sha256") != contract.source_sha256
            or report.get("contract") != json.loads(json.dumps(asdict(contract)))
            or report.get("transformed_sha256") != hashlib.sha256(paths["selected"].read_bytes()).hexdigest()
            or report.get("implementation_sources_sha256") != _implementation_sources()):
        raise ValueError("host graph contract, selected source or implementation bytes changed")
    return {role: {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for role, path in paths.items()}


def optimize_host_graph(source: str, contract: GraphOptimizationContract | HostGraphPolicy) -> tuple[str, dict]:
    """Select the existing layout/fusion mechanisms before host preprocessing.

    The admitted source-check domain is static pointwise tensors/permutations.
    Fusion concerns logical host tensor SSA only. Physical representations,
    device residency and target conversion emitters require separate OOT APIs.
    Lowering/LLVM conformance and independent numerical execution remain required.
    """
    from .host_graph_policy import HostGraphPolicy

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
        from .layout_propagation import rewrite_module

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
                "floating layout propagation requires its nontrapping/unobserved environment contract")
        original = module.clone()
        bodies = _layout_bodies(original)
        before_bytes = _transpose_bytes(original)
        from merlin.transforms.pipeline import PassSpec, PipelineContext, PipelineSession, validate_pass_order

        sources = report["implementation_sources_sha256"]
        session = PipelineSession(module, ir_level="host_tensor", context=PipelineContext(
            numerical_policy=hashlib.sha256(json.dumps(asdict(contract), sort_keys=True).encode()).hexdigest(),
            resources=f"added_operations={contract.maximum_added_operations}",
            library=hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest(),
        ))
        specs = (
            PassSpec("layout-source", 0, produces=("coordinates",), mutates_ir=False, ir_level="host_tensor"),
            PassSpec("layout-propagation", 1, requires=("coordinates",), produces=("rewrite",),
                     invalidates=("*",), ir_level="host_tensor"),
            PassSpec("layout-conformance", 2, requires=("rewrite",), produces=("conformance",),
                     mutates_ir=False, ir_level="host_tensor"),
        )
        validate_pass_order(specs)
        session.run(specs[0], lambda: {"coordinates": _layout_expression(module)})
        session.run(specs[1], lambda: {"rewrite": rewrite_module(
            module, reduction_channel_block=0, pointwise_only=True,
            maximum_added_operations=contract.maximum_added_operations,
        )})
        layout = session.analysis("rewrite")
        session.run(specs[2], lambda: {"conformance": _check_layout_output(original, module, bodies)})
        after_bytes = _transpose_bytes(module)
        if before_bytes - after_bytes < contract.minimum_transpose_bytes_saved:
            raise GraphOptimizationUnavailable("layout rewrite does not satisfy its materialization-byte cost contract")
        report["layout"] = {
            "rewrite": layout.to_dict(),
            "before_transpose_bytes": before_bytes,
            "after_transpose_bytes": after_bytes,
            "cost_kind": "structural_byte_proxy",
            "library_sources_sha256": sources,
            "passes": [asdict(record) for record in session.records],
            "pointwise_work": session.analysis("conformance"),
        }
    if contract.fusion_groups:
        from merlin.xdsl_dialects.lowering.dispatch_program import build_dispatch_program
        from merlin.xdsl_dialects.lowering.global_plan import ValueRepresentation
        from merlin.xdsl_dialects.lowering.global_plan_emission import emit_global_plan
        from merlin.xdsl_dialects.lowering.outline import outline_dispatches
        from merlin.xdsl_dialects.lowering.outlined_plan_emission import (
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
