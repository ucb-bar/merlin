"""Batch independent readout epochs in the actual counted panel schedule.

The target supplies ordered store/configuration snapshot and backpressure
premises. This pass preserves data-call order, operand ownership and every
source output; it moves completion only across pure readouts. It supplies no
hardware capability, cycle score or physical qualification.
"""

from dataclasses import asdict, replace

from merlin.sched.ir import Call, Loop

from .block_schedule import InitializeAccumulator, Store
from .compact_dense import _digest
from .dense_contracts import DenseTiledIntegerReadout, TiledIntegerStore
from .dense_panel_pipeline import (
    DRAIN,
    DensePanelOverlapAdmission,
    DensePanelPipelineResult,
    DenseRetainedReductionAdmission,
    verify_dense_panel_pipeline,
    verify_dense_retained_reduction,
)


def batch_dense_panel_readouts(result, problem, block, resources, storage, adapter, admission,
                               *, static_budget=128, timeout_ms=5000):
    """Return a checked schedule with fewer pure-readout completion boundaries.

    Complete loops stay counted. A group contains at most the selected issued
    readout budget; its pending queue can be smaller because the selected target
    must apply backpressure. No movement, initialization or compute is crossed.
    Per-tile varying numerical configuration remains unsupported here. An
    original tiled readout with identical coefficient bits at every source
    column retains those exact calls under the same ordered snapshot premise.
    """
    from .panel_tile_loads import PanelTileLoadResult

    if type(result) is PanelTileLoadResult:
        return _batch_wide_panel_readouts(
            result, problem, block, resources, storage, adapter, admission,
            static_budget=static_budget, timeout_ms=timeout_ms,
        )
    if (type(result) is not DensePanelPipelineResult
            or type(admission) not in (DensePanelOverlapAdmission, DenseRetainedReductionAdmission)
            or type(admission) is not type(result.admission)):
        raise ValueError("readout batching requires an original checked panel result and selected admission")
    result.revalidate()
    admission.__post_init__()
    if replace(admission,
               max_issued_readouts=result.admission.max_issued_readouts,
               pending_readouts=result.admission.pending_readouts,
               readout_configuration=result.admission.readout_configuration) != result.admission:
        raise ValueError("readout batching cannot change original operand/compute execution premises")
    if (type(result.output_readout) is DenseTiledIntegerReadout
            and len(set(result.output_readout.column_scale_bits)) != 1):
        raise ValueError("varying per-tile readout configuration needs an independent snapshot mapping")
    kernel, groups = _propose(result.kernel, adapter, admission.max_issued_readouts)
    verifier = (verify_dense_retained_reduction if type(admission) is DenseRetainedReductionAdmission
                else verify_dense_panel_pipeline)
    report = verifier(
        kernel, problem, block, resources, storage, adapter, admission,
        static_budget=static_budget, timeout_ms=timeout_ms,
        output_readout=result.output_readout, bias_source=result.bias_source,
    )
    report["input_schedule"] = result.realization
    report["schedule_ir"] = result.realization["schedule_ir"]
    report["readout_batching"] = {
        "source_kernel_sha256": result.kernel.digest(),
        "actual_kernel_sha256": kernel.digest(),
        "groups": groups,
        "removed_dynamic_drains": (result.realization["dynamic_epoch_drains"] - report["dynamic_epoch_drains"]),
        "target_execution": "not_established",
    }
    if type(result.output_readout) is DenseTiledIntegerReadout:
        report["readout_batching"]["uniform_original_scale_bits"] = result.output_readout.column_scale_bits[0]
    rewritten = replace(result, kernel=kernel, realization=report, admission=admission,
                        checked_kernel_sha256=kernel.digest(), checked_admission_sha256=_digest(asdict(admission)))
    result.revalidate()
    rewritten.revalidate()
    return rewritten


def _batch_wide_panel_readouts(result, problem, block, resources, storage, adapter, admission,
                              *, static_budget, timeout_ms):
    """Compose on physical wide commands; check every decoded primitive band.

    The original wide result already proves its complete checking expansion.
    Readout grouping must commute exactly with that expansion. We run the
    ordinary completion/source checker on the grouped expansion and bind the
    physical proposal to it, without reinterpreting wide calls as one tile or
    granting a new load, configuration, address or completion premise.
    """
    from .panel_tile_loads import expand_panel_tile_loads

    result.revalidate()
    expanded, _ = expand_panel_tile_loads(
        result.kernel, project=result.project, encode_band=result.encode_band,
        max_static_nodes=result.max_static_nodes,
    )
    logical = replace(
        result.original, kernel=expanded, realization=result.realization,
        checked_kernel_sha256=expanded.digest(),
    )
    grouped = batch_dense_panel_readouts(
        logical, problem, block, resources, storage, adapter, admission,
        static_budget=static_budget, timeout_ms=timeout_ms,
    )
    proposal, _ = _propose(result.kernel, adapter, admission.max_issued_readouts)
    final_expansion, projection = expand_panel_tile_loads(
        proposal, project=result.project, encode_band=result.encode_band,
        max_static_nodes=result.max_static_nodes,
    )
    if final_expansion != grouped.kernel:
        raise ValueError("physical readout grouping changed the checked wide-load bands or their source order")
    report = dict(grouped.realization)
    report["readout_batching"] = {
        **report["readout_batching"],
        "checking_source_kernel_sha256": expanded.digest(),
        "source_kernel_sha256": result.kernel.digest(),
        "actual_kernel_sha256": proposal.digest(),
        "wide_load_composition": "exact expansion commutes with checked readout grouping",
    }
    report["tile_load_projection"] = projection
    report["kernel_sha256"] = proposal.digest()
    report["checking_expansion_sha256"] = final_expansion.digest()
    rewritten = replace(
        result, kernel=proposal, original=grouped, realization=report,
        checked_kernel_sha256=proposal.digest(), checked_projection_sha256=_digest(projection),
    )
    result.revalidate()
    rewritten.revalidate()
    return rewritten


def _propose(original, adapter, max_readouts, *, operations=(Store, TiledIntegerStore), prefix="batched_readout_epoch"):
    names = set()

    def collect(body):
        for statement in body:
            if type(statement) is Loop:
                names.add(statement.var)
                collect(statement.body)

    collect(original.body)
    groups = []

    def strip(statement):
        if type(statement) is Call:
            if statement == Call(DRAIN, ()):
                return (), 0
            if type(adapter.project(statement)) not in operations:
                return None
            return (statement,), 1
        if type(statement) is not Loop:
            return None
        parts = [strip(child) for child in statement.body]
        if any(part is None for part in parts):
            return None
        body = tuple(child for part, _count in parts for child in part)
        count = statement.extent * sum(count for _part, count in parts)
        # Only the original explicit singleton completion wrappers disappear.
        if statement.extent == 1 and statement.body and statement.body[-1] == Call(DRAIN, ()):
            return body, count
        return (replace(statement, body=body),), count

    def rewrite(body, outer=1):
        output = []
        for statement in body:
            proposed = strip(statement)
            if proposed is not None and 1 < proposed[1] <= max_readouts:
                index = len(groups)
                name = f"{prefix}_{index}"
                while name in names:
                    name += "_"
                names.add(name)
                output.append(Loop(name, 1, (*proposed[0], Call(DRAIN, ()))))
                groups.append({"readouts_per_epoch": proposed[1], "dynamic_epochs": outer})
            elif type(statement) is Loop:
                output.append(replace(statement, body=rewrite(statement.body, outer * statement.extent)))
            else:
                output.append(statement)
        return tuple(output)

    kernel = replace(original, body=rewrite(original.body))
    # Removing only explicit completion wrappers must leave the complete
    # symbolic data-call tree unchanged, including every numerical operand.
    def data(body):
        output = []
        for statement in body:
            if statement == Call(DRAIN, ()):
                continue
            if type(statement) is Loop:
                inner = data(statement.body)
                if statement.extent == 1 and statement.body and statement.body[-1] == Call(DRAIN, ()):
                    output.extend(inner)
                else:
                    output.append(replace(statement, body=inner))
            else:
                output.append(statement)
        return tuple(output)

    if data(original.body) != data(kernel.body):
        raise ValueError("readout batching changed original symbolic data calls or source order")
    return kernel, groups


def batch_dense_panel_initializers(result, problem, block, resources, storage, adapter, admission,
                                   *, static_budget=128, timeout_ms=5000):
    """Batch pure bias loads into proven pairwise-disjoint private ACC rows.

    The selected retained admission must explicitly cover ordered load-state
    capture and bounded initializer issue. No staging, compute or readout is
    crossed; every original initializer/data call and numerical operand stays
    in order. The final actual epoch checker proves all simultaneous ACC write
    intervals disjoint and retains a real full drain before any consumer.
    """
    if (type(result) is not DensePanelPipelineResult
            or type(result.admission) is not DenseRetainedReductionAdmission
            or type(admission) is not DenseRetainedReductionAdmission):
        raise ValueError("initializer batching requires the original checked retained result and admission")
    result.revalidate()
    admission.__post_init__()
    if replace(admission,
            max_issued_initializations=result.admission.max_issued_initializations,
            initialization_configuration=result.admission.initialization_configuration) != result.admission:
        raise ValueError("initializer batching cannot change original operand/compute/readout premises")
    kernel, groups = _propose(result.kernel, adapter, admission.max_issued_initializations,
                              operations=(InitializeAccumulator,), prefix="batched_initializer_epoch")
    report = verify_dense_retained_reduction(kernel, problem, block, resources, storage, adapter, admission,
        static_budget=static_budget, timeout_ms=timeout_ms,
        output_readout=result.output_readout, bias_source=result.bias_source)
    report["input_schedule"] = result.realization
    report["schedule_ir"] = result.realization["schedule_ir"]
    report["initializer_batching"] = {
        "source_kernel_sha256": result.kernel.digest(), "actual_kernel_sha256": kernel.digest(),
        "groups": [{"initializations_per_epoch": g["readouts_per_epoch"], "dynamic_epochs": g["dynamic_epochs"]}
                   for g in groups],
        "removed_dynamic_drains": result.realization["dynamic_epoch_drains"]-report["dynamic_epoch_drains"],
        "target_execution": "not_established",
    }
    rewritten = replace(result, kernel=kernel, realization=report, admission=admission,
                        checked_kernel_sha256=kernel.digest(), checked_admission_sha256=_digest(asdict(admission)))
    result.revalidate()
    rewritten.revalidate()
    return rewritten
