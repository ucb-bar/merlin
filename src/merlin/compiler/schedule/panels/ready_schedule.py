"""Final-K readout frontiers with explicit causal accumulator completion.

Actual counted PRELOAD/COMPUTE pairs remain adjacent. Each original output store
follows its complete final K run in the same epoch; the full accumulator view
and all input staging epochs stay intact. The target must separately establish
producer retirement, pending-write read readiness, store snapshot/backpressure
and the final drain. This module supplies no cycle spacing or execution grant.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from pathlib import Path

from merlin.compiler.kernel_ir.ir import Call, Kernel, Loop, check_integer_domain, check_structure, free_vars
from merlin.compiler.kernel_ir.primitives import _loop_at, _splice
from merlin.compiler.schedule.blocks import Compute, InitializeAccumulator, Load, Preload, Store
from merlin.compiler.schedule.dense.compact import _digest
from merlin.compiler.schedule.dense.contracts import TiledIntegerStore
from merlin.compiler.schedule.library_sources import admission_point
from merlin.compiler.schedule.loops.counted_loops import LoopTransform, propose_loop_transforms
from merlin.compiler.schedule.panels.load_coalescing import _linear, _plain
from merlin.compiler.schedule.panels.pipeline import (
    DRAIN,
    DensePanelPipelineResult,
    DenseReadyReadoutAdmission,
    DenseRetainedReductionAdmission,
    _calls,
    _remap,
    _verify_dense_epochs,
)
from merlin.compiler.schedule.panels.ready_readout import (
    DenseReadyInputPlacement,
    propose_dense_ready_input_pipeline,
    propose_dense_ready_readout,
)
from merlin.compiler.schedule.panels.segmented_pipeline import DenseSegmentedPipelineResult, _assignment

# ---- dense_ready_readout -----------------------------------------------------------------------


@dataclass(frozen=True)
class DenseReadyReadoutCheckingProjection:
    """Exact deterministic transform transport, usable only after actual hazards."""

    original_kernel: Kernel
    problem: object
    canonical_load_runs: bool = False
    input_pipeline_row_offset: int | None = None
    input_pipeline_placement: DenseReadyInputPlacement | None = None
    final_k_prefix: bool = False
    array_prefix_admission: object = None

    def _expected(self, adapter, storage, *, retained=False):
        if (
            self.input_pipeline_placement is not None
            and self.input_pipeline_row_offset != self.input_pipeline_placement.lhs_slot_offset
        ):
            raise ValueError("ready input emission must consume the selected placement")
        expected, regions = propose_dense_ready_readout(
            self.original_kernel,
            self.problem,
            adapter,
            final_k_prefix=self.final_k_prefix,
            array_prefix_admission=self.array_prefix_admission if retained else None,
        )
        input_pipeline = []
        if self.input_pipeline_row_offset is not None:
            expected, input_pipeline = propose_dense_ready_input_pipeline(
                expected,
                adapter,
                storage,
                row_offset=self.input_pipeline_row_offset,
                placement=self.input_pipeline_placement,
            )
        if type(self.canonical_load_runs) is not bool:
            raise ValueError("explicit exact ready-readout load representation required")
        canonical_changes = []
        if self.canonical_load_runs:
            from merlin.compiler.schedule.panels.tile_loads import _canonical_load_runs

            def original_load(call):
                if call == Call(DRAIN, ()):
                    return None
                operation = adapter.project(call)
                return operation if type(operation) is Load else None

            expected, canonical_changes = _canonical_load_runs(expected, original_load)
        return expected, regions, input_pipeline, canonical_changes

    def hazard_kernel(self, actual, adapter, storage):
        """Check exact prefix STAY, then strengthen its unchanged scratch reads.

        The existing typed array owner rewrites ONLY pure K-prefix pairs. All
        loads, input reads, ACC updates, final fresh pairs, stores and drains
        retain their sequence/addresses. Checking fresh RHS scratch reads is
        stronger than reading their captured array values: the epoch owner
        still proves immutable RHS rows through every original consume. No
        STAY overlaps a store and no target store-preserves-array law is used.
        """
        if self.array_prefix_admission is None:
            return actual, None
        expected, regions, _, _ = self._expected(adapter, storage, retained=True)
        if actual != expected or actual.digest() != expected.digest():
            raise ValueError("actual array prefix differs from the exact pure retained-prefix transformation")
        fresh, _, _, _ = self._expected(adapter, storage)
        return fresh, dict(
            actual_kernel_sha256=actual.digest(),
            fresh_read_kernel_sha256=fresh.digest(),
            admission=asdict(self.array_prefix_admission),
            regions=regions,
            premise="pure prefix only; same input/ACC/store accesses and issue order; stronger immutable RHS reads",
            final_k="all original final-K consumes are fresh; original causal stores and full drain",
        )

    def checking_kernel(self, actual, adapter, storage):
        expected, regions, input_pipeline, canonical_changes = self._expected(adapter, storage)
        if actual != expected or actual.digest() != expected.digest():
            raise ValueError("ready readout actual commands differ from the exact original transformation")
        return self.original_kernel, dict(
            rule="exact_final_k_producer_store_frontier_after_actual_hazards",
            original_kernel_sha256=self.original_kernel.digest(),
            actual_kernel_sha256=actual.digest(),
            regions=regions,
            canonical_load_runs=canonical_changes,
            input_pipeline=input_pipeline,
            geometry="original owned views; only explicit checked LHS slot relocation",
            correspondence="same complete counted products/stores; only independent outputs interchanged",
            premise="all overlapping producer instances precede each store; no later ACC reuse in epoch",
            emission="actual frontier commands, never the checking projection",
        )


@dataclass(frozen=True)
class DenseReadyReadoutResult:
    original: DenseSegmentedPipelineResult | DensePanelPipelineResult
    kernel: Kernel
    realization: dict
    admission: DenseReadyReadoutAdmission
    problem: object
    checked_identity: str
    owner_sha256: str
    _adapter: object
    _storage: object
    input_pipeline_row_offset: int | None = None
    input_pipeline_placement: DenseReadyInputPlacement | None = None
    array_prefix_admission: object = None

    @property
    def source_identity(self):
        return self.original.source_identity

    @property
    def output_readout(self):
        return self.original.output_readout

    @property
    def bias_source(self):
        return self.original.bias_source

    @property
    def readout_qualification_identity(self):
        return self.original.readout_qualification_identity

    def revalidate(self):
        self.original.revalidate()
        if (
            self.input_pipeline_placement is not None
            and self.input_pipeline_row_offset != self.input_pipeline_placement.lhs_slot_offset
        ):
            raise ValueError("ready input emission must consume the selected placement")
        expected, _ = propose_dense_ready_readout(
            self.original.kernel,
            self.problem,
            self._adapter,
            final_k_prefix=self.array_prefix_admission is not None,
            array_prefix_admission=self.array_prefix_admission,
        )
        if self.input_pipeline_row_offset is not None:
            expected, _ = propose_dense_ready_input_pipeline(
                expected,
                self._adapter,
                self._storage,
                row_offset=self.input_pipeline_row_offset,
                placement=self.input_pipeline_placement,
            )
        if (
            self.kernel != expected
            or self.kernel.digest() != self.realization["kernel_sha256"]
            or self.checked_identity
            != _digest(
                (
                    self.original.kernel.digest(),
                    self.kernel.digest(),
                    asdict(self.admission),
                    asdict(self.problem),
                    repr(self._adapter),
                    asdict(self._storage),
                    None if self.input_pipeline_placement is None else asdict(self.input_pipeline_placement),
                    None if self.array_prefix_admission is None else asdict(self.array_prefix_admission),
                )
            )
            or sha256(Path(__file__).read_bytes()).hexdigest() != self.owner_sha256
        ):
            raise ValueError("ready readout original source, actual commands, capability or owner changed")

    def materialize_scf(self):
        from merlin.compiler.schedule.loops.scf_commands import materialize_scf_commands

        self.revalidate()
        module = materialize_scf_commands(self.kernel)
        self.revalidate()
        return module

    def emit_index_program(self, control, lower_effect, *, complete_epoch):
        """Emit actual checked effects with the same encoder and full-drain callback."""
        from merlin.compiler.schedule.loops.counted_loops import emit_index_effect_program
        from merlin.compiler.schedule.panels.pipeline import DRAIN

        self.revalidate()
        report = emit_index_effect_program(
            self.kernel, control, lower_effect, complete_epoch=complete_epoch, completion_instruction=DRAIN
        )
        self.revalidate()
        return report

    def emit_llvm(self, builder, lower_call):
        from merlin.compiler.schedule.loops.counted_loops import FnBuilderIndexAdapter
        from merlin.compiler.schedule.loops.scf_commands import emit_scf_commands

        module = self.materialize_scf()
        report = emit_scf_commands(module, FnBuilderIndexAdapter(builder), lower_call)
        self.revalidate()
        return report


def _ready_execution_bound(result, problem, block, resources, storage):
    if type(result) not in (DenseSegmentedPipelineResult, DensePanelPipelineResult):
        raise ValueError("ready readout requires its original checked owner")
    result.revalidate()
    original = result.admission
    if type(result) is DenseSegmentedPipelineResult:
        _assignment(problem, block, resources, storage)
        return replace(
            original.overlap,
            max_issued_compute_pairs=min(
                original.overlap.max_issued_compute_pairs, original.reduction.max_issued_compute_pairs
            ),
            max_issued_loads=min(original.overlap.max_issued_loads, original.reduction.max_issued_loads),
            max_issued_readouts=min(original.overlap.max_issued_readouts, original.reduction.max_issued_readouts),
        )
    else:
        if (
            type(original) is not DenseRetainedReductionAdmission
            or block.k != problem.tiles[1]
            or not storage.admits(problem, block, resources)
        ):
            raise ValueError("ready readout requires the original complete full-K storage and reduction owner")
        return original


@admission_point(
    "merlin.compiler.schedule.panels.ready_schedule.schedule_dense_ready_readout",
    fallback_route="caller-selected original route; the unchanged input checked result keeps its original readout",
)
def schedule_dense_ready_readout(
    result,
    problem,
    block,
    resources,
    storage,
    adapter,
    admission,
    *,
    static_budget=128,
    timeout_ms=5000,
    input_pipeline_row_offset=None,
    input_pipeline_placement=None,
    array_prefix_admission=None,
):
    """Consume a checked segmented/full-K result and separate causal law.

    The final compute/store frontier retains original ownership and reduction.
    An explicit ``input_pipeline_row_offset`` additionally consumes the selected
    disjoint-load conjunction and two LHS slots inside the original owned view.
    Complete drains retain the original single ACC region between groups.
    """
    from merlin.compiler.schedule.panels.ready_readout import (
        DenseReadyInputPlacement,
        propose_dense_ready_input_pipeline,
        propose_dense_ready_readout,
    )

    if type(admission) is not DenseReadyReadoutAdmission:
        raise ValueError("ready readout requires its original checked owner and separate typed capability")
    if input_pipeline_placement is not None:
        if type(input_pipeline_placement) is not DenseReadyInputPlacement or input_pipeline_row_offset is not None:
            raise ValueError("ready input needs one typed placement or one explicit manual offset")
        input_pipeline_placement.validate_for(problem, resources, storage)
        if input_pipeline_placement.block != block:
            raise ValueError("ready input placement cannot replace the explicit selected block")
        input_pipeline_row_offset = input_pipeline_placement.lhs_slot_offset
    bound = _ready_execution_bound(result, problem, block, resources, storage)
    admission.validate_for(adapter, bound)
    if array_prefix_admission is not None and type(result) is not DensePanelPipelineResult:
        raise ValueError("array prefix requires the original retained full-K owner")
    actual, changes = propose_dense_ready_readout(
        result.kernel,
        problem,
        adapter,
        final_k_prefix=array_prefix_admission is not None,
        array_prefix_admission=array_prefix_admission,
    )
    input_pipeline = []
    if input_pipeline_row_offset is not None:
        if admission.disjoint_load_qualification_identity is None:
            raise ValueError("input pipeline needs the separately selected load/ready-readout conjunction")
        if type(result) is not DensePanelPipelineResult:
            raise ValueError("input pipeline requires its original retained full-K owner")
        if input_pipeline_placement is None:
            lhs_rows = block.m * block.k * problem.tile_m
            rhs_rows = block.k * block.n * problem.tile_k
            if 2 * lhs_rows * storage.lhs.unit_bytes + rhs_rows * storage.rhs.unit_bytes > resources.operand_bytes:
                raise ValueError("two live LHS panels and one RHS panel exceed the original operand byte budget")
        actual, input_pipeline = propose_dense_ready_input_pipeline(
            actual, adapter, storage, row_offset=input_pipeline_row_offset, placement=input_pipeline_placement
        )
    projection = DenseReadyReadoutCheckingProjection(
        result.kernel,
        problem,
        input_pipeline_row_offset=input_pipeline_row_offset,
        input_pipeline_placement=input_pipeline_placement,
        final_k_prefix=array_prefix_admission is not None,
        array_prefix_admission=array_prefix_admission,
    )
    fresh, array_proof = projection.hazard_kernel(actual, adapter, storage)
    if type(static_budget) is not int or static_budget <= 0 or sum(1 for _ in _calls(actual.body)) > static_budget:
        raise ValueError("actual retained prefix exceeds the original static command budget")
    report = _verify_dense_epochs(
        fresh,
        problem,
        resources,
        storage,
        adapter,
        bound,
        static_budget=static_budget,
        timeout_ms=timeout_ms,
        output_readout=result.output_readout,
        bias_source=result.bias_source,
        ordered_accumulator_commits=True,
        source_phase_fission=True,
        readout_compute_admission=admission,
        readout_source_projection=projection,
    )
    report.update(
        schema="merlin.dense_ready_readout.v1",
        transformations=changes,
        original_schedule=result.realization,
        source_identity=result.source_identity,
        additional_capability=asdict(admission),
        schedule_ir=result.realization["schedule_ir"],
    )
    if input_pipeline:
        report["input_pipeline"] = input_pipeline
    if input_pipeline_placement is not None:
        report["input_pipeline_placement"] = asdict(input_pipeline_placement)
    if array_proof is not None:
        report["array_prefix_projection"] = array_proof
        report["kernel_sha256"] = actual.digest()
    checked = DenseReadyReadoutResult(
        result,
        actual,
        report,
        admission,
        problem,
        _digest(
            (
                result.kernel.digest(),
                actual.digest(),
                asdict(admission),
                asdict(problem),
                repr(adapter),
                asdict(storage),
                None if input_pipeline_placement is None else asdict(input_pipeline_placement),
                None if array_prefix_admission is None else asdict(array_prefix_admission),
            )
        ),
        sha256(Path(__file__).read_bytes()).hexdigest(),
        adapter,
        storage,
        input_pipeline_row_offset,
        input_pipeline_placement,
        array_prefix_admission,
    )
    checked.revalidate()
    return checked


def admit_ready_readout_tile_loads(
    original,
    proposal,
    problem,
    block,
    resources,
    storage,
    adapter,
    *,
    project,
    encode_band,
    static_budget=128,
    timeout_ms=5000,
    max_static_nodes=4096,
    external_sources=None,
):
    """Compose final physical wide bands with the SAME causal frontier proof.

    Every actual band is decoded and all actual epochs are verified before the
    exact original counted source representation is used. Emission retains the
    wide commands. No old typed consumer or execution admission is broadened.
    """
    from merlin.compiler.schedule.panels.tile_loads import _admit_dense_tile_loads

    if type(original) is not DenseReadyReadoutResult:
        raise ValueError("wide causal readout requires its exact original checked result")
    original.revalidate()
    if problem != original.problem or storage != original._storage or adapter != original._adapter:
        raise ValueError("wide causal readout cannot change original problem, storage or adapter")
    bound = _ready_execution_bound(original.original, problem, block, resources, storage)
    projection = DenseReadyReadoutCheckingProjection(
        original.original.kernel,
        problem,
        canonical_load_runs=True,
        input_pipeline_row_offset=original.input_pipeline_row_offset,
        input_pipeline_placement=original.input_pipeline_placement,
        final_k_prefix=original.array_prefix_admission is not None,
        array_prefix_admission=original.array_prefix_admission,
    )

    def verifier(actual, p, b, r, s, a, admission, **options):
        if admission != original.admission:
            raise ValueError("wide addressing cannot change the causal readout capability")
        _ready_execution_bound(original.original, p, b, r, s)
        fresh, array_proof = projection.hazard_kernel(actual, a, s)
        report = _verify_dense_epochs(
            fresh,
            p,
            r,
            s,
            a,
            bound,
            **options,
            ordered_accumulator_commits=True,
            source_phase_fission=True,
            readout_compute_admission=admission,
            readout_source_projection=projection,
        )
        if array_proof is not None:
            report.update(array_prefix_projection=array_proof, kernel_sha256=actual.digest())
        return report

    return _admit_dense_tile_loads(
        original,
        proposal,
        problem,
        block,
        resources,
        storage,
        adapter,
        verifier=verifier,
        project=project,
        encode_band=encode_band,
        static_budget=static_budget,
        timeout_ms=timeout_ms,
        max_static_nodes=max_static_nodes,
        external_sources=external_sources,
    )


@admission_point(
    "merlin.compiler.schedule.panels.ready_schedule.schedule_dense_pipeline",
    fallback_route="caller-selected original route; this consumer emits nothing on refusal",
)
def schedule_dense_pipeline(
    problem,
    resources,
    storage,
    adapter,
    *,
    name,
    dtypes,
    source_identity=None,
    external_buffers_disjoint,
    schedule_plan=None,
    pipeline_admission=None,
    batch_readouts=False,
    readout_mode="separate",
    readout_admission=None,
    readout_group_path=(),
    static_budget=128,
    timeout_ms=5000,
    output_readout=None,
    bias_source=None,
    readout_qualification_identity=None,
    epoch_law=None,
    input_pipeline_row_offset=None,
    input_pipeline_placement=None,
    array_prefix_admission=None,
):
    """Consume one source-bound decision through its original checked owner.

    A compact backend can keep the same problem/resources/storage/adapter and
    lowering callback while trying panel or segmented decisions. None retains
    the ordinary compact planner. Explicit plans never silently choose another
    block/family on a capacity, tail, proof or execution-law refusal.

    The caller supplies independently selected target premises, never inferred
    from capacity or the plan: panel/retained/segmented pipeline_admission and,
    optionally, a separate causal-ready or disjoint-ACC readout_admission.
    ``batch_readouts`` consumes only the pipeline law's existing store budget.
    All original product, effect, alias, numerical, finite-epoch and completion
    checks remain in the original owners. Returned results emit their checked
    actual commands with ``emit_llvm`` or through their counted kernel. A common
    lowering callback accepts (instruction, operands, metadata=None): ready
    emission supplies its actual SCF metadata as the third argument. The target
    must lower every explicit completion, not erase it or fence every data call.
    No preferred schedule, performance score, simulator or new grant is added.

    ``input_pipeline_row_offset`` selects an explicit second owned LHS slot for
    a retained full-K ready-readout decision and requires its separate disjoint
    load conjunction. Other routes reject this option instead of ignoring it.
    Alternatively, ``input_pipeline_placement`` consumes the same capacity
    proposal returned by ``dense_ready_input_choices``. It must match this
    explicit plan and actual projected row windows; it never repairs a plan.
    ``array_prefix_admission`` selects counted final-K peeling with the
    existing pure K-prefix fresh/STAY law. Final-K pairs remain fresh and
    their original stores consume the same causal law. No store/STAY
    conjunction, extra ACC window or implicit execution permission is added.

    ``epoch_law`` (a ``dense_panel_epochs.DensePanelEpochLaw``) selects fact-backed
    epochs for a two-window decision: drains inside each output block's K stream
    (and before its readout, with result visibility) go where the law's ordering
    premises hold, and the result emits only the remaining drains. None keeps
    every original drain; the returned schedule is then unchanged.
    """
    if (
        type(batch_readouts) is not bool
        or readout_mode not in ("separate", "ready", "disjoint")
        or type(readout_group_path) is not tuple
        or any(type(v) is not str or not v for v in readout_group_path)
        or readout_mode == "separate"
        and (readout_admission is not None or readout_group_path)
        or readout_mode == "ready"
        and (readout_admission is None or readout_group_path)
        or readout_mode == "disjoint"
        and (readout_admission is None or not readout_group_path)
    ):
        raise ValueError("dense pipeline needs explicit consumed readout mode, law and group path")
    from merlin.compiler.schedule.dense.plans import read_dense_schedule_plan

    boundary = dict(
        output_readout=output_readout,
        bias_source=bias_source,
        readout_qualification_identity=readout_qualification_identity,
    )
    common = dict(
        name=name,
        dtypes=dtypes,
        source_identity=source_identity,
        external_buffers_disjoint=external_buffers_disjoint,
        static_budget=static_budget,
        schedule_plan=schedule_plan,
        **boundary,
    )
    decision = (
        None
        if schedule_plan is None
        else read_dense_schedule_plan(
            schedule_plan,
            problem,
            resources,
            storage,
            adapter,
            source_identity=source_identity,
            dtypes=dtypes,
            static_budget=static_budget,
            output_readout=output_readout,
            bias_source=bias_source,
        )
    )
    if input_pipeline_row_offset is not None and input_pipeline_placement is not None:
        raise ValueError("ready input needs one typed placement or one explicit manual offset")
    if (
        input_pipeline_row_offset is not None
        or input_pipeline_placement is not None
        or array_prefix_admission is not None
    ) and (readout_mode != "ready" or decision is None or decision.pipeline_kind != "retained_full_k"):
        raise ValueError("advancing input slots require an explicit retained full-K ready-readout decision")
    if epoch_law is not None and (decision is None or decision.pipeline_kind != "two_window"):
        raise ValueError("dense panel epochs apply to an explicit two-window panel decision")
    if decision is None or decision.pipeline_prefetch is None:
        if pipeline_admission is not None or batch_readouts or readout_mode != "separate":
            raise ValueError("compact decision cannot consume asynchronous pipeline or readout laws")
        from merlin.compiler.schedule.dense.compact import schedule_compact_dense

        return schedule_compact_dense(
            problem,
            resources,
            storage,
            adapter,
            verification_timeout_ms=timeout_ms,
            max_repairs=0 if decision is not None else 1,
            **common,
        )
    if pipeline_admission is None:
        raise ValueError("advanced dense decision requires its independently selected pipeline law")
    # Layout decisions may relocate the original assigned windows. Every later
    # proof consumes this same checked actual layout, rather than stale bases.
    chosen_storage = decision.storage
    args = (problem, decision.block, resources, chosen_storage, adapter)
    if decision.pipeline_kind == "retained_lhs_segmented_k":
        from merlin.compiler.schedule.panels.segmented_pipeline import (
            batch_dense_segmented_readouts,
            schedule_dense_segmented_pipeline,
        )

        result = schedule_dense_segmented_pipeline(
            problem, resources, storage, adapter, pipeline_admission, timeout_ms=timeout_ms, **common
        )
        if batch_readouts:
            result = batch_dense_segmented_readouts(
                result, *args, pipeline_admission, static_budget=static_budget, timeout_ms=timeout_ms
            )
        if readout_mode == "ready":
            result = schedule_dense_ready_readout(
                result,
                *args,
                readout_admission,
                static_budget=static_budget,
                timeout_ms=timeout_ms,
                input_pipeline_row_offset=input_pipeline_row_offset,
                input_pipeline_placement=input_pipeline_placement,
                array_prefix_admission=array_prefix_admission,
            )
        elif readout_mode == "disjoint":
            from merlin.compiler.schedule.panels.readouts import schedule_dense_readout_overlap

            result = schedule_dense_readout_overlap(
                result,
                *args,
                readout_admission,
                group_path=readout_group_path,
                static_budget=static_budget,
                timeout_ms=timeout_ms,
            )
    else:
        if readout_mode == "disjoint" or (readout_mode == "ready" and decision.pipeline_kind != "retained_full_k"):
            raise ValueError("readout mode requires its original checked segmented/full-K pipeline")
        from merlin.compiler.schedule.panels.pipeline import schedule_dense_panel_pipeline

        result = schedule_dense_panel_pipeline(
            problem, resources, storage, adapter, pipeline_admission, timeout_ms=timeout_ms, **common
        )
        if batch_readouts:
            from merlin.compiler.schedule.panels.readouts import batch_dense_panel_readouts

            result = batch_dense_panel_readouts(
                result, *args, pipeline_admission, static_budget=static_budget, timeout_ms=timeout_ms
            )
        if epoch_law is not None:
            from merlin.compiler.schedule.panels.epochs import schedule_dense_panel_epochs

            result = schedule_dense_panel_epochs(
                result, *args, pipeline_admission, epoch_law, static_budget=static_budget, timeout_ms=timeout_ms
            )
        if readout_mode == "ready":
            result = schedule_dense_ready_readout(
                result,
                *args,
                readout_admission,
                static_budget=static_budget,
                timeout_ms=timeout_ms,
                input_pipeline_row_offset=input_pipeline_row_offset,
                input_pipeline_placement=input_pipeline_placement,
                array_prefix_admission=array_prefix_admission,
            )
    result.revalidate()
    return result
