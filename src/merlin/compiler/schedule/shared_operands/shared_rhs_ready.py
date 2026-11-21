"""Causal readout of the original checked shared-RHS member regions.

The common source prefix is emitted once. Each member's exact operations are
checked in the original common/private windows, with the same prefix as an
immutable input. Only the existing final-K producer/store transform is used.
Its real mixed epochs pass the ordinary causal hazard engine before the exact
source transport runs. Member boundaries remain complete device drains.
"""

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from pathlib import Path

from merlin.common.paths import module_source_path
from merlin.compiler.kernel_ir.ir import Call, Loop, TensorArg, check_structure
from merlin.compiler.schedule.blocks import Compute, Preload
from merlin.compiler.schedule.dense.compact import _digest
from merlin.compiler.schedule.library_sources import admission_point
from merlin.compiler.schedule.loops.counted_loops import LoopTransform, propose_loop_transforms
from merlin.compiler.schedule.panels.pipeline import (
    DRAIN,
    DenseReadyReadoutAdmission,
    DenseRetainedReductionAdmission,
    _calls,
    _remap,
    _verify_dense_epochs,
    complete_dense_retained_reduction,
    verify_dense_retained_reduction,
)
from merlin.compiler.schedule.panels.ready_readout import propose_dense_ready_readout
from merlin.compiler.schedule.panels.ready_schedule import DenseReadyReadoutCheckingProjection
from merlin.compiler.schedule.shared_operands.shared_lhs import _callback, _same_callback
from merlin.compiler.schedule.shared_operands.shared_rhs import DenseSharedRHSResult
from merlin.compiler.schedule.shared_operands.shared_rhs_checks import verify_dense_shared_rhs


@dataclass(frozen=True)
class DenseSharedRHSReadyResult:
    original: object
    kernel: object
    blocks: tuple
    realization: dict
    plan_text: str
    _guard: object

    def revalidate(self):
        self.original.revalidate()
        self._guard(self)


@admission_point(
    "merlin.compiler.schedule.shared_operands.shared_rhs_ready.schedule_dense_shared_rhs_ready",
    fallback_route="caller-selected unchanged original shared RHS schedule",
)
def schedule_dense_shared_rhs_ready(
    original,
    problems,
    resources,
    storages,
    adapter,
    *,
    encode_member,
    project_member,
    retention_law,
    epoch_admissions,
    readout_admissions,
    static_budget=512,
    timeout_ms=30000,
):
    """Consume an exact shared owner and independently selected causal laws.

    This binding admits only zero-seeded raw i32 outputs and complete full-K
    private member windows. It neither allocates new storage nor overlaps
    members. The caller must emit one uniform CONFIG_ST after the entry drain,
    preserve it through the final drain, and encode every explicit completion.
    """
    if type(original) is not DenseSharedRHSResult:
        raise ValueError("shared ready requires its original checked shared-RHS owner")
    original.revalidate()
    count = len(problems)
    if (
        not 2 <= count <= 32
        or len(storages) != count
        or len(epoch_admissions) != count
        or len(readout_admissions) != count
        or any(type(a) is not DenseRetainedReductionAdmission for a in epoch_admissions)
        or any(type(a) is not DenseReadyReadoutAdmission for a in readout_admissions)
        or any(p.bias for p in problems)
        or any(arg.dtype != ("i32" if arg.access == "write" else "i8") for arg in original.kernel.args)
    ):
        raise ValueError("shared ready requires bounded uniform zero-seeded raw-i32 members")
    # Reopen the original full source/pointer/lifetime proof with these exact
    # arguments. Neither a caller-supplied member map nor a prior report can
    # replace this check.
    source = verify_dense_shared_rhs(
        original.kernel,
        problems,
        original.blocks,
        resources,
        storages,
        adapter,
        project_member=project_member,
        retention_law=retention_law,
        dtypes=("i8", "i8", "i32"),
        static_budget=static_budget,
        external_buffers_disjoint=True,
        timeout_ms=timeout_ms,
    )
    prefix = original.kernel.body[0]

    def canonical(body):
        result = []
        for op in body:
            if type(op) is Loop:
                result.append(replace(op, body=canonical(op.body)))
            elif op == Call(DRAIN, ()):
                continue
            else:
                member, operation = project_member(op)
                if encode_member(member, operation) != op:
                    raise ValueError("shared ready member map failed actual encode/project roundtrip")
                result.append(adapter.encode(operation))
        return tuple(result)

    def lift(body, member, namespace):
        result = []
        for op in body:
            if type(op) is Loop:
                name = namespace + op.var
                renamed = _remap(op.body, op.var, name)
                result.append(replace(op, var=name, body=lift(renamed, member, namespace)))
            elif op == Call(DRAIN, ()):
                result.append(op)
            else:
                result.append(encode_member(member, adapter.project(op)))
        return tuple(result)

    body = []
    member_reports = []
    common = None
    for i, (p, b, s, epochs, ready) in enumerate(
        zip(problems, original.blocks, storages, epoch_admissions, readout_admissions, strict=True)
    ):
        if b.k != p.tiles[1] or not s.admits(p, b, resources):
            raise ValueError("shared ready cannot change original full-K private windows")
        ready.validate_for(adapter, epochs)
        kernel = replace(
            original.kernel,
            args=(
                TensorArg("lhs", (p.m, p.k), "i8", "read"),
                TensorArg("rhs", (p.k, p.n), "i8", "read"),
                TensorArg("output", (p.m, p.n), "i32", "write"),
            ),
            body=canonical((prefix, original.kernel.body[i + 1])),
        )
        # Shared construction keeps its singleton full-K panel wrapper. The
        # existing dense ready frontier expects its three primitive axes.
        # Consume a real loop transformation, substituting the wrapper's zero;
        # all non-singleton reduction loops and M/N tails remain counted.
        requests = []

        def panels(items, path=()):
            for op in items:
                if type(op) is not Loop:
                    continue
                current = op
                depth = 0
                while type(current) is Loop:
                    depth += 1
                    if len(current.body) != 1:
                        break
                    current = current.body[0]
                kinds = {type(adapter.project(c)) for c in _calls((op,))}
                if op.extent == 1 and depth == 4 and kinds == {Preload, Compute}:
                    requests.append(LoopTransform("hoist", (*path, op.var), index=0))
                else:
                    panels(op.body, (*path, op.var))

        panels(kernel.body)
        if requests:
            kernel = propose_loop_transforms(
                kernel, tuple(requests), index_bits=adapter.index_bits, max_static_operations=static_budget
            )
        completed = complete_dense_retained_reduction(kernel, p, b, resources, s, adapter)
        retained = verify_dense_retained_reduction(
            completed, p, b, resources, s, adapter, epochs, static_budget=static_budget, timeout_ms=timeout_ms
        )
        actual, changes = propose_dense_ready_readout(completed, p, adapter)
        projection = DenseReadyReadoutCheckingProjection(completed, p)
        report = _verify_dense_epochs(
            actual,
            p,
            resources,
            s,
            adapter,
            epochs,
            static_budget=static_budget,
            timeout_ms=timeout_ms,
            ordered_accumulator_commits=True,
            source_phase_fission=True,
            readout_compute_admission=ready,
            readout_source_projection=projection,
        )
        # The immutable common source is proved with each consumer, but has
        # only one actual emitted copy. No private work may enter that prefix.
        if common is None:
            common = actual.body[0]
            body.extend(lift((common,), None, "shared_ready_common_"))
        elif actual.body[0] != common:
            raise ValueError("shared ready changed the common RHS preparation between members")
        member_body = actual.body[1:]
        if not member_body or not list(_calls(member_body))[-1] == Call(DRAIN, ()):
            raise ValueError("shared ready lost a complete inter-member boundary")
        body.extend(lift(member_body, i, f"shared_ready_member{i}_"))
        member_reports.append(
            dict(
                member=i,
                retained=retained,
                causal=report,
                transformations=changes,
                singleton_panel_transforms=[asdict(r) for r in requests],
                actual_member_kernel_sha256=actual.digest(),
                admission=asdict(ready),
                original_windows=asdict(s),
            )
        )
    kernel = replace(original.kernel, body=tuple(body))
    if check_structure(kernel) or sum(1 for _ in _calls(kernel.body)) > static_budget:
        raise ValueError("shared ready exceeded original closed counted command budget")
    callbacks = tuple(_callback(f) for f in (encode_member, project_member, adapter.encode, adapter.project))
    paths = tuple(
        dict.fromkeys(
            (
                Path(__file__).resolve(),
                module_source_path("merlin.compiler.schedule.panels.ready_readout").resolve(),
                module_source_path("merlin.compiler.schedule.panels.ready_schedule").resolve(),
                module_source_path("merlin.compiler.schedule.panels.pipeline").resolve(),
                module_source_path("merlin.compiler.schedule.panels.construction").resolve(),
                module_source_path("merlin.compiler.schedule.panels.retained").resolve(),
                module_source_path("merlin.compiler.schedule.panels.verification").resolve(),
                module_source_path("merlin.compiler.schedule.shared_operands.shared_rhs_checks").resolve(),
                module_source_path("merlin.compiler.schedule.loops.counted_loops").resolve(),
                module_source_path("merlin.compiler.schedule.shared_operands.shared_lhs").resolve(),
                *(c[3] for c in callbacks),
            )
        )
    )
    pins = tuple(sha256(p.read_bytes()).hexdigest() for p in paths)
    report = dict(
        schema="merlin.dense_shared_rhs_ready.v1",
        source=source,
        original_shared_kernel_sha256=original.kernel.digest(),
        kernel_sha256=kernel.digest(),
        members=member_reports,
        common_rhs_preparations=1,
        inter_member_completion="full device drain; no cross-member overlap",
        numerical_state="one uniform raw-i32 CONFIG_ST through final drain",
        passes=[
            "original_shared_source_and_lifetime_check",
            "retained_member_epochs",
            "exact_final_k_producer_store_frontier",
            "actual_causal_hazards",
            "exact_source_transport",
            "single_common_prefix_and_closed_member_composition",
        ],
    )
    stamp = _digest((kernel.digest(), report, tuple(asdict(b) for b in original.blocks), original.plan_text))

    def guard(result):
        if (
            result.original is not original
            or pins != tuple(sha256(p.read_bytes()).hexdigest() for p in paths)
            or any(
                not _same_callback(f, c)
                for f, c in zip(
                    (encode_member, project_member, adapter.encode, adapter.project), callbacks, strict=True
                )
            )
            or _digest(
                (result.kernel.digest(), result.realization, tuple(asdict(b) for b in result.blocks), result.plan_text)
            )
            != stamp
        ):
            raise ValueError("checked shared ready effects, member map, or proof changed; recheck")

    result = DenseSharedRHSReadyResult(original, kernel, original.blocks, report, original.plan_text, guard)
    result.revalidate()
    return result
