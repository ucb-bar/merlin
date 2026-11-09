"""Check actual dense panel issue epochs and original source effects."""

from __future__ import annotations

from dataclasses import asdict
from math import prod

from merlin.compiler.schedule.blocks import Compute, InitializeAccumulator, Load, Preload, Store
from merlin.compiler.schedule.dense_contracts import DenseTiledIntegerReadout, TiledIntegerStore
from merlin.compiler.schedule.dense_panel_contracts import DensePanelOverlapAdmission as DensePanelOverlapAdmission
from merlin.compiler.schedule.dense_panel_contracts import DenseReadoutComputeAdmission as DenseReadoutComputeAdmission
from merlin.compiler.schedule.dense_panel_contracts import (
    DenseReadoutConfigurationSnapshot as DenseReadoutConfigurationSnapshot,
)
from merlin.compiler.schedule.dense_panel_contracts import DenseReadyReadoutAdmission as DenseReadyReadoutAdmission
from merlin.compiler.schedule.dense_panel_ir import DRAIN as DRAIN
from merlin.compiler.schedule.dense_panel_ir import _calls as _calls
from merlin.compiler.schedule.dense_panel_ir import _serial_projection as _serial_projection
from merlin.sched.ir import Call, Const, Loop, Scaled, Select, Sum, Var, check_structure
from merlin.sched.ir.integer_domain import check_integer_domain

from .dense_panel_build import _windows
from .dense_panel_projection import _collapse_compute_radices, _fission_noninterfering_epoch_loads


def verify_dense_panel_pipeline(
    kernel,
    problem,
    block,
    resources,
    storage,
    adapter,
    admission,
    *,
    static_budget,
    timeout_ms=5000,
    output_readout=None,
    bias_source=None,
):
    """Check all source products plus actual epoch read/write/completion hazards."""
    if (
        type(admission) is not DensePanelOverlapAdmission
        or admission.hardware_identity != adapter.configuration_identity
        or admission.numerical_policy_identity != adapter.numerical_policy_identity
    ):
        raise ValueError("panel overlap differs from original selected execution/numerical capability")
    admission.__post_init__()
    _windows(problem, block, storage, resources)
    return _verify_dense_epochs(
        kernel,
        problem,
        resources,
        storage,
        adapter,
        admission,
        static_budget=static_budget,
        timeout_ms=timeout_ms,
        output_readout=output_readout,
        bias_source=bias_source,
    )


def _verify_dense_epochs(
    kernel,
    problem,
    resources,
    storage,
    adapter,
    admission,
    *,
    static_budget,
    timeout_ms,
    output_readout=None,
    bias_source=None,
    ordered_accumulator_commits=False,
    source_phase_fission=False,
    readout_compute_admission=None,
    readout_source_projection=None,
):
    """Shared proof engine; public owners validate their distinct typed laws.

    Ordered accumulator commitment is an explicit additional implementation
    premise, never inferred from storage, issue order or this private switch.
    Operand/load disjointness and the complete serial source proof always run.
    """
    import time

    import z3

    if readout_compute_admission is not None:
        if type(readout_compute_admission) not in (DenseReadoutComputeAdmission, DenseReadyReadoutAdmission):
            raise ValueError("typed additional readout/compute conjunction required")
        readout_compute_admission.validate_for(adapter, admission)
        snapshot = readout_compute_admission.configuration_snapshot
        varying = type(output_readout) is DenseTiledIntegerReadout and len(set(output_readout.column_scale_bits)) != 1
        if varying and snapshot is None:
            raise ValueError("mixed per-tile readout configuration has no selected conjunction")
        if snapshot is not None:
            if type(snapshot) is not DenseReadoutConfigurationSnapshot:
                raise ValueError("typed exact readout configuration snapshot required")
            snapshot.validate_for(adapter, output_readout)

    from merlin.compiler.schedule.compact_verification import _project, verify_compact_dense

    if (
        check_structure(kernel)
        or type(static_budget) is not int
        or static_budget <= 0
        or sum(1 for _ in _calls(kernel.body)) > static_budget
        or type(timeout_ms) is not int
        or timeout_ms <= 0
    ):
        raise ValueError("invalid panel pipeline structure or original checking/command budget")
    check_integer_domain(kernel, adapter.index_bits)
    deadline = time.monotonic() + timeout_ms / 1000
    sites = _project(_serial_projection(kernel), adapter, fold_singleton_loops=False)
    epochs = []
    serial = 0

    def structure(body, ancestry=(), inside=False):
        nonlocal serial
        for statement in body:
            if type(statement) is Loop:
                serial += 1
                scope = (serial, statement.var, statement.extent)
                drains = [s for s in statement.body if type(s) is Call and s.instr == DRAIN]
                if drains:
                    if (
                        inside
                        or statement.extent != 1
                        or len(drains) != 1
                        or statement.body[-1] is not drains[0]
                        or drains[0] != Call(DRAIN, ())
                    ):
                        raise ValueError("each panel epoch needs one final explicit full drain")
                    epochs.append((*ancestry, scope))
                    structure(statement.body[:-1], (*ancestry, scope), True)
                else:
                    structure(statement.body, (*ancestry, scope), inside)
            elif not inside:
                raise ValueError("every movement, compute and readout must be inside a drained panel epoch")

    structure(kernel.body)
    if not epochs:
        raise ValueError("panel pipeline has no explicit completion epochs")
    views = {"lhs": storage.lhs, "weight": storage.rhs, "acc": storage.accumulator}

    def expression(value, env):
        if type(value) is Const:
            return z3.IntVal(value.value)
        if type(value) is int:
            return z3.IntVal(value)
        if type(value) is Var:
            return env[value.name]
        if type(value) is Sum:
            return z3.Sum(*(expression(v, env) for v in value.terms))
        if type(value) is Scaled:
            return expression(value.expr, env) * value.factor
        if type(value) is Select:
            return z3.If(env[value.var] == value.equals, expression(value.then, env), expression(value.other, env))
        raise ValueError("unknown panel address expression")

    def domain(site, tag):
        env = {name: z3.Int(f"{tag}:{scope}:{name}") for scope, name, _ in site.scopes}
        return env, [z3.And(env[name] >= 0, env[name] < extent) for _, name, extent in site.scopes]

    def interval(view, row, rows, env):
        start = expression(row, env) * view.unit_bytes
        return view.backing, start, start + expression(rows, env) * view.unit_bytes

    def prove_disjoint(a, b, outer, *, exclude_same=False):
        sa, va, ra, na = a
        sb, vb, rb, nb = b
        if va.backing != vb.backing:
            return
        ae, ac = domain(sa, "a")
        be, bc = domain(sb, "b")
        same = [ae[name] == be[name] for _, name, _ in outer]
        if exclude_same and sa.ordinal == sb.ordinal:
            local = [ae[name] != be[name] for scope, name, _ in sa.scopes if scope not in {s[0] for s in outer}]
            if not local:
                return
            same.append(z3.Or(*local))
        _, al, ah = interval(va, ra, na, ae)
        _, bl, bh = interval(vb, rb, nb, be)
        remaining = int((deadline - time.monotonic()) * 1000)
        if remaining <= 0:
            raise ValueError("panel epoch checking budget exhausted")
        solver = z3.Solver()
        solver.set(timeout=remaining)
        solver.add(*ac, *bc, *same, al < bh, bl < ah)
        if solver.check() != z3.unsat:
            raise ValueError("panel epoch has overlapping in-flight operand or accumulator accesses")

    def prove_producer_before_readout(read, write, outer):
        # Each overlapping producer INSTANCE must already be issued before the
        # store. The new target law supplies the causal commit/readiness edge;
        # this source proof never substitutes issue for hardware completion.
        reader, rv, rr, rn = read
        writer, wv, wr, wn = write
        if rv.backing != wv.backing:
            return
        re, rc = domain(reader, "readout")
        we, wc = domain(writer, "producer")
        same = [re[name] == we[name] for _, name, _ in outer]
        _, rl, rh = interval(rv, rr, rn, re)
        _, wl, wh = interval(wv, wr, wn, we)

        def position(site, env):
            return tuple(z3.IntVal(v) if type(v) is int else env[v[1]] for v in site.path)

        left, right = position(writer, we), position(reader, re)
        before = z3.Or(
            *(
                z3.And(*(left[j] == right[j] for j in range(i)), left[i] < right[i])
                for i in range(min(len(left), len(right)))
            )
        )
        remaining = int((deadline - time.monotonic()) * 1000)
        if remaining <= 0:
            raise ValueError("causal readout epoch checking budget exhausted")
        solver = z3.Solver()
        solver.set(timeout=remaining)
        solver.add(*rc, *wc, *same, rl < wh, wl < rh, z3.Not(before))
        if solver.check() != z3.unsat:
            raise ValueError("overlapping ACC producer must precede readout; later writes/reuse are unqualified")

    def prove_assigned(access):
        site, view, row, rows = access
        env, constraints = domain(site, "assigned")
        _, low, high = interval(view, row, rows, env)
        solver = z3.Solver()
        remaining = int((deadline - time.monotonic()) * 1000)
        if remaining <= 0:
            raise ValueError("panel assigned-access checking budget exhausted")
        solver.set(timeout=remaining)
        solver.add(
            *constraints,
            z3.Or(low < view.base * view.unit_bytes, high > (view.base + view.capacity) * view.unit_bytes, high <= low),
        )
        if solver.check() != z3.unsat:
            raise ValueError("actual panel access escapes its original assigned private view")

    summaries = []
    for outer in epochs:
        marker = outer[-1][0]
        selected = [s for s in sites if marker in {scope for scope, _, _ in s.scopes}]
        loads = []
        reads = []
        writes = []
        stores = 0
        store_reads = []
        initializations = 0
        pairs = 0
        load_count = 0
        pending = None
        for site in selected:
            op = site.operation
            local = prod(extent for scope, _, extent in site.scopes if scope not in {s[0] for s in outer})
            if type(op) is Load:
                if pending is not None:
                    raise ValueError("epoch splits a physical preload/compute pair")
                load_count += local
                loads.append((site, views[op.role], op.row, op.rows))
            elif type(op) is Preload:
                if pending is not None:
                    raise ValueError("epoch overwrites an unconsumed preload descriptor")
                pending = site
            elif type(op) is Compute:
                if pending is None or pending.scopes != site.scopes:
                    raise ValueError("epoch has an unpaired compute")
                pre = pending.operation
                pairs += local
                reads.extend(
                    ((site, storage.lhs, op.input_row, op.rows), (site, storage.rhs, pre.weight_row, pre.weight_rows))
                )
                writes.append((site, storage.accumulator, pre.accumulator_row, pre.rows))
                pending = None
            elif type(op) in (Store, TiledIntegerStore):
                stores += local
                store_reads.append((site, storage.accumulator, op.accumulator_row, op.rows))
            elif type(op) is InitializeAccumulator:
                initializations += local
            else:
                raise ValueError("unknown panel effect")
        if (
            pending is not None
            or stores
            and (loads or writes or initializations)
            and readout_compute_admission is None
            or stores > admission.max_issued_readouts
            or initializations
            and (loads or writes or stores)
            or initializations > 1
        ):
            raise ValueError("initialization or readout exceeds the selected separate completion epoch")
        if stores and (loads or writes):
            if not writes or initializations or stores > readout_compute_admission.max_issued_readouts:
                raise ValueError("mixed readout epoch needs disjoint compute and its explicit issued store bound")
            if (
                type(readout_compute_admission) is DenseReadyReadoutAdmission
                and loads
                and readout_compute_admission.disjoint_load_qualification_identity is None
            ):
                raise ValueError("causal-ready readout has no mixed DMA-load execution conjunction")
            for read in store_reads:
                for load in loads:
                    prove_disjoint(read, load, outer)
            for read in store_reads:
                for write in writes:
                    if type(readout_compute_admission) is DenseReadyReadoutAdmission:
                        prove_producer_before_readout(read, write, outer)
                    else:
                        prove_disjoint(read, write, outer)
        if pairs > admission.max_issued_compute_pairs or load_count > admission.max_issued_loads:
            raise ValueError("panel epoch exceeds original selected in-flight command bounds")
        for access in (*loads, *reads, *writes, *store_reads):
            prove_assigned(access)
        for i, a in enumerate(loads):
            for b in loads[i:]:
                prove_disjoint(a, b, outer, exclude_same=True)
        for a in reads:
            for b in loads:
                prove_disjoint(a, b, outer)
        if not ordered_accumulator_commits:
            for i, a in enumerate(writes):
                for b in writes[i:]:
                    prove_disjoint(a, b, outer, exclude_same=True)
        summaries.append(
            {
                "compute_pairs": pairs,
                "future_or_initial_loads": load_count,
                "readout": bool(stores),
                "readout_commands": stores,
                "disjoint_readout_compute": bool(
                    stores and writes and type(readout_compute_admission) is DenseReadoutComputeAdmission
                ),
                "causal_ready_readout_compute": bool(
                    stores and writes and type(readout_compute_admission) is DenseReadyReadoutAdmission
                ),
                "initialization": bool(initializations),
                "issue_order": tuple(type(site.operation).__name__ for site in selected),
                "completion": DRAIN,
                "awaits": "all issued movement, operand consumption, accumulator writes and readout "
                "before window reuse",
            }
        )
    # Every ACTUAL epoch and physical interval above is checked first. Only the
    # exact typed inverse of this one counted transformation may then restore
    # its original source representation. Arbitrary kernels/callbacks cannot
    # supply a checking projection.
    source_base, readout_projection = kernel, None
    if readout_source_projection is not None:
        from merlin.compiler.schedule.dense_readout_overlap import DenseReadoutCheckingProjection

        if readout_compute_admission is None:
            raise ValueError("readout source projection needs exact typed transformation and selected conjunction")
        if type(readout_source_projection) is not DenseReadoutCheckingProjection:
            from merlin.compiler.schedule.dense_ready_readout import DenseReadyReadoutCheckingProjection

            if (
                type(readout_source_projection) is not DenseReadyReadoutCheckingProjection
                or type(readout_compute_admission) is not DenseReadyReadoutAdmission
            ):
                raise ValueError("readout source projection needs exact typed transformation and selected conjunction")
        elif type(readout_compute_admission) is not DenseReadoutComputeAdmission:
            raise ValueError("paired disjoint readout projection cannot claim a different execution law")
        source_base, readout_projection = readout_source_projection.checking_kernel(kernel, adapter, storage)
    source_kernel, phase_changes = (
        _fission_noninterfering_epoch_loads(source_base, adapter) if source_phase_fission else (source_base, [])
    )
    phase_kernel_sha256 = source_kernel.digest()
    source_kernel, radix_changes = (
        _collapse_compute_radices(source_kernel, adapter) if source_phase_fission else (source_kernel, [])
    )
    remaining = int((deadline - time.monotonic()) * 1000)
    if remaining <= 0:
        raise ValueError("panel source checking budget exhausted")
    source = verify_compact_dense(
        _serial_projection(source_kernel),
        problem,
        resources,
        storage,
        adapter,
        static_budget=static_budget,
        timeout_ms=remaining,
        output_readout=output_readout,
        bias_source=bias_source,
    )
    return {
        "schema": "merlin.dense_panel_pipeline.v1",
        "kernel_sha256": kernel.digest(),
        "source": source,
        **({"readout_checking_projection": readout_projection} if readout_projection is not None else {}),
        **(
            {"readout_compute_admission": asdict(readout_compute_admission)}
            if readout_compute_admission is not None
            else {}
        ),
        **(
            {
                "source_phase_fission": {
                    "actual_kernel_sha256": kernel.digest(),
                    "checking_kernel_sha256": source_kernel.digest(),
                    "epochs": phase_changes,
                    "phase_fission_kernel_sha256": phase_kernel_sha256,
                    "compute_radix_collapses": radix_changes,
                    "premise": "all actual per-epoch load writes pairwise disjoint and disjoint from every compute read",
                    "preserves": "all actual commands, complete counted domains, per-phase order and full epoch completion",
                    "emission": "original actual Kernel unchanged",
                }
            }
            if source_phase_fission
            else {}
        ),
        "epochs": summaries,
        "admission": asdict(admission),
        "dynamic_epoch_drains": sum(prod(extent for _, _, extent in scopes) for scopes in epochs),
        "overlapped_compute_pairs": sum(
            prod(extent for _, _, extent in scopes) * summary["compute_pairs"]
            for scopes, summary in zip(epochs, summaries)
            if summary["future_or_initial_loads"]
        ),
        "qualification": "conditional original implementation premises; no independent physical qualification",
        "target_execution": "not_established",
    }


# Preserve the original public lookup and pickle identities.
for _definition in (
    verify_dense_panel_pipeline,
    _verify_dense_epochs,
):
    _definition.__module__ = "merlin.compiler.schedule.dense_panel_pipeline"
del _definition
