"""Check complete shared RHS groups through closed member epochs.

Member scratch windows may be reused only after their complete drain. The RHS
is read-only after the prefix. Each proof view retains those actual writes and
uses the existing dense source/lifetime and retained-epoch verifiers.
"""

from dataclasses import asdict, dataclass, field, replace

from merlin.sched.ir import Call, Expr, Kernel, Loop, Ptr, TensorArg, as_expr, check_structure
from merlin.sched.ir.kernel import DTYPE_BYTES

from .block_schedule import Compute, Load, Preload, Store
from .compact_dense import DenseLoopStorage, _static_calls
from .compact_dense_group_check import _group_output_contracts, _range, _span
from .compact_verification import _project, _requested_elements, verify_compact_dense
from .dense_contracts import TiledIntegerStore
from .dense_panel_pipeline import DRAIN, _serial_projection, verify_dense_retained_reduction


@dataclass(frozen=True)
class DenseRHSReuseLaw:
    """Caller-qualified omitted-load and shared-state premises, never a grant."""

    configuration_identity: str
    numerical_policy_identity: str
    adapter_qualification_identity: str
    qualification_identity: str
    rhs_unchanged_by_member_operations: bool = field(default=False, kw_only=True)
    independent_member_interleaving: bool = field(default=False, kw_only=True)
    suppressed_loads_no_faults: bool = field(default=False, kw_only=True)
    suppressed_loads_no_extra_observations: bool = field(default=False, kw_only=True)

    def record(self):
        values = asdict(self)
        if any(type(v) is not str or not v.strip() for v in tuple(values.values())[:4]) or any(
            type(v) is not bool for v in tuple(values.values())[4:]
        ):
            raise ValueError("shared RHS requires explicit identities and Boolean original qualifications")
        return values

    def admits(self, adapter):
        values = self.record()
        return all(tuple(values.values())[4:]) and (
            self.configuration_identity,
            self.numerical_policy_identity,
            self.adapter_qualification_identity,
        ) == (adapter.configuration_identity, adapter.numerical_policy_identity, adapter.qualification_identity)


def shared_rhs_views(problems, storages, adapter):
    if (
        type(storages) is not tuple
        or len(storages) != len(problems)
        or any(type(s) is not DenseLoopStorage or s.accumulator is None for s in storages)
    ):
        raise ValueError("shared RHS requires a complete explicit member storage roster")
    rhs = storages[0].rhs
    maximum = (1 << (adapter.index_bits - 1)) - 1
    for s in storages:
        if s.rhs != rhs:
            raise ValueError("shared RHS members require the identical physical RHS view")
        views = (s.lhs, s.rhs, s.accumulator)
        for i, a in enumerate(views):
            a.__post_init__()
            if (a.base + a.capacity) * a.unit_bytes > maximum:
                raise ValueError("shared RHS storage overflows selected integer addressing")
            for b in views[:i]:
                if (
                    a.backing == b.backing
                    and a.base * a.unit_bytes < (b.base + b.capacity) * b.unit_bytes
                    and b.base * b.unit_bytes < (a.base + a.capacity) * a.unit_bytes
                ):
                    raise ValueError("shared RHS overlaps a member's private staging or accumulator")
    return maximum


def _closed_synchronous(body, project):
    """No asynchronous batch permission: one load, pair or readout per drain."""
    index = 0
    while index < len(body):
        item = body[index]
        if type(item) is Loop:
            _closed_synchronous(item.body, project)
            index += 1
            continue
        op = project(item)[1]
        end = index + 1
        if type(op) is Preload:
            if end >= len(body) or type(body[end]) is not Call or type(project(body[end])[1]) is not Compute:
                raise ValueError("synchronous shared RHS descriptor lost its adjacent compute")
            end += 1
        elif type(op) not in (Load, Store, TiledIntegerStore):
            raise ValueError("synchronous shared RHS has an uncompleted or unknown effect")
        if end >= len(body) or body[end] != Call(DRAIN, ()):
            raise ValueError("unknown epoch law requires original per-effect full completion")
        index = end + 1


def verify_dense_shared_rhs(
    kernel,
    problems,
    blocks,
    resources,
    storages,
    adapter,
    *,
    project_member,
    retention_law,
    dtypes,
    static_budget,
    external_buffers_disjoint,
    epoch_admission=None,
    epoch_law=None,
    output_readouts=None,
    readout_qualification_identities=None,
    selected_contracts=None,
    timeout_ms=5000,
):
    """Reconstruct actual pointers, pure epochs, full source covers and liveouts."""
    from .dense_panel_epochs import DensePanelEpochLaw

    if (
        not retention_law.admits(adapter)
        or external_buffers_disjoint is not True
        or not adapter.synchronous_completion
        or not adapter.exact_accumulator_continuation
    ):
        raise ValueError("shared RHS lacks retention, external alias or synchronous fallback qualification")
    maximum = shared_rhs_views(problems, storages, adapter)
    contracts, readouts, _ = _group_output_contracts(
        len(problems),
        selected_contracts,
        output_readouts,
        readout_qualification_identities,
        require_selected=False,
    )
    widths = tuple(DTYPE_BYTES[dtypes[2] if r is None else r.output_dtype] for r in readouts)
    first = problems[0]
    expected = (
        TensorArg("rhs", (first.k, first.n), dtypes[1], "read"),
        *(TensorArg(f"lhs{i}", (p.m, p.k), dtypes[0], "read") for i, p in enumerate(problems)),
        *(
            TensorArg(f"output{i}", (p.m, p.n), dtypes[2] if readouts[i] is None else readouts[i].output_dtype, "write")
            for i, p in enumerate(problems)
        ),
    )
    if (
        type(kernel) is not Kernel
        or check_structure(kernel)
        or kernel.args != expected
        or len(kernel.body) != len(problems) + 1
        or sum(1 for _ in _static_calls(kernel.body)) > static_budget
    ):
        raise ValueError("shared RHS changed its complete source/liveout roster or structural budget")
    asynchronous = epoch_admission is not None and type(epoch_law) is DensePanelEpochLaw and not epoch_law.unknown()
    if asynchronous:
        epoch_law.validate_for(adapter, epoch_admission)
    reports, totals = (
        [],
        dict(
            rhs_loads=0,
            requested_rhs_bytes=0,
            lhs_loads=0,
            requested_lhs_bytes=0,
            computes=0,
            stores=0,
            requested_output_bytes=0,
            dynamic_epoch_drains=0,
        ),
    )
    prefix = kernel.body[0]
    for owner, section in enumerate(kernel.body):
        if type(section) is not Loop or section.extent != 1 or not section.body:
            raise ValueError("shared RHS requires a closed prefix and one closed sequential member region")
        if not asynchronous:
            _closed_synchronous(section.body, project_member)
        for call in _static_calls(section.body):
            if call.instr == DRAIN:
                if call != Call(DRAIN, ()):
                    raise ValueError("shared RHS completion carries hidden operands")
                continue
            member, op = project_member(call)
            if (owner == 0 and (member is not None or type(op) is not Load or op.role != "weight")) or (
                owner > 0 and (type(member) is not int or member != owner - 1 or type(op) is Load and op.role != "lhs")
            ):
                raise ValueError("actual shared RHS effect escaped its prefix/member region")

    for index, (p, block, storage) in enumerate(zip(problems, blocks, storages, strict=True)):
        selected = replace(
            kernel,
            args=(
                TensorArg("lhs", (p.m, p.k), dtypes[0], "read"),
                TensorArg("rhs", (p.k, p.n), dtypes[1], "read"),
                TensorArg("output", (p.m, p.n), expected[1 + len(problems) + index].dtype, "write"),
            ),
            body=(prefix, kernel.body[index + 1]),
        )
        actual = {}

        def rename(body):
            result = []
            for item in body:
                if type(item) is Loop:
                    result.append(replace(item, body=rename(item.body)))
                elif item.instr == DRAIN:
                    result.append(item)
                else:
                    member, op = project_member(item)
                    names = {"rhs": "rhs", f"lhs{index}": "lhs", f"output{index}": "output"}
                    cloned = replace(
                        item,
                        args=tuple(
                            (name, Ptr(names.get(v.tensor, v.tensor), v.offset) if type(v) is Ptr else v)
                            for name, v in item.args
                        ),
                    )
                    actual[id(cloned)] = (member, op)
                    result.append(cloned)
            return tuple(result)

        selected = replace(selected, body=rename(selected.body))
        member_adapter = replace(adapter, project=lambda c: actual[id(c)][1])
        serial = _serial_projection(selected)
        bindings = []

        def projection(call):
            bindings.append((call, *actual[id(call)]))
            return actual[id(call)][1]

        sites = _project(serial, replace(adapter, project=projection), fold_singleton_loops=False)
        used = {"lhs": 0, "rhs": 0, "acc": 0}
        lhs_bytes = rhs_bytes = 0
        for site, (call, member, op) in zip(sites, bindings, strict=True):
            if any(not isinstance(v, (Expr, Ptr)) for _, v in call.args):
                raise ValueError("shared RHS has an unknown operand or implicit memory effect")
            ptrs = [v for _, v in call.args if type(v) is Ptr]
            spans = []
            if type(op) is Load:
                weight = op.role == "weight"
                width, stride = (p.rhs_bytes, p.n) if weight else (p.lhs_bytes, p.k)
                if op.gather is not None or ptrs != [
                    Ptr("rhs" if weight else "lhs", (as_expr(op.dram_row) * stride + op.dram_col) * width)
                ]:
                    raise ValueError("shared RHS load pointer differs from its exact original tensor")
                spans = [
                    (
                        "rhs" if weight else "lhs",
                        storage.rhs if weight else storage.lhs,
                        op.row,
                        op.rows,
                        op.cols,
                        width,
                    )
                ]
                payload = _requested_elements(site, op.rows, op.cols) * width
                if weight:
                    rhs_bytes += payload
                else:
                    lhs_bytes += payload
                if not weight or index == 0:
                    role = "rhs" if weight else "lhs"
                    totals[role + "_loads"] += site.instances
                    totals["requested_" + role + "_bytes"] += payload
            elif type(op) in (Store, TiledIntegerStore):
                if ptrs != [Ptr("output", (as_expr(op.dram_row) * p.n + op.dram_col) * widths[index])]:
                    raise ValueError("shared RHS output pointer/readout width changed")
                spans = [("acc", storage.accumulator, op.accumulator_row, op.rows, op.cols, p.accumulator_bytes)]
                totals["stores"] += site.instances
                totals["requested_output_bytes"] += _requested_elements(site, op.rows, op.cols) * widths[index]
            elif type(op) is Preload:
                if ptrs:
                    raise ValueError("shared RHS descriptor has hidden external memory")
                spans = [
                    ("rhs", storage.rhs, op.weight_row, op.weight_rows, op.cols, p.rhs_bytes),
                    ("acc", storage.accumulator, op.accumulator_row, op.rows, op.cols, p.accumulator_bytes),
                ]
            elif type(op) is Compute:
                if ptrs or site.ordinal == 0:
                    raise ValueError("shared RHS compute lost its adjacent member descriptor")
                previous = sites[site.ordinal - 1]
                if (
                    type(previous.operation) is not Preload
                    or previous.scopes != site.scopes
                    or previous.path[:-1] != site.path[:-1]
                    or previous.path[-1] + 1 != site.path[-1]
                ):
                    raise ValueError("shared RHS compute lost its adjacent member descriptor")
                spans = [("lhs", storage.lhs, op.input_row, op.rows, op.cols, p.lhs_bytes)]
                totals["computes"] += site.instances
            else:
                raise ValueError("shared RHS has unsupported actual effects")
            for ptr in ptrs:
                lo, hi = _range(ptr.offset, site.scopes)
                if lo < 0 or hi > maximum:
                    raise ValueError("shared RHS external pointer overflows selected index width")
            for role, view, row, rows, cols, width in spans:
                used[role] = max(used[role], _span(view, row, rows, cols, width, site.scopes, maximum))
        if lhs_bytes != p.m * p.k * p.lhs_bytes or rhs_bytes != p.k * p.n * p.rhs_bytes:
            raise ValueError("shared RHS must load each source exactly once across all output columns/members")
        if used["lhs"] + used["rhs"] > resources.operand_bytes or used["acc"] > resources.accumulator_bytes:
            raise ValueError("shared RHS actual combined live footprint exceeds selected capacity")
        contract = contracts[index]
        if contract is not None:
            contract.revalidate()
            _, member_adapter = contract.dense.bind(
                p,
                storage,
                member_adapter,
                dtypes=dtypes,
                hardware=contract.hardware_identity,
                policy=contract.numerical_policy,
                resources=resources,
                output_readout=readouts[index],
            )
        report = (
            verify_dense_retained_reduction(
                selected,
                p,
                block,
                resources,
                storage,
                member_adapter,
                epoch_admission,
                static_budget=static_budget,
                timeout_ms=timeout_ms,
                output_readout=readouts[index],
            )
            if asynchronous
            else verify_compact_dense(
                serial,
                p,
                resources,
                storage,
                member_adapter,
                static_budget=static_budget,
                timeout_ms=timeout_ms,
                output_readout=readouts[index],
            )
        )
        reports.append(report)

    def drains(body):
        return sum(s.extent * drains(s.body) if type(s) is Loop else int(s.instr == DRAIN) for s in body)

    totals["dynamic_epoch_drains"] = drains(kernel.body)
    return {
        **totals,
        "members": reports,
        "source_products": "checked",
        "complete_liveouts": "checked",
        "epoch_mode": "pure_phase_batches" if asynchronous else "per_effect_drains",
        "retained_rhs_scope": "complete_group_invocation",
        "retained_lhs_scope": "full_K_across_all_N_per_member_M_block",
        "target_execution": "not_established",
        "performance": "unknown",
    }
