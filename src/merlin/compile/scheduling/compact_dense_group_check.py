"""Check actual full-K shared-LHS groups through physical noninterference.

Every member has separate RHS and accumulator windows. Only after checking all
actual role footprints and external pointer bindings may the existing compact
checker project a member view. These proof views neither reset machine memory
nor execute a member separately. Original qualification remains required.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field, replace

from merlin.sched.ir import Call, Const, Expr, Kernel, Loop, Ptr, Scaled, Select, Sum, TensorArg, Var, as_expr
from merlin.sched.ir.kernel import DTYPE_BYTES, check_structure

from .block_schedule import Compute, Load, Preload, Store
from .compact_dense import DenseLoopAdapter, DenseLoopStorage, _static_calls
from .compact_verification import DenseVerificationUnavailable, _project, _requested_elements, verify_compact_dense


@dataclass(frozen=True)
class DenseLHSReuseLaw:
    """Original implementation premises; constructing this data grants nothing.

    The original qualifier covers the full admitted operand/selector domain,
    including every RHS load, preload, compute, store and implicit/configuration
    effect, independent-member interleaving, and omitted-load fault/observation
    effects. RTL capacity and a syntactic pointer do not establish these laws.
    """

    configuration_identity: str
    numerical_policy_identity: str
    adapter_qualification_identity: str
    qualification_identity: str
    lhs_unchanged_by_member_operations: bool = field(default=False, kw_only=True)
    independent_member_interleaving: bool = field(default=False, kw_only=True)
    suppressed_loads_no_faults: bool = field(default=False, kw_only=True)
    suppressed_loads_no_extra_observations: bool = field(default=False, kw_only=True)

    def __post_init__(self):
        if any(
            type(v) is not str or not v.strip()
            for v in (
                self.configuration_identity,
                self.numerical_policy_identity,
                self.adapter_qualification_identity,
                self.qualification_identity,
            )
        ) or any(
            type(v) is not bool
            for v in (
                self.lhs_unchanged_by_member_operations,
                self.independent_member_interleaving,
                self.suppressed_loads_no_faults,
                self.suppressed_loads_no_extra_observations,
            )
        ):
            raise ValueError("shared LHS requires explicit original identities and typed qualification premises")

    def record(self):
        self.__post_init__()
        return asdict(self)

    def admits(self, adapter):
        return (
            self.lhs_unchanged_by_member_operations
            and self.independent_member_interleaving
            and self.suppressed_loads_no_faults
            and self.suppressed_loads_no_extra_observations
            and (self.configuration_identity, self.numerical_policy_identity, self.adapter_qualification_identity)
            == (adapter.configuration_identity, adapter.numerical_policy_identity, adapter.qualification_identity)
        )


def _range(value, scopes):
    """Conservative bounds over the existing integer/select expression language."""
    if type(value) in (int, bool):
        return int(value), int(value)
    if type(value) is Const:
        return value.value, value.value
    if type(value) is Var:
        matches = [extent for _, name, extent in scopes if name == value.name]
        if len(matches) != 1:
            raise ValueError("shared LHS has an unbound or ambiguous index")
        return 0, matches[0] - 1
    if type(value) is Scaled:
        lo, hi = _range(value.expr, scopes)
        return min(lo * value.factor, hi * value.factor), max(lo * value.factor, hi * value.factor)
    if type(value) is Sum:
        values = [_range(v, scopes) for v in value.terms]
        return sum(v[0] for v in values), sum(v[1] for v in values)
    if type(value) is Select:
        lo, hi = _range(Var(value.var), scopes)
        branches = (
            (value.then,)
            if lo == hi == value.equals
            else ((value.other,) if not lo <= value.equals <= hi else (value.then, value.other))
        )
        values = [_range(v, scopes) for v in branches]
        return min(v[0] for v in values), max(v[1] for v in values)
    raise ValueError("shared LHS has an unsupported address expression")


def _span(view, row, rows, cols, width, scopes, maximum):
    start, last = _range(row, scopes)
    rlo, rhi = _range(rows, scopes)
    clo, chi = _range(cols, scopes)
    end = (last + rhi - 1) * view.unit_bytes + chi * width
    if (
        start < view.base
        or rlo <= 0
        or clo <= 0
        or chi * width > view.unit_bytes
        or end > (view.base + view.capacity) * view.unit_bytes
        or end > maximum
        or start * view.unit_bytes > maximum
    ):
        raise ValueError("actual shared-LHS operation leaves its assigned private role window")
    return end - view.base * view.unit_bytes


def _views(problems, storages, adapter, *, reject_aliases=True):
    if (
        type(storages) is not tuple
        or len(storages) != len(problems)
        or any(type(s) is not DenseLoopStorage or s.accumulator is None for s in storages)
    ):
        raise ValueError("shared LHS requires explicit member RHS and accumulator assignments")
    lhs = storages[0].lhs
    if any(s.lhs != lhs for s in storages):
        raise ValueError("shared LHS members do not have the identical assigned activation view")
    views = (lhs, *(s.rhs for s in storages), *(s.accumulator for s in storages))
    if reject_aliases and len({v.name for v in views}) != len(views):
        raise ValueError("shared LHS private role names must identify unique assigned windows")
    maximum = (1 << (adapter.index_bits - 1)) - 1
    for i, a in enumerate(views):
        a.__post_init__()
        if (a.base + a.capacity) * a.unit_bytes > maximum:
            raise ValueError("shared LHS private address exceeds its selected index width")
        for b in views[:i]:
            if (
                reject_aliases
                and a.backing == b.backing
                and (
                    a.base * a.unit_bytes < (b.base + b.capacity) * b.unit_bytes
                    and b.base * b.unit_bytes < (a.base + a.capacity) * a.unit_bytes
                )
            ):
                raise ValueError("shared LHS member private windows overlap through physical backing")
    return views, maximum


def _group_output_contracts(count, selected, readouts, qualifications, *, require_selected=True):
    """Bind every independent exit to its original source declaration."""
    from .dense_contracts import DenseIntegerReadout
    from .selected_contracts import SelectedSchedulingContracts, _parse

    contracts = selected if type(selected) is tuple else (selected,) * count
    readouts = (None,) * count if readouts is None else readouts
    qualifications = (None,) * count if qualifications is None else qualifications
    if (
        type(contracts) is not tuple
        or len(contracts) != count
        or type(readouts) is not tuple
        or len(readouts) != count
        or type(qualifications) is not tuple
        or len(qualifications) != count
        or any(c is not None and type(c) is not SelectedSchedulingContracts for c in contracts)
        or (type(selected) is tuple and any(c is None for c in contracts))
    ):
        raise ValueError("shared LHS exits require one complete original member contract roster")
    for contract, readout, qualification in zip(contracts, readouts, qualifications, strict=True):
        if readout is None:
            if qualification is not None:
                raise ValueError("shared LHS readout qualification lacks its selected output path")
            continue
        if (
            type(readout) is not DenseIntegerReadout
            or (require_selected and contract is None)
            or type(qualification) is not str
            or not qualification.strip()
        ):
            raise ValueError("each shared LHS readout needs original selected semantics and qualification")
        if contract is not None:
            original = _parse(contract.facts_bytes, contract.software_bytes, contract.target)[4]
            if original is None or original.output_readout != readout:
                raise ValueError("shared LHS member readout differs from its original source declaration")
        else:
            graph = readout.graph()
            if graph.input_bits != (32,) or graph.output_bits != 8 or readout.output_dtype != "i8":
                raise ValueError("conditional shared LHS readout requires exact unary i32 to i8 semantics")
    return contracts, readouts, qualifications


def _group_contract_record(selected):
    if selected is None:
        return None
    return tuple(c.record() for c in selected) if type(selected) is tuple else selected.record()


def verify_compact_dense_group(
    kernel,
    problems,
    resources,
    storages,
    adapter,
    *,
    project_member,
    retention_law,
    dtypes,
    static_budget,
    external_buffers_disjoint,
    timeout_ms=2000,
    max_members=8,
    selected_contracts=None,
    reservations=(),
    output_readouts=None,
    readout_qualification_identities=None,
):
    """Reconstruct actual effects, coverage, retention and each final output.

    Like the single-member verifier, this low-level checker may consume a
    caller-owned scalar readout. That conditional arithmetic result does not
    establish original source selection or physical readout qualification. The
    scheduling entry requires original selected contracts for every typed exit.
    """
    from merlin.transforms.dense_planning import DenseProblem, DenseResources

    if (
        type(kernel) is not Kernel
        or type(problems) is not tuple
        or type(max_members) is not int
        or not 2 <= len(problems) <= max_members <= 128
        or any(type(p) is not DenseProblem for p in problems)
        or type(resources) is not DenseResources
        or type(adapter) is not DenseLoopAdapter
        or type(retention_law) is not DenseLHSReuseLaw
        or not retention_law.admits(adapter)
        or not adapter.synchronous_completion
        or not adapter.exact_accumulator_continuation
        or external_buffers_disjoint is not True
        or type(timeout_ms) is not int
        or timeout_ms <= 0
        or type(static_budget) is not int
        or static_budget < 0
        or type(dtypes) is not tuple
        or len(dtypes) != 3
    ):
        raise ValueError("shared LHS checking needs original qualified synchronous typed contracts and bounded work")
    contracts, readouts, qualifications = _group_output_contracts(
        len(problems), selected_contracts, output_readouts, readout_qualification_identities, require_selected=False
    )
    retention_law.__post_init__()
    resources.__post_init__()
    first = problems[0]

    def common(p):
        return p.m, p.k, p.tile_m, p.tile_k, p.tile_n, p.lhs_bytes, p.rhs_bytes, p.accumulator_bytes, p.bias

    if any(common(p) != common(first) for p in problems) or first.bias:
        raise ValueError("shared LHS members have different source geometry, formats or initialization")
    if first.tiles[1] > resources.max_k_blocks:
        raise ValueError("shared LHS complete reduction does not fit the selected K panel domain")
    if tuple(DTYPE_BYTES.get(t) for t in dtypes) != (first.lhs_bytes, first.rhs_bytes, first.accumulator_bytes):
        raise ValueError("shared LHS boundary formats differ from the original arithmetic widths")
    output_dtypes = tuple(dtypes[2] if r is None else r.output_dtype for r in readouts)
    output_widths = tuple(DTYPE_BYTES[t] for t in output_dtypes)
    expected = (
        TensorArg("lhs", (first.m, first.k), dtypes[0], "read"),
        *(TensorArg(f"rhs{i}", (p.k, p.n), dtypes[1], "read") for i, p in enumerate(problems)),
        *(TensorArg(f"output{i}", (p.m, p.n), output_dtypes[i], "write") for i, p in enumerate(problems)),
    )
    work, nodes = [(kernel.body, 0)], 0
    while work:
        body, depth = work.pop()
        if type(body) is not tuple or depth > 32:
            raise ValueError("group actual program exceeds its bounded structural domain")
        for statement in body:
            nodes += 1
            if nodes > max(32, static_budget * 16):
                raise ValueError("group actual program exceeds its bounded structural domain")
            if type(statement) is Loop:
                work.append((statement.body, depth + 1))
            elif type(statement) is not Call:
                raise ValueError("group actual program has an unknown statement")
    if kernel.args != expected or kernel.attrs or check_structure(kernel):
        raise ValueError("actual shared LHS kernel does not retain its complete multi-buffer boundary")
    if sum(1 for _ in _static_calls(kernel.body)) > static_budget:
        raise ValueError("actual group exceeds its static instruction budget")
    views, maximum = _views(problems, storages, adapter)
    if any(a.nbytes > maximum for a in kernel.args):
        raise ValueError("group external buffer exceeds the selected index domain")

    def snapshot():
        return (
            kernel.digest(),
            tuple(asdict(p) for p in problems),
            asdict(resources),
            tuple(asdict(s) for s in storages),
            retention_law.record(),
            adapter.configuration_identity,
            adapter.numerical_policy_identity,
            adapter.qualification_identity,
            adapter.synchronous_completion,
            adapter.exact_accumulator_continuation,
            adapter.index_bits,
            tuple(None if r is None else asdict(r) for r in readouts),
            qualifications,
        )

    before = snapshot()

    def callbacks():
        return (adapter.encode, adapter.project, project_member) + tuple(
            callback
            for contract in contracts
            if contract is not None
            for callback in (contract.revalidate, contract.record)
        )

    frozen_callbacks = tuple(
        (getattr(f, "__self__", None), getattr(f, "__func__", f), getattr(getattr(f, "__func__", f), "__code__", None))
        for f in callbacks()
    )
    if any(code is None for _, _, code in frozen_callbacks):
        raise ValueError("group checker requires original Python callback implementations")

    def static_fresh():
        current = callbacks()
        if before != snapshot() or any(
            getattr(f, "__self__", None) is not owner
            or getattr(f, "__func__", f) is not target
            or target.__code__ is not code
            for f, (owner, target, code) in zip(current, frozen_callbacks, strict=True)
        ):
            raise ValueError("original group source, adapter or projection changed during checking")

    static_fresh()
    selection = _group_contract_record(selected_contracts)
    static_fresh()

    def fresh():
        static_fresh()
        if selected_contracts is not None:
            for contract in contracts:
                contract.revalidate()
            static_fresh()
            current_selection = _group_contract_record(selected_contracts)
            static_fresh()
            if current_selection != selection:
                raise ValueError("original group selected contracts changed during checking")

    fresh()
    bindings = []

    def projection(call):
        fresh()
        value = project_member(call)
        fresh()
        if type(value) is not tuple or len(value) != 2:
            raise ValueError("group projection requires actual member identity and an existing dense operation")
        member, operation = value
        if member is not None and (type(member) is not int or not 0 <= member < len(problems)):
            raise ValueError("actual group member identity is outside its original buffer roster")
        if (member is None) != (type(operation) is Load and operation.role == "lhs"):
            raise ValueError("only an actual shared activation load may omit member identity")
        bindings.append((call, member, operation))
        return operation

    # Bindings retain original call operands and external pointers. Their
    # singleton variables still need the original scopes for range checks;
    # folding only the projected operation would leave these pointers unbound.
    sites = _project(kernel, replace(adapter, project=projection), fold_singleton_loops=False)
    used = {v.name: 0 for v in views}
    lhs_loads = lhs_bytes = rhs_loads = rhs_bytes = stores = output_bytes = 0
    for site, (call, member, op) in zip(sites, bindings, strict=True):
        if any(not isinstance(value, (Expr, Ptr)) for _, value in call.args):
            raise ValueError("group instructions contain unknown operands or hidden memory effects")
        ptrs = [value for _, value in call.args if type(value) is Ptr]
        p, storage = (first, storages[0]) if member is None else (problems[member], storages[member])
        spans = []
        if type(op) is Load:
            if op.gather is not None or op.role not in ("lhs", "weight"):
                raise ValueError("group has an unsupported gathered or implicit load")
            view, width, name, stride = (
                (storage.lhs, p.lhs_bytes, "lhs", p.k)
                if op.role == "lhs"
                else (storage.rhs, p.rhs_bytes, f"rhs{member}", p.n)
            )
            offset = (as_expr(op.dram_row) * stride + op.dram_col) * width
            if ptrs != [Ptr(name, offset)]:
                raise ValueError("actual group load pointer does not identify the complete original input buffer")
            spans.append((view, op.row, op.rows, op.cols, width))
            payload = _requested_elements(site, op.rows, op.cols) * width
            if op.role == "lhs":
                lhs_loads += site.instances
                lhs_bytes += payload
            else:
                rhs_loads += site.instances
                rhs_bytes += payload
        elif type(op) is Store:
            if ptrs != [Ptr(f"output{member}", (as_expr(op.dram_row) * p.n + op.dram_col) * output_widths[member])]:
                raise ValueError("actual group store pointer does not identify the original member output buffer")
            spans.append((storage.accumulator, op.accumulator_row, op.rows, op.cols, p.accumulator_bytes))
            stores += site.instances
            output_bytes += _requested_elements(site, op.rows, op.cols) * output_widths[member]
        elif type(op) is Preload:
            if ptrs or op.weight_row is None or op.weight_rows is None:
                raise ValueError("group preload has an unsupported retained descriptor or external pointer")
            spans.extend(
                (
                    (storage.rhs, op.weight_row, op.weight_rows, op.cols, p.rhs_bytes),
                    (storage.accumulator, op.accumulator_row, op.rows, op.cols, p.accumulator_bytes),
                )
            )
        elif type(op) is Compute:
            if ptrs or site.ordinal == 0 or bindings[site.ordinal - 1][1] != member:
                raise ValueError("group compute does not consume its own actual adjacent descriptor")
            previous = sites[site.ordinal - 1]
            if (
                type(previous.operation) is not Preload
                or previous.scopes != site.scopes
                or (previous.path[:-1] != site.path[:-1] or previous.path[-1] + 1 != site.path[-1])
            ):
                raise ValueError("group compute does not consume its own actual adjacent descriptor")
            pre = previous.operation
            spans.extend(
                (
                    (storage.lhs, op.input_row, op.rows, op.cols, p.lhs_bytes),
                    (storage.rhs, pre.weight_row, op.cols, pre.cols, p.rhs_bytes),
                    (storage.accumulator, pre.accumulator_row, op.rows, pre.cols, p.accumulator_bytes),
                )
            )
        else:
            raise ValueError("unknown group operation or implicit effect")
        for ptr in ptrs:
            lo, hi = _range(ptr.offset, site.scopes)
            if lo < 0 or hi > maximum:
                raise ValueError("group external pointer offset overflows its selected index domain")
        for view, row, rows, cols, width in spans:
            used[view.name] = max(used[view.name], _span(view, row, rows, cols, width, site.scopes, maximum))
    if lhs_loads != first.tiles[0] * first.tiles[1] or lhs_bytes != first.m * first.k * first.lhs_bytes:
        raise ValueError("actual group does not load each complete logical activation tile exactly once")
    if used[views[0].name] + sum(used[s.rhs.name] for s in storages) > resources.operand_bytes or (
        sum(used[s.accumulator.name] for s in storages) > resources.accumulator_bytes
    ):
        raise ValueError("actual group exceeds the original combined private storage budget")
    # All foreign writes are confined to disjoint windows; every shared LHS
    # write remains in each proof view, preserving its global latest-write state.
    deadline = time.monotonic() + timeout_ms / 1000
    member_reports = []
    for index, (p, storage) in enumerate(zip(problems, storages, strict=True)):
        operations = {}
        selected = {id(call): (member, op) for call, member, op in bindings}

        def body(statements):
            result = []
            for statement in statements:
                if type(statement) is Loop:
                    nested = body(statement.body)
                    if nested:
                        result.append(replace(statement, body=nested))
                else:
                    member, op = selected[id(statement)]
                    if member is None or member == index:
                        rename = {f"rhs{index}": "rhs", f"output{index}": "output", "lhs": "lhs"}
                        call = replace(
                            statement,
                            args=tuple(
                                (name, Ptr(rename[value.tensor], value.offset) if type(value) is Ptr else value)
                                for name, value in statement.args
                            ),
                        )
                        operations[id(call)] = op
                        result.append(call)
            return tuple(result)

        view = replace(
            kernel,
            args=(
                expected[0],
                TensorArg("rhs", (p.k, p.n), dtypes[1], "read"),
                TensorArg("output", (p.m, p.n), output_dtypes[index], "write"),
            ),
            body=body(kernel.body),
        )
        remaining = int((deadline - time.monotonic()) * 1000)
        if remaining <= 0:
            raise DenseVerificationUnavailable("shared LHS verification budget exhausted")
        member_adapter = replace(adapter, project=lambda call: operations[id(call)])
        contract = contracts[index]
        if contract is not None:
            _, member_adapter = contract.dense.bind(
                p,
                storage,
                member_adapter,
                dtypes=dtypes,
                hardware=contract.hardware_identity,
                policy=contract.numerical_policy,
                resources=resources,
                reservations=reservations,
                output_readout=readouts[index],
            )
        member_reports.append(
            verify_compact_dense(
                view,
                p,
                resources,
                storage,
                member_adapter,
                static_budget=static_budget,
                timeout_ms=remaining,
                output_readout=readouts[index],
            )
        )
    fresh()
    return {
        "schema": "merlin.compact_dense_group.v1",
        "members": len(problems),
        "kernel_sha256": kernel.digest(),
        "static_commands": len(sites),
        "lhs_loads": lhs_loads,
        "rhs_loads": rhs_loads,
        "output_tiles": stores,
        "requested_lhs_bytes": lhs_bytes,
        "requested_rhs_bytes": rhs_bytes,
        "requested_output_bytes": output_bytes,
        "output_dtypes": output_dtypes,
        "output_readouts": tuple(None if r is None else r.identity for r in readouts),
        "readout_qualifications": qualifications,
        "readout_source_binding": "selected_original" if selected_contracts is not None else "caller_owned_conditional",
        "private_high_water_bytes": used,
        "member_realizations": member_reports,
        "retention_law": retention_law.record(),
        "scope": "conditional original qualified full-K LHS reuse; no target qualification or timing",
        "proof": "global actual role noninterference plus unchanged compact per-member coverage/latest-write proofs",
    }
