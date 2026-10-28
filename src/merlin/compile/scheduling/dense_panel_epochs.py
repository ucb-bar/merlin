"""Fact-backed completion epochs for the two-window dense panel pipeline.

``build_dense_panel_pipeline`` closes every K step of an output block with a full device drain: the
current window's compute and the next load into the other window share one epoch, and the next step
waits until both have completed. That drain orders three kinds of pairs: a compute after the load
into its rows (read after write), a load into a window after the computes that read it (write after
read), and successive accumulator updates of one output block. When the selected target already
orders all three, those drains order nothing the hardware does not. ``DensePanelEpochLaw`` states the
premises flag by flag. Each flag is a claim about the selected hardware that its command-ordering
facts must back; a flag left False is UNKNOWN and keeps every drain it would remove.

``propose_dense_panel_epochs`` removes only drains inside one output block: between the K steps of
its two-window stream and, with accumulator result visibility, between its final compute and its
readout epoch. Every remaining drain is an original one (initialization, output-block boundaries,
the closing completion before host-visible results). No data call, operand or issue order changes:
the serial projection equals the checked original's.

``schedule_dense_panel_epochs`` consumes the checked original two-window result. For every removed
boundary it proves that each load of the earlier step is read by the next step, so a later load into
the same rows is ordered after the earlier one through that reader (read after write, then write
after read) even where loads complete out of order. It bounds every merged epoch by the selected
issued budgets and returns a result whose emitters invoke the completion callback only for the
remaining drains. It grants no hardware qualification and makes no performance claim.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from itertools import product
from math import prod
from pathlib import Path

from merlin.sched.ir import Call, Const, Kernel, Loop, Scaled, Select, Sum, Var, check_structure

from .block_schedule import Compute, Load, Preload, Store
from .compact_dense import _digest
from .dense_contracts import TiledIntegerStore
from .dense_panel_pipeline import (
    DRAIN,
    DensePanelOverlapAdmission,
    DensePanelPipelineResult,
    _calls,
    _serial_projection,
)

COMPLETION = Call(DRAIN, ())
#: Premises every merged K stream needs; the readout flag additionally merges the readout epoch.
STREAM_FLAGS = (
    "local_read_after_write",
    "local_write_after_read",
    "ordered_accumulator_writes",
    "ordered_configuration_capture",
    "finite_queue_progress",
    "complete_device_drain",
    "exclusive_private_state",
)
READOUT_FLAGS = ("accumulator_result_visibility",)
EPOCH_FLAGS = (*STREAM_FLAGS[:3], *READOUT_FLAGS, *STREAM_FLAGS[3:])
_OPERANDS = frozenset((Load, Preload, Compute))
_READOUTS = frozenset((Store, TiledIntegerStore))


@dataclass(frozen=True)
class DensePanelEpochLaw:
    """Selected target ordering premises that replace drains inside one output block.

    ``local_read_after_write``: a compute or descriptor waits for the completion of an older load into
    overlapping operand rows. ``local_write_after_read``: a load waits for the completion of older
    computes and descriptors that read its rows. ``ordered_accumulator_writes``: accumulator updates of
    the compute queue complete in issue order. ``accumulator_result_visibility``: a readout waits for
    the older descriptor that wrote its accumulator rows. ``ordered_configuration_capture``: latched
    configuration is captured in its queue's issue order. ``finite_queue_progress``: finite queues
    backpressure issue and a descriptor/compute pair always progresses. ``complete_device_drain``: the
    completion callback is a real full device drain. ``exclusive_private_state``: the schedule owns its
    operand windows and accumulator rows for the invocation. Load completion order is not a premise:
    rewrites of one row are ordered through the step that reads it, which the scheduler proves.
    """

    hardware_identity: str
    numerical_policy_identity: str
    qualification_identity: str
    local_read_after_write: bool = False
    local_write_after_read: bool = False
    ordered_accumulator_writes: bool = False
    accumulator_result_visibility: bool = False
    ordered_configuration_capture: bool = False
    finite_queue_progress: bool = False
    complete_device_drain: bool = False
    exclusive_private_state: bool = False

    def __post_init__(self):
        identities = (self.hardware_identity, self.numerical_policy_identity, self.qualification_identity)
        if any(type(v) is not str or not v.strip() for v in identities) or any(
            type(getattr(self, flag)) is not bool for flag in EPOCH_FLAGS
        ):
            raise ValueError("dense panel epochs require typed identities and boolean ordering premises")

    def unknown(self, flags=EPOCH_FLAGS):
        """The named premises this law leaves UNKNOWN (False)."""
        return tuple(flag for flag in flags if not getattr(self, flag))

    def validate_for(self, adapter, admission):
        self.__post_init__()
        if (
            self.hardware_identity != adapter.configuration_identity
            or self.numerical_policy_identity != adapter.numerical_policy_identity
            or self.hardware_identity != admission.hardware_identity
            or self.numerical_policy_identity != admission.numerical_policy_identity
        ):
            raise ValueError("dense panel epoch law selects a different hardware or numerical contract")


def _is_epoch(statement):
    return (
        type(statement) is Loop
        and statement.extent == 1
        and bool(statement.body)
        and statement.body[-1] == COMPLETION
        and not any(call == COMPLETION for call in _calls(statement.body[:-1]))
    )


def _is_chain(statement):
    """An epoch, or a counted loop whose whole body is a sequence of chains."""
    if _is_epoch(statement):
        return True
    return (
        type(statement) is Loop
        and bool(statement.body)
        and all(type(child) is Loop and _is_chain(child) for child in statement.body)
    )


def _kinds(statement, adapter):
    return {type(adapter.project(call)) for call in _calls((statement,)) if call != COMPLETION}


def _count(statement, adapter, kinds):
    """Dynamic data calls of the projected kinds in one execution of ``statement``."""
    if type(statement) is not Loop:
        return int(statement != COMPLETION and type(adapter.project(statement)) in kinds)
    return statement.extent * sum(_count(child, adapter, kinds) for child in statement.body)


def _drains(body):
    """Dynamic completions in one execution of ``body``."""
    return sum(
        statement.extent * _drains(statement.body) if type(statement) is Loop else int(statement == COMPLETION)
        for statement in body
    )


def _epochs(statement):
    return 1 if _is_epoch(statement) else statement.extent * sum(_epochs(child) for child in statement.body)


def _strip(statement):
    """Drop the completion of every epoch of a chain; every scope and data call stays."""
    if _is_epoch(statement):
        return replace(statement, body=statement.body[:-1])
    return replace(statement, body=tuple(_strip(child) for child in statement.body))


def _plan(kernel, adapter, law, admission):
    """Deterministic structural proposal: (actual kernel, merged streams on the original kernel).

    A stream is a maximal run of sibling chains whose effects are operand loads and paired computes
    (one output block's K steps), ending in an epoch. It merges only when the law backs every stream
    premise and the merged epoch stays inside the selected issued budgets; otherwise its loops are
    searched for smaller streams. The readout epoch right after it joins under result visibility.
    """
    stream_ok = not law.unknown(STREAM_FLAGS)
    readout_ok = stream_ok and not law.unknown(READOUT_FLAGS)
    streams = []

    def operand_chain(statement):
        if not _is_chain(statement):
            return False
        kinds = _kinds(statement, adapter)
        return bool(kinds) and kinds <= _OPERANDS

    def walk(body, ancestry):
        output, index = [], 0
        while index < len(body):
            end = index
            while end < len(body) and operand_chain(body[end]):
                end += 1
            last = end - 1
            while last >= index and not _is_epoch(body[last]):
                last -= 1
            elements = tuple(body[index : last + 1])
            pairs = sum(_count(e, adapter, {Compute}) for e in elements)
            loads = sum(_count(e, adapter, {Load}) for e in elements)
            if (
                stream_ok
                and elements
                and sum(_epochs(e) for e in elements) > 1
                and pairs
                and pairs <= admission.max_issued_compute_pairs
                and loads <= admission.max_issued_loads
            ):
                following = body[last + 1] if last + 1 < len(body) else None
                readout = None
                if (
                    readout_ok
                    and _is_epoch(following)
                    and _kinds(following, adapter)
                    and _kinds(following, adapter) <= _READOUTS
                    and _count(following, adapter, _READOUTS) <= admission.max_issued_readouts
                ):
                    readout = following
                if readout is None:
                    merged = (*(_strip(e) for e in elements[:-1]), elements[-1])
                else:
                    merged = (*(_strip(e) for e in elements), readout)
                output.extend(merged)
                streams.append(
                    {
                        "ancestry": ancestry,
                        "elements": elements,
                        "readout": readout,
                        "compute_pairs": pairs,
                        "loads": loads,
                        "readouts": 0 if readout is None else _count(readout, adapter, _READOUTS),
                        "removed_drains": _drains(elements) + (0 if readout is None else 1) - 1,
                    }
                )
                index = last + 1 + (readout is not None)
                continue
            statement = body[index]
            if type(statement) is Loop and not _is_epoch(statement):
                statement = replace(
                    statement, body=walk(statement.body, (*ancestry, (statement.var, statement.extent)))
                )
            output.append(statement)
            index += 1
        return tuple(output)

    actual = replace(kernel, body=walk(kernel.body, ()))
    return actual, streams


def propose_dense_panel_epochs(kernel, adapter, law, admission):
    """Propose the actual kernel without the drains the law backs; checking must follow.

    Returns (kernel, records). Only completions inside one output block move: data calls, scopes and
    issue order are unchanged, so the serial projection equals the input's. Initialization epochs,
    output-block boundaries and streams over the issued budgets keep their drains.
    """
    if type(kernel) is not Kernel or check_structure(kernel) or type(law) is not DensePanelEpochLaw:
        raise ValueError("dense panel epochs need a closed counted kernel and a typed epoch law")
    if type(admission) is not DensePanelOverlapAdmission:
        raise ValueError("dense panel epochs need the selected two-window overlap admission")
    law.__post_init__()
    actual, streams = _plan(kernel, adapter, law, admission)
    return actual, [_record(s) for s in streams]


def _record(stream):
    instances = prod(extent for _, extent in stream["ancestry"])
    return {
        "scope": [var for var, _ in stream["ancestry"]],
        "instances": instances,
        "steps": sum(_epochs(e) for e in stream["elements"]),
        "readout_merged": stream["readout"] is not None,
        "compute_pairs_per_epoch": stream["compute_pairs"],
        "loads_per_epoch": stream["loads"],
        "readouts_per_epoch": stream["readouts"],
        "removed_dynamic_drains": stream["removed_drains"] * instances,
    }


# ------------------------------------------------------------------------- consumption proof
def _sites(body, local=()):
    """(call, local scopes) of every data call in issue order; scopes are (var, extent)."""
    for statement in body:
        if type(statement) is Loop:
            yield from _sites(statement.body, (*local, (statement.var, statement.extent)))
        elif statement != COMPLETION:
            yield statement, local


def _ends(element):
    """((first epoch, binding), (last epoch, binding)) of one chain; bindings fix chain variables."""
    if _is_epoch(element):
        return (element, {}), (element, {})
    (first, fb), _ = _ends(element.body[0])
    _, (last, lb) = _ends(element.body[-1])
    return (first, {**fb, element.var: ("const", 0)}), (last, {**lb, element.var: ("const", element.extent - 1)})


def _inner(element):
    """Consecutive dynamic epoch pairs inside one chain: (a, a binding, b, b binding, guards)."""
    if _is_epoch(element):
        return []
    var, body, here = element.var, element.body, ("sym", 0)
    out = [
        (a, {**ab, var: here}, b, {**bb, var: here}, guards) for child in body for a, ab, b, bb, guards in _inner(child)
    ]
    for left, right in zip(body, body[1:]):
        (a, ab), (b, bb) = _ends(left)[1], _ends(right)[0]
        out.append((a, {**ab, var: here}, b, {**bb, var: here}, ()))
    if element.extent > 1:  # iteration i's last epoch -> iteration i+1's first
        (a, ab), (b, bb) = _ends(body[-1])[1], _ends(body[0])[0]
        out.append((a, {**ab, var: here}, b, {**bb, var: ("sym", 1)}, (var,)))
    return out


def _boundaries(elements):
    out = [boundary for element in elements for boundary in _inner(element)]
    for left, right in zip(elements, elements[1:]):
        (a, ab), (b, bb) = _ends(left)[1], _ends(right)[0]
        out.append((a, ab, b, bb, ()))
    return out


def _chain_extents(elements):
    extents = {}

    def visit(statement):
        if not _is_epoch(statement):
            extents[statement.var] = statement.extent
            for child in statement.body:
                visit(child)

    for element in elements:
        visit(element)
    return extents


def _prove_stream(stream, adapter, storage, deadline, max_instances):
    """Each load of a removed boundary's earlier epoch is read inside the next epoch.

    The next epoch's local loops are enumerated (at most ``max_instances`` reads); everything else
    stays symbolic: enclosing output-block scopes, chain iterations and the earlier load's own loops.
    """
    import z3

    views = {"lhs": storage.lhs, "weight": storage.rhs}
    extents = _chain_extents(stream["elements"])
    proved = 0

    def expression(value, env):
        if type(value) is int:
            return z3.IntVal(value)
        if type(value) is Const:
            return z3.IntVal(value.value)
        if type(value) is Var:
            return env[value.name]
        if type(value) is Sum:
            return z3.Sum(*(expression(term, env) for term in value.terms))
        if type(value) is Scaled:
            return expression(value.expr, env) * value.factor
        if type(value) is Select:
            return z3.If(env[value.var] == value.equals, expression(value.then, env), expression(value.other, env))
        raise ValueError("unknown dense panel address expression")

    def base(binding, constraints):
        env = {}
        for var, extent in stream["ancestry"]:
            env[var] = z3.Int("outer:" + var)
            constraints.append(z3.And(env[var] >= 0, env[var] < extent))
        for var, (kind, value) in binding.items():
            if kind == "const":
                env[var] = z3.IntVal(value)
            else:
                symbol = z3.Int("chain:" + var)
                constraints.append(z3.And(symbol >= 0, symbol < extents[var]))
                env[var] = symbol + value
        return env

    def accesses(operation):
        if type(operation) is Load:
            if operation.gather is not None or operation.role not in views:
                raise ValueError("dense panel epochs need contiguous operand loads")
            return "write", views[operation.role], operation.row, operation.rows
        if type(operation) is Compute:
            return "read", storage.lhs, operation.input_row, operation.rows
        if type(operation) is Preload:
            if operation.weight_row is None:
                return None
            return "read", storage.rhs, operation.weight_row, operation.weight_rows
        raise ValueError("dense panel stream has an unknown effect")

    for a, a_binding, b, b_binding, guards in _boundaries(stream["elements"]):
        writes = []
        for call, local in _sites((a,)):
            access = accesses(adapter.project(call))
            if access is not None and access[0] == "write":
                writes.append((access, local))
        if not writes:
            continue
        constraints = []
        a_env = base(a_binding, constraints)
        b_env = base(b_binding, constraints)
        for var in guards:
            constraints.append(z3.Int("chain:" + var) <= extents[var] - 2)
        reads, count = [], 0
        for call, local in _sites((b,)):
            access = accesses(adapter.project(call))
            if access is None or access[0] != "read":
                continue
            for values in product(*(range(extent) for _, extent in local)):
                count += 1
                if count > max_instances:
                    raise ValueError("dense panel epoch consumption proof exceeds its read-instance budget")
                env = {**b_env, **{var: z3.IntVal(v) for (var, _), v in zip(local, values)}}
                _, view, row, rows = access
                low = expression(row, env) * view.unit_bytes
                reads.append((view.backing, low, low + expression(rows, env) * view.unit_bytes))
        for (_, view, row, rows), local in writes:
            env = dict(a_env)
            local_constraints = []
            for var, extent in local:
                env[var] = z3.Int("load:" + var)
                local_constraints.append(z3.And(env[var] >= 0, env[var] < extent))
            low = expression(row, env) * view.unit_bytes
            high = low + expression(rows, env) * view.unit_bytes
            byte = z3.Int("byte")
            uncovered = [
                z3.Or(byte < r_low, byte >= r_high) for backing, r_low, r_high in reads if backing == view.backing
            ]
            remaining = int((deadline - time.monotonic()) * 1000)
            if remaining <= 0:
                raise ValueError("dense panel epoch consumption proof budget exhausted")
            solver = z3.Solver()
            solver.set(timeout=remaining)
            solver.add(*constraints, *local_constraints, low <= byte, byte < high, *uncovered)
            if solver.check() != z3.unsat:
                raise ValueError(
                    "a load of a merged K step is not read by the next step; its rewrite would be "
                    "unordered without a drain"
                )
            proved += 1
    return proved


# ------------------------------------------------------------------------- checked owner
def _identity(original, kernel, law, adapter, storage):
    return _digest((original.kernel.digest(), kernel.digest(), asdict(law), repr(adapter), asdict(storage)))


def _owner():
    return sha256(Path(__file__).read_bytes()).hexdigest()


@dataclass(frozen=True)
class DensePanelEpochResult:
    """Checked actual program with fact-backed epochs; execution qualification stays with its owner."""

    original: DensePanelPipelineResult
    kernel: Kernel
    realization: dict
    law: DensePanelEpochLaw
    checked_identity: str
    owner_sha256: str
    _adapter: object
    _storage: object

    @property
    def admission(self):
        return self.original.admission

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
        expected, _ = _plan(self.original.kernel, self._adapter, self.law, self.original.admission)
        if (
            self.kernel != expected
            or self.kernel.digest() != self.realization["kernel_sha256"]
            or self.checked_identity != _identity(self.original, self.kernel, self.law, self._adapter, self._storage)
            or self.owner_sha256 != _owner()
        ):
            raise ValueError("dense panel epochs: original schedule, actual commands, law or owner changed")

    def materialize_scf(self):
        from .scf_commands import materialize_scf_commands

        self.revalidate()
        module = materialize_scf_commands(self.kernel)
        self.revalidate()
        return module

    def emit_llvm(self, builder, lower_call):
        """Emit the actual commands; ``lower_call`` sees only the remaining drains."""
        from .llvm_program import FnBuilderIndexAdapter, emit_index_program

        self.revalidate()
        receipt = emit_index_program(self.kernel, FnBuilderIndexAdapter(builder), lower_call)
        self.revalidate()
        return receipt

    def emit_index_program(self, control, lower_effect, *, complete_epoch):
        """Lower effects and the remaining full completions through separate callbacks."""
        from .llvm_program import emit_index_effect_program

        self.revalidate()
        receipt = emit_index_effect_program(
            self.kernel, control, lower_effect, complete_epoch=complete_epoch, completion_instruction=DRAIN
        )
        self.revalidate()
        return receipt


def schedule_dense_panel_epochs(
    result,
    problem,
    block,
    resources,
    storage,
    adapter,
    admission,
    law,
    *,
    static_budget=128,
    timeout_ms=5000,
    max_read_instances=4096,
):
    """Consume a checked two-window result and a typed epoch law; check the merged epochs.

    The original result already proved source cover, window disjointness and its per-epoch issued
    budgets. This adds: identical serial projection, the consumption proof for every removed
    boundary, separate operand and accumulator stores for every merged epoch, and the issued budgets
    of the merged epochs. A law with an UNKNOWN stream premise removes nothing.
    """
    if type(result) is not DensePanelPipelineResult or type(law) is not DensePanelEpochLaw:
        raise ValueError("dense panel epochs need an original checked panel result and a typed epoch law")
    result.revalidate()
    if (result.realization.get("schedule_ir") or {}).get("pipeline_kind") != "two_window":
        raise ValueError("dense panel epochs apply to the two-window panel pipeline")
    if type(admission) is not DensePanelOverlapAdmission or admission != result.admission:
        raise ValueError("dense panel epochs cannot change the original overlap admission")
    law.validate_for(adapter, admission)
    if block.k != 1:
        raise ValueError("dense panel epochs need the original one-K-slice panel block")
    if type(timeout_ms) is not int or timeout_ms <= 0 or type(max_read_instances) is not int or max_read_instances <= 0:
        raise ValueError("dense panel epochs need explicit positive checking budgets")
    deadline = time.monotonic() + timeout_ms / 1000
    actual, streams = _plan(result.kernel, adapter, law, admission)
    if check_structure(actual) or _serial_projection(actual) != _serial_projection(result.kernel):
        raise ValueError("dense panel epochs changed a data call, scope or issue order")
    if sum(1 for _ in _calls(actual.body)) > static_budget:
        raise ValueError("dense panel epochs exceed the original static command budget")
    if streams and storage.accumulator.backing in (storage.lhs.backing, storage.rhs.backing):
        raise ValueError("merged epochs need operand windows outside the accumulator store")
    proved = sum(_prove_stream(s, adapter, storage, deadline, max_read_instances) for s in streams)
    records = [_record(s) for s in streams]
    before, after = _drains(result.kernel.body), _drains(actual.body)
    unknown = law.unknown()
    report = {
        "schema": "merlin.dense_panel_epochs.v1",
        "kernel_sha256": actual.digest(),
        "source": result.realization["source"],
        "schedule_ir": result.realization["schedule_ir"],
        "input_schedule": result.realization,
        "admission": asdict(admission),
        "epoch_law": asdict(law),
        "unknown_premises": list(unknown),
        "dynamic_epoch_drains": after,
        "panel_epochs": {
            "input_dynamic_drains": before,
            "dynamic_drains": after,
            "removed_dynamic_drains": before - after,
            "streams": records,
            "consumption_proofs": proved,
            "kept": [
                "initialization epochs and every output-block boundary (issued budgets, accumulator reuse)",
                "the closing completion before host-visible results",
                *(
                    ["readout epochs: accumulator_result_visibility is UNKNOWN"]
                    if "accumulator_result_visibility" in unknown
                    else []
                ),
                *(
                    [f"every K-step drain: {', '.join(law.unknown(STREAM_FLAGS))} UNKNOWN"]
                    if law.unknown(STREAM_FLAGS)
                    else []
                ),
            ],
            "ordering": "read after write, write after read and accumulator order from the law; rewrites of one "
            "row ordered through the proved reader of the next step",
        },
        "qualification": "conditional selected ordering premises; no independent physical qualification",
        "target_execution": "not_established",
    }
    checked = DensePanelEpochResult(
        result, actual, report, law, _identity(result, actual, law, adapter, storage), _owner(), adapter, storage
    )
    checked.revalidate()
    return checked
