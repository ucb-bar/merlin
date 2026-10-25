"""Counted array-weight reuse with a conservative, source-bound proof projection.

Only a pure K,N,M descriptor/compute run is rewritten. The first M consume
captures the RHS; later consumes keep it. The unchanged source/epoch verifier
checks the original fresh-read stream, including continued scratchpad residency.
That stronger condition deliberately excludes reuse after overwriting the RHS.
The checking stream is never the emitted stream. Target array behavior remains
an independently selected premise, separate from storage and ordered commits.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from math import prod
from pathlib import Path

from merlin.sched.ir import Call, Const, Kernel, Loop, Var, check_structure
from merlin.sched.ir.expr import free_vars
from merlin.sched.ir.integer_domain import check_integer_domain
from merlin.sched.primitives import _bound_vars, _loop_at, _splice, _subst_call

from .block_schedule import Compute, Preload
from .dense_panel_pipeline import DRAIN, dense_retained_compute_nests


@dataclass(frozen=True)
class DenseArrayRetentionAdmission:
    hardware_identity: str
    numerical_policy_identity: str
    qualification_identity: str
    execution: str = "fresh_consume_sets_array_stay_preserves_pair_output_only"

    def __post_init__(self):
        if any(type(value) is not str or not value or len(value) > 512 for value in (
            self.hardware_identity, self.numerical_policy_identity, self.qualification_identity,
        )):
            raise ValueError("array retention requires independently selected source identities")
        if self.execution != "fresh_consume_sets_array_stay_preserves_pair_output_only":
            raise ValueError("array retention lacks the fresh/stay/output-descriptor execution premise")


def _owner_sha256():
    return sha256(Path(__file__).read_bytes()).hexdigest()


def _check_admission(adapter, admission):
    if type(admission) is not DenseArrayRetentionAdmission:
        raise ValueError("array retention needs a separate typed target admission")
    admission.__post_init__()
    if (admission.hardware_identity != adapter.configuration_identity
            or admission.numerical_policy_identity != adapter.numerical_policy_identity):
        raise ValueError("array retention differs from the original selected adapter")


def _rewrite(kernel, adapter):
    if type(kernel) is not Kernel or check_structure(kernel):
        raise ValueError("array retention needs well-formed original counted IR")
    check_integer_domain(kernel, adapter.index_bits)

    def project(call):
        # Completion remains in place and cannot be mistaken for an array op.
        return None if call.instr == DRAIN else adapter.project(call)

    streams = dense_retained_compute_nests(kernel, replace(adapter, project=project))
    actual, changes = kernel, []
    used = _bound_vars(kernel.body) | {arg.name for arg in kernel.args}
    for prefix, names, order, extents in streams:
        if order != ["k", "n", "m"]:
            raise ValueError("array retention requires actual source K,N,M compute order")
        path = (*prefix, names["k"], names["n"], names["m"])
        loop = _loop_at(actual.body, path)
        if loop.extent < 2:
            continue
        preload_call, compute_call = loop.body
        if any(call != Call(call.instr, call.args) for call in loop.body):
            raise ValueError("array retention cannot erase additional call or configuration effects")
        preload, compute = map(adapter.project, loop.body)
        if (type(preload) is not Preload or type(compute) is not Compute
                or preload.weight_row is None or preload.weight_block is None
                or preload.weight_rows is None or compute.fresh_weights is not True
                or compute.input_gather is not None):
            raise ValueError("array retention needs an adjacent fresh source descriptor/compute pair")
        retained_fields = (preload.weight_row, *preload.weight_block, preload.weight_rows, preload.cols,
                           compute.input_block[1], compute.cols)
        if any(names["m"] in free_vars(value) for value in retained_fields):
            raise ValueError("array RHS identity, depth and width must be invariant across M")
        # An encoder must not drop extra target operands or metadata. Only the
        # two explicit semantic fields below may change in the retained pair.
        if adapter.encode(preload) != preload_call or adapter.encode(compute) != compute_call:
            raise ValueError("array retention requires complete actual encoder/projector round trips")
        kept = (replace(preload, weight_row=None, weight_block=None), replace(compute, fresh_weights=False))
        encoded = tuple(adapter.encode(operation) for operation in kept)
        if any(type(call) is not Call or call != Call(call.instr, call.args) for call in encoded):
            raise ValueError("retained encoding introduced unknown call effects")
        if tuple(adapter.project(call) for call in encoded) != kept:
            raise ValueError("target adapter does not project actual retained-weight operands")

        serial = 0
        while (first_name := names["m"] + "_array_first_" + str(serial)) in used:
            serial += 1
        used.add(first_name)
        first = Loop(first_name, 1, tuple(_subst_call(call, names["m"], Const(0)) for call in loop.body))
        following = Loop(loop.var, loop.extent - 1,
                         tuple(_subst_call(call, loop.var, Var(loop.var) + 1) for call in encoded))
        actual = replace(actual, body=_splice(actual.body, path, (first, following)))
        # The enclosing multiplicity is reconstructed from current real loops,
        # including panel and completion scopes. It is not a physical limit.
        multiplicity = prod(_loop_at(kernel.body, path[:i]).extent for i in range(1, len(path)))
        changes.append({"path": path, "m_extent": loop.extent, "fresh_pairs": multiplicity,
                        "retained_pairs": multiplicity * (loop.extent - 1),
                        "first_m": 0, "remaining_m": "original M = counted M + 1",
                        "retention_scope": "one pure M run; no cross-run or cross-epoch reuse"})
    if not changes:
        raise ValueError("array retention found no reusable M run")
    if check_structure(actual):
        raise ValueError("array retention changed counted binder scope")
    check_integer_domain(actual, adapter.index_bits)
    return actual, tuple(changes)


@dataclass(frozen=True)
class DenseArrayRetentionProposal:
    kernel: Kernel
    checking_kernel: Kernel
    admission: DenseArrayRetentionAdmission
    changes: tuple
    actual_sha256: str
    checking_sha256: str
    owner_sha256: str

    def revalidate(self, adapter, admission):
        _check_admission(adapter, admission)
        if (self.admission != admission or self.kernel.digest() != self.actual_sha256
                or self.checking_kernel.digest() != self.checking_sha256
                or _owner_sha256() != self.owner_sha256):
            raise ValueError("array retention source, actual commands, law or proof owner changed")
        actual, changes = _rewrite(self.checking_kernel, adapter)
        if actual != self.kernel or changes != self.changes:
            raise ValueError("actual array-retained stream differs from its exact source projection")


def propose_dense_array_retention(kernel, adapter, admission):
    """Emit a counted first/remaining M split; supply no source or timing waiver."""
    _check_admission(adapter, admission)
    actual, changes = _rewrite(kernel, adapter)
    return DenseArrayRetentionProposal(actual, kernel, admission, changes, actual.digest(), kernel.digest(),
                                       _owner_sha256())


def verify_dense_array_retention(
    proposal, problem, block, resources, storage, adapter, base_admission, *, static_budget,
    timeout_ms=5000, output_readout=None, bias_source=None,
):
    """Conjoin array semantics with unchanged complete source/epoch checks.

    The projection reads the same unmodified RHS scratch cells on every M,
    which is a conservative strengthening of reading their captured array
    values. No load, configuration, observer or completion crosses the pure
    M run; the base checker still proves all concurrent DMA/read disjointness.
    A proposal is not independently executable evidence or target admission.
    """
    from .dense_panel_pipeline import DenseRetainedReductionAdmission, verify_dense_retained_reduction

    if type(proposal) is not DenseArrayRetentionProposal:
        raise ValueError("array retention requires its exact source-custodied proposal")
    proposal.revalidate(adapter, proposal.admission)
    if type(static_budget) is not int or static_budget <= 0:
        raise ValueError("array retention needs the original finite static command budget")

    def size(body):
        return sum(size(s.body) if type(s) is Loop else 1 for s in body)

    def compute_count(body, multiplicity=1):
        return sum(compute_count(s.body, multiplicity * s.extent) if type(s) is Loop else
                   multiplicity if s.instr != DRAIN and type(adapter.project(s)) is Compute else 0 for s in body)

    if size(proposal.kernel.body) > static_budget:
        raise ValueError("array-retained commands exceed the original static command budget")
    if type(base_admission) is DenseRetainedReductionAdmission:
        verify = verify_dense_retained_reduction
    else:
        # A retained-only public library need not expose the independent
        # segmented family. Do not import its owner unless it is requested.
        from .dense_segmented_pipeline import DenseSegmentedAdmission, verify_dense_segmented_pipeline

        if type(base_admission) is not DenseSegmentedAdmission:
            raise ValueError("array retention needs an existing typed complete source/epoch admission")
        verify = verify_dense_segmented_pipeline
    checked = verify(proposal.checking_kernel, problem, block, resources, storage, adapter, base_admission,
                     static_budget=static_budget, timeout_ms=timeout_ms,
                     output_readout=output_readout, bias_source=bias_source)
    proposal.revalidate(adapter, proposal.admission)
    retained = sum(c["retained_pairs"] for c in proposal.changes)
    return {"schema": "merlin.dense_array_retention.v1", "actual_kernel_sha256": proposal.actual_sha256,
            "checking_kernel_sha256": proposal.checking_sha256, "admission": asdict(proposal.admission),
            "changes": proposal.changes, "fresh_pairs": compute_count(proposal.kernel.body) - retained,
            "retained_pairs": retained,
            "source_and_epochs": checked, "projection": "checking only; never emitted",
            "scratchpad_rhs_retained": True, "machine_emission": "not_established", "timing": "UNKNOWN"}


def verify_dense_array_tile_loads(
    proposal, problem, block, resources, storage, adapter, base_admission, *, project, encode_band,
    static_budget, timeout_ms=5000, max_static_nodes=4096, output_readout=None, bias_source=None,
):
    """Conjoin final wide addressing, exact array reuse, source and epochs.

    Both actual and fresh streams are decoded from their final wide calls.
    Every actual physical band must still match the original source pitch,
    local stride, width and element layout. Expanding the fresh stream and
    applying the exact counted rewrite must reproduce the expanded actual
    stream. Only those expanded, conservative fresh reads enter the complete
    source/hazard checker; emission retains the actual wide/STAY commands.
    """
    from .panel_tile_loads import expand_dense_panel_tile_loads

    if type(proposal) is not DenseArrayRetentionProposal:
        raise ValueError("wide array retention requires its exact source-custodied proposal")
    proposal.revalidate(adapter, proposal.admission)
    actual, actual_mapping = expand_dense_panel_tile_loads(
        proposal.kernel, problem, storage, adapter, project=project, encode_band=encode_band,
        max_static_nodes=max_static_nodes,
    )
    checking, checking_mapping = expand_dense_panel_tile_loads(
        proposal.checking_kernel, problem, storage, adapter, project=project, encode_band=encode_band,
        max_static_nodes=max_static_nodes,
    )
    if (actual_mapping["actual_mapping_sha256"] != checking_mapping["actual_mapping_sha256"]
            or actual_mapping["physical_load_commands"] != checking_mapping["physical_load_commands"]
            or actual_mapping["expanded_logical_tile_bands"] != checking_mapping["expanded_logical_tile_bands"]):
        raise ValueError("actual array reuse changed final physical wide-load mapping or order")
    expanded = propose_dense_array_retention(checking, adapter, proposal.admission)
    if expanded.kernel != actual:
        raise ValueError("final expanded wide bands do not preserve the exact counted array reuse stream")
    report = verify_dense_array_retention(
        expanded, problem, block, resources, storage, adapter, base_admission,
        static_budget=static_budget, timeout_ms=timeout_ms,
        output_readout=output_readout, bias_source=bias_source,
    )
    proposal.revalidate(adapter, proposal.admission)
    return {"schema": "merlin.dense_array_tile_loads.v1", "actual_kernel_sha256": proposal.actual_sha256,
            "fresh_wide_kernel_sha256": proposal.checking_sha256,
            "expanded_actual_kernel_sha256": actual.digest(), "expanded_checking_kernel_sha256": checking.digest(),
            "tile_load_projection": actual_mapping, "fresh_tile_load_projection": checking_mapping,
            "array_retention": report, "emission": "original final wide/STAY kernel only",
            "machine_emission": "not_established", "timing": "UNKNOWN"}
