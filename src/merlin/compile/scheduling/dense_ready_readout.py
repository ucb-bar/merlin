"""Final-K readout frontiers with explicit causal accumulator completion.

Actual counted PRELOAD/COMPUTE pairs remain adjacent. Each original output store
follows its complete final K run in the same epoch; the full accumulator view
and all input staging epochs stay intact. The target must separately establish
producer retirement, pending-write read readiness, store snapshot/backpressure
and the final drain. This module supplies no cycle spacing or execution grant.
"""

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from pathlib import Path

from merlin.compile.scheduling.block_schedule import Compute, Load, Preload, Store
from merlin.compile.scheduling.compact_dense import _digest
from merlin.compile.scheduling.dense_panel_pipeline import (
    DRAIN,
    DenseReadyReadoutAdmission,
    _calls,
    _remap,
    _verify_dense_epochs,
)
from merlin.compile.scheduling.dense_segmented_pipeline import DenseSegmentedPipelineResult, _assignment
from merlin.compile.scheduling.loop_schedule import LoopTransform, propose_loop_transforms
from merlin.compile.scheduling.tile_load_coalescing import _linear, _plain
from merlin.sched.ir import Call, Kernel, Loop, check_structure
from merlin.sched.ir.expr import free_vars
from merlin.sched.ir.integer_domain import check_integer_domain
from merlin.sched.primitives import _loop_at, _splice


def _chain(statement, count):
    loops = []
    for index in range(count):
        if type(statement) is not Loop:
            raise ValueError("ready readout requires perfect counted compute and output domains")
        loops.append(statement)
        if index + 1 < count:
            if len(statement.body) != 1:
                raise ValueError("ready readout cannot move effects across an imperfect nest")
            statement = statement.body[0]
    return loops, statement.body


def _strip_store_epochs(statement):
    """Remove only singleton store completion wrappers, substituting their zero."""
    if type(statement) is Call:
        if statement == Call(DRAIN, ()):
            raise ValueError("ready readout encountered a drain outside its original epoch")
        return (statement,)
    if type(statement) is not Loop:
        raise ValueError("unknown ready readout statement")
    if statement.body and statement.body[-1] == Call(DRAIN, ()):
        if statement.extent != 1:
            raise ValueError("ready readout cannot erase a counted completion epoch")
        return tuple(s for child in _remap(statement.body[:-1], statement.var, None)
                     for s in _strip_store_epochs(child))
    return (replace(statement, body=tuple(s for child in statement.body for s in _strip_store_epochs(child))),)


def propose_dense_ready_readout(kernel, problem, adapter):
    """Propose actual M,N,K final reduction followed by each original store.

    Nested outer counted groups and primitive/output tails stay counted. Only a
    pure compute epoch immediately followed by one complete original store nest
    is fused. Partial store partitions, unrolled/imperfect compute nests and
    per-tile readouts decline rather than silently selecting another schedule.
    Complete source and actual epoch checks MUST follow this proposal.
    """
    if type(kernel) is not Kernel or check_structure(kernel):
        raise ValueError("ready readout needs original closed counted commands")
    check_integer_domain(kernel, adapter.index_bits)

    def kinds(statement):
        return {type(adapter.project(c)) for c in _calls((statement,)) if c != Call(DRAIN, ())}

    paths = []

    def discover(body, path=()):
        for index, statement in enumerate(body):
            if type(statement) is not Loop:
                continue
            if statement.body and statement.body[-1] == Call(DRAIN, ()):
                if (kinds(statement) == {Preload, Compute} and index + 1 < len(body)
                        and kinds(body[index + 1]) == {Store}):
                    paths.append((*path, statement.var))
            else:
                discover(statement.body, (*path, statement.var))

    discover(kernel.body)
    if not paths or len(paths) > 32:
        raise ValueError("ready readout needs 1..32 final pure compute/store regions")
    actual, records = kernel, []
    for path in paths:
        epoch = _loop_at(actual.body, path)
        if epoch.extent != 1 or len(epoch.body) != 2:
            raise ValueError("ready readout needs one complete final compute nest and its original drain")
        parent = _loop_at(actual.body, path[:-1]) if path[:-1] else None
        siblings = parent.body if parent is not None else actual.body
        index = next(i for i, s in enumerate(siblings) if s is epoch)
        store_tree = _strip_store_epochs(siblings[index + 1])
        if len(store_tree) != 1:
            raise ValueError("ready readout needs one complete store domain, not partial partitions")
        loops, pair = _chain(epoch.body[0], 3)
        stores, store_calls = _chain(store_tree[0], 2)
        if (len(pair) != 2 or len(store_calls) != 1
                or any(type(c) is not Call or not _plain(c) for c in (*pair, *store_calls))):
            raise ValueError("ready readout requires plain adjacent physical pairs and one original store")
        pre, compute = (adapter.project(c) for c in pair)
        store = adapter.project(store_calls[0])
        if type(pre) is not Preload or type(compute) is not Compute or type(store) is not Store:
            raise ValueError("ready readout cannot split descriptors or vary output numerical configuration")
        variables = {s.var for s in loops}
        names = {}
        for axis, coordinate in zip(("m", "k", "n"),
                                    (compute.input_block[0], compute.input_block[1], pre.weight_block[1]), strict=True):
            selected = variables & free_vars(coordinate)
            if len(selected) != 1:
                raise ValueError("ready readout needs independent original M/K/N coordinates")
            names[axis] = selected.pop()
        if set(names.values()) != variables or variables & free_vars(pre.weight_block[0]) != {names["k"]}:
            raise ValueError("ready readout reduction coordinates disagree")
        extents = {s.var: s.extent for s in loops}
        store_variables = {s.var for s in stores}
        store_names = {}
        for axis, coordinate in (("m", store.dram_row), ("n", store.dram_col)):
            selected = store_variables & free_vars(coordinate)
            if len(selected) != 1:
                raise ValueError("ready readout store coordinates are not independent original M/N axes")
            store_names[axis] = selected.pop()
        if set(store_names.values()) != store_variables or any(
            next(s.extent for s in stores if s.var == store_names[a]) != extents[names[a]] for a in ("m", "n")
        ):
            raise ValueError("ready readout store domain differs from the complete final compute output domain")
        renamed = store_calls
        for axis in ("m", "n"):
            renamed = _remap(renamed, store_names[axis], names[axis])
        renamed_store = adapter.project(renamed[0])
        # Exact original scalar descriptors, including piecewise output tails.
        # Only affine logical coordinates need algebraic normalization.
        if (pre.accumulator_row != renamed_store.accumulator_row or pre.rows != renamed_store.rows
                or pre.cols != renamed_store.cols
                or _linear(renamed_store.dram_row) != _linear(compute.input_block[0] * problem.tile_m)
                or _linear(renamed_store.dram_col) != _linear(pre.weight_block[1] * problem.tile_n)
                or names["k"] in (free_vars(pre.accumulator_row) | free_vars(pre.rows) | free_vars(pre.cols))):
            raise ValueError("ready readout does not match the exact original producer ACC/output footprint")
        if adapter.encode(renamed_store) != renamed[0] or any(adapter.encode(adapter.project(c)) != c for c in pair):
            raise ValueError("ready readout requires complete descriptor/effect roundtrips")
        current = [next(a for a, name in names.items() if name == s.var) for s in loops]
        requests = []
        for position, axis in enumerate(("m", "n", "k")):
            cursor = current.index(axis)
            while cursor > position:
                request = LoopTransform("reorder", (*path, *(names[a] for a in current[:cursor])))
                actual = propose_loop_transforms(actual, (request,), index_bits=adapter.index_bits)
                requests.append(request)
                current[cursor - 1], current[cursor] = current[cursor], current[cursor - 1]
                cursor -= 1
        epoch = _loop_at(actual.body, path)
        (m, n, k), _ = _chain(epoch.body[0], 3)
        frontier = replace(m, body=(replace(n, body=(k, *renamed)),))
        replacement = replace(epoch, body=(frontier, Call(DRAIN, ())))
        parent = _loop_at(actual.body, path[:-1]) if path[:-1] else None
        siblings = parent.body if parent is not None else actual.body
        rewritten = (*siblings[:index], replacement, *siblings[index + 2:])
        actual = (replace(actual, body=_splice(actual.body, path[:-1], (replace(parent, body=rewritten),)))
                  if parent is not None else replace(actual, body=rewritten))
        records.append(dict(path=path, loop_transforms=[asdict(r) for r in requests],
                            final_k_tiles=k.extent, output_tiles=m.extent * n.extent,
                            geometry="unchanged original ACC and operand views",
                            frontier="each original store immediately follows its complete final K paired run",
                            completion="one original full drain after the complete frontier"))
    if check_structure(actual):
        raise ValueError("ready readout changed original binder custody")
    check_integer_domain(actual, adapter.index_bits)
    return actual, records


@dataclass(frozen=True)
class DenseReadyReadoutCheckingProjection:
    """Exact deterministic transform transport, usable only after actual hazards."""

    original_kernel: Kernel
    problem: object
    canonical_load_runs: bool = False

    def checking_kernel(self, actual, adapter, storage):
        expected, regions = propose_dense_ready_readout(self.original_kernel, self.problem, adapter)
        if type(self.canonical_load_runs) is not bool:
            raise ValueError("explicit exact ready-readout load representation required")
        canonical_changes = []
        if self.canonical_load_runs:
            from merlin.compile.scheduling.panel_tile_loads import _canonical_load_runs

            def original_load(call):
                if call == Call(DRAIN, ()):
                    return None
                operation = adapter.project(call)
                return operation if type(operation) is Load else None

            expected, canonical_changes = _canonical_load_runs(expected, original_load)
        if actual != expected or actual.digest() != expected.digest():
            raise ValueError("ready readout actual commands differ from the exact original transformation")
        return self.original_kernel, dict(
            rule="exact_final_k_producer_store_frontier_after_actual_hazards",
            original_kernel_sha256=self.original_kernel.digest(), actual_kernel_sha256=actual.digest(),
            regions=regions, canonical_load_runs=canonical_changes,
            geometry="no private allocation, offset or capacity changed",
            correspondence="same complete counted products/stores; only independent outputs interchanged",
            premise="all overlapping producer instances precede each store; no later ACC reuse in epoch",
            emission="actual frontier commands, never the checking projection",
        )


@dataclass(frozen=True)
class DenseReadyReadoutResult:
    original: DenseSegmentedPipelineResult
    kernel: Kernel
    realization: dict
    admission: DenseReadyReadoutAdmission
    problem: object
    checked_identity: str
    owner_sha256: str
    _adapter: object
    _storage: object

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
        expected, _ = propose_dense_ready_readout(self.original.kernel, self.problem, self._adapter)
        if (self.kernel != expected or self.kernel.digest() != self.realization["kernel_sha256"]
                or self.checked_identity != _digest((self.original.kernel.digest(), self.kernel.digest(),
                                                    asdict(self.admission), asdict(self.problem),
                                                    repr(self._adapter), asdict(self._storage)))
                or sha256(Path(__file__).read_bytes()).hexdigest() != self.owner_sha256):
            raise ValueError("ready readout original source, actual commands, capability or owner changed")

    def materialize_scf(self):
        from merlin.compile.scheduling.scf_commands import materialize_scf_commands

        self.revalidate()
        module = materialize_scf_commands(self.kernel)
        self.revalidate()
        return module

    def emit_llvm(self, builder, lower_call):
        from merlin.compile.scheduling.llvm_program import FnBuilderIndexAdapter
        from merlin.compile.scheduling.scf_commands import emit_scf_commands

        module = self.materialize_scf()
        report = emit_scf_commands(module, FnBuilderIndexAdapter(builder), lower_call)
        self.revalidate()
        return report


def schedule_dense_ready_readout(result, problem, block, resources, storage, adapter, admission, *,
                                 static_budget=128, timeout_ms=5000):
    """Consume an original checked segmented result and separate causal law."""
    if type(result) is not DenseSegmentedPipelineResult or type(admission) is not DenseReadyReadoutAdmission:
        raise ValueError("ready readout requires its original checked segmented owner and separate typed capability")
    result.revalidate()
    _assignment(problem, block, resources, storage)
    original = result.admission
    bound = replace(original.overlap,
                    max_issued_compute_pairs=min(original.overlap.max_issued_compute_pairs,
                                                 original.reduction.max_issued_compute_pairs),
                    max_issued_loads=min(original.overlap.max_issued_loads, original.reduction.max_issued_loads),
                    max_issued_readouts=min(original.overlap.max_issued_readouts,
                                           original.reduction.max_issued_readouts))
    admission.validate_for(adapter, bound)
    actual, changes = propose_dense_ready_readout(result.kernel, problem, adapter)
    report = _verify_dense_epochs(
        actual, problem, resources, storage, adapter, bound,
        static_budget=static_budget, timeout_ms=timeout_ms,
        output_readout=result.output_readout, bias_source=result.bias_source,
        ordered_accumulator_commits=True, source_phase_fission=True,
        readout_compute_admission=admission,
        readout_source_projection=DenseReadyReadoutCheckingProjection(result.kernel, problem),
    )
    report.update(schema="merlin.dense_ready_readout.v1", transformations=changes,
                  original_schedule=result.realization, source_identity=result.source_identity,
                  additional_capability=asdict(admission), schedule_ir=result.realization["schedule_ir"])
    checked = DenseReadyReadoutResult(
        result, actual, report, admission, problem,
        _digest((result.kernel.digest(), actual.digest(), asdict(admission), asdict(problem),
                 repr(adapter), asdict(storage))), sha256(Path(__file__).read_bytes()).hexdigest(), adapter, storage,
    )
    checked.revalidate()
    return checked


def admit_ready_readout_tile_loads(original, proposal, problem, block, resources, storage, adapter, *,
                                   project, encode_band, static_budget=128, timeout_ms=5000,
                                   max_static_nodes=4096):
    """Compose final physical wide bands with the SAME causal frontier proof.

    Every actual band is decoded and all actual epochs are verified before the
    exact original counted source representation is used. Emission retains the
    wide commands. No old typed consumer or execution admission is broadened.
    """
    from merlin.compile.scheduling.panel_tile_loads import _admit_dense_tile_loads

    if type(original) is not DenseReadyReadoutResult:
        raise ValueError("wide causal readout requires its exact original checked result")
    original.revalidate()
    if problem != original.problem or storage != original._storage or adapter != original._adapter:
        raise ValueError("wide causal readout cannot change original problem, storage or adapter")
    components = original.original.admission
    bound = replace(components.overlap,
                    max_issued_compute_pairs=min(components.overlap.max_issued_compute_pairs,
                                                 components.reduction.max_issued_compute_pairs),
                    max_issued_loads=min(components.overlap.max_issued_loads, components.reduction.max_issued_loads),
                    max_issued_readouts=min(components.overlap.max_issued_readouts,
                                           components.reduction.max_issued_readouts))
    projection = DenseReadyReadoutCheckingProjection(original.original.kernel, problem, canonical_load_runs=True)

    def verifier(actual, p, b, r, s, a, admission, **options):
        if admission != original.admission:
            raise ValueError("wide addressing cannot change the causal readout capability")
        _assignment(p, b, r, s)
        return _verify_dense_epochs(
            actual, p, r, s, a, bound, **options,
            ordered_accumulator_commits=True, source_phase_fission=True,
            readout_compute_admission=admission, readout_source_projection=projection,
        )

    return _admit_dense_tile_loads(
        original, proposal, problem, block, resources, storage, adapter,
        verifier=verifier, project=project, encode_band=encode_band,
        static_budget=static_budget, timeout_ms=timeout_ms, max_static_nodes=max_static_nodes,
    )
