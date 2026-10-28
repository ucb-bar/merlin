"""Counted output pairs with explicit disjoint ACC readout/compute lifetimes.

This pass does not derive an execution grant from spare accumulator capacity.
The target owner supplies an additional snapshot/backpressure/completion law;
the unchanged complete source checker and extended epoch checker validate the
actual commands. Only a final common full drain permits either window reuse.
"""

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from pathlib import Path

from merlin.compile.scheduling.block_schedule import Compute, Load, Preload, Store
from merlin.compile.scheduling.compact_dense import _digest
from merlin.compile.scheduling.dense_panel_pipeline import (
    DRAIN,
    DenseReadoutComputeAdmission,
    _calls,
    _remap,
    _verify_dense_epochs,
)
from merlin.compile.scheduling.dense_segmented_pipeline import DenseSegmentedPipelineResult, _assignment
from merlin.sched.ir import Call, Const, Kernel, Loop, Scaled, Select, Sum, Var, check_structure


def propose_dense_readout_overlap(kernel, adapter, storage, *, group_path, accumulator_rows):
    """Pair an actual counted output loop, retaining an odd final group.

    The second group gets a distinct ACC window. The first group's final stores
    move past only the second group's completed staging, into its first compute
    epoch. PRELOAD/COMPUTE adjacency and every existing operand drain remain.
    Full source and temporal verification are REQUIRED after this proposal.
    """
    if (
        type(kernel) is not Kernel
        or check_structure(kernel)
        or type(group_path) is not tuple
        or not group_path
        or any(type(v) is not str for v in group_path)
        or type(accumulator_rows) is not int
        or accumulator_rows <= 0
        or storage.accumulator is None
        or 2 * accumulator_rows > storage.accumulator.capacity
        or accumulator_rows % storage.accumulator.alignment
    ):
        raise ValueError("readout pairing needs a counted source loop and two aligned owned ACC windows")
    names = {arg.name for arg in kernel.args}

    def collect(body):
        for s in body:
            if type(s) is Loop:
                names.add(s.var)
                collect(s.body)

    collect(kernel.body)

    def fresh(prefix):
        while prefix in names:
            prefix += "_"
        names.add(prefix)
        return prefix

    def renamed(body, suffix):
        result = []
        for s in body:
            if type(s) is Loop:
                name = fresh(s.var + suffix)
                result.append(replace(s, var=name, body=renamed(_remap(s.body, s.var, name), suffix)))
            else:
                result.append(s)
        return tuple(result)

    def kinds(body):
        return {type(adapter.project(c)) for c in _calls(body) if c != Call(DRAIN, ())}

    def stores_only(s):
        body = (s,)
        return kinds(body) == {Store}

    def data(body):
        result = []
        for s in body:
            if s == Call(DRAIN, ()):
                continue
            if type(s) is Loop:
                inner = data(s.body)
                result.extend(inner if s.extent == 1 and s.body[-1] == Call(DRAIN, ()) else (replace(s, body=inner),))
            else:
                result.append(s)
        return tuple(result)

    def shifted_acc(body):
        result = []
        for s in body:
            if type(s) is Loop:
                result.append(replace(s, body=shifted_acc(s.body)))
            elif s == Call(DRAIN, ()):
                result.append(s)
            else:
                op = adapter.project(s)
                if type(op) in (Preload, Store):
                    if adapter.encode(op) != s:
                        raise ValueError("ACC relocation requires an exact original effect/descriptor roundtrip")
                    op = replace(op, accumulator_row=op.accumulator_row + accumulator_rows)
                    encoded = adapter.encode(op)
                    if type(encoded) is not Call or adapter.project(encoded) != op:
                        raise ValueError("ACC relocation must roundtrip every complete physical descriptor")
                    result.append(encoded)
                else:
                    result.append(s)
        return tuple(result)

    changed = []

    def pair(loop):
        if loop.extent < 2 or not kinds(loop.body) <= {Load, Preload, Compute, Store}:
            raise ValueError("readout pairing requires at least two unbiased known dense output groups")
        period = fresh("readout_output_pair")
        first = renamed(_remap(loop.body, loop.var, period, 2, 0), "_readout_first")
        second = shifted_acc(renamed(_remap(loop.body, loop.var, period, 2, 1), "_readout_second"))
        cut = len(first)
        while cut and stores_only(first[cut - 1]):
            cut -= 1
        if cut == len(first) or not cut or Store in kinds(first[:cut]):
            raise ValueError("output group must have one final pure readout suffix")
        deferred = data(first[cut:])
        merged = []
        found = False
        for s in second:
            if (
                not found
                and type(s) is Loop
                and s.extent == 1
                and s.body[-1] == Call(DRAIN, ())
                and Compute in kinds(s.body[:-1])
            ):
                if Store in kinds(s.body[:-1]):
                    raise ValueError("next compute epoch already contains an unqualified readout")
                s = replace(s, body=(*deferred, *s.body))
                found = True
            merged.append(s)
        if not found:
            raise ValueError("next output has no independently completed staging then compute epoch")
        pairs, tail = divmod(loop.extent, 2)
        result = [Loop(period, pairs, (*first[:cut], *merged))]
        if tail:
            result.extend(renamed(_remap(loop.body, loop.var, None, 1, 2 * pairs), "_readout_tail"))
        changed.append(
            dict(
                path=group_path,
                groups=loop.extent,
                pairs=pairs,
                odd_tail=tail,
                accumulator_rows=accumulator_rows,
                completion="mixed epoch full drain before either ACC window reuse",
            )
        )
        return tuple(result)

    def walk(body, path):
        matches = [i for i, s in enumerate(body) if type(s) is Loop and s.var == path[0]]
        if len(matches) != 1:
            raise ValueError("readout group path must identify exactly one actual counted loop")
        i = matches[0]
        s = body[i]
        replacement = pair(s) if len(path) == 1 else (replace(s, body=walk(s.body, path[1:])),)
        return (*body[:i], *replacement, *body[i + 1 :])

    actual = replace(kernel, body=walk(kernel.body, group_path))
    if check_structure(actual):
        raise ValueError("readout pairing changed original index scopes or pointer ownership")
    return actual, changed


def _affine_window_bounds(expr, domains):
    if type(expr) is Const:
        return expr.value, expr.value
    if type(expr) is int:
        return expr, expr
    if type(expr) is Var:
        if expr.name not in domains:
            raise ValueError("unbound original ACC window expression")
        return domains[expr.name]
    if type(expr) is Sum:
        values = [_affine_window_bounds(v, domains) for v in expr.terms]
        return sum(v[0] for v in values), sum(v[1] for v in values)
    if type(expr) is Scaled:
        values = _affine_window_bounds(expr.expr, domains)
        return tuple(sorted(v * expr.factor for v in values))
    if type(expr) is Select:
        if expr.var not in domains:
            raise ValueError("unbound original ACC window select")
        lo, hi = domains[expr.var]
        parts = []
        if lo <= expr.equals <= hi:
            parts.append(_affine_window_bounds(expr.then, {**domains, expr.var: (expr.equals, expr.equals)}))
        if lo != hi or lo != expr.equals:
            parts.append(_affine_window_bounds(expr.other, domains))
        return min(v[0] for v in parts), max(v[1] for v in parts)
    raise ValueError("unknown original ACC window expression; no checking projection")


@dataclass(frozen=True)
class DenseReadoutCheckingProjection:
    """Exact inverse of ONE verified counted output-pair transformation.

    No caller-supplied replacement kernel or digest grants equivalence. Actual
    commands are reconstructed and compared, and every original ACC access is
    independently bounded inside the injectively renamed source window. The
    epoch verifier may use this only AFTER actual movement/compute/store hazards.
    """

    original_kernel: Kernel
    group_path: tuple
    accumulator_rows: int
    canonical_load_runs: bool = False

    def checking_kernel(self, actual, adapter, storage):
        expected, changes = propose_dense_readout_overlap(
            self.original_kernel, adapter, storage, group_path=self.group_path, accumulator_rows=self.accumulator_rows
        )
        if type(self.canonical_load_runs) is not bool:
            raise ValueError("explicit exact load-run checking representation required")
        canonical_changes = []
        if self.canonical_load_runs:
            from merlin.compile.scheduling.panel_tile_loads import _canonical_load_runs

            def original_load(call):
                if call == Call(DRAIN, ()):
                    return None
                op = adapter.project(call)
                return op if type(op) is Load else None

            expected, canonical_changes = _canonical_load_runs(expected, original_load)
        if actual != expected:
            raise ValueError("readout checking projection does not reproduce all actual commands")

        footprints = 0
        view = storage.accumulator

        def walk(body, domains):
            nonlocal footprints
            for stmt in body:
                if type(stmt) is Loop:
                    walk(stmt.body, {**domains, stmt.var: (0, stmt.extent - 1)})
                elif stmt != Call(DRAIN, ()):
                    op = adapter.project(stmt)
                    if type(op) not in (Preload, Store, Compute, Load):
                        raise ValueError("unknown original effect in readout source projection")
                    if type(op) in (Preload, Store):
                        low, high = _affine_window_bounds(op.accumulator_row, domains)
                        row_low, row_high = _affine_window_bounds(op.rows, domains)
                        if low < view.base or high + row_high > view.base + self.accumulator_rows or row_low <= 0:
                            raise ValueError("original ACC footprint escapes one injectively renamed window")
                        footprints += 1

        walk(self.original_kernel.body, {})
        if not footprints:
            raise ValueError("source projection has no original ACC ownership footprints")
        return self.original_kernel, dict(
            rule="exact_counted_output_pair_inverse_after_actual_hazards",
            original_kernel_sha256=self.original_kernel.digest(),
            actual_kernel_sha256=actual.digest(),
            canonical_load_runs=canonical_changes,
            accumulator_footprints=footprints,
            accumulator_rows=self.accumulator_rows,
            paired_domains=changes,
            correspondence="each original index occurs once as 2*p or 2*p+1; odd final index retained",
            ownership="all original ACC accesses bounded in first window; second window is disjoint translation",
            commutation="only first stores cross next-group effects, with disjoint ACC and unchanged input lifetimes",
            numerical_boundary="original store/readout operands unchanged except injective private ACC translation",
            emission="actual paired commands unchanged",
        )


@dataclass(frozen=True)
class DenseReadoutOverlapResult:
    """Actual emitted program with original source custody and new explicit law."""

    original: DenseSegmentedPipelineResult
    kernel: Kernel
    realization: dict
    admission: DenseReadoutComputeAdmission
    group_path: tuple
    accumulator_rows: int
    checked_identity: str
    owner_sha256: str

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
        expected, _ = propose_dense_readout_overlap(
            self.original.kernel,
            self._adapter,
            self._storage,
            group_path=self.group_path,
            accumulator_rows=self.accumulator_rows,
        )
        if (
            self.kernel != expected
            or self.kernel.digest() != self.realization["kernel_sha256"]
            or _digest(
                (
                    self.original.kernel.digest(),
                    self.kernel.digest(),
                    asdict(self.admission),
                    repr(self._adapter),
                    asdict(self._storage),
                )
            )
            != self.checked_identity
            or sha256(Path(__file__).read_bytes()).hexdigest() != self.owner_sha256
        ):
            raise ValueError("readout overlap original source, actual commands, capability or owner changed")

    # Original checked adapter/storage are part of the explicit result, never
    # recovered from a target name or an assertion attached to emitted IR.
    _adapter: object
    _storage: object

    def materialize_scf(self):
        from merlin.compile.scheduling.scf_commands import materialize_scf_commands

        self.revalidate()
        result = materialize_scf_commands(self.kernel)
        self.revalidate()
        return result

    def emit_index_program(self, control, lower_effect, *, complete_epoch):
        """Emit actual checked effects with the same encoder and full-drain callback."""
        from .dense_panel_pipeline import DRAIN
        from .llvm_program import emit_index_effect_program

        self.revalidate()
        report = emit_index_effect_program(self.kernel, control, lower_effect,
            complete_epoch=complete_epoch, completion_instruction=DRAIN)
        self.revalidate()
        return report

    def emit_llvm(self, builder, lower_call):
        from merlin.compile.scheduling.llvm_program import FnBuilderIndexAdapter, emit_index_program

        self.revalidate()
        result = emit_index_program(self.kernel, FnBuilderIndexAdapter(builder), lower_call)
        self.revalidate()
        return result


def schedule_dense_readout_overlap(
    result, problem, block, resources, storage, adapter, admission, *, group_path, static_budget=128, timeout_ms=5000
):
    """Consume a checked segmented schedule and independently selected new law.

    This initial narrow owner accepts unbiased raw/uniform readouts and narrow
    tile transfers. Wide-load composition needs its own final-band expansion;
    it must not be represented as one primitive load here.
    """
    if type(result) is not DenseSegmentedPipelineResult or type(admission) is not DenseReadoutComputeAdmission:
        raise ValueError("readout overlap requires the exact original segmented owner and separate selected law")
    result.revalidate()
    if result.bias_source is not None:
        raise ValueError("mixed initialization/readout needs an additional explicit ownership law")
    _assignment(problem, block, resources, storage)
    original = result.admission
    bound = replace(
        original.overlap,
        max_issued_compute_pairs=min(
            original.overlap.max_issued_compute_pairs, original.reduction.max_issued_compute_pairs
        ),
        max_issued_loads=min(original.overlap.max_issued_loads, original.reduction.max_issued_loads),
        max_issued_readouts=min(original.overlap.max_issued_readouts, original.reduction.max_issued_readouts),
    )
    admission.validate_for(adapter, bound)
    rows = block.m * block.n * problem.tile_m
    if 2 * rows * storage.accumulator.unit_bytes > resources.accumulator_bytes:
        raise ValueError("two live ACC output windows exceed the original selected byte budget")
    actual, changes = propose_dense_readout_overlap(
        result.kernel, adapter, storage, group_path=group_path, accumulator_rows=rows
    )
    report = _verify_dense_epochs(
        actual,
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
        readout_source_projection=DenseReadoutCheckingProjection(result.kernel, group_path, rows),
    )
    report.update(
        schema="merlin.dense_readout_compute_overlap.v1",
        input_schedule=result.realization,
        source_identity=result.source_identity,
        schedule_ir=result.realization["schedule_ir"],
        transformations=changes,
        physical_qualification="not established",
    )
    checked = DenseReadoutOverlapResult(
        result,
        actual,
        report,
        admission,
        group_path,
        rows,
        _digest((result.kernel.digest(), actual.digest(), asdict(admission), repr(adapter), asdict(storage))),
        sha256(Path(__file__).read_bytes()).hexdigest(),
        adapter,
        storage,
    )
    checked.revalidate()
    return checked


def admit_readout_overlap_tile_loads(
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
):
    """Admit complete physical wide bands under this exact readout owner.

    The shared mapper decodes every actual band and canonicalizes only exact
    counted affine load runs. All expanded ACTUAL epoch hazards still run.
    Only then does the strict pair inverse match the canonicalized original
    command stream and transport its complete source proof. Old public panel
    and segmented consumers keep their own strict result types unchanged.
    """
    from merlin.compile.scheduling.panel_tile_loads import _admit_dense_tile_loads

    if type(original) is not DenseReadoutOverlapResult:
        raise ValueError("wide readout overlap requires its exact original checked result")
    original.revalidate()
    if storage != original._storage or adapter != original._adapter:
        raise ValueError("wide readout overlap cannot change original storage or adapter")
    components = original.original.admission
    bound = replace(
        components.overlap,
        max_issued_compute_pairs=min(
            components.overlap.max_issued_compute_pairs, components.reduction.max_issued_compute_pairs
        ),
        max_issued_loads=min(components.overlap.max_issued_loads, components.reduction.max_issued_loads),
        max_issued_readouts=min(components.overlap.max_issued_readouts, components.reduction.max_issued_readouts),
    )
    projection = DenseReadoutCheckingProjection(
        original.original.kernel, original.group_path, original.accumulator_rows, canonical_load_runs=True
    )

    def verifier(actual, p, b, r, s, a, admission, **options):
        if admission != original.admission:
            raise ValueError("wide addressing cannot change the readout/compute conjunction")
        _assignment(p, b, r, s)
        return _verify_dense_epochs(
            actual,
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
    )
