"""Full-K shared-LHS groups using the existing dense blocks and counted loops.

This conditional original-host entry needs independently qualified retention,
interleaving and explicit pointer transport. Missing capabilities retain the
original separate contraction route. It issues no target or Phase 2 grants.
"""

from __future__ import annotations

import hashlib
import inspect
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from merlin.sched.ir import Call, Const, Kernel, Loop, TensorArg, Var
from merlin.sched.ir.kernel import check_structure
from merlin.transforms.dense_planning import DenseGroupChoice, DenseProblem, DenseResources, choose_dense_group
from merlin.transforms.pipeline_state import AnalysisStamp, PassSpec, PipelineContext, PipelineSession

from .compact_dense import (
    DenseLoopAdapter,
    _dense_library_sources,
    _digest,
    _emit_dense_tile_group,
    _emit_lhs_panel,
    _Group,
    _groups,
    _static_calls,
)
from .compact_dense_group_check import DenseLHSReuseLaw, _views, verify_compact_dense_group
from .compact_verification import DenseVerificationUnavailable


def implementation_paths() -> tuple[Path, ...]:
    """Direct implementation plus the unchanged original scheduling library."""
    import importlib

    from . import compact_dense_group_check
    from .library_sources import library_modules

    modules = (importlib.import_module(name) for name in library_modules())
    return tuple(
        dict.fromkeys(
            (
                Path(__file__).resolve(),
                Path(compact_dense_group_check.__file__).resolve(),
                *(Path(module.__file__).resolve() for module in modules),
            )
        )
    )


def _callback(function):
    owner = getattr(function, "__self__", None)
    target = getattr(function, "__func__", function)
    code = getattr(target, "__code__", None)
    source = inspect.getsourcefile(target)
    if code is None or source is None:
        raise ValueError("shared LHS callbacks require original Python implementation source")
    return owner, target, code, Path(source).resolve()


def _same_callback(function, frozen):
    owner, target, code, _ = frozen
    return (
        getattr(function, "__self__", None) is owner
        and getattr(function, "__func__", function) is target
        and target.__code__ is code
    )


def emit_compact_dense_group(problems, blocks, storages, *, encode_member, name, dtypes):
    """One M outer nest, one LHS load panel, then complete ordered members."""
    first = problems[0]
    serial = 0

    def variable(prefix):
        nonlocal serial
        serial += 1
        return prefix + str(serial)

    def nest(extents, action):
        names = tuple(variable("tile") for _ in extents)
        body = tuple(action(*(Var(name) for name in names)))
        for var, extent in reversed(tuple(zip(names, extents, strict=True))):
            body = (Loop(var, extent, body),)
        return body

    def call(member, operation):
        actual = encode_member(member, operation)
        if type(actual) is not Call:
            raise ValueError("group encoder must emit an actual existing instruction Call")
        return actual

    statements = []
    for mg in _groups(first.tiles[0], blocks[0].m, "outer_m"):
        kg = _Group(None, 1, Const(0), first.tiles[1], True)
        body = list(_emit_lhs_panel(first, blocks[0], storages[0], lambda op: call(None, op), nest, mg, kg))
        for member, (problem, block, storage) in enumerate(zip(problems, blocks, storages, strict=True)):
            for ng in _groups(problem.tiles[2], block.n, variable("outer_n")):
                emitted = _emit_dense_tile_group(
                    problem, block, storage, lambda op: call(member, op), variable, nest, mg, ng, include_lhs=False
                )
                body.extend((Loop(ng.var, ng.trips, emitted),) if ng.var else emitted)
        statements.extend((Loop(mg.var, mg.trips, tuple(body)),) if mg.var else body)
    kernel = Kernel(
        name,
        (
            TensorArg("lhs", (first.m, first.k), dtypes[0], "read"),
            *(TensorArg(f"rhs{i}", (p.k, p.n), dtypes[1], "read") for i, p in enumerate(problems)),
            *(TensorArg(f"output{i}", (p.m, p.n), dtypes[2], "write") for i, p in enumerate(problems)),
        ),
        tuple(statements),
    )
    if check_structure(kernel):
        raise ValueError("invalid shared LHS counted-loop structure")
    return kernel


@dataclass(frozen=True)
class CompactDenseGroupResult:
    kernel: Kernel
    choice: DenseGroupChoice
    realization: dict
    index_bits: int
    output_shapes: tuple[tuple[int, int], ...]
    output_dtypes: tuple[str, ...]
    checked_emission: AnalysisStamp
    _guard: object
    _paths: tuple[Path, ...]
    _kernel_digest: str
    _checked_index_bits: int

    def revalidate(self):
        self._guard(self)
        if (
            self.kernel.digest() != self._kernel_digest
            or self.checked_emission.source_sha256 != self._kernel_digest
            or self.index_bits != self._checked_index_bits
            or self.output_shapes != tuple(a.shape for a in self.kernel.args if a.access == "write")
            or self.output_dtypes != tuple(a.dtype for a in self.kernel.args if a.access == "write")
        ):
            raise ValueError("checked shared LHS kernel or complete outputs changed; replan")

    def implementation_paths(self) -> tuple[Path, ...]:
        self.revalidate()
        return self._paths

    def emit_c(self, instruction_set, *, symbol):
        from merlin.sched.codegen import emit_c_function

        self.revalidate()
        result = emit_c_function(self.kernel, instruction_set, symbol=symbol, index_bits=self.index_bits)
        self.revalidate()
        return result


def schedule_compact_dense_group(
    problems: tuple[DenseProblem, ...],
    resources: DenseResources | None,
    storages: tuple,
    adapter: DenseLoopAdapter,
    *,
    encode_member,
    project_member,
    retention_law,
    name,
    dtypes,
    static_budget,
    external_buffers_disjoint,
    selected_contracts=None,
    reservations=(),
    verification_timeout_ms=2000,
    max_members=8,
):
    """Choose and check one bounded full-K group before actual emission.

    Distinct RHS/accumulator assigned windows and complete explicit external
    pointers are required. A declaration or this result alone grants no effect
    permission. Missing retention/full-K capacity yields no candidate.
    """
    from .selected_contracts import SelectedSchedulingContracts

    if (
        type(problems) is not tuple
        or type(max_members) is not int
        or not 2 <= len(problems) <= max_members <= 128
        or any(type(p) is not DenseProblem for p in problems)
        or type(adapter) is not DenseLoopAdapter
        or type(dtypes) is not tuple
        or len(dtypes) != 3
        or type(static_budget) is not int
        or static_budget < 0
        or type(verification_timeout_ms) is not int
        or verification_timeout_ms <= 0
    ):
        raise ValueError("shared LHS requires bounded typed group inputs")
    if retention_law is None:
        return None
    if type(retention_law) is not DenseLHSReuseLaw:
        raise ValueError("shared LHS needs the original typed retention law")
    retention_law.__post_init__()
    if (
        not retention_law.admits(adapter)
        or not adapter.synchronous_completion
        or not adapter.exact_accumulator_continuation
    ):
        return None
    if external_buffers_disjoint is not True:
        return None
    for p in problems:
        p.__post_init__()
    adapter.__post_init__()
    assigned, _ = _views(problems, storages, adapter, reject_aliases=False)
    if len({view.name for view in assigned}) != len(assigned) or any(
        a.backing == b.backing
        and a.base * a.unit_bytes < (b.base + b.capacity) * b.unit_bytes
        and b.base * b.unit_bytes < (a.base + a.capacity) * a.unit_bytes
        for i, a in enumerate(assigned)
        for b in assigned[:i]
    ):
        return None
    if selected_contracts is not None and type(selected_contracts) is not SelectedSchedulingContracts:
        raise ValueError("shared LHS needs an original selected dense contract")

    def callbacks():
        return (encode_member, project_member, adapter.encode, adapter.project) + (
            () if selected_contracts is None else (selected_contracts.revalidate, selected_contracts.record)
        )

    frozen_callbacks = tuple(_callback(f) for f in callbacks())
    paths = tuple(dict.fromkeys((*implementation_paths(), *(v[3] for v in frozen_callbacks))))
    pins = tuple((p, hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths)
    original = _digest(
        (
            tuple(asdict(p) for p in problems),
            tuple(asdict(s) for s in storages),
            asdict(resources) if resources is not None else None,
            retention_law.record(),
            dtypes,
            name,
            static_budget,
            external_buffers_disjoint,
            tuple(asdict(r) for r in reservations),
            verification_timeout_ms,
            max_members,
            adapter.configuration_identity,
            adapter.numerical_policy_identity,
            adapter.qualification_identity,
            adapter.synchronous_completion,
            adapter.exact_accumulator_continuation,
            adapter.retained_weights,
            adapter.zero_initialization,
            *((asdict(adapter.load_grouping),) if adapter.load_grouping is not None else ()),
            adapter.index_bits,
        )
    )
    library = _dense_library_sources()

    def functions():
        return (
            emit_compact_dense_group,
            verify_compact_dense_group,
            choose_dense_group,
            implementation_paths,
            _callback,
            _same_callback,
            _digest,
            _dense_library_sources,
            _views,
            _groups,
            _emit_lhs_panel,
            _emit_dense_tile_group,
            DenseLHSReuseLaw.__post_init__,
            DenseLHSReuseLaw.record,
            DenseLHSReuseLaw.admits,
            CompactDenseGroupResult.revalidate,
            CompactDenseGroupResult.emit_c,
            CompactDenseGroupResult.implementation_paths,
        )

    original_functions = tuple((f, f.__code__) for f in functions())

    def static_fresh():
        if any(
            f is not original or original.__code__ is not code
            for f, (original, code) in zip(functions(), original_functions, strict=True)
        ):
            raise ValueError("shared LHS direct implementation changed")
        now = _digest(
            (
                tuple(asdict(p) for p in problems),
                tuple(asdict(s) for s in storages),
                asdict(original_resources) if original_resources is not None else None,
                retention_law.record(),
                dtypes,
                name,
                static_budget,
                external_buffers_disjoint,
                tuple(asdict(r) for r in reservations),
                verification_timeout_ms,
                max_members,
                adapter.configuration_identity,
                adapter.numerical_policy_identity,
                adapter.qualification_identity,
                adapter.synchronous_completion,
                adapter.exact_accumulator_continuation,
                adapter.retained_weights,
                adapter.zero_initialization,
                *((asdict(adapter.load_grouping),) if adapter.load_grouping is not None else ()),
                adapter.index_bits,
            )
        )
        current = callbacks()
        if original != now or any(
            not _same_callback(f, saved) for f, saved in zip(current, frozen_callbacks, strict=True)
        ):
            raise ValueError("original shared LHS inputs or callback implementation changed")
        if (
            pins != tuple((p, hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths)
            or library != _dense_library_sources()
        ):
            raise ValueError("shared LHS implementation source changed; replan")

    original_resources = resources
    static_fresh()
    selection = None if selected_contracts is None else _digest(selected_contracts.record())
    static_fresh()

    def fresh():
        static_fresh()
        if selected_contracts is not None:
            selected_contracts.revalidate()
            static_fresh()
            current_selection = _digest(selected_contracts.record())
            static_fresh()
            if selection != current_selection:
                raise ValueError("selected shared LHS contracts changed; reselect")

    fresh()
    if selected_contracts is not None:
        if type(selected_contracts) is not SelectedSchedulingContracts or selected_contracts.dense is None:
            raise ValueError("shared LHS needs an original selected dense contract")
        dense = selected_contracts.dense
        if not dense.external_pointer_arguments:
            return None
        limits = dense.block_limits
        views = (storages[0].lhs, *(s.rhs for s in storages), *(s.accumulator for s in storages))
        operands = sum(v.capacity * v.unit_bytes for v in views[: len(problems) + 1])
        accumulators = sum(v.capacity * v.unit_bytes for v in views[len(problems) + 1 :])
        derived = DenseResources(
            operands,
            accumulators,
            *limits,
            candidate_budget=resources.candidate_budget if resources is not None else 4096,
            partial_output_tiles=dense.partial_output_tiles,
            partial_output_blocks=dense.partial_output_blocks,
        )
        if resources is not None:
            derived = replace(
                derived,
                **{
                    key: min(getattr(resources, key), getattr(derived, key))
                    for key in ("operand_bytes", "accumulator_bytes", "max_m_blocks", "max_n_blocks", "max_k_blocks")
                },
                partial_output_tiles=resources.partial_output_tiles and derived.partial_output_tiles,
                partial_output_blocks=resources.partial_output_blocks and derived.partial_output_blocks,
            )
        for p, storage in zip(problems, storages, strict=True):
            dense.bind(
                p,
                storage,
                adapter,
                dtypes=dtypes,
                hardware=selected_contracts.hardware_identity,
                policy=selected_contracts.numerical_policy,
                resources=derived,
                reservations=reservations,
            )
        resources = derived
    elif resources is None or reservations:
        raise ValueError("resource derivation and reservations require selected contracts")
    fresh()
    if problems[0].tiles[1] > resources.max_k_blocks:
        return None
    checked_resources = asdict(resources)

    def guarded_encode(member, operation):
        fresh()
        actual = encode_member(member, operation)
        fresh()
        if asdict(resources) != checked_resources:
            raise ValueError("derived shared LHS resources changed")
        return actual

    def guarded_project(call):
        fresh()
        actual = project_member(call)
        fresh()
        if asdict(resources) != checked_resources:
            raise ValueError("derived shared LHS resources changed")
        return actual

    def emit(blocks):
        return emit_compact_dense_group(
            problems, blocks, storages, encode_member=guarded_encode, name=name, dtypes=dtypes
        )

    def admits(blocks):
        if not all(s.admits(p, b, resources) for p, b, s in zip(problems, blocks, storages, strict=True)):
            return False
        counts = tuple(s.row_counts(p, b) for p, b, s in zip(problems, blocks, storages, strict=True))
        operand = counts[0][0] * storages[0].lhs.unit_bytes + sum(
            c[1] * s.rhs.unit_bytes for c, s in zip(counts, storages, strict=True)
        )
        accumulator = sum(c[2] * s.accumulator.unit_bytes for c, s in zip(counts, storages, strict=True))
        return operand <= resources.operand_bytes and accumulator <= resources.accumulator_bytes

    context = PipelineContext(
        hardware=adapter.configuration_identity,
        numerical_policy=adapter.numerical_policy_identity,
        resources=_digest((checked_resources, tuple(asdict(s) for s in storages))),
        execution=_digest((retention_law.record(), selection, adapter.index_bits)),
        library=_digest(pins and tuple((str(p), h) for p, h in pins)),
    )
    state = {"kernel": None}

    def structure(current):
        if current["kernel"] is not None and check_structure(current["kernel"]):
            raise ValueError("shared LHS pipeline requires valid counted-loop IR")

    session = PipelineSession(
        state,
        context=context,
        ir_level="compact_dense_group",
        max_repairs=0,
        fingerprint=lambda current: _digest(None if current["kernel"] is None else current["kernel"].digest()),
        verify_ir=structure,
    )
    choice = None
    report = None
    specs = (
        PassSpec("admission", 0, produces=("legality",), mutates_ir=False, ir_level="compact_dense_group"),
        PassSpec(
            "blocking",
            2,
            requires=("legality",),
            produces=("choice",),
            mutates_ir=False,
            ir_level="compact_dense_group",
        ),
        PassSpec(
            "emission",
            5,
            requires=("choice",),
            produces=("emitted",),
            invalidates=("*",),
            ir_level="compact_dense_group",
        ),
        PassSpec(
            "realization",
            5,
            requires=("emitted",),
            produces=("realization",),
            mutates_ir=False,
            ir_level="compact_dense_group",
        ),
    )

    def blocking():
        nonlocal choice
        choice = choose_dense_group(
            problems,
            resources,
            static_budget=static_budget,
            price=lambda blocks: sum(1 for _ in _static_calls(emit(blocks).body)),
            admits=admits,
            k_limits=tuple(
                (lambda m, n, p=p, s=s: s.k_limit(p, resources, m, n)) for p, s in zip(problems, storages, strict=True)
            ),
            max_members=max_members,
        )
        return {"choice": choice}

    def emission():
        state["kernel"] = emit(choice.blocks)
        return {"emitted": state["kernel"].digest()}

    def realization():
        nonlocal report
        report = verify_compact_dense_group(
            state["kernel"],
            problems,
            resources,
            storages,
            adapter,
            project_member=guarded_project,
            retention_law=retention_law,
            dtypes=dtypes,
            static_budget=static_budget,
            external_buffers_disjoint=external_buffers_disjoint,
            timeout_ms=verification_timeout_ms,
            max_members=max_members,
            selected_contracts=selected_contracts,
            reservations=reservations,
        )
        return {"realization": report}

    actions = (lambda: {"legality": retention_law.record()}, blocking, emission, realization)
    for spec, action in zip(specs, actions, strict=True):
        fresh()
        try:
            session.run(spec, action)
        except DenseVerificationUnavailable:
            return None
        fresh()
        if spec.name == "blocking" and choice is None:
            return None
    report["pipeline"] = {
        "context": asdict(context),
        "passes": [asdict(r) for r in session.records],
        "selected_contracts": None if selected_contracts is None else selected_contracts.record(),
        "library_sources_sha256": library,
    }
    kernel = state["kernel"]
    stamp = AnalysisStamp(kernel.digest(), context, "compact_dense_group")
    checked = kernel.digest(), asdict(stamp), asdict(choice), adapter.index_bits, implementation_paths()

    def checked_guard(current):
        fresh()
        if (
            current.kernel.digest(),
            asdict(current.checked_emission),
            asdict(current.choice),
            current.index_bits,
            current._paths,
        ) != checked:
            raise ValueError("checked shared LHS emission, context or choice changed; replan")

    result = CompactDenseGroupResult(
        kernel,
        choice,
        report,
        adapter.index_bits,
        tuple((p.m, p.n) for p in problems),
        (dtypes[2],) * len(problems),
        stamp,
        checked_guard,
        implementation_paths(),
        kernel.digest(),
        adapter.index_bits,
    )
    result.revalidate()
    return result
