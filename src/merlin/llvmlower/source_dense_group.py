"""Bind independent source contractions to the shared counted-loop group path.

This optional original-host selection supplies qualified implementations and
effects. It discovers no target and grants no author access. Missing group
support retains the existing individual-contraction route.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from weakref import WeakKeyDictionary, ref

from merlin.common.digest import sha256_file
from merlin.transforms.certificates import ir_digest
from merlin.transforms.pipeline_state import PassSpec, PipelineContext, PipelineSession

if TYPE_CHECKING:
    from merlin.compile.scheduling.compact_dense import DenseLoopStorage
    from merlin.compile.scheduling.compact_dense_group_check import DenseLHSReuseLaw


# Weak keys and weak guard references retain no completed model or callback
# owner. A replacement guard cannot acquire an existing result's issuance.
_ISSUED_GROUPS = WeakKeyDictionary()


@dataclass(frozen=True)
class SourceDenseGrouping:
    """Original physical assignments/effects, plus a bounded grouping choice.

    The callbacks implement and project actual instruction operands. Their
    qualification and the retention law remain original integration inputs;
    candidate policy cannot derive them from numerical agreement.
    """

    storages: tuple[DenseLoopStorage, ...]
    encode_member: Callable
    project_member: Callable
    retention_law: DenseLHSReuseLaw
    max_members: int = 2

    def __post_init__(self):
        from merlin.compile.scheduling.compact_dense import DenseLoopStorage
        from merlin.compile.scheduling.compact_dense_group_check import DenseLHSReuseLaw

        if (
            type(self.storages) is not tuple
            or not 2 <= len(self.storages) <= 8
            or any(type(storage) is not DenseLoopStorage for storage in self.storages)
            or type(self.max_members) is not int
            or not 2 <= self.max_members <= len(self.storages)
            or not callable(self.encode_member)
            or not callable(self.project_member)
            or type(self.retention_law) is not DenseLHSReuseLaw
        ):
            raise ValueError("dense grouping requires original typed assignments, effects and bounded members")
        self.retention_law.__post_init__()


def _digest(value):
    from merlin.common.jsonio import canonical_sha256

    return canonical_sha256(value)


def _callbacks(selection):
    return (
        selection.adapter.encode,
        selection.adapter.project,
        selection.lhs_grouping.encode_member,
        selection.lhs_grouping.project_member,
    )


def _callback_snapshot(selection):
    from merlin.compile.scheduling.compact_dense_group import _callback

    return tuple(_callback(function) for function in _callbacks(selection))


def _check_callbacks(selection, snapshot):
    from merlin.compile.scheduling.compact_dense_group import _same_callback

    if any(
        not _same_callback(function, frozen) for function, frozen in zip(_callbacks(selection), snapshot, strict=True)
    ):
        raise ValueError("original dense group callback identity or code changed; replan")


def _validator_snapshot(validate):
    from merlin.compile.scheduling.compact_dense_group import _callback

    return _callback(validate)


def _check_validator(validate, snapshot):
    from merlin.compile.scheduling.compact_dense_group import _same_callback

    if not _same_callback(validate, snapshot):
        raise ValueError("original dense group validation callback changed; replan")


def _functions():
    from merlin.compile.scheduling.selected_contracts import SelectedSchedulingContracts

    from .source_model import _region_results, _SourceModelBuild

    return (
        _functions,
        _function_snapshot,
        _check_functions,
        _callbacks,
        _callback_snapshot,
        _check_callbacks,
        _validator_snapshot,
        _check_validator,
        _source_snapshot,
        _check_source,
        grouping_record,
        schedule_source_dense_group,
        discover_dense_groups,
        SourceDenseGrouping.__post_init__,
        SourceDenseGroupResult.revalidate,
        SourceDenseGroupResult.emit_c,
        SourceDenseGroupResult.inputs.fget,
        SourceDenseGroupResult.outputs.fget,
        _region_results,
        _SourceModelBuild.revalidate,
        _SourceModelBuild._check_group_callbacks,
        SelectedSchedulingContracts.revalidate,
        SelectedSchedulingContracts.record,
    )


def _function_snapshot():
    from merlin.compile.scheduling.compact_dense_group import _callback

    return tuple(_callback(function) for function in _functions())


def _check_functions(snapshot):
    from merlin.compile.scheduling.compact_dense_group import _same_callback

    if any(not _same_callback(function, frozen) for function, frozen in zip(_functions(), snapshot, strict=True)):
        raise ValueError("original dense group implementation function or code changed; replan")


def grouping_record(selection):
    """Finite original bindings for context identity, never qualification.

    Code serialization identifies the current Python implementation on this
    interpreter. Its digest supplies no numerical, runtime or timing permission.
    Live identity and source checks remain separate from this JSON record.
    """
    import hashlib
    import marshal

    grouping = selection.lhs_grouping
    grouping.__post_init__()
    callbacks = []
    roles = ("adapter.encode", "adapter.project", "grouping.encode_member", "grouping.project_member")
    for role, (_, target, code, path) in zip(roles, _callback_snapshot(selection), strict=True):
        if path.is_symlink() or not path.is_file():
            raise ValueError("original dense group callback requires a regular source file")
        callbacks.append(
            dict(
                role=role,
                module=target.__module__,
                qualname=target.__qualname__,
                source_path=str(path),
                source_sha256=sha256_file(path),
                code_sha256=hashlib.sha256(marshal.dumps(code)).hexdigest(),
            )
        )
    return dict(
        schema="merlin.source_dense_grouping.v1",
        storages=[asdict(storage) for storage in grouping.storages],
        retention_law=asdict(grouping.retention_law),
        max_members=grouping.max_members,
        callbacks=callbacks,
    )


def _source_snapshot(module, operations):
    return ir_digest(module), tuple(
        (operation, operation.parent, tuple(operation.operands), tuple(operation.results)) for operation in operations
    )


def _check_source(module, operations, snapshot):
    if _source_snapshot(module, operations) != snapshot:
        raise ValueError("original dense group source or SSA bindings changed; replan")


def implementation_paths():
    """Direct original owners to freeze before any group implementation runs."""
    from merlin.compile.scheduling.compact_dense_group import implementation_paths as group_paths

    return tuple(
        dict.fromkeys(
            (
                Path(__file__).resolve(),
                *(
                    Path(__file__).with_name(name)
                    for name in (
                        "source_dense.py",
                        "source_model.py",
                        "fresh_tensor_writer.py",
                        "borrowed_buffer_calls.py",
                    )
                ),
                *group_paths(),
            )
        )
    )


@dataclass(frozen=True, eq=False)
class SourceDenseGroupResult:
    source: object
    operations: tuple
    operation_ordinals: tuple[int, ...]
    source_stamp: object
    lowered_stamp: object
    selected_contracts: object
    selection_identity: str
    compiled: object
    source_passes: tuple
    source_pins: tuple[tuple[Path, str], ...]
    _guard: object

    @property
    def operation(self):
        return self.operations[0]

    @property
    def operation_ordinal(self):
        return self.operation_ordinals[0]

    @property
    def inputs(self):
        return (self.operations[0].inputs[0], *(operation.inputs[1] for operation in self.operations))

    @property
    def outputs(self):
        return tuple(operation.results[0] for operation in self.operations)

    def revalidate(self):
        from merlin.compile.scheduling.compact_dense_group import CompactDenseGroupResult
        from merlin.compile.scheduling.library_sources import shared_library_sources

        issued = _ISSUED_GROUPS.get(self)
        if issued is None or self._guard is not issued[0]() or self._guard.__code__ is not issued[1]:
            raise ValueError("original dense group issuance or guard changed; replan")
        guard = self._guard
        guard(self)
        self.selected_contracts.revalidate()
        if self._guard is not guard or guard.__code__ is not issued[1]:
            raise ValueError("original dense group guard changed during validation; replan")
        guard(self)
        current = tuple(self.source.walk())
        selection_identity = _digest(self.selected_contracts.record())
        if self._guard is not guard or guard.__code__ is not issued[1]:
            raise ValueError("original dense group guard changed during validation; replan")
        guard(self)
        if (
            ir_digest(self.source) != self.source_stamp.source_sha256
            or selection_identity != self.selection_identity
            or any(
                not 0 <= index < len(current) or current[index] is not operation
                for index, operation in zip(
                    self.operation_ordinals,
                    self.operations,
                    strict=True,
                )
            )
            or type(self.compiled) is not CompactDenseGroupResult
            or self.compiled.checked_emission != self.lowered_stamp
            or self.compiled.kernel.digest() != self.lowered_stamp.source_sha256
            or _digest(shared_library_sources()) != self.source_stamp.context.library
            or not set(implementation_paths()) <= {path for path, _ in self.source_pins}
            or not set(self.compiled.implementation_paths()) <= {path for path, _ in self.source_pins}
            or any(
                path.is_symlink() or not path.is_file() or sha256_file(path) != digest
                for path, digest in self.source_pins
            )
        ):
            raise ValueError("dense group source, selected contracts or implementation changed; replan")
        self.compiled.revalidate()
        if self._guard is not guard or guard.__code__ is not issued[1]:
            raise ValueError("original dense group guard changed during validation; replan")
        guard(self)

    def emit_c(self, instruction_set, *, symbol):
        self.revalidate()
        code = self.compiled.emit_c(instruction_set, symbol=symbol)
        self.revalidate()
        return code


def schedule_source_dense_group(module, operations, selection, validate):
    """Require current complete independent source domains before grouping."""
    from merlin.compile.scheduling.compact_dense_group import schedule_compact_dense_group
    from merlin.compile.scheduling.dense_source import DenseSourceUnavailable, read_source_dense
    from merlin.compile.scheduling.library_sources import shared_library_sources

    from .source_model import SourceRegion, _region_results, _values

    grouping, contracts = selection.lhs_grouping, selection.selected_contracts
    grouping.__post_init__()
    callbacks = _callback_snapshot(selection)
    functions = _function_snapshot()
    validator = _validator_snapshot(validate)
    paths = tuple(dict.fromkeys((*implementation_paths(), *(snapshot[3] for snapshot in (*callbacks, validator)))))
    pins = tuple((path, sha256_file(path)) for path in paths)
    original_source = _source_snapshot(module, operations)
    original_selection = _values(selection)

    def unchanged():
        _check_source(module, operations, original_source)
        _check_callbacks(selection, callbacks)
        _check_functions(functions)
        _check_validator(validate, validator)
        if _values(selection) != original_selection:
            raise ValueError("original dense group selection changed; replan")
        if any(path.is_symlink() or not path.is_file() or sha256_file(path) != digest for path, digest in pins):
            raise ValueError("dense group implementation changed during compilation; replan")

    def fresh():
        unchanged()
        validate()
        unchanged()

    if type(operations) is not tuple or not 2 <= len(operations) <= grouping.max_members:
        raise ValueError("source group requires a bounded immutable member tuple")
    current = frozenset(module.walk())
    if any(operation not in current for operation in operations):
        raise ValueError("source group member is not in the current original module")
    fresh()

    def admit():
        problems = tuple(
            read_source_dense(
                operation,
                selected_contracts=contracts,
                max_source_elements=selection.max_source_elements,
            )
            for operation in operations
        )
        lhs = operations[0].inputs[0]
        inputs = (lhs, *(operation.inputs[1] for operation in operations))
        outputs = tuple(operation.results[0] for operation in operations)
        if (
            any(operation.inputs[0] is not lhs for operation in operations)
            or len(set(inputs)) != len(inputs)
            or any(problem.m != problems[0].m or problem.k != problems[0].k for problem in problems)
            or sum(problem.m * problem.k + problem.k * problem.n + problem.m * problem.n for problem in problems)
            > selection.max_source_elements
        ):
            raise DenseSourceUnavailable("group requires bounded independent contractions with one identical LHS")
        region = SourceRegion(None, operations, inputs, (problems[0].m, problems[0].n), outputs=outputs)
        try:
            _region_results(region)
        except ValueError as error:
            raise DenseSourceUnavailable(str(error)) from error
        fresh()
        return {"independent_dense_sources": problems}

    module.verify()
    context = PipelineContext(
        hardware=contracts.hardware_identity,
        numerical_policy=contracts.numerical_policy,
        resources=_digest(
            (
                [asdict(storage) for storage in grouping.storages],
                None if selection.resources is None else asdict(selection.resources),
                [asdict(window) for window in selection.reservations],
                grouping.max_members,
                selection.static_budget,
                selection.max_source_elements,
            )
        ),
        execution=_digest(asdict(grouping.retention_law)),
        library=_digest(shared_library_sources()),
    )
    session = PipelineSession(module, context=context, ir_level="linalg", max_repairs=0)
    passes = (
        PassSpec(
            "group-source-admission", 0, produces=("independent_dense_sources",), mutates_ir=False, ir_level="linalg"
        ),
        PassSpec(
            "group-dense-lowering",
            2,
            requires=("independent_dense_sources",),
            produces=("compiled_dense_group",),
            mutates_ir=False,
            ir_level="linalg",
        ),
    )
    session.run(passes[0], admit)

    def lower():
        compiled = schedule_compact_dense_group(
            session.analysis("independent_dense_sources"),
            selection.resources,
            grouping.storages[: len(operations)],
            selection.adapter,
            encode_member=grouping.encode_member,
            project_member=grouping.project_member,
            retention_law=grouping.retention_law,
            name="source_dense_group",
            dtypes=contracts.dense.dtypes,
            static_budget=selection.static_budget,
            external_buffers_disjoint=selection.external_buffers_disjoint,
            selected_contracts=contracts,
            reservations=selection.reservations,
            verification_timeout_ms=selection.verification_timeout_ms,
            max_members=grouping.max_members,
        )
        fresh()
        return {"compiled_dense_group": compiled}

    session.run(passes[1], lower)
    compiled = session.analysis("compiled_dense_group")
    if compiled is None:
        return None
    source_stamp = session.stamp()
    lowered_stamp = compiled.checked_emission
    ordinal = {operation: index for index, operation in enumerate(module.walk())}
    from merlin.sched.ir import TensorArg

    problems = session.analysis("independent_dense_sources")
    dtypes = contracts.dense.dtypes
    expected_arguments = (
        TensorArg("lhs", (problems[0].m, problems[0].k), dtypes[0], "read"),
        *(TensorArg(f"rhs{i}", (problem.k, problem.n), dtypes[1], "read") for i, problem in enumerate(problems)),
        *(TensorArg(f"output{i}", (problem.m, problem.n), dtypes[2], "write") for i, problem in enumerate(problems)),
    )
    ordinals = tuple(ordinal[operation] for operation in operations)
    source_passes = tuple(session.records)
    fresh()
    selection_identity = _digest(contracts.record())
    fresh()

    def guard(candidate):
        unchanged()
        if (
            candidate.source is not module
            or candidate.operations != operations
            or candidate.operation_ordinals != ordinals
            or candidate.compiled is not compiled
            or candidate.source_stamp != source_stamp
            or candidate.lowered_stamp != lowered_stamp
            or candidate.selected_contracts is not contracts
            or candidate.source_pins != pins
            or candidate.source_passes != source_passes
            or candidate.selection_identity != selection_identity
            or candidate.compiled.kernel.args != expected_arguments
        ):
            raise ValueError("original dense group source-to-program binding changed; replan")

    result = SourceDenseGroupResult(
        module,
        operations,
        ordinals,
        source_stamp,
        compiled.checked_emission,
        contracts,
        selection_identity,
        compiled,
        source_passes,
        pins,
        guard,
    )
    _ISSUED_GROUPS[result] = ref(guard), guard.__code__
    result.revalidate()
    return result


def discover_dense_groups(module, selection, validate):
    """Try bounded adjacent groups; unsupported candidates retain singles."""
    from merlin.compile.scheduling.dense_contracts import DenseSchedulingUnavailable
    from merlin.compile.scheduling.dense_source import DenseSourceUnavailable, _problem

    from .source_model import SourceRegion

    grouping = selection.lhs_grouping
    if grouping is None:
        return (), ()
    if not selection.selected_contracts.dense.external_pointer_arguments:
        return (), ("dense grouping lacks explicit external pointer transport; original singles retained",)
    if (
        selection.logical_abi is not None
        or selection.offload_plan is not None
        or selection.max_convolution_packing_bytes is not None
    ):
        return (), ("dense grouping retains singles for logical-ABI, placement and convolution compositions",)
    original_source = _source_snapshot(module, tuple(module.walk()))
    callbacks = _callback_snapshot(selection)
    functions = _function_snapshot()
    validator = _validator_snapshot(validate)
    callback_pins = tuple((snapshot[3], sha256_file(snapshot[3])) for snapshot in (*callbacks, validator))

    def fresh():
        def unchanged():
            _check_source(module, tuple(item[0] for item in original_source[1]), original_source)
            _check_callbacks(selection, callbacks)
            _check_functions(functions)
            _check_validator(validate, validator)
            if any(
                path.is_symlink() or not path.is_file() or sha256_file(path) != digest for path, digest in callback_pins
            ):
                raise ValueError("original dense group callback source changed during discovery; replan")

        unchanged()
        validate()
        unchanged()

    claimed, regions, declined = set(), [], []
    attempts = 0
    for operation in module.walk():
        if operation.name != "linalg.generic" or operation in claimed:
            continue
        candidates = []
        current = operation
        while current is not None and len(candidates) < grouping.max_members:
            if current.name != "linalg.generic" or current in claimed:
                break
            try:
                _problem(current, selection.selected_contracts, selection.max_source_elements)
            except DenseSourceUnavailable:
                break
            if candidates and current.inputs[0] is not candidates[0].inputs[0]:
                break
            candidates.append(current)
            current = current.next_op
        for count in range(len(candidates), 1, -1):
            if attempts >= selection.max_islands * grouping.max_members:
                return tuple(regions), (
                    *declined,
                    "dense grouping search budget exhausted; remaining work retains singles",
                )
            attempts += 1
            if len(regions) >= selection.max_islands:
                raise ValueError("source dense grouping exceeds the selected island budget")
            members = tuple(candidates[:count])
            fresh()
            try:
                result = schedule_source_dense_group(module, members, selection, validate)
            except (DenseSourceUnavailable, DenseSchedulingUnavailable) as error:
                fresh()
                declined.append(str(error))
                continue
            if result is None:
                declined.append("no legal full-reduction shared-LHS group fits assigned storage and code budgets")
                continue
            inputs = (members[0].inputs[0], *(member.inputs[1] for member in members))
            outputs = tuple(member.results[0] for member in members)
            regions.append(SourceRegion(result, members, inputs, tuple(outputs[0].type.get_shape()), outputs=outputs))
            claimed.update(members)
            break
    return tuple(regions), tuple(declined)
