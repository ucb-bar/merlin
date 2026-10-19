"""Explicit backend selection for ordinary checked source preparation.

Merlin matches current typed source, checks schedules and inverts its call
replacement. The original caller supplies target instruction selection, ABI
glue, capability revalidation and independent execution qualification. No
target is discovered and no qualified device implementation is supplied here.
"""

from __future__ import annotations

import copy
import inspect
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from types import FunctionType, MethodType

from merlin.common import mlir_query as mq
from merlin.common.digest import sha256_file
from merlin.common.jsonio import write_pretty_json
from merlin.compile.scheduling.dense_source import DenseSourceUnavailable, read_dense_source, schedule_source_dense
from merlin.compile.scheduling.fusion_semantics import UnsupportedElementwise
from merlin.compile.scheduling.pointwise_source import compile_source_pointwise, read_pointwise_source
from merlin.compile.scheduling.schedule_pipeline import ScheduleBudgets
from merlin.compile.scheduling.selected_contracts import SelectedSchedulingContracts
from merlin.transforms.certificates import ir_digest, ir_text

from .fresh_tensor_writer import FreshTensorWriterContract, rewrite_fresh_tensor_writers


@dataclass(frozen=True)
class PointwiseSourceBackend:
    """Host-selected implementation callbacks; declarations alone qualify nothing.

    propose(source, operation, directory) returns ordinary pointwise compilation
    kwargs or None for unsupported semantics/layout/resources. emit returns the
    independently qualified fresh-writer ABI contract after emitting actual glue.
    finalize assembles those fragments. build retains the ordinary checked
    HostProviderBuild interface. verify reopens original capability selections.
    """

    contracts: SelectedSchedulingContracts
    propose: object
    emit: object
    finalize: object
    build: object
    verify: object
    budgets: ScheduleBudgets = ScheduleBudgets()


@dataclass(frozen=True)
class DenseSourceLimits:
    """Original source/verification ceilings; policies may only reduce work caps."""

    max_source_elements: int = 64 << 20
    static_budget: int = 80
    verification_timeout_ms: int = 2000
    max_repairs: int = 1

    def __post_init__(self):
        for key, value in asdict(self).items():
            if type(value) is not int or value < (0 if key == "max_repairs" else 1):
                raise ValueError("dense source limits require bounded original integer ceilings")
        if self.max_repairs not in (0, 1):
            raise ValueError("dense source supports only the original zero or one repair")


@dataclass(frozen=True)
class DenseSourceBackend:
    """Host-owned exact dense implementation; missing runtime proof stays off.

    The propose callback receives a freshly matched DenseProblem and current
    operation. Merlin supplies source, selected contracts, name and original
    runtime ownership to the fixed checked scheduler. No field is a target
    qualification issuer; the original independent callback remains mandatory.
    """

    contracts: SelectedSchedulingContracts
    propose: object
    emit: object
    finalize: object
    build: object
    verify: object
    limits: DenseSourceLimits = DenseSourceLimits()
    external_buffers_disjoint: bool = False


@dataclass(frozen=True)
class SharedSourceBackend:
    """One explicitly reviewed implementation for dense and pointwise islands.

    Callbacks dispatch on the typed source descriptor, never model identities.
    One finalize/build owner sees the complete selected island set, including
    supported readouts; this does not imply fusion or device residency.
    """

    contracts: SelectedSchedulingContracts
    propose: object
    emit: object
    finalize: object
    build: object
    verify: object
    budgets: ScheduleBudgets = ScheduleBudgets()
    limits: DenseSourceLimits = DenseSourceLimits()
    external_buffers_disjoint: bool = False


def _callback(value):
    function = value.__func__ if type(value) is MethodType else value
    if type(function) is not FunctionType:
        raise ValueError("source routing requires explicit original Python callback owners")
    path = inspect.getsourcefile(function)
    if path is None or Path(path).is_symlink() or not Path(path).is_file():
        raise ValueError("source routing callback source is absent or linked")
    return value, function.__code__, Path(path).resolve(), sha256_file(Path(path))


@dataclass(frozen=True)
class _Prepared:
    source: Path
    source_sha256: str
    selected: Path
    selected_sha256: str
    original_module: object
    original_digest: str
    compiled: tuple
    report_path: Path


class SharedSourceRouting:
    """One ordinary preparation/build invocation; missing qualification stays off."""

    def __init__(self, backend=None, *, qualify=None):
        if backend is not None and type(backend) not in (
            PointwiseSourceBackend,
            DenseSourceBackend,
            SharedSourceBackend,
        ):
            raise ValueError("source routing requires an explicitly selected typed backend")
        self._backend, self._qualify, self._prepared = backend, qualify, None
        self._backend_type = type(backend)
        self._families = (
            ("dense", "pointwise")
            if type(backend) is SharedSourceBackend
            else ("dense",)
            if type(backend) is DenseSourceBackend
            else ("pointwise",)
        )
        self._reason = (
            "no original selected pointwise backend"
            if backend is None
            else "no original independent execution qualifier"
            if qualify is None
            else None
        )
        self._callbacks = ()
        self._contracts = None
        self._budgets = None
        self._dense_limits = None
        self._runtime = None
        self._host_sources = ()
        if self._reason is None:
            if type(backend.contracts) is not SelectedSchedulingContracts:
                raise ValueError("source routing requires selected RTL/software contracts")
            self._callbacks = tuple(
                _callback(value)
                for value in (backend.propose, backend.emit, backend.finalize, backend.build, backend.verify, qualify)
            )
            self._contracts = backend.contracts.record()
            if "pointwise" in self._families:
                if type(backend.budgets) is not ScheduleBudgets:
                    raise ValueError("source routing requires original fixed verification budgets")
                self._budgets = asdict(backend.budgets)
            if "dense" in self._families:
                if type(backend.limits) is not DenseSourceLimits or type(backend.external_buffers_disjoint) is not bool:
                    raise ValueError("source routing requires original dense limits and runtime ownership")
                self._dense_limits = asdict(backend.limits)
                self._runtime = backend.external_buffers_disjoint
            from merlin.common.paths import module_source_path

            paths = tuple(
                module_source_path(name)
                for name in (
                    __name__,
                    "merlin.llvmlower.shared_pointwise_routing",
                    "merlin.llvmlower.fresh_tensor_writer",
                    "merlin.runtime.host_provider",
                    "merlin.llvmlower.prepared_model_transform",
                )
            )
            self._host_sources = tuple((path, sha256_file(path)) for path in paths)
            self._verify()

    def record(self):
        result = {
            "schema": (
                "merlin.shared_pointwise_source_routing.v1"
                if self._families == ("pointwise",)
                else "merlin.shared_source_routing.v1"
            ),
            "status": "disabled" if self._reason is not None else "explicit_selection",
            "refusal": self._reason,
            "qualification": (
                "original independent execution callback required; no hardware/timing authority supplied"
            ),
            "selected_contracts": copy.deepcopy(self._contracts),
            "original_budget_ceilings": copy.deepcopy(self._budgets),
            "host_sources": [{"path": str(path), "sha256": digest} for path, digest in self._host_sources],
        }
        if "dense" in self._families:
            result.update(
                families=list(self._families),
                original_dense_limits=copy.deepcopy(self._dense_limits),
                original_external_buffers_disjoint=self._runtime,
            )
        return result

    def build_arguments(self):
        """Use existing ordinary preparation and checked provider build seams."""
        if self._reason is not None:
            return {}
        self._verify()
        return {"prepared_model_transform": self.prepare, "host_provider_builder": self.build}

    def _verify(self):
        if type(self) not in (SharedSourceRouting, PointwiseSourceRouting) or self._reason is not None:
            raise ValueError("source routing has no enabled original backend/qualification selection")
        if any(path.is_symlink() or sha256_file(path) != digest for path, digest in self._host_sources):
            raise ValueError("source routing original preparation/provider implementation changed")
        backend = self._backend
        if type(backend) is not self._backend_type or type(backend.contracts) is not SelectedSchedulingContracts:
            raise ValueError("source routing original backend or contract owner changed")
        values = (backend.propose, backend.emit, backend.finalize, backend.build, backend.verify, self._qualify)
        if len(values) != len(self._callbacks):
            raise ValueError("source routing callback membership changed")
        for value, (original, code, path, digest) in zip(values, self._callbacks, strict=True):
            function = value.__func__ if type(value) is MethodType else value
            if value is not original or function.__code__ is not code or sha256_file(path) != digest:
                raise ValueError("source routing original callback source or owner changed")
        backend.contracts.revalidate()
        if (
            backend.contracts.record() != self._contracts
            or "pointwise" in self._families
            and asdict(backend.budgets) != self._budgets
            or "dense" in self._families
            and (asdict(backend.limits) != self._dense_limits or backend.external_buffers_disjoint is not self._runtime)
        ):
            raise ValueError("source routing selected numerical/software/RTL contracts changed")
        backend.verify()

    def _revalidate_prepared(self):
        self._verify()
        state = self._prepared
        if type(state) is not _Prepared:
            raise ValueError("source routing requires completed original preparation")
        for path, digest in ((state.source, state.source_sha256), (state.selected, state.selected_sha256)):
            if path.is_symlink() or not path.is_file() or sha256_file(path) != digest:
                raise ValueError("source routing original/selected prepared source changed")
        if ir_digest(state.original_module) != state.original_digest:
            raise ValueError("source routing original source module changed")
        for result in state.compiled:
            result.revalidate()
        return state

    def _match_source(self, operation):
        reasons = []
        for family in self._families:
            try:
                if family == "dense":
                    source = read_dense_source(
                        operation,
                        selected_contracts=self._backend.contracts,
                        max_source_elements=self._dense_limits["max_source_elements"],
                    )
                else:
                    source = read_pointwise_source(
                        operation, max_elements=64 << 20, max_operations=128, max_ir_bytes=65536
                    )
                return family, source, reasons
            except (UnsupportedElementwise, DenseSourceUnavailable) as error:
                reasons.append(f"{family}: {error}")
        return None, None, reasons

    def _compile_source(self, original, operation, family, kwargs, symbol):
        if type(kwargs) is not dict or set(kwargs) & {
            "module",
            "operation",
            "selected_contracts",
            "max_source_elements",
        }:
            raise ValueError("source backend cannot replace current source or selected contracts")
        if family == "dense":
            if "external_buffers_disjoint" in kwargs or "name" in kwargs:
                raise ValueError("dense policy cannot replace original runtime ownership or symbol")
            limits = self._dense_limits
            for name in ("static_budget", "verification_timeout_ms", "max_repairs"):
                value = kwargs.get(name, limits[name])
                if (
                    type(value) is not int
                    or value < (0 if name == "max_repairs" else 1)
                    or value > limits[name]
                    or name == "verification_timeout_ms"
                    and value != limits[name]
                ):
                    raise ValueError("dense policy cannot change original verification limits or ceilings")
            if not self._runtime:
                raise DenseSourceUnavailable("original dense runtime buffer ownership/disjointness is unavailable")
            return schedule_source_dense(
                original,
                operation,
                selected_contracts=self._backend.contracts,
                name=symbol,
                external_buffers_disjoint=self._runtime,
                max_source_elements=limits["max_source_elements"],
                **{
                    **{name: limits[name] for name in ("static_budget", "verification_timeout_ms", "max_repairs")},
                    **kwargs,
                },
            )
        budgets = kwargs.get("budgets", self._backend.budgets)
        if type(budgets) is not ScheduleBudgets:
            raise ValueError("source routing requires typed fixed verification limits")
        policy_fields = {"eliminate_copies", "eliminate_configuration", "optimize_storage"}
        bounded_fields = {"max_fusions", "max_spills", "max_repairs"}
        for key, value in asdict(budgets).items():
            if key not in policy_fields and (
                key not in bounded_fields
                and value != self._budgets[key]
                or key in bounded_fields
                and value > self._budgets[key]
            ):
                raise ValueError("pointwise policy cannot change original verification limits or ceilings")
        return compile_source_pointwise(
            original, operation, selected_contracts=self._backend.contracts, **{**kwargs, "budgets": budgets}
        )

    def prepare(self, source_path, output):
        from xdsl.dialects import builtin, func, tensor
        from xdsl.rewriter import InsertPoint, Rewriter

        self._verify()
        if self._prepared is not None:
            raise ValueError("source routing cannot reuse an earlier model preparation")
        source_path, output = Path(source_path), Path(output)
        if source_path.is_symlink() or not source_path.is_file():
            raise ValueError("source routing requires regular original prepared bytes")
        source_hash = sha256_file(source_path)
        original = mq.parse(source_path)
        original_digest = ir_digest(original)
        replacement = original.clone()
        targets = dict(enumerate(replacement.walk()))
        rows, results, writers, replacements = [], [], [], []
        for ordinal, operation in enumerate(original.walk()):
            if operation.name != "linalg.generic":
                continue
            family, source, reasons = self._match_source(operation)
            if source is None:
                rows.append(
                    {
                        "ordinal": ordinal,
                        "status": "source_fallback",
                        "reason": "; ".join(reasons),
                        "unavailable_families": reasons,
                    }
                )
                continue
            directory = output / f"{family}-{ordinal}"
            directory.mkdir()
            kwargs = self._backend.propose(source, operation, directory)
            self._verify()
            if kwargs is None:
                rows.append(
                    {"ordinal": ordinal, "status": "source_fallback", "reason": "backend capability unavailable"}
                )
                continue
            symbol = f"merlin_source_{family}_{ordinal}"
            try:
                checked = self._compile_source(original, operation, family, kwargs, symbol)
            except (UnsupportedElementwise, DenseSourceUnavailable) as error:
                rows.append({"ordinal": ordinal, "status": "source_fallback", "reason": str(error)})
                continue
            if checked is None:
                rows.append(
                    {
                        "ordinal": ordinal,
                        "status": "source_fallback",
                        "reason": "checked scheduler found no legal realization within original ceilings",
                    }
                )
                continue
            checked.revalidate()
            if any(
                getattr(op, "sym_name", None) is not None and op.sym_name.data == symbol
                for op in replacement.body.block.ops
            ):
                raise ValueError("source routing generated symbol collides with original source")
            writer = self._backend.emit(checked, symbol, source, directory)
            index = len(operation.inputs)
            if (
                type(writer) is not FreshTensorWriterContract
                or writer.symbol != symbol
                or writer.result_argument != index
                or writer.fully_written_arguments != (index,)
                or writer.declaration_abi != "ranked_c"
                or writer.wrapper_symbol is not None
                or writer.allow_initialized_writers
                or writer.private_workspaces
            ):
                raise ValueError("source routing requires exact qualified fresh-result ABI effects")
            target = targets[ordinal]
            original_operands = tuple(target.operands)
            empty = tensor.EmptyOp([], target.results[0].type)
            Rewriter.insert_op(empty, InsertPoint.before(target))
            operands = [*target.inputs, empty.tensor]
            declaration = func.FuncOp.external(symbol, [value.type for value in operands], [target.results[0].type])
            declaration.attributes["llvm.emit_c_interface"] = builtin.UnitAttr()
            declaration.properties["arg_attrs"] = builtin.ArrayAttr(
                [
                    builtin.DictionaryAttr({"bufferization.access": builtin.StringAttr(access)})
                    for access in (*("read" for _ in target.inputs), "write")
                ]
            )
            call = func.CallOp(symbol, operands, [target.results[0].type])
            replacements.append((operation, call, declaration, empty, original_operands))
            Rewriter.replace_op(target, call)
            replacement.body.block.add_op(declaration)
            results.append(checked)
            writers.append(writer)
            if family == "pointwise":
                evidence = {
                    "shape": list(source.shape),
                    "conformance": checked.conformance,
                    "program_sha256": [sha256(image).hexdigest() for image in checked.images()],
                }
            else:
                evidence = {
                    "shape": [source.m, source.k, source.n],
                    "conformance": checked.compiled.realization,
                    "checked_kernel_sha256": checked.compiled.kernel.digest(),
                    "machine_program": "independent target emitted-program verification required",
                }
            rows.append(
                {"ordinal": ordinal, "status": "source_bound_schedule", "family": family, "symbol": symbol, **evidence}
            )
        # Independently invert every new call before applying the existing
        # fresh-writer lowering; all unrelated operations and carried SSA stay.
        values = {}
        inverse = replacement.clone(value_mapper=values)
        correspondence = dict(zip(replacement.walk(), inverse.walk(), strict=True))
        for source, call, declaration, empty, original_operands in reversed(replacements):
            inverse_call = correspondence[call]
            restored = source.clone(
                value_mapper=dict(zip(source.operands, (values[value] for value in original_operands), strict=True))
            )
            Rewriter.replace_op(inverse_call, restored)
            Rewriter.erase_op(correspondence[empty])
            Rewriter.erase_op(correspondence[declaration])
        inverse.verify()
        if ir_digest(inverse) != original_digest or ir_digest(original) != original_digest:
            raise ValueError("source routing inverse did not preserve the complete original module")
        if results:
            self._backend.finalize(tuple(results), output)
            self._verify()
            self._qualify(tuple(results), output)
            self._verify()
            rewrite_fresh_tensor_writers(replacement, writers)
        replacement.verify()
        selected = output / "selected.mlir" if results else source_path
        if results:
            selected.write_text(ir_text(replacement))
        if sha256_file(source_path) != source_hash:
            raise ValueError("source routing modified original input bytes")
        report_path = output / (
            "source_pointwise_routing.json" if self._families == ("pointwise",) else "shared_source_routing.json"
        )
        report = {
            **self.record(),
            "source_sha256": source_hash,
            "original_module_sha256": original_digest,
            "selected_sha256": sha256_file(selected),
            "members": rows,
            "routed_count": len(results),
            "fallback_count": sum(row["status"] == "source_fallback" for row in rows),
            "inverse": "complete original source restored before fresh-writer lowering",
        }
        write_pretty_json(report_path, report)
        self._prepared = _Prepared(
            source_path,
            source_hash,
            selected,
            sha256_file(selected),
            original,
            original_digest,
            tuple(results),
            report_path,
        )
        self._revalidate_prepared()
        return selected

    def provider_for(self, context):
        """Original provider admission may skip only a checked empty source route."""
        state = self._revalidate_prepared()
        if Path(context.prepared_path).resolve() != state.selected.resolve():
            raise ValueError("provider build selects another prepared source")
        return self._backend.build if state.compiled else None

    def build(self, context):
        builder = self.provider_for(context)
        if builder is None:
            raise ValueError("no source-bound provider was selected")
        result = builder(context)
        self._revalidate_prepared()
        return result


class PointwiseSourceRouting(SharedSourceRouting):
    """Compatibility entry retaining the original pointwise-only selection."""

    def __init__(self, backend=None, *, qualify=None):
        if backend is not None and type(backend) is not PointwiseSourceBackend:
            raise ValueError("source routing requires an explicitly selected pointwise backend")
        super().__init__(backend, qualify=qualify)


def resolve_pointwise_provider(builder, context):
    """Recognize only this original typed owner, never a callback-selected bypass."""
    if type(builder) is MethodType and builder.__func__ is SharedSourceRouting.build:
        if type(builder.__self__) not in (SharedSourceRouting, PointwiseSourceRouting):
            raise ValueError("foreign source routing provider owner")
        selected = builder.__self__.provider_for(context)
        return None if selected is None else builder
    return builder
