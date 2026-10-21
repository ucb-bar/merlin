"""Consume symbolically checked counted-loop dense schedules in device LLVM.

The original provider supplies exhaustive module projection and emission. The
shared planner never discovers instruction semantics from a backend name. Both
the original and consumed programs are checked without expanding their loops.
This host-only integration is outside frozen compiler-authoring inventories.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from pathlib import Path
from types import FunctionType

from xdsl.dialects.builtin import ModuleOp
from xdsl.parser import Parser
from xdsl.printer import Printer

from merlin.common import invocation_record as I
from merlin.common.dense_float_ir import DenseFloatPrinter
from merlin.common.jsonio import write_pretty_json
from merlin.compile.scheduling import compact_dense as D
from merlin.compile.scheduling import compact_verification as V
from merlin.compile.scheduling import schedule_pipeline as P
from merlin.compile.scheduling.library_sources import library_modules, shared_library_sources
from merlin.compile.scheduling.schedule_pipeline import _rewrite_snapshot
from merlin.sched.ir import Kernel
from merlin.targetgen.oot_starterkit.llvm_context import make_llvm_context
from merlin.transforms.certificates import ir_digest
from merlin.transforms.dense_planning import DenseProblem, DenseResources

from . import counted_llvm_cfg as C
from .counted_llvm_cfg import canonical_counted_kernel
from .host_transform_chain import (
    HostLLVMArtifact,
    HostLLVMEmission,
    HostLLVMStageContract,
    HostLLVMTransformStage,
)
from .source_multioutput import _callback, _check_callback


def _functions():
    return (
        _functions,
        _rewrite_snapshot,
        _callback,
        _check_callback,
        ir_digest,
        I.run,
        I.verify,
        Parser.__init__,
        Parser.parse_module,
        DenseFloatPrinter.print_op,
        make_llvm_context,
        D.schedule_compact_dense,
        D.emit_compact_dense,
        V.verify_compact_dense,
        canonical_counted_kernel,
        library_modules,
        shared_library_sources,
        *(
            fn
            for module in (D, V, C)
            for fn in vars(module).values()
            if type(fn) is FunctionType and fn.__module__ == module.__name__
        ),
        *(
            getattr(CompactDenseLLVMTransform, name)
            for name in (
                "__post_init__",
                "_selection",
                "_callbacks",
                "revalidate",
                "_run",
                "_import",
                "_project",
                "_verify",
                "_plan",
                "transform",
                "verify_emission",
                "stage",
            )
        ),
    )


@dataclass(frozen=True)
class CompactDenseLLVMTransform:
    """Original-host selection for one fully represented dense LLVM module.

    ``project_module`` reads actual current LLVM and returns a counted-loop
    Kernel, or None for an unknown module. It must account for all configuration,
    globals, ABI, arithmetic and effects, rejecting changes outside its domain.
    ``emit_module`` consumes that original module and the selected Kernel and
    returns a new ModuleOp. Original qualification of both callbacks and the
    instruction adapter remains required; complete_effects is only a premise.

    source_storage describes the original allocation when it differs from the
    proposed allocation. Both must be original-owned assignments, not a claim
    that all physical SRAM is free. Solver limits are not performance knobs.
    """

    problem: DenseProblem
    resources: DenseResources
    storage: D.DenseLoopStorage
    adapter: D.DenseLoopAdapter
    project_module: object
    emit_module: object
    translator: HostLLVMArtifact
    semantic_proofs: tuple[HostLLVMArtifact, ...]
    complete_effects: bool
    external_buffers_disjoint: bool
    source_storage: D.DenseLoopStorage | None = None
    implementation_sources: tuple[HostLLVMArtifact, ...] = ()
    enabled: bool = True
    reuse_pe_weights: bool = False
    static_budget: int = 4096
    verification_timeout_ms: int = 20_000
    max_ir_bytes: int = 8 << 20
    max_operations: int = 4096
    timeout_seconds: int = 120
    outer_order: tuple[str, str] = ("m", "n")
    _snapshot: object = field(init=False, repr=False)
    _owners: tuple = field(init=False, repr=False)
    _readers: tuple = field(init=False, repr=False)
    _aliases: tuple = field(init=False, repr=False)
    _library: tuple[HostLLVMArtifact, ...] = field(init=False, repr=False)
    _constructor: tuple = field(init=False, repr=False)

    def __post_init__(self):
        if (
            type(self.problem) is not DenseProblem
            or type(self.resources) is not DenseResources
            or type(self.storage) is not D.DenseLoopStorage
            or self.source_storage is not None
            and type(self.source_storage) is not D.DenseLoopStorage
            or type(self.adapter) is not D.DenseLoopAdapter
            or any(
                type(value) is not bool
                for value in (
                    self.complete_effects,
                    self.external_buffers_disjoint,
                    self.enabled,
                    self.reuse_pe_weights,
                )
            )
            or self.outer_order not in (("m", "n"), ("n", "m"))
            or any(
                type(value) is not int or not 0 < value <= 1 << 30
                for value in (
                    self.static_budget,
                    self.verification_timeout_ms,
                    self.max_ir_bytes,
                    self.max_operations,
                    self.timeout_seconds,
                )
            )
            or type(self.translator) is not HostLLVMArtifact
            or type(self.semantic_proofs) is not tuple
            or not self.semantic_proofs
            or type(self.implementation_sources) is not tuple
            or any(type(item) is not HostLLVMArtifact for item in (*self.semantic_proofs, *self.implementation_sources))
        ):
            raise ValueError("compact LLVM requires typed original selections, evidence and bounded verification")
        object.__setattr__(
            self,
            "_readers",
            tuple(
                (fn, fn.__code__)
                for fn in (
                    _functions,
                    _rewrite_snapshot,
                    _callback,
                    _check_callback,
                )
            ),
        )
        object.__setattr__(self, "_aliases", (Parser, DenseFloatPrinter, make_llvm_context, P))
        # Printer's dataclass-generated constructor has no inspectable source
        # function. Freeze its live code explicitly and pin its defining module.
        constructor = DenseFloatPrinter.__init__
        object.__setattr__(
            self,
            "_constructor",
            (constructor, constructor.__code__, constructor.__defaults__, dict(constructor.__kwdefaults__ or {})),
        )
        object.__setattr__(
            self,
            "_library",
            tuple(
                HostLLVMArtifact.capture(importlib.import_module(name).__file__)
                for name in (*library_modules(), Printer.__module__)
            ),
        )
        object.__setattr__(self, "_snapshot", self._selection())
        object.__setattr__(
            self,
            "_owners",
            tuple(
                (
                    fn,
                    _callback(fn),
                    _rewrite_snapshot(
                        (getattr(fn, "__func__", fn).__defaults__, getattr(fn, "__func__", fn).__kwdefaults__)
                    ),
                )
                for fn in self._callbacks()
            ),
        )
        self.revalidate()

    def _selection(self):
        return _rewrite_snapshot(
            (
                self.problem,
                self.resources,
                self.storage,
                self.source_storage,
                self.adapter,
                self.project_module,
                self.emit_module,
                self.translator,
                self.semantic_proofs,
                self.implementation_sources,
                self.complete_effects,
                self.external_buffers_disjoint,
                self.enabled,
                self.reuse_pe_weights,
                self.static_budget,
                self.verification_timeout_ms,
                self.max_ir_bytes,
                self.max_operations,
                self.timeout_seconds,
                self.outer_order,
            )
        )

    def _callbacks(self):
        return (self.project_module, self.emit_module, self.adapter.encode, self.adapter.project, *_functions())

    def revalidate(self):
        for current, (original, code) in zip(
            (_functions, _rewrite_snapshot, _callback, _check_callback),
            self._readers,
            strict=True,
        ):
            if current is not original or original.__code__ is not code:
                raise ValueError("compact LLVM original reader changed; replan")
        if (Parser, DenseFloatPrinter, make_llvm_context, P) != self._aliases:
            raise ValueError("compact LLVM original observer changed; replan")
        if P._rewrite_snapshot is not _rewrite_snapshot:
            raise ValueError("compact LLVM recursive snapshot binding changed; replan")
        if (
            DenseFloatPrinter.__init__ is not self._constructor[0]
            or self._constructor[0].__code__ is not self._constructor[1]
            or self._constructor[0].__defaults__ != self._constructor[2]
            or self._constructor[0].__kwdefaults__ != self._constructor[3]
        ):
            raise ValueError("compact LLVM printer constructor changed; replan")
        if self._selection() != self._snapshot:
            raise ValueError("compact LLVM original selection changed; replan")
        callbacks = self._callbacks()
        if len(callbacks) != len(self._owners):
            raise ValueError("compact LLVM implementation changed; replan")
        for current, (original, snapshot, defaults) in zip(callbacks, self._owners, strict=True):
            fn = getattr(current, "__func__", current)
            if (
                fn is not getattr(original, "__func__", original)
                or getattr(current, "__self__", None) is not getattr(original, "__self__", None)
                or _rewrite_snapshot((fn.__defaults__, fn.__kwdefaults__)) != defaults
            ):
                raise ValueError("compact LLVM original implementation changed; replan")
            _check_callback(current, snapshot)
        for item in (self.translator, *self.semantic_proofs, *self.implementation_sources, *self._library):
            item.validate()

    def _run(self, arguments, *, source, output):
        self.revalidate()
        result = I.run(
            [str(self.translator.path), *map(str, arguments)],
            directory=output.parent,
            stage="compact_dense_llvm_translation",
            inputs=(source,),
            outputs=(output,),
            dependencies=tuple(item.path for item in (*self.implementation_sources, *self.semantic_proofs)),
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
        )
        self.revalidate()
        if result.returncode:
            raise ValueError(f"compact LLVM translation failed ({result.returncode}): {result.stderr[-2000:]}")

    def _import(self, source, directory):
        if source.stat().st_size > self.max_ir_bytes:
            raise ValueError("compact LLVM input exceeds its IR byte budget")
        directory.mkdir()
        actual = directory / "actual.mlir"
        self._run(["--import-llvm", "--mlir-print-op-generic", source, "-o", actual], source=source, output=actual)
        if actual.stat().st_size > self.max_ir_bytes:
            raise ValueError("imported compact LLVM exceeds its IR byte budget")
        module = Parser(make_llvm_context(), actual.read_text()).parse_module()
        module.verify()
        if sum(1 for _ in module.walk()) > self.max_operations:
            raise ValueError("compact LLVM module exceeds its operation budget")
        return module

    def _project(self, module):
        self.revalidate()
        before = ir_digest(module)
        result = self.project_module(module)
        self.revalidate()
        if ir_digest(module) != before:
            raise ValueError("compact LLVM projection changed the actual input")
        if result is not None and type(result) is not Kernel:
            raise ValueError("compact LLVM projection must return actual counted-loop Kernel or None")
        return result

    def _verify(self, kernel, storage):
        self.revalidate()
        report = V.verify_compact_dense(
            kernel,
            self.problem,
            self.resources,
            storage,
            self.adapter,
            static_budget=self.static_budget,
            timeout_ms=self.verification_timeout_ms,
        )
        self.revalidate()
        return report

    def _plan(self, module):
        if not self.complete_effects:
            return None, {"status": "original_llvm_retained", "reason": "complete module effects unavailable"}
        if not self.external_buffers_disjoint:
            return None, {"status": "original_llvm_retained", "reason": "external buffer disjointness unavailable"}
        if not self.adapter.synchronous_completion or not self.adapter.exact_accumulator_continuation:
            return None, {"status": "original_llvm_retained", "reason": "compact execution semantics unavailable"}
        source = self._project(module)
        if source is None:
            return None, {"status": "original_llvm_retained", "reason": "unknown module or effects"}
        baseline = self._verify(source, self.source_storage or self.storage)
        if not self.enabled:
            return None, {"status": "original_llvm_retained", "reason": "optimization disabled", "baseline": baseline}
        selected = D.schedule_compact_dense(
            self.problem,
            self.resources,
            self.storage,
            self.adapter,
            name=source.name,
            dtypes=tuple(arg.dtype for arg in source.args),
            static_budget=self.static_budget,
            external_buffers_disjoint=self.external_buffers_disjoint,
            verification_timeout_ms=self.verification_timeout_ms,
            outer_order=self.outer_order,
            reuse_pe_weights=self.reuse_pe_weights,
        )
        self.revalidate()
        if selected is None:
            return None, {
                "status": "original_llvm_retained",
                "reason": "no admitted compact candidate",
                "baseline": baseline,
            }
        selected.revalidate()
        if selected.kernel.digest() == source.digest():
            return None, {
                "status": "original_llvm_retained",
                "reason": "unchanged compact candidate",
                "baseline": baseline,
            }
        return selected, {"status": "compact_candidate", "baseline": baseline, "selected": selected.realization}

    def transform(self, source, workdir):
        source, workdir = Path(source), Path(workdir)
        self.revalidate()
        original = self._import(source, workdir / "original")
        selected, report = self._plan(original)
        write_pretty_json(workdir / "proposal.json", report)
        if selected is None:
            return source
        before, kernel_before = ir_digest(original), selected.kernel.digest()
        module = self.emit_module(original, selected.kernel)
        self.revalidate()
        selected.revalidate()
        if ir_digest(original) != before or selected.kernel.digest() != kernel_before or type(module) is not ModuleOp:
            raise ValueError("compact LLVM emitter changed its inputs or returned an invalid module")
        module.verify()
        actual = self._project(module)
        if (
            actual is None
            or canonical_counted_kernel(actual).digest() != canonical_counted_kernel(selected.kernel).digest()
        ):
            raise ValueError("compact LLVM emitter did not consume the selected schedule")
        self._verify(actual, self.storage)
        typed, output = workdir / "selected.mlir", workdir / "selected.ll"
        with typed.open("x") as stream:
            DenseFloatPrinter(stream=stream, print_generic_format=True).print_op(module)
        if typed.stat().st_size > self.max_ir_bytes or sum(1 for _ in module.walk()) > self.max_operations:
            raise ValueError("selected compact LLVM exceeds its static IR budget")
        self._run(["--mlir-to-llvmir", typed, "-o", output], source=typed, output=output)
        return output

    def verify_emission(self, source, selected, workdir):
        """Check the actual LLVM file consumed by ordinary object compilation."""
        source, selected, workdir = Path(source), Path(selected), Path(workdir)
        original_input = HostLLVMArtifact.capture(source)
        actual_output = HostLLVMArtifact.capture(selected)
        original = self._import(source, workdir / "verify_original")
        planned, report = self._plan(original)
        if planned is None:
            if source.read_bytes() != selected.read_bytes():
                raise ValueError("unavailable compact selection must preserve exact original LLVM bytes")
        else:
            actual = self._project(self._import(selected, workdir / "verify_actual"))
            if (
                actual is None
                or canonical_counted_kernel(actual).digest() != canonical_counted_kernel(planned.kernel).digest()
            ):
                raise ValueError("actual consumed LLVM differs from the selected compact schedule")
            report["actual"] = self._verify(actual, self.storage)
            report["status"] = "actual_compact_llvm_checked"
        self.revalidate()
        evidence = workdir / "actual-emission.json"
        write_pretty_json(evidence, report)
        invocations = tuple(sorted(workdir.rglob("invocation.json")))
        for path in invocations:
            I.verify(path)
        self.revalidate()
        original_input.validate()
        actual_output.validate()
        return HostLLVMEmission(
            original_input,
            actual_output,
            tuple(HostLLVMArtifact.capture(path) for path in (evidence, *invocations)),
        )

    def stage(self, name="compact_dense"):
        self.revalidate()
        paths = {
            Path(__file__).resolve(),
            self.translator.path,
            Path(D.__file__).resolve(),
            Path(V.__file__).resolve(),
            Path(importlib.import_module(Printer.__module__).__file__).resolve(),
        }
        paths.update(snapshot[3] for _, snapshot, _ in self._owners)
        sources = {item.path: item for item in self.implementation_sources}
        sources.update((item.path, item) for item in self._library)
        sources.update((Path(path).resolve(), HostLLVMArtifact.capture(path)) for path in paths)
        contract = HostLLVMStageContract(name, tuple(sources.values()), self.semantic_proofs)
        stage = HostLLVMTransformStage(contract, self.transform, self.verify_emission)
        stage.validate()
        return stage
