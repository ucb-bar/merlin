"""Consume checked staged contraction rewrites in the ordinary LLVM object path.

The selected provider owns the exhaustive instruction/effect adapter. This
bridge imports actual LLVM, runs existing shared algorithms and checks the
translated output again. It supplies neither instruction meanings nor target
qualification. Its stage composes through the existing host transform chain.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from xdsl.parser import Parser

from merlin.common import invocation_record as I
from merlin.common.dense_float_ir import DenseFloatPrinter
from merlin.common.jsonio import write_pretty_json
from merlin.compile.scheduling.schedule_pipeline import _rewrite_snapshot
from merlin.compile.scheduling.staged_dense import StagedDenseContract, StagedDenseProblem, StagedDenseRewrite
from merlin.targetgen.oot_starterkit.llvm_context import make_llvm_context
from merlin.transforms.pipeline_state import PassSpec, PipelineContext, PipelineSession

from . import target_staged_dense as T
from .host_transform_chain import (
    HostLLVMArtifact,
    HostLLVMEmission,
    HostLLVMStageContract,
    HostLLVMTransformStage,
)
from .source_multioutput import _callback, _check_callback
from .target_staged_dense import (
    StagedDenseTargetAdapter,
    rewrite_target_staged_dense,
    verify_target_staged_dense_emission,
)


def _functions():
    return (
        _functions,
        StagedDenseLLVMTransform.__post_init__,
        StagedDenseLLVMTransform._bounds,
        StagedDenseLLVMTransform._callbacks,
        StagedDenseLLVMTransform._selection_record,
        StagedDenseLLVMTransform.revalidate,
        StagedDenseLLVMTransform._kwargs,
        StagedDenseLLVMTransform._run,
        StagedDenseLLVMTransform._import,
        StagedDenseLLVMTransform._rewrite,
        StagedDenseLLVMTransform.transform,
        StagedDenseLLVMTransform.verify_emission,
        StagedDenseLLVMTransform.stage,
        I.run,
        I.verify,
        PipelineSession.__init__,
        PipelineSession.run,
        PipelineSession.analysis,
        PipelineSession.stamp,
        PassSpec.__post_init__,
        PipelineContext.__post_init__,
        *T._functions(),
        _rewrite_snapshot,
    )


@dataclass(frozen=True)
class StagedDenseLLVMTransform:
    """An original host's bounded selection, outside frozen authoring SDKs.

    ``semantic_proofs`` retains the original provider's selected evidence; the
    files' presence is not an independent qualification. Unknown projection
    retains the byte-identical LLVM input. Bounds are verification limits, not
    performance knobs. The provider must preserve compact loops when emitting.
    """

    problem: StagedDenseProblem
    contract: StagedDenseContract
    adapter: StagedDenseTargetAdapter
    translator: HostLLVMArtifact
    semantic_proofs: tuple[HostLLVMArtifact, ...]
    rewrites: tuple[StagedDenseRewrite, ...] = ()
    implementation_sources: tuple[HostLLVMArtifact, ...] = ()
    max_commands: int = 1_000_000
    max_products: int = 4 << 20
    max_ir_bytes: int = 8 << 20
    max_operations: int = 4096
    timeout_seconds: int = 120
    _record: str = field(init=False, repr=False)
    _owners: tuple = field(init=False, repr=False)
    _readers: tuple = field(init=False, repr=False)
    _aliases: tuple = field(init=False, repr=False)
    _observers: tuple = field(init=False, repr=False)

    def __post_init__(self):
        if (
            type(self.problem) is not StagedDenseProblem
            or type(self.contract) is not StagedDenseContract
            or type(self.adapter) is not StagedDenseTargetAdapter
            or type(self.translator) is not HostLLVMArtifact
            or type(self.rewrites) is not tuple
            or any(type(item) is not StagedDenseRewrite for item in self.rewrites)
            or type(self.implementation_sources) is not tuple
            or type(self.semantic_proofs) is not tuple
            or not self.semantic_proofs
            or any(type(item) is not HostLLVMArtifact for item in (*self.implementation_sources, *self.semantic_proofs))
            or any(type(value) is not int or not 0 < value <= 1 << 30 for value in self._bounds().values())
        ):
            raise ValueError("staged LLVM requires original typed contracts, tools, evidence and positive bounds")
        object.__setattr__(
            self,
            "_readers",
            tuple((fn, fn.__code__) for fn in (_functions, T._functions, _rewrite_snapshot, _check_callback)),
        )
        object.__setattr__(self, "_aliases", (Parser, DenseFloatPrinter, make_llvm_context))
        object.__setattr__(
            self,
            "_observers",
            tuple(
                (fn, fn.__code__, tuple(map(id, fn.__defaults__ or ())), _rewrite_snapshot(fn.__kwdefaults__))
                for fn in (Parser.__init__, Parser.parse_module, DenseFloatPrinter.__init__, DenseFloatPrinter.print_op)
            ),
        )
        object.__setattr__(self, "_record", self._selection_record())
        object.__setattr__(
            self,
            "_owners",
            tuple(
                (callback, _callback(callback), _rewrite_snapshot((fn.__defaults__, fn.__kwdefaults__)))
                for callback in self._callbacks()
                for fn in (getattr(callback, "__func__", callback),)
            ),
        )
        self.revalidate()

    def _bounds(self):
        return {
            "max_commands": self.max_commands,
            "max_products": self.max_products,
            "max_ir_bytes": self.max_ir_bytes,
            "max_operations": self.max_operations,
            "timeout_seconds": self.timeout_seconds,
        }

    def _callbacks(self):
        return (
            self.adapter.project,
            self.adapter.emit,
            *(item.implementation for item in self.rewrites),
            *_functions(),
            rewrite_target_staged_dense,
            verify_target_staged_dense_emission,
            make_llvm_context,
            _callback,
            _check_callback,
        )

    def _selection_record(self):
        callbacks = []
        for fn in self._callbacks():
            _, target, _, path, digest = _callback(fn)
            callbacks.append((target.__qualname__, str(path), digest))
        return json.dumps(
            {
                "problem": asdict(self.problem),
                "contract": asdict(self.contract),
                "complete_effects": self.adapter.complete_effects,
                "rewrites": [item.name for item in self.rewrites],
                "callbacks": callbacks,
                "bounds": self._bounds(),
                "translator": self.translator.to_dict(),
                "sources": [item.to_dict() for item in self.implementation_sources],
                "evidence": [item.to_dict() for item in self.semantic_proofs],
            },
            sort_keys=True,
            allow_nan=False,
        )

    def revalidate(self):
        for current, (original, code) in zip(
            (_functions, T._functions, _rewrite_snapshot, _check_callback), self._readers, strict=True
        ):
            if current is not original or original.__code__ is not code:
                raise ValueError("staged LLVM implementation reader changed; replan")
        if any(
            current is not original
            for current, original in zip((Parser, DenseFloatPrinter, make_llvm_context), self._aliases, strict=True)
        ):
            raise ValueError("staged LLVM original observer changed; replan")
        for fn, (original, code, defaults, keywords) in zip(
            (Parser.__init__, Parser.parse_module, DenseFloatPrinter.__init__, DenseFloatPrinter.print_op),
            self._observers,
            strict=True,
        ):
            if (
                fn is not original
                or fn.__code__ is not code
                or tuple(map(id, fn.__defaults__ or ())) != defaults
                or _rewrite_snapshot(fn.__kwdefaults__) != keywords
            ):
                raise ValueError("staged LLVM original observer implementation changed; replan")
        if self._selection_record() != self._record:
            raise ValueError("staged LLVM selection changed; replan")
        for callback, (original, snapshot, defaults) in zip(self._callbacks(), self._owners, strict=True):
            fn = getattr(callback, "__func__", callback)
            if (
                getattr(callback, "__self__", None) is not getattr(original, "__self__", None)
                or fn is not getattr(original, "__func__", original)
                or _rewrite_snapshot((fn.__defaults__, fn.__kwdefaults__)) != defaults
            ):
                raise ValueError("staged LLVM original callback changed; replan")
            _check_callback(callback, snapshot)
        for item in (self.translator, *self.semantic_proofs, *self.implementation_sources):
            item.validate()

    def _kwargs(self):
        return {
            "problem": self.problem,
            "contract": self.contract,
            "adapter": self.adapter,
            **{key: value for key, value in self._bounds().items() if key != "timeout_seconds"},
        }

    def _run(self, arguments, *, source, output):
        self.revalidate()
        result = I.run(
            [str(self.translator.path), *map(str, arguments)],
            directory=output.parent,
            stage="staged_dense_llvm_translation",
            inputs=(source,),
            outputs=(output,),
            dependencies=tuple(item.path for item in (*self.implementation_sources, *self.semantic_proofs)),
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
        )
        self.revalidate()
        if result.returncode:
            raise ValueError(f"staged LLVM translation failed ({result.returncode}): {result.stderr[-2000:]}")

    def _import(self, source, directory):
        if source.stat().st_size > self.max_ir_bytes:
            raise ValueError("staged LLVM input exceeds its explicit IR byte budget")
        directory.mkdir()
        path = directory / "actual.mlir"
        self._run(["--import-llvm", "--mlir-print-op-generic", source, "-o", path], source=source, output=path)
        if path.stat().st_size > self.max_ir_bytes:
            raise ValueError("imported staged LLVM exceeds its explicit IR byte budget")
        module = Parser(make_llvm_context(), path.read_text()).parse_module()
        module.verify()
        return module

    def _rewrite(self, module):
        context = PipelineContext(
            hardware=self.contract.hardware,
            numerical_policy=self.contract.numerical_policy,
            resources=json.dumps(asdict(self.contract), sort_keys=True),
            execution=self.contract.execution,
            library=hashlib.sha256(
                json.dumps(sorted({(str(snapshot[3]), snapshot[4]) for _, snapshot, _ in self._owners})).encode()
            ).hexdigest(),
        )
        session = PipelineSession(module, context=context, ir_level="device_llvm", max_repairs=0)
        session.run(
            PassSpec("bind_device_llvm", 0, produces=("current_llvm",), mutates_ir=False, ir_level="device_llvm"),
            lambda: {"current_llvm": session.stamp().source_sha256},
        )

        def apply():
            result = rewrite_target_staged_dense(module, rewrites=self.rewrites, **self._kwargs())
            result.revalidate()
            session.module = result.module
            return {"checked_staged_ir": result}

        session.run(
            PassSpec(
                "rewrite_staged_device_llvm",
                1,
                requires=("current_llvm",),
                produces=("checked_staged_ir",),
                invalidates=("*",),
                ir_level="device_llvm",
            ),
            apply,
        )
        result = session.analysis("checked_staged_ir")
        result.revalidate()
        return result, [asdict(record) for record in session.records]

    def transform(self, source, workdir):
        source, workdir = Path(source), Path(workdir)
        self.revalidate()
        if source.stat().st_size > self.max_ir_bytes:
            raise ValueError("staged LLVM input exceeds its explicit IR byte budget")
        original = self._import(source, workdir / "original")
        result, pipeline = self._rewrite(original)
        result.revalidate()
        write_pretty_json(workdir / "proposal.json", {**result.report, "pipeline": pipeline})
        # An unavailable adapter is a real fallback, not a roundtrip rewrite.
        if result.source_operations is None or result.source_operations == result.selected_operations:
            self.revalidate()
            return source
        typed, selected = workdir / "selected.mlir", workdir / "selected.ll"
        with typed.open("x") as stream:
            DenseFloatPrinter(stream=stream, print_generic_format=True).print_op(result.module)
        self._run(["--mlir-to-llvmir", typed, "-o", selected], source=typed, output=selected)
        result.revalidate()
        self.revalidate()
        return selected

    def verify_emission(self, source, selected, workdir):
        """Reimport the actual file that object compilation will consume."""
        source, selected, workdir = Path(source), Path(selected), Path(workdir)
        self.revalidate()
        original = self._import(source, workdir / "verify_original")
        planned, pipeline = self._rewrite(original)
        planned.revalidate()
        if planned.source_operations is None or planned.source_operations == planned.selected_operations:
            if source.read_bytes() != selected.read_bytes():
                raise ValueError("unavailable or unchanged staged selection must preserve exact LLVM bytes")
            report = {"status": "original_llvm_retained", "proposal": planned.report}
        else:
            actual = self._import(selected, workdir / "verify_actual")
            report = verify_target_staged_dense_emission(
                original, actual, selected_operations=planned.selected_operations, **self._kwargs()
            )
        report["pipeline"] = pipeline
        planned.revalidate()
        self.revalidate()
        evidence = workdir / "actual-emission.json"
        write_pretty_json(evidence, report)
        invocations = tuple(sorted(workdir.rglob("invocation.json")))
        for path in invocations:
            I.verify(path)
        return HostLLVMEmission(
            HostLLVMArtifact.capture(source),
            HostLLVMArtifact.capture(selected),
            tuple(HostLLVMArtifact.capture(path) for path in (evidence, *invocations)),
        )

    def stage(self, name="staged_dense"):
        """Use with DeviceRouting.llvm_transform_chain or an existing chain."""
        self.revalidate()
        paths = [Path(__file__), self.translator.path]
        paths.extend(snapshot[3] for _, snapshot, _ in self._owners)
        paths.extend(Path(path) for fn, *_ in self._observers if (path := inspect.getsourcefile(fn)) != "<string>")
        paths.extend(Path(inspect.getsourcefile(owner)) for owner in (Parser, DenseFloatPrinter, make_llvm_context))
        sources = {item.path: item for item in self.implementation_sources}
        sources.update((path.resolve(), HostLLVMArtifact.capture(path)) for path in paths)
        contract = HostLLVMStageContract(name, tuple(sources.values()), self.semantic_proofs)
        stage = HostLLVMTransformStage(contract, self.transform, self.verify_emission)
        stage.validate()
        return stage
