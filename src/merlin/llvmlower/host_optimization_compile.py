"""Ordinary compile transport for original host choices and distinct products.

This private bridge is called by the ordinary object producer. The public request
is data only; the original scoped executor owns its reviewed semantic selection.
No reports or callback results issue formal or physical execution authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .host_optimization_request import (
    PreparedHostOptimization,
    optimize_host_llvm,
    prepare_host_optimization,
    write_host_optimization_report,
)


@dataclass(frozen=True)
class HostCompilationOptimization:
    prepared: PreparedHostOptimization
    executor: object
    explicit_selection: bool
    source_path: Path
    selected_source_path: Path

    @classmethod
    def select(cls, source, workdir, *, explicit=None, finished_llvm_only=False):
        from merlin.targetgen.host_optimization_admission import active_selection
        from merlin.targetgen.package_runtime import active_package_executor

        selected = getattr(active_package_executor(), "host_optimization_selection", None)
        original_buffer = active_selection()
        if original_buffer is not None:
            if selected is not None and selected is not original_buffer:
                raise ValueError("host buffer choices differ from the original execution owner")
            selected = original_buffer
        if explicit is not None:
            if selected is not None and selected is not explicit:
                raise ValueError("host optimization override differs from the original execution owner")
            selected = explicit
        if selected is None:
            return None
        prepared = prepare_host_optimization(source, selected, finished_llvm_only=finished_llvm_only)
        selected_source = Path(workdir) / "kernel.host.mlir"
        selected_source.write_text(prepared.selected_mlir, encoding="utf-8")
        return cls(
            prepared,
            active_package_executor(),
            explicit is not None,
            Path(workdir) / "kernel.llvm.mlir",
            selected_source,
        )

    def lowering_options(self):
        options = self.prepared.lowering_options()
        observation = {} if options.get("index_bits") is not None else None
        if observation is not None:
            options["lowering_selection"] = observation
        return options, observation

    def translation_arguments(self):
        return {
            "workdir": str(self.source_path.parent.resolve()),
            "host_optimization": self.prepared.selection.record(),
        }

    def validate_lowering(self, observation):
        if observation is not None and (
            observation.get("index_bits") != self.prepared.selection.context.index_bits
            or not observation.get("effective_pipeline")
        ):
            raise ValueError("selected host lowering lost its actual index-bound pass pipeline")

    def optimize_translated(self, llvm_path, *, observation):
        self.validate_lowering(observation)
        llvm_path = Path(llvm_path)
        llvm, report = optimize_host_llvm(llvm_path.read_text(encoding="utf-8"), self.prepared)
        report["observed_lowering"] = observation
        # Completed translation observations continue to bind their exact raw
        # product. A later selected host transform emits a distinct LLVM file.
        if report["llvm_before_sha256"] != report["llvm_after_sha256"]:
            llvm_path = llvm_path.parent / "kernel.host.ll"
            llvm_path.write_text(llvm, encoding="utf-8")
        return llvm_path, report

    def complete(self, obj, llvm, *, report):
        from merlin.targetgen.host_optimization_admission import active_selection

        if (
            not self.explicit_selection
            and getattr(self.executor, "host_optimization_selection", None) is not self.prepared.selection
            and active_selection() is not self.prepared.selection
        ):
            raise ValueError("original execution owner changed its host optimization selection")
        write_host_optimization_report(
            report,
            selection=self.prepared.selection,
            source_path=self.source_path,
            selected_source_path=self.selected_source_path,
            llvm_path=llvm,
            object_path=obj,
            workdir=self.source_path.parent,
        )
        return obj
