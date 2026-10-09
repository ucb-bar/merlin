"""Evaluator-only CPU instrumentation fixture; never an author/runtime seed.

Events sample native stock-CPU counter/control getters. All stage labels here
exercise only framing/accounting: they do not establish actual cost boundaries.
"""

import sys
from pathlib import Path

from merlin_experiments.phase2 import component_runtime_controls as controls

from merlin.common import invocation_record as I
from merlin.perf.component_cost import COMPLETE_STAGES
from merlin.runtime.direct_kernel_harness import render_direct_kernel
from merlin.targetgen import package_runtime as P


class ActualCommands:
    def build_package(self, package, **kwargs):
        assert not package.manifest.get("build")

    def run_entrypoint(self, package, name, source, output=None, *, timeout, invocation_directory, **kwargs):
        argv = P._resolve_argv(package, name, source, output)
        return I.run(
            [sys.executable, "-I", "-B", *argv[1:]],
            directory=invocation_directory,
            stage=name,
            inputs=(Path(source),),
            outputs=(Path(output),) if output is not None else (),
            dependencies=tuple(package.directory.rglob("*.py")),
            cwd=package.directory,
            env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"},
            capture_output=True,
            text=True,
            timeout=timeout,
        )


def verify_original(*, source, lowered_mlir, **kwargs):
    return controls.verify_primitive_llvm(source.read_text(), lowered_mlir.read_text(), entry_symbol="control_entry")


def render(cb, *, inputs, readback_policy, abi):
    return render_direct_kernel(cb, inputs=inputs, readback_policy=readback_policy, abi=abi)


def support_source(mode):
    stages = list(COMPLETE_STAGES)
    if mode == "missing":
        stages.remove("packing")
    elif mode == "moved":
        stages[1], stages[2] = stages[2], stages[1]
    elif mode != "complete":
        raise ValueError("unsupported diagnostic mode")
    calls = "\n".join(f' sample("{stage} begin"); sample("{stage} end");' for stage in stages)
    return (
        """#include "htif.h"
static void sample(const char *boundary) {
  unsigned long counter, control;
  __asm__ volatile("csrr %0, mcycle" : "=r"(counter) :: "memory");
  __asm__ volatile("csrr %0, mcountinhibit" : "=r"(control) :: "memory");
  htif_puts(boundary); htif_putc(' '); htif_putd(counter);
  htif_putc(' '); htif_putd(control); htif_putc('\\n');
}
extern void __real_htif_exit(int code) __attribute__((noreturn));
void __wrap_htif_exit(int code) {
  if (code) __real_htif_exit(code);
  htif_puts("MERLIN_MEASUREMENT_V1\\n");
"""
        + calls
        + """
  htif_puts("MERLIN_MEASUREMENT_END_V1\\n");
  __real_htif_exit(code);
}
"""
    )
