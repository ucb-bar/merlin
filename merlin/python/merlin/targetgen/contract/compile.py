"""Runner-owned compile + execute of a *package-produced* lowered LLVM/RoCC MLIR.

The contract splits responsibility: the package emits ``lowered.llvm.mlir`` (a module defining a
kernel function under the entry symbol its contract declares, with the kernel ABI in
``mlir_oot_backend_contract.yaml``); the runner owns the harness (which embeds the deterministic leaf
tensors by name + output buffers and prints ``OUT/METRIC/DONE``), the link, and the oracle
invocation. This path is uniform for Python and C++ packages — the only difference is who produced
the MLIR.

**This module names no target and imports none.** ``target`` is a required argument throughout, and
everything target-specific is resolved through it: the harness ABI from the target's contract
(:mod:`.harness_abi`), and the harness renderer, build recipe and oracle from its backend via
:mod:`merlin.runtime.backends.base`. What remains here is orchestration — lower, render, link, run.
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Any


def llvm_mlir_to_object(lowered_mlir_text: str, workdir: Path) -> Path:
    """Lower package-emitted llvm-dialect MLIR to a rv64gcv object (.o)."""
    from merlin.llvmlower.pipeline import lower_to_llvm_ir
    from merlin.llvmlower import codegen
    workdir.mkdir(parents=True, exist_ok=True)
    ll = lower_to_llvm_ir(lowered_mlir_text, workdir=workdir)
    (workdir / "kernel.ll").write_text(ll, encoding="utf-8")
    return Path(codegen.compile_ll(workdir / "kernel.ll", workdir / "kernel.o", "riscv"))


def link_elf(cb: dict[str, Any], obj: Path, workdir: Path, *, target: str) -> Path:
    """Build the runner-owned harness from ``cb`` and link it with the package object -> ELF.

    Orchestration only: the harness TEXT comes from ``target``'s declared harness ABI and the BUILD
    from its declared recipe, both resolved through the backend registry. This module names no target
    and imports no target's module — ``target`` is a required argument precisely so no default can
    reintroduce one.
    """
    from merlin.runtime.backends import base as _backends
    recipe = _backends.harness_build_recipe(target)
    harness = _backends.harness_renderer(target)(cb, target=target)
    (workdir / "harness.c").write_text(harness, encoding="utf-8")
    # Linker load address DERIVED from the RTL memory map (platform DRAM base), reusing the curated
    # script's proven section layout but replacing its BAKED origin — so the base is a HW fact, not a
    # hardcoded literal in a vendored file.
    from ..runtime_build import derived_link_script
    link_ld = derived_link_script(recipe.load_address, recipe.link_script, Path(workdir))
    elf = workdir / "package_kernel.elf"
    cmd = recipe.command(sources=[workdir / "harness.c", obj], output=elf, link_script=link_ld)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise recipe.error_cls(f"link failed:\n{proc.stderr[-2000:]}")
    return elf


def compile_lowered_to_elf(cb: dict[str, Any], lowered_mlir_text: str,
                           workdir: str | Path | None = None, *, target: str) -> Path:
    """Full package-lowered-MLIR -> rv64 ELF (object + runner harness + link)."""
    work = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="oot_compile_"))
    obj = llvm_mlir_to_object(lowered_mlir_text, work)
    return link_elf(cb, obj, work, target=target)


def run_on_oracle(cb: dict[str, Any], lowered_mlir_text: str, *, simulator: str, target: str,
                  workdir: str | Path | None = None, timeout: int = 600) -> dict[str, Any]:
    """Compile the package's lowered MLIR + run on ``simulator``; return outputs/metrics/console.

    ``timing`` splits the work: ``build_s`` (ELF compile/link) and ``sim_active_s`` (the simulator
    subprocess) are *active* time; ``oracle_wait_s`` is queue/FPGA-slot wait (0 for local sims like
    spike/verilator — only VCS/FireSim adapters that route through a queue set it).
    """
    import time
    from merlin.runtime.backends import base as _backends
    backend = _backends.get_backend(target)
    work = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="oot_run_"))
    _t0 = time.perf_counter()
    elf = compile_lowered_to_elf(cb, lowered_mlir_text, work, target=target)
    _t1 = time.perf_counter()
    console = backend.run_elf(elf, simulator=simulator, timeout=timeout)
    _t2 = time.perf_counter()
    outputs, raw = backend.parse_output(console)
    return {"outputs": outputs, "raw_metrics": raw, "cycles": raw.get("cycles", 0),
            "oracle": backend.ORACLE[simulator], "elf": str(elf), "console": console,
            "timing": {"build_s": round(_t1 - _t0, 3), "sim_active_s": round(_t2 - _t1, 3),
                       "oracle_wait_s": 0.0}}
