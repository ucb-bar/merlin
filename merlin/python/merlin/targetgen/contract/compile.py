"""Runner-owned compile + execute of a *package-produced* lowered LLVM/RoCC MLIR.

The contract splits responsibility: the package emits ``lowered.llvm.mlir`` (a module defining
``llvm.func @gemmini_kernel`` with the kernel ABI in ``mlir_oot_backend_contract.yaml``); the
runner owns the harness (which embeds the deterministic leaf tensors by name + output buffers and
prints ``OUT/METRIC/DONE``), the link, and the oracle invocation. This path is uniform for Python
and C++ packages — the only difference is who produced the MLIR.

It deliberately reuses the proven low-level pieces from the certified MLIR-faithful path
(``lower_to_llvm_ir`` + ``codegen.compile_ll`` + ``_harness_c`` + the gemmini backend link) rather
than reimplementing them — but takes the lowered MLIR as input instead of re-deriving it.
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


def _is_movement_cb(cb: dict[str, Any]) -> bool:
    cmds = cb.get("commands", [])
    return (not any(c.get("opcode") == "RES_PACK" for c in cmds)
            and any(c.get("opcode") == "VECTOR_MAP"
                    and c.get("attributes", {}).get("combine") == "identity" for c in cmds))


def _movement_harness_c(cb: dict[str, Any]) -> str:
    """Harness for a pure-movement kernel gemmini_kernel(src*, dst*): embed src, print dst."""
    from merlin.runtime.backends.gemmini_codegen import _ceil_dim, _pad_rowmajor
    from merlin.runtime.commandbuffer import materialize_inputs
    mv = next(c for c in cb["commands"]
              if c.get("opcode") == "VECTOR_MAP" and c["attributes"].get("combine") == "identity")
    src, dst = mv["operands"]["lhs"], mv["operands"]["dst"]
    m, n = cb["tensors"][src]["shape"]
    mp, np_ = _ceil_dim(m), _ceil_dim(n)
    leaves = materialize_inputs(cb)
    sp = _pad_rowmajor(list(leaves[src].data), m, n, mp, np_)
    decls = [f"static const elem_t T_{src}[{mp * np_}] row_align(1) = "
             f"{{{','.join(str(int(v)) for v in sp)}}};",
             f"static elem_t T_{dst}[{mp * np_}] row_align(1);"]
    prints = [f'  printf("OUT {dst} {m} {n}");',
              f"  for (long i = 0; i < {m}; i++) for (long j = 0; j < {n}; j++)"
              f" printf(\" %d\", (int)T_{dst}[i * {np_} + j]);", '  printf("\\n");']
    # Print METRIC cycles BEFORE the (possibly huge) OUT tensor dump: large-output kernels flood the
    # UART and the per-ELF capture truncates mid-dump, so a trailing METRIC line would be lost. Emitting
    # the (tiny) cycle metric first guarantees it is always captured; the OUT dump follows for correctness.
    return ("#include <stdint.h>\n#include <stdio.h>\n#include \"include/gemmini_testutils.h\"\n"
            "extern void gemmini_kernel();\n" + "\n".join(decls) + "\nint main() {\n"
            "  uint64_t c0 = read_cycles();\n"
            f"  gemmini_kernel((void*)T_{src}, (void*)T_{dst});\n  gemmini_fence();\n"
            "  uint64_t c1 = read_cycles();\n"
            '  printf("METRIC cycles %lu\\n", (unsigned long)(c1 - c0));\n'
            '  printf("METRIC cycle_window_gemmini_region 1\\n");\n'
            + "\n".join(prints) + "\n"
            '  printf("DONE\\n");\n  return 0;\n}\n')


def link_elf(cb: dict[str, Any], obj: Path, workdir: Path) -> Path:
    """Build the runner-owned harness from ``cb`` and link it with the package object -> ELF."""
    from merlin.runtime.backends import gemmini as gem
    from merlin.runtime.backends.gemmini_codegen_mlir import _harness_c
    harness = _movement_harness_c(cb) if _is_movement_cb(cb) else _harness_c(cb)
    (workdir / "harness.c").write_text(harness, encoding="utf-8")
    rt, common = gem.rocc_tests_dir(), gem._common_dir()
    # Linker load address DERIVED from the RTL memory map (platform DRAM base), reusing the curated
    # script's proven section layout but replacing its BAKED origin — so the base is a HW fact, not a
    # hardcoded literal in a vendored file.
    from ..runtime_build import derived_link_script
    link_ld = derived_link_script(gem.platform_dram_base(), common / "test.ld", Path(workdir))
    elf = workdir / "package_kernel.elf"
    cmd = [str(gem.gcc_path()), "-DPREALLOCATE=1", "-DMULTITHREAD=1", "-mcmodel=medany",
           "-std=gnu99", "-O2", "-ffast-math", "-fno-common", "-fno-builtin-printf",
           "-fno-tree-loop-distribute-patterns", "-march=rv64gc", "-Wa,-march=rv64gc",
           "-lm", "-lgcc", "-I", str(rt / "riscv-tests"), "-I", str(rt / "riscv-tests/env"),
           "-I", str(rt), "-I", str(common), "-DID_STRING=", "-DPRINT_TILE=0",
           "-nostdlib", "-nostartfiles", "-static", "-T", str(link_ld), "-DBAREMETAL=1",
           str(workdir / "harness.c"), str(obj), "-o", str(elf),
           *(str(p) for p in sorted(common.glob("*.c"))),
           *(str(p) for p in sorted(common.glob("*.S")))]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise gem.GemminiError(f"link failed:\n{proc.stderr[-2000:]}")
    return elf


def compile_lowered_to_elf(cb: dict[str, Any], lowered_mlir_text: str,
                           workdir: str | Path | None = None) -> Path:
    """Full package-lowered-MLIR -> rv64 ELF (object + runner harness + link)."""
    work = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="oot_compile_"))
    obj = llvm_mlir_to_object(lowered_mlir_text, work)
    return link_elf(cb, obj, work)


def run_on_oracle(cb: dict[str, Any], lowered_mlir_text: str, *, simulator: str,
                  workdir: str | Path | None = None, timeout: int = 600) -> dict[str, Any]:
    """Compile the package's lowered MLIR + run on ``simulator``; return outputs/metrics/console.

    ``timing`` splits the work: ``build_s`` (ELF compile/link) and ``sim_active_s`` (the simulator
    subprocess) are *active* time; ``oracle_wait_s`` is queue/FPGA-slot wait (0 for local sims like
    spike/verilator — only VCS/FireSim adapters that route through a queue set it).
    """
    import time
    from merlin.runtime.backends import gemmini as gem
    work = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="oot_run_"))
    _t0 = time.perf_counter()
    elf = compile_lowered_to_elf(cb, lowered_mlir_text, work)
    _t1 = time.perf_counter()
    console = gem.run_elf(elf, simulator=simulator, timeout=timeout)
    _t2 = time.perf_counter()
    outputs, raw = gem.parse_output(console)
    return {"outputs": outputs, "raw_metrics": raw, "cycles": raw.get("cycles", 0),
            "oracle": gem.ORACLE[simulator], "elf": str(elf), "console": console,
            "timing": {"build_s": round(_t1 - _t0, 3), "sim_active_s": round(_t2 - _t1, 3),
                       "oracle_wait_s": 0.0}}
