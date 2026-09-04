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


def link_elf(cb: dict[str, Any], obj: Path, workdir: Path, *, target: str,
             inputs: dict | None = None) -> Path:
    """Build the runner-owned harness from ``cb`` and link it with the package object -> ELF.

    Orchestration only: the harness TEXT comes from ``target``'s declared harness ABI and the BUILD
    from its declared recipe, both resolved through the backend registry. This module names no target
    and imports no target's module — ``target`` is a required argument precisely so no default can
    reintroduce one.
    """
    from merlin.runtime.backends import base as _backends
    recipe = _backends.harness_build_recipe(target)
    # ``inputs`` INJECTS the caller's real operands into the device harness. A renderer written before
    # this parameter existed still works and still materializes from names -- but silently doing that
    # while the reference and simulator use injected data produces a guaranteed three-way mismatch that
    # reads as a functional failure of the TARGET, so an injecting caller is told instead.
    _render = _backends.harness_renderer(target)
    if inputs:
        import inspect
        if "inputs" not in inspect.signature(_render).parameters:
            raise NotImplementedError(
                f"backend for target {target!r} declares a render_harness that cannot take `inputs`, so "
                f"the device would compute on name-materialized operands while the reference and the "
                f"simulator use the injected ones. Add an `inputs` parameter to its render_harness.")
        harness = _render(cb, target=target, inputs=inputs)
    else:
        harness = _render(cb, target=target)
    (workdir / "harness.c").write_text(harness, encoding="utf-8")
    # Linker load address DERIVED from the RTL memory map (platform DRAM base), reusing the curated
    # script's proven section layout but replacing its BAKED origin — so the base is a HW fact, not a
    # hardcoded literal in a vendored file.
    from ..runtime_build import derived_link_script
    link_ld = derived_link_script(recipe.load_address, recipe.link_script, Path(workdir))
    elf = workdir / "package_kernel.elf"
    # REPRODUCIBLE BUILD, in two phases. A single compile+link invocation lets the driver name its
    # intermediate objects `ccXXXXXX.o`, and those random names are recorded in the ELF as STT_FILE
    # symbols -- so two builds of byte-identical sources differ (measured: 6 bytes) while producing
    # identical cycles. That defeats content-addressed reuse of a measurement for no reason. Naming
    # each object explicitly makes the artifact a function of its inputs again.
    # ORDER IS PRESERVED EXACTLY. The single-step command linked
    # `harness.c, <kernel obj>, *support_sources`; object order decides placement within a section,
    # so reordering could move code and change cycles. This build changes how each object is NAMED,
    # nothing about which objects are linked or in what order.
    objects: list[Path] = []
    for source in [workdir / "harness.c", obj, *recipe.support_sources]:
        source = Path(source)
        # Assembly counts: the driver assembles a .S through the same temp-named intermediate that
        # a .c goes through, so leaving crt.S to the link step reintroduced the very STT_FILE symbol
        # this two-phase build exists to remove.
        if source.suffix not in (".c", ".S", ".s"):
            objects.append(source)
            continue
        unit = workdir / f"{source.stem}.o"
        step = subprocess.run(recipe.compile_command(source=source, output=unit),
                              capture_output=True, text=True)
        if step.returncode != 0:
            raise recipe.error_cls(f"compile of {source.name} failed:\n{step.stderr[-2000:]}")
        objects.append(unit)
    cmd = recipe.link_command(objects=objects, output=elf, link_script=link_ld)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise recipe.error_cls(f"link failed:\n{proc.stderr[-2000:]}")
    return elf


def compile_lowered_to_elf(cb: dict[str, Any], lowered_mlir_text: str,
                           workdir: str | Path | None = None, *, target: str,
                           inputs: dict | None = None) -> Path:
    """Full package-lowered-MLIR -> rv64 ELF (object + runner harness + link)."""
    work = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="oot_compile_"))
    obj = llvm_mlir_to_object(lowered_mlir_text, work)
    return link_elf(cb, obj, work, target=target, inputs=inputs)


def run_on_oracle(cb: dict[str, Any], lowered_mlir_text: str, *, simulator: str, target: str,
                  workdir: str | Path | None = None, timeout: int = 600,
                  inputs: dict | None = None) -> dict[str, Any]:
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
    elf = compile_lowered_to_elf(cb, lowered_mlir_text, work, target=target, inputs=inputs)
    _t1 = time.perf_counter()
    console = backend.run_elf(elf, simulator=simulator, timeout=timeout)
    _t2 = time.perf_counter()
    outputs, raw = backend.parse_output(console)
    result = {"outputs": outputs, "raw_metrics": raw, "cycles": raw.get("cycles", 0),
              "oracle": backend.ORACLE[simulator], "elf": str(elf), "console": console,
              "timing": {"build_s": round(_t1 - _t0, 3), "sim_active_s": round(_t2 - _t1, 3),
                         "oracle_wait_s": 0.0}}
    # Counter markers are a target-independent wire protocol.  The event names/codes remain the
    # target's own: this boundary merely preserves readings the runner already paid to collect.  If
    # they exactly cover a structurally derived joint-occupancy block, compute eta; otherwise retain
    # the raw named readings without guessing what they mean.
    from merlin.perf import hw_counters
    readings = hw_counters.parse_counter_output(console)
    if readings:
        discovery = hw_counters.counters_for_target(target)
        measured_schema = hw_counters.parse_counter_schema(console)
        report: dict[str, Any] = {"status": "measured", "readings": readings,
                                  "discovery": discovery,
                                  "measured_header_sha256": measured_schema}
        if (discovery.get("status") == "derived"
                and measured_schema == discovery.get("header_sha256")):
            header = Path(discovery["header"]).read_text(encoding="utf-8", errors="replace")
            occupancy = hw_counters.derive_occupancy_counters(header)
            required = set(occupancy.by_combination.values())
            if required and required <= set(readings):
                report["occupancy"] = occupancy.to_dict()
                partition_reader = getattr(backend, "counter_partition_inputs", None)
                partition = partition_reader() if callable(partition_reader) else {
                    "status": "unknown",
                    "why": "the target backend exposes no CIRCT counter-partition artifact",
                }
                if partition.get("status") == "available":
                    report["overlap"] = hw_counters.eta_from_counters(
                        readings, occupancy, hw_text=partition["hw_text"],
                        codes=hw_counters.event_codes(header), module=partition["module"],
                        counter_module=partition["counter_module"],
                        measurement_cycles=raw.get("cycles"), source=partition["source"])
                else:
                    report["overlap"] = {
                        "state": "unknown", "eta": None,
                        "why": partition.get("why", "CIRCT counter-partition proof is unavailable"),
                    }
        elif discovery.get("status") == "derived":
            report["status"] = "unknown"
            report["overlap"] = {
                "state": "unknown", "eta": None,
                "why": "the measured ELF counter-schema digest does not match current discovery",
            }
        result["counters"] = report
    return result
