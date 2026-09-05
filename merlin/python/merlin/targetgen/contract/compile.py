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


def llvm_mlir_to_object(lowered_mlir_text: str, workdir: Path, *, target: str | None = None) -> Path:
    """Lower package-emitted llvm-dialect MLIR to an rv64 object (.o) for ``target``'s own ISA.

    THE MARCH IS THE TARGET'S, NOT A DEFAULT. This object and the runner-owned harness are linked into
    one ELF and executed on one core, so they have to be built for the same instruction set. The
    default here is the vector-capable ``rv64gcv``, while a systolic target's harness recipe declares
    ``rv64gc`` -- and for years that disagreement was invisible, because every kernel on such a target
    was inline-asm accelerator instructions with nothing for the auto-vectorizer to take. The first
    kernel that gives it something (a scalar float program placed on the host lane) was compiled with
    ``vsetivli``/``vle32.v``/``vfadd.vv`` for a core with no vector unit, trapped on the first one, and
    was reported as the submission's kernel faulting at runtime.

    ``target=None`` keeps the previous default, for callers with no target in hand.
    """
    from merlin.llvmlower.pipeline import lower_to_llvm_ir
    from merlin.llvmlower import codegen
    workdir.mkdir(parents=True, exist_ok=True)
    ll = lower_to_llvm_ir(lowered_mlir_text, workdir=workdir)
    (workdir / "kernel.ll").write_text(ll, encoding="utf-8")
    extra: tuple[str, ...] = ()
    if target is not None:
        from merlin.runtime.backends import base as _backends
        extra = (_backends.harness_build_recipe(target).march(),)
    return Path(codegen.compile_ll(workdir / "kernel.ll", workdir / "kernel.o", "riscv",
                                   extra_flags=extra))


def _recorded_operands(cb: dict[str, Any]) -> dict[str, list] | None:
    """The operands a FLOAT-graded buffer must be run on, or ``None``.

    A capsule graded under a float policy cannot have had its answer recomputed on the integer engine,
    so its golden is the INDEPENDENT one — computed off-device, on the operands the runner attached to
    the buffer as ``canonical_inputs``. The device is only answerable against that golden if it runs
    on those same operands; nothing consumed them on this path, so the harness materialized each leaf
    from its NAME and the device computed the right function of the wrong inputs — a guaranteed
    mismatch, reported as a functional failure of the submission.

    THE FLOAT CONDITION IS LOAD-BEARING, not decoration. A buffer whose declared output is an INTEGER
    is graded against a golden RECOMPUTED from the deterministic name-materialized fill, and its
    capsule may still record different operands beside it — measured on ``GS0_matmul_spec``, whose
    recorded ``W``/``A0`` and materialized ``W``/``A0`` are different numbers and which passes today
    precisely because the device materializes. Embedding recorded operands unconditionally breaks
    exactly those capsules; embedding them nowhere leaves every float capsule unwinnable. The declared
    output dtype is what separates the two, and it is read from the buffer itself.
    """
    from merlin.runtime.backends import base as _backends
    from merlin.runtime.commandbuffer import declared_output_dtypes
    recorded = cb.get("canonical_inputs") or {}
    tensors = cb.get("tensors") or {}
    if not recorded:
        return None
    dtypes = declared_output_dtypes(cb)
    outputs = [n for n, s in tensors.items() if (s or {}).get("role") == "output"]
    if not outputs or not all(_backends.float_format_of(dtypes.get(n, "")) for n in outputs):
        return None
    return {name: spec["values"] for name, spec in recorded.items()
            if isinstance(spec, dict) and spec.get("values") is not None and name in tensors} or None


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
    inputs = inputs or _recorded_operands(cb) or None
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
    obj = llvm_mlir_to_object(lowered_mlir_text, work, target=target)
    return link_elf(cb, obj, work, target=target, inputs=inputs)


def simulator_provenance(backend, simulator: str) -> dict[str, Any] | None:
    """Identify the simulator BINARY an engine ran on: its path, its digest, and its recorded lineage.

    A result that claims a hardware verdict must record which hardware revision it came from. A cert
    record already pins the ELF, the RTL commits and the toolchain — and said nothing at all about the
    prebuilt simulator that produced the numbers, which for GSIM is an out-of-tree build whose bytes are
    the only thing tying the verdict to an elaboration. A stale or mis-provenanced emulator was therefore
    undetectable after the fact, which is the same failure mode the binary provenance stamp was added to
    catch on the ELF side.

    Derived from the backend, not from a table: it is asked for ``<engine>_status()`` (a sentence about
    how the engine resolved) and ``<engine>_path()`` (where its binary is), both DERIVED attribute names,
    so an engine or a backend that does not publish them contributes nothing rather than a fabricated
    entry. Never raises — provenance that cannot be established is recorded as absent.
    """
    from pathlib import Path as _Path
    rec: dict[str, Any] = {"engine": simulator}
    try:
        getter = getattr(backend, f"{simulator}_path", None)
        if callable(getter):
            binary = _Path(str(getter()))
            rec["binary"] = str(binary)
            if binary.is_file():
                from merlin.common import provenance as _prov
                rec["sha256"] = _prov.file_digest(binary)
        status = getattr(backend, f"{simulator}_status", None)
        if callable(status):
            _ok, _why = status()
            rec["resolution"] = str(_why)
    except Exception as exc:  # noqa: BLE001 — unestablished provenance is recorded, never invented
        rec["error"] = f"{type(exc).__name__}: {exc}"
    return rec if len(rec) > 1 else None


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
    # DECODE A FLOAT RESULT THAT CAME BACK AS ITS CONTAINER WORD. `parse_output` yields whatever the
    # console carried; a target whose harness has integer-only formatting prints a float destination
    # buffer's stored PATTERN, so an f32 result arrives as its 32-bit word and a bf16 result as its
    # 16-bit one. That is a lossless hand-back, not a broken one -- but only once the pattern is read
    # back as the value, which needs the dtype the buffer was DECLARED in. Read here, from the command
    # buffer itself (the same declaration the harness sized the buffer from), so the writer and the
    # reader cannot disagree; keyed on that dtype and on nothing target-specific. A no-op for an
    # integer-declared output and for a backend that already prints decimals, so every existing
    # readback is byte-identical.
    from merlin.runtime.commandbuffer import declared_output_dtypes
    outputs = _backends.decode_float_readback(outputs, declared_output_dtypes(cb))
    # WHICH BUILD of the simulator answered — recorded beside the oracle's declared kind, not inferred
    # afterwards. The tier record identifies the ELF, the RTL pins and the tools, and identified the one
    # remaining input to the verdict not at all: the prebuilt simulator binary. Derived, never assumed:
    # the backend is asked where its ``<engine>_path()`` is and the bytes there are digested. A backend
    # that does not expose one contributes nothing rather than a guess.
    _oracle = dict(backend.ORACLE[simulator])
    _prov = simulator_provenance(backend, simulator)
    if _prov:
        _oracle["provenance"] = _prov
    result = {"outputs": outputs, "raw_metrics": raw, "cycles": raw.get("cycles", 0),
              "oracle": _oracle, "elf": str(elf), "console": console,
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
