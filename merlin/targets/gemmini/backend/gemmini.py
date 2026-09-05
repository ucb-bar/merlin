"""Run Merlin command buffers on Gemmini, via the chipyard bare-metal flow.

Pipeline: command buffer -> :mod:`gemmini_codegen` C driver (low-level libgemmini intrinsics)
-> compile with the chipyard ``riscv64-unknown-elf-gcc`` against the gemmini-rocc-tests
bare-metal harness -> run on an oracle:

  - ``spike --extension=gemmini``   : functional model, **bootstrap only** (derived_from_rtl=False)
  - the prebuilt Verilator RTL sim  : **certification** (derived_from_rtl=True)
  - the prebuilt GSIM emulator      : **certification** (derived_from_rtl=True) — the same elaborated
    design compiled FIRRTL->C++ instead of Verilog->C++, so it answers at the same fidelity and is
    chosen over Verilator by :mod:`merlin.targetgen.rtl_engine_policy` purely on cost.

-> parse OUT/METRIC/DONE -> gate the outputs against
:func:`merlin.runtime.reference.reference_outputs` (the same oracle the Python simulator
backend is held to). All three oracles run the *exact same ELF*.

Toolchain resolution mirrors ``build_tools/scripts/probe_gemmini_oracle.py``:
``MERLIN_CHIPYARD`` (default ``/path/to/chipyard``), plus optional
``MERLIN_RISCV_GCC`` / ``MERLIN_GEMMINI_SPIKE`` / ``MERLIN_GEMMINI_VERILATOR`` /
``MERLIN_GEMMINI_GSIM_EMU`` / ``MERLIN_GEMMINI_HARNESS_DIR`` overrides.
"""
from __future__ import annotations

import os
import resource
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from merlin.runtime.metrics import COMMON_METRIC_NAMES
from merlin.runtime.reference import outputs_match, reference_outputs
from merlin.runtime.backends.base import BackendInfo, BackendKind, TargetClass, register
from .gemmini_codegen import DIM, CodegenError, generate_driver   # sibling — moves with this backend package

# Self-register this reference NPU backend with the class registry (base._REGISTRY). Discovery in
# base._ensure_discovered imports this module to run the call, so the core carries no name -> module
# map for the accelerator; the identity lives with the backend that owns it.
register(BackendInfo("gemmini", TargetClass.NPU, BackendKind.KERNEL, __name__))

DEFAULT_CHIPYARD = "/path/to/chipyard"
VERILATOR_CONFIG = "GemminiRocketConfig"

#: Wall-cycle budget for the GSIM emulator: the cap IS the hang bound, since a kernel that never reaches
#: its stop condition otherwise runs until the caller's wall timeout. NOT derived from a measurement on
#: this harness — it is a starting bound, raised or lowered per run with MERLIN_GEMMINI_GSIM_MAXCYCLES
#: rather than edited here (for scale, the SIMT target's GSIM oracle defaults to 2M).
GSIM_MAX_CYCLES = 20_000_000

ORACLE = {
    "spike": {"kind": "spike_gemmini_functional", "derived_from_rtl": False},
    "verilator": {"kind": "rtl_verilator", "derived_from_rtl": True},
    # Same elaborated design, different lowering of it (FIRRTL->C++). `derived_from_rtl` is the claim
    # the grade cites, and it is true of GSIM for exactly the reason it is true of Verilator; a missing
    # entry here is not "no oracle metadata" but a KeyError in `contract.compile.run_on_oracle`, i.e. the
    # engine would be selectable and then crash at grade time.
    "gsim": {"kind": "rtl_gsim", "derived_from_rtl": True},
}


class GemminiError(RuntimeError):
    pass


def counter_partition_inputs() -> dict[str, Any]:
    """Target boundary for the generic CIRCT occupancy-partition verifier.

    Module identities select structures in this target's elaborated artifact; no event predicate,
    engine name, code, or numeric parameter is copied here.  The generic verifier must still prove
    the boolean partition from the artifact before an overlap value can be called measured.
    """
    from merlin.targetgen.rtl import mlc_bridge

    path = mlc_bridge.core_hw_mlir("gemmini")
    if path is None or not Path(path).is_file():
        return {"status": "unknown", "why": "elaborated CIRCT core HW is unavailable"}
    return {"status": "available", "hw_text": Path(path).read_text(
        encoding="utf-8", errors="replace"), "module": "Gemmini",
        "counter_module": "CounterController", "source": str(path)}


def chipyard_root() -> Path:
    """Chipyard root, honoring ``.env`` (not just the process env). ``os.environ.get`` alone missed
    ``MERLIN_CHIPYARD`` when it lives in the repo ``.env`` (the repo-wide contract), leaving the
    ``/path/to/chipyard`` placeholder — which made ``available('spike'/'verilator')`` False even with
    a real toolchain, so every oracle reported NOT_RUN_IS_NOT_PASS. Resolve through
    ``merlin.common.paths`` (env/.env → ext_path) with the placeholder only as a last resort."""
    from merlin.common.paths import env as _env, ext_path as _ext_path
    root = os.environ.get("MERLIN_CHIPYARD") or _env("MERLIN_CHIPYARD")
    if root:
        return Path(root)
    cy = _ext_path("chipyard")
    return Path(cy) if cy else Path(DEFAULT_CHIPYARD)


def gcc_path() -> Path:
    env = os.environ.get("MERLIN_RISCV_GCC")
    if env:
        return Path(env)
    return chipyard_root() / ".conda-env/riscv-tools/bin/riscv64-unknown-elf-gcc"


def spike_path() -> Path:
    env = os.environ.get("MERLIN_GEMMINI_SPIKE")
    if env:
        return Path(env)
    return chipyard_root() / ".conda-env/riscv-tools/bin/spike"


def libgemmini_dir() -> Path:
    return chipyard_root() / ".conda-env/riscv-tools/lib"


def platform_dram_base() -> int:
    """The bare-metal linker load address, DERIVED from this target's chipyard RTL build memory map
    (``runtime_build.platform_dram_base`` reads the ``memory@`` region of the sim config's ``memmap.json``)
    rather than baked in the linker script. Falls back to the platform default if the build is absent."""
    from merlin.targetgen import runtime_build as _rb
    return _rb.platform_dram_base("gemmini", "chipyard")


def _rtl_sim_config() -> str:
    """The verilator harness config that realizes gemmini — a DECLARED target fact (capability manifest
    ``runtime.rtl_sim_config``), read via the target registry rather than a hardcoded backend constant.
    Env override wins; then the manifest; then the module fallback (kept coherent with the facts config)."""
    env = os.environ.get("MERLIN_GEMMINI_VERILATOR_CONFIG")
    if env:
        return env
    try:
        from merlin.targetgen.target_experiment import load_capability_manifest
        cfg = (load_capability_manifest("gemmini").contract.get("runtime") or {}).get("rtl_sim_config")
        if cfg:
            return str(cfg)
    except Exception:  # noqa: BLE001 — manifest unavailable ⇒ fall back, never crash the backend
        pass
    return VERILATOR_CONFIG


def verilator_path() -> Path:
    env = os.environ.get("MERLIN_GEMMINI_VERILATOR")
    if env:
        return Path(env)
    return (chipyard_root() / "sims/verilator"
            / f"simulator-chipyard.harness-{_rtl_sim_config()}")


def gsim_path() -> Path:
    """The prebuilt GSIM emulator binary for this target's RTL, or where one would live.

    Env-first like every other tool here, but the env override is the ONLY realistic source: GSIM emits a
    standalone C++ model of the elaborated design and that model is built out of tree (chipyard has no
    ``sims/gsim`` rule), so the derived path below exists to keep the resolution shape identical to
    :func:`verilator_path` and to fail closed — an absent binary makes ``available('gsim')`` False, never
    an exception. Same variable spelling as the SIMT target's ``MERLIN_MUON_GSIM_EMU``.
    """
    env = os.environ.get("MERLIN_GEMMINI_GSIM_EMU", "").strip()
    if env:
        return Path(env)
    return chipyard_root() / "sims/gsim" / f"emu-chipyard.harness-{_rtl_sim_config()}"


def rocc_tests_dir() -> Path:
    env = os.environ.get("MERLIN_GEMMINI_HARNESS_DIR")
    if env:
        return Path(env)
    # Default to the in-repo CURATED int8 harness (elem_t=int8_t, HAS_MVIN_SCALE defined) that we own,
    # NOT chipyard's generators/gemmini/software/gemmini-rocc-tests. That chipyard header is externally
    # regenerated: the shared tree was switched to an fp8/OPU config whose gemmini_params.h drops
    # HAS_MVIN_SCALE -> the mvin scale field encodes 0 -> the int8 Gemmini multiplies every loaded input
    # by 0 -> all-zero matmul in the C driver (the MLIR .insn path is immune; it hardcodes the float-1.0
    # scale bits). The sandbox already binds this curated harness via MERLIN_GEMMINI_HARNESS_DIR; using it
    # as the default makes the direct (non-sandboxed) path header-correct and chipyard-contamination-proof.
    from merlin.common.paths import merlin_dir
    curated = merlin_dir() / "experiments/capsule_bench/targets/gemmini/contracts/harness_curated/gemmini-rocc-tests"
    if curated.is_dir():
        return curated
    return chipyard_root() / "generators/gemmini/software/gemmini-rocc-tests"


def _common_dir() -> Path:
    return rocc_tests_dir() / "riscv-tests/benchmarks/common"


def _test_ld() -> Path:
    return _common_dir() / "test.ld"


def available(simulator: str = "verilator") -> bool:
    """True when gcc + the harness + the requested simulator are all present.

    An engine this backend KNOWS but cannot find answers False; an engine it does not know still RAISES.
    The distinction is load-bearing for :func:`merlin.targetgen.capsule_runner.chipyard_l3_selection`,
    which probes every engine in the priority list: a raise is recorded as "this target has no such
    engine", a False as "it has one and the binary is absent" — two different things to fix.
    """
    base = gcc_path().is_file() and _test_ld().is_file() and _common_dir().is_dir()
    if simulator == "spike":
        return base and spike_path().is_file()
    if simulator == "verilator":
        return base and verilator_path().is_file()
    if simulator == "gsim":
        # Executability is checked as well as existence, unlike the Verilator path: that binary is a
        # build product of a chipyard rule that leaves it executable, whereas the GSIM emu arrives via an
        # env var a caller points anywhere — at a copied-without-mode artifact, or at the emitted .cpp.
        emu = gsim_path()
        return base and emu.is_file() and os.access(emu, os.X_OK)
    raise GemminiError(f"unknown simulator {simulator!r}")


def compile_command_buffer(cb: dict[str, Any], workdir: str | Path,
                           driver_src: str | None = None) -> Path:
    """Generate the Gemmini C driver and compile the bare-metal ELF; return the ELF path.

    ``driver_src`` overrides the in-tree codegen with externally-provided C (used to certify
    an agent-generated kernel) — the rest of the build/run/gate path is identical.
    """
    work = Path(workdir)
    work.mkdir(parents=True, exist_ok=True)
    main_c = work / "main.c"
    main_c.write_text(driver_src if driver_src is not None else generate_driver(cb),
                      encoding="utf-8")
    elf = work / "merlin_gemmini_c0.elf"
    rt = rocc_tests_dir()
    common = _common_dir()
    # Mirror gemmini-rocc-tests/bareMetalC/Makefile (CFLAGS_BAREMETAL) EXACTLY — both the
    # flag set and the include ORDER matter: a wrong order shadows the riscv-tests/env
    # syscall headers and corrupts the tohost protocol ("bad syscall" on spike).
    cmd = [
        str(gcc_path()),
        "-DPREALLOCATE=1", "-DMULTITHREAD=1",
        "-mcmodel=medany", "-std=gnu99", "-O2", "-ffast-math",
        "-fno-common", "-fno-builtin-printf", "-fno-tree-loop-distribute-patterns",
        "-march=rv64gc", "-Wa,-march=rv64gc",
        "-lm", "-lgcc",
        "-I", str(rt / "riscv-tests"),
        "-I", str(rt / "riscv-tests/env"),
        "-I", str(rt),
        "-I", str(common),
        "-DID_STRING=", "-DPRINT_TILE=0",
        "-nostdlib", "-nostartfiles", "-static",
        "-T", str(_test_ld()), "-DBAREMETAL=1",
        str(main_c),
        "-o", str(elf),
        *(str(p) for p in sorted(common.glob("*.c"))),
        *(str(p) for p in sorted(common.glob("*.S"))),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise GemminiError(f"riscv gcc failed:\n{' '.join(cmd)}\n{proc.stderr}")
    return elf


def _unlimited_stack() -> None:  # pragma: no cover - runs in the child process
    """Lift RLIMIT_STACK for a simulator child.

    The Verilator model needs a large stack; the default (e.g. 12500 kb) makes it warn ("%Warning: System
    has stack size ...") onto the console, corrupting output capture. The GSIM-emitted model holds the
    whole design state in one C++ object and is driven from the same harness, so it is raised the same way.
    """
    try:
        resource.setrlimit(resource.RLIMIT_STACK,
                           (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
    except (ValueError, OSError):
        pass


def gsim_max_cycles() -> str:
    """The GSIM wall-cycle cap, as the plusarg value. Env override wins; see :data:`GSIM_MAX_CYCLES`."""
    return os.environ.get("MERLIN_GEMMINI_GSIM_MAXCYCLES", "").strip() or str(GSIM_MAX_CYCLES)


def run_elf(elf: str | Path, simulator: str = "verilator", timeout: int = 600) -> str:
    """Run the ELF on the chosen oracle; return raw console output."""
    preexec = None
    if simulator == "spike":
        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = str(libgemmini_dir()) + ":" + env.get("LD_LIBRARY_PATH", "")
        cmd = [str(spike_path()), "--extension=gemmini", str(elf)]
    elif simulator == "verilator":
        env = dict(os.environ)
        cmd = [str(verilator_path()), str(elf)]
        preexec = _unlimited_stack
    elif simulator == "gsim":
        env = dict(os.environ)
        # The SAME ELF the Verilator path runs, so the console it prints is the same OUT/METRIC/DONE text
        # and `parse_output` is unchanged. GSIM re-roots the circuit at ChipTop and so has no SimTSI to
        # load the image: `+loadmem` is the backdoor that writes it into the backing store, and it is
        # passed BESIDE the positional argument rather than instead of it (the emitted harness reads the
        # symbol table from the positional path). `+max-cycles` is the hang bound, not a perf knob.
        cmd = [str(gsim_path()), str(elf), f"+max-cycles={gsim_max_cycles()}", f"+loadmem={elf}"]
        preexec = _unlimited_stack
        # NB the console is buffered in this process, like every other oracle here. That is fine for the
        # OUT/METRIC/DONE protocol but NOT for a model built with a per-instruction commit trace: on the
        # SIMT target such a console reached 72 GB in one run and had to be spooled to disk. If a gemmini
        # GSIM model is ever emitted with tracing on, this branch needs that same treatment.
    else:
        raise GemminiError(f"unknown simulator {simulator!r}")
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env,
                          preexec_fn=preexec)
    # The Verilator harness exits 0 on $finish; spike exits 0 on htif_exit(0); the GSIM-emitted model
    # exits 0 when the design's own stop condition fires before +max-cycles.
    if proc.returncode != 0:
        raise GemminiError(
            f"{simulator} exited {proc.returncode}:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
    return proc.stdout


def parse_output(text: str) -> tuple[dict[str, list], dict[str, int]]:
    """Parse the OUT/METRIC/DONE console into (outputs, raw metrics) — shared protocol parser, with
    the gemmini-specific robustness: strip stray Verilator ``%Warning:`` fragments + tolerate a
    malformed METRIC line instead of raising."""
    from merlin.runtime.backends.base import parse_console
    return parse_console(text, error_cls=GemminiError, strip_warnings=True, tolerant_metric=True)


def _metrics(raw: dict[str, int], simulator: str) -> dict[str, Any]:
    metrics = {name: int(raw.get(name, 0)) for name in COMMON_METRIC_NAMES}
    metrics["cycles"] = int(raw.get("cycles", 0))
    metrics["cycle_source"] = "rdcycle" if "cycles" in raw else "unknown"
    metrics["cycle_window"] = ("gemmini_region"
                               if raw.get("cycle_window_gemmini_region") else "unknown")
    metrics["memory_model"] = "functional_model" if simulator == "spike" else "unknown"
    return metrics


def run_command_buffer(cb: dict[str, Any], *, workdir: str | Path | None = None,
                       simulator: str = "verilator", timeout: int = 600,
                       driver_src: str | None = None) -> dict[str, Any]:
    """Compile + run a command buffer on Gemmini and gate on reference equality.

    ``driver_src`` certifies an externally-provided (e.g. agent-generated) kernel instead of
    the in-tree codegen. Returns {outputs, metrics, raw_metrics, correct, oracle, elf, console}.
    """
    if not available(simulator):
        raise GemminiError(f"gemmini {simulator} oracle not available (set MERLIN_CHIPYARD)")
    own_tmp = workdir is None
    work = Path(tempfile.mkdtemp(prefix="merlin_gemmini_")) if own_tmp else Path(workdir)
    elf = compile_command_buffer(cb, work, driver_src=driver_src)
    console = run_elf(elf, simulator=simulator, timeout=timeout)
    outputs, raw = parse_output(console)
    ref = reference_outputs(cb)
    return {
        "outputs": outputs,
        "metrics": _metrics(raw, simulator),
        "raw_metrics": raw,
        "correct": outputs_match(outputs, ref),
        "oracle": dict(ORACLE[simulator]),
        "elf": str(elf),
        "console": console,
    }


def preflight_codegen_smoke(*, target: str) -> tuple[bool, str]:
    """Compile the production command-buffer emitter and run it bit-exact on RTL.

    This is the target-owned implementation of the generic pre-spend codegen-smoke hook.  It exercises
    the same ``generate_driver -> riscv gcc -> selected L3 engine -> parse -> reference equality`` path
    used by a real grade.  The engine is resolved through the shared RTL policy, including
    ``MERLIN_REQUIRED_RTL_ENGINE``; otherwise a GSIM-pinned run would silently pay for an unrelated
    Verilator pass before every resume.  Merely finding the simulator or compiling an empty file is
    insufficient: both have been true while the emitted kernel itself was wrong.
    """
    try:
        # Import at call time: capsule_runner discovers this backend while it is itself importing, so a
        # module-level import would create a cycle.  This is the SAME selector used to bind the L3 grade;
        # the preflight must not maintain a second engine policy.
        from merlin.targetgen.capsule_runner import chipyard_l3_selection
        selection = chipyard_l3_selection(target)
        rtl_engine = str(selection["engine"])
    except Exception as e:  # noqa: BLE001 — no policy-selected L3 means the smoke cannot certify codegen
        return False, ("Gemmini production codegen smoke cannot select its L3 RTL engine: "
                       f"{type(e).__name__}: {str(e)[-200:]}")
    if not available(rtl_engine):
        return False, (f"Gemmini production codegen smoke cannot run: the selected {rtl_engine} RTL "
                       "oracle, RISC-V compiler, or curated harness is unavailable")
    tile = int(DIM)
    cb = {
        "abi_version": "0.1",
        "target": target,
        "tensors": {
            "probe_w": {"shape": [tile, tile], "dtype": "i8", "role": "weight"},
            "probe_a": {"shape": [tile, tile], "dtype": "i8", "role": "input"},
            "probe_y": {"shape": [tile, tile], "dtype": "i32", "role": "output"},
        },
        "commands": [
            {"opcode": "RES_PACK", "operands": {"src": "probe_w", "dst": "probe_w_res"},
             "attributes": {"layout": "packed_rhs"}},
            {"opcode": "MATMUL_RESIDENT",
             "operands": {"lhs": "probe_a", "rhs": "probe_w_res", "dst": "probe_acc"}},
            {"opcode": "COMMIT", "operands": {"src": "probe_acc", "dst": "probe_y"},
             "attributes": {"epilogue": [], "output_dtype": "i32"}},
            {"opcode": "EVICT", "operands": {"handle": "probe_w_res"}},
        ],
    }
    try:
        with tempfile.TemporaryDirectory(prefix="merlin_gemmini_codegen_smoke_") as td:
            result = run_command_buffer(cb, workdir=td, simulator=rtl_engine, timeout=600)
            elf_present = Path(str(result.get("elf") or "")).is_file()
    except Exception as e:  # noqa: BLE001 — this is the failure the launch gate exists to surface
        return False, f"Gemmini production codegen smoke failed: {type(e).__name__}: {str(e)[-240:]}"
    oracle = result.get("oracle") or {}
    output = (result.get("outputs") or {}).get("probe_y")
    if (result.get("correct") is not True or not output or not elf_present
            or oracle.get("derived_from_rtl") is not True):
        return False, ("Gemmini production codegen smoke ran but lacked a bit-exact RTL proof "
                       f"(correct={result.get('correct')!r}, output={bool(output)}, "
                       f"elf={elf_present}, oracle={oracle!r})")
    return True, (f"production command-buffer codegen compiled and ran a {tile}x{tile} kernel "
                  f"bit-exact on {rtl_engine} RTL")


def harness_build_recipe():
    """How to compile + link a runner-owned harness against this target's bare-metal environment.

    Declared here, where the target is owned, so the GENERIC contract-compile path can orchestrate the
    build without importing this module. Every value is resolved the same way the backend's own build
    resolves it — the curated harness tree (env-overridable), the riscv-tests include layout, the
    toolchain gcc, and a link script whose ORIGIN is derived from the RTL memory map rather than baked
    into the vendored script. Nothing new is hardcoded: this is the existing recipe, named.
    """
    # Absolute, like the other two imports of this module: the package registers OUT-OF-TREE as
    # `merlin._oot_backends.gemmini`, so a relative `.base` resolves to a sibling that does not
    # exist there and the spike/verilator invocation dies with ModuleNotFoundError at grade time.
    from merlin.runtime.backends.base import HarnessBuildRecipe

    rt, common = rocc_tests_dir(), _common_dir()
    return HarnessBuildRecipe(
        compiler=gcc_path(),
        include_roots=(rt / "riscv-tests", rt / "riscv-tests/env", rt, common),
        support_sources=tuple(sorted(common.glob("*.c"))) + tuple(sorted(common.glob("*.S"))),
        link_script=_test_ld(),
        load_address=platform_dram_base(),
        cflags=("-DPREALLOCATE=1", "-DMULTITHREAD=1", "-mcmodel=medany", "-std=gnu99", "-O2",
                "-ffast-math", "-fno-common", "-fno-builtin-printf",
                "-fno-tree-loop-distribute-patterns", "-march=rv64gc", "-Wa,-march=rv64gc",
                "-lm", "-lgcc", "-DID_STRING=", "-DPRINT_TILE=0",
                "-nostdlib", "-nostartfiles", "-static", "-DBAREMETAL=1"),
        error_cls=GemminiError,
    )


# --- runner-owned harness rendering (the `harness_renderer` backend capability) ---------------------
# Moved here from the GENERIC contract-compile path, which had to import this module to render a
# harness at all. Both renderers are target-owned for the same underlying reason: they pad to this
# accelerator's tile edge and lay out its accumulator readout. That is codegen, not four declarable
# strings, so it belongs with the backend rather than behind a contract key no second target could
# implement. What the CONTRACT still supplies is the harness ABI (entry symbol, fence, includes,
# metric), read through `harness_abi.for_target` below.
from .gemmini_codegen_mlir import _harness_c, _measurement_c_fragments


def _is_movement_cb(cb: dict) -> bool:
    cmds = cb.get("commands", [])
    return (not any(c.get("opcode") == "RES_PACK" for c in cmds)
            and any(c.get("opcode") == "MOVEMENT" or (
                    c.get("opcode") == "VECTOR_MAP"
                    and c.get("attributes", {}).get("combine") == "identity") for c in cmds))


_NATIVE_INTERFACE_OPS = frozenset({"ATTENTION_QK", "ATTENTION_PV", "CONV2D"})


def _native_interface_command(cb: dict) -> dict | None:
    """Return the one whole interface op whose LLVM ABI is the interface tensor list.

    The resident-matmul backend has an intentionally different internal ABI
    (weights ++ materialized lhs matrices ++ outputs).  A whole-op artifact does not: its pointers are
    the tensors declared by the interface, including an original NHWC activation rather than a
    runner-materialized im2col matrix.  Keep the distinction explicit so the harness cannot silently
    call one ABI with the other's argument list.
    """
    native = [command for command in cb.get("commands", [])
              if command.get("opcode") in _NATIVE_INTERFACE_OPS]
    if not native:
        return None
    if len(native) != 1:
        raise CodegenError(
            f"native interface harness supports exactly one whole op, got {len(native)}")
    return native[0]


#: Declared output dtype -> (C type, alignment macro) for a harness-allocated destination buffer. The
#: SAME two spellings the tiled harness (`gemmini_codegen_mlir._harness_c`) allocates its outputs with:
#: an i8 readout lands in the operand type `elem_t`, and a full accumulator readout lands in `int32_t`
#: (NOT `acc_t` — see that function for why the typedef cannot be trusted). Kept as a table, and
#: consulted rather than defaulted, so an output dtype nobody has sized is a REFUSAL and never a buffer
#: quietly allocated at the wrong width.
_DEST_CTYPE = {"i8": ("elem_t", "row_align"), "i32": ("int32_t", "row_align_acc")}


def _declared_output_dtype(cb: dict, cmd: dict, dst: str) -> str:
    """The dtype the destination of ``cmd`` is DECLARED to land in: the command's own ``output_dtype``
    attribute, else the declared dtype of the ``dst`` tensor. Raises when neither states one — a movement
    capsule's whole point is the container widening, so an unstated output dtype is not a thing to guess.
    """
    dtype = (cmd.get("attributes") or {}).get("output_dtype") \
        or ((cb.get("tensors") or {}).get(dst) or {}).get("dtype")
    if not dtype:
        raise CodegenError(
            f"movement destination {dst!r} declares no output dtype (neither the command's "
            f"output_dtype attribute nor the tensor's dtype), so its buffer cannot be sized")
    return str(dtype)


def _movement_harness_c(cb: dict, *, target: str, inputs: dict | None = None) -> str:
    """Harness for a pure-movement kernel ``<entry>(src*, dst*)``: embed src, print dst.

    The entry symbol, the fence, the includes and the cycle-window metric are read from ``target``'s
    declared harness ABI rather than written here; see :mod:`.harness_abi` for why they cannot be
    derived. ``target`` is required: a default would be one target's name, which is the weld this is
    removing.
    """
    from .gemmini_codegen import _ceil_dim, _pad_rowmajor
    # Absolute: this package registers OUT-OF-TREE as `merlin._oot_backends.gemmini`, where `..` is the
    # synthetic namespace rather than `merlin.runtime` — the layout these relative imports were written
    # against before the eviction. Left relative they raise ModuleNotFoundError at GRADE time, inside the
    # harness renderer, which the runner reports as an opaque "spike invocation failed".
    from merlin.runtime.commandbuffer import materialize_inputs
    from merlin.targetgen.contract.harness_abi import for_target
    mv = next(c for c in cb["commands"]
              if c.get("opcode") == "MOVEMENT" or (
                  c.get("opcode") == "VECTOR_MAP"
                  and c.get("attributes", {}).get("combine") == "identity"))
    src = mv["operands"].get("src") or mv["operands"].get("lhs")
    dst = mv["operands"]["dst"]
    m, n = cb["tensors"][src]["shape"]
    mp, np_ = _ceil_dim(m), _ceil_dim(n)
    leaves = materialize_inputs(cb, inputs)
    sp = _pad_rowmajor(list(leaves[src].data), m, n, mp, np_)
    # The destination is allocated from the DECLARED OUTPUT dtype, not from the operand dtype. Movement
    # is a container widening (operand dtype in, accumulate dtype out), so pinning the destination to
    # `elem_t` under-allocated it by the width ratio: two shipped capsules declare an i32 output, and a
    # CORRECT 4-byte store into a 1-byte-per-element buffer ran ~700 bytes off the end of .bss and
    # trapped AFTER printing DONE — a harness overrun reported as a failure of the submission.
    odt = _declared_output_dtype(cb, mv, dst)
    if odt not in _DEST_CTYPE:
        raise CodegenError(
            f"movement destination {dst!r} declares output dtype {odt!r}, which this harness has no "
            f"buffer width for (sized: {sorted(_DEST_CTYPE)}). Refusing rather than allocating a "
            f"guessed width")
    dst_ctype, dst_align = _DEST_CTYPE[odt]
    decls = [f"static const elem_t T_{src}[{mp * np_}] row_align(1) = "
             f"{{{','.join(str(int(v)) for v in sp)}}};",
             f"static {dst_ctype} T_{dst}[{mp * np_}] {dst_align}(1);"]
    prints = [f'  printf("OUT {dst} {m} {n}");',
              f"  for (long i = 0; i < {m}; i++) for (long j = 0; j < {n}; j++)"
              f" printf(\" %d\", (int)T_{dst}[i * {np_} + j]);", '  printf("\\n");']
    # Print METRIC cycles BEFORE the (possibly huge) OUT tensor dump: large-output kernels flood the
    # UART and the per-ELF capture truncates mid-dump, so a trailing METRIC line would be lost. Emitting
    # the (tiny) cycle metric first guarantees it is always captured; the OUT dump follows for correctness.
    abi = for_target(target)
    window = abi.cycle_window_line()
    measured_call = abi.call(f"(void*)T_{src}, (void*)T_{dst}")
    fragments = _measurement_c_fragments(measured_call)
    return ("#include <stdint.h>\n#include <stdio.h>\n" + abi.declarations() + "\n"
            + fragments["include"]
            + "\n".join(decls) + "\nint main() {\n"
            + fragments["warmup"]
            + fragments["prologue"]
            + "  uint64_t c0 = read_cycles();\n"
            + measured_call + "\n"
            + "  uint64_t c1 = read_cycles();\n"
            + fragments["epilogue"]
            + '  printf("METRIC cycles %lu\\n", (unsigned long)(c1 - c0));\n'
            + (window + "\n" if window else "")
            + "\n".join(prints) + "\n"
            '  printf("DONE\\n");\n  return 0;\n}\n')


def _flat_matrix_shape(spec: dict, *, name: str) -> tuple[int, int]:
    """Flatten leading logical dimensions into rows, preserving the final dimension as columns.

    Gemmini's DRAM-facing matrix layout pads both dimensions to its tile edge.  For NHWC this means
    one physical row per pixel and, critically, a padded physical C stride; compact NHWC bytes do not
    match the addresses emitted by a tiled whole-convolution artifact.
    """
    shape = spec.get("shape")
    if not isinstance(shape, list) or len(shape) < 2 or any(
            not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0 for dim in shape):
        raise CodegenError(f"native interface tensor {name!r} needs a positive rank >= 2 shape, got {shape!r}")
    rows = 1
    for dim in shape[:-1]:
        rows *= dim
    return rows, shape[-1]


def _native_interface_harness_c(cb: dict, command: dict, *, inputs: dict | None = None) -> str:
    """Harness for a schema-native whole op with the interface's pointer ABI.

    The tensor table is emitted by the interface parser in declaration order.  Preserve that order
    exactly: unlike the in-tree resident-matmul emitter, a package artifact receives neither reordered
    weights nor codegen-only derived buffers.  Every physical buffer is a zero-padded matrix whose last
    logical dimension is its padded row stride.
    """
    from .gemmini_codegen import _ceil_dim, _pad_rowmajor
    from merlin.runtime.commandbuffer import materialize_inputs

    tensors = cb.get("tensors") or {}
    opcode = command.get("opcode")
    operands = command.get("operands") or {}
    packs = {item.get("operands", {}).get("dst"): item.get("operands", {}).get("src")
             for item in cb.get("commands", []) if item.get("opcode") == "RES_PACK"}
    if opcode == "ATTENTION_QK":
        required = [operands.get("q"), operands.get("k"), operands.get("dst")]
    elif opcode == "ATTENTION_PV":
        required = [operands.get("p"), operands.get("v"), operands.get("dst")]
    elif opcode == "CONV2D":
        weight = operands.get("weight")
        required = [operands.get("ifm"), packs.get(weight, weight), operands.get("dst")]
    else:  # pragma: no cover - caller selects only _NATIVE_INTERFACE_OPS
        raise CodegenError(f"unsupported native interface opcode {opcode!r}")
    if any(not isinstance(name, str) or not name for name in required):
        raise CodegenError(f"native interface {opcode} has incomplete operands {operands!r}")
    missing = [name for name in required if name not in tensors]
    if missing:
        raise CodegenError(
            f"native interface {opcode} operand(s) {missing} have no declared tensor buffer")

    # Dict insertion order is the only representation of interface declaration order retained by the
    # command-buffer JSON.  Restrict it to external buffers, then prove every whole-op operand is present;
    # do not reconstruct a plausible order from roles (CONV commonly declares IFM before its weight).
    external_roles = {"input", "weight", "bias", "output"}
    args = [name for name, spec in tensors.items() if spec.get("role") in external_roles]
    if any(name not in args for name in required):
        raise CodegenError(
            f"native interface {opcode} operands are not all declared external buffers: {required!r}")

    leaves = materialize_inputs(cb, inputs)
    output_names = [name for name in args if tensors[name].get("role") == "output"]
    if operands["dst"] not in output_names:
        raise CodegenError(
            f"native interface destination {operands['dst']!r} is not a declared output tensor")

    decls: list[str] = []
    layouts: dict[str, tuple[int, int, int, int]] = {}
    for name in args:
        spec = tensors[name]
        rows, cols = _flat_matrix_shape(spec, name=name)
        prows, pcols = _ceil_dim(rows), _ceil_dim(cols)
        layouts[name] = (rows, cols, prows, pcols)
        if spec.get("role") == "output":
            dtype = str(spec.get("dtype") or "")
            if dtype not in _DEST_CTYPE:
                raise CodegenError(
                    f"native interface output {name!r} declares dtype {dtype!r}, which this harness "
                    f"has no buffer width for (sized: {sorted(_DEST_CTYPE)})")
            ctype, align = _DEST_CTYPE[dtype]
            decls.append(f"static {ctype} T_{name}[{prows * pcols}] {align}(1);")
            continue
        if spec.get("dtype") != "i8":
            raise CodegenError(
                f"native interface input {name!r} must use the target operand dtype i8, "
                f"got {spec.get('dtype')!r}")
        if name not in leaves:
            raise CodegenError(f"native interface input {name!r} was not materialized")
        padded = _pad_rowmajor(list(leaves[name].data), rows, cols, prows, pcols)
        decls.append(f"static const elem_t T_{name}[{prows * pcols}] row_align(1) = "
                     f"{{{','.join(str(int(value)) for value in padded)}}};")

    call = ", ".join(f"(void*)T_{name}" for name in args)
    prints: list[str] = []
    for name in output_names:
        shape = tensors[name].get("shape")
        if not isinstance(shape, list) or len(shape) != 2:
            raise CodegenError(
                f"native interface output {name!r} must be a rank-2 flattened tensor, got {shape!r}")
        rows, cols, _, pcols = layouts[name]
        prints.extend([
            f'  printf("OUT {name} {rows} {cols}");',
            f"  for (long i = 0; i < {rows}; i++) for (long j = 0; j < {cols}; j++)"
            f" printf(\" %d\", (int)T_{name}[i * {pcols} + j]);",
            '  printf("\\n");',
        ])

    measured_call = f"  gemmini_kernel({call});\n  gemmini_fence();"
    fragments = _measurement_c_fragments(measured_call)
    return ("#include <stdint.h>\n#include <stdio.h>\n#include \"include/gemmini_testutils.h\"\n"
            + fragments["include"]
            + "extern void gemmini_kernel();\n" + "\n".join(decls) + "\nint main() {\n"
            + fragments["warmup"]
            + fragments["prologue"]
            + "  uint64_t c0 = read_cycles();\n"
            + measured_call + "\n"
            + "  uint64_t c1 = read_cycles();\n"
            + fragments["epilogue"]
            + '  printf("METRIC cycles %lu\\n", (unsigned long)(c1 - c0));\n'
            '  printf("METRIC cycle_window_gemmini_region 1\\n");\n'
            + "\n".join(prints) + "\n"
            '  printf("DONE\\n");\n  return 0;\n}\n')


def render_harness(cb: dict, *, target: str, inputs: dict | None = None) -> str:
    """Render the runner-owned harness for ``cb`` — the `harness_renderer` capability.

    Chooses between the pure-movement and tiled forms itself, because which one applies is a property
    of this target's command vocabulary rather than something the generic path can decide.

    ``inputs`` (name -> nested-list) INJECTS explicit operand values so the DEVICE computes on the same
    data the reference and the simulator were given. Without it the harness materializes each leaf from
    its NAME, so a caller injecting real activations got a three-way gate in which the reference and the
    simulator saw the injected operands and the device saw different ones -- guaranteed to mismatch, and
    reported as a functional failure of the target.
    """
    if _is_movement_cb(cb):
        return _movement_harness_c(cb, target=target, inputs=inputs)
    native = _native_interface_command(cb)
    if native is not None:
        return _native_interface_harness_c(cb, native, inputs=inputs)
    return _harness_c(cb, inputs)
