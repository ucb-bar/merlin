"""Realize Merlin's command buffer on this SIMT cluster's own hardware.

This is the package's runtime backend: the core loads it because the contract's ``plugin.backend``
names it, so nothing in the core tree carries a name -> module map for this target. Everything below
the command buffer that is target-specific lives here.

What this module is for
-----------------------
The staged pipeline already descends to a command buffer for this target, and the fork-free device
path (boot object, transcode, link, oracles) is already generic. The one thing missing was the
**tile body**: nothing turned a command buffer into the per-warp computation the scaffold runs.
:func:`emit_kernel` is that step, and it is the only new content needed to take this target from
"emits a command buffer" to "executes on its own RTL".

Why the command buffer and not the target dialect
-------------------------------------------------
This backend consumes Merlin's abstract opcodes (``MATMUL``/``COMMIT``/``VECTOR_MAP``), never a
target-dialect op name. That matters here specifically: this package's dialect is hand-authored and
prototype-grade, so codegen written against its op names would be certifying names somebody chose
rather than hardware. The opcode set is Merlin's own and is what a *generated* dialect would lower to
as well, so this emitter survives the dialect being replaced.

Scope, stated so a green run cannot be over-read
------------------------------------------------
1. The computation is realized on the cluster's **base integer ISA, mapped across warps**. It does
   NOT use the tensor-core / MX datapath: those op encodings are not derived, and the command buffer's
   element types are integer in the first place. A pass here says the command buffer executes
   correctly on this hardware — not that the tensor core was exercised.
2. Work is partitioned across **warps** only, not across the lanes within a warp. Lane-level mapping
   needs the thread-mask/lane-id ops, which this module does not derive. So the ``must_map_to_warps``
   obligation is discharged; SIMD width is left on the table and is a later increment.
3. Residency (``RES_PACK``/``MATMUL_RESIDENT``) is REFUSED, not approximated. Staging an operand into
   shared memory needs the scratchpad's base address, and the derived runtime ABI has no such
   aperture yet — ``IsaModel.aperture`` raises rather than offering one. Realizing a "resident" pack
   in global memory would produce a kernel that runs, grades green, and has not made anything
   resident. The refusal names the missing derivation.

Two deliberate emission choices
-------------------------------
**Operands are ``volatile``.** With constant inputs, constant loop bounds and no aliasing, the
compiler is entitled to fold the whole contraction at build time and store the answer. That kernel
grades green while executing no arithmetic — a vacuous pass of exactly the kind this repo has been
bitten by before. ``volatile`` forces the loads and the multiply-accumulate to survive into the image.

**Every command uses the same partitioning.** Each command's outer loop is strided by warp id in the
same way, so a warp only ever reads elements it wrote itself. That is what makes the absence of a
cross-warp barrier safe rather than lucky — this module does not derive a barrier op, so it must not
emit a schedule that would need one. A future command whose partitioning differs MUST introduce one.
"""
from __future__ import annotations

import struct
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from merlin.runtime.backends import muon
from merlin.runtime.backends.base import BackendInfo, BackendKind, TargetClass, register
from merlin.runtime.reference import outputs_match, reference_outputs
from merlin.runtime.commandbuffer import materialize_inputs

HERE = Path(__file__).resolve().parent

#: The target this package is for. A target name is legitimate inside its own package — it is the
#: subject here — and is what keeps it out of the core, where it would be an overfit.
TARGET = "radiance"
BACKEND_NAME = "radiance_simt"

#: Opcodes this backend can realize honestly today.
SUPPORTED = ("MATMUL", "COMMIT", "VECTOR_MAP")

#: Opcodes refused with the derivation that is missing, so a failure names a boundary rather than
#: reading "unsupported". Fail closed: none of these has a safe approximation.
REFUSED = {
    "RES_PACK": ("staging an operand into shared memory needs the scratchpad base address; the "
                 "derived runtime ABI has no such aperture (IsaModel.aperture raises). Realizing a "
                 "resident pack in global memory would not make anything resident"),
    "MATMUL_RESIDENT": ("a resident matmul reads an operand staged in shared memory; see RES_PACK. "
                        "Lower this workload without residency (a non-reused RHS emits MATMUL) or "
                        "derive the aperture first"),
    "EVICT": ("releasing a shared-memory allocation is meaningless while nothing can be staged; see "
              "RES_PACK"),
}

#: Element types the emitter can express, with their C spelling and struct code. The command-buffer
#: ABI is integer-only, which is what makes this table short rather than a restriction chosen here.
CTYPE = {"i8": ("int8_t", "b", 1), "i16": ("int16_t", "h", 2),
         "i32": ("int32_t", "i", 4), "i64": ("int64_t", "q", 8)}

#: Epilogue stages realizable as a pure elementwise function of the accumulator.
EPILOGUE = {"relu"}

#: Combines the emitter can realize (the same set the interface tier accepts).
COMBINES = {"add": "+", "mul": "*"}

#: Identifiers the emitted scaffold and loop nests already own. A command-buffer tensor named ``s``
#: or ``n`` would shadow a loop variable and miscompile in silence, so the collision is refused.
RESERVED = frozenset({"main", "s", "m", "n", "i", "k", "wid", "a_row", "MU_NUM_WARPS",
                      "_wid", "_wmask", "_park", "_spawn", "_mu_worker", "MERLIN_KERNEL_DONE"})

#: The most warps this backend will spawn, regardless of how many the hardware declares.
#:
#: NOT a hardware fact and not a preference. The contract correctly declares 8 warp slots (the RTL config
#: sets ``WithSIMTConfig(numWarps = 8)``), but 8 does not produce correct results on the oracle we grade
#: against, so this caps what is spawned and says so in every result.
#:
#: What was MEASURED, on the RTL-arc model with three unrelated bodies (a constant store, a 32-bit load,
#: an 8-bit load): at 8 warps every case returned element 0 as zero with elements 1..7 correct; at 4 warps
#: all three were exact. Element 0 is the manager warp's, and the manager is the one warp whose store is
#: followed by almost no execution.
#:
#: The MECHANISM is deliberately not asserted. It was first read as the scaffold losing the spawning
#: warp's tile, but later measurements showed this oracle also drops stores that little or no execution
#: follows — which would explain the same table without any spawn defect at all. Two candidate causes fit
#: the evidence and separating them needs work on the model, not on this package. Naming a mechanism here
#: would put a guess in the one place a reader would trust it.
#:
#: Applied visibly: :func:`run_command_buffer` reports ``warps_declared`` alongside ``warps_used``, so a
#: capped run cannot be mistaken for one that used the whole machine.
ORACLE_SPAWN_WARPS = 4

#: Written by warp 0 only after every other warp has parked, so it means "the whole kernel finished",
#: not "the kernel started". Without it, a run that exhausts its cycle budget mid-computation returns a
#: partially written buffer, which is indistinguishable from a miscompile — and that is exactly what
#: happened here: the contraction was bit-exact and the commit had simply not been reached yet.
#: One cache line wide because the readback recovers whole lines.
SENTINEL = "MERLIN_KERNEL_DONE"

#: The arc model advances roughly a thousand cycles a second, so this budget is a real time cost, not
#: a formality. It is the value the fork-free multi-warp path is already proven at
#: (``muon.run_elf_arc``'s own default) rather than a number chosen here. A kernel that outruns it
#: leaves the tail of its output buffer at the .bss zero it started as — which reads as a wrong
#: answer with a correct prefix, i.e. loudly and distinguishably, never as a pass.
ARC_MAX_CYCLES = 120_000

#: The readback recovers device memory a cache line at a time. Not a free parameter: it is the line
#: size the model's recovery indexes its coverage array by, so a span that is shorter or unaligned
#: indexes past the end of it. It agrees with the target's own ``CACHE_LINE_BYTES``. A disagreement
#: with the model would surface as an IndexError out of the model — loudly — not as a wrong answer,
#: which is why this is a stated constant rather than a derivation with a fallback.
ARC_READBACK_LINE_BYTES = 64


class EmitError(RuntimeError):
    """The command buffer cannot be realized on this target — always names what is missing."""


@dataclass(frozen=True)
class ResultBuffer:
    """One graded output: its command-buffer name, C symbol, shape and element type."""

    name: str
    symbol: str
    rows: int
    cols: int
    dtype: str

    @property
    def n_bytes(self) -> int:
        return self.rows * self.cols * CTYPE[self.dtype][2]


@dataclass(frozen=True)
class EmittedKernel:
    source: str
    results: tuple[ResultBuffer, ...]
    num_warps: int                   # what the kernel actually spawns
    warps_declared: int              # what the contract says the hardware has

    @property
    def warps_capped(self) -> bool:
        return self.num_warps < self.warps_declared


# --------------------------------------------------------------------------- package facts
def contract() -> dict[str, Any]:
    """This package's own contract (the generated one, with its derived facts)."""
    import yaml

    path = HERE / "contracts" / "target_contract.yaml"
    if not path.is_file():
        raise EmitError(f"package contract missing: {path} (run derive_facts.py)")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def warps_per_core(spec: dict[str, Any] | None = None) -> int:
    """How many warps to map work across — DERIVED, never defaulted.

    A warp count guessed here would silently under-use the hardware (or over-spawn it) while every
    check downstream still passed, because the strided partitioning is correct for any count.
    """
    simt = ((spec or contract()).get("capabilities") or {}).get("simt") or {}
    warps = simt.get("warps_per_core")
    if warps is None:
        raise EmitError("contract declares no capabilities.simt.warps_per_core — re-run "
                        "derive_facts.py; a default warp count would be a fabricated hardware fact")
    return int(warps)


def _shape_of(tensor: dict[str, Any], name: str) -> tuple[int, int]:
    shape = list(tensor.get("shape") or [])
    if len(shape) != 2:
        raise EmitError(f"tensor {name!r} has shape {shape}; this emitter realizes 2-D operands only")
    return int(shape[0]), int(shape[1])


def _dtype_of(tensor: dict[str, Any], name: str) -> str:
    dtype = str(tensor.get("dtype") or "")
    if dtype not in CTYPE:
        raise EmitError(f"tensor {name!r} has element type {dtype!r}; this backend realizes integer "
                        f"element types {sorted(CTYPE)} (the command-buffer ABI is integer-only)")
    return dtype


# --------------------------------------------------------------------------- emission
def _c_array(symbol: str, dtype: str, values: list[int]) -> str:
    """A ``volatile`` initialised global. See the module docstring on why volatile is load-bearing."""
    ctype = CTYPE[dtype][0]
    body = ",".join(str(int(v)) for v in values)
    return f"volatile {ctype} {symbol}[{len(values)}] = {{{body}}};"


def _flat(rows: list[list[int]]) -> list[int]:
    return [v for row in rows for v in row]


def _row_strided(rows: int, cols: int, statement: str) -> str:
    """An elementwise loop nest partitioned by ROW, matching how the matmul splits its work.

    Striding a flat index by warp id instead would give every warp elements from every row — so a
    commit would read accumulator entries another warp is still writing. There is no barrier op
    derived for this target, so the emitter cannot fix that after the fact; the only safe schedule is
    one where the ownership of each command is identical, which is what this enforces.
    """
    return (f"  for (uint32_t m = wid; m < {rows}u; m += MU_NUM_WARPS)\n"
            f"    for (uint32_t n = 0; n < {cols}u; ++n) {{\n"
            f"      const uint32_t i = m * {cols}u + n;\n"
            f"      {statement}\n"
            f"    }}")


def accumulator_dtype(lhs_dtype: str, rhs_dtype: str, k: int) -> str:
    """The narrowest integer type that holds the contraction EXACTLY, computed from the operands.

    The command buffer declares the commit's output type but not the accumulator's, so this is derived
    rather than assumed: a product of a signed n-bit and a signed m-bit value needs n+m bits, and
    summing ``k`` of them needs ``ceil(log2 k)`` more. Picking a width by habit is how an accumulator
    silently wraps on a larger K while every small-K test still passes.
    """
    product_bits = CTYPE[lhs_dtype][2] * 8 + CTYPE[rhs_dtype][2] * 8
    sum_bits = 0
    while (1 << sum_bits) < max(k, 1):
        sum_bits += 1
    width = product_bits + sum_bits
    for dtype in ("i8", "i16", "i32", "i64"):
        if CTYPE[dtype][2] * 8 >= width:
            return dtype
    raise EmitError(
        f"a {lhs_dtype} x {rhs_dtype} contraction over K={k} needs {width} accumulator bits, wider "
        f"than any integer type this backend can express ({sorted(CTYPE)})")


def emit_kernel(cb: dict[str, Any], *, num_warps: int | None = None) -> EmittedKernel:
    """Turn a command buffer into the SIMT program that realizes it.

    Raises :class:`EmitError` naming the missing capability for anything it cannot realize honestly.
    """
    spec = contract()
    declared = warps_per_core(spec)
    warps = int(num_warps) if num_warps is not None else min(declared, ORACLE_SPAWN_WARPS)
    if warps < 1:
        raise EmitError(f"num_warps must be >= 1, got {warps}")

    tensors = cb.get("tensors") or {}
    commands = list(cb.get("commands") or [])
    if not commands:
        raise EmitError("command buffer has no commands")

    clashing = sorted(set(tensors) & RESERVED)
    if clashing:
        raise EmitError(
            f"tensor name(s) {clashing} collide with identifiers the emitted kernel already uses "
            f"({sorted(RESERVED)}); a shadowed loop variable miscompiles without any diagnostic")

    for cmd in commands:
        opcode = cmd.get("opcode")
        if opcode in REFUSED:
            raise EmitError(f"{opcode} cannot be realized on this target: {REFUSED[opcode]}")
        if opcode not in SUPPORTED:
            raise EmitError(f"no emission for opcode {opcode!r} (realizable: {sorted(SUPPORTED)}; "
                            f"refused with a reason: {sorted(REFUSED)})")

    # Leaf operands, materialized by the runtime so the emitter embeds exactly the values the
    # reference is computed from (never its own idea of the inputs).
    env = materialize_inputs(cb)
    globals_: list[str] = []
    # cb value name -> (C symbol, rows, cols, dtype)
    buffers: dict[str, tuple[str, int, int, str]] = {}

    for name, tensor in sorted(tensors.items()):
        if tensor.get("role") != "input":
            continue
        rows, cols = _shape_of(tensor, name)
        dtype = _dtype_of(tensor, name)
        if name not in env:
            raise EmitError(f"input tensor {name!r} was not materialized by the runtime")
        values = _flat(env[name].to_list())
        if len(values) != rows * cols:
            raise EmitError(f"input {name!r}: {len(values)} values for a {rows}x{cols} tensor")
        globals_.append(_c_array(name, dtype, values))
        buffers[name] = (name, rows, cols, dtype)

    body: list[str] = []
    results: list[ResultBuffer] = []

    def declare(symbol: str, rows: int, cols: int, dtype: str) -> None:
        globals_.append(f"volatile {CTYPE[dtype][0]} {symbol}[{rows * cols}];")
        buffers[symbol] = (symbol, rows, cols, dtype)

    def require(name: str) -> tuple[str, int, int, str]:
        if name not in buffers:
            raise EmitError(f"command reads {name!r}, which no earlier command produced")
        return buffers[name]

    for cmd in commands:
        opcode, ops = cmd["opcode"], cmd.get("operands") or {}
        attrs = cmd.get("attributes") or {}

        if opcode == "MATMUL":
            lhs, m, k, lhs_dt = require(ops["lhs"])
            rhs, k2, n, rhs_dt = require(ops["rhs"])
            if k != k2:
                raise EmitError(f"MATMUL shape mismatch: {m}x{k} @ {k2}x{n}")
            dst = ops["dst"]
            # The command buffer declares the commit's output type but never the accumulator's, so the
            # width is COMPUTED from the operands and K (see accumulator_dtype) rather than assumed.
            acc_dt = str(attrs.get("acc_dtype") or accumulator_dtype(lhs_dt, rhs_dt, k))
            if acc_dt not in CTYPE:
                raise EmitError(f"MATMUL accumulator type {acc_dt!r} is not an integer type")
            declare(dst, m, n, acc_dt)
            acc_c = CTYPE[acc_dt][0]
            # The LHS row is copied into a NON-volatile local once per row, then reused across the N
            # columns. Reading it straight out of the volatile operand instead is what the obvious
            # version does, and it costs M*N*K volatile loads rather than M*K — a 16x inflation on a
            # 16-wide tile, which is enough to run a whole cycle budget out inside the contraction.
            # `volatile` is still doing its job: the values entered through a volatile read, so the
            # compiler cannot know them and cannot fold the arithmetic away. It just no longer has to
            # re-read them, which is the hoist it would have done by itself for a normal array.
            body.append(
                f"  for (uint32_t m = wid; m < {m}u; m += MU_NUM_WARPS) {{\n"
                f"    {CTYPE[lhs_dt][0]} a_row[{k}];\n"
                f"    for (uint32_t k = 0; k < {k}u; ++k) a_row[k] = {lhs}[m * {k}u + k];\n"
                f"    for (uint32_t n = 0; n < {n}u; ++n) {{\n"
                f"      {acc_c} s = 0;\n"
                f"      for (uint32_t k = 0; k < {k}u; ++k)\n"
                f"        s += ({acc_c})a_row[k] * ({acc_c}){rhs}[k * {n}u + n];\n"
                f"      {dst}[m * {n}u + n] = s;\n"
                f"    }}\n"
                f"  }}")

        elif opcode == "COMMIT":
            src, rows, cols, src_dt = require(ops["src"])
            dst = ops["dst"]
            out_dt = str(attrs.get("output_dtype") or src_dt)
            if out_dt not in CTYPE:
                raise EmitError(f"COMMIT output_dtype {out_dt!r} is not an integer type")
            stages = [str(s) for s in (attrs.get("epilogue") or [])]
            unknown = [s for s in stages if s not in EPILOGUE]
            if unknown:
                raise EmitError(
                    f"COMMIT epilogue stages {unknown} are not realizable here (realizable: "
                    f"{sorted(EPILOGUE)}). A bias add needs the bias operand in the command buffer; "
                    "requantization must match Merlin's integer rounding exactly, which this emitter "
                    "does not derive — so it refuses rather than rounding differently")
            if attrs.get("requant_shift") is not None:
                raise EmitError(
                    "COMMIT carries requant_shift: this emitter does not reproduce Merlin's integer "
                    "requant rounding, and an approximation would differ from the reference in the "
                    "low bit while looking correct")
            declare(dst, rows, cols, out_dt)
            expr = f"{src}[i]"
            if "relu" in stages:
                expr = f"({expr} < 0 ? 0 : {expr})"
            body.append(_row_strided(rows, cols, f"{dst}[i] = ({CTYPE[out_dt][0]}){expr};"))
            results.append(ResultBuffer(dst, dst, rows, cols, out_dt))

        elif opcode == "VECTOR_MAP":
            lhs, rows, cols, lhs_dt = require(ops["lhs"])
            rhs, rrows, rcols, _ = require(ops["rhs"])
            if (rows, cols) != (rrows, rcols):
                raise EmitError(f"VECTOR_MAP operands differ in shape: {rows}x{cols} vs {rrows}x{rcols}")
            combine = str(attrs.get("combine") or "")
            if combine not in COMBINES:
                raise EmitError(f"VECTOR_MAP combine {combine!r} not realizable "
                                f"(realizable: {sorted(COMBINES)})")
            activations = [str(a) for a in (attrs.get("activation") or [])]
            unknown = [a for a in activations if a not in EPILOGUE]
            if unknown:
                raise EmitError(f"VECTOR_MAP activations {unknown} are not realizable here "
                                f"(realizable: {sorted(EPILOGUE)})")
            dst = ops["out"] if "out" in ops else ops["dst"]
            declare(dst, rows, cols, lhs_dt)
            expr = f"{lhs}[i] {COMBINES[combine]} {rhs}[i]"
            if "relu" in activations:
                expr = f"(({expr}) < 0 ? 0 : ({expr}))"
            body.append(_row_strided(rows, cols, f"{dst}[i] = ({CTYPE[lhs_dt][0]})({expr});"))
            results.append(ResultBuffer(dst, dst, rows, cols, lhs_dt))

    if not results:
        raise EmitError("command buffer commits no output; there would be nothing to grade")

    model = muon._model_for(TARGET)
    # NO FENCE IS EMITTED — a MEASURED decision now, not the missing capability it used to be.
    #
    # These stores ought to be ordered before the warp parks, and the target's own runtime does exactly
    # that (`mu_fence()` is plain `asm volatile("fence")`). The transcoder can re-map MISC_MEM now, so one
    # BUILDS. It also makes things worse. Measured on the arc oracle, same kernel shape each time:
    #
    #   body store + tail store, no fence      -> both land
    #   body store + fence, tail store + fence -> tail store lost
    #   body store + fence, no tail            -> even the store BEFORE the fence is lost
    #   body store alone, no fence, no tail    -> lost
    #
    # Two things are visible in that table and neither is this emitter's: a store is unreliably visible
    # when little or no execution follows it, and nothing after a fence executes at all. Whether the
    # fixed-format encoding of MISC_MEM's ordering bits is not what this decoder expects, or the model
    # simply does not implement the instruction, is not decidable from here.
    #
    # So the fence is left out. The alternative — padding the tail with filler stores until the real one
    # happens to become visible — would make the grade pass by exploiting the very race that makes the
    # grade meaningless, which is worse than an honest expected failure.
    globals_.append(f"volatile int32_t {SENTINEL}[{ARC_READBACK_LINE_BYTES // 4}];")
    # The scaffold already opens with <stdint.h>, so the globals need no include of their own. The
    # manager tail runs on warp 0 after it has waited for every other warp to park, which is the only
    # point in the program where "all the work is done" is true.
    source = muon.render_simt_runtime(model, num_warps=warps, worker_body="\n".join(body),
                                      manager_tail=f"{SENTINEL}[0] = 1;",
                                      globals="\n".join(globals_))
    return EmittedKernel(source=source, results=tuple(results), num_warps=warps,
                         warps_declared=declared)


# --------------------------------------------------------------------------- Backend protocol
def available() -> bool:
    """True when the stock toolchain, the derived ISA facts and the RTL-arc oracle are all present."""
    try:
        from merlin.targetgen.contract.toolchain import mlir_bin
        from merlin.targetgen.rtl import mlc_bridge

        if not mlir_bin("clang").is_file():
            return False
        if not mlc_bridge.isa_encoding_for(TARGET):
            return False
        return muon.arc_oracle_available(TARGET)
    except Exception:  # noqa: BLE001 — an unavailable dependency is an unavailable backend
        return False


def compile_command_buffer(cb: dict[str, Any], workdir: Any, **kw: Any) -> Path:
    """Emit + build the device ELF for ``cb``. Returns the ELF path."""
    emitted = emit_kernel(cb, num_warps=kw.get("num_warps"))
    work = Path(workdir)
    work.mkdir(parents=True, exist_ok=True)
    (work / "kernel.c").write_text(emitted.source, encoding="utf-8")
    return muon.compile_kernel_forkfree(emitted.source, work, target=TARGET,
                                        num_warps=emitted.num_warps)


def _symbol_address(elf: Path, symbol: str) -> int:
    """The link-time address of ``symbol``, read from the ELF — never a hardcoded memory map."""
    from merlin.targetgen.contract.toolchain import mlir_bin

    proc = subprocess.run([str(mlir_bin("llvm-objdump")), "-t", str(elf)],
                          capture_output=True, text=True)
    for line in proc.stdout.splitlines():
        fields = line.split()
        if fields and fields[-1] == symbol:
            return int(fields[0], 16)
    raise EmitError(f"symbol {symbol!r} not found in {elf} — the output buffer was optimized away "
                    "or renamed, so there is nothing to read back")


#: The pin declaring the oracle model this backend grades against. Named here because this package is
#: what chose that oracle; the pin registry holds the revision.
ORACLE_PIN = "muon_arc_model"


def oracle_provenance() -> dict[str, Any]:
    """Which oracle build this backend is about to grade against, verified and digested.

    A result attributed to the wrong device is worse than no result, and the arc model is a *built*
    artifact whose path says nothing about which build it is — so the model binary is digested, not just
    located. ``$MERLIN_MLC_DIR`` is normally unset and the checkout is found by an implicit default, so the
    path is resolved the way the code resolves it and handed to ``verify`` explicitly; verifying the env
    var instead would report drift for a checkout that is present and correct.

    Distinguishes MATERIAL drift (wrong revision, a required path absent) from a merely dirty tree. Both
    are recorded; only the former means the run cannot be attributed to the hardware it claims.
    """
    from merlin.common import provenance as PROV
    from merlin.targetgen.rtl import mlc_bridge

    checkout = mlc_bridge.mlc_dir()
    verification = PROV.verify(ORACLE_PIN, checkout=checkout)
    model_so = (Path(checkout) / "runs" / "circt-arc" / mlc_bridge._arc_target(TARGET)
                / "native_run" / "libmuon_model.so") if checkout else None
    record = PROV.record(
        pins={ORACLE_PIN: verification},
        artifacts={"arc_model": model_so} if model_so and model_so.is_file() else None)
    material = [*verification.missing_paths, *verification.forbidden_present,
                *(d for d in verification.drift if "uncommitted" not in d)]
    record["attributable"] = not material
    record["gaps"] = material
    return record


def _read_back(elf: Path, symbol: str, n_bytes: int, max_cycles: int, timeout: int) -> bytes:
    """Run the model and read ``n_bytes`` of ``symbol`` back out of device memory.

    The recovery works a CACHE LINE at a time — it indexes a per-line coverage array — so the span is
    aligned down to a line and rounded up to whole lines. Asking for the buffer's exact extent raises
    inside the model for anything sub-line or unaligned, and rounding the span DOWN would quietly
    return less of the buffer than was asked for.
    """
    address = _symbol_address(elf, symbol)
    low = address & ~(ARC_READBACK_LINE_BYTES - 1)
    span = (((address - low) + n_bytes + ARC_READBACK_LINE_BYTES - 1)
            // ARC_READBACK_LINE_BYTES * ARC_READBACK_LINE_BYTES)
    data = muon.run_elf_arc(elf, target=TARGET, base=low, length=span,
                            max_cycles=max_cycles, timeout=timeout)
    if data is None:
        raise muon.MuonUnavailable("the RTL-arc model went absent mid-run")
    offset = address - low
    window = data[offset:offset + n_bytes]
    if len(window) != n_bytes:
        raise EmitError(f"readback returned {len(window)} bytes for {symbol}, expected {n_bytes} — "
                        "a short readback must not be graded")
    return window


def run_elf(elf: Any, **kw: Any) -> str:
    """Run a device ELF on a console-producing simulator (delegates to the target's runner)."""
    console, _cycles, _summary = muon.run_elf(elf, simulator=kw.get("simulator", "cyclotron"),
                                             timeout=kw.get("timeout", 600))
    return console


def parse_output(text: str) -> tuple[dict, dict]:
    """Parse the shared console protocol. Present for protocol completeness — the graded path reads
    results back from memory instead, because a console print races across the lanes of a warp."""
    return muon.parse_output(text, None)


def run_command_buffer(cb: dict[str, Any], **kw: Any) -> dict[str, Any]:
    """Build ``cb``, execute it on the target's RTL-derived arc model, and grade it bit-exact.

    Grading reads the output buffers back from device memory rather than parsing a console: a print
    from a SIMT kernel interleaves across lanes. Each output costs one run of the model — the kernel
    is deterministic, so re-running is sound, and it is honest about the cost rather than reading one
    span across buffers whose layout it does not control.
    """
    if not available():
        raise muon.MuonUnavailable(
            "the SIMT backend needs the stock LLVM tools, the derived ISA encoding fact and the "
            "RTL-arc model (set MERLIN_MLIR_INSTALL / MERLIN_CHIPYARD; build the arc model)")
    own_tmp = kw.get("workdir") is None
    work = Path(tempfile.mkdtemp(prefix="merlin_radiance_")) if own_tmp else Path(kw["workdir"])
    emitted = emit_kernel(cb, num_warps=kw.get("num_warps"))
    (work / "kernel.c").write_text(emitted.source, encoding="utf-8")
    elf = muon.compile_kernel_forkfree(emitted.source, work, target=TARGET,
                                       num_warps=emitted.num_warps)

    cycles = int(kw.get("max_cycles") or ARC_MAX_CYCLES)
    timeout = int(kw.get("timeout") or 900)

    # Captured BEFORE the run, so what is recorded is the oracle that actually executed it.
    provenance = oracle_provenance()

    # Completion FIRST: a partially written output buffer must never be compared to the reference.
    done = struct.unpack("<i", _read_back(elf, SENTINEL, 4, cycles, timeout))[0]
    if done != 1:
        raise EmitError(
            f"the kernel never reached its completion sentinel (it reads {done}, not 1) with "
            f"max_cycles={cycles} and {emitted.num_warps} warp(s). The output buffer is therefore "
            "incomplete and has NOT been graded. Two causes look identical here and both are real: the "
            "cycle budget ran out mid-computation (raise max_cycles), or the manager warp never reached "
            f"the tail — which is what spawning more than {ORACLE_SPAWN_WARPS} warps does on this "
            "oracle (see ORACLE_SPAWN_WARPS). Either way, refusing to grade beats comparing a partial "
            "buffer to a reference and calling the difference a miscompile.")

    outputs: dict[str, list] = {}
    for buf in emitted.results:
        data = _read_back(elf, buf.symbol, buf.n_bytes, cycles, timeout)
        code = CTYPE[buf.dtype][1]
        flat = list(struct.unpack("<" + code * (buf.rows * buf.cols), data))
        outputs[buf.name] = [flat[r * buf.cols:(r + 1) * buf.cols] for r in range(buf.rows)]

    expected = reference_outputs(cb)
    return {
        "outputs": outputs,
        "expected": expected,
        "correct": outputs_match(outputs, expected),
        "oracle": {"kind": "rtl_arc_muon_cosim", "derived_from_rtl": True},
        # Which hardware build this verdict belongs to. `correct` says the numbers matched; `provenance`
        # says what they matched ON. A run whose provenance is not attributable must not be read as a
        # certification, however green `correct` is.
        "provenance": provenance,
        "attributable": provenance.get("attributable"),
        "elf": str(elf),
        # Both numbers, always: a run that used half the declared machine must not read as a run that
        # used all of it. See ORACLE_SPAWN_WARPS for why they differ.
        "warps_used": emitted.num_warps,
        "warps_declared": emitted.warps_declared,
        "warps_capped": emitted.warps_capped,
        "source": emitted.source,
    }


register(BackendInfo(BACKEND_NAME, TargetClass.GPU, BackendKind.KERNEL, __name__))
