"""Emit the host/device boundary of a ``device_native`` device: a DRAM ADDRESS CONTRACT.

The sibling :mod:`~merlin.llvmlower.device_build` compiles the boundary of a device the host drives
with its own instructions -- the package emits LLVM-dialect MLIR, that becomes an object, and the
seam is a linkable call. A ``device_native`` device has no such call. It fetches and decodes its own
instruction stream, and the two sides meet at agreed DRAM addresses: the host stages every operand at
the address the device's program will load from, the harness starts the device, and the host reads
the result back from the address the program stored to. Nothing links; the addresses ARE the seam.

That is why ``device_build`` refused these devices, and why the composition axis then reported
UNKNOWN for them -- correctly, because an axis that says "this capsule crosses the host/device
boundary" on a target whose boundary nothing can emit is asserting something no artifact backs.

**What this module emits, and why each half is needed.**

* the DEVICE half -- the package's own artifact for one interface capsule, ASSEMBLED to the bytes the
  device fetches (stock ``llvm-mc`` + ``llvm-objcopy`` over the emitted ``.word``/``.insn``
  directives, the same target-agnostic path the program oracle uses). Assembling is not decoration:
  it is what distinguishes an artifact that is machine code from an artifact that is text;
* the ADDRESS CONTRACT -- every tensor's address, taken from the command buffer the SAME package
  emits for the SAME capsule, checked against the device's own derived DRAM window and converted to
  an offset within it;
* the HOST half -- a C translation unit that stages each operand at its offset and collects each
  result from its offset, compiled to an object. This is the piece that was missing: without it the
  address contract exists only as a document, and a host that writes an operand somewhere else is
  exactly the defect that graded a conformant backend 0 (the oracle preloaded operand A where the
  layout said, the kernel read it from where it guessed).

**Two things are deliberately NOT emitted, because nothing here derives them.**

*The address translation.* Whether a host address and a device address are the same number is not
derivable for this transport (``Link.address_translation`` is None, with the evidence saying so), so
the generated code never names an absolute host address. It works in OFFSETS from the window and
takes the window's host-visible base as a parameter. A fabricated identity mapping would be a wrong
address that looks like a measurement.

*The launch.* No fact in this repo says how the host rings this device's doorbell -- there is no
control-aperture interface among its derived facts. So the launch is a FUNCTION POINTER the harness
supplies, not an ``extern`` the archive fails to define: the emitted object is self-contained and
links on its own, and the one step that is genuinely the harness's stays the harness's.

Nothing here knows which device it is building for. The window base, the artifact, the addresses and
the extents are all read from the device's own derived facts and its package's own output.
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from .device_build import _ar, _flags, _run  # noqa: PLC2701 -- one toolchain locator per package

__all__ = ["DeviceNativeSeam", "SeamProgram", "SeamTensor", "build_device_native_seam",
           "emit_host_staging_unit", "seam_emittable", "seam_entry"]

#: The operand placement this seam IS. A ``device_native`` device whose operands arrive some other way
#: (pointer arguments it pulls over DMA, say) has a different boundary, and emitting an address
#: contract for it would stage operands the device never reads.
_SEAM_PLACEMENT = "preload_at_base"
_SEAM_TRANSPORT = "device_native"

#: Roles whose bytes the HOST writes before the device runs, and roles it reads back afterwards. Read
#: off the command buffer's own tensor entries; a role in neither set is carried in the contract but
#: staged by nobody, and is reported rather than silently dropped.
_INPUT_ROLES = frozenset({"input", "weight"})
_OUTPUT_ROLES = frozenset({"output"})


# ---------------------------------------------------------------------------------------------
# the predicate
# ---------------------------------------------------------------------------------------------

def seam_emittable(device: str) -> str | None:
    """Why this device's ``device_native`` seam cannot be emitted, or None when it can.

    Every clause is a fact the emitted artifact would otherwise have to invent, and each is quoted
    with the evidence the Link recorded for it, so a refusal names what to go and derive.
    """
    try:
        from merlin.system.derive import link_for
        from merlin.targetgen.target_experiment import load_capability_manifest
        endpoint = getattr(load_capability_manifest(device), "endpoint_kind", None)
        link = link_for(device, endpoint)
    except Exception as exc:                 # noqa: BLE001
        return (f"{device!r}: its link could not be derived ({type(exc).__name__}), so nothing is "
                f"known about how the host reaches it")
    if link.command_transport != _SEAM_TRANSPORT:
        return (f"{device!r} is reached by {link.command_transport!r}, not {_SEAM_TRANSPORT!r}; this "
                f"path emits a DRAM address contract and that transport's boundary is a different one")
    if link.operand_placement != _SEAM_PLACEMENT:
        return (f"{device!r} places its operands by {link.operand_placement!r}, not "
                f"{_SEAM_PLACEMENT!r} ({link.evidence.get('operand_placement', 'no evidence')}); an "
                f"address contract would stage operands this device does not read from there")
    if link.device_dram_base is None:
        return (f"{device!r} declares no derivable DRAM window base "
                f"({link.evidence.get('device_dram_base', 'nothing looked at it')}); the seam IS the "
                f"address contract, so without the window the host cannot know where to stage an "
                f"operand and an assumed base is a wrong address that looks like a measurement")
    if not link.emitted_artifact:
        return (f"{device!r} declares no emitted artifact "
                f"({link.evidence.get('emitted_artifact', 'nothing looked at it')}); the device half "
                f"of the seam is whatever its package emits, and that is unknown here")
    return None


# ---------------------------------------------------------------------------------------------
# what a seam is made of
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class SeamTensor:
    """One tensor's place in the address contract, as the package declared it."""

    name: str
    role: str
    dtype: str
    shape: tuple[int, ...]
    #: The absolute device address the package declared for this tensor.
    device_address: int
    #: That address as a byte offset from the start of the device's derived DRAM window.
    window_offset: int
    nbytes: int
    #: The package's own physical-layout note for a result, passed through untouched (the host
    #: collects BYTES; how they are unpacked into a logical tensor is the reader's business).
    physical: dict | None = None

    def to_dict(self) -> dict:
        return {"name": self.name, "role": self.role, "dtype": self.dtype, "shape": list(self.shape),
                "device_address": self.device_address, "window_offset": self.window_offset,
                "nbytes": self.nbytes, "physical": self.physical}


@dataclass(frozen=True)
class SeamProgram:
    """One signature's seam: the bytes the device fetches, and where its operands live."""

    #: The C identifier the host entries are emitted under.
    entry: str
    #: The offload signature symbol this was built for.
    signature: str
    extents: tuple[int, ...]
    #: The assembled device program, exactly as the device fetches it.
    image: bytes
    tensors: tuple[SeamTensor, ...]

    @property
    def inputs(self) -> tuple[SeamTensor, ...]:
        return tuple(t for t in self.tensors if t.role in _INPUT_ROLES)

    @property
    def outputs(self) -> tuple[SeamTensor, ...]:
        return tuple(t for t in self.tensors if t.role in _OUTPUT_ROLES)

    def to_dict(self) -> dict:
        return {"entry": self.entry, "signature": self.signature, "extents": list(self.extents),
                "program_bytes": len(self.image),
                "tensors": [t.to_dict() for t in self.tensors]}


@dataclass(frozen=True)
class DeviceNativeSeam:
    """What was emitted, and what could not be."""

    device: str
    #: The device's own DRAM window base, derived. Every offset in ``programs`` is relative to it.
    window_base: int | None = None
    programs: tuple[SeamProgram, ...] = ()
    host_source: Path | None = None
    host_object: Path | None = None
    #: The machine-readable address contract, for a harness that stages the operands itself.
    contract_path: Path | None = None
    skipped: tuple[tuple[str, str], ...] = ()

    @property
    def ok(self) -> bool:
        """Both halves present. A seam with a device image and no host stager is not a seam."""
        return bool(self.programs) and self.host_object is not None

    def to_dict(self) -> dict:
        return {"device": self.device, "window_base": self.window_base,
                "programs": [p.to_dict() for p in self.programs],
                "host_source": str(self.host_source) if self.host_source else None,
                "host_object": str(self.host_object) if self.host_object else None,
                "skipped": [list(s) for s in self.skipped]}

    def archive(self, path: str | Path) -> Path | None:
        """Bundle the host object into a static archive the board build links, or None.

        Same shape and same reason as :meth:`DeviceBuild.archive`: the final link belongs to the
        board build, which owns the toolchain. There is no device object to bundle -- the device's
        bytes are not linked, they are fetched -- so the archive carries the host half and the image
        rides inside it as data.
        """
        if self.host_object is None:
            return None
        ar = _ar()
        if ar is None:
            return None
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists():
            out.unlink()                 # ar appends; a stale member would shadow a rebuilt one
        r = _run([ar, "rcs", str(out), str(self.host_object)], timeout=300)
        return out if r.returncode == 0 and out.exists() else None


# ---------------------------------------------------------------------------------------------
# reading the address contract
# ---------------------------------------------------------------------------------------------

def seam_entry(signature: str) -> str:
    """A C identifier for one signature's host entries.

    Structural, not cosmetic: the offload rewrite mints a symbol per signature and the host side
    needs one stager per signature, so two signatures sharing an entry name would stage the second
    layer's operands through the first layer's offsets -- the same silent mis-binding the kernel
    symbol rename exists to prevent, one level up.
    """
    kept = [c if (c.isalnum() or c == "_") else "_" for c in str(signature)]
    name = "".join(kept) or "seam"
    return name if not name[0].isdigit() else f"s_{name}"


def _nbytes(shape: Sequence[int], dtype: str) -> int:
    """Storage bytes for a whole tensor, from the format registry rather than a width table.

    Ceil over the packed bit total, so a sub-byte format occupies what it actually occupies. An
    unknown token raises: a silent wrong size mis-stages this tensor AND every tensor after it, which
    is precisely the class of defect the address contract exists to close.
    """
    from merlin.common import quant_formats as qf

    key = str(dtype)
    if key.startswith("torch."):
        key = key[len("torch."):]
    if qf.has(key):
        fmt = qf.get(key)
        bits = int(fmt.pack_bits or fmt.element_bits)
    else:
        machine = qf.machine_bits(key)
        if machine is None:
            raise KeyError(f"cannot size dtype {key!r}: it is neither a registered format "
                           f"({qf.names()}) nor a machine width; register it rather than assuming")
        bits = int(machine)
    n = 1
    for d in shape:
        n *= int(d)
    return (n * bits + 7) // 8


def address_contract(command_buffer: Mapping, *, window_base: int) -> tuple[SeamTensor, ...]:
    """The declared tensors as window-relative slots, or raise saying which fact is missing.

    Fails closed on three separate ways the contract can be unhonourable, each of which produced a
    wrong answer rather than an error before there was a contract to check:

    * a tensor with no declared address -- the host would have to guess where to put it;
    * an address BELOW the device's own DRAM window -- either the window or the address is wrong, and
      staging at a negative offset writes outside whatever the harness mapped;
    * two tensors whose byte ranges overlap -- staging the second silently destroys the first.
    """
    tensors = (command_buffer or {}).get("tensors") or {}
    if not tensors:
        raise ValueError("the package's command buffer declares no tensors; there is no address "
                         "contract to honour")
    out: list[SeamTensor] = []
    for name, spec in tensors.items():
        spec = spec or {}
        addr = spec.get("base")
        if addr is None:
            raise ValueError(f"tensor {name!r} carries no address; the host cannot stage a tensor "
                             f"whose address the device program was never told")
        addr = int(addr)
        if addr < window_base:
            raise ValueError(f"tensor {name!r} is addressed at {addr:#x}, below this device's DRAM "
                             f"window base {window_base:#x}; one of the two is wrong and staging at "
                             f"a negative offset writes outside the window")
        shape = tuple(int(d) for d in (spec.get("shape") or ()))
        dtype = str(spec.get("dtype") or "")
        out.append(SeamTensor(name=str(name), role=str(spec.get("role") or ""), dtype=dtype,
                              shape=shape, device_address=addr, window_offset=addr - window_base,
                              nbytes=_nbytes(shape, dtype), physical=spec.get("physical")))
    ordered = sorted(out, key=lambda t: t.window_offset)
    for prev, nxt in zip(ordered, ordered[1:]):
        if prev.window_offset + prev.nbytes > nxt.window_offset:
            raise ValueError(
                f"tensors {prev.name!r} ({prev.window_offset:#x}+{prev.nbytes}) and {nxt.name!r} "
                f"({nxt.window_offset:#x}) overlap in the device window; staging the second would "
                f"destroy the first")
    return tuple(out)


# ---------------------------------------------------------------------------------------------
# the host half
# ---------------------------------------------------------------------------------------------

_PREAMBLE = """/* Generated by merlin.llvmlower.device_native for device {device!r}. Do not edit.
 *
 * THE SEAM IS AN ADDRESS CONTRACT, NOT A CALL. This device fetches and decodes its own instruction
 * stream, so there is nothing here to link against it. The two sides meet at agreed addresses in the
 * device's DRAM window: the host stages every operand where the device program will load it from,
 * the harness starts the device, and the host reads each result back from where the program stored
 * it. Every offset below came from the command buffer the device's own backend package emitted for
 * the same capsule as the program image beside it, so both halves describe one layout by
 * construction rather than by two independent guesses agreeing.
 *
 * NO ABSOLUTE HOST ADDRESS APPEARS HERE. Whether a host address and a device address are the same
 * number is not derivable for this transport, so everything is an offset and the window's
 * host-visible base is a parameter the harness passes in.
 *
 * THE LAUNCH IS A PARAMETER, NOT AN EXTERN. Nothing in this repo derives a control aperture for this
 * device, so the step that starts it is supplied by the caller; this object links on its own.
 */
#include <stdint.h>
#include <string.h>

/* The device's DRAM window base, DERIVED from its own memory map: {window_base:#x}.
 * Recorded for the reader and for a harness that maps the window by absolute address; the code below
 * never uses it, because it works entirely in offsets. */
#define MERLIN_DEVICE_WINDOW_BASE UINT64_C({window_base})

typedef struct {{
  uint64_t offset;    /* bytes from the start of the device's DRAM window */
  uint64_t nbytes;    /* storage bytes, from the tensor's own extents and element format */
}} merlin_seam_slot;

/* Start the device on `program` and return 0 once it has finished. Supplied by the harness. */
typedef int (*merlin_device_launch_fn)(void *ctx, const unsigned char *program,
                                       uint64_t program_bytes);
"""

_ENTRY = """
/* ---- {entry}: {extents} ---- */
static const unsigned char {entry}_program[] = {{
{image}
}};
static const merlin_seam_slot {entry}_inputs[] = {{
{inputs}
}};
static const merlin_seam_slot {entry}_outputs[] = {{
{outputs}
}};

uint64_t {entry}_program_bytes(void) {{ return (uint64_t)sizeof {entry}_program; }}
const unsigned char *{entry}_program_image(void) {{ return {entry}_program; }}
uint64_t {entry}_input_count(void) {{ return {n_in}u; }}
uint64_t {entry}_output_count(void) {{ return {n_out}u; }}

/* Write each operand where this program will read it from. `operands` is in the order the package
 * declared the tensors, which is the order the offsets below are in. */
void {entry}_stage(unsigned char *window, const void *const *operands) {{
  for (unsigned i = 0; i < {n_in}u; ++i)
    memcpy(window + {entry}_inputs[i].offset, operands[i], (size_t){entry}_inputs[i].nbytes);
}}

/* Read each result back from where this program stored it. */
void {entry}_collect(const unsigned char *window, void *const *results) {{
  for (unsigned i = 0; i < {n_out}u; ++i)
    memcpy(results[i], window + {entry}_outputs[i].offset, (size_t){entry}_outputs[i].nbytes);
}}

/* The whole crossing: stage, hand the program to the harness, collect. A null launch is refused
 * rather than treated as success -- collecting without running reads whatever was in the window. */
int {entry}_dispatch(unsigned char *window, const void *const *operands, void *const *results,
                     merlin_device_launch_fn launch, void *ctx) {{
  int rc;
  if (window == 0 || launch == 0) return -1;
  {entry}_stage(window, operands);
  rc = launch(ctx, {entry}_program, (uint64_t)sizeof {entry}_program);
  if (rc != 0) return rc;
  {entry}_collect(window, results);
  return 0;
}}
"""


def _image_lines(image: bytes, per_line: int = 12) -> str:
    rows = []
    for i in range(0, len(image), per_line):
        rows.append("  " + " ".join(f"0x{b:02x}," for b in image[i:i + per_line]))
    return "\n".join(rows)


def _slot_lines(slots: Sequence[SeamTensor]) -> str:
    if not slots:
        # A zero-length array is not C. An entry with no operands on one side still needs the array
        # to exist, and a single zero-byte slot is never iterated because the count is 0.
        return "  { 0u, 0u }  /* none declared */"
    return "\n".join(f"  {{ {t.window_offset}u, {t.nbytes}u }},   /* {t.name}: {t.dtype}"
                     f"{list(t.shape)} at {t.device_address:#x} */" for t in slots)


def emit_host_staging_unit(device: str, programs: Sequence[SeamProgram], *,
                           window_base: int) -> str:
    """The host half of the seam, as one C translation unit."""
    parts = [_PREAMBLE.format(device=device, window_base=int(window_base))]
    for prog in programs:
        parts.append(_ENTRY.format(
            entry=prog.entry,
            extents="x".join(str(e) for e in prog.extents) or "unknown extents",
            image=_image_lines(prog.image),
            inputs=_slot_lines(prog.inputs),
            outputs=_slot_lines(prog.outputs),
            n_in=len(prog.inputs), n_out=len(prog.outputs)))
    return "".join(parts)


# ---------------------------------------------------------------------------------------------
# the device half
# ---------------------------------------------------------------------------------------------

def assemble_device_image(source: Path, workdir: Path, *, timeout: int = 900) -> bytes:
    """Assemble a package's emitted device artifact into the bytes the device fetches.

    Stock ``llvm-mc`` + ``llvm-objcopy`` over the artifact's own ``.word``/``.insn`` directives -- the
    same target-agnostic path the program oracle assembles a submission with. Merlin holds no opcode
    table: the encoding lives in the directives the package emitted, which it derived from the
    target's own ISA. An empty ``.text`` is an error, not an empty program: an artifact that
    assembled to nothing is one this pipeline could not turn into anything the device can fetch.
    """
    from merlin.targetgen.contract.toolchain import mlir_bin

    mc, objcopy = mlir_bin("llvm-mc"), mlir_bin("llvm-objcopy")
    if not mc.is_file() or not objcopy.is_file():
        raise FileNotFoundError(f"stock LLVM assembler absent ({mc} / {objcopy}); set "
                                f"MERLIN_MLIR_INSTALL")
    obj, binf = workdir / f"{source.stem}.dev.o", workdir / f"{source.stem}.dev.bin"
    a = _run([str(mc), "-triple=riscv64", "-filetype=obj", "-o", str(obj), str(source)],
             timeout=timeout)
    if a.returncode != 0:
        raise ValueError(f"llvm-mc declined the emitted device artifact: {(a.stderr or '')[-300:]}")
    b = _run([str(objcopy), "-O", "binary", "--only-section=.text", str(obj), str(binf)],
             timeout=timeout)
    if b.returncode != 0:
        raise ValueError(f"llvm-objcopy: {(b.stderr or '')[-300:]}")
    image = binf.read_bytes()
    if not image:
        raise ValueError("the emitted device artifact assembled to zero .text bytes; there is no "
                         "program for the device to fetch")
    return image


# ---------------------------------------------------------------------------------------------
# the build
# ---------------------------------------------------------------------------------------------

def build_device_native_seam(device: str,
                             signatures: Mapping[str, Sequence[int]],
                             *,
                             package_dir: str | Path,
                             workdir: str | Path,
                             operand_dtype: str,
                             accum_dtype: str,
                             codegen_target: str = "riscv",
                             cflags: "Sequence[str] | None" = None,
                             timeout: int = 900) -> DeviceNativeSeam:
    """Emit one device program plus its address contract per signature, and the host stager for all.

    ``signatures`` comes from the offload rewrite, exactly as for
    :func:`~merlin.llvmlower.device_build.build_device_objects`; ``operand_dtype`` / ``accum_dtype``
    are the device's own datapath tokens, derived by the caller rather than assumed.

    Every per-signature failure is recorded and skipped rather than raised, for the same reason the
    linkable path does it: a model whose third extent the package declines should still emit the seam
    for its other two and say what it lost.
    """
    from merlin.targetgen import corpus_spec as CS
    from merlin.targetgen.oot_runner import load_package, run_entrypoint

    from .toolchain import clang

    work = Path(workdir)
    work.mkdir(parents=True, exist_ok=True)
    skipped: list[tuple[str, str]] = []

    unemittable = seam_emittable(device)
    if unemittable:
        return DeviceNativeSeam(device=device, skipped=(("all", unemittable),))

    from merlin.system.derive import link_for
    from merlin.targetgen.target_experiment import load_capability_manifest
    link = link_for(device, getattr(load_capability_manifest(device), "endpoint_kind", None))
    window_base = int(link.device_dram_base)

    try:
        pkg = load_package(str(package_dir))
    except Exception as exc:                     # noqa: BLE001
        return DeviceNativeSeam(device=device, window_base=window_base,
                                skipped=(("all", f"package unusable: {exc}"),))

    from merlin.compile_cli import _mesh_tile_binding
    binding = _mesh_tile_binding(device, operand_dtype, accum_dtype)

    programs: list[SeamProgram] = []
    for sym in sorted(signatures):
        key = tuple(int(v) for v in signatures[sym])
        if len(key) not in (3, 4):
            skipped.append((sym, f"signature {key} has neither 3 nor 4 extents; no kernel shape "
                                 f"for it"))
            continue
        # A batched signature is the same device program as its unbatched form -- the batch is a loop
        # over disjoint slices on the host side, not an axis the device sees.
        m, n, k = key[-3:]
        entry = {"name": sym, "op": "matmul", "kind": "op",
                 "source_role": "mesh_tile_synthesized",
                 "source_reference": f"offloaded layer {m}x{k}x{n} for {device}",
                 "M": m, "K": k, "N": n}
        try:
            _capsule, iface = CS.build(entry, binding)
        except Exception as exc:                 # noqa: BLE001
            skipped.append((sym, f"interface capsule: {exc}"))
            continue
        stem = work / f"{seam_entry(sym)}"
        ifc = stem.with_suffix(".iface.mlir")
        ifc.write_text(iface, encoding="utf-8")

        art = stem.with_suffix(".device.S")
        r = run_entrypoint(pkg, "emit_target_artifact", ifc, timeout=timeout)
        if r.returncode != 0:
            skipped.append((sym, f"package declined {m}x{k}x{n}: {(r.stderr or '').strip()[:200]}"))
            continue
        art.write_text(r.stdout, encoding="utf-8")

        # THE ADDRESSES COME FROM THE SAME PACKAGE RUN ON THE SAME CAPSULE as the program above.
        # Deriving them a second way here -- a layout of our own -- is how the two halves come to
        # describe different layouts while each looks right on its own.
        cbf = stem.with_suffix(".contract.json")
        c = run_entrypoint(pkg, "emit_command_buffer", ifc, cbf, timeout=timeout)
        if c.returncode != 0 or not cbf.is_file():
            skipped.append((sym, f"package emitted no address contract for {m}x{k}x{n}: "
                                 f"{(c.stderr or '').strip()[:200]}"))
            continue
        try:
            tensors = address_contract(json.loads(cbf.read_text(encoding="utf-8")),
                                       window_base=window_base)
        except Exception as exc:                 # noqa: BLE001
            skipped.append((sym, f"address contract: {exc}"))
            continue

        try:
            image = assemble_device_image(art, work, timeout=timeout)
        except Exception as exc:                 # noqa: BLE001
            skipped.append((sym, f"device image: {exc}"))
            continue

        programs.append(SeamProgram(entry=seam_entry(sym), signature=sym, extents=key,
                                    image=image, tensors=tensors))

    if not programs:
        return DeviceNativeSeam(device=device, window_base=window_base, skipped=tuple(skipped))

    host_c = work / "device_seam.c"
    host_c.write_text(emit_host_staging_unit(device, programs, window_base=window_base),
                      encoding="utf-8")
    contract_p = work / "seam_contract.json"
    contract_p.write_text(json.dumps(
        {"device": device, "window_base": window_base,
         "programs": [p.to_dict() for p in programs]}, indent=2), encoding="utf-8")

    host_o = work / "device_seam.o"
    s = _run([clang(), *_flags(codegen_target, cflags), "-c", str(host_c), "-o", str(host_o)],
             timeout=timeout)
    if s.returncode != 0:
        skipped.append(("host", f"clang: {(s.stderr or '').strip()[:300]}"))
        return DeviceNativeSeam(device=device, window_base=window_base, programs=tuple(programs),
                                host_source=host_c, contract_path=contract_p, skipped=tuple(skipped))

    return DeviceNativeSeam(device=device, window_base=window_base, programs=tuple(programs),
                            host_source=host_c, host_object=host_o, contract_path=contract_p,
                            skipped=tuple(skipped))
