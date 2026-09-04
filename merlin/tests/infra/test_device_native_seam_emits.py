"""EMIT a ``device_native`` host/device seam end to end, and check the artifact rather than the plan.

The composition axis (:mod:`merlin.targetgen.boundary`) refuses to call a capsule ``H->A->H`` on a
target whose seam nothing in this repo can build, and it is right to: being able to CLASSIFY which
side a region falls on is not evidence that the crossing is exercised. The only thing that lifts that
refusal is an artifact. So this module builds one -- for whatever ``device_native`` target this
checkout actually has a backend package for -- and asserts on the bytes:

* the DEVICE half is the package's own emitted directives ASSEMBLED (``llvm-mc`` + ``llvm-objcopy``),
  so the image is machine code and not text that looks like it;
* the ADDRESS CONTRACT is window-relative, every address inside the device's own derived DRAM window,
  every size taken from the tensor's declared format;
* the HOST half compiles to an object that defines one stager per signature and leaves nothing but
  libc undefined -- and, linked against a harness whose launch is a stub, actually writes each operand
  at the offset the contract declares and reads each result back from the offset it declares.

NOTHING HERE IS HARDCODED PER TARGET. The target is discovered (any target whose derived link says
``device_native`` and whose window is derivable), the extents come from its own mesh tile dim, the
dtypes from its own compute unit, and the window base from its own memory map. A target-name literal
in this file would make the test prove something about one device instead of about the seam.

The host half is compiled for the machine running the harness (x86 here), which is what it is: host
code. The device half is assembled for the device. That the two are different architectures is the
whole point of a seam.
"""
from __future__ import annotations

import struct
import subprocess
from pathlib import Path

import pytest

from merlin.common.paths import artifacts_dir


# ------------------------------------------------------------------------------------------------
# discovery -- what this checkout can actually build a seam for
# ------------------------------------------------------------------------------------------------

def _emittable_target_with_a_package() -> tuple[str, Path] | None:
    """A (target, package_dir) whose seam this repo can emit, or None if this checkout has none.

    Backend packages are generated output (``out/artifacts/targets/<target>/<package_id>/``) and only
    the hand-authored baselines are tracked, so a fresh clone legitimately has none. Discovering the
    pair rather than naming it keeps this test about the transport instead of about one device.
    """
    from merlin.llvmlower.device_native import seam_emittable
    from merlin.targetgen.target_registry import all_targets

    root = artifacts_dir() / "targets"
    for target in all_targets():
        if seam_emittable(target) is not None:
            continue
        for pkg in sorted((root / target).glob("*/manifest.yaml")):
            return target, pkg.parent
    return None


def _toolchain_missing() -> str | None:
    """Which tool the seam needs and this environment lacks."""
    from merlin.llvmlower.toolchain import clang
    from merlin.targetgen.contract.toolchain import mlir_bin

    for name in ("llvm-mc", "llvm-objcopy", "llvm-nm"):
        if not mlir_bin(name).is_file():
            return name
    if not Path(clang()).exists():
        return "clang"
    return None


@pytest.fixture(scope="module")
def seam(tmp_path_factory):
    """One emitted seam, built once: the package run and the assembler are not free."""
    from merlin.compile_cli import _mesh_tile_binding
    from merlin.llvmlower.device_native import build_device_native_seam

    missing = _toolchain_missing()
    if missing:
        pytest.skip(f"no {missing} in this environment; the seam cannot be assembled or compiled")
    found = _emittable_target_with_a_package()
    if found is None:
        pytest.skip("this checkout has no backend package for any device_native target "
                    "(they are generated output); nothing to emit a seam from")
    target, package_dir = found

    # Extents and datapath DERIVED from the target: its own mesh tile is the shape its package emits a
    # kernel for, and its own compute unit declares the formats.
    binding = _mesh_tile_binding(target, None, None)
    d = int(binding.tile_dim)
    work = tmp_path_factory.mktemp("seam")
    built = build_device_native_seam(
        target, {"layer0": (d, d, d)}, package_dir=package_dir, workdir=work,
        operand_dtype=binding.operand_dtype, accum_dtype=binding.accum_dtype,
        codegen_target="x86", timeout=900)
    if not built.programs:
        pytest.skip(f"{target}'s package emitted no seam for a {d}x{d}x{d} tile: {built.skipped}")
    return built, target, binding, work


# ------------------------------------------------------------------------------------------------
# the artifact
# ------------------------------------------------------------------------------------------------

def test_a_complete_seam_is_emitted(seam):
    """Both halves and the contract, in one build. A device image with no stager is not a seam."""
    built, _target, _binding, _work = seam
    assert built.ok, f"seam incomplete: {built.skipped}"
    assert not built.skipped, f"a complete seam should skip nothing: {built.skipped}"
    assert built.programs, "no device program"
    assert built.host_object is not None and built.host_object.is_file()
    assert built.host_source is not None and built.host_source.is_file()
    assert built.contract_path is not None and built.contract_path.is_file()
    assert built.window_base is not None


def test_the_device_image_is_the_assembled_directives_not_their_text(seam):
    """The image is what ``llvm-mc`` made of the package's own words, byte for byte.

    This is the assertion that separates machine code from a string that resembles it. Merlin holds
    no opcode table, so the words are read back out of the artifact the package emitted rather than
    compared against anything written here.
    """
    built, _target, _binding, work = seam
    prog = built.programs[0]
    art = next(iter(sorted(work.glob("*.device.S"))), None)
    assert art is not None, "the package's emitted device artifact was not kept"

    words: list[int] = []
    only_words = True
    for raw in art.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()          # structural: strip the comment, no regex
        if not line or line.endswith(":") or not line.startswith("."):
            continue
        directive, _, rest = line.partition(" ")
        if directive == ".word":
            words.append(int(rest.strip(), 0))
        elif directive not in (".text", ".globl", ".global"):
            only_words = False
    assert words, "the emitted artifact carries no .word directives to assemble"
    assert prog.image, "the assembled image is empty"
    if only_words:
        assert prog.image == b"".join(struct.pack("<I", w) for w in words), (
            "the image is not the little-endian assembly of the package's own words")
    else:
        assert prog.image[:4] == struct.pack("<I", words[0])


def test_the_address_contract_is_window_relative_and_sized_from_the_format(seam):
    """Every slot inside the device's own derived window, sized from its declared format."""
    from merlin.targetgen import corpus_spec as CS

    built, target, _binding, _work = seam
    from merlin.system.derive import link_for
    from merlin.targetgen.target_experiment import load_capability_manifest
    link = link_for(target, getattr(load_capability_manifest(target), "endpoint_kind", None))
    assert built.window_base == int(link.device_dram_base), (
        "the window was not the one the device's own memory map declares")

    prog = built.programs[0]
    assert prog.tensors, "an address contract with no tensors is not a contract"
    assert prog.inputs, "nothing to stage: the host half would be inert"
    assert prog.outputs, "nothing to collect: the result would never leave the device"
    for t in prog.tensors:
        assert t.device_address >= built.window_base
        assert t.window_offset == t.device_address - built.window_base
        width = CS.dtype_info(t.dtype)[2]
        if width is not None:                        # sub-byte formats have no byte width
            n = 1
            for dim in t.shape:
                n *= dim
            assert t.nbytes == n * width, f"{t.name}: {t.nbytes} != {n}*{width}"
    ordered = sorted(prog.tensors, key=lambda t: t.window_offset)
    for prev, nxt in zip(ordered, ordered[1:]):
        assert prev.window_offset + prev.nbytes <= nxt.window_offset, (
            f"{prev.name} and {nxt.name} overlap; staging one would destroy the other")


def test_the_host_unit_carries_the_whole_image_and_the_same_offsets(seam):
    """The stager the host compiles describes the SAME layout as the contract beside it."""
    built, _target, _binding, _work = seam
    text = built.host_source.read_text(encoding="utf-8")
    prog = built.programs[0]
    assert f"{built.window_base:#x}" in text, "the derived window base is not recorded for the reader"
    assert text.count("0x") >= len(prog.image), "the image is not embedded in full"
    for t in prog.inputs:
        assert f"{{ {t.window_offset}u, {t.nbytes}u }}" in text, f"{t.name} slot missing"
    for t in prog.outputs:
        assert f"{{ {t.window_offset}u, {t.nbytes}u }}" in text, f"{t.name} slot missing"
    # No absolute host address may appear in generated code: the mapping is not derivable.
    for t in prog.tensors:
        assert f"{t.device_address}u" not in text, (
            f"{t.name}'s absolute address leaked into the generated code")


def test_the_host_object_defines_the_entries_and_leaves_only_libc_undefined(seam):
    """The emitted object links on its own: the launch is a parameter, not an extern to satisfy."""
    from merlin.targetgen.contract.toolchain import mlir_bin

    built, _target, _binding, _work = seam
    r = subprocess.run([str(mlir_bin("llvm-nm")), str(built.host_object)],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    defined, undefined = set(), set()
    for line in r.stdout.splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        kind, name = parts[-2], parts[-1]
        (undefined if kind.upper() == "U" else defined).add(name)
    prog = built.programs[0]
    for suffix in ("_stage", "_collect", "_dispatch", "_program_bytes", "_program_image"):
        assert prog.entry + suffix in defined, f"{prog.entry}{suffix} not defined by the host object"
    assert undefined <= {"memcpy"}, f"the seam object needs more than libc: {sorted(undefined)}"


def test_the_seam_links_and_the_stager_honours_its_own_contract(seam, tmp_path):
    """LINK it and RUN it: operands land at the declared offsets, results come back from theirs.

    The harness below is generated FROM the emitted contract -- no offset is written here -- so it
    fails if the stager and the contract ever describe different layouts. Its launch is a stub that
    stamps a marker where the program stores its result, which is the one step no fact in this repo
    derives (there is no control aperture for this transport) and which the emitted code therefore
    takes as a function pointer.
    """
    from merlin.llvmlower.toolchain import clang

    built, _target, _binding, _work = seam
    prog = built.programs[0]
    out = prog.outputs[0]
    ins = prog.inputs
    window_bytes = max(t.window_offset + t.nbytes for t in prog.tensors) + 64

    src = tmp_path / "harness.c"
    src.write_text(f"""
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef int (*launch_fn)(void *ctx, const unsigned char *program, uint64_t nbytes);
uint64_t {prog.entry}_program_bytes(void);
uint64_t {prog.entry}_input_count(void);
uint64_t {prog.entry}_output_count(void);
int {prog.entry}_dispatch(unsigned char *window, const void *const *operands, void *const *results,
                          launch_fn launch, void *ctx);
static unsigned char *g_window;
static uint64_t g_program_bytes;
static int stub_launch(void *ctx, const unsigned char *program, uint64_t nbytes) {{
  (void)ctx; (void)program; g_program_bytes = nbytes;
  memset(g_window + {out.window_offset}u, 0xAB, {out.nbytes}u);   /* "the device wrote its result" */
  return 0;
}}
int main(void) {{
  g_window = calloc({window_bytes}u, 1);
  {"".join(f"unsigned char *op{i} = malloc({t.nbytes}u); memset(op{i}, {0x11 + i}, {t.nbytes}u);"
           for i, t in enumerate(ins))}
  unsigned char *res0 = calloc({out.nbytes}u, 1);
  const void *ops[] = {{ {", ".join(f"op{i}" for i in range(len(ins)))} }};
  void *res[] = {{ res0 }};
  int rc = {prog.entry}_dispatch(g_window, ops, res, stub_launch, 0);
  int staged = 1;
  {"".join(f'if (g_window[{t.window_offset}u] != {0x11 + i} '
           f'|| g_window[{t.window_offset + t.nbytes - 1}u] != {0x11 + i}) staged = 0;'
           for i, t in enumerate(ins))}
  printf("rc=%d bytes=%llu in=%llu out=%llu staged=%d first=%02x last=%02x head=%02x\\n",
         rc, (unsigned long long)g_program_bytes,
         (unsigned long long){prog.entry}_input_count(),
         (unsigned long long){prog.entry}_output_count(), staged,
         res0[0], res0[{out.nbytes} - 1], g_window[0]);
  return 0;
}}
""", encoding="utf-8")

    exe = tmp_path / "harness"
    c = subprocess.run([str(clang()), "-O1", str(src), str(built.host_object), "-o", str(exe)],
                       capture_output=True, text=True, timeout=600)
    if c.returncode != 0:
        pytest.skip(f"no host link in this environment (clang cannot link a hosted binary): "
                    f"{c.stderr.strip()[-300:]}")
    r = subprocess.run([str(exe)], capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    fields = dict(tok.split("=", 1) for tok in r.stdout.split())
    assert fields["rc"] == "0", r.stdout
    assert int(fields["bytes"]) == len(prog.image), "the launch was handed a different program"
    assert int(fields["in"]) == len(ins) and int(fields["out"]) == len(prog.outputs)
    assert fields["staged"] == "1", "an operand did not land at its declared offset"
    assert fields["first"] == "ab" and fields["last"] == "ab", (
        "the result was not collected from the offset the contract declares")
    assert fields["head"] == "00", "the stager wrote outside the slots it declared"


def test_the_seam_bundles_into_an_archive_the_board_build_can_link(seam, tmp_path):
    built, _target, _binding, _work = seam
    a = built.archive(tmp_path / "libseam.a")
    if a is None:
        pytest.skip("no ar in this environment")
    assert a.is_file() and a.stat().st_size > 0


def test_a_device_without_a_derivable_window_is_refused(seam):
    """The predicate stays a predicate: it says yes here because facts said so, not by default.

    A refusal has to remain reachable, or ``seam_emittable`` is a constant dressed as a check.
    """
    from merlin.llvmlower.device_native import seam_emittable

    _built, target, _binding, _work = seam
    assert seam_emittable(target) is None
    refused = [t for t in _all_targets() if seam_emittable(t) is not None]
    assert refused, "no target is refused by seam_emittable; the predicate is vacuous"


def _all_targets() -> list[str]:
    from merlin.targetgen.target_registry import all_targets
    return all_targets()
