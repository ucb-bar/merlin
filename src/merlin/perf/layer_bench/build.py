"""Build a layer program with the target's own bare-metal recipe, and refuse images too big to load.

The recipe (compiler, flags, include roots, support sources, link template, load address) comes from
the target's registered backend (``runtime.backends.base.harness_build_recipe``), exactly as the
generic contract-compile path obtains it. The build is two-phase (each source compiled to a named
object, then linked) for the same reason ``targetgen.contract.compile.link_elf`` gives: a one-step
build embeds random temp-object names and makes identical programs hash differently.

The image guard exists because of a measurement, not a preference: the RTL engines' loader moves
every loaded byte (file bytes AND the zero-fill of NOBITS/.bss) over a slow serial link, measured at
under ~0.47 B/cycle. A 4 MB static array keeps a program from starting within 9M cycles. So the
guard counts p_memsz of every PT_LOAD segment, which is what the loader moves.
"""

from __future__ import annotations

import hashlib
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

#: Default ceiling on loaded bytes. At the measured loader rate this keeps load under ~0.6M cycles.
DEFAULT_MAX_LOADED_BYTES = 256 * 1024

_PT_LOAD = 1


class BuildError(RuntimeError):
    pass


@dataclass(frozen=True)
class BuiltProgram:
    elf: Path
    elf_sha256: str
    loaded_bytes: int


def loaded_bytes(elf: Path) -> int:
    """Sum of ``p_memsz`` over PT_LOAD segments of an ELF64 little-endian file."""
    data = elf.read_bytes()
    if data[:4] != b"\x7fELF" or data[4] != 2 or data[5] != 1:
        raise BuildError(f"{elf} is not an ELF64 little-endian file")
    (e_phoff,) = struct.unpack_from("<Q", data, 0x20)
    e_phentsize, e_phnum = struct.unpack_from("<HH", data, 0x36)
    total = 0
    for i in range(e_phnum):
        off = e_phoff + i * e_phentsize
        (p_type,) = struct.unpack_from("<I", data, off)
        if p_type == _PT_LOAD:
            (p_memsz,) = struct.unpack_from("<Q", data, off + 0x28)
            total += p_memsz
    return total


def build_program(
    sources: Sequence[Path],
    workdir: Path,
    *,
    target: str,
    extra_cflags: Sequence[str] = (),
    max_loaded_bytes: int | None = DEFAULT_MAX_LOADED_BYTES,
    elf_name: str = "layer.elf",
    support_first: bool = False,
) -> BuiltProgram:
    """Compile ``sources`` plus the recipe's support sources and link one ELF.

    ``max_loaded_bytes=None`` disables the guard; do that only when the run will use the target's load
    backdoor (``run_on_gsim(..., backdoor=True)`` on a target declaring one), where image size costs no
    simulated cycles.

    ``support_first`` links the recipe's support objects (crt, syscalls) AHEAD of ``sources``. Needed
    when a source is a large straight-line kernel object: with the default order it lands between the
    crt's ``_start`` and ``_init``, and their ``jal`` (reach +-1 MiB) no longer fits. The default keeps
    the order existing receipts were built with."""
    from merlin.runtime.backends import base as backends
    from merlin.targetgen.runtime_build import derived_link_script

    recipe = backends.harness_build_recipe(target)
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    link_ld = derived_link_script(recipe.load_address, recipe.link_script, workdir)
    objects: list[Path] = []
    ordered = (
        [*recipe.support_sources, *map(Path, sources)]
        if support_first
        else [*map(Path, sources), *recipe.support_sources]
    )
    for source in ordered:
        source = Path(source)
        if source.suffix not in (".c", ".S", ".s"):
            objects.append(source)
            continue
        unit = workdir / f"{source.stem}.o"
        cmd = recipe.compile_command(source=source, output=unit)
        if extra_cflags:
            cmd = cmd[:1] + list(extra_cflags) + cmd[1:]
        # Compiled INSIDE the work directory so a program's `.incbin "<name>"` resolves to the blob
        # beside it by a relative name (an absolute path would make the source differ per directory).
        step = subprocess.run(cmd, capture_output=True, text=True, cwd=str(workdir))
        if step.returncode != 0:
            raise BuildError(f"compile of {source.name} failed:\n{step.stderr[-2000:]}")
        objects.append(unit)
    elf = workdir / elf_name
    link = subprocess.run(
        recipe.link_command(objects=objects, output=elf, link_script=link_ld), capture_output=True, text=True
    )
    if link.returncode != 0:
        raise BuildError(f"link failed:\n{link.stderr[-2000:]}")
    size = loaded_bytes(elf)
    if max_loaded_bytes is not None and size > max_loaded_bytes:
        raise BuildError(
            f"{elf.name} loads {size} bytes (> {max_loaded_bytes}); the RTL loader would "
            "spend most of the run transferring it. Move operands out of the image."
        )
    return BuiltProgram(elf=elf, elf_sha256=hashlib.sha256(elf.read_bytes()).hexdigest(), loaded_bytes=size)
