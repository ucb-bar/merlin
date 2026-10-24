"""Shared fixtures for the targetgen bucket.

``mm_build`` compiles a host driver against the REAL testchipip ``mm.h``/``mm.cc`` that a chipyard Verilator
build generated, together with every simulator memory model in ``merlin/contract/external/sim_memory/``. So
the models are tested against the interface they are linked into, not a copy of it. It skips -- never
fakes -- when no such build, no fesvr header or no host compiler is present.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


def chipyard_mm_dir() -> Path | None:
    """The gen-collateral directory of a chipyard Verilator build under ``MERLIN_CHIPYARD``."""
    from merlin.common.paths import env

    root = env("MERLIN_CHIPYARD")
    if not root:
        return None
    for mk in sorted(Path(root).glob("sims/verilator/generated-src/*/gen-collateral/mm.h")):
        if (mk.parent / "mm.cc").is_file():
            return mk.parent
    return None


def fesvr_include() -> Path | None:
    """fesvr's headers (``mm.h`` includes ``fesvr/memif.h``): beside the chipyard toolchain, else under
    ``$RISCV``, chipyard's own variable for that toolchain prefix -- a pinned tree built with an outside
    toolchain has none of its own."""
    from merlin.common.paths import env

    candidates = []
    if env("MERLIN_CHIPYARD"):
        candidates.append(Path(env("MERLIN_CHIPYARD")) / ".conda-env" / "riscv-tools" / "include")
    if env("RISCV"):
        candidates.append(Path(env("RISCV")) / "include")
    return next((c for c in candidates if (c / "fesvr" / "memif.h").is_file()), None)


class MmBuild:
    """``build(driver_source, out_dir, replace={name: path})`` -> the driver executable.

    Every model source is compiled in; ``replace`` swaps one, by file name, for another file (a mutant).
    """

    def __init__(self, cxx: str, mm_dir: Path, fesvr: Path):
        self.cxx, self.mm_dir, self.fesvr = cxx, mm_dir, fesvr

    def __call__(self, driver: str, out: Path, *, replace: dict[str, Path] | None = None) -> Path:
        from merlin.targetgen import mem_perturb as MP

        sources = {n: MP.model_sources_dir() / n for n in MP.MODEL_SOURCES}
        sources.update(replace or {})
        (out / "driver.cc").write_text(driver)
        exe = out / "driver"
        subprocess.run(
            [self.cxx, "-std=c++17", "-O1", f"-I{MP.model_sources_dir()}", f"-I{self.mm_dir}", f"-I{self.fesvr}",
             str(out / "driver.cc"), *map(str, sources.values()), str(self.mm_dir / "mm.cc"), "-o", str(exe)],
            check=True, capture_output=True, text=True,
        )  # fmt: skip
        return exe


@pytest.fixture(scope="session")
def mm_build() -> MmBuild:
    mm_dir = chipyard_mm_dir()
    cxx = shutil.which("g++") or shutil.which("c++")
    if mm_dir is None or cxx is None:
        pytest.skip("needs MERLIN_CHIPYARD with a generated testchipip mm.h/mm.cc and a host C++ compiler")
    fesvr = fesvr_include()
    if fesvr is None:
        pytest.skip("needs the fesvr headers beside the chipyard toolchain or under $RISCV")
    return MmBuild(cxx, mm_dir, fesvr)
