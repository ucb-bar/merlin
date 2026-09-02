#!/usr/bin/env python
"""Harvest per-family XNNPACK RVV ukernel objdump fixtures for the beam's per-op teacher.

The per-op teacher (``merlin.mining.wholemodel_proposer``) pairs a per-FAMILY expert CCA against our
per-family section CCA. The expert CCA is lifted from a REAL RVV disassembly of the family's XNNPACK
ukernel. GEMM fixtures pre-exist (``run_expert_gemm``); this script harvests the rest of the census
byte-traffic families that HAVE an XNNPACK primitive — transpose, reduce (rsum/rmax), the transcendental
activations (vgelu/vsigmoid), clamp, and elementwise binary — into the SAME ``merlin/tests/data/cca_asm/``
directory, named ``xnnpack_<family>_rvv.objdump``.

Method (all HOST cross-compile, no board): compile the family's ``*rvv*.c`` ukernel to a ``.o`` with the
SpacemiT K1 clang (``--target=riscv64-unknown-linux-gnu -march=rv64gcv_zfh_zvfh -mabi=lp64d -O3
-DNDEBUG``, ``-I ceiling_drivers`` for the ``src/xnnpack/*.h`` header shim + ``-I <XNNPACK>/src``),
disassemble with ``kernels/decode/objdump.disassemble_text`` (llvm-objdump, NOT GNU). A ukernel that
will not compile is SKIPPED and logged honestly, never faked.

The family->fixture->ukernel registry is ``wholemodel_proposer.FAMILY_TEACHERS`` (single source of
truth, shared with the proposer). Run: ``.venv/bin/python build_tools/scripts/harvest_xnnpack_fixtures.py``.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

from merlin.common.paths import repo_root, work_dir
from merlin.kernels.decode.objdump import disassemble_text
from merlin.mining import k1
from merlin.mining.wholemodel_proposer import FAMILY_TEACHERS, FamilyTeacher, cca_asm_dir

_CFLAGS = ["--target=riscv64-unknown-linux-gnu", "-march=rv64gcv_zfh_zvfh", "-mabi=lp64d",
           "-O3", "-DNDEBUG"]


def _ceiling_inc() -> Path:
    """The ceiling_drivers dir supplying the minimal ``src/xnnpack/*.h`` header shim."""
    return repo_root() / "merlin" / "python" / "merlin" / "kernels" / "ceiling_drivers"


def _xnnpack_src() -> Path:
    import os
    env = os.environ.get("MERLIN_XNNPACK_REPO")
    root = Path(env) if env else work_dir() / "tmp" / "kernels" / "XNNPACK"
    return root / "src"


def _harvestable() -> list[FamilyTeacher]:
    """The teacher entries this script harvests: a ukernel source + a fixture basename, deduped by
    fixture (so ``silu``/``sub`` which reuse ``sigmoid``/``add`` fixtures are covered once)."""
    seen: set[str] = set()
    out: list[FamilyTeacher] = []
    from merlin.mining.wholemodel_proposer import dtype_fixture_teachers
    # the dtype-matched GEMM fixtures too: they are not census-family entries, but they must be
    # re-harvestable or they stay stuck as the unlinked versions that teach nothing loop-scoped.
    for t in list(FAMILY_TEACHERS.values()) + dtype_fixture_teachers():
        if t.ukernel_src is None or t.fixture is None or t.fixture in seen:
            continue
        seen.add(t.fixture)
        out.append(t)
    return out


def harvest_one(t: FamilyTeacher, cc: Path, xsrc: Path, ceil_inc: Path, out_dir: Path,
                *, timeout: int = 300) -> tuple[bool, str]:
    """Compile + disassemble one ukernel into its fixture. Returns (ok, message)."""
    src = xsrc / t.ukernel_src
    if not src.is_file():
        return False, f"ukernel source not found: {src}"
    with tempfile.TemporaryDirectory(prefix="merlin_harvest_") as tmp:
        obj = Path(tmp) / (t.fixture.replace(".objdump", "") + ".o")
        cmd = [str(cc), *_CFLAGS, "-I", str(ceil_inc), "-I", str(xsrc), "-c", str(src),
               "-o", str(obj)]
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except (subprocess.TimeoutExpired, OSError) as e:
            return False, f"compile exec failed: {e}"
        if p.returncode != 0 or not obj.is_file():
            return False, f"compile failed (rc={p.returncode}): {p.stderr.strip()[-400:]}"
        # LINK before disassembling. In an unlinked .o every branch displacement is still a zero
        # placeholder awaiting relocation, so llvm-objdump resolves each branch to ITS OWN address --
        # `bne a0, t5, 0x5c <sym+0x5c>` at 0x5c. No target is ever < addr, `loop_spans()` reads EMPTY,
        # and every LOOP-SCOPED facet the expert should teach (register_block, accumulator_resident,
        # nr_is_vsetvlmax, calls_in_loop, and the whole memory facet) lifts as None.
        #
        # That is not a small loss: those facets ARE the register-blocking lesson. MEASURED on the
        # pre-existing qd8 fixture, which was harvested unlinked: 48 instructions, 0 loop spans,
        # spans_reliable=False, and a plainly visible K loop around `vwmacc.vx` that the lifter could
        # not see. `compare` then skipped those axes in silence, so choosing the (correct) dtype-matched
        # int8 expert quietly retired five axes instead of answering them.
        #
        # -shared -nostdlib is enough: intra-function branches only need the object laid out, and a
        # ukernel's undefined externals stay undefined in a shared object. If the link fails the
        # fixture is SKIPPED with the reason, never written unlinked.
        linked = Path(tmp) / (t.fixture.replace(".objdump", "") + ".so")
        lp = subprocess.run([str(cc), "--target=riscv64-unknown-linux-gnu", "-shared", "-nostdlib",
                             "-o", str(linked), str(obj)], capture_output=True, text=True,
                            timeout=timeout)
        target = linked if (lp.returncode == 0 and linked.is_file()) else None
        if target is None:
            return False, (f"link failed (rc={lp.returncode}), refusing to write an unlinked fixture "
                           f"whose loop structure the lifter cannot read: {lp.stderr.strip()[-300:]}")
        try:
            text = disassemble_text(target)
        except Exception as e:  # noqa: BLE001
            return False, f"disassemble failed: {e}"
    if "vsetvli" not in text and "vle" not in text and "vfmacc" not in text and "vadd" not in text:
        return False, "disassembly has no vector ops (not an RVV ukernel object?)"
    # A fixture whose loop structure is unreadable is a SILENTLY DEGRADED expert: it lifts every
    # loop-scoped facet as None, and `cca_compare` then skips those axes without saying so. Refuse it
    # rather than write one -- the decoder already knows how to tell, so use it.
    from merlin.kernels.decode import rvv as _rvv
    _stream = _rvv.decode_text(text)
    if not _stream.spans_reliable():
        return False, ("loop spans unreadable (branch displacements look unrelocated) -- this fixture "
                       "would teach nothing about register blocking, accumulator residency or memory, "
                       "and would do it silently. Refusing to write it.")
    (out_dir / t.fixture).write_text(_machine_independent(text, t.fixture), encoding="utf-8")
    return True, f"wrote {t.fixture} ({len(text)} chars) from {t.ukernel_src}"


def _machine_independent(text: str, fixture: str) -> str:
    """Replace objdump's leading ``<path>.o:\tfile format ...`` with the fixture name.

    These fixtures are TRACKED test data and the expert CCA is lifted from them, so they should be a
    function of the ukernel and nothing else. objdump names the object by its full path, which lives in
    a temp dir -- so re-running the harvester rewrote all six existing fixtures with the new TMPDIR
    (``/tmp/...`` -> ``/scratch/.../tmp/...``) and zero instruction changes. That churn is pure noise,
    and it invites exactly the wrong reading: a diff on an expert fixture should mean the SEARCH TARGET
    moved. The line is not parsed by the lifter (``rvv.decode_text`` keys on the ``<sym>`` section
    headers), so normalising it costs nothing.

    Gate on the ``:\tfile format `` separator ALONE, never on the object's extension. This function
    also required ``head.endswith(".o")``, and adding the link step (``.o`` -> ``.so``) silently turned
    the whole normalisation off: every fixture churned its temp path again on the next harvest with zero
    instruction change, which is precisely the misreading the normalisation exists to prevent. A guard
    keyed on a filename detail owned by another part of the pipeline stops firing when that detail
    moves; the separator is what objdump actually emits.
    """
    out = []
    for line in text.splitlines():
        head, sep, rest = line.partition(":\tfile format ")
        out.append(f"{fixture}{sep}{rest}" if sep else line)
    return "\n".join(out) + ("\n" if text.endswith("\n") else "")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Harvest per-family XNNPACK RVV objdump fixtures.")
    ap.add_argument("--out-dir", default=None, help="fixture dir (default: merlin/tests/data/cca_asm)")
    ap.add_argument("--only", default=None, help="comma-separated fixture basenames to harvest")
    args = ap.parse_args(argv)

    cc = k1.toolchain_cc()
    if cc is None:
        print("SKIP-ALL: SpacemiT K1 clang not found (k1.toolchain_cc() is None); "
              "set MERLIN_K1_TOOLCHAIN. No fixtures written.", file=sys.stderr)
        return 1
    xsrc = _xnnpack_src()
    if not xsrc.is_dir():
        print(f"SKIP-ALL: XNNPACK src not found at {xsrc} (set MERLIN_XNNPACK_REPO).", file=sys.stderr)
        return 1
    ceil_inc = _ceiling_inc()
    out_dir = Path(args.out_dir) if args.out_dir else cca_asm_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    only = {s.strip() for s in args.only.split(",")} if args.only else None

    ok = skipped = 0
    for t in _harvestable():
        if only is not None and t.fixture not in only:
            continue
        good, msg = harvest_one(t, cc, xsrc, ceil_inc, out_dir)
        print(f"[{'OK ' if good else 'SKIP'}] {t.census_family:12s} {msg}")
        ok += int(good)
        skipped += int(not good)
    print(f"\nharvested {ok} fixture(s), skipped {skipped} into {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
