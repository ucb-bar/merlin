"""Bridge to the sibling ``mlc`` model-ladder compiler — the CIRCT+xDSL RTL frontend we reuse for
RTL *needle* extraction (finding specific ISA facts in the HW-dialect op graph).

``mlc`` is an EXTERNAL dependency resolved via ``.env MERLIN_MLC_DIR`` (it will be upstreamed to its own
open-source repo). Rather than re-implement the op-graph parser + decoder analysis, we reuse mlc's
``discover.irgraph`` (``circt-opt --mlir-print-op-generic`` -> xDSL ``HwGraph``) and ``discover.decode``
(the legal opcode set from the decoder's ``comb.icmp eq`` fan-out), plus mlc's own prebuilt CIRCT
binaries. Imports are function-local behind an availability guard (``chia_bridge`` style) so importing
this module never hard-requires mlc, and a machine without mlc degrades honestly rather than crashing.

Why this matters: an ISA *header* parse (hand table / ``val NAME = N.U``) is provably wrong vs the
silicon — it lists command codes the decoder never matches and omits ones it does. The decoder-derived
set here is the actual ISA the hardware implements, which is exactly what a functionally-correct compiler
must target.

TARGET-AGNOSTIC: every entry point here takes a ``target`` argument and holds no target name. mlc is
target-parameterized (``artifact_paths(target)`` / ``discovered_memory_map(target)`` /
``discover_opcode_set(graph)`` / per-target ``runs/circt-arc/<target>``), so the same code plugs any HW
RTL repo mlc knows (gemmini, atlas, otbn, muon, nvdla, rocket, ...).
"""
from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path

_DEFAULT_MLC = "/scratch2/agustin/mvp-lhwir/modeling"


def mlc_dir() -> Path | None:
    """The mlc package root from ``.env MERLIN_MLC_DIR`` (default the current sibling checkout), or None
    if it does not contain an ``mlc/`` package."""
    from ...common.paths import env
    d = env("MERLIN_MLC_DIR") or _DEFAULT_MLC
    p = Path(d).expanduser()
    return p if (p / "mlc").is_dir() else None


def circt_opt_bin() -> Path | None:
    """mlc's prebuilt ``circt-opt`` (used to lower CIRCT custom syntax to generic MLIR)."""
    d = mlc_dir()
    if d is None:
        return None
    b = d / "third_party" / "circt" / "build" / "bin" / "circt-opt"
    return b if b.exists() else None


@contextmanager
def _mlc_on_path():
    d = mlc_dir()
    added = d is not None and str(d) not in sys.path
    if added:
        sys.path.insert(0, str(d))
    try:
        yield
    finally:
        if added:
            try:
                sys.path.remove(str(d))
            except ValueError:
                pass


def mlc_available() -> tuple[bool, str]:
    """(ok, reason). ok iff MERLIN_MLC_DIR resolves, circt-opt is built, and mlc imports."""
    d = mlc_dir()
    if d is None:
        return False, "MERLIN_MLC_DIR unset/invalid (no mlc/ package) — set it in .env"
    if circt_opt_bin() is None:
        return False, f"circt-opt not built under {d}/third_party/circt/build/bin"
    try:
        with _mlc_on_path():
            import mlc.discover.irgraph  # noqa: F401
            import mlc.discover.decode  # noqa: F401
    except Exception as e:  # noqa: BLE001 — surface the import failure honestly
        return False, f"mlc import failed: {type(e).__name__}: {e}"
    return True, "ok"


def require_mlc() -> None:
    ok, why = mlc_available()
    if not ok:
        raise RuntimeError(f"mlc unavailable: {why}")


def matmul_reuse_prediction(M: int, N: int, K: int, *, dim: int, capacity_bytes: int,
                            elem_bytes: int = 1) -> dict | None:
    """M3 STATIC data-movement prediction for an M×K·K×N matmul — no arc, no sim, no cycles.

    Reuses mlc's pure-arithmetic Model-3 functions (``predict_dma_volume.matmul_refetch_factor`` +
    ``spills.matmul_footprint_rows``) so the reuse/spill math is the mlc source of truth, not a
    re-derivation here. Returns the resident operand footprint (rows), the on-chip capacity (rows), a
    fits flag, and the loop-nest refetch factor (=1 when the operands are fully resident; >1 when they
    spill and the outer loop re-streams). Returns None (fail-closed) when mlc is not importable — these
    functions need only the mlc package on ``sys.path``, NOT circt, so we gate on the import alone."""
    with _mlc_on_path():
        try:
            from mlc.passes.predict_dma_volume import matmul_refetch_factor, ReuseFacts
            from mlc.passes.spills import matmul_footprint_rows
        except Exception:  # noqa: BLE001 — mlc absent/unimportable ⇒ honest unavailable, never a guess
            return None
        # beat_bytes is irrelevant to the refetch/footprint math (it only scales byte→beat counts), so a
        # benign 1 keeps us from fabricating a DMA beat width we have not discovered for this target.
        facts = ReuseFacts(beat_bytes=1, dim=int(dim), capacity_bytes=int(capacity_bytes))
        refetch = int(matmul_refetch_factor(M, K, N, facts, in_bytes=elem_bytes))
        footprint_rows = int(matmul_footprint_rows(M, N, K, int(dim)))
    capacity_rows = int(capacity_bytes) // (int(dim) * max(elem_bytes, 1))
    return {"footprint_rows": footprint_rows, "capacity_rows": capacity_rows,
            "fits": footprint_rows <= capacity_rows, "refetch": refetch,
            "footprint_tiles": footprint_rows // int(dim)}


# ------------------------------------------------------------------- target-agnostic RTL extraction
# Everything below is parameterized by ``target`` and holds NO target name — mlc is target-parameterized
# (artifact_paths/discovered_memory_map/discover_opcode_set/per-target runs/circt-arc/<target>), so the
# same code plugs any HW RTL repo mlc knows (gemmini, atlas, otbn, muon, nvdla, rocket, ...).

def core_hw_mlir(target: str) -> Path | None:
    """The version-matched CORE HW dialect (the module carrying the command decoder) for ANY target,
    from mlc's per-target arc outputs (``runs/circt-arc/<target>/outputs``). Prefers ``*_core_hw.mlir``
    (the core parses cleanly; the SoC dialect carries unparseable sv.verbatim blobs)."""
    d = mlc_dir()
    if d is None:
        return None
    outs = d / "runs" / "circt-arc" / target / "outputs"
    cands = sorted(outs.glob("*_core_hw.mlir")) or sorted(outs.glob("*_hw.mlir"))
    return next((p for p in cands if p.exists() and ".generic." not in p.name), None)


def discover_legal_opcodes(target: str, *, opcode_width: int | None = None) -> dict:
    """Derive the legal command-opcode set the RTL DECODER matches, for ANY target — via mlc's
    ``comb.icmp eq`` fan-out over the target's core HW dialect. The ISA the silicon implements, not a
    hand table/header. Returns ``{legal_opcodes, width, fanout, module, hw_source, method, evidence}``
    (``legal_opcodes=None`` if no decode signal is found / mlc or the HW dialect is unavailable)."""
    require_mlc()
    hw = core_hw_mlir(target)
    if hw is None:
        return {"legal_opcodes": None, "width": opcode_width, "fanout": 0, "module": None,
                "hw_source": None, "method": "decoder_icmp_fanout(mlc)",
                "evidence": f"no core HW dialect for target {target!r} under mlc runs/circt-arc"}
    with _mlc_on_path():
        from mlc.discover import decode, irgraph
        graph = irgraph.load_hw_graph(hw, circt_opt=circt_opt_bin())
        sig = decode.discover_opcode_set(graph, expected_width=opcode_width)
    if sig is None:
        return {"legal_opcodes": None, "width": opcode_width, "fanout": 0, "module": None,
                "hw_source": str(hw), "method": "decoder_icmp_fanout(mlc)",
                "evidence": f"no decode signal in {hw.name}"}
    legal = sorted(int(v) for v in sig.values)
    return {"legal_opcodes": legal, "width": sig.width, "fanout": sig.fanout, "module": sig.module,
            "hw_source": str(hw), "method": "decoder_icmp_fanout(mlc)",
            "evidence": f"union of comb.icmp-eq {sig.width}-bit decode signals in module {sig.module} "
                        f"({sig.fanout} comparisons) -> {len(legal)} legal opcodes"}


def discovered_memory_map(target: str) -> dict | None:
    """The target's operand-scratchpad / accumulator bank map, DISCOVERED from the RTL by mlc (row
    widths, not hand paths). None if unavailable. Target-agnostic."""
    if mlc_dir() is None:
        return None
    try:
        with _mlc_on_path(), _mlc_cwd():
            _ensure_interface_cache(target)
            from mlc.discover.cache import discovered_memory_map as _mm
            return dict(_mm(target))
    except Exception:  # noqa: BLE001
        return None


def discovered_dim(target: str) -> int | None:
    """The target's systolic mesh DIM, DISCOVERED from the RTL by mlc (not a hand literal). None if the
    target has no mesh / is unavailable. Target-agnostic."""
    if mlc_dir() is None:
        return None
    try:
        with _mlc_on_path(), _mlc_cwd():
            _ensure_interface_cache(target)
            from mlc.discover.cache import discovered_dim as _dim
            return int(_dim(target))
    except Exception:  # noqa: BLE001
        return None


def discovered_memories(target: str) -> list[dict] | None:
    """The target's SRAM banks (name/depth/row_bytes), DISCOVERED from the RTL by mlc. Target-agnostic."""
    if mlc_dir() is None:
        return None
    try:
        with _mlc_on_path(), _mlc_cwd():
            _ensure_interface_cache(target)
            from mlc.discover.cache import load_interface
            return list(load_interface(target).get("memories", []))
    except Exception:  # noqa: BLE001
        return None


def discovered_capacities(target: str) -> dict | None:
    """Total operand-scratchpad + accumulator capacity (bytes) DERIVED from mlc discovery — sum the
    sibling banks of the representative operand/accumulator memory named in the discovered ``memory_map``
    (banks share the name with the bank index stripped). Target-agnostic: no hardcoded bank names.
    Returns ``{operand_bytes, accumulator_bytes, operand_depth, accumulator_depth}`` (values None when
    not derivable)."""
    mm = discovered_memory_map(target)
    mems = discovered_memories(target)
    if not mm or not mems:
        return None

    def _bank_prefix(rep_name: str | None) -> str | None:
        # sibling banks share the name with the bank-index segment ('<base>_<int>') truncated to
        # '<base>_' — structural split, no regex (e.g. 'spad/spad_mems_0/mem_ext' -> 'spad/spad_mems_').
        if not rep_name:
            return None
        segs = rep_name.split("/")
        for i, s in enumerate(segs):
            base, sep, num = s.rpartition("_")
            if sep and num.isdigit():
                return "/".join(segs[:i] + [base + "_"])
        return None

    def _group(rep_name: str | None):
        prefix = _bank_prefix(rep_name)
        if not prefix:
            return None, None
        banks = [x for x in mems if x.get("name", "").startswith(prefix)]
        if not banks:
            return None, None
        total = sum(b["depth"] * b["row_bytes"] for b in banks)
        depth = banks[0]["depth"] if len({b["depth"] for b in banks}) == 1 else None
        return total, depth

    op_bytes, op_depth = _group(mm.get("operand_mem"))
    acc_bytes, acc_depth = _group(mm.get("accum_mem"))
    return {"operand_bytes": op_bytes, "accumulator_bytes": acc_bytes,
            "operand_depth": op_depth, "accumulator_depth": acc_depth}


@contextmanager
def _mlc_cwd():
    """mlc resolves its ``runs/...`` arc artifacts by paths RELATIVE to its own root, so its cosim +
    discovery entry points run with CWD = the mlc dir."""
    d = mlc_dir()
    prev = os.getcwd()
    if d is not None:
        os.chdir(d)
    try:
        yield
    finally:
        os.chdir(prev)


def arc_available(target: str) -> bool:
    """True iff mlc has a prebuilt arc model (.so/.o) + state manifest for ``target`` (any target)."""
    if mlc_dir() is None:
        return False
    try:
        with _mlc_on_path(), _mlc_cwd():
            from mlc.runtime.backend import available
            return bool(available(target))
    except Exception:  # noqa: BLE001
        return False


def _ensure_interface_cache(target: str) -> None:
    from mlc.discover.cache import discovered_memory_map, dump_cache
    try:
        discovered_memory_map(target)
    except Exception:  # noqa: BLE001 — build the fingerprint-gated discovery cache on demand
        dump_cache(target)


def arc_core(target: str):
    """The target-AGNOSTIC ctypes arc model (mlc ``CosimCore``) for ``target``: poke/peek/step any signal
    by NAME. Works for every target mlc compiled from RTL — the compile-from-RTL oracle primitive. The
    high-level per-op driver (matmul-WS, SIMT, ...) is target-specific and provided by mlc; use
    :func:`arc_run_command_buffer` for the agnostic high-level path."""
    require_mlc()
    if not arc_available(target):
        raise RuntimeError(f"mlc arc model absent for target {target!r} (runs/circt-arc/{target})")
    with _mlc_on_path(), _mlc_cwd():
        _ensure_interface_cache(target)
        from mlc.backends.cosim_core import CosimCore
        from mlc.discover.fingerprint import artifact_paths
        p = artifact_paths(target)
        d = mlc_dir()
        return CosimCore(str((d / p["so"]).resolve()), str((d / p["man"]).resolve()))


def arc_run_command_buffer(cb: dict) -> dict:
    """Answer a merlin command buffer on the RTL-derived arc model — target-AGNOSTIC (mlc infers the
    target/config from the command buffer). Returns mlc's backend contract ``{outputs, metrics, correct,
    oracle, ...}`` where ``metrics`` carries the RTL's internal counts (cycles, bytes_moved,
    resident_hits, accumulator_commits, ...) — verilator-level state without verilator, for ANY target."""
    require_mlc()
    with _mlc_on_path(), _mlc_cwd():
        from mlc.runtime.backend import run_command_buffer
        return run_command_buffer(cb)
