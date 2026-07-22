"""Bridge to the sibling ``mlc`` model-ladder compiler — the CIRCT+xDSL RTL frontend we reuse for
RTL *needle* extraction (finding specific ISA facts in the HW-dialect op graph).

``mlc`` is an EXTERNAL dependency resolved via ``.env MERLIN_MLC_DIR`` (it will be upstreamed to its own
open-source repo). Rather than re-implement the op-graph parser + decoder analysis, we reuse mlc's
``discover.irgraph`` (``circt-opt --mlir-print-op-generic`` -> xDSL ``HwGraph``) and ``discover.decode``
(the legal opcode set from the decoder's ``comb.icmp eq`` fan-out), plus mlc's own prebuilt CIRCT
binaries. Imports are function-local behind an availability guard (``chia_bridge`` style) so importing
this module never hard-requires mlc, and a machine without mlc degrades honestly rather than crashing.

Why this matters: our legacy ``circt_introspect.extract_funct_table`` parses the ISA *header*
(``GemminiISA.scala`` ``val NAME = N.U``), which is provably wrong vs the silicon — it lists funct codes
the decoder never matches and omits ones it does. The decoder-derived set here is the actual ISA the
hardware implements, which is exactly what a functionally-correct compiler must target.
"""
from __future__ import annotations

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


# RoCC funct7 is 7 bits — the width of the Gemmini command opcode field the decoder matches on.
_FUNCT_WIDTH = 7


def discover_legal_functs(hw_mlir_path: str | Path, *, funct_width: int = _FUNCT_WIDTH) -> dict:
    """Derive the legal RoCC funct set from the DECODER's ``comb.icmp eq`` fan-out in the HW dialect —
    the actual silicon, not the ISA header. Returns ``{legal_funct, width, fanout, module, method,
    evidence}`` (``legal_funct=None`` if no decode signal of that width is found)."""
    require_mlc()
    with _mlc_on_path():
        from mlc.discover import decode, irgraph
        graph = irgraph.load_hw_graph(Path(hw_mlir_path), circt_opt=circt_opt_bin())
        sig = decode.discover_opcode_set(graph, expected_width=funct_width)
    if sig is None:
        return {"legal_funct": None, "width": funct_width, "fanout": 0, "module": None,
                "method": "decoder_icmp_fanout(mlc)",
                "evidence": f"no width-{funct_width} decode signal in {Path(hw_mlir_path).name}"}
    legal = sorted(int(v) for v in sig.values)
    return {
        "legal_funct": legal, "width": sig.width, "fanout": sig.fanout, "module": sig.module,
        "method": "decoder_icmp_fanout(mlc)",
        "evidence": f"union of comb.icmp-eq {sig.width}-bit decode signals in module {sig.module} "
                    f"({sig.fanout} comparisons) -> {len(legal)} legal functs",
    }
