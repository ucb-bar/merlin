"""Operand and accumulator capacity of a target's matrix unit, and the tile that fits it.

Element widths come from the format registry (``_dtype_bits``); store sizes come from the target's RTL
facts (``_operand_store_bytes``, ``_accumulator_capacity_elems``); ``capacity_fit`` is the obligation a
mesh layer is checked against and ``_capacity_fit_tile`` the blocking that discharges it.
"""
from __future__ import annotations

from pathlib import Path


def _dtype_bits(tok: str | None) -> int:
    """Width in BITS of one stored element of an operand dtype token. Floats come from the quant-format
    registry's own layout; integer tokens (``i8``/``int8``/``i32``) from their trailing digit run.
    Defaults to 8 (int8-class) when the token says nothing.

    Not a digit scrape over the whole token: ``fp8_e4m3`` contains three digit runs, and concatenating
    them yields 843, which made an fp8 target's on-chip capacity read as 624 elements instead of 65536.
    """
    if not tok:
        return 8
    t = str(tok)
    try:
        from ..runtime.fp8_formats import storage_bits
        return storage_bits(t)
    except Exception:  # noqa: BLE001 — not a registered float format: try the integer spelling
        pass
    tail = ""
    for ch in reversed(t):                    # trailing digit run, structurally (no regex)
        if not ch.isdigit():
            break
        tail = ch + tail
    return int(tail) if tail else 8


def _dtype_bytes(tok: str | None) -> int:
    """Byte width of an operand dtype token, rounded up (a sub-byte format still occupies a byte when
    something must address it individually). Prefer :func:`_dtype_bits` where packing matters."""
    return max(1, (_dtype_bits(tok) + 7) // 8)


def _operand_store_bytes(target: str) -> int | None:
    """On-chip OPERAND-store capacity in bytes for ``target``, derived from that target's own RTL. Tries
    three sources in descending order of independence, and returns None (never a default) when none of
    them can answer.

    1. mlc's discovered memory MAP — fully structural: mlc names the representative operand bank and this
       sums its sibling banks' ``depth x row_bytes``. Nothing is declared anywhere.
    2. The DESCRIPTOR's ``rtl.operand_store`` bank-name prefix, summed over mlc's discovered memory list.
       mlc discovers every SRAM on some targets and still refuses to classify them ("no memory map
       discovered"), which left the ``capacity_fit`` obligation decidable on one target and undecidable
       on another. Here the descriptor supplies only the LABEL; the bytes are still read out of the RTL.
       Deliberately not a guess: picking the largest discovered memory would select the INSTRUCTION
       memory on a device whose IMEM (128 KiB) is larger than its operand register file (64 KiB).
    3. The extracted RTL facts' ``memories`` list, keyed by the role name merlin's own facts schema
       assigns ("scratchpad" is the schema's operand-store role, not a target's spelling).
    """
    from ..targetgen.rtl import mlc_bridge as _mb
    try:
        caps = _mb.discovered_capacities(target)
        if caps and caps.get("operand_bytes"):
            return int(caps["operand_bytes"])
    except Exception:  # noqa: BLE001 — mlc unavailable → try the declared group next
        pass
    try:
        from ..targetgen.corpora import experiment_for
        prefix = getattr(experiment_for(target), "operand_store", None)
        if prefix:
            banks = _mb.discovered_memories(target) or []
            total = sum(int(b["depth"]) * int(b["row_bytes"]) for b in banks
                        if str(b.get("name", "")).startswith(prefix))
            if total:
                return total
    except Exception:  # noqa: BLE001 — no descriptor / no discovery → fall through to facts
        pass
    try:
        from ..targetgen.rtl import facts as _facts
        mems = (_facts.load_facts(target).get("facts") or {}).get("memories") or []
    except Exception:  # noqa: BLE001 — no facts bundle → capacity unknown, caller falls back
        return None
    sp = next((m for m in mems if m.get("name") == "scratchpad"), None)
    nbytes = sp.get("bytes") if sp else None
    return int(nbytes) if nbytes else None


def _operand_store_capacity_elems(target: str, operand_dtype: str | None) -> int | None:
    """The operand store's capacity in ELEMENTS of ``operand_dtype`` — counted in bits, so a sub-byte
    packed format (fp4/fp6) is not silently rounded up to a byte apiece."""
    nbytes = _operand_store_bytes(target)
    return (int(nbytes) * 8) // max(1, _dtype_bits(operand_dtype)) if nbytes else None


def _accumulator_capacity_elems(target: str, accum_dtype: str | None) -> int | None:
    """The ACCUMULATOR's capacity in elements of ``accum_dtype``, or None when it cannot be derived.

    Separate from the operand store because it is a separate, much smaller SRAM, and it binds a
    different quantity: the operand store bounds the INPUT working set (weight tile + activation tile),
    the accumulator bounds the OUTPUT tile that the contraction accumulates into. A layer can sit
    comfortably inside a 256 KiB scratchpad and still overrun a 64 KiB accumulator, because the output
    grows as M*N while the operands grow as K*(M+N).

    Measured: the whole-model capsule routed four layers whose operands fit the scratchpad by a wide
    margin -- (345,32)@(32,256) and (96,64)@(64,512) -- so nothing tiled them, and every one aborted the
    simulator with ``vector::_M_range_check: __n (which is 1024) >= this->size() (which is 1024)``: the
    1024 accumulator rows (65536 B / (16 elems * 4 B)) addressed one past the end. The tiles that DID
    run all passed, so this cost four unmeasurable layers rather than four wrong ones -- the capsule
    failed for want of a measurement, not for a wrong answer.
    """
    from ..targetgen.rtl import mlc_bridge as _mb
    try:
        caps = _mb.discovered_capacities(target) or {}
    except Exception:  # noqa: BLE001 -- mlc unavailable => undecidable, never assumed
        return None
    nbytes = caps.get("accumulator_bytes")
    return (int(nbytes) * 8) // max(1, _dtype_bits(accum_dtype)) if nbytes else None


def declared_primitive_tile(package: str | Path | None) -> tuple[int, int, int] | None:
    """The (m, k, n) tile a backend DECLARES it implements, from its ``manifest.yaml``, or ``None``.

    Why this exists alongside the capacity derivation: they answer different questions and only one of
    them is always answerable. Capacity asks "how much fits on chip", which needs a classified operand
    store -- and on a device whose SRAMs cannot be classified there is nothing to compute, so the whole
    residency tiler becomes unreachable and an oversized layer is handed to a backend that cannot take
    it. The declared tile asks "what did you build", which the backend always knows.

    A backend that declares one is promising to lower EXACTLY that shape; the loop nest over larger
    extents is then the runtime's, and every result produced that way is attributed as such (see the
    ``discharged_by`` block in :func:`run_matmul_on_mesh`) -- never as evidence the backend generalizes.
    Declaring nothing is legal and keeps the previous behaviour.
    """
    if not package:
        return None
    try:
        import yaml as _yaml
        man = Path(package) / "manifest.yaml"
        if not man.is_file():
            return None
        t = ((_yaml.safe_load(man.read_text(encoding="utf-8")) or {}).get("primitive_tile") or {})
        m, k, n = int(t.get("m") or 0), int(t.get("k") or 0), int(t.get("n") or 0)
        return (m, k, n) if m > 0 and k > 0 and n > 0 else None
    except Exception:  # noqa: BLE001 -- an unreadable/absent declaration is simply no declaration
        return None


def _capacity_fit_tile(M: int, K: int, N: int, D: int, cap_elems: int,
                       acc_elems: int | None = None) -> tuple[int, int, int, int]:
    """Shrink a matmul (M,K,N) to the largest D-aligned tile that fits on chip, and return
    (mt,kt,nt,n_tiles) where n_tiles is how many such tiles cover the layer.

    TWO stores bound the tile, and they bind different dimensions:
      * the OPERAND store holds the weight tile K·N plus the activation tile M·K (``cap_elems``);
      * the ACCUMULATOR holds the OUTPUT tile M·N (``acc_elems``), when it can be derived.

    Modelling only the first is what let four layers through whose operands fit the scratchpad easily
    while their output overran the accumulator. Output residency is the constraint that can force the
    ROW dim to split: with nt already at the tile edge, shrinking K does nothing for M·N, so M is the
    only dimension left to give. That is why M is no longer unconditionally whole -- but only when an
    accumulator bound is known AND the output actually exceeds it, so a caller that passes no
    ``acc_elems`` gets exactly the previous extents. Pure arithmetic, target-agnostic."""
    def _half(x):
        return max(D, (x // 2 // D) * D or D)

    def _operands_fit(mt, kt, nt):
        return kt * nt + mt * kt <= cap_elems

    def _output_fits(mt, nt):
        return (not acc_elems) or mt * nt <= acc_elems

    mt, kt, nt = M, K, N
    while not (_operands_fit(mt, kt, nt) and _output_fits(mt, nt)):
        if not _output_fits(mt, nt):
            # the OUTPUT is what does not fit: only M or N relieve it (K does not appear in M·N)
            if nt >= mt and nt > D:
                nt = _half(nt)
            elif mt > D:
                mt = _half(mt)
            elif nt > D:
                nt = _half(nt)
            else:
                break                       # already a single tile edge: nothing left to give
        elif kt > D or nt > D:
            if nt >= kt and nt > D:
                nt = _half(nt)
            elif kt > D:
                kt = _half(kt)
            else:
                nt = _half(nt)
        else:
            break
    import math
    n_tiles = math.ceil(M / mt) * math.ceil(K / kt) * math.ceil(N / nt)
    return mt, kt, nt, n_tiles


def capacity_fit(target: str, m: int, k: int, n: int, operand_dtype: str | None,
                 tile_dim: int, accum_dtype: str | None = None) -> dict:
    """Evaluate the ``capacity_fit`` CONTRACT OBLIGATION for one contraction on ``target``.

    ``capacity_fit`` is not a heuristic of ours — it is a predicate the interface contract already
    names (``xdsl_dialects.contract.KNOWN_PREDICATES``) and already demands, alongside
    ``rhs_immutable``, on every ``resident_packed_tensor`` requirement. What was missing is anyone
    evaluating it. A backend that assumes unbounded on-chip storage therefore did not fail its
    contract; it segfaulted the simulator three layers away, and the result surfaced as a generic
    "the oracle returned nothing".

    Measured on the gemmini backend under grade: its lowering tiles the ITERATION space into DIM x DIM
    tiles correctly, but addresses every one of the ``kt*nt`` weight tiles as simultaneously resident
    (``b_spad = weight_base + (kk*nt + nn)*DIM``). At 512x512 that is 32*32*16 = 16384 scratchpad rows
    against a 16384-row scratchpad, and spike aborts with
    ``vector::_M_range_check: __n (which is 16384) >= this->size() (which is 16384)``. Compute tiling
    without a residency loop above it.

    WHAT ``required_elems`` MODELS, stated because it is a modelling choice and not a property of the
    layer: it is the working set of a lowering that keeps the contraction's OPERANDS RESIDENT -- the
    whole weight tile plus the activation tile at once. A backend that instead STREAMS, DMA-ing one
    mesh tile at a time from DRAM, has a resident set of a few tiles no matter how large the layer, and
    satisfies the obligation at every extent. Both kinds are under grade here: one target's generated
    backend addresses all kt*nt weight tiles as simultaneously resident, and the other's issues a DMA
    per 32x32 tile. So a ``holds: False`` predicts a decline only for the resident kind, which is why
    the backend is CHARGED only when the dispatch actually declines -- the predicate alone never
    convicts. Measured: an extent this predicate called too large for one target's 64 KiB register file
    ran unblocked on its cosim and returned output, because that backend streams.

    Returns ``{"holds", "required_elems", "capacity_elems", "obligation", "assumes"}``; ``holds`` is
    None when the target declares no capacity (unknown, never assumed true).
    """
    cap = _operand_store_capacity_elems(target, operand_dtype)
    need = int(k) * int(n) + int(m) * int(k)      # weight tile + activation tile, both resident
    # The OUTPUT must also be resident, in the accumulator -- a separate and much smaller store. It is
    # reported and conjoined separately because it fails on different layers: the output grows as M*N
    # while the operands grow as K*(M+N), so a wide-output layer overruns the accumulator while its
    # operands sit well inside the scratchpad. Undecidable (None) stays undecidable, never assumed true.
    acc = _accumulator_capacity_elems(target, accum_dtype)
    out_need = int(m) * int(n)
    op_holds = None if not cap else bool(need <= cap)
    acc_holds = None if not acc else bool(out_need <= acc)
    # Three-valued conjunction, fail-closed: a violated term convicts, but an UNEVALUATED one leaves the
    # obligation undecidable -- never "holds". Reporting True because the one term we could evaluate
    # passed would claim an obligation nobody checked.
    _terms = (op_holds, acc_holds)
    holds = (False if False in _terms else None if None in _terms else True)
    return {"obligation": "capacity_fit",
            "holds": holds,
            "required_elems": need, "capacity_elems": cap,
            "output_elems": out_need, "accumulator_capacity_elems": acc,
            "operands_hold": op_holds, "output_holds": acc_holds,
            "tile_dim": int(tile_dim or 0),
            "assumes": "operands resident (a streaming lowering needs far less and always fits) "
                       "and the output tile resident in the accumulator"}
