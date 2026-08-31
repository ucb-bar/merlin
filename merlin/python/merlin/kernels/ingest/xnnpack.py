"""Ingest XNNPACK microkernels, attributing each to the endpoint it actually drives.

XNNPACK ships generated C under ``src/<op>/gen/*.c``. The optimization-relevant facts are encoded in
the in-file symbol ``xnn_<dtype>_<op>_<variant>_ukernel_<MRxNR>__<isa>`` and, redundantly, in the
filename. We parse the symbol first (most regular) and fall back to the filename. No file is executed;
this is pure text parsing, so it scales to the full corpus.

⚠️ **The ISA suffix is not the target.** Measured: a shipped matrix-extension kernel is named
``..._ukernel_16x4v__rvv`` and its body issues the outer-product unit's own instructions. Attributing
by suffix ingests it as a plain vector kernel and makes every matrix-unit decision in it INVISIBLE —
the corpus would then teach the mining loop that the expert used lanes where the expert used an array.

So the suffix names the base ISA a kernel is compiled for, and the ENDPOINT it drives is established
separately, by looking in the source for the macros that endpoint's own header defines
(:func:`endpoint_markers`, derived from the declared compute endpoints). A kernel carrying those
markers is attributed to that endpoint's target and records why.

The symbol grammar is a real grammar, so it is *tokenized* here (:func:`find_ukernel_symbol`) rather
than pattern-matched: the identifier run is walked and each field validated. A file whose symbol does
not fit falls back to the filename tokens, and a shape token that fits neither form is preserved
verbatim as ``{"tile": <token>}`` — never dropped.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

from merlin.kernels.types import NormalizedKernel, normalize_dtype

# Ordered longest-first so e.g. "igemm"/"dwconv" win over "gemm"/"conv".
_OP_KEYWORDS = (
    "igemm", "gemm", "dwconv2d", "dwconv", "conv",
    "argmaxpool", "maxpool", "avgpool", "vmulcaddc",
    "rdsum", "rsum", "rdminmax", "rminmax",
    "vadd", "vmul", "vdiv", "vsub", "vbinary", "vclamp", "velu",
    "vsigmoid", "vtanh", "vgelu", "vsqrt", "vrsqrt", "vexp", "vcvt",
    "ibilinear", "transpose",
)

#: The base ISA suffix to ingest. A PARAMETER, not a target: see the module docstring.
DEFAULT_ISA = "rvv"


#: The ukernel symbol grammar: ``xnn_<core>_ukernel_<shape>__<isa>``. ``core`` is lowercase
#: alphanumerics plus ``_``; ``shape`` is lowercase alphanumerics with NO separator.
_SYM_PREFIX, _UKERNEL_INFIX, _ISA_SEP = "xnn_", "_ukernel_", "__"
_CORE_ALPHABET = frozenset("abcdefghijklmnopqrstuvwxyz0123456789_")
_SHAPE_ALPHABET = frozenset("abcdefghijklmnopqrstuvwxyz0123456789")


def find_ukernel_symbol(text: str, isa: str) -> tuple[str, str] | None:
    """``(core, shape_token)`` for the first ukernel symbol of base ISA ``isa``, else ``None``.

    Walks each ``xnn_`` occurrence left to right and reads the maximal identifier run of the symbol
    alphabet — so an uppercase or punctuation character ends the symbol, as it did before. Within one
    run the LAST ``_ukernel_`` wins (the old greedy ``core`` capture), and the ISA is a PREFIX test of
    what follows ``__``: ``…__rvvfp16arith`` is ingested under base ISA ``rvv``, which is how the
    fp16 kernels reach the corpus at all.
    """
    at = text.find(_SYM_PREFIX)
    while at != -1:
        end = at + len(_SYM_PREFIX)
        while end < len(text) and text[end] in _CORE_ALPHABET:
            end += 1
        found = _split_ukernel_run(text[at + len(_SYM_PREFIX):end], isa)
        if found is not None:
            return found
        at = text.find(_SYM_PREFIX, at + 1)
    return None


def _split_ukernel_run(run: str, isa: str) -> tuple[str, str] | None:
    """Split one ``<core>_ukernel_<shape>__<isa>…`` run; last ``_ukernel_`` first (greedy core)."""
    at = run.rfind(_UKERNEL_INFIX)
    while at > 0:                        # > 0: `core` must be non-empty
        tail = run[at + len(_UKERNEL_INFIX):]
        sep = tail.find(_ISA_SEP)
        shape = tail[:sep] if sep != -1 else ""
        if (sep > 0 and tail[sep + len(_ISA_SEP):].startswith(isa)
                and all(ch in _SHAPE_ALPHABET for ch in shape)):
            return run[:at], shape
        at = run.rfind(_UKERNEL_INFIX, 0, at)
    return None


def endpoint_markers() -> dict[str, tuple[str, ...]]:
    """target -> the macro names a compute endpoint's own header defines.

    Derived from the declared endpoints rather than listed here, so an endpoint whose instruction set
    changes does not leave a stale marker list behind. A kernel whose SOURCE contains any of these is
    driving that endpoint whatever its ISA suffix says.
    """
    out: dict[str, list[str]] = {}
    try:
        from merlin.kernels import endpoints as _ep
        for name in _ep.endpoint_names():
            ep = _ep.load_endpoint(name)
            names = [n for names in ep.roles.values() for n in names]
            if ep.target and names:
                out.setdefault(ep.target, []).extend(names)
            # The RTL's vocabulary and the expert HEADER's differ -- the array calls it OPMACC and the
            # header's macro is VOPACC -- and it is the header spelling that appears in a kernel's
            # source. matrix_units.yaml already declares the correspondence for the crosscheck, so both
            # spellings are markers; using only the RTL names finds nothing in the very file this
            # attribution exists to catch.
            for rtl_name, header_name in (ep.crosscheck.get("pairs") or {}).items():
                out.setdefault(ep.target, []).extend([str(rtl_name), str(header_name)])
    except Exception:  # noqa: BLE001 — no endpoint data: attribute by suffix and say nothing more
        return {}
    return {t: tuple(sorted(set(v))) for t, v in out.items()}


def attribute_target(text: str, default_target: str,
                     markers: "dict[str, tuple[str, ...]] | None" = None) -> tuple[str, tuple[str, ...]]:
    """``(target, matched_markers)`` for one kernel source.

    Returns ``default_target`` and an empty tuple when nothing matches — the honest outcome for a plain
    vector kernel. When markers DO match, the kernel is attributed to that endpoint's target and the
    matched names are recorded, so the attribution can be checked rather than trusted.
    """
    for target, names in sorted((markers if markers is not None else endpoint_markers()).items()):
        if target == default_target:
            # A target's own endpoint markers cannot RE-attribute it to itself. Without this the base
            # ISA's own mnemonics (now that it has a declared endpoint too) match every kernel
            # compiled for it, and every ordinary kernel reports as "re-attributed" — noise that would
            # bury the one case this exists to catch.
            continue
        hit = tuple(n for n in names if n in text)
        if hit:
            return target, hit
    return default_target, ()
#: Shape-token suffixes that qualify a tile: ``v`` scalable, ``c`` channel, ``vc`` both. Longest
#: first, since the token continues past them (``4x8vc`` and ``4x8vcx`` both read as ``vc``).
_TILE_SUFFIXES = ("vc", "v", "c")


def _guess_op(text_lower: str) -> str:
    for kw in _OP_KEYWORDS:
        if kw in text_lower:
            return "gemm" if kw == "igemm" else kw  # treat igemm as gemm family
    return "other"


def _decimal_run(token: str, start: int) -> int:
    """End index of the decimal-digit run at ``start`` (``str.isdecimal()`` == the old ``\\d``)."""
    i = start
    while i < len(token) and token[i].isdecimal():
        i += 1
    return i


def _suffix_at(token: str, start: int, allowed: tuple[str, ...]) -> str:
    """The longest of ``allowed`` present at ``start``, or ``""``."""
    return next((s for s in allowed if token.startswith(s, start)), "")


def parse_shape_token(token: str) -> dict[str, object] | None:
    """``4x8vc`` -> MRxNR tile; ``9p2v`` -> kernel-points tile; ``None`` when it is neither.

    Both forms are read as a PREFIX of the token (as the old pattern's ``match`` was), so a token
    with trailing text still yields its leading tile. The MRxNR reading is tried first, exactly as
    the alternation was ordered, and ``vc``/``v``/``c`` are optional in both.
    """
    a = _decimal_run(token, 0)
    if a:
        if token.startswith("x", a):                       # <MR>x<NR>[vc|v|c]
            b = _decimal_run(token, a + 1)
            if b > a + 1:
                return {"MR": int(token[:a]),
                        "NR": f"{token[a + 1:b]}{_suffix_at(token, b, _TILE_SUFFIXES)}"}
        if token.startswith("p", a):                       # <points>p[<channels>][vc|v]
            b = _decimal_run(token, a + 1)
            # `c` alone is NOT a suffix of this form -- the old alternation spelled it `(vc?)?`.
            return {"kernel_points": int(token[:a]),
                    "channel_tile": f"{token[a + 1:b]}{_suffix_at(token, b, ('vc', 'v'))}"}
    return None


def _parse_shape(token: str) -> dict[str, object]:
    """Parse an MRxNR or NpKvc shape token into a small dict.

    A token fitting neither form is kept verbatim as ``{"tile": token}`` rather than discarded — the
    shape is then visibly unparsed instead of silently absent.
    """
    return parse_shape_token(token) or {"tile": token}


def _record_from_file(path: Path, repo: Path, target: str,
                      markers: "dict[str, tuple[str, ...]] | None" = None) -> NormalizedKernel:
    text = path.read_text(encoding="utf-8", errors="replace")
    sym = find_ukernel_symbol(text, target)
    if sym:
        core, shape_tok = sym
        dtype = normalize_dtype(core.split("_", 1)[0])
        op = _guess_op(core)
        shape = _parse_shape(shape_tok)
    else:  # fall back to filename tokens
        stem = path.stem  # e.g. f32-gemm-1x4v-minmax-rvv
        toks = stem.split("-")
        dtype = normalize_dtype(toks[0] if toks else "")
        op = _guess_op(stem.replace("-", "_"))
        shape_tok = next((t for t in toks
                          if parse_shape_token(t) and any(c.isdigit() for c in t)), "")
        shape = _parse_shape(shape_tok) if shape_tok else {}
    try:
        rel = str(path.relative_to(repo))
    except ValueError:
        rel = str(path)
    # The ISA suffix says which base ISA this was compiled for; the SOURCE says which endpoint it
    # drives. A shipped matrix-extension kernel carries the vector suffix and issues the array's own
    # instructions, so attributing by suffix files it as a lane kernel and hides every array decision
    # in it.
    attributed, hit = attribute_target(text, target, markers)
    meta = {"isa_suffix": target}
    if hit:
        meta["endpoint_markers"] = list(hit)
        meta["reattributed_from"] = target
    return NormalizedKernel(
        source="xnnpack", target=attributed, path=rel, op=op, dtype=dtype,
        shape=shape, raw_text=text, meta=meta,
    )


def ingest_xnnpack(repo: str, target: str = "rvv", limit: int | None = None) -> Iterator[NormalizedKernel]:
    """Yield NormalizedKernels for XNNPACK RVV microkernels under ``repo``.

    Globs ``src/*/gen/*<target>*.c`` (matches both ``rvv`` and ``rvvfp16arith``).
    """
    root = Path(repo)
    pattern = f"src/*/gen/*{target}*.c"
    count = 0
    markers = endpoint_markers()          # derived once; the per-file check is a substring scan
    for path in sorted(root.glob(pattern)):
        yield _record_from_file(path, root, target, markers)
        count += 1
        if limit is not None and count >= limit:
            return
