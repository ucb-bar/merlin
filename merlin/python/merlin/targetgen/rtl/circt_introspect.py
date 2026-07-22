"""merlin-rtl circt-introspect (v2) — deterministic RTL fact extraction via the CIRCT HW dialect.

Where the grep-over-FIRRTL v1 (:mod:`introspect`) stops, this picks up: it parses the **CIRCT HW
dialect** lowered by ``firtool --ir-hw`` from the elaborated Gemmini SoC, plus the Gemmini ISA Chisel
source, to extract the facts the RTL-derived checks layer needs *from the hardware itself* — not from a
hand-curated contract:

  * accumulator memory depth / lane count / byte capacity / address width  (HW dialect ``@AccumulatorMem``
    port widths — v1 left this ``None``);
  * scratchpad + mesh                                                       (reused from v1 grep, which
    already reproduces the contract);
  * the legal RoCC **funct decode table** + custom opcode                   (``GemminiISA.scala`` funct
    block — the ground truth the hand-curated ``rocc_decode._FUNCT_CLASS`` is a subset of).

Every fact carries ``evidence`` (the exact RTL/source token it came from). The result is cached to
``facts.json`` keyed by the hashes of its inputs; re-extraction is a no-op cache hit on unchanged RTL.
``validate(...)`` cross-checks the facts against the hand-curated ``target_contract.yaml`` and
``rocc_decode`` — agreement is what lets the curated numbers be retired; a *disagreement is surfaced*
(e.g. an unconfirmed contract capacity) rather than silently reconciled.

Deterministic, no LLM: the hardware is the source of truth.

CLI::

    python -m merlin.targetgen.rtl.circt_introspect [--out facts.json] [--hw <gemmini_soc.hw.mlir>]
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from merlin.common.paths import repo_root
from typing import Any

from . import introspect as V1
from .facts import rtl_cache_dir, rtl_facts_path

_REPO = repo_root()  # .../merlin
# Introspect scratch (the 21 MB firtool hw.mlir input + intermediates) lives in the PURGEABLE cache,
# never inside merlin/. The committed facts.json is a promoted PIN of a run (see promote_facts).
_CACHE_DIR = rtl_cache_dir("gemmini")
DEFAULT_HW = _CACHE_DIR / "gemmini_soc.hw.mlir"      # firtool --ir-hw output (cached run product)
FACTS_PIN = rtl_facts_path("gemmini")                # committed certified facts pin (checks read this)
GEMMINI_ISA = Path(V1.DEFAULT_CHIPYARD) / "generators/gemmini/src/main/scala/gemmini/GemminiISA.scala"

GENERATOR_VERSION = "rtl-introspect-v2-circt-hw"
# RoCC custom-3 opcode + the Gemmini funct3 group, confirmed against rocc_decode's ABI block.
CUSTOM_OPCODE = 0x7B
FUNCT3 = 0x3


# --------------------------------------------------------------- HW-dialect accumulator extraction
def _int_width(typ: str) -> int | None:
    """Bit width of an ``iN`` HW/MLIR integer type token (``i9`` -> 9), else None."""
    typ = typ.strip()
    return int(typ[1:]) if typ.startswith("i") and typ[1:].isdigit() else None


def _paren_span(line: str, open_idx: int) -> int:
    """Index of the ``)`` that closes the ``(`` at ``open_idx`` (balanced), or -1."""
    depth = 0
    for j in range(open_idx, len(line)):
        if line[j] == "(":
            depth += 1
        elif line[j] == ")":
            depth -= 1
            if depth == 0:
                return j
    return -1


def _module_port_sig(hw_text: str, module: str) -> str | None:
    """The port-list text of ``hw.module ... @<module>( ... )`` — the ports between the balanced
    parens after the module name. None if the module is absent."""
    marker = f"@{module}("
    for line in hw_text.splitlines():
        if "hw.module" not in line or marker not in line:
            continue
        open_idx = line.find(marker) + len(marker) - 1   # the '(' itself
        close = _paren_span(line, open_idx)
        if close != -1:
            return line[open_idx + 1:close]
    return None


def extract_accumulator(hw_text: str) -> dict[str, Any] | None:
    """Accumulator capacity from the @AccumulatorMem HW-dialect port signature, scaled by bank count.

    Per bank: depth = 2**addr_width (the write address port); a row is ``lanes`` i32 words; the byte
    mask width confirms bytes/row. ``banks`` = the number of @AccumulatorMem instantiations (Gemmini's
    acc is multi-bank: acc_capacity is split across acc_banks). Total bytes = banks * per-bank.

    The signature is read *structurally* — the module's port list is enumerated and each ``name : type``
    port is matched by exact identity and its integer type width parsed — not pattern-matched, so a
    port rename or reorder cannot silently mis-derive a capacity."""
    sig = _module_port_sig(hw_text, "AccumulatorMem")
    if sig is None:
        return None
    addr_w = None
    lane_bits_seen: list[int] = []
    mask_bits = 0
    for decl in (d.strip() for d in sig.split(",")):        # ports have no nested parens in their types
        if " : " not in decl:
            continue
        lhs, typ = decl.rsplit(" : ", 1)
        name = lhs.split()[-1].lstrip("%")                  # drop the `in`/`out` dir + `%`
        if name == "io_write_bits_addr":
            addr_w = _int_width(typ)
        elif name.startswith("io_write_bits_data_") and name.endswith("_0"):
            w = _int_width(typ)
            if w is not None:
                lane_bits_seen.append(w)
        elif name.startswith("io_write_bits_mask_"):
            mask_bits += 1
    if addr_w is None:
        return None
    depth = 1 << addr_w
    lane_bits = lane_bits_seen[0] if lane_bits_seen else 32
    n_lanes = len(lane_bits_seen)
    row_bytes = mask_bits or (n_lanes * (lane_bits // 8))
    banks = sum(1 for ln in hw_text.splitlines()
                if "hw.instance" in ln and "@AccumulatorMem" in ln) or 1
    per_bank = depth * row_bytes
    return {
        "name": "accumulator",
        "banks": banks,
        "addr_width": addr_w,
        "depth": depth,
        "lanes": n_lanes,
        "lane_bits": lane_bits,
        "row_bytes": row_bytes,
        "bytes": banks * per_bank,
        "bytes_per_bank": per_bank,
        "elem_bits": lane_bits,
        "evidence": (f"{banks}x @AccumulatorMem instance; per-bank io_write_bits_addr:i{addr_w} "
                     f"(depth={depth}); {n_lanes}x io_write_bits_data_*_0:i{lane_bits}; "
                     f"{mask_bits} byte-mask bits -> {row_bytes} B/row; total={banks * per_bank} B"),
    }


# ------------------------------------------------------------------ Chisel funct decode-table extract
def extract_funct_table(isa_src: str) -> dict[str, Any]:
    """The legal RoCC funct7 set + names from the GemminiISA funct block.

    Parses ``val NAME = N.U`` (no bit-width annotation) from the ``// funct values`` block, stopping at
    the ``CONFIG_EX`` line which begins the rs1-subfield group (those reuse small numbers and are NOT
    funct codes). The result is the authoritative legal-funct set; ``rocc_decode._FUNCT_CLASS`` is a
    narrower classifier subset of it."""
    lines = isa_src.splitlines()
    start = next((i for i, ln in enumerate(lines) if "// funct values" in ln), 0)
    table: dict[int, str] = {}
    # Structural parse of ``val NAME = <n>.U`` (no bit-width annotation) — split the line, not a
    # regex. Reject ``<n>.U(<w>.W)`` (bit-width-annotated) by requiring only whitespace/comment after
    # ``.U``, matching the funct block (the annotated rs1-subfields begin at CONFIG_EX and are skipped).
    for ln in lines[start + 1:]:
        if "CONFIG_EX" in ln:                 # start of the rs1-subfield block -> stop
            break
        s = ln.strip()
        if not s.startswith("val "):
            continue
        lhs, sep, rhs = s.partition("=")
        if not sep or ".U" not in rhs:
            continue
        name = lhs[4:].strip()
        num, after = rhs.strip().split(".U", 1)
        num = num.strip()
        if not (num.isdigit() and name and all(c.isupper() or c.isdigit() or c == "_" for c in name)):
            continue
        if after and not (after[:1].isspace() or after.startswith("//")):
            continue                          # e.g. ``.U(3.W)`` -> not a funct code
        table.setdefault(int(num), name)
    legal = sorted(table)
    return {
        "name": "funct_decode_table",
        "custom_opcode": CUSTOM_OPCODE,
        "funct3": FUNCT3,
        "legal_funct": legal,
        "names": {str(k): table[k] for k in legal},
        "evidence": f"GemminiISA.scala // funct values block: {len(legal)} codes "
                    f"[{legal[0]}..{legal[-1]}] up to CONFIG_EX",
    }


# ------------------------------------------------------- decoder-derived funct set (the true ISA)
def extract_funct_table_via_decoder(hw_path: Path) -> dict[str, Any] | None:
    """The legal RoCC funct set derived from the HW-dialect DECODER (mlc's comb.icmp-eq fan-out) — the
    actual ISA the silicon implements. Returns a funct-table dict (same shape as
    :func:`extract_funct_table`, ``method='decoder_icmp_fanout'``), or None if mlc is unavailable or the
    HW dialect cannot be parsed (e.g. a firtool/circt-opt version skew) — an honest fallback, never a
    fake pass."""
    try:
        from . import mlc_bridge
        ok, why = mlc_bridge.mlc_available()
        if not ok:
            return None
        res = mlc_bridge.discover_legal_functs(hw_path)
    except Exception:  # noqa: BLE001 — a parse/version skew means "no decoder facts", fall back cleanly
        return None
    legal = res.get("legal_funct")
    if not legal:
        return None
    return {
        "name": "funct_decode_table",
        "custom_opcode": CUSTOM_OPCODE,
        "funct3": FUNCT3,
        "legal_funct": legal,
        "names": {},  # the decoder yields numeric codes; names are cross-referenced from the header below
        "method": res.get("method", "decoder_icmp_fanout(mlc)"),
        "evidence": res.get("evidence", "mlc decoder comb.icmp-eq fan-out"),
    }


def _reconcile_funct(decoder: dict[str, Any] | None, header: dict[str, Any] | None) -> dict[str, Any] | None:
    """Choose the authoritative funct table. The decoder-derived set wins (it is the silicon); the
    header parse is the fallback. When BOTH are present, record the discrepancy (functs the header
    claims but the silicon never decodes, and vice-versa) + borrow the header's names for the codes the
    silicon actually decodes — so the pin is both correct AND named."""
    if decoder is None:
        if header is not None:
            header.setdefault("method", "scala_header_parse")
        return header
    if header is not None:
        hs, ds = set(header.get("legal_funct", [])), set(decoder["legal_funct"])
        hnames = header.get("names", {})
        decoder["names"] = {str(k): hnames.get(str(k), "?") for k in decoder["legal_funct"]}
        decoder["header_only_functs"] = sorted(hs - ds)   # header claims, silicon never decodes (phantom)
        decoder["decoder_only_functs"] = sorted(ds - hs)  # silicon decodes, header omits (missing)
        decoder["evidence"] += (f"; vs header: phantom={sorted(hs - ds)} missing={sorted(ds - hs)}")
    return decoder


# ---------------------------------------------------------------------------------- assemble + cache
def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16] if path.is_file() else "missing"


def build_facts(hw_path: Path = DEFAULT_HW, isa_path: Path = GEMMINI_ISA,
                chipyard_root: str | Path = V1.DEFAULT_CHIPYARD) -> dict[str, Any]:
    """Combine v1 grep facts (mesh + scratchpad) with v2 CIRCT/Chisel facts (accumulator + funct
    table). Returns a provenance-stamped record (does not write; see :func:`dump_facts`)."""
    arts = V1.find_artifacts(chipyard_root)
    v1 = V1.extract_facts(arts["fir"], arts["hierarchy"])

    hw_text = hw_path.read_text(errors="replace") if hw_path.is_file() else ""
    acc = extract_accumulator(hw_text) if hw_text else None
    # replace v1's size-less accumulator stub with the HW-extracted one
    mems = [m for m in v1.get("memories", []) if m.get("name") != "accumulator"]
    if acc:
        mems.append(acc)
    v1["memories"] = mems

    # Funct decode table: PREFER the decoder-derived legal set (the actual ISA the silicon implements,
    # via mlc's comb.icmp-eq fan-out analysis) over the header parse (GemminiISA.scala), which is
    # provably wrong (lists functs the decoder never matches, omits ones it does). Fall back to the
    # header parse when mlc / a version-matched HW dialect is unavailable — recording which method was
    # used and, when both are available, the header-vs-decoder discrepancy as evidence.
    header_funct = extract_funct_table(isa_path.read_text(errors="replace")) if isa_path.is_file() else None
    decoder_funct = extract_funct_table_via_decoder(hw_path) if hw_path.is_file() else None
    funct = _reconcile_funct(decoder_funct, header_funct)
    if funct:
        v1.setdefault("interfaces", []).append(funct)

    return {
        "schema_version": "2.0",
        "generator": {
            "name": "merlin.targetgen.rtl.circt_introspect",
            "version": GENERATOR_VERSION,
            "method": "firtool --ir-hw CIRCT HW-dialect port widths (accumulator) + GemminiISA.scala "
                      "funct block (decode table) + v1 grep (mesh/scratchpad)",
        },
        "inputs": {
            "hw_mlir": hw_path.name, "hw_sha": _sha(hw_path),
            "fir_sha": _sha(arts["fir"]), "isa_sha": _sha(isa_path),
            "extractor_sha": _sha(Path(__file__)),  # code change -> cache invalidates
        },
        "facts": v1,
    }


def dump_facts(out_path: Path | str | None = None, **kw) -> dict[str, Any]:
    """Build facts and cache to ``out_path`` (default: the purgeable run/cache dir, NOT merlin/);
    cache-hit (no rebuild) when input SHAs are unchanged. Promote the result into the committed pin
    with :func:`promote_facts`."""
    out = Path(out_path) if out_path is not None else _CACHE_DIR / "facts.json"
    rec = build_facts(**kw)
    if out.is_file():
        try:
            old = json.loads(out.read_text())
            if old.get("inputs") == rec["inputs"] and old.get("generator") == rec["generator"]:
                return old  # deterministic cache hit
        except Exception:
            pass
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    return rec


# Small distilled artifacts promoted alongside facts.json (the derived RoCC replay header/source are
# tiny and hand-consumed; the heavy hw.mlir/*.ll/*.o/arc bins stay in the purgeable cache).
_PROMOTE_EXTRAS = ("gemmini.state.json", "arc_results.json", "gemmini_arc_ports.h", "gemmini_arc_replay.c")


def promote_facts(target: str = "gemmini", *, src: Path | str | None = None) -> Path:
    """Promote a certified run's ``facts.json`` (+ small derived extras, if present next to it) into
    the committed target pin ``merlin/targets/<target>/contracts/rtl_facts/facts.json``. The run is
    the source of truth; this pins the certified snapshot so the checks run without the toolchain."""
    import shutil
    src_facts = Path(src) if src is not None else _CACHE_DIR / "facts.json"
    if not src_facts.is_file():
        raise FileNotFoundError(f"no run facts.json to promote: {src_facts} (run dump_facts first)")
    pin = rtl_facts_path(target)
    pin.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src_facts, pin)
    src_dir = src_facts.parent
    for name in _PROMOTE_EXTRAS:
        cand = src_dir / name
        if cand.is_file():
            shutil.copyfile(cand, pin.parent / name)
    return pin


# ------------------------------------------------------------------------- cross-check / validation
def validate(facts_rec: dict, contract: dict | None = None,
             rocc_funct_class: dict | None = None) -> dict[str, list[str]]:
    """Cross-check RTL facts against the hand-curated sources. Returns {'agree':[...], 'diverge':[...]}.

    Divergence is information (e.g. an unconfirmed contract capacity), not a hard error — the whole
    point of RTL extraction is to *correct* curated guesses."""
    facts = facts_rec["facts"]
    agree, diverge = [], []

    if contract:
        # reuse v1's mesh/scratchpad/dtype checks (they operate on the same facts shape)
        for p in V1.validate_against_contract(facts, contract):
            diverge.append(f"contract: {p}")
        if not [d for d in diverge if "contract" in d]:
            agree.append("contract: mesh + scratchpad + dtypes reproduced")
        acc = next((m for m in facts["memories"] if m["name"] == "accumulator"), None)
        want = (contract.get("capabilities", {}) or {}).get("accumulator_storage_bytes")
        if acc and want is not None:
            (agree if acc.get("bytes") == want else diverge).append(
                f"accumulator bytes RTL={acc.get('bytes')} vs contract={want}"
                + ("" if acc.get("bytes") == want else "  (RTL is authoritative; contract unconfirmed)"))

    if rocc_funct_class is not None:
        funct = next((i for i in facts.get("interfaces", []) if i.get("name") == "funct_decode_table"), None)
        if funct:
            legal = set(funct["legal_funct"])
            classifier = set(int(k) for k in rocc_funct_class)
            extra = classifier - legal
            if extra:
                diverge.append(f"rocc_decode classifies funct(s) {sorted(extra)} not in RTL legal set")
            else:
                agree.append(f"rocc_decode funct classifier {sorted(classifier)} ⊆ RTL legal "
                             f"{funct['legal_funct'][0]}..{funct['legal_funct'][-1]}")
    return {"agree": agree, "diverge": diverge}


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Deterministic RTL facts via CIRCT HW dialect (v2).")
    ap.add_argument("--out", default=str(_CACHE_DIR / "facts.json"),
                    help="run/cache output (default: purgeable artifacts/cache/rtl_introspect/)")
    ap.add_argument("--hw", default=str(DEFAULT_HW))
    ap.add_argument("--promote", action="store_true",
                    help="after building, promote the certified facts.json into the committed target pin")
    ap.add_argument("--validate", action="store_true", help="cross-check vs contract + rocc_decode")
    a = ap.parse_args(argv)
    rec = dump_facts(a.out, hw_path=Path(a.hw))
    if a.promote:
        pin = promote_facts("gemmini", src=a.out)
        print(f"promoted -> {pin}")
    facts = rec["facts"]
    acc = next((m for m in facts["memories"] if m["name"] == "accumulator"), {})
    funct = next((i for i in facts.get("interfaces", []) if i.get("name") == "funct_decode_table"), {})
    print(f"wrote {a.out}")
    print(f"  accumulator: depth={acc.get('depth')} bytes={acc.get('bytes')} "
          f"addr_width={acc.get('addr_width')}")
    print(f"  funct legal: {funct.get('legal_funct')}")
    if a.validate:
        import yaml
        from .. import rocc_decode  # noqa
        contract = yaml.safe_load(
            (_REPO / "merlin/targets/gemmini/contracts/target_contract.yaml").read_text())
        res = validate(rec, contract, rocc_decode._FUNCT_CLASS)
        print("  AGREE:");  [print(f"    + {x}") for x in res["agree"]]
        print("  DIVERGE:"); [print(f"    ! {x}") for x in res["diverge"]] or print("    (none)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
