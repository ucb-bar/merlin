"""merlin-rtl circt-introspect (v2) — deterministic, TARGET-AGNOSTIC RTL fact extraction.

Where the grep-over-FIRRTL v1 (:mod:`introspect`) stops, this picks up. It PREFERS mlc's
target-agnostic RTL discovery (the version-matched core HW dialect: the decoder-derived legal-opcode
set + the discovered mesh DIM + memory capacities) and falls back to a legacy chipyard/FIRRTL grep +
HW-port parse only for a target that actually ships those artifacts (gemmini). Nothing here bakes one
target's paths at import: every path is resolved *from the ``target`` argument* —

  * the HW dialect / accumulator capacity                    (per-target purgeable ``<t>_soc.hw.mlir``
    cache — the ``@AccumulatorMem`` port widths, superseded by discovery when mlc is available);
  * scratchpad + mesh + accumulator capacities               (mlc discovery, target-agnostic);
  * the legal command **funct decode table** + custom opcode (mlc's decoder ``comb.icmp-eq`` fan-out;
    the CODES are the fact). The funct NAMES are ISA vocabulary sourced generically: a chipyard
    target's Chisel ISA source (``<T>ISA.scala``, resolved by the generator convention) when present,
    else the target's DECLARED ISA headers (``target_experiment.yaml`` -> ``#define k_<NAME> <N>``),
    else a generic ``funct_<code>`` label.

Every fact carries ``evidence`` (the exact RTL/source token it came from). The result is cached to a
per-target ``facts.json`` keyed by the hashes of its inputs; re-extraction is a no-op cache hit.
``validate(...)`` cross-checks the facts against the target's hand-curated ``target_contract.yaml`` +
``rocc_decode`` — a disagreement is *surfaced*, not silently reconciled.

Deterministic, no LLM: the hardware is the source of truth. (Extraction still needs mlc to KNOW the
target's RTL — a novel accelerator registers its RTL with mlc first; that is mlc's coverage, not a
gemmini assumption baked here.)

CLI::

    python -m merlin.targetgen.rtl.circt_introspect [--target <t>] [--out facts.json] [--hw <t.hw.mlir>]
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from merlin.common.paths import repo_root
from typing import Any, Iterable

from . import introspect as V1
from .facts import rtl_cache_dir

_REPO = repo_root()  # the repo root (contains merlin/)

GENERATOR_VERSION = "rtl-introspect-v3-slot-derived-opcode"
# RISC-V ISA STANDARD custom-N major opcodes — fixed by the base ISA for EVERY RISC-V chip, NOT a
# per-target fact. WHICH custom slot a RoCC accelerator is wired to IS target-specific; it is resolved
# from the target's own reviewed encoding (contract ``encoding.rocc_custom_slot``) — never a baked
# gemmini=custom3 assumption. The major opcode is a SoC-config (OpcodeSet) fact the accelerator's own
# decoder does not carry, so it cannot be recovered from the core HW dialect the funct fan-out uses.
_RISCV_CUSTOM_OPCODES = {0: 0x0b, 1: 0x2b, 2: 0x5b, 3: 0x7b}  # derived-ok: RISC-V standard custom-0/1/2/3 encodings
# RoCC ``.insn`` func3 is the xd/xs1/xs2 register-usage field — a RoCC ABI field that varies per
# instruction and is NOT an instruction-identity constraint (identity is func7; see rocc_decode), and
# not per-target. Recorded as the standard reg-usage default so the pin carries a value.
_ROCC_FUNCT3_DEFAULT = 0x3  # derived-ok: RoCC xd/xs1/xs2 reg-usage field (standard, not identity/per-target)


def _contract_rocc_slot(target: str | None) -> int | None:
    """The RISC-V custom SLOT (0..3) this target's RoCC is wired to, read from its reviewed contract
    (``encoding.rocc_custom_slot``). Reads the raw contract yaml (NOT the manifest — loading the manifest
    would re-enter facts regeneration). None when the target declares no slot (a non-RoCC target, or one
    that has not declared it) -> the opcode fails closed rather than defaulting to gemmini's custom-3."""
    if not target:
        return None
    try:
        import yaml  # function-local: off the hot path, avoids an import cycle
        from .facts import target_contract_path
        p = target_contract_path(target)
        if not p.is_file():
            return None
        enc = (yaml.safe_load(p.read_text(encoding="utf-8")) or {}).get("encoding") or {}
        slot = enc.get("rocc_custom_slot")
        return int(slot) if slot is not None else None
    except Exception:  # noqa: BLE001 — a missing/broken contract is simply "no declared slot"
        return None


def _rocc_custom_opcode(target: str | None) -> int | None:
    """The RoCC MAJOR opcode for ``target`` = the RISC-V-standard encoding of the custom SLOT it uses.
    The slot is the per-target fact (from the reviewed contract encoding); slot->opcode is a RISC-V
    standard. Returns None (UNKNOWN — fail closed) when the slot is undeclared, never a baked default."""
    slot = _contract_rocc_slot(target)
    return _RISCV_CUSTOM_OPCODES.get(slot) if slot is not None else None


# ---------------------------------------------------------------------- per-target path resolution
def _soc_hw_path(target: str) -> Path:
    """The per-target SoC HW-dialect cache (``firtool --ir-hw`` output) the accumulator port-parser
    reads, under the PURGEABLE rtl cache — never inside merlin/, never a baked gemmini path. This is
    the SoC dialect the ``@AccumulatorMem`` port-parse needs (mlc's core dialect drives the decoder)."""
    return rtl_cache_dir(target) / f"{target}_soc.hw.mlir"


def isa_scala_path(target: str, chipyard_root: str | Path | None = None) -> Path:
    """The target's Chisel ISA source, by the chipyard generator convention
    ``generators/<t>/src/main/scala/<t>/<T>ISA.scala`` — DERIVED from the target name, not a hardcoded
    ``GemminiISA.scala`` (gemmini resolves the identical file it always did). Returns the path whether
    or not it exists; callers gate on ``.is_file()``."""
    root = Path(chipyard_root) if chipyard_root is not None else Path(V1.DEFAULT_CHIPYARD)
    cap = target[:1].upper() + target[1:]
    return root / "generators" / target / "src" / "main" / "scala" / target / f"{cap}ISA.scala"


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
        "legal_funct": legal,
        "names": {str(k): table[k] for k in legal},
        "evidence": f"GemminiISA.scala // funct values block: {len(legal)} codes "
                    f"[{legal[0]}..{legal[-1]}] up to CONFIG_EX",
    }


# ------------------------------------------------------- decoder-derived funct set (the true ISA)
def extract_funct_table_via_decoder(target: str) -> dict[str, Any] | None:
    """The legal command-opcode set derived from the HW-dialect DECODER (mlc's comb.icmp-eq fan-out) —
    the actual ISA the silicon implements. Target-parameterized (mlc resolves the target's core HW
    dialect). Returns a table dict (same shape as :func:`extract_funct_table`), or None if mlc is
    unavailable or the HW dialect cannot be parsed — an honest fallback, never a fake pass."""
    try:
        from . import mlc_bridge
        if not mlc_bridge.mlc_available()[0]:
            return None
        # Agnostic extraction: mlc resolves the target's version-matched core HW dialect (the passed
        # hw_path is only a legacy fallback for the accumulator/mesh path). ``target`` is a parameter.
        res = mlc_bridge.discover_legal_opcodes(target)
    except Exception:  # noqa: BLE001 — a parse/version skew means "no decoder facts", fall back cleanly
        return None
    legal = res.get("legal_opcodes")
    if not legal:
        return None
    return {
        "name": "funct_decode_table",
        "legal_funct": legal,
        "names": {},  # the decoder yields numeric codes; names are cross-referenced from the header below
        "hw_source": res.get("hw_source"),
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
    else:
        # No name vocabulary at all (target declares no ISA source) -> generic ``funct_<code>`` labels.
        # The CODES are the derived fact; the names are a convenience the target can supply later.
        decoder["names"] = {str(k): f"funct_{k}" for k in decoder["legal_funct"]}
    return decoder


# ------------------------------------------------------------- funct NAMES, sourced target-agnostically
def _functs_from_headers(headers: Iterable[str | Path]) -> dict[int, str]:
    """Map funct code -> name from a target's C ISA headers by the generic ``#define k_<NAME> <N>``
    convention — a structured split (no regex), mirroring
    :func:`merlin.targetgen.oot_starterkit.verify.legal_functs`. Earlier headers win a code."""
    out: dict[int, str] = {}
    for h in headers:
        try:
            txt = Path(h).read_text(errors="ignore")
        except OSError:
            continue
        for line in txt.splitlines():
            parts = line.split()   # ``#define  k_<NAME>  <N>`` — any whitespace
            if len(parts) >= 3 and parts[0] == "#define" and parts[1].startswith("k_") and parts[2].isdigit():
                name = parts[1][2:]
                if name and all(c.isupper() or c.isdigit() or c == "_" for c in name):
                    out.setdefault(int(parts[2]), name)
    return out


def _defines_from_headers(headers: Iterable[str | Path], prefix: str) -> dict[str, int]:
    """Map NAME -> value from a target's C ISA headers by the generic ``#define <prefix><NAME> <N>``
    convention (structured split, no regex). Used for the ``k_`` funct defines and the un-prefixed
    ``CONFIG_*`` subtype defines. Earlier headers win a name."""
    out: dict[str, int] = {}
    for h in headers:
        try:
            txt = Path(h).read_text(errors="ignore")
        except OSError:
            continue
        for line in txt.splitlines():
            parts = line.split()   # ``#define  <prefix><NAME>  <N>`` — any whitespace
            if len(parts) >= 3 and parts[0] == "#define" and parts[1].startswith(prefix) and parts[2].isdigit():
                name = parts[1][len(prefix):]
                if name and all(c.isupper() or c.isdigit() or c == "_" for c in name):
                    out.setdefault(name, int(parts[2]))
    return out


def crosscheck_semantic_class_names(target: str, semantic_class: dict | None = None) -> dict[str, list]:
    """Prove the manifest ``semantic_class`` NAMES are HEADER-DERIVED: for each ``code -> name`` the manifest
    declares, compare the name to the target's ISA-header ``#define k_<NAME> <code>`` at the SAME code.
    Returns ``{exact, alias, missing, mismatch}``:

      * ``exact``    — code present in the headers with the identical name (the derivation is byte-exact);
      * ``alias``    — the manifest name is a PREFIX of the header name (a compiler alias that drops a
                       dataflow suffix, e.g. ``LOOP_CONV`` for the header's ``LOOP_CONV_WS``) — reviewed, benign;
      * ``missing``  — code the headers do not define (an ungrounded manifest name — a real alarm);
      * ``mismatch`` — code present but the name neither matches nor prefixes the header (a real alarm).

    The gate is ``missing == mismatch == []`` (every declared class is a real header funct, named by the
    header modulo a reviewed prefix-alias). Skips entirely (empty lists) when the target declares no ISA
    headers. No mlc needed — a pure header parse."""
    from ..target_experiment import load_capability_manifest
    if semantic_class is None:
        semantic_class = (load_capability_manifest(target).encoding or {}).get("semantic_class") or {}
    hdr_by_code = _functs_from_headers(_declared_isa_headers(target))  # already {code: name}
    out: dict[str, list] = {"exact": [], "alias": [], "missing": [], "mismatch": []}
    if not hdr_by_code:
        return out
    for code, name in semantic_class.items():
        code, name = int(code), str(name)
        hname = hdr_by_code.get(code)
        if hname is None:
            out["missing"].append((code, name))
        elif hname == name:
            out["exact"].append((code, name))
        elif hname.startswith(name):
            out["alias"].append((code, name, hname))
        else:
            out["mismatch"].append((code, name, hname))
    return out


def crosscheck_config_subtype_names(target: str, config_subtype: dict | None = None) -> dict[str, list]:
    """Prove the manifest ``config_subtype`` NAMES are HEADER-DERIVED: compare each ``code -> name`` to the
    ISA-header ``#define <NAME> <code>`` (the un-prefixed ``CONFIG_EX``/``CONFIG_LD``/``CONFIG_ST`` defines).
    Returns ``{exact, missing, mismatch}``; the gate is ``missing == mismatch == []``. Pure header parse."""
    from ..target_experiment import load_capability_manifest
    if config_subtype is None:
        config_subtype = (load_capability_manifest(target).encoding or {}).get("config_subtype") or {}
    hdr = _defines_from_headers(_declared_isa_headers(target), "")  # un-prefixed ``#define NAME N``
    out: dict[str, list] = {"exact": [], "missing": [], "mismatch": []}
    if not hdr:
        return out
    for code, name in config_subtype.items():
        code, name = int(code), str(name)
        hcode = hdr.get(name)
        if hcode is None:
            out["missing"].append((code, name))
        elif hcode == code:
            out["exact"].append((code, name))
        else:
            out["mismatch"].append((code, name, hcode))
    return out


def _declared_isa_headers(target: str) -> list[Path]:
    """The ISA header files the target's ``target_experiment.yaml`` declares (bundle-convention,
    repo-root-relative strings), resolved to absolute paths. Empty when the target has no descriptor
    or declares no headers (e.g. arc/cyclotron targets) — the caller then falls back to generic names."""
    from ..target_experiment import load_target_experiment
    exp = _REPO / "merlin" / "experiments"
    if not exp.is_dir():
        return []
    for desc in sorted(exp.glob("*/target_experiment.yaml")):
        try:
            te = load_target_experiment(desc)
        except Exception:  # noqa: BLE001 — a malformed descriptor is just "no header source for this target"
            continue
        if te.target == target:
            return [_REPO / h for h in te.isa_headers]
    return []


def _funct_name_table(target: str, isa_path: Path | None) -> dict[str, Any] | None:
    """The funct-name vocabulary for ``target``, sourced agnostically: the Chisel ISA source
    (``<T>ISA.scala``) when the target ships one, else the target's DECLARED ISA headers (``#define
    k_*``), else None (codes-only, generic ``funct_<code>`` names applied downstream). Returns a
    funct-table dict (``legal_funct``/``names``), whose names the decoder-derived set borrows."""
    if isa_path is not None and Path(isa_path).is_file():
        return extract_funct_table(Path(isa_path).read_text(errors="replace"))
    headers = _declared_isa_headers(target)
    code_name = _functs_from_headers(headers)
    if not code_name:
        return None
    legal = sorted(code_name)
    return {
        "name": "funct_decode_table",
        "legal_funct": legal,
        "names": {str(k): code_name[k] for k in legal},
        "evidence": f"ISA headers {[Path(h).name for h in headers]} '#define k_* <N>': "
                    f"{len(legal)} names [{legal[0]}..{legal[-1]}]",
    }


# ---------------------------------------------------------------------------------- assemble + cache
def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16] if path.is_file() else "missing"


def _facts_from_discovery(target: str, facts: dict) -> list[str]:
    """Override mesh + operand-scratchpad + accumulator facts with mlc RTL discovery (target-agnostic:
    DIM from the discovered mesh, capacities summed from the discovered banks). Mutates ``facts`` in
    place; returns the list of fact names that came from discovery (for provenance)."""
    from . import mlc_bridge
    sourced: list[str] = []
    dim = mlc_bridge.discovered_dim(target)
    if dim:
        facts["arrays"] = [a for a in facts.get("arrays", []) if a.get("name") != "mesh"]
        facts["arrays"].append({"name": "mesh", "rows": dim, "cols": dim, "source": "mlc_discovery"})
        sourced.append("mesh")
    caps = mlc_bridge.discovered_capacities(target)
    if caps:
        keep = [m for m in facts.get("memories", []) if m.get("name") not in ("scratchpad", "accumulator")]
        if caps.get("operand_bytes"):
            keep.append({"name": "scratchpad", "bytes": caps["operand_bytes"],
                         "depth": caps.get("operand_depth"), "source": "mlc_discovery"})
            sourced.append("scratchpad")
        if caps.get("accumulator_bytes"):
            keep.append({"name": "accumulator", "bytes": caps["accumulator_bytes"],
                         "depth": caps.get("accumulator_depth"), "source": "mlc_discovery"})
            sourced.append("accumulator")
        facts["memories"] = keep
    return sourced


def build_facts(hw_path: Path | str | None = None, isa_path: Path | str | None = None,
                chipyard_root: str | Path | None = None, target: str | None = None) -> dict[str, Any]:
    """Assemble the RTL facts for ``target``. PREFERS mlc RTL discovery (target-agnostic: mesh DIM +
    memory capacities + the decoder-derived ISA); the chipyard FIRRTL grep + HW-port parse is the
    legacy FALLBACK, run only for a target that ships a Chisel ISA source (gemmini) and skipped
    entirely otherwise (a non-chipyard target relies wholly on discovery — no gemmini path is touched).

    Every input path is resolved FROM ``target`` when not explicitly overridden: the SoC HW cache
    (``<t>_soc.hw.mlir``), the Chisel ISA source (``<T>ISA.scala`` by generator convention), the
    declared ISA headers. Provenance-stamped; does not write (see :func:`dump_facts`).

    ``target`` is required (last so the path overrides can stay positional); a missing target is a
    loud error, never a silent gemmini fallback."""
    if target is None:
        raise ValueError("build_facts requires an explicit target (no default is assumed)")
    chipyard_root = V1.DEFAULT_CHIPYARD if chipyard_root is None else chipyard_root
    hw_path = _soc_hw_path(target) if hw_path is None else Path(hw_path)
    isa_path = isa_scala_path(target, chipyard_root) if isa_path is None else Path(isa_path)

    fir_sha = isa_sha = "n/a"
    v1: dict[str, Any] = {}
    if isa_path.is_file():   # a chipyard target with a Chisel ISA source -> legacy FIRRTL grep + HW-port
        try:
            arts = V1.find_artifacts(chipyard_root)
            v1 = V1.extract_facts(arts["fir"], arts["hierarchy"])
            hw_text = hw_path.read_text(errors="replace") if hw_path.is_file() else ""
            acc = extract_accumulator(hw_text) if hw_text else None
            mems = [m for m in v1.get("memories", []) if m.get("name") != "accumulator"]
            if acc:
                mems.append(acc)
            v1["memories"] = mems
            fir_sha, isa_sha = _sha(arts["fir"]), _sha(isa_path)
        except Exception:  # noqa: BLE001 — chipyard artifacts absent/broken: rely on mlc discovery
            v1 = {}

    # PREFER mlc discovery (target-agnostic) for mesh + memory capacities.
    sourced = _facts_from_discovery(target, v1)

    # Funct decode table: PREFER the decoder-derived legal set (the ISA the silicon implements) over the
    # name parse; fall back to the names when mlc / a version-matched HW dialect is unavailable. NAMES
    # are sourced target-agnostically (Chisel ISA source > declared ISA headers > generic funct_<code>).
    header_funct = _funct_name_table(target, isa_path)
    decoder_funct = extract_funct_table_via_decoder(target)
    funct = _reconcile_funct(decoder_funct, header_funct)
    if funct:
        # The RoCC major opcode is a SoC-config fact the accelerator decoder does not carry: resolve it
        # from the target's reviewed custom-slot (RISC-V standard slot->opcode), UNKNOWN if undeclared.
        funct["custom_opcode"] = _rocc_custom_opcode(target)
        funct.setdefault("funct3", _ROCC_FUNCT3_DEFAULT)
        v1.setdefault("interfaces", []).append(funct)
        sourced.append("funct" if funct.get("method", "").startswith("decoder") else "funct(header)")

    return {
        "schema_version": "2.0",
        "generator": {
            "name": "merlin.targetgen.rtl.circt_introspect",
            "version": GENERATOR_VERSION,
            "method": f"mlc RTL discovery (target-agnostic: {', '.join(sourced) or 'none'}) + chipyard "
                      "FIRRTL grep/HW-port fallback",
        },
        "inputs": {
            "target": target,
            "hw_mlir": hw_path.name, "hw_sha": _sha(hw_path),
            "fir_sha": fir_sha, "isa_sha": isa_sha,
            "extractor_sha": _sha(Path(__file__)),  # code change -> cache invalidates
        },
        "facts": v1,
    }


def dump_facts(out_path: Path | str | None = None, **kw) -> dict[str, Any]:
    """Build facts and write the GENERATED artifact to ``out_path`` (default: the purgeable cache dir,
    NOT merlin/); cache-hit (no rebuild) when input SHAs are unchanged. This is the writer
    :func:`merlin.targetgen.rtl.facts.ensure_facts` calls to fill a cold cache."""
    rec = build_facts(**kw)
    out = Path(out_path) if out_path is not None else rtl_cache_dir(kw["target"]) / "facts.json"
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


# ------------------------------------------------------------------------- cross-check / validation
def validate(facts_rec: dict, contract: dict | None = None,
             rocc_funct_class: dict | None = None) -> dict[str, list[str]]:
    """Cross-check RTL facts against the hand-curated sources. Returns {'agree':[...], 'diverge':[...]}.

    Divergence is information (e.g. an unconfirmed contract capacity), not a hard error — the whole
    point of RTL extraction is to *correct* curated guesses."""
    facts = facts_rec["facts"]
    agree, diverge = [], []

    if contract:
        # reuse v1's compute_units-coverage check (mesh/scratchpad/dtype/accumulator capacities are
        # no longer hand-declared in the contract — they ARE these facts, so nothing to cross-check).
        for p in V1.validate_against_contract(facts, contract):
            diverge.append(f"contract: {p}")
        if not [d for d in diverge if "contract" in d]:
            agree.append("contract: RTL datapaths covered by declared compute_units")

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
    ap = argparse.ArgumentParser(description="Deterministic, target-agnostic RTL facts via CIRCT HW (v2).")
    ap.add_argument("--target", required=True,
                    help="the accelerator target whose RTL facts to extract")
    ap.add_argument("--out", default=None,
                    help="output path (default: purgeable artifacts/cache/rtl_introspect/<target>/facts.json)")
    ap.add_argument("--hw", default=None, help="override the SoC HW-dialect input (default: per-target cache)")
    ap.add_argument("--validate", action="store_true", help="cross-check vs contract + rocc_decode")
    a = ap.parse_args(argv)
    out = a.out or str(rtl_cache_dir(a.target) / "facts.json")
    rec = dump_facts(out, target=a.target, hw_path=(Path(a.hw) if a.hw else None))
    facts = rec["facts"]
    acc = next((m for m in facts.get("memories", []) if m.get("name") == "accumulator"), {})
    funct = next((i for i in facts.get("interfaces", []) if i.get("name") == "funct_decode_table"), {})
    print(f"wrote {out}")
    print(f"  accumulator: depth={acc.get('depth')} bytes={acc.get('bytes')} "
          f"addr_width={acc.get('addr_width')}")
    print(f"  funct legal: {funct.get('legal_funct')}")
    if a.validate:
        import yaml
        from .facts import target_contract_path
        contract = yaml.safe_load(target_contract_path(a.target).read_text())
        rocc_class = None
        try:
            from .. import rocc_decode  # RoCC classifier for this target; best-effort cross-check
            rocc_class = rocc_decode.funct_class_for(a.target)
        except Exception:  # noqa: BLE001 — no classifier for this target: skip that cross-check
            pass
        res = validate(rec, contract, rocc_class)
        print("  AGREE:");  [print(f"    + {x}") for x in res["agree"]]
        print("  DIVERGE:"); [print(f"    ! {x}") for x in res["diverge"]] or print("    (none)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
