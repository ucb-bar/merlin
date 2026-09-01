"""Deterministic, no-LLM extraction of Muon (RadianceMuonConfig) hardware facts from the real RTL.

The Muon analog of :mod:`merlin.targetgen.rtl.circt_introspect` (Gemmini). The hardware is the source
of truth: every geometry fact here is read out of the **elaborated FIRRTL + module hierarchy + memory
macro list + device tree** that CIRCT/firtool produced for ``RadianceMuonConfig`` (under
``$MERLIN_CHIPYARD/sims/vcs/generated-src/...``), and emitted into one provenance-stamped
``muon_facts.json`` -- each fact carrying the exact RTL token it came from. Consumers (digest, advisory
checker, the manifest deriver, the fact-bundle renderer) read that artifact instead of re-deriving the
geometry by crawling Radiance Scala.

FACT SOURCE RULE (why this module was rewritten
===============================================
A hardware fact may only be derived from CIRCT output or the target's own RTL repo. The cyclotron
perf model (``MERLIN_MUON_CONFIG`` -> ``.../cyclotron/config.toml``) is a Rust cycle model's own
configuration -- it may CROSS-CHECK a derived fact, it may not BE its source. Two measured failures
forced this:

1. **The geometry was baked, not read.** The previous version resolved its config through raw
   ``os.environ`` (not :func:`merlin.common.paths.env`), so in any plain process -- pytest, a CLI run,
   an agent sandbox -- ``MERLIN_MUON_CONFIG`` was unset, ``config_path()`` returned the placeholder
   ``/path/to/autocomp/scripts/muon/config_muon.toml``, the TOML parse returned ``{}``, and
   ``cfg.get("num_lanes", 16)`` / ``("num_warps", 8)`` / ``("num_cores", 2)`` published the DEFAULTS as
   facts -- under an evidence string that read ``"config_muon.toml: num_lanes=16, num_warps=8,
   num_cores=2"`` while no config file had been opened at all. Measured, before this change::

       config_path: /path/to/autocomp/scripts/muon/config_muon.toml exists: False
       rtl_present: False
       simt: {"lanes_per_warp": 16, "warps_per_core": 8, "cores": 2, "threads_per_core": 128}

   The numbers happened to match the real config, which made them UNFALSIFIABLE rather than wrong: no
   run could ever distinguish "read from the hardware" from "never read anything".

2. **The perf model and the RTL DISAGREE.** With the paths actually resolved, the elaborated
   ``top_module_hierarchy.json`` for ``RadianceMuonConfig`` contains exactly ONE ``MuonCore`` instance
   (``/ChipTop/system/cluster_prci_domain/element_reset_domain_element/tile_prci_domain/
   element_reset_domain_muon_tile/core``), while the cyclotron config declares ``num_cores = 2``. The
   old code published ``cores = 2`` and a ``64 flop/cycle`` / ``32 GFLOP/s`` peak built on it. The RTL
   wins; the config's reading is kept beside it as a recorded ``cross_check`` disagreement rather than
   silently dropped, because a perf model configured for a different core count is itself a finding.

THREE STATES, never a default
=============================
Every fact is one of:

* ``derived``        -- read out of the RTL, with the token quoted in ``evidence``;
* ``absent``         -- the input that would carry it is not on this machine (no ``$MERLIN_CHIPYARD``,
  no elaboration) -- value ``None``;
* ``undeterminable`` -- the input IS present but carries no sound reading (e.g. the design's fixed
  clocks disagree, so there is no single clock to report) -- value ``None``.

There is no fourth state where a plausible number stands in. A consumer that needs a value it did not
get must say so; that is strictly better than a wrong number nobody can trace.

This module is target-gen for Muon ONLY; it imports nothing Gemmini-specific and never runs the agent.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# The single target this SIMT introspect serves. It lives HERE (the target-specific introspect
# module — the Muon analog of circt_introspect for Gemmini), not in the kind-routed core dispatch
# (mlc_bridge), so the core stays target-name-free: the bridge routes by compute-unit KIND and asks
# this module which target it is the introspect for.
TARGET = "muon"
VCS_CONFIG = "RadianceMuonConfig"

# RTL module names in the target's OWN elaborated hierarchy. These are not "constants about hardware"
# in the prohibited sense — they are the identifiers the RTL repo itself elaborates under, and they are
# used to LOOK UP a fact in the hierarchy, never to substitute for one. Every number below comes out of
# the elaboration; nothing here asserts a width, a count, or a size.
MOD_CORE = "MuonCore"
MOD_TILE = "MuonTile"
MOD_CLUSTER = "RadianceCluster"
MOD_SHARED_MEM = "RadianceSharedMem"

#: Signal names inside ``MuonCore``'s ``io`` bundle whose FIRRTL TYPE carries a geometry fact.
#: ``tmask`` is the per-thread execution mask: definitionally one bit per lane. ``perWarp`` is the
#: per-warp performance-counter bundle: definitionally one element per warp. ``warpId`` is the warp
#: index field: its width bounds the warp count and serves as the cross-check on ``perWarp``.
SIG_TMASK = "tmask"
SIG_PERWARP = "perWarp"
SIG_WARPID = "warpId"
SIG_REGS_BUNDLE = "regs"      # the trace bundle's register-read record
SIG_REG_ADDR = "address"      # inside it: the architectural register index

#: The Muon compiler's live-register policy, from the Radiance toolchain doc. NOT a hardware fact and
#: never reported as one — see the ``compiler_limit_source`` label in :func:`build_facts`.
COMPILER_REG_LIMIT = 128

DERIVED, ABSENT, UNDETERMINABLE = "derived", "absent", "undeterminable"


# ------------------------------------------------------------------------------- input resolution
def chipyard_root() -> Path | None:
    """The chipyard checkout holding the elaboration, or None when unset.

    Resolved through :func:`merlin.common.paths.env`, which honors the gitignored ``.env`` as a real
    config source. Reading raw ``os.environ`` here is the bug described in the module docstring: every
    process that did not export the variable by hand saw "no RTL" and fell through to baked numbers.
    There is deliberately NO default path — a placeholder that cannot exist on any machine is not a
    fallback, it is a way to make "absent" look like "configured"."""
    from merlin.common.paths import env
    d = env("MERLIN_CHIPYARD")
    return Path(d).expanduser() if d else None


def config_path() -> Path | None:  # fact-source-ok: accessor for the CROSS-CHECK only; the geometry above it comes from the elaboration
    """The cyclotron perf-model config, or None when unset.

    NOT a fact source (see the module docstring): a Rust cycle model's own config is not the hardware.
    It is read only to CROSS-CHECK geometry already derived from the RTL, and its reading is recorded
    beside the derived one when the two disagree."""
    from merlin.common.paths import env
    d = env("MERLIN_MUON_CONFIG")  # fact-source-ok: cross-check only — the value never reaches a fact; see _cross_check(), which annotates but never supplies
    return Path(d).expanduser() if d else None


def _gen_src_dir() -> Path | None:
    root = chipyard_root()
    if root is None:
        return None
    return root / "sims/vcs/generated-src" / f"chipyard.harness.TestHarness.{VCS_CONFIG}"


def _read(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


# --------------------------------------------------------------------------- FIRRTL type parsing
def _uint_width(text: str, key: str) -> int | None:
    """The ``W`` of the first ``<key> : UInt<W>`` in ``text``, parsed structurally (no regex, per the
    repo's no-regex rule: a pattern narrow enough to match one spelling silently drops the others)."""
    marker = f"{key} : UInt<"
    i = text.find(marker)
    if i < 0:
        return None
    width, sep, _ = text[i + len(marker):].partition(">")
    return int(width) if sep and width.isdigit() else None


def _bundle_span(text: str, key: str, start: int = 0) -> tuple[int, int] | None:
    """``(body_start, close_index)`` of the first ``<key> : { ... }`` at or after ``start``.

    Brace-matched rather than pattern-matched, because the bundle nests arbitrarily deep and the
    closing brace that matters is not the first one after the key."""
    marker = f"{key} : {{"
    i = text.find(marker, start)
    if i < 0:
        return None
    j = i + len(marker) - 1          # index of the opening '{'
    depth = 0
    while j < len(text):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return (i + len(marker), j)
        j += 1
    return None


def _bundle_body(text: str, key: str) -> str | None:
    """The contents of ``<key> : { ... }``, so a field name can be resolved WITHIN its own bundle.

    Needed because a bare ``str.find`` for a field name finds the FIRST one in the whole flattened
    port line, and the names repeat across sub-bundles: reading ``address`` off the top level picked
    up ``imem.req.bits.address : UInt<32>`` and published a 4-billion-entry register file. Scoping the
    lookup to the enclosing bundle is what makes the reading mean what it says."""
    span = _bundle_span(text, key)
    return text[span[0]:span[1]] if span else None


def _bundle_cardinality(text: str, key: str, start: int = 0) -> int | None:
    """The ``N`` of ``<key> : { ... }[N]`` — the element count of a FIRRTL vector-of-bundle."""
    span = _bundle_span(text, key, start)
    if span is None:
        return None
    rest = text[span[1] + 1:]
    if not rest.startswith("["):
        return None
    n, sep, _ = rest[1:].partition("]")
    return int(n) if sep and n.isdigit() else None


def _scan_fir(gs: Path) -> dict[str, str | None]:
    """ONE streaming pass over the elaborated FIRRTL, collecting the two lines that carry facts.

    Streamed line-by-line and stopped early because the elaborated ``.fir`` is ~430 MB; the previous
    version read the whole file into a string for a single ``str.find``.

    * ``core_io``  -- ``MuonCore``'s ``output io : {...}`` line, whose TYPE encodes lanes/warps/regs;
    * ``smem_line`` -- the first shared-memory ``smem mem : UInt<W>[E] [D]`` declaration (bank evidence).
    """
    fir = gs / f"chipyard.harness.TestHarness.{VCS_CONFIG}.fir"
    core_io: str | None = None
    smem_line: str | None = None
    in_core = False
    try:
        with fir.open(encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                s = line.strip()
                if s.startswith("module ") or s.startswith("public module "):
                    name = s.split("module ", 1)[1].split(":")[0].split("@")[0].strip()
                    in_core = name == MOD_CORE
                elif in_core and core_io is None and s.startswith("output io :"):
                    core_io = s
                elif smem_line is None and "smem mem : UInt<" in s:
                    smem_line = s.split("@[")[0].strip()
                if core_io is not None and smem_line is not None:
                    break
    except OSError:
        return {"core_io": None, "smem_line": None}
    return {"core_io": core_io, "smem_line": smem_line}


# ------------------------------------------------------------------- module-hierarchy (instance) walk
def _hierarchy(gs: Path) -> dict | None:
    """The elaborated INSTANCE tree (``top_module_hierarchy.json``), or None when absent."""
    try:
        return json.loads((gs / "top_module_hierarchy.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _instance_counts(node: dict | None) -> dict[str, int]:
    """Count instances per module name by walking the instance tree.

    Structural walk, not a text ``count('"MuonCore"')``: the old version counted quoted occurrences in
    the raw JSON, which conflates ``module_name`` with ``instance_name`` and cannot tell one instance
    from twenty. The tree really is per-instance (``plusarg_reader`` occurs 456 times in it), so the
    walk is the only reading that answers "how many cores does this elaboration have"."""
    out: dict[str, int] = {}
    stack = [node] if node else []
    while stack:
        n = stack.pop()
        name = n.get("module_name")
        if name:
            out[name] = out.get(name, 0) + 1
        stack.extend(n.get("instances") or [])
    return out


def _subtree(node: dict | None, module_name: str) -> dict | None:
    """The first subtree whose ``module_name`` matches, or None."""
    stack = [node] if node else []
    while stack:
        n = stack.pop()
        if n.get("module_name") == module_name:
            return n
        stack.extend(n.get("instances") or [])
    return None


# --------------------------------------------------------------------------- memory-macro geometry
def _mems_conf(gs: Path) -> dict[str, dict[str, str]]:
    """The elaboration's SRAM macro list (``*.mems.conf``) as ``{macro: {depth, width, ...}}``.

    Format is ``name <macro> depth <D> width <W> ports <P> [mask_gran <G>]`` — whitespace-delimited
    key/value pairs, parsed with ``str.split`` (no regex)."""
    out: dict[str, dict[str, str]] = {}
    for p in sorted(gs.glob("*.mems.conf")):
        for line in _read(p).splitlines():
            tok = line.split()
            if len(tok) < 2 or tok[0] != "name":
                continue
            rec: dict[str, str] = {}
            i = 2
            while i + 1 < len(tok):
                rec[tok[i]] = tok[i + 1]
                i += 2
            out[tok[1]] = rec
    return out


def _shared_memory_bytes(hier: dict | None, mems: dict[str, dict[str, str]]) -> dict[str, Any]:
    """Shared-memory CAPACITY summed from the RTL's own SRAM macros under the shared-memory module.

    Replaces a literal ``128 * 1024`` that was published under an evidence string naming a Scala config
    ("RadianceBaseConfig TapeoutSmemConfig (128 KiB/cluster)") no code here ever opened. The derived
    sum for ``RadianceMuonConfig`` is 64 x ``radiance_smem_bank_ext`` x (depth 512 x width 32 bits) =
    131072 bytes — the same number, now falsifiable: change the RTL and this moves."""
    sm = _subtree(hier, MOD_SHARED_MEM)
    if sm is None:
        return {"bytes_per_cluster": None, "state": ABSENT,
                "evidence": f"no {MOD_SHARED_MEM} instance in the elaborated hierarchy"}
    counts = _instance_counts(sm)
    total = 0
    parts: list[str] = []
    for macro, n in sorted(counts.items()):
        rec = mems.get(macro or "")
        if not rec or not rec.get("depth", "").isdigit() or not rec.get("width", "").isdigit():
            continue
        per = int(rec["depth"]) * int(rec["width"]) // 8
        total += n * per
        parts.append(f"{n}x {macro} (depth {rec['depth']} x width {rec['width']}b = {per} B)")
    if not parts:
        # The module is there but no macro under it appears in *.mems.conf: present input, no sound
        # reading. That is UNDETERMINABLE, not zero — a zero here would read as "no shared memory".
        return {"bytes_per_cluster": None, "state": UNDETERMINABLE,
                "evidence": f"{MOD_SHARED_MEM} present but none of its leaf modules "
                            f"{sorted(k for k in counts if k)} appear in *.mems.conf"}
    return {"bytes_per_cluster": total, "state": DERIVED,
            "evidence": f"RTL SRAM macros under {MOD_SHARED_MEM}: " + " + ".join(parts)
                        + f" = {total} B/cluster"}


# ----------------------------------------------------------------------------- device-tree readings
def _dt_json(gs: Path) -> dict | None:
    """The elaborated device-tree JSON (``<config>.json``)."""
    try:
        return json.loads((gs / f"chipyard.harness.TestHarness.{VCS_CONFIG}.json")
                          .read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _clock_hz(dt: dict | None) -> dict[str, Any]:
    """The design clock, from the ``fixed-clock`` nodes the elaboration emitted.

    Replaces a module-level ``CLOCK_HZ = 500_000_000`` literal — an assumed constant, and the exact
    shape the repo's cardinal rule forbids: it would have kept reporting 500 MHz for a design
    elaborated at any other frequency, silently rescaling every GFLOP/s number downstream.

    All fixed clocks must AGREE for a single number to be reportable. When they do not, there is no
    "the clock" and the state is ``undeterminable`` — reported with the disagreeing set, never
    resolved by picking one."""
    if dt is None:
        return {"clock_hz": None, "state": ABSENT, "evidence": "no elaborated device-tree JSON"}
    found: dict[str, int] = {}

    def walk(node: Any, path: str) -> None:
        if not isinstance(node, dict):
            return
        if "fixed-clock" in (node.get("compatible") or []):
            f = (node.get("clock-frequency") or [None])[0]
            if isinstance(f, int):
                found[path] = f
        for k, v in node.items():
            walk(v, f"{path}/{k}")

    walk(dt, "")
    if not found:
        return {"clock_hz": None, "state": ABSENT,
                "evidence": "device tree declares no fixed-clock node"}
    distinct = sorted(set(found.values()))
    if len(distinct) != 1:
        return {"clock_hz": None, "state": UNDETERMINABLE,
                "evidence": f"fixed-clock nodes disagree: {found} — no single design clock"}
    return {"clock_hz": distinct[0], "state": DERIVED,
            "evidence": f"device tree: {len(found)} fixed-clock node(s) all at {distinct[0]} Hz "
                        f"({', '.join(sorted(found))})"}


def _address_map(gs: Path) -> dict[str, Any] | None:
    """The SoC address map from the elaborated design's ``*.memmap.json``.

    Every MMIO/memory region the design actually exposes, with the device-tree name it was elaborated
    under. This is the ONLY sound source for a device address on this target: a literal would bake in
    one config's map, and the facts block carried no address information at all, which is how the
    device console came to store into an address the SoC does not map.

    Fails closed (``None``) when the elaboration output is absent -- a missing map must read as
    UNKNOWN, never as an empty map that a consumer could mistake for "the SoC has no devices".
    """
    for p in sorted(gs.glob("*.memmap.json")):
        try:
            doc = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        regions = []
        for e in doc.get("mapping") or []:
            names = e.get("names") or []
            base = (e.get("base") or [None])[0]
            size = (e.get("size") or [None])[0]
            if base is None or size is None or not names:
                continue
            regions.append({
                "name": str(names[0]), "base": int(base), "size": int(size),
                "r": bool((e.get("r") or [False])[0]), "w": bool((e.get("w") or [False])[0]),
            })
        if regions:
            return {"regions": sorted(regions, key=lambda r: r["base"]),
                    "config": gs.name.rpartition(".")[2],
                    "evidence": f"elaborated address map {p.name}: {len(regions)} region(s)"}
    return None


def _console_device(gs: Path, amap: dict[str, Any] | None) -> dict[str, Any] | None:
    """The device the elaborated design designates as its console, from the device tree's
    ``stdout-path``.

    Parsed STRUCTURALLY (partition/split, never a regex, per the repo's no-regex rule): ``stdout-path``
    names a label (``&L44``); the label is then defined as ``L44: serial@10020000``. Resolving through
    the label -- rather than assuming a node called "serial" -- is what keeps this target-agnostic.

    Returns ``None`` when the dts is absent or names no console, so a consumer fails closed instead of
    guessing an address.
    """
    label = None
    for p in sorted(gs.glob("*.dts")):
        text = _read(p)
        if not text:
            continue
        for line in text.splitlines():
            if "stdout-path" in line and "&" in line:
                _, _, rest = line.partition("&")
                label = rest.strip().rstrip(";").strip()
                break
        if not label:
            continue
        # the label's own definition line: "<label>: <node>@<hexbase> {"
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped.startswith(f"{label}:"):
                continue
            node = stripped.partition(":")[2].strip().partition("{")[0].strip()
            name, _, base_hex = node.partition("@")
            if not base_hex:
                continue
            try:
                base = int(base_hex, 16)
            except ValueError:
                continue
            size = None
            for r in ((amap or {}).get("regions") or []):
                if r["base"] == base:
                    size = r["size"]
                    break
            return {"node": node, "device": name.strip(), "base": base, "size": size,
                    "label": label,
                    "evidence": f"{p.name}: chosen/stdout-path -> &{label} -> {node}"}
    return None


# -------------------------------------------------------------------------- cross-check (NOT source)
def _perf_model_geometry() -> dict[str, Any]:
    """The cyclotron perf model's declared geometry — a CROSS-CHECK, never a source.

    Returned separately from the facts so no consumer can mistake it for a hardware reading. When it
    agrees with the RTL that is a useful confirmation; when it disagrees (it does: ``num_cores = 2``
    against one elaborated ``MuonCore``) the disagreement is the finding."""
    p = config_path()  # fact-source-ok: this function returns a cross_check record, never a fact
    if p is None:
        return {"state": ABSENT, "values": {}, "path": None,
                "evidence": "MERLIN_MUON_CONFIG unset — no perf-model cross-check "  # fact-source-ok: naming the unset var in a diagnostic, not reading it
                            "available"}
    text = _read(p)
    if not text:
        return {"state": ABSENT, "values": {}, "path": str(p),
                "evidence": f"perf-model config unreadable at {p}"}
    import tomllib
    try:
        table = tomllib.loads(text).get("muon", {})
    except tomllib.TOMLDecodeError as e:
        return {"state": UNDETERMINABLE, "values": {}, "path": str(p),
                "evidence": f"perf-model config at {p} is not valid TOML: {e}"}
    vals = {k: int(table[k]) for k in ("num_lanes", "num_warps", "num_cores", "num_regs")
            if isinstance(table.get(k), int)}
    return {"state": DERIVED if vals else UNDETERMINABLE, "values": vals, "path": str(p),
            "evidence": f"cyclotron perf model config {p.name} [muon]: {vals or 'no geometry keys'}"}


def _cross_check(fact: dict[str, Any], key: str, xchk: dict[str, Any]) -> None:
    """Annotate ``fact`` with the perf model's reading of ``key`` — agreement or disagreement.

    Mutates in place and NEVER supplies a value: a fact with no RTL reading stays ``None``. That is the
    whole point — the perf model may confirm the hardware, it may not stand in for it."""
    declared = xchk.get("values", {}).get(key)
    if declared is None:
        return
    got = fact.get("value")
    fact["cross_check"] = {
        "source": "cyclotron perf model config (NOT RTL)", "key": key, "value": declared,
        "agrees": (got == declared) if got is not None else None,
    }


def _where(gs: Path | None) -> str:
    """A path for an evidence string, or a phrase saying there was no path at all — so an "absent"
    reading never reads as ``under None``."""
    return str(gs) if gs is not None else "(no elaboration directory: MERLIN_CHIPYARD is unset)"


def _fact(value: Any, state: str, evidence: str) -> dict[str, Any]:
    return {"value": value, "state": state, "evidence": evidence}


def _weakest(states) -> str:
    """The weakest of several states — a block is only ``derived`` when every part of it was read."""
    seen = set(states)
    if ABSENT in seen:
        return ABSENT
    if UNDETERMINABLE in seen:
        return UNDETERMINABLE
    return DERIVED if seen else ABSENT


# ------------------------------------------------------------------------------------ the ISA block
def _isa_block() -> dict[str, Any]:
    """The ISA facts, DERIVED from the mlc ``isa_encoding`` fact (the RTL-decoder field layout + opcode
    table + address-space macros) rather than a hand-copied list. Encoding width, the src/dst operand
    counts, predication, the opcode classes, and the address spaces all come from the derived fact; the
    intrinsic-helper names stay from the HW's own headers.

    When the fact is not derived (mlc/CIRCT absent) the block is ABSENT — every field ``None``. The
    previous version substituted the ISA-doc reading (``encoding_bits=64``, ``4`` src, ``1`` dst,
    predication ``True``) as a "fallback", which is the same failure as the geometry defaults one level
    up: an unfalsifiable number wearing a fact's clothes. The doc values are not even right — the
    RTL-derived reading is ``max_src_operands=3`` (the trace bundle carries three source-register
    slots), not the 4 the doc claims. A consumer that needs the encoding and cannot get it must see
    that, and :func:`merlin.targetgen.rtl.mlc_bridge.simt_facts` already returns ``{}`` on a missing
    ``encoding_bits`` so the manifest deriver falls back to the family default honestly.
    """
    intrinsics = {"smem": ["store_shared", "load32_shared", "load16_shared", "store64_shared"],
                  "sync": ["mu_barrier", "mu_fence_smem", "mu_fence"],
                  "simt": ["vx_thread_id", "vx_warp_id", "vx_core_id", "vx_split", "vx_join", "vx_tmc"]}
    try:
        from merlin.targetgen.rtl import mlc_bridge
        fact = mlc_bridge.isa_encoding_for(TARGET)
    except Exception:  # noqa: BLE001 — mlc/cache absent
        fact = None
    if fact and fact.get("fields"):
        fields = fact.get("fields", {})
        return {
            "encoding_bits": fact.get("inst_width"),
            "max_src_operands": sum(1 for k in fields if k.startswith("rs")),
            "max_dst_operands": sum(1 for k in fields if k == "rd"),
            "predicated_execution": "pred" in fields,
            "instruction_classes": sorted(fact.get("opcodes", {})),
            "address_spaces": fact.get("address_spaces", {}),
            "intrinsics": intrinsics,
            "state": DERIVED,
            "evidence": "DERIVED from mlc isa_encoding fact (muon_isa.json): RTL-decoder field layout + "
                        "opcode table + address-space macros; intrinsic names from lib/include/*intrinsics.h",
        }
    return {
        "encoding_bits": None, "max_src_operands": None, "max_dst_operands": None,
        "predicated_execution": None, "instruction_classes": [], "address_spaces": {},
        "intrinsics": intrinsics,
        "state": ABSENT,
        "evidence": "isa_encoding fact not derived (mlc/CIRCT absent — set MERLIN_MLC_DIR): encoding "
                    "UNKNOWN. No ISA-doc substitute: the doc's 4-src reading contradicts the RTL's 3.",
    }


# ------------------------------------------------------------------------------------------ assembly
def _geometry(gs: Path | None, hier: dict | None, core_io: str | None,
              xchk: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """lanes / warps / cores, each DERIVED from the elaboration or explicitly unknown."""
    if gs is None:
        why = ("no elaboration to read: MERLIN_CHIPYARD is unset, or its "
               f"sims/vcs/generated-src/…{VCS_CONFIG} tree is not present")
        out = {k: _fact(None, ABSENT, why) for k in ("lanes_per_warp", "warps_per_core", "cores")}
        for key, name in (("lanes_per_warp", "num_lanes"), ("warps_per_core", "num_warps"),
                          ("cores", "num_cores")):
            _cross_check(out[key], name, xchk)   # the model's reading is still worth recording
        return out
    if core_io is None:
        why = (f"no {MOD_CORE} 'output io' bundle in the elaborated FIRRTL under {_where(gs)} — "
               "geometry UNKNOWN")
        lanes = _fact(None, ABSENT, why)
        warps = _fact(None, ABSENT, why)
    else:
        # lanes: the thread mask has exactly one bit per lane, so its FIRRTL width IS the lane count.
        w = _uint_width(core_io, SIG_TMASK)
        lanes = (_fact(w, DERIVED, f"{MOD_CORE} io trace bits {SIG_TMASK} : UInt<{w}> — one mask bit "
                                   f"per lane")
                 if w else _fact(None, UNDETERMINABLE,
                                 f"{MOD_CORE} io bundle carries no '{SIG_TMASK} : UInt<..>' field"))
        # warps: the per-warp perf-counter bundle is instantiated once per warp.
        n = _bundle_cardinality(core_io, SIG_PERWARP)
        if n:
            warps = _fact(n, DERIVED,
                          f"{MOD_CORE} io perf {SIG_PERWARP} : {{...}}[{n}] — one counter set per warp")
            bits = _uint_width(core_io, SIG_WARPID)
            if bits is not None:
                # An independent RTL reading of the same fact: the warp index field must address them
                # all. Recorded as a bound, not as the value — a width of 3 admits <= 8 warps.
                warps["rtl_cross_check"] = {
                    "signal": f"{SIG_WARPID} : UInt<{bits}>", "max_warps": 1 << bits,
                    "consistent": n <= (1 << bits)}
        else:
            warps = _fact(None, UNDETERMINABLE,
                          f"{MOD_CORE} io bundle carries no '{SIG_PERWARP} : {{...}}[N]' vector")
    counts = _instance_counts(hier)
    if hier is None:
        cores = _fact(None, ABSENT,
                      f"no top_module_hierarchy.json under {_where(gs)} — core count UNKNOWN")
    elif MOD_CORE in counts:
        cores = _fact(counts[MOD_CORE], DERIVED,
                      f"elaborated instance tree: {counts[MOD_CORE]} x {MOD_CORE} "
                      f"({counts.get(MOD_TILE, 0)} x {MOD_TILE}, {counts.get(MOD_CLUSTER, 0)} x "
                      f"{MOD_CLUSTER})")
    else:
        cores = _fact(None, UNDETERMINABLE,
                      f"elaborated hierarchy contains no {MOD_CORE} instance")
    _cross_check(lanes, "num_lanes", xchk)
    _cross_check(warps, "num_warps", xchk)
    _cross_check(cores, "num_cores", xchk)
    return {"lanes_per_warp": lanes, "warps_per_core": warps, "cores": cores}


def _flatten(spec: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """``{name: {value, state, ...}}`` -> the flat ``{name: value}`` shape consumers already read,
    plus a per-field ``provenance`` sidecar so the state and the evidence survive into the artifact.

    Kept flat on purpose: :func:`mlc_bridge.simt_facts` and ``gen_muon_digest`` index ``simt['cores']``
    directly, and an unknown must arrive there as ``None`` (which their ``isinstance(..., int)`` guards
    already reject) rather than as a shape they cannot read."""
    out: dict[str, Any] = {k: v["value"] for k, v in spec.items()}
    out["provenance"] = {k: {kk: vv for kk, vv in v.items() if kk != "value"}
                         for k, v in spec.items()}
    return out


def build_facts() -> dict[str, Any]:
    """RTL-derived Muon facts. Every number below is read out of the elaboration or is ``None``."""
    gs = _gen_src_dir()
    rtl_present = bool(gs and gs.is_dir())
    hier = _hierarchy(gs) if rtl_present else None
    scan = _scan_fir(gs) if rtl_present else {"core_io": None, "smem_line": None}
    mems = _mems_conf(gs) if rtl_present else {}
    dt = _dt_json(gs) if rtl_present else None
    amap = _address_map(gs) if rtl_present else None
    console = _console_device(gs, amap) if rtl_present else None
    xchk = _perf_model_geometry()

    geo = _geometry(gs if rtl_present else None, hier, scan["core_io"], xchk)
    lanes, warps, cores = (geo["lanes_per_warp"]["value"], geo["warps_per_core"]["value"],
                           geo["cores"]["value"])
    simt = _flatten(geo)
    simt["threads_per_core"] = lanes * warps if (lanes and warps) else None
    # The block-level state the bundle adapter reads (mlc_bridge._simt_field). Aggregated from the
    # parts, not asserted: the geometry as a whole is only derived when EVERY dimension was read out of
    # the elaboration. A partial reading reports the weakest state it contains, so a consumer never
    # sees "derived" over a block one of whose numbers is missing.
    simt["state"] = _weakest(v["state"] for v in geo.values())
    simt["evidence"] = "; ".join(f"{k}: {v['evidence']}" for k, v in geo.items())

    # Register budget. arch_max is an RTL reading: the trace bundle's register-index field must address
    # every architectural register, so its width fixes the file size. It is resolved INSIDE the ``regs``
    # bundle (see _bundle_body): read off the top level it matched imem's address port and published
    # arch_max = 4294967296.
    regs_body = _bundle_body(scan["core_io"] or "", SIG_REGS_BUNDLE)
    regs_bits = _uint_width(regs_body or "", SIG_REG_ADDR)
    registers: dict[str, Any] = {
        "arch_max": (1 << regs_bits) if regs_bits else None,
        # NOT a hardware fact and never presented as one: a toolchain policy the Muon compiler applies,
        # carried here because the advisory checker's occupancy rule needs it. Labelled with its own
        # non-RTL source so it can never be mistaken for something read out of the elaboration.
        "compiler_limit": COMPILER_REG_LIMIT,
        "compiler_limit_source": "Radiance muon.md toolchain doc — a COMPILER policy, not an RTL fact",
        "state": DERIVED if regs_bits else (ABSENT if scan["core_io"] is None else UNDETERMINABLE),
        "evidence": (f"{MOD_CORE} io trace {SIG_REGS_BUNDLE} {SIG_REG_ADDR} : UInt<{regs_bits}> -> "
                     f"{1 << regs_bits} architectural registers"
                     if regs_bits else
                     f"no '{SIG_REG_ADDR} : UInt<..>' in the {MOD_CORE} {SIG_REGS_BUNDLE} bundle — "
                     f"register file size UNKNOWN"),
    }
    declared_regs = xchk.get("values", {}).get("num_regs")
    if declared_regs is not None:
        registers["cross_check"] = {"source": "cyclotron perf model config (NOT RTL)",
                                    "key": "num_regs", "value": declared_regs,
                                    "agrees": (registers["arch_max"] == declared_regs
                                               if registers["arch_max"] is not None else None)}

    smem = _shared_memory_bytes(hier, mems) if rtl_present else {
        "bytes_per_cluster": None, "state": ABSENT,
        "evidence": "no elaboration on this machine — shared-memory capacity UNKNOWN"}
    smem["rtl_bank_evidence"] = scan["smem_line"] or "no 'smem mem : UInt<..>' line in the FIRRTL"

    clk = _clock_hz(dt)
    clock_hz = clk["clock_hz"]
    peak_flops_cycle = (cores * lanes * 2) if (cores and lanes) else None
    fp: dict[str, Any] = {
        "dtype": "f32", "flop_per_fma": 2,
        "peak_flops_per_cycle": peak_flops_cycle,
        "clock_hz": clock_hz,
        "peak_gflops": (peak_flops_cycle * clock_hz / 1e9)
                       if (peak_flops_cycle and clock_hz) else None,
        "state": DERIVED if (peak_flops_cycle and clock_hz) else (
            clk["state"] if peak_flops_cycle else geo["cores"]["state"]),
        "clock_evidence": clk["evidence"],
        "evidence": (f"{cores} cores x {lanes} lanes x 2 flop/FMA = {peak_flops_cycle} flop/cycle "
                     f"@ {clock_hz / 1e6:g} MHz = {peak_flops_cycle * clock_hz / 1e9:g} GFLOP/s"
                     if (peak_flops_cycle and clock_hz) else
                     "peak UNKNOWN: it is the product of derived geometry and the derived clock, and "
                     f"at least one is not derived (cores={cores}, lanes={lanes}, clock_hz={clock_hz})"),
    }

    facts = {
        "schema_version": "1.1",
        "generator": {
            "name": "merlin.targets.muon.backend.muon_introspect",
            "version": "muon-introspect-v2",
            "method": (f"{VCS_CONFIG} elaborated FIRRTL type widths + module-instance hierarchy + "
                       "SRAM macro list + device tree (CIRCT/firtool output only); cyclotron perf-model "
                       "config used as a CROSS-CHECK, never as a fact source"),
        },
        "inputs": {
            "rtl_generated_src": str(gs) if gs else None,
            "rtl_present": rtl_present,
            "perf_model_config": xchk.get("path"),
            "perf_model_role": "cross-check only (not a fact source)",
        },
        "cross_check": xchk,
        "facts": {
            "simt": simt,
            "registers": registers,
            "shared_memory": smem,
            # The SoC address map and the console device it designates. Absent from this facts block
            # until recently, which is why the device console had no derived address to target and kept
            # storing to the Vortex IO_COUT_ADDR aperture -- an address this SoC maps no device at, so
            # every OUT/DONE byte was dropped and every RTL grade came back completion-only.
            # UNKNOWN (null) when the elaboration output is absent: fail closed, never guess an address.
            "address_map": amap or {"regions": None, "state": ABSENT,
                                    "evidence": f"no *.memmap.json under {_where(gs)} — "
                                                f"address map UNKNOWN"},
            "console": console or {"base": None, "state": ABSENT,
                                   "evidence": f"no *.dts stdout-path under {_where(gs)} — "
                                               f"console UNKNOWN"},
            "fp_datapath": fp,
            "isa": _isa_block(),
        },
        "rtl_hierarchy_counts": {m: _instance_counts(hier).get(m, 0)
                                 for m in (MOD_CORE, MOD_TILE, MOD_CLUSTER, MOD_SHARED_MEM)},
    }
    return facts


def default_facts_path() -> Path:
    # muon has no hand-curated merlin/targets/muon; the resolver routes it to artifacts/targets/muon
    # (or $MERLIN_RTL_FACTS). Muon uses a distinct filename (muon_facts.json) next to the pin.
    from merlin.targetgen.rtl.facts import rtl_facts_path
    return rtl_facts_path(TARGET).with_name("muon_facts.json")


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="extract Muon RTL facts -> muon_facts.json")
    ap.add_argument("--out", default=None)
    ap.add_argument("--allow-downgrade", action="store_true",
                    help="permit a regeneration that HOLLOWS OUT facts the existing artifact carries. "
                         "Only for a genuine hardware change that really lost them — a hollowed fact is "
                         "otherwise the signature of a missing extractor (see MERLIN_MLC_DIR).")
    a = ap.parse_args(argv)
    facts = build_facts()
    out = Path(a.out) if a.out else default_facts_path()
    # Guarded write: an extraction that silently lost the isa block looks identical to a good one on the
    # way out, and this is the path a person runs by hand (ensure_facts' _warn_if_degraded does not cover
    # it). Refuse the downgrade here rather than leave it to be caught by eye.
    from merlin.targetgen.rtl.facts import FactsDowngrade, write_facts_guarded
    try:
        write_facts_guarded(out, facts, allow_downgrade=a.allow_downgrade)
    except FactsDowngrade as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    f = facts["facts"]
    s, fp = f["simt"], f["fp_datapath"]

    def _n(v: Any) -> str:
        return "UNKNOWN" if v is None else f"{v:g}" if isinstance(v, float) else str(v)

    print(f"wrote {out}")
    print(f"  simt: {_n(s['cores'])} cores x {_n(s['warps_per_core'])} warps x "
          f"{_n(s['lanes_per_warp'])} lanes; peak {_n(fp['peak_gflops'])} GFLOP/s; "
          f"rtl_present={facts['inputs']['rtl_present']}")
    disagree = [f"{k}={v['cross_check']['value']} (RTL {v.get('value')})"
                for k, v in facts["facts"]["simt"]["provenance"].items()
                if isinstance(v, dict) and v.get("cross_check", {}).get("agrees") is False]
    if disagree:
        print(f"  CROSS-CHECK DISAGREES with the cyclotron perf model: {', '.join(disagree)}",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
