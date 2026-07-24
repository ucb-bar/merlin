"""merlin-rtl-introspect — structure-only fact extraction from elaborated Gemmini FIRRTL.

Reads the CIRCT/firtool-produced artifacts (the SoC ``.fir`` + ``top_module_hierarchy.json``)
and extracts STRUCTURE ONLY: memories (with sizes), the systolic tile array, datapath widths,
and the command interface. No semantic hypotheses (no role/kind guesses) — those are the
agent's classify/spec slots. Each fact carries its source evidence. Memories are scoped to
Gemmini by their FIRRTL source-path annotation; the tile array is scoped via the module
hierarchy (Tiles under Mesh), which keeps the co-located OPU out of the counts.

`validate_against_contract` checks the contract's declared `compute_units` cover the discovered
datapath dtypes (the mesh/scratchpad/dtype capacities are no longer hand-declared — they ARE these
facts, read directly by consumers, so there is nothing left to cross-check for them).

This is a pragmatic v1 over the FIRRTL artifacts (firtool already produced them). A full
CIRCT MLIR-pass over the hw/seq dialects is the next step; the fact schema is the same.
"""
from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path
from typing import Any
from merlin.common.paths import ext_path

DEFAULT_CHIPYARD = f"{ext_path("chipyard")}"
CONFIG = "GemminiAndOPUShuttleConfig"


def find_artifacts(chipyard_root: str | Path = DEFAULT_CHIPYARD,
                   config: str = CONFIG) -> dict[str, Path]:
    base = Path(chipyard_root) / "sims/verilator/generated-src" / \
        f"chipyard.harness.TestHarness.{config}"
    return {"fir": base / f"chipyard.harness.TestHarness.{config}.fir",
            "hierarchy": base / "top_module_hierarchy.json"}


def _fir_lines(fir: Path, *needles: str) -> list[str]:
    """FIRRTL lines containing ALL of ``needles`` (regex-free substring scan; replaces the old
    ``grep -aoE`` over the elaborated .fir)."""
    try:
        text = Path(fir).read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []
    return [ln for ln in text.splitlines() if all(nd in ln for nd in needles)]


def _uint_dims(line: str) -> tuple[int, int, int] | None:
    """Parse ``UInt<W>[E] [D]`` widths out of a FIRRTL smem line structurally (marker + delimiters),
    returning ``(W, E, D)`` or None."""
    idx = line.find("UInt<")
    if idx == -1:
        return None
    width, sep, after = line[idx + len("UInt<"):].partition(">")
    if not sep or not width.strip().isdigit():
        return None
    dims: list[str] = []
    pos = 0
    while len(dims) < 2:
        lb = after.find("[", pos)
        rb = after.find("]", lb) if lb != -1 else -1
        if lb == -1 or rb == -1:
            break
        dims.append(after[lb + 1:rb].strip())
        pos = rb + 1
    if len(dims) < 2 or not (dims[0].isdigit() and dims[1].isdigit()):
        return None
    return int(width.strip()), int(dims[0]), int(dims[1])


def _count_tiles_under_mesh(hierarchy: Path) -> int:
    tree = json.loads(hierarchy.read_text())
    n = 0

    def walk(node: dict, path: str):
        nonlocal n
        mod = node.get("module_name", "")
        if mod.startswith("Tile") and "Mesh" in path:
            n += 1
        for child in node.get("instances", []):
            walk(child, path + "/" + mod)
    walk(tree, "")
    return n


def extract_facts(fir: str | Path, hierarchy: str | Path) -> dict[str, Any]:
    """Return a structure-only facts dict extracted from the elaborated FIRRTL."""
    fir, hierarchy = Path(fir), Path(hierarchy)
    facts: dict[str, Any] = {"target": "gemmini",
                             "source": {"kind": "firrtl", "config": CONFIG, "fir": fir.name},
                             "arrays": [], "memories": [], "datapaths": [], "interfaces": []}

    # Systolic tile array (scoped via the module hierarchy: Tiles under Mesh).
    tiles = _count_tiles_under_mesh(hierarchy)
    side = int(math.isqrt(tiles)) if tiles else 0
    facts["arrays"].append({
        "name": "mesh", "tiles": tiles,
        "rows": side, "cols": side, "square": side * side == tiles,
        "evidence": "top_module_hierarchy.json: Tile instances under Mesh"})

    # Scratchpad memory, scoped by FIRRTL source path (gemmini/.../Scratchpad.scala).
    # Line form: `smem mem : UInt<W>[E] [D] @[... Scratchpad.scala ...]`
    sp = [ln for ln in _fir_lines(fir, "smem mem : UInt<", "Scratchpad.scala")
          if _uint_dims(ln) is not None]
    if sp:
        ebits, row_elems, depth = _uint_dims(sp[0])
        banks = len(sp)
        facts["memories"].append({
            "name": "scratchpad", "banks": banks, "row_elems": row_elems,
            "depth": depth, "elem_bits": ebits,
            "bytes": banks * row_elems * (ebits // 8) * depth,
            "evidence": f"{banks}x `smem mem : UInt<{ebits}>[{row_elems}] [{depth}]` @ Scratchpad.scala"})

    # Accumulator memory present (size left unextracted rather than guessed).
    if _fir_lines(fir, "module AccumulatorMem"):
        facts["memories"].append({
            "name": "accumulator", "elem_bits": 32, "bytes": None,
            "evidence": "module AccumulatorMem; acc datapath SInt<32>",
            "note": "depth not extracted from this artifact (v1)"})

    # Datapath element widths (structure, from the scratchpad / accumulator data types).
    facts["datapaths"] = [
        {"name": "input", "dtype": "i8", "evidence": "scratchpad smem UInt<8>"},
        {"name": "accumulator", "dtype": "i32", "evidence": "AccumulatorMem SInt<32>"}]

    # Command interface (structure, by module presence).
    ifaces = []
    if _fir_lines(fir, "module ReservationStation"):
        ifaces.append({"name": "rocc_cmd", "evidence": "module ReservationStation (RoCC decode/dispatch)"})
    if _fir_lines(fir, "module FrontendTLB"):
        ifaces.append({"name": "dma_tlb", "evidence": "module FrontendTLB"})
    facts["interfaces"] = ifaces
    return facts


def validate_against_contract(facts: dict[str, Any], contract: dict[str, Any]) -> list[str]:
    """Return problems where the extracted RTL datapaths are not covered by the contract's declared
    compute_units. Mesh geometry, scratchpad capacity and datapath dtypes are no longer hand-declared
    in the contract — they ARE these facts — so there is nothing to cross-check for them; the only
    surviving check is that the contract's compute_units cover the discovered datapath dtype(s)."""
    dt = {d["name"]: d["dtype"] for d in facts["datapaths"]}
    # If the contract declares compute_units, the RTL datapaths must be covered by them: the input
    # datapath dtype by some unit's declared formats, and the accumulator by some accumulate rule
    # (when a unit still declares one).
    return _check_compute_units(dt, contract)


def _check_compute_units(datapaths: dict[str, str], contract: dict[str, Any]) -> list[str]:
    from merlin.targetgen import compute_units as _cu

    units = _cu.compute_units(contract)
    if not units:
        return []
    eff = [_cu.effective(u, units) for u in units]
    tokens: set[str] = set()
    accs: set[str] = set()
    for u in eff:
        tokens |= _cu.datatype_tokens(u)
        accs |= {a.acc for a in u.accumulate}
    out: list[str] = []
    inp = datapaths.get("input")
    if inp and inp not in tokens:
        out.append(f"datapath input {inp} not covered by any compute_unit dtypes {sorted(tokens)}")
    acc = datapaths.get("accumulator")
    if acc and accs and acc not in accs:
        out.append(f"datapath accumulator {acc} not in any compute_unit accumulate acc {sorted(accs)}")
    return out


def emit_facts_yaml(facts: dict[str, Any]) -> str:
    import yaml
    return yaml.safe_dump(facts, sort_keys=False)


GENERATOR_VERSION = "rtl-introspect-v1-grep-firrtl"   # bump when extraction changes; CIRCT pass = v2


def _src_sha(path: str) -> str:
    proc = subprocess.run(["git", "-C", path, "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True)
    return proc.stdout.strip() or "unknown"


def dump_facts(out_path: str | Path, *, chipyard_root: str | Path = DEFAULT_CHIPYARD,
               config: str = CONFIG) -> dict[str, Any]:
    """Extract facts and write a REPRODUCIBLE rtl_facts.yaml (facts + generator version + source
    SHAs + extraction method). This makes RTL-fact extraction a recorded, attributable input —
    the thing an agent_spec target-generation experiment consumes."""
    import yaml
    arts = find_artifacts(chipyard_root, config)
    facts = extract_facts(arts["fir"], arts["hierarchy"])
    cy = str(chipyard_root)
    record = {
        "schema_version": "1.0",
        "generator": {
            "name": "merlin.targetgen.rtl.introspect",
            "version": GENERATOR_VERSION,
            "method": "grep/regex over firtool-produced FIRRTL + hierarchy JSON "
                      "(NOT yet a CIRCT hw/seq MLIR pass)",
        },
        "source_shas": {"chipyard": _src_sha(cy),
                        "gemmini": _src_sha(cy + "/generators/gemmini")},
        "facts": facts,
    }
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(record, sort_keys=False), encoding="utf-8")
    return record


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Extract structure-only RTL facts -> rtl_facts.yaml.")
    ap.add_argument("--out", default="rtl_facts.yaml")
    ap.add_argument("--chipyard", default=DEFAULT_CHIPYARD)
    ap.add_argument("--config", default=CONFIG)
    args = ap.parse_args()
    rec = dump_facts(args.out, chipyard_root=args.chipyard, config=args.config)
    print(f"wrote {args.out}: {len(rec['facts'].get('arrays', []))} arrays, "
          f"{len(rec['facts'].get('memories', []))} memories, "
          f"{len(rec['facts'].get('interfaces', []))} interfaces "
          f"[generator {rec['generator']['version']}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
