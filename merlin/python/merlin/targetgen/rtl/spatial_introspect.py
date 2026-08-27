"""Deterministic, no-LLM extraction of SPATIAL tensor-tile hardware facts from the real RTL.

The spatial-tile analog of :mod:`merlin.targetgen.rtl.circt_introspect` (systolic/Gemmini) and
:mod:`merlin.targetgen.rtl.muon_introspect` (SIMT/Muon). Its first instance is the **Saturn
OuterProductUnit (OPU)** — a non-systolic spatial tensor tile: a ``cluster x cell`` grid of accumulator
cells that computes a matmul as a sequence of rank-1 (outer-product) accumulate steps, NOT a
stationary-weight systolic wavefront. So the geometry is a *tile of cells*, and the facts are the tile
dimensions, the per-cell multi-register-file (MRF) depth, the operand/accumulator element widths, the
int8 + FP8 datapaths, and the FP8 FMA pipeline latency.

Two properties make this its own fact family rather than a reuse of the systolic path:

* **Geometry comes from the state manifest cell count, NOT ``discover_mesh_dim``.** The mesh-DIM
  discovery mis-derives ``DIM=4`` from the OPU's two-level ``cluster x cell`` hierarchy (it sees the
  4x4 cluster layer). The output tile is ``(l_groups*l_cells) x (t_groups*t_cells)`` — e.g. 16x16 = 256
  cells for ``mxv256d128`` — read from the ``io_op_in_l_<g>_<i>`` / ``io_op_in_t_<g>_<j>`` operand ports
  and the ``clusters_*/cells_*/regs_*`` accumulator registers (the same geometry ``mlc``'s
  ``cosim_opu._discover_geometry`` reads).
* **There is NO RoCC funct decode**, so this is STRUCTURAL ONLY — no arc effect-probe, no
  ``derive_and_cache_roles`` (which raises for the OPU). The behavioural effect probe (op categories by
  measured datapath effect) is a later mlc-branch task; here every fact is a static structural read.

The hw.mlir + state manifest are resolved through ``mlc``'s per-target fingerprint map
(:func:`merlin.targetgen.rtl.mlc_bridge.opu_artifact_paths`), which knows the two-level
``runs/circt-arc/saturn_opu/<config>/outputs`` layout. Facts land in the PURGEABLE cache
``out/artifacts/cache/rtl_introspect/<target>/facts.json`` (like the systolic path), each field carrying
``{value, derived, source, evidence}`` — a field that cannot be grounded is reported ``derived=False``
(value None), never guessed.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import circt_introspect as CI
from .facts import rtl_cache_dir

GENERATOR_VERSION = "spatial-introspect-v1-opu"


# ------------------------------------------------------------------- arc state-manifest geometry parse
def _load_state_names(manifest_path: Path) -> tuple[str, list[dict], list[str]]:
    """Parse an arc state manifest -> (top-module name, states list, state names).

    The manifest is a JSON list whose single element is the top module dict
    ``{"name", "numStateBytes", "states": [...]}`` (mirrors ``mlc``'s ``CosimCore.manifest`` /
    ``cosim_opu``). Each state carries ``{name, offset, numBits, type}``."""
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    top = data[0] if isinstance(data, list) and data else data
    states = top.get("states", []) if isinstance(top, dict) else []
    names = [s["name"] for s in states if isinstance(s, dict) and "name" in s]
    return (top.get("name", "") if isinstance(top, dict) else ""), states, names


def _grid_from_operand_ports(names: list[str]) -> dict[str, int] | None:
    """The output-tile geometry from the ``io_op_in_l_<g>_<i>`` / ``io_op_in_t_<g>_<j>`` operand ports.

    ``in_l`` drives the tile rows (left/stationary operand columns), ``in_t`` the tile cols (top operand
    rows); each is a ``<group>_<cell>`` grid. Output tile = ``(l_groups*l_cells) x (t_groups*t_cells)``.
    Structural split on ``_`` (no regex): ``io_op_in_l_<g>_<i>`` -> ['io','op','in','l','<g>','<i>'].
    Returns None if the operand ports are absent (not an OPU-shaped manifest)."""
    il = [n for n in names if n.startswith("io_op_in_l_")]
    it = [n for n in names if n.startswith("io_op_in_t_")]
    if not il or not it:
        return None
    l_groups = 1 + max(int(n.split("_")[4]) for n in il)
    l_cells = 1 + max(int(n.split("_")[5]) for n in il)
    t_groups = 1 + max(int(n.split("_")[4]) for n in it)
    t_cells = 1 + max(int(n.split("_")[5]) for n in it)
    return {"l_groups": l_groups, "l_cells": l_cells, "t_groups": t_groups, "t_cells": t_cells,
            "rows": l_groups * l_cells, "cols": t_groups * t_cells}


def _mrf_depth(names: list[str]) -> int | None:
    """Per-cell multi-register-file depth = number of ``.../regs_<idx>`` banks (regs_0..regs_{n-1})."""
    idxs = [int(n.split("/regs_")[1]) for n in names if "/regs_" in n and n.split("/regs_")[1].isdigit()]
    return (1 + max(idxs)) if idxs else None


def _cell_count(names: list[str]) -> int:
    """Number of accumulator cells = number of ``.../cells_<r>_<c>/regs_0`` register-0 states."""
    return sum(1 for n in names if n.endswith("/regs_0"))


def _op_categories(names: list[str]) -> list[str]:
    """The command op categories the tile exposes, from the ``io_op_<cat>_<g>`` control-port groups
    (macc / mvin / shift + the mvin broadcast variants). Structural prefix scan, no regex."""
    cats: list[str] = []
    for cat in ("macc", "mvin", "shift"):
        if any(n.startswith(f"io_op_{cat}_") or n == f"io_op_{cat}" for n in names):
            cats.append(cat)
    return cats


def _port_bits(states: list[dict], name: str) -> int | None:
    """Bit width of a named state (exact identity match), or None."""
    for s in states:
        if isinstance(s, dict) and s.get("name") == name:
            b = s.get("numBits")
            return int(b) if isinstance(b, int) else None
    return None


def _first_bits(states: list[dict], predicate) -> int | None:
    """Bit width of the first state whose name satisfies ``predicate``, or None."""
    for s in states:
        if isinstance(s, dict) and predicate(s.get("name", "")):
            b = s.get("numBits")
            if isinstance(b, int):
                return int(b)
    return None


# --------------------------------------------------------------- hw.mlir cross-checks (reuse CI parse)
def _cell_port_widths(hw_text: str) -> dict[str, int]:
    """Operand + accumulator widths from the ``@OuterProductCell`` port signature, parsed with the
    shared :func:`circt_introspect._module_port_sig` / ``_int_width`` helpers (no re-invented parser).
    Returns ``{operand_bits, accumulator_bits, mrf_idx_bits}`` for the ports present."""
    sig = CI._module_port_sig(hw_text, "OuterProductCell")
    out: dict[str, int] = {}
    if sig is None:
        return out
    for decl in (d.strip() for d in sig.split(",")):
        if " : " not in decl:
            continue
        lhs, typ = decl.rsplit(" : ", 1)
        name = lhs.split()[-1].lstrip("%")
        w = CI._int_width(typ.strip())
        if w is None:
            continue
        if name == "io_in_l":
            out["operand_bits"] = w
        elif name == "io_out":
            out["accumulator_bits"] = w
        elif name == "io_mrf_idx":
            out["mrf_idx_bits"] = w
    return out


def _fp8_fma(hw_text: str) -> dict[str, int] | None:
    """The FP8 FMA pipeline signature from the ``@MulAddRecFNPipe_l<L>_e<E>_s<S>`` module name.

    ``l<L>`` = the pipeline latency in cycles (l1 -> 1), ``e<E>_s<S>`` = the accumulate float format
    (e8_s24 -> fp32: 8 exponent + 24 significand bits). Returns ``{latency_cycles, exp_bits, sig_bits}``
    or None when the module is absent (an int8-only tile, e.g. v128d64). Structural ``find``+``split``
    on the module name — no regex."""
    marker = "@MulAddRecFNPipe_l"
    idx = hw_text.find(marker)
    if idx == -1:
        return None
    tail = hw_text[idx + len(marker):]
    tok = tail.split("(", 1)[0].strip()          # "1_e8_s24"
    parts = tok.split("_")
    if len(parts) < 3 or not parts[0].isdigit():
        return None
    lat = int(parts[0])
    exp = parts[1][1:] if parts[1].startswith("e") and parts[1][1:].isdigit() else None
    sig = parts[2][1:] if parts[2].startswith("s") and parts[2][1:].isdigit() else None
    return {"latency_cycles": lat,
            "exp_bits": int(exp) if exp else None,
            "sig_bits": int(sig) if sig else None}


# ----------------------------------------------------------------------------- fp8 format identity (derive-or-UNKNOWN)
# (exp_bits, significand_bits_including_hidden) -> canonical name. This is MATH (IEEE/OCP format
# definitions), not a per-target table — the only way to tell two same-width floats apart is the exponent
# width, so a name is emitted ONLY when the RTL's own float tokens carry that fact.
_FLOAT_BY_ES = {(4, 4): "fp8_e4m3", (5, 3): "fp8_e5m2", (5, 11): "fp16",
                (8, 8): "bf16", (8, 24): "f32", (11, 53): "f64"}


def _lead_int(s: str) -> int | None:
    """The leading run of digits of ``s`` as an int (``"24("`` -> 24), or None — structural, no regex."""
    i = 0
    while i < len(s) and s[i].isdigit():
        i += 1
    return int(s[:i]) if i else None


def _fp8_format_names(hw_text: str, width: int) -> list[str]:
    """DERIVE the float format name(s) whose total width == ``width`` from the RTL's OWN float tokens — a
    hardfloat ``e<E>_s<S>`` module name or a design ``E<d>M<d>`` token — parsed structurally (split /
    partition, no regex). Returns ``[]`` when the RTL names no format at that width: the identity is then
    UNKNOWN and must NOT be fabricated (a ``io_op_altfmt_`` select proves a second mode exists, never
    WHICH format it is)."""
    names: list[str] = []

    def add(pair: tuple[int, int]) -> None:
        nm = _FLOAT_BY_ES.get(pair)
        if nm and nm not in names:
            names.append(nm)

    for raw in hw_text.split():
        tok = raw.lstrip("@%")
        parts = tok.split("_")
        for a, b in zip(parts, parts[1:]):                       # hardfloat e<E>_s<S>
            if a[:1] == "e" and b[:1] == "s":
                e, s = _lead_int(a[1:]), _lead_int(b[1:])
                if e is not None and s is not None and e + s == width:
                    add((e, s))
        if tok[:1] == "E" and "M" in tok:                        # design E<d>M<d>
            ep, _, mp = tok.partition("M")
            e, m = _lead_int(ep[1:]), _lead_int(mp)
            if e is not None and m is not None and e + m + 1 == width:
                add((e, m + 1))
    return names


# ----------------------------------------------------------------------------- assemble the fact bundle
def _unavailable(target: str, reason: str) -> dict[str, Any]:
    """An honest all-unavailable bundle (mlc / the OPU artifacts absent) — never a fabricated tile."""
    fields = {k: {"value": None, "derived": False, "source": None, "evidence": reason}
              for k in ("tile_dim", "mrf_depth", "element_widths", "dtypes", "fma_latency",
                        "op_categories", "accum_kind")}
    return {"target": target, "method": "static OPU state-manifest + HW-dialect discovery (no model run)",
            "kind": "spatial", "fields": fields, "n_derived": 0, "reason": reason}


def build_fact_bundle(target: str) -> dict[str, Any]:
    """The STATIC, provenance-tagged spatial-tile fact bundle for ``target`` — no model run, no arc probe.

    Mirrors :func:`merlin.targetgen.rtl.mlc_bridge.target_fact_bundle` (the systolic sibling): each field
    carries ``{value, derived, source, evidence}``; a field that cannot be grounded is reported
    ``derived=False`` (value None), never guessed. The geometry is read from the state-manifest cell count
    (NOT ``discover_mesh_dim``); the hw.mlir cross-checks the widths and supplies the FP8 FMA latency.
    """
    from . import mlc_bridge
    paths = mlc_bridge.opu_artifact_paths(target)
    if not paths:
        return _unavailable(target, "mlc unavailable / target not in the OPU fingerprint map")
    man, hw = paths.get("man"), paths.get("hw")
    if man is None or not Path(man).is_file():
        return _unavailable(target, f"OPU state manifest absent ({man})")

    module, states, names = _load_state_names(Path(man))
    grid = _grid_from_operand_ports(names)
    if grid is None:
        return _unavailable(target, f"{module or target!r}: no io_op_in_l/in_t operand ports "
                                    "(not an OuterProductUnit-shaped manifest)")

    hw_text = Path(hw).read_text(errors="replace") if hw and Path(hw).is_file() else ""
    cell_w = _cell_port_widths(hw_text)
    fma = _fp8_fma(hw_text)

    # -- geometry (state-manifest cell count, NOT discover_mesh_dim) --------------------------------
    cells = _cell_count(names)
    macc_lanes = sum(1 for n in names if n.startswith("io_op_macc_"))
    tile_dim = {
        "rows": grid["rows"], "cols": grid["cols"], "dim": grid["rows"],
        "clusters": {"rows": grid["l_groups"], "cols": grid["t_groups"]},
        "cells_per_cluster": {"rows": grid["l_cells"], "cols": grid["t_cells"]},
        "cells": cells, "cluster_row_lanes": macc_lanes,
    }
    tile_ev = (f"io_op_in_l groups x cells = {grid['l_groups']}x{grid['l_cells']} (rows={grid['rows']}), "
               f"io_op_in_t = {grid['t_groups']}x{grid['t_cells']} (cols={grid['cols']}); "
               f"{cells} accumulator cells (clusters_*/cells_*/regs_0)")

    # -- MRF depth ---------------------------------------------------------------------------------
    mrf = _mrf_depth(names)
    mrf_idx_bits = cell_w.get("mrf_idx_bits") or _first_bits(states, lambda n: n.startswith("io_op_mrf_idx_"))
    mrf_ev = (f"{mrf} regs banks/cell (.../regs_0..regs_{(mrf or 1) - 1})"
              + (f"; io_op_mrf_idx:i{mrf_idx_bits}" if mrf_idx_bits else ""))

    # -- element widths ----------------------------------------------------------------------------
    operand_bits = cell_w.get("operand_bits") or _first_bits(states, lambda n: n.startswith("io_op_in_l_"))
    accum_bits = cell_w.get("accumulator_bits") or _port_bits(states, next(
        (n for n in names if n.endswith("/regs_0")), "")) or _first_bits(states, lambda n: n.endswith("/regs_0"))
    widths = {"operand_bits": operand_bits, "accumulator_bits": accum_bits}
    widths_ev = (f"@OuterProductCell io_in_l:i{operand_bits}, io_out:i{accum_bits}"
                 if cell_w else f"manifest operand:i{operand_bits}, accumulator reg:i{accum_bits}")

    # -- datapaths / dtypes ------------------------------------------------------------------------
    has_fp8 = any(n.startswith("io_op_fp8_") for n in names)
    has_altfmt = any(n.startswith("io_op_altfmt_") for n in names)
    dtypes: list[dict[str, Any]] = [{
        "name": "int8", "operand": "i8", "accumulator": f"i{accum_bits}" if accum_bits else "i32",
        "path": "combinational i8*i8 -> i32 MAC (io_op_fp8=0)"}]
    fp8_id_derived = True
    if has_fp8:
        fw = operand_bits or 8
        fp8_names = _fp8_format_names(hw_text, fw)          # RTL-named formats at the operand width, or []
        if fp8_names:
            use = fp8_names if has_altfmt else fp8_names[:1]  # altfmt makes >1 named format reachable
            for altfmt, fmt in enumerate(use):
                dtypes.append({
                    "name": fmt, "operand": fmt.split("_", 1)[1] if "_" in fmt else fmt,
                    "accumulator": "f32", "altfmt": altfmt,
                    "path": "fp8 -> fp32 widen; MulAddRecFNPipe fp32 FMA (io_op_fp8=1, "
                            f"io_op_altfmt={altfmt})"})
        else:
            # fail closed: an fp8 datapath exists, io_op_altfmt_ says how many sub-format modes, but the
            # RTL names no format at this width -> report the width-only identity, never a guessed pair.
            fp8_id_derived = False
            dtypes.append({
                "name": f"float{fw}", "operand": f"f{fw}", "accumulator": "f32",
                "modes": 2 if has_altfmt else 1, "altfmt_selectable": has_altfmt, "identity_derived": False,
                "path": f"fp8 -> fp32 widen; MulAddRecFNPipe fp32 FMA; io_op_altfmt selects "
                        f"{'2 sub-formats' if has_altfmt else '1 sub-format'} whose identity is NOT named "
                        f"in the RTL facts (UNKNOWN; not fabricating fp8_e4m3/fp8_e5m2)"})
    if not has_fp8:
        dtypes_ev = "no io_op_fp8_ ports -> int8-only tile"
    elif fp8_id_derived:
        fp8_list = [d["name"] for d in dtypes if d["name"] != "int8"]
        dtypes_ev = f"io_op_fp8_/io_op_altfmt_ ports present -> int8 + {fp8_list} (fp32 accumulate, RTL-named)"
    else:
        w = operand_bits or 8
        dtypes_ev = (f"io_op_fp8_ present -> int8 + a {w}-bit fp8 datapath with "
                     f"{'2 io_op_altfmt-selected sub-formats' if has_altfmt else '1 sub-format'}, fp32 "
                     f"accumulate; fp8 format identity NOT named in the RTL -> float{w} (UNKNOWN, not fabricated)")

    # -- FMA latency -------------------------------------------------------------------------------
    fma_val = {"int8_mac_cycles": 0,
               "fp8_fma_cycles": (fma or {}).get("latency_cycles") if has_fp8 else None}
    fma_derived = (not has_fp8) or (fma is not None)
    fma_ev = ("int8 MAC is combinational (same-cycle accumulate); "
              + (f"fp8 FMA @MulAddRecFNPipe_l{fma['latency_cycles']} "
                 f"(e{fma.get('exp_bits')}_s{fma.get('sig_bits')} = fp32)" if (has_fp8 and fma)
                 else "no fp8 datapath" if not has_fp8 else "fp8 FMA module not found in hw.mlir"))

    # -- accumulator kind --------------------------------------------------------------------------
    accum_kind = {"int8": f"i{accum_bits}" if accum_bits else "i32"}
    if has_fp8:
        accum_kind["fp8"] = "f32"
    accum_ev = (f"int8 datapath accumulates into i{accum_bits} registers"
                + ("; fp8 datapath accumulates into fp32 (RecFN e8_s24)" if has_fp8 else ""))

    fields = {
        "tile_dim": {"value": tile_dim, "derived": True,
                     "source": "OPU state-manifest cell-count geometry (NOT discover_mesh_dim)",
                     "evidence": tile_ev},
        "mrf_depth": {"value": mrf, "derived": mrf is not None,
                      "source": "per-cell regs_* bank count + io_op_mrf_idx width", "evidence": mrf_ev},
        "element_widths": {"value": widths, "derived": operand_bits is not None and accum_bits is not None,
                           "source": "@OuterProductCell HW-dialect port widths (manifest fallback)",
                           "evidence": widths_ev},
        "dtypes": {"value": dtypes, "derived": True,
                   "source": "io_op_fp8/altfmt datapath-select ports", "evidence": dtypes_ev},
        "fma_latency": {"value": fma_val, "derived": fma_derived,
                        "source": "MulAddRecFNPipe_l<L> module name + combinational int8 MAC",
                        "evidence": fma_ev},
        "op_categories": {"value": _op_categories(names), "derived": bool(_op_categories(names)),
                          "source": "io_op_<cat>_ command control-port groups",
                          "evidence": f"io_op control ports -> {_op_categories(names)}"},
        "accum_kind": {"value": accum_kind, "derived": accum_bits is not None,
                       "source": "accumulator register width + fp8 RecFN format", "evidence": accum_ev},
    }
    return {
        "target": target,
        "method": "static OPU state-manifest + HW-dialect discovery (no model run, no arc probe)",
        "kind": "spatial",
        "generator": {"name": "merlin.targetgen.rtl.spatial_introspect", "version": GENERATOR_VERSION},
        "inputs": {"module": module, "state_manifest": str(man),
                   "hw_mlir": str(hw) if hw else None, "n_states": len(names)},
        "fields": fields,
        "n_derived": sum(1 for f in fields.values() if f["derived"]),
    }


def render_fact_bundle(target: str, bundle: dict | None = None) -> str:
    """Render :func:`build_fact_bundle` as an agent-facing spatial-tile brief (Markdown). Every line is
    DERIVED and stamped with provenance; an ungrounded field is stated unavailable, never invented."""
    b = bundle or build_fact_bundle(target)
    f = b["fields"]
    n = b.get("n_derived", 0)
    lines = [f"# Spatial tensor-tile facts: {b['target']} (kind={b.get('kind', 'spatial')})",
             f"_Derived by {b['method']}. {n}/{len(f)} fields grounded; ungrounded = unavailable, "
             f"not guessed._", ""]
    td = f["tile_dim"]
    if td["derived"]:
        v = td["value"]
        lines.append(f"- **Tile**: {v['rows']}x{v['cols']} ({v['cells']} cells) = "
                     f"{v['clusters']['rows']}x{v['clusters']['cols']} clusters of "
                     f"{v['cells_per_cluster']['rows']}x{v['cells_per_cluster']['cols']} cells "
                     f"(source: {td['source']})")
    else:
        lines.append(f"- **Tile**: unavailable ({td['evidence']})")
    if f["mrf_depth"]["derived"]:
        lines.append(f"- **MRF depth**: {f['mrf_depth']['value']} banks/cell")
    ew = f["element_widths"]["value"] if f["element_widths"]["derived"] else None
    if ew:
        lines.append(f"- **Element widths**: operand i{ew['operand_bits']} -> "
                     f"accumulator i{ew['accumulator_bits']}")
    dts = f["dtypes"]["value"] if f["dtypes"]["derived"] else None
    if dts:
        lines.append(f"- **Dtypes**: {', '.join(d['name'] for d in dts)}")
    else:
        lines.append(f"- **Dtypes**: unavailable ({f['dtypes']['evidence']})")
    if f["fma_latency"]["derived"]:
        fv = f["fma_latency"]["value"]
        lines.append(f"- **FMA latency**: int8 MAC {fv['int8_mac_cycles']} cyc; "
                     f"fp8 FMA {fv['fp8_fma_cycles']} cyc")
    lines.append(f"- **Op categories**: {f['op_categories']['value'] if f['op_categories']['derived'] else 'unavailable'}")
    lines.append(f"- **Accumulator**: {f['accum_kind']['value'] if f['accum_kind']['derived'] else 'unavailable'}")
    return "\n".join(lines) + "\n"


def dump_fact_bundle(target: str, out_path: Path | str | None = None) -> dict[str, Any]:
    """Build the spatial fact bundle and write it to the PURGEABLE cache
    ``out/artifacts/cache/rtl_introspect/<target>/facts.json`` (like the systolic path). Returns the
    written bundle."""
    out = Path(out_path) if out_path is not None else rtl_cache_dir(target, ensure=True) / "facts.json"
    bundle = build_fact_bundle(target)
    # Guarded: the spatial family's facts come from the OPU state manifest, and a manifest the extractor
    # cannot see yields a well-formed bundle with the geometry hollowed out. Same ratchet as every other
    # fact-extraction family (see facts.write_facts_guarded).
    from .facts import write_facts_guarded
    write_facts_guarded(out, bundle)
    bundle["_path"] = str(out)
    return bundle


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(
        description="Emit a spatial tensor-tile (OPU) DERIVED fact bundle. Prints Markdown; --json prints "
                    "the raw provenance-tagged bundle; --dump writes the purgeable facts.json cache.")
    ap.add_argument("target", help="the OPU target id (e.g. saturn_opu_mxv256d128 / saturn_opu_v128d64)")
    ap.add_argument("--json", action="store_true", help="print the raw provenance-tagged bundle as JSON")
    ap.add_argument("--dump", action="store_true", help="write the purgeable facts.json cache")
    a = ap.parse_args(argv)
    bundle = dump_fact_bundle(a.target) if a.dump else build_fact_bundle(a.target)
    print(json.dumps(bundle, indent=2) if a.json else render_fact_bundle(a.target, bundle), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
