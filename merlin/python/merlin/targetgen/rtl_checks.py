"""RTL-derived cheap checks layer (Phase 0, ADVISORY / UN-WIRED).

A pure-static pre-screen + richer-feedback layer that consumes the **already-decoded** RoCC
instruction trace (:mod:`merlin.targetgen.rocc.decode` output) plus the target's RTL facts, and
produces graded, RTL-grounded check results the codegen agent can act on.

Why this exists (task #131, untouched half):

* The full oracle ladder (spike L2, verilator L3, FireSim L5) is *expensive*. Most broken codegen
  is broken in ways visible BEFORE simulating — a malformed ISA stream, a tile count that cannot
  cover the declared output shape, a scratchpad address past the array capacity. This layer catches
  those for free (no simulation) so a caller can skip the costly oracle (pre-screen), and gives the
  agent expected-vs-got deltas instead of a bare pass/fail (richer feedback).

NON-OVERFIT INVARIANT (project principle ``abstract-into-compiler-not-overfit``): every ``expected``
value here is derived ONLY from (i) the target RTL facts (mesh dims, scratchpad capacity), (ii) the
capsule's *declared* shapes/modes (part of the problem statement), and (iii) Gemmini ISA structural
rules. There is NO path that reads a golden trace or a per-capsule expected instruction count. These
are GENERAL compiler-level invariants of the hardware, identical for every shape — never hand-kernels
and never per-shape magic numbers.

Four of the checks are STRUCTURAL rather than numeric, and each exists because the plane that would
otherwise catch the defect reports it as something else:

* :func:`_check_output_store_coverage` — a declared output with no covering store, or a store past its
  declared extent. Those bytes are never produced, so the numeric plane reports ``mismatch_count > 0``
  with ``max_abs_error == 0``: a writeback failure wearing an arithmetic failure's clothes.
* :func:`_check_extent_tile_legalization` — a declared extent the RTL-derived array edge does not
  divide, with nothing in the stream legalizing the ragged band. Silent in both directions: the tail is
  either dropped or written past.
* :func:`_check_conv_lowering` — a declared convolution whose ``Kh*Kw`` window was never folded into the
  contraction. The stream is well-formed and fully covering; it computes a different operation.
* :func:`_check_encoded_field_intent` — a move whose DRAM base pointer is not the argument slot the
  kernel ABI puts its tensor in. A correctly-NAMED instruction carrying a wrong FIELD is not an illegal
  instruction, and both command-buffer planes (numeric + trace) read the DECLARATION — so a package
  whose command buffer is right and whose emitted encoding binds the pointers in another order passes
  both and diverges only on the oracle, as a value error. The argument order is RESOLVED from the ABI
  contract for the buffer's own command shape (there is one order per shape), never assumed.

All four are severity ``warn``, so they can never move this report's verdict to ``reject`` and can never
cost a run its oracle. This module changes no pass/fail verdict.

Run it against any existing decoded trace::

    python -m merlin.targetgen.rtl_checks <instruction_trace.json> [--capsule <capsule.yaml>]
"""
from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import Any

SCHEMA = "rtl_checks/v0"  # advisory artifact tag — deliberately NOT a frozen merlin/contract schema

# The deterministic CIRCT-extracted facts — the ONLY source (no baked gemmini-shaped fallback).
from .rtl.facts import load_facts


def load_default_facts(target: str) -> dict[str, Any]:
    """The RTL-derived facts for ``target`` (mesh, scratchpad capacity, legal funct set, custom opcode),
    flattened from the CIRCT-extracted facts.json (regenerated from the RTL on demand). FAIL-CLOSED: a
    fact that cannot be derived is left UNKNOWN (``None``) — never a gemmini-shaped default (an assumed
    16x16 mesh / 0x7B opcode / 256 KB scratchpad would silently mis-screen a non-gemmini target). The
    consuming checks SKIP any fact that is UNKNOWN. ``target`` is REQUIRED — the facts are the run's
    resolved target's facts, never an assumed default."""
    facts: dict[str, Any] = {"mesh": None, "scratchpad_bytes": None,
                             "scratchpad_rows": None, "scratchpad_row_mask": None,
                             "accumulator_rows": None,
                             "accumulator_select_bit": None, "accumulator_row_mask": None,
                             "local_address_data_width": None, "local_address_data_mask": None,
                             "config_mvout_fields": None, "max_pool_supported": None,
                             "legal_funct": None, "custom_opcode": None, "funct3": None,
                             "from": "UNKNOWN (RTL facts not derivable)"}
    try:
        rec = load_facts(target)
        f = rec.get("facts", rec)
        mesh = next((a for a in f.get("arrays", []) if a["name"] == "mesh"), None)
        if mesh:
            facts["mesh"] = [mesh["rows"], mesh["cols"]]
        sp = next((m for m in f.get("memories", []) if m["name"] == "scratchpad"), None)
        if sp and sp.get("bytes"):
            facts["scratchpad_bytes"] = sp["bytes"]
        # A local-address operand is not necessarily a flat scratchpad row number.  Derive both
        # memory capacities in the units the ISA addresses (rows) from the same CIRCT facts record;
        # address_space also accounts for the accumulator's wider element type and banked depth.
        try:
            from .address_space import derive_address_space
            space = derive_address_space(target, facts=rec)
            scratchpad = space.store("scratchpad")
            accumulator = space.store("accumulator")
            if scratchpad is not None:
                facts["scratchpad_rows"] = scratchpad.total_rows
                if isinstance(scratchpad.total_rows, int):
                    width = max(1, (scratchpad.total_rows - 1).bit_length())
                    facts["scratchpad_row_mask"] = (1 << width) - 1
            if accumulator is not None:
                facts["accumulator_rows"] = accumulator.total_rows
                if isinstance(accumulator.total_rows, int):
                    # Physical accumulator addressing uses the low ceil(log2(total_rows)) data bits.
                    width = max(1, (accumulator.total_rows - 1).bit_length())
                    facts["accumulator_row_mask"] = (1 << width) - 1
            if space.separate_accumulator_space is True:
                # The accumulator selector comes from the generated ISA constants (ultimately the
                # target contract/header's ADDR_LEN convention), never from a capsule name or bit-31
                # literal. The row mask above is derived independently from the memory capacity.
                try:
                    from .rocc.decode import isa_constants
                    isa = isa_constants(target)
                    select = isa.get("ACC_I8")
                    if isinstance(select, int):
                        facts["accumulator_select_bit"] = select
                except Exception:
                    # Capacities, row masks, and common LocalAddr.data width remain independently
                    # useful. Only the selector stays UNKNOWN.
                    pass
            # ``LocalAddr.data`` is wide enough for the largest local store. A smaller store's physical
            # address accessor selects only its own low address bits; any remaining data bits are thus
            # non-canonical payload which aliases a lower physical row. Derive that common data width
            # from the two independently-derived store address widths, not from a target literal.
            row_masks = [m for m in (facts["scratchpad_row_mask"],
                                     facts["accumulator_row_mask"])
                         if isinstance(m, int)]
            if row_masks and (space.separate_accumulator_space is not True or len(row_masks) == 2):
                facts["local_address_data_mask"] = max(row_masks)
                facts["local_address_data_width"] = max(row_masks).bit_length()
        except Exception:
            # The base RTL facts remain useful on a target whose address-space/ISA contract cannot be
            # derived. Leave only these optional fields UNKNOWN; do not erase mesh/decode facts.
            pass
        ft = next((i for i in f.get("interfaces", []) if i.get("name") == "funct_decode_table"), None)
        if ft:
            facts["legal_funct"] = ft.get("legal_funct")
            facts["custom_opcode"] = ft.get("custom_opcode")
            facts["funct3"] = ft.get("funct3")
        layouts = next((i.get("bundles", {}) for i in f.get("interfaces", [])
                        if i.get("name") == "register_bundle_layouts"), {})
        mvout_layout = layouts.get("ConfigMvoutRs1") if isinstance(layouts, dict) else None
        if isinstance(mvout_layout, dict):
            facts["config_mvout_fields"] = sorted((mvout_layout.get("fields") or {}).keys())
        build_features = next((i for i in f.get("interfaces", [])
                               if i.get("name") == "elaborated_rtl_features"), {})
        max_pool = (build_features.get("features") or {}).get("max_pool")
        if isinstance(max_pool, bool) and build_features.get("status") == "derived":
            facts["max_pool_supported"] = max_pool
        facts["from"] = f"circt_introspect facts.json ({rec.get('generator', {}).get('version', '?')})"
    except Exception:  # noqa: BLE001 — facts not derivable -> everything stays UNKNOWN (checks skip)
        pass
    return facts

# capsule.operation.op values whose semantics REQUIRE matrix compute (used by movement/compute
# balance). Movement-only ops legitimately emit no COMPUTE; these do not.
_COMPUTE_OPS = {"matmul", "resident_reuse", "conv2d", "conv", "matmul_resident"}
_COMPUTE_CLASSES = {"COMPUTE_PRELOADED", "COMPUTE_ACCUMULATE"}
_MVIN_CLASSES = {"MVIN", "MVIN2", "MVIN3"}


@dataclasses.dataclass
class Check:
    id: str
    tier: str               # "T0" (pure static) | "T1" (cheap spike byproduct)
    severity: str           # "error" | "warn" | "info"
    status: str             # "pass" | "fail" | "skipped"
    message: str
    expected: Any = None
    got: Any = None
    ratio: float | None = None
    evidence: dict | None = None
    fix_hint: str | None = None

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        return {k: v for k, v in d.items() if v is not None or k in ("expected", "got")}


@dataclasses.dataclass
class CheckReport:
    capsule: str | None
    source_trace: str | None
    rtl_facts: dict
    checks: list[Check] = dataclasses.field(default_factory=list)

    @property
    def n_error(self) -> int:
        return sum(1 for c in self.checks if c.severity == "error" and c.status == "fail")

    @property
    def n_warn(self) -> int:
        return sum(1 for c in self.checks if c.severity == "warn" and c.status == "fail")

    @property
    def verdict(self) -> str:
        if self.n_error:
            return "reject"
        if self.n_warn:
            return "warn"
        return "ok"

    def to_dict(self) -> dict:
        return {
            "schema": SCHEMA,
            "capsule": self.capsule,
            "source_trace": self.source_trace,
            "rtl_facts": self.rtl_facts,
            "verdict": self.verdict,
            "n_error": self.n_error,
            "n_warn": self.n_warn,
            "checks": [c.to_dict() for c in self.checks if c.status != "skipped"],
            "skipped": [{"id": c.id, "reason": c.message}
                        for c in self.checks if c.status == "skipped"],
        }


def _classes(trace: dict) -> list[str]:
    return [i.get("class") for i in trace.get("instructions", [])]


def _declared_op(capsule: dict | None) -> str | None:
    if not capsule:
        return None
    op = (capsule.get("operation") or {}).get("op")
    return op.lower() if isinstance(op, str) else None


def _declared_output_shape(capsule: dict | None) -> tuple[int, int] | None:
    """Best-effort (M, N) of the declared output, GENERAL (declared shapes only, no golden).

    For a plain matmul the output is (M, N) with lhs (M,K) and weight (K,N). We read the declared
    input roles. Returns None when the shape cannot be derived generally (then tile_coverage skips,
    honestly, rather than guessing).
    """
    if not capsule:
        return None
    inputs = capsule.get("inputs") or []
    by_role: dict[str, list] = {}
    for t in inputs:
        by_role.setdefault(t.get("role"), []).append(t.get("shape"))
    lhs = by_role.get("input")
    w = by_role.get("weight")
    if not lhs or not w:
        return None
    lhs_shape, w_shape = lhs[0], w[0]
    if not (isinstance(lhs_shape, list) and isinstance(w_shape, list)
            and len(lhs_shape) == 2 and len(w_shape) == 2):
        return None
    M = lhs_shape[0]
    N = w_shape[1]
    return (int(M), int(N))


def _declared_mkn(capsule: dict | None) -> tuple[int, int, int] | None:
    """(M, N, K) for a declared matmul: lhs (M,K) · weight (K,N). None if not derivable generally."""
    shape = _declared_output_shape(capsule)
    if shape is None:
        return None
    inputs = (capsule or {}).get("inputs") or []
    lhs = next((t.get("shape") for t in inputs if t.get("role") == "input"), None)
    if not (isinstance(lhs, list) and len(lhs) == 2):
        return None
    M, N = shape
    return (M, N, int(lhs[1]))                        # K is the contraction dim (lhs columns)


_DTYPE_BYTES = {"i8": 1, "int8": 1, "u8": 1, "i16": 2, "bf16": 2, "fp16": 2, "f16": 2,
                "i32": 4, "fp32": 4, "f32": 4}


def _elem_bytes(capsule: dict | None) -> int:
    """Element byte width from the declared input dtype (default 1 = i8, the corpus default)."""
    inputs = (capsule or {}).get("inputs") or []
    dt = next((t.get("dtype") for t in inputs if t.get("role") == "input"), None) \
        or ((capsule or {}).get("operation") or {}).get("attributes", {}).get("input_dtype")
    return _DTYPE_BYTES.get(str(dt).lower(), 1) if dt else 1


def _mesh(rtl_facts: dict) -> tuple[int, int] | None:
    """The (rows, cols) systolic mesh, or None when UNKNOWN (no RTL facts derived) — callers SKIP the
    check rather than screen against an assumed default."""
    m = rtl_facts.get("mesh")
    return (int(m[0]), int(m[1])) if m else None


# --------------------------------------------------------------------------- individual T0 checks
def _check_decode_clean(trace: dict) -> Check:
    classes = _classes(trace)
    n_unknown = classes.count("UNKNOWN")
    n = len(classes)
    if n == 0:
        return Check("T0.decode_clean", "T0", "error", "fail",
                     "empty instruction trace (no RoCC instructions decoded)",
                     expected=">0", got=0,
                     fix_hint="backend emitted no custom-3 .insn — lowering produced nothing")
    if n_unknown:
        idxs = [i["index"] for i in trace.get("instructions", []) if i.get("class") == "UNKNOWN"]
        return Check("T0.decode_clean", "T0", "error", "fail",
                     f"trace contains {n_unknown} UNKNOWN instruction(s) (fail-closed decode)",
                     expected=0, got=n_unknown, evidence={"instruction_indices": idxs[:8]},
                     fix_hint="emitted a custom-3 form the ABI does not recognise; RTL would reject")
    return Check("T0.decode_clean", "T0", "error", "pass",
                 f"all {n} instructions decoded to known ISA classes", expected=0, got=0)


def _check_movement_compute_balance(trace: dict, capsule: dict | None) -> Check:
    classes = _classes(trace)
    n_cmp = sum(classes.count(c) for c in _COMPUTE_CLASSES)
    n_mvin = sum(classes.count(c) for c in _MVIN_CLASSES)
    n_mvout = classes.count("MVOUT")
    op = _declared_op(capsule)
    if op is None:
        return Check("T0.movement_compute_balance", "T0", "error", "skipped",
                     "capsule declares no operation.op; cannot judge compute requirement")
    needs_compute = op in _COMPUTE_OPS
    if needs_compute and n_cmp == 0 and (n_mvin or n_mvout):
        return Check("T0.movement_compute_balance", "T0", "error", "fail",
                     f"declared op '{op}' needs matrix compute but trace has 0 COMPUTE "
                     f"(MVIN={n_mvin}, MVOUT={n_mvout}) — all movement, no compute",
                     expected=">0", got=0,
                     fix_hint="emit PRELOAD+COMPUTE for the matmul; movement alone is not the op")
    if n_cmp and n_mvout == 0:
        return Check("T0.movement_compute_balance", "T0", "error", "fail",
                     f"trace has {n_cmp} COMPUTE but 0 MVOUT — computed result never committed",
                     expected=">0", got=0,
                     fix_hint="add MVOUT to commit the accumulator; compute without commit is dead")
    return Check("T0.movement_compute_balance", "T0", "error", "pass",
                 f"compute/movement balance ok (COMPUTE={n_cmp}, MVIN={n_mvin}, MVOUT={n_mvout})")


def _check_tile_coverage(trace: dict, capsule: dict | None, rtl_facts: dict) -> Check:
    classes = _classes(trace)
    n_mvout = classes.count("MVOUT")
    mesh = _mesh(rtl_facts)
    if mesh is None:
        return Check("T0.tile_coverage", "T0", "warn", "skipped",
                     "mesh dims UNKNOWN (no RTL facts derived) — cannot screen tile coverage")
    mr, mc = mesh
    shape = _declared_output_shape(capsule)
    # multi-matmul: count declared matmuls; if >1 we can only assert a LOWER BOUND generally.
    op_attrs = ((capsule or {}).get("operation") or {}).get("attributes") or {}
    matmuls = op_attrs.get("matmuls")
    if shape is None and not matmuls:
        return Check("T0.tile_coverage", "T0", "warn", "skipped",
                     "could not derive a declared output shape (general, no golden) — skipped")
    if isinstance(matmuls, list) and len(matmuls) >= 1 and shape is not None:
        # N matmuls reusing one resident weight over an MxN output: lower bound N * Mt * Nt tiles.
        M, N = shape
        per = math.ceil(M / mr) * math.ceil(N / mc)
        lo = per * len(matmuls)
        if n_mvout < lo:
            return Check("T0.tile_coverage", "T0", "warn", "fail",
                         f"MVOUT={n_mvout} < lower-bound {lo} tiles for {len(matmuls)} matmul(s) "
                         f"of {M}x{N} over {mr}x{mc} mesh",
                         expected=f">={lo}", got=n_mvout,
                         ratio=round(n_mvout / lo, 3) if lo else None,
                         fix_hint="under-committing output tiles; check the per-matmul tiling loop")
        return Check("T0.tile_coverage", "T0", "warn", "pass",
                     f"MVOUT={n_mvout} >= lower-bound {lo} tiles ({len(matmuls)} matmuls of {M}x{N})",
                     expected=f">={lo}", got=n_mvout)
    # Single declared matmul: the store-controller mode determines the exact commit count. A normal
    # MVOUT commits one mesh-row tile; native pooling walks every output spatial row inside one MVOUT,
    # so it needs one command per output channel tile (and per independent spatial plane), not Mt*Nt.
    expected = expected_mvout_count(capsule, rtl_facts)
    if expected is None:
        return Check("T0.tile_coverage", "T0", "warn", "skipped",
                     "exact MVOUT geometry is not derivable from the declaration and RTL mesh")
    exp, geometry = expected
    if n_mvout != exp:
        return Check("T0.tile_coverage", "T0", "warn", "fail",
                     f"MVOUT={n_mvout} but {geometry} needs {exp} store command(s)",
                     expected=exp, got=n_mvout,
                     ratio=round(n_mvout / exp, 3) if exp else None,
                     fix_hint=("over-committing outputs" if n_mvout > exp else "under-committing "
                               "outputs") + f" ~{round(n_mvout/exp,1) if exp else '?'}x; "
                               "check the M/N tiling loop bounds")
    return Check("T0.tile_coverage", "T0", "warn", "pass",
                 f"MVOUT={n_mvout} == {exp} for {geometry}",
                 expected=exp, got=n_mvout)


def _check_data_movement_reuse(trace: dict, capsule: dict | None, rtl_facts: dict) -> Check:
    """ADVISORY (M3): does the emitted movement reflect on-chip operand REUSE, or is it re-streaming?

    A numerically-correct kernel can still move far more data than necessary (re-loading a resident
    operand instead of reusing it) — a data-movement inefficiency spike is blind to and that today needs
    a perf sim. mlc's Model-3 static reuse math (predict_dma_volume + spills, no arc/cycles) gives, from
    the declared shape + discovered DIM/capacity: the resident footprint (rows/tiles), whether it FITS
    on-chip, and the loop-nest REFETCH factor. We report that budget and compare it to the observed MVIN
    count. Severity=info: this NEVER moves the verdict (which rides the correctness checks) — it is
    feedback. Fail-closed to skipped when mlc or the shape is unavailable."""
    op = _declared_op(capsule)
    if op not in ("matmul", "matmul_resident"):
        return Check("T0.data_movement_reuse", "T0", "info", "skipped",
                     "advisory reuse check applies to a single declared matmul only")
    mkn = _declared_mkn(capsule)
    if mkn is None:
        return Check("T0.data_movement_reuse", "T0", "info", "skipped",
                     "could not derive (M,N,K) generally — skipped")
    M, N, K = mkn
    mesh = _mesh(rtl_facts)
    spad_bytes = rtl_facts.get("scratchpad_bytes")
    if mesh is None or spad_bytes is None:
        return Check("T0.data_movement_reuse", "T0", "info", "skipped",
                     "mesh/scratchpad UNKNOWN (no RTL facts) — advisory reuse check skipped")
    mr, _mc = mesh
    spad_bytes = int(spad_bytes)
    from .rtl import mlc_bridge  # lazy: keeps rtl_checks import-light and mlc optional
    pred = mlc_bridge.matmul_reuse_prediction(
        M, N, K, dim=mr, capacity_bytes=spad_bytes, elem_bytes=_elem_bytes(capsule))
    if pred is None:
        return Check("T0.data_movement_reuse", "T0", "info", "skipped",
                     "mlc Model-3 reuse model unavailable (MERLIN_MLC_DIR) — advisory skipped")
    classes = _classes(trace)
    n_mvin = sum(classes.count(c) for c in _MVIN_CLASSES)
    ideal_loads = pred["footprint_tiles"] * pred["refetch"]     # resident tiles, re-streamed on spill
    ev = {"footprint_rows": pred["footprint_rows"], "capacity_rows": pred["capacity_rows"],
          "fits": pred["fits"], "refetch": pred["refetch"], "ideal_tile_loads": ideal_loads,
          "observed_mvin": n_mvin}
    # Unambiguous under-reuse: the shape is FULLY resident (fits, refetch==1) yet the trace loads
    # operands many times over. 3x slack absorbs the coarse MVIN-instruction↔tile-load mapping, so a
    # flag means gross re-streaming, not a granularity artefact.
    if pred["fits"] and pred["refetch"] == 1 and ideal_loads > 0 and n_mvin >= 3 * ideal_loads:
        return Check("T0.data_movement_reuse", "T0", "info", "fail",
                     f"excess operand movement: {n_mvin} MVIN vs ~{ideal_loads} resident tile-loads "
                     f"(~{round(n_mvin/ideal_loads,1)}x) — shape is fully on-chip (refetch=1) so "
                     f"operands are being re-streamed instead of reused",
                     expected=f"~{ideal_loads}", got=n_mvin,
                     ratio=round(n_mvin / ideal_loads, 3), evidence=ev,
                     fix_hint="keep A/B tiles resident across the reuse loop; avoid re-MVIN per output tile")
    note = ("spills capacity" if not pred["fits"] else "fully resident") + \
        f" (footprint {pred['footprint_rows']} vs {pred['capacity_rows']} rows, refetch={pred['refetch']})"
    return Check("T0.data_movement_reuse", "T0", "info", "pass",
                 f"movement within reuse budget: {n_mvin} MVIN, ideal ~{ideal_loads} tile-loads; {note}",
                 expected=f"~{ideal_loads}", got=n_mvin, evidence=ev)


def _check_spad_capacity(trace: dict, rtl_facts: dict) -> Check:
    """Every MVIN local row must fit the memory space selected by its encoded address.

    Gemmini-style local addresses multiplex scratchpad and accumulator rows in one operand.  The
    accumulator selector is derived by :func:`load_default_facts` from the target's ISA contract,
    while its row-payload mask and both row capacities are independently derived from CIRCT
    memory/datapath facts.
    Comparing the raw encoded value to scratchpad capacity confuses a space tag with a row bit (for
    example ``ACC_I8 | 16`` is accumulator row 16, not scratchpad row 2147483664).

    The historical check id is retained for artifact compatibility even though the invariant now
    covers both local spaces.
    """
    mesh = _mesh(rtl_facts)
    spad_bytes = rtl_facts.get("scratchpad_bytes")
    if mesh is None or spad_bytes is None:
        return Check("T0.spad_capacity", "T0", "error", "skipped",
                     "mesh/scratchpad UNKNOWN (no RTL facts) — cannot bound scratchpad capacity")
    mr, _mc = mesh
    spad_bytes = int(spad_bytes)
    # Prefer the address-space derivation.  Keep the byte/mesh calculation as a compatibility path for
    # callers that inject the old flat facts shape directly into this pure helper.
    spad_capacity = rtl_facts.get("scratchpad_rows")
    if not isinstance(spad_capacity, int):
        spad_capacity = int(spad_bytes) // (mr * 1)
    spad_row_mask = rtl_facts.get("scratchpad_row_mask")
    if isinstance(spad_capacity, int) and not isinstance(spad_row_mask, int):
        spad_row_mask = (1 << max(1, (spad_capacity - 1).bit_length())) - 1
    acc_capacity = rtl_facts.get("accumulator_rows")
    acc_select = rtl_facts.get("accumulator_select_bit")
    acc_row_mask = rtl_facts.get("accumulator_row_mask")
    if isinstance(acc_capacity, int) and not isinstance(acc_row_mask, int):
        acc_row_mask = (1 << max(1, (acc_capacity - 1).bit_length())) - 1
    local_data_mask = rtl_facts.get("local_address_data_mask")
    if not isinstance(local_data_mask, int):
        # Compatibility for callers injecting the old flat fact shape: LocalAddr.data is the widest
        # local-store address payload, so its mask is the maximum of the derived per-store masks.
        known_masks = [m for m in (spad_row_mask, acc_row_mask) if isinstance(m, int)]
        local_data_mask = max(known_masks) if known_masks else None
    local_data_width = (local_data_mask.bit_length()
                        if isinstance(local_data_mask, int) and local_data_mask >= 0 else None)
    selector_is_derived = (isinstance(acc_select, int) and not isinstance(acc_select, bool)
                           and acc_select > 0 and (acc_select & (acc_select - 1)) == 0)

    encoded = [(i.get("index"), i.get("decoded", {}).get("spad_addr"),
                i.get("decoded", {}).get("rows"))
               for i in trace.get("instructions", []) if i.get("class") in _MVIN_CLASSES]
    encoded = [(idx, addr, rows) for idx, addr, rows in encoded
               if isinstance(addr, int) and not isinstance(addr, bool)]
    if not encoded:
        return Check("T0.spad_capacity", "T0", "error", "skipped",
                     "no MVIN scratchpad addresses to check")

    # Each entry is (instruction index, decoded base row, exclusive transfer end). The decoded `rows`
    # field is the exact number of consecutive local rows touched. An absent count is UNKNOWN; zero or a
    # negative value is an invalid encoded transfer, not a one-row default (LoadController consumes the
    # encoded field and its zero case can underflow the repeat count / violate its non-zero-byte assert).
    spad: list[tuple[int | None, int, int]] = []
    accumulator: list[tuple[int | None, int, int]] = []
    unresolved_acc: list[int | None] = []
    unresolved_space: list[int | None] = []
    unresolved_rows: list[int | None] = []
    invalid_rows: list[int | None] = []
    noncanonical_acc: list[int | None] = []
    noncanonical_spad: list[int | None] = []
    for idx, addr, transfer_rows in encoded:
        if not isinstance(transfer_rows, int) or isinstance(transfer_rows, bool):
            unresolved_rows.append(idx)
            continue
        if transfer_rows <= 0:
            invalid_rows.append(idx)
            continue

        if selector_is_derived and addr & acc_select:
            if isinstance(acc_row_mask, int) and isinstance(acc_capacity, int):
                row = addr & acc_row_mask
                if isinstance(local_data_mask, int):
                    payload = addr & local_data_mask
                    if payload & ~acc_row_mask:
                        noncanonical_acc.append(idx)
                accumulator.append((idx, row, row + transfer_rows))
            else:
                unresolved_acc.append(idx)
        else:
            # A high/control bit on an address cannot safely be discarded when a separate accumulator
            # exists but its selector was not derived. It might select that space. Low-only values carry
            # no set selector bit and remain unambiguously scratchpad addresses.
            if (isinstance(acc_capacity, int) and not selector_is_derived
                    and (not isinstance(local_data_mask, int) or addr & ~local_data_mask)):
                unresolved_space.append(idx)
                continue
            if not isinstance(spad_row_mask, int):
                unresolved_space.append(idx)
                continue
            row = addr & spad_row_mask
            if isinstance(local_data_mask, int):
                payload = addr & local_data_mask
                if payload & ~spad_row_mask:
                    noncanonical_spad.append(idx)
            spad.append((idx, row, row + transfer_rows))

    bad_spad = [(idx, row, end) for idx, row, end in spad if end > spad_capacity]
    bad_acc = ([(idx, row, end) for idx, row, end in accumulator if end > acc_capacity]
               if isinstance(acc_capacity, int) else [])
    spad_max = max((row for _idx, row, _end in spad), default=None)
    acc_max = max((row for _idx, row, _end in accumulator), default=None)
    spad_end = max((end for _idx, _row, end in spad), default=None)
    acc_end = max((end for _idx, _row, end in accumulator), default=None)
    evidence = {
        "scratchpad_max_row": spad_max,
        "scratchpad_max_row_exclusive": spad_end,
        "scratchpad_rows_capacity": spad_capacity,
        "scratchpad_row_mask": spad_row_mask,
        "accumulator_max_row": acc_max,
        "accumulator_max_row_exclusive": acc_end,
        "accumulator_rows_capacity": acc_capacity,
        "accumulator_select_bit": acc_select,
        "accumulator_row_mask": acc_row_mask,
        "local_address_data_width": local_data_width,
        "local_address_data_mask": local_data_mask,
        "noncanonical_accumulator_instruction_indices": noncanonical_acc[:8],
        "noncanonical_scratchpad_instruction_indices": noncanonical_spad[:8],
        "unresolved_address_space_instruction_indices": unresolved_space[:8],
        "invalid_row_count_instruction_indices": invalid_rows[:8],
        "unresolved_row_count_instruction_indices": unresolved_rows[:8],
    }
    if bad_spad or bad_acc or noncanonical_acc or noncanonical_spad or invalid_rows:
        bad = ([idx for idx, _row, _end in (bad_spad + bad_acc)] + noncanonical_acc
               + noncanonical_spad + invalid_rows)[:8]
        details = []
        if bad_spad:
            details.append(
                f"scratchpad transfer end {max(end for _idx, _row, end in bad_spad)} > {spad_capacity}"
            )
        if bad_acc:
            details.append(
                f"accumulator transfer end {max(end for _idx, _row, end in bad_acc)} > {acc_capacity}"
            )
        if noncanonical_acc:
            details.append(
                f"{len(noncanonical_acc)} accumulator address(es) set LocalAddr.data bits outside "
                "the accumulator row payload and alias lower physical rows"
            )
        if noncanonical_spad:
            details.append(
                f"{len(noncanonical_spad)} scratchpad address(es) set LocalAddr.data bits outside "
                "the scratchpad row payload and alias lower physical rows"
            )
        if invalid_rows:
            details.append(f"{len(invalid_rows)} MVIN address(es) have a non-positive row count")
        return Check("T0.spad_capacity", "T0", "error", "fail",
                     "; ".join(details) + " — local transfer is invalid or would alias/wrap on silicon",
                     expected={"scratchpad": f"<{spad_capacity}",
                               "accumulator": (f"<{acc_capacity}" if isinstance(acc_capacity, int)
                                               else "derived capacity")},
                     got={"scratchpad_max": spad_max, "accumulator_max": acc_max},
                     evidence={**evidence, "instruction_indices": bad},
                     fix_hint="resident footprint exceeds its selected local memory; tile smaller or evict")
    unresolved = unresolved_acc + unresolved_space + unresolved_rows
    if unresolved:
        reasons = []
        if unresolved_acc:
            reasons.append("an accumulator-selected MVIN has UNKNOWN row mask/capacity")
        if unresolved_space:
            reasons.append("the accumulator selector/local-address payload is UNKNOWN")
        if unresolved_rows:
            reasons.append("an MVIN row count is undecodable")
        return Check("T0.spad_capacity", "T0", "error", "skipped",
                     "; ".join(reasons) + " — cannot safely prove the local transfer bound",
                     evidence={**evidence, "instruction_indices": unresolved[:8]})
    spaces = [f"scratchpad max {spad_max} < {spad_capacity}"] if spad_max is not None else []
    if acc_max is not None:
        spaces.append(f"accumulator max {acc_max} < {acc_capacity}")
    return Check("T0.spad_capacity", "T0", "error", "pass",
                 "; ".join(spaces),
                 expected={"scratchpad": f"<{spad_capacity}",
                           "accumulator": (f"<{acc_capacity}" if isinstance(acc_capacity, int)
                                           else None)},
                 got={"scratchpad_max": spad_max, "accumulator_max": acc_max}, evidence=evidence)


def _check_decode_funct_legal(trace: dict, rtl_facts: dict) -> Check:
    """Every emitted RoCC funct must be in the RTL-derived legal set (GemminiISA funct block).

    This is the hardware-legality invariant: a funct outside the decoder's legal range is an encoding
    the silicon would not accept. Mirrors the FileCheck ILLEGAL_FUNCT_COUNT assertion for the no-FileCheck
    path + a structured verdict."""
    legal_list = rtl_facts.get("legal_funct")
    if not legal_list:
        return Check("T0.decode_funct_legal", "T0", "error", "skipped",
                     "legal funct set UNKNOWN (no RTL facts derived) — cannot screen funct legality")
    legal = set(legal_list)
    bad = [i["index"] for i in trace.get("instructions", [])
           if isinstance(i.get("funct"), int) and i["funct"] not in legal]
    if bad:
        functs = sorted({i["funct"] for i in trace.get("instructions", [])
                         if isinstance(i.get("funct"), int) and i["funct"] not in legal})
        return Check("T0.decode_funct_legal", "T0", "error", "fail",
                     f"{len(bad)} instruction(s) use funct {functs} outside RTL legal set "
                     f"[{min(legal)}..{max(legal)}]",
                     expected=0, got=len(bad), evidence={"instruction_indices": bad[:8]},
                     fix_hint="emitted a custom funct the target's RoCC decoder rejects; RTL would not accept")
    return Check("T0.decode_funct_legal", "T0", "error", "pass",
                 f"all emitted functs ∈ RTL legal set [{min(legal)}..{max(legal)}]", expected=0, got=0)


def _first(classes: list[str], names) -> int:
    for i, c in enumerate(classes):
        if c in names:
            return i
    return -1


def _check_preload_before_compute(trace: dict) -> Check:
    """WS protocol: a COMPUTE_PRELOADED must be preceded by a PRELOAD (sets the stationary weight + the
    output/accumulator address). A preloaded-compute with no prior PRELOAD is a real protocol violation."""
    cls = _classes(trace)
    fc = _first(cls, {"COMPUTE_PRELOADED"})
    fp = _first(cls, {"PRELOAD"})
    if fc == -1:
        return Check("T0.preload_before_compute", "T0", "error", "pass",
                     "no preloaded-compute to pair", expected=None, got=None)
    if fp == -1 or fp > fc:
        return Check("T0.preload_before_compute", "T0", "error", "fail",
                     f"COMPUTE_PRELOADED at idx {fc} has no preceding PRELOAD "
                     f"({'none in trace' if fp==-1 else f'first PRELOAD at {fp}'})",
                     expected="PRELOAD before COMPUTE_PRELOADED", got="compute-before-preload",
                     evidence={"instruction_indices": [fc]},
                     fix_hint="emit PRELOAD (weight + C addr) before the preloaded matmul COMPUTE")
    return Check("T0.preload_before_compute", "T0", "error", "pass",
                 f"PRELOAD@{fp} precedes COMPUTE_PRELOADED@{fc}")


def _check_fence_bracket(trace: dict) -> Check:
    """The RoCC trace must OPEN and CLOSE with a FENCE: a leading fence quiesces prior accelerator state,
    a trailing fence guarantees the result is committed before the host reads it. Missing either is a real
    correctness hazard (general — not shape-specific), and is the dominant real-draft protocol violation."""
    cls = _classes(trace)
    if not cls:
        return Check("T0.fence_bracket", "T0", "error", "skipped", "empty trace")
    opens, closes = cls[0] == "FENCE", cls[-1] == "FENCE"
    if opens and closes:
        return Check("T0.fence_bracket", "T0", "error", "pass", "trace opens and closes with FENCE")
    miss = []
    if not opens:
        miss.append(f"does not open with FENCE (first={cls[0]})")
    if not closes:
        miss.append(f"does not close with FENCE (last={cls[-1]})")
    return Check("T0.fence_bracket", "T0", "error", "fail", "; ".join(miss),
                 expected="FENCE … FENCE", got=f"{cls[0]} … {cls[-1]}",
                 fix_hint="bracket the kernel with a leading and trailing gemmini_fence()")


def _check_config_before_use(trace: dict) -> Check:
    """A COMPUTE before any CONFIG_EX (dataflow/activation undefined) or an MVOUT before any CONFIG_ST
    (output scale/stride undefined) uses uninitialised config — a real correctness hazard, general to WS."""
    cls = _classes(trace)
    problems = []
    fc = _first(cls, {"COMPUTE_PRELOADED", "COMPUTE_ACCUMULATE"})
    fx = _first(cls, {"CONFIG_EX"})
    if fc != -1 and (fx == -1 or fx > fc):
        problems.append(f"COMPUTE@{fc} before CONFIG_EX")
    fo = _first(cls, {"MVOUT"})
    fs = _first(cls, {"CONFIG_ST"})
    if fo != -1 and (fs == -1 or fs > fo):
        problems.append(f"MVOUT@{fo} before CONFIG_ST")
    if problems:
        return Check("T0.config_before_use", "T0", "error", "fail",
                     "use before config: " + "; ".join(problems),
                     expected="CONFIG before first use", got="; ".join(problems),
                     fix_hint="emit CONFIG_EX before the first COMPUTE and CONFIG_ST before the first MVOUT")
    return Check("T0.config_before_use", "T0", "error", "pass", "configs precede their first use")


# ------------------------------------------------------------- declared geometry (declarations only)
# Everything below derives from (a) the capsule's DECLARED shapes/attributes -- the problem statement the
# author was handed -- and (b) the RTL facts, and compares them against the author's OWN emitted stream.
# No golden, no expected value and no hidden capsule name is read on any path here.


def _attrs(capsule: dict | None) -> dict:
    return ((capsule or {}).get("operation") or {}).get("attributes") or {}


def _input_shape(capsule: dict | None, name: str | None, role: str | None = None):
    """A declared input's shape, by NAME first (the operation attributes reference inputs by name) and by
    ROLE as the fallback. None when the capsule declares neither."""
    ins = (capsule or {}).get("inputs") or []
    for t in ins:
        if name is not None and t.get("name") == name:
            return t.get("shape")
    if role is not None:
        for t in ins:
            if t.get("role") == role:
                return t.get("shape")
    return None


def _declared_elem_bytes(entry: dict, capsule: dict | None) -> int | None:
    """Byte width of a declared commit, from the commit's own ``output_dtype`` and, failing that, from the
    capsule's declared numeric policy. Both are DECLARATIONS (the problem statement), never a golden."""
    dt = entry.get("output_dtype") or ((capsule or {}).get("numeric_policy") or {}).get("dtype")
    return _DTYPE_BYTES.get(str(dt).lower()) if dt else None


def _pooled_rows(rows: int, entry: dict) -> tuple[int | None, str]:
    """Rows after a declared pooling stage, or ``(None, reason)``. The window and the extent the rows
    unflatten to come from the DECLARED pool attributes through the runtime's own geometry function, so
    this check and the engines cannot disagree about what extent a pooled commit has."""
    pid, ps, pst = entry.get("pool_in_dims"), entry.get("pool_size"), entry.get("pool_stride")
    pp = entry.get("pool_padding") or [0, 0, 0, 0]
    if not (pid and ps and pst):
        return None, ("the declared epilogue contains a pooling stage but no pool_in_dims/pool_size/"
                      "pool_stride is declared, so the committed extent is not derivable")
    from merlin.runtime.tensor import pool_out_dims          # the ONE pooled-extent definition
    H, W = int(pid[0]), int(pid[1])
    if H * W <= 0 or rows % (H * W):
        return None, (f"declared pool_in_dims [{H}, {W}] does not divide the {rows} accumulator rows, "
                      f"so the pooled extent is not derivable")
    Ho, Wo = pool_out_dims(H, W, ps, pst, pp)
    return (rows // (H * W)) * Ho * Wo, ""


_POOL_CONFIG_FIELDS = ("pool_stride", "pool_size", "pool_out_dim", "porows", "pocols",
                       "orows", "ocols", "upad", "lpad")


def _declared_pool_geometry(capsule: dict | None) -> tuple[dict | None, str]:
    """The native store-controller geometry implied by a declared max-pool epilogue.

    Gemmini's RTL bundle has scalar (square-window) size/stride fields and explicit top/left padding;
    bottom/right padding affects the derived output dimensions.  No field width or offset is assumed
    here: this is declaration-side intent, compared later with fields decoded from the RTL layout.
    """
    a = _attrs(capsule)
    if "maxpool" not in (a.get("epilogue") or []):
        return None, "no maxpool epilogue is declared"
    pid, size, stride = a.get("pool_in_dims"), a.get("pool_size"), a.get("pool_stride")
    padding = a.get("pool_padding") or [0, 0, 0, 0]
    if not all(isinstance(v, (list, tuple)) and len(v) == n
               for v, n in ((pid, 2), (size, 2), (stride, 2), (padding, 4))):
        return None, "pool_in_dims/size/stride/padding do not have declared arities 2/2/2/4"
    try:
        H, W = (int(pid[0]), int(pid[1]))
        sh, sw = (int(size[0]), int(size[1]))
        th, tw = (int(stride[0]), int(stride[1]))
        top, left, bottom, right = (int(x) for x in padding)
    except (TypeError, ValueError):
        return None, "pool geometry contains a non-integer value"
    if min(H, W, sh, sw, th, tw) <= 0:
        return None, "pool input, window, and stride dimensions must be positive"
    if sh != sw or th != tw:
        return None, "the RTL CONFIG_ST bundle represents only square pool_size and pool_stride"
    from merlin.runtime.tensor import pool_out_dims
    try:
        Ho, Wo = pool_out_dims(H, W, [sh, sw], [th, tw], [top, left, bottom, right])
    except (TypeError, ValueError) as e:
        return None, f"declared pool output geometry is invalid ({type(e).__name__})"
    if Ho <= 0 or Wo <= 0:
        return None, "declared pool geometry has no output positions"

    pre_rows = None
    op = _declared_op(capsule)
    if op in ("matmul", "matmul_resident"):
        shape = _declared_output_shape(capsule)
        pre_rows = shape[0] if shape is not None else None
    elif op in ("conv", "conv2d"):
        ifm = _input_shape(capsule, a.get("ifm"), "input")
        if isinstance(ifm, list) and len(ifm) == 4:
            try:
                from merlin.runtime.commandbuffer import conv_out_dims
                ch, cw = conv_out_dims(int(ifm[1]), int(ifm[2]), int(a["kh"]), int(a["kw"]),
                                       a.get("stride", [1, 1]),
                                       a.get("padding", [0, 0, 0, 0]),
                                       a.get("dilation", [1, 1]))
                pre_rows = int(ifm[0]) * ch * cw
            except (KeyError, TypeError, ValueError):
                pre_rows = None
    if not isinstance(pre_rows, int) or pre_rows <= 0 or pre_rows % (H * W):
        return None, (f"pool_in_dims [{H}, {W}] does not partition the operation's pre-pool "
                      f"row extent {pre_rows!r}")
    planes = pre_rows // (H * W)
    return {
        "pool_stride": th, "pool_size": sh, "pool_out_dim": Wo,
        "porows": Ho, "pocols": Wo, "orows": H, "ocols": W,
        "upad": top, "lpad": left, "planes": planes,
        "rows_per_store": Ho * Wo,
    }, ""


def _pool_config_agreement(config: dict | None, capsule: dict | None,
                           rtl_facts: dict) -> tuple[bool, str, dict]:
    """Compare one emitted CONFIG_ST snapshot with declaration-derived pooling intent."""
    want, why = _declared_pool_geometry(capsule)
    if want is None:
        return False, why, {}
    supported = rtl_facts.get("max_pool_supported")
    if supported is not True:
        state = "disabled in the elaborated RTL configuration" if supported is False else "UNKNOWN"
        return False, f"native max-pool capability is {state}", want
    config = config or {}
    missing = [f for f in _POOL_CONFIG_FIELDS if not isinstance(config.get(f), int)]
    if missing:
        return False, f"RTL-derived decoder did not recover CONFIG_ST field(s) {missing}", want
    mismatch = {f: {"expected": want[f], "got": int(config[f])}
                for f in _POOL_CONFIG_FIELDS if int(config[f]) != want[f]}
    if mismatch:
        return False, f"CONFIG_ST pooling fields disagree with the declaration: {mismatch}", want
    return True, "", want


def expected_mvout_count(capsule: dict | None, rtl_facts: dict) -> tuple[int, str] | None:
    """Exact single-matmul store count from declared geometry and the RTL-derived mesh.

    Shared by the Python screen and FileCheck compiler so their count model cannot drift.
    """
    if _declared_op(capsule) not in ("matmul", "matmul_resident"):
        return None
    a = _attrs(capsule)
    if isinstance(a.get("matmuls"), list):
        return None
    shape, mesh = _declared_output_shape(capsule), _mesh(rtl_facts)
    if shape is None or mesh is None:
        return None
    M, N = shape
    mr, mc = mesh
    if "maxpool" in (a.get("epilogue") or []):
        if rtl_facts.get("max_pool_supported") is not True:
            return None
        recovered = rtl_facts.get("config_mvout_fields")
        if not isinstance(recovered, list) or not set(_POOL_CONFIG_FIELDS) <= set(recovered):
            return None
        pool, _why = _declared_pool_geometry(capsule)
        if pool is None:
            return None
        count = int(pool["planes"]) * math.ceil(N / mc)
        return count, (f"native pooled {M}x{N} commit: {pool['planes']} spatial plane(s) x "
                       f"ceil({N}/{mc}) channel tiles")
    count = math.ceil(M / mr) * math.ceil(N / mc)
    return count, f"{M}x{N} over {mr}x{mc} mesh (Mt*Nt)"


def declared_outputs(capsule: dict | None) -> tuple[list[dict], str]:
    """Every DECLARED output's committed extent, in kernel-argument order — or ``([], reason)``.

    ``[{name, rows, cols, elem_bytes, arg_index}]``, derived only from the capsule's declaration:

    * a contraction commits ``[M, N]`` — M the declared lhs rows, N the declared weight columns;
    * a residency capsule commits one such tensor per declared matmul, in declaration order;
    * a convolution commits ``[N*Ho*Wo, Co]``, the spatial extent through the runtime's own
      ``conv_out_dims`` so this check and the engines share one geometry;
    * a movement commits its declared source's extent;
    * any of those whose DECLARED epilogue contains a pooling stage pools the row axis down.

    ``arg_index`` is the harness ABI: the harness calls ``<kernel>(<declared inputs, in order>,
    <outputs>)``, so the j-th output is argument ``len(inputs) + j``. That order is the HARNESS's, not the
    kernel's — the kernel does not get to choose it.

    FAIL CLOSED: anything not derivable returns ``([], reason)`` and every consumer reports *skipped with
    that reason*, never a pass."""
    a = _attrs(capsule)
    op = _declared_op(capsule)
    ins = (capsule or {}).get("inputs")
    if not isinstance(ins, list):
        return [], "capsule declares no input list, so the harness argument order is not derivable"
    base = len(ins)

    def _commit(name, rows, cols, entry) -> tuple[dict | None, str]:
        eb = _declared_elem_bytes(entry, capsule)
        if eb is None:
            return None, (f"commit {name!r} declares no output_dtype and the capsule declares no numeric "
                          f"policy dtype, so a DRAM byte extent is not derivable")
        if "maxpool" in (entry.get("epilogue") or []):
            rows, why = _pooled_rows(rows, entry)
            if rows is None:
                return None, f"commit {name!r}: {why}"
        return {"name": str(name), "rows": int(rows), "cols": int(cols), "elem_bytes": int(eb)}, ""

    outs: list[dict] = []
    if op == "resident_reuse":
        w = _input_shape(capsule, a.get("weight"), "weight")
        mms = a.get("matmuls")
        if not (isinstance(w, list) and len(w) == 2 and isinstance(mms, list) and mms):
            return [], "declared residency capsule has no [K, N] weight or no matmul list"
        for mm in mms:
            lhs = _input_shape(capsule, mm.get("lhs"))
            if not (isinstance(lhs, list) and len(lhs) == 2) or not mm.get("out"):
                return [], "a declared matmul has no 2-D lhs shape or no output name"
            rec, why = _commit(mm["out"], int(lhs[0]), int(w[1]), mm)
            if rec is None:
                return [], why
            outs.append(rec)
    elif op in ("matmul", "matmul_resident"):
        shape = _declared_output_shape(capsule)
        if shape is None or not a.get("out"):
            return [], "could not derive the declared (M, N) and output name from the declaration"
        rec, why = _commit(a["out"], shape[0], shape[1], a)
        if rec is None:
            return [], why
        outs.append(rec)
    elif op in ("conv2d", "conv"):
        ifm = _input_shape(capsule, a.get("ifm"), "input")
        w = _input_shape(capsule, a.get("weight"), "weight")
        if not (isinstance(ifm, list) and len(ifm) == 4 and isinstance(w, list) and len(w) == 2
                and a.get("out")):
            return [], "declared conv is not [N, H, W, Ci] x [Kh*Kw*Ci, Co] with a named output"
        try:
            from merlin.runtime.commandbuffer import conv_out_dims   # the ONE conv-geometry definition
            Ho, Wo = conv_out_dims(int(ifm[1]), int(ifm[2]), int(a["kh"]), int(a["kw"]),
                                   a.get("stride", [1, 1]), a.get("padding", [0, 0, 0, 0]),
                                   a.get("dilation", [1, 1]))
        except (KeyError, TypeError, ValueError) as e:
            return [], f"declared conv geometry is incomplete ({type(e).__name__})"
        rec, why = _commit(a["out"], int(ifm[0]) * Ho * Wo, int(w[1]), a)
        if rec is None:
            return [], why
        outs.append(rec)
    elif op == "movement":
        src = _input_shape(capsule, a.get("src"), "input")
        if not (isinstance(src, list) and len(src) == 2) or not a.get("out"):
            return [], "declared movement has no 2-D source shape or no output name"
        rec, why = _commit(a["out"], int(src[0]), int(src[1]), a)
        if rec is None:
            return [], why
        outs.append(rec)
    else:
        return [], f"declared op {op or '<absent>'!r} has no derived commit extent in this check"
    for j, rec in enumerate(outs):
        rec["arg_index"] = base + j
    return outs, ""


def declared_contractions(capsule: dict | None) -> list[tuple[int, int, int]]:
    """``[(M, N, K)]`` for every DECLARED matrix product, in declaration order. A convolution contributes
    its im2col form ``(N*Ho*Wo, Co, Kh*Kw*Ci)`` — the contraction the mesh actually has to perform, which
    is the whole point: a convolution lowered as if K were the raw channel count is a different
    operation. Empty when the capsule declares no product (a pure movement)."""
    a = _attrs(capsule)
    op = _declared_op(capsule)
    if op == "resident_reuse":
        w = _input_shape(capsule, a.get("weight"), "weight")
        out = []
        for mm in (a.get("matmuls") or []):
            lhs = _input_shape(capsule, mm.get("lhs"))
            if isinstance(lhs, list) and len(lhs) == 2 and isinstance(w, list) and len(w) == 2:
                out.append((int(lhs[0]), int(w[1]), int(lhs[1])))
        return out
    if op in ("matmul", "matmul_resident"):
        mkn = _declared_mkn(capsule)
        return [mkn] if mkn else []
    if op in ("conv2d", "conv"):
        outs, _ = declared_outputs(capsule)
        try:
            K = int(a["kh"]) * int(a["kw"]) * int(a["ci"])
        except (KeyError, TypeError, ValueError):
            return []
        return [(outs[0]["rows"], outs[0]["cols"], K)] if outs else []
    return []


# ------------------------------------------------------------------- emitted-stream move extraction
def _moves(trace: dict, move_class: str, stride_class: str, stride_key: str) -> list[dict]:
    """Every ``move_class`` instruction with the DRAM row pitch in force at that point.

    The class names callers pass are the SHARED, human-owned semantic vocabulary the decoder maps every
    RoCC target's funct CODES into (the codes are RTL-derived; the vocabulary is the compiler's), which
    is the same vocabulary the checks above this one already use — not a per-target spelling. A
    self-hosted-ISA target never reaches this module at all: the runner routes it to the kernel-decode
    check instead.

    The pitch is the one the author's OWN most recent ``stride_class`` command configured (its decoded
    ``stride_key``): a move's DRAM operand is a byte offset from a kernel argument, and turning that into
    a (row, column) needs the pitch the kernel itself declared. A move reached with no pitch configured,
    or missing an extent, keeps ``None`` in that field — the caller reports UNKNOWN and asserts nothing,
    rather than assuming a pitch."""
    stride = None
    config: dict = {}
    out: list[dict] = []
    for i in trace.get("instructions", []):
        d = i.get("decoded") or {}
        if i.get("class") == stride_class:
            v = d.get(stride_key)
            stride = int(v) if isinstance(v, int) and v > 0 else None
            config = dict(d)
        elif i.get("class") == move_class:
            dram = d.get("dram") if isinstance(d.get("dram"), dict) else {}
            out.append({"index": i.get("index"), "kind": dram.get("kind"),
                        "arg_index": dram.get("arg_index"), "offset": dram.get("offset"),
                        "rows": d.get("rows"), "cols": d.get("cols"), "stride": stride,
                        "config": dict(config)})
    return out


def _effective_store_rows(store: dict, capsule: dict | None,
                          rtl_facts: dict) -> tuple[int | None, str]:
    """Rows physically committed by one MVOUT under its active CONFIG_ST.

    In native pooling mode the StoreController ignores MVOUT.num_rows and walks porows*pocols output
    positions itself.  We use that replacement only after every encoded geometry field agrees with the
    declaration; otherwise the footprint is UNKNOWN, never silently interpreted as zero rows.
    """
    if "maxpool" not in (_attrs(capsule).get("epilogue") or []):
        rows = store.get("rows")
        return (int(rows), "") if isinstance(rows, int) else (None, "MVOUT rows are undecodable")
    ok, why, want = _pool_config_agreement(store.get("config"), capsule, rtl_facts)
    if not ok:
        return None, why
    return int(want["rows_per_store"]), ""


def _store_coverage(trace: dict, outputs: list[dict], capsule: dict | None = None,
                    rtl_facts: dict | None = None) -> dict:
    """Per-declared-output DRAM store coverage, computed from the author's own stores.

    ``{name: {"status": "covered"|"uncovered"|"absent"|"unknown", ...}}``. Row/column coverage is tracked
    as SETS OF INDICES so a kernel that legalizes a ragged extent by any means (a shorter final band, or
    many one-row bands) is judged by what it actually covers, not by the shape of its loop."""
    stores = _moves(trace, "MVOUT", "CONFIG_ST", "out_stride_bytes")
    for store in stores:
        store["rows"], store["rows_unknown_reason"] = _effective_store_rows(
            store, capsule, rtl_facts or {})
    rep: dict[str, dict] = {}
    for o in outputs:
        mine = [s for s in stores if s["kind"] == "argbase" and s["arg_index"] == o["arg_index"]]
        rec: dict = {"n_stores_total": len(stores), "n_stores_to_output": len(mine),
                     "arg_index": o["arg_index"], "extent": [o["rows"], o["cols"]],
                     "arg_indices_stored_to": sorted({s["arg_index"] for s in stores
                                                      if s["arg_index"] is not None}),
                     "n_baked_address_stores": sum(1 for s in stores if s["kind"] == "const")}
        if not mine:
            # "No store addresses this output" is a CLAIM, and it is only sound when every store's
            # address operand was decodable. A store whose DRAM operand the decoder could not resolve
            # (no `dram` provenance at all) might be the missing one, so it makes the answer UNKNOWN
            # rather than making the output absent — the difference between "your kernel dropped this
            # store" and "our decoder could not read your addresses", which must never be collapsed.
            blind = [s["index"] for s in stores if s["kind"] not in ("argbase", "const")]
            if blind:
                rec.update(status="unknown", undecodable_instruction_indices=blind[:8],
                           unknown_reason="store(s) carry no decodable DRAM address operand")
            else:
                rec["status"] = "absent"
            rep[o["name"]] = rec
            continue
        undecodable = [s["index"] for s in mine
                       if not isinstance(s["stride"], int)
                       or not all(isinstance(s[k], int) for k in ("offset", "rows", "cols"))]
        if undecodable:
            rec.update(status="unknown", undecodable_instruction_indices=undecodable[:8],
                       unknown_reason="; ".join(sorted({
                           str(s.get("rows_unknown_reason") or
                               "store carries no decodable offset/rows/cols/row-pitch")
                           for s in mine if s["index"] in undecodable
                       })))
            rep[o["name"]] = rec
            continue
        R, C, eb = o["rows"], o["cols"], o["elem_bytes"]
        # Exact coverage by a ROW-STRIP sweep rather than a per-cell set: a whole-model commit can be
        # millions of cells, and a check that runs out of memory on a big shape is a check that does not
        # run. Strip boundaries come from the stores themselves, so the sweep is O(stores^2) at worst and
        # trivial for the tens of stores a real kernel emits.
        tiles = [(s_["offset"] // s_["stride"], (s_["offset"] % s_["stride"]) // eb,
                  s_["rows"], s_["cols"], s_["index"]) for s_ in mine]
        overrun = [{"instruction_index": ix, "rows": [r0, r0 + nr], "cols": [c0, c0 + nc]}
                   for (r0, c0, nr, nc, ix) in tiles if r0 + nr > R or c0 + nc > C]
        bounds = sorted({0, R} | {b for (r0, _c, nr, _nc, _i) in tiles
                                  for b in (r0, r0 + nr) if 0 <= b <= R})
        covered_cells = 0
        first_missing = None
        for a_, b_ in zip(bounds, bounds[1:]):
            if b_ <= a_:
                continue
            spans = sorted((max(c0, 0), min(c0 + nc, C)) for (r0, c0, nr, nc, _ix) in tiles
                           if r0 <= a_ and r0 + nr >= b_)
            merged: list[list[int]] = []
            for lo_, hi_ in spans:
                if hi_ <= lo_:
                    continue
                if merged and lo_ <= merged[-1][1]:
                    merged[-1][1] = max(merged[-1][1], hi_)
                else:
                    merged.append([lo_, hi_])
            width = sum(hi_ - lo_ for lo_, hi_ in merged)
            covered_cells += (b_ - a_) * width
            if width < C and first_missing is None:
                gap = 0
                for lo_, hi_ in merged:
                    if gap < lo_:
                        break
                    gap = max(gap, hi_)
                first_missing = (a_, gap)
        # A row/column "band nothing writes" is one no store's extent reaches at all — the shape a
        # dropped tail band takes. Reporting every merely-PARTIALLY covered index instead names both
        # axes for a one-axis defect, which reads as noise.
        touched_rows = {r for (r0, _c0, nr, _nc, _ix) in tiles
                        for r in range(max(r0, 0), min(r0 + nr, R))}
        touched_cols = {c for (_r0, c0, _nr, nc, _ix) in tiles
                        for c in range(max(c0, 0), min(c0 + nc, C))}
        miss_rows = [r for r in range(R) if r not in touched_rows]
        miss_cols = [c for c in range(C) if c not in touched_cols]
        rec.update(status="covered" if covered_cells == R * C and not overrun else "uncovered",
                   covered_cells=covered_cells, declared_cells=R * C, overrunning_stores=overrun[:8],
                   uncovered_rows=miss_rows[:16], uncovered_cols=miss_cols[:16],
                   first_uncovered_cell=list(first_missing) if first_missing else None)
        rep[o["name"]] = rec
    return rep


def _name_tokens(name: str) -> set[str]:
    """Identifier -> its lowercase underscore-separated TOKENS (never substrings, so ``ws`` can never
    match inside ``rows``). Same discipline as the capability-discovery lexicons."""
    return {t for t in str(name).lower().replace("-", "_").split("_") if t}


#: Word stems that mark an instruction class, in a target's OWN funct->class vocabulary, as a HARDWARE
#: LOOP performing a whole tiled nest in one command (so the per-tile instructions a software nest emits
#: are legitimately absent). Generic dataflow words in the vocabulary any accelerator's interface uses;
#: whether this target HAS one, and at which funct code, is derived from its own facts.
_LOOP_STEMS = ("loop",)
#: …and the stem that additionally marks such a loop as the CONVOLUTION loop (window gather in hardware).
_CONV_STEMS = ("conv",)


def _vocabulary_classes(target: str, stems: tuple[str, ...]) -> tuple[set[str], str]:
    """Instruction-class names in the TARGET'S OWN funct->class vocabulary whose tokens match ``stems``.

    The vocabulary is the target's reviewed funct-CODE -> class map — the same map the decoder classified
    THIS trace with — so the names compared here are this target's own names, never a class list written
    down in advance. An underivable vocabulary returns an empty set WITH a reason; callers then report
    UNKNOWN and assert nothing, never 'absent'."""
    try:
        from .rocc.decode import funct_class_for
        vocab = funct_class_for(target)
    except Exception as e:  # noqa: BLE001 - not derivable -> UNKNOWN, never a substituted default
        return set(), f"UNKNOWN (funct->class vocabulary not derivable: {type(e).__name__})"
    return ({str(c) for c in vocab.values() if _name_tokens(c) & set(stems)},
            f"derived from the target's funct->class map ({len(vocab)} codes)")


# ----------------------------------------------------------------------------- new structural checks
def _check_pool_config(trace: dict, capsule: dict | None, rtl_facts: dict) -> Check:
    """Native CONFIG_ST pooling mode must encode the declared geometry exactly.

    This gate is intentionally separate from coverage: a malformed or underived config must make the
    pooled footprint UNKNOWN, not let the checker manufacture a plausible coverage result from capsule
    intent alone.
    """
    cid = "T0.pool_config"
    stores = _moves(trace, "MVOUT", "CONFIG_ST", "out_stride_bytes")
    declared = "maxpool" in (_attrs(capsule).get("epilogue") or [])
    if not declared:
        enabled = [s["index"] for s in stores
                   if isinstance((s.get("config") or {}).get("pool_stride"), int)
                   and s["config"]["pool_stride"] > 0]
        if enabled:
            return Check(cid, "T0", "warn", "fail",
                         "CONFIG_ST enables native pooling although no maxpool epilogue is declared",
                         expected="pool_stride == 0", got="pool_stride > 0",
                         evidence={"instruction_indices": enabled[:8]})
        return Check(cid, "T0", "warn", "skipped", "no maxpool epilogue is declared")
    supported = rtl_facts.get("max_pool_supported")
    if supported is False:
        return Check(cid, "T0", "warn", "fail",
                     "a maxpool epilogue is declared, but the RTL-derived elaborated configuration "
                     "disables native max-pool",
                     expected=True, got=False)
    if supported is not True:
        return Check(cid, "T0", "warn", "skipped",
                     "native max-pool capability is UNKNOWN because no derived elaborated-config "
                     "feature fact is available")
    want, declaration_why = _declared_pool_geometry(capsule)
    if want is None:
        return Check(cid, "T0", "warn", "fail",
                     f"declared maxpool geometry cannot be represented/derived: {declaration_why}",
                     expected="representable native pool geometry", got=declaration_why)
    if not stores:
        return Check(cid, "T0", "warn", "skipped",
                     "no MVOUT exists at which to observe the active CONFIG_ST pooling fields")
    bad = []
    for store in stores:
        ok, why, _ = _pool_config_agreement(store.get("config"), capsule, rtl_facts)
        if not ok:
            bad.append({"instruction_index": store["index"], "reason": why,
                        "decoded_config": {f: (store.get("config") or {}).get(f)
                                           for f in _POOL_CONFIG_FIELDS}})
    if bad:
        return Check(cid, "T0", "warn", "fail",
                     f"{len(bad)} pooled MVOUT(s) do not carry a fully derived CONFIG_ST matching "
                     "the declared geometry; their store footprint is UNKNOWN",
                     expected={f: want[f] for f in _POOL_CONFIG_FIELDS}, got=bad[0]["decoded_config"],
                     evidence={"mismatches": bad[:8]},
                     fix_hint="encode CONFIG_ST pooling fields from the declared geometry using the "
                              "target's RTL-derived ConfigMvoutRs1 layout")
    return Check(cid, "T0", "warn", "pass",
                 f"all {len(stores)} pooled MVOUT(s) use CONFIG_ST geometry matching the declaration",
                 expected={f: want[f] for f in _POOL_CONFIG_FIELDS},
                 got={f: want[f] for f in _POOL_CONFIG_FIELDS})


def _check_output_store_coverage(trace: dict, capsule: dict | None, rtl_facts: dict) -> Check:
    """Every DECLARED output must be covered by stores in the emitted stream, and by no store past it.

    This is detectable with NO simulation, and it is the failure mode that reads as ``mismatch_count > 0``
    with ``max_abs_error == 0``: bytes that were never produced have no value to diff, so the numeric
    plane reports what looks like an arithmetic error for what is really a DROPPED or MIS-ADDRESSED
    store. Everything it needs is a declaration or the author's own artifact: the extent from the declared
    shapes, the output's kernel argument from the harness ABI, the DRAM row pitch from the author's own
    store configuration, and the store extents from the author's own instructions.

    ANSWER-FREE: it says which part of the declared output nothing writes. It never reads, computes, or
    hints at a value."""
    outs, why = declared_outputs(capsule)
    if not outs:
        return Check("T0.output_store_coverage", "T0", "warn", "skipped",
                     f"declared output extent not derivable: {why}")
    cov = _store_coverage(trace, outs, capsule, rtl_facts)
    unknown = {n: r for n, r in cov.items() if r["status"] == "unknown"}
    if unknown:
        return Check("T0.output_store_coverage", "T0", "warn", "skipped",
                     f"DRAM coverage of output(s) {sorted(unknown)} is UNKNOWN — asserted neither way ("
                     + "; ".join(f"{n}: {r.get('unknown_reason')}" for n, r in unknown.items()) + ")",
                     evidence=unknown)
    absent = [n for n, r in cov.items() if r["status"] == "absent"]
    bad = [n for n, r in cov.items() if r["status"] == "uncovered"]
    n_ok = len(outs) - len(absent) - len(bad)
    if absent:
        r0 = cov[absent[0]]
        seen = r0["arg_indices_stored_to"]
        baked = r0["n_baked_address_stores"]
        return Check("T0.output_store_coverage", "T0", "warn", "fail",
                     f"declared output(s) {absent} have NO store addressing them in the emitted stream "
                     f"(your kernel stores to {n_ok} of {len(outs)} declared outputs). Under the harness "
                     f"ABI — the declared inputs in order, then the outputs — "
                     + ", ".join(f"{n} is kernel argument #{cov[n]['arg_index']}" for n in absent)
                     + f"; the {r0['n_stores_total']} store(s) present address argument(s) {seen or '[]'}"
                     + (f", of which {baked} use a baked constant address" if baked else "")
                     + ". This is a DROPPED or MIS-ADDRESSED store, not an arithmetic problem: those "
                       "bytes are never produced, so nothing downstream has anything to diff.",
                     expected=f"a store to argument(s) "
                              f"{[cov[n]['arg_index'] for n in absent]}",
                     got=f"stores to {seen or '[]'}", evidence={n: cov[n] for n in absent},
                     fix_hint="address every declared output's store from ITS kernel-argument pointer, "
                              "and make sure the commit is not optimized away before the readback")
    if bad:
        parts = []
        for n in bad:
            r = cov[n]
            if r["overrunning_stores"] and r["covered_cells"] == r["declared_cells"]:
                parts.append(f"{n} ({r['extent'][0]}x{r['extent'][1]}): "
                             f"{len(r['overrunning_stores'])} store(s) write PAST the declared extent — "
                             f"a whole tile stored over an extent the tile does not divide writes into "
                             f"memory this buffer does not own")
            else:
                where = []
                if r["uncovered_rows"]:
                    where.append(f"row band {r['uncovered_rows']} is never written by any store")
                if r["uncovered_cols"]:
                    where.append(f"column band {r['uncovered_cols']} is never written by any store")
                if not where:
                    where.append(f"the first cell nothing writes is {r['first_uncovered_cell']}")
                parts.append(f"{n} ({r['extent'][0]}x{r['extent'][1]}): stores cover "
                             f"{r['covered_cells']} of {r['declared_cells']} declared cells; "
                             + "; ".join(where))
        return Check("T0.output_store_coverage", "T0", "warn", "fail",
                     "declared output(s) not exactly covered by the emitted stores: " + "; ".join(parts),
                     expected="every declared cell stored exactly once, none past the extent",
                     got=f"{len(bad)} of {len(outs)} output(s) mis-covered",
                     evidence={n: cov[n] for n in bad},
                     fix_hint="extend the store loop over every declared band (including the final "
                              "partial one) and clamp each band's store extent to the declared remainder")
    return Check("T0.output_store_coverage", "T0", "warn", "pass",
                 f"all {len(outs)} declared output(s) "
                 + ", ".join(f"{o['name']} ({o['rows']}x{o['cols']})" for o in outs)
                 + " are fully covered by the emitted stores, with no store past the declared extent",
                 evidence=cov)


def _check_extent_tile_legalization(trace: dict, capsule: dict | None, rtl_facts: dict,
                                    target: str) -> Check:
    """A DECLARED extent the RTL-derived tile edge does not divide must be LEGALIZED.

    Three axes, each measured against the edge DERIVED from the introspected array geometry (never a
    constant — a target with a different array is screened against its own edge, and a target whose array
    is UNKNOWN is skipped with that reason rather than screened against a guess):

    * the committed ROWS and COLUMNS — legalized iff the author's own stores cover the declared band
      EXACTLY (a shorter final band, many short bands, anything: the test is coverage, not loop shape);
    * the CONTRACTION LENGTH — legalized iff the emitted compute steps reach the count the tile geometry
      forces, ``ceil(M/edge_rows) * ceil(K/edge_rows) * ceil(N/edge_cols)`` summed over the declared
      products. A ragged K whose final partial band is never accumulated lands exactly here: the stream
      is well-formed, every store covers its output, and one slice of the sum is simply missing.

    A hardware loop that walks the nest itself legalizes all three, so its presence (derived from the
    target's own class vocabulary) is accepted in place of the per-tile evidence."""
    mesh = _mesh(rtl_facts)
    if mesh is None:
        return Check("T0.extent_tile_legalization", "T0", "warn", "skipped",
                     "mesh dims UNKNOWN (no RTL facts derived) — no tile edge to legalize against")
    mr, mc = mesh
    outs, why = declared_outputs(capsule)
    contractions = declared_contractions(capsule)
    if not outs and not contractions:
        return Check("T0.extent_tile_legalization", "T0", "warn", "skipped",
                     f"no declared extent to legalize: {why}")
    axes: list[dict] = []
    for o in outs:
        axes.append({"output": o["name"], "axis": "committed rows", "extent": o["rows"], "edge": mr,
                     "edge_from": "RTL facts arrays[mesh].rows"})
        axes.append({"output": o["name"], "axis": "committed columns", "extent": o["cols"], "edge": mc,
                     "edge_from": "RTL facts arrays[mesh].cols"})
    for (M, N, K) in contractions:
        axes.append({"output": None, "axis": "contraction length", "extent": K, "edge": mr,
                     "edge_from": "RTL facts arrays[mesh].rows"})
    ragged = [a for a in axes if a["edge"] and a["extent"] % a["edge"]]
    if not ragged:
        return Check("T0.extent_tile_legalization", "T0", "warn", "pass",
                     f"every declared extent is a whole multiple of the RTL-derived {mr}x{mc} tile edge",
                     evidence={"axes": axes})
    loops, loop_basis = _vocabulary_classes(target, _LOOP_STEMS)
    in_trace = loops & set(_classes(trace))
    if in_trace:
        return Check("T0.extent_tile_legalization", "T0", "warn", "pass",
                     f"the ragged extent(s) {[a['axis'] for a in ragged]} are legalized by the hardware "
                     f"loop {sorted(in_trace)}, which walks the tail itself",
                     evidence={"axes": axes, "loop_classes": sorted(in_trace), "basis": loop_basis})
    cov = _store_coverage(trace, outs, capsule, rtl_facts)
    n_compute = sum(1 for c in _classes(trace) if c in _COMPUTE_CLASSES)
    min_compute = sum(math.ceil(M / mr) * math.ceil(K / mr) * math.ceil(N / mc)
                      for (M, N, K) in contractions)
    unlegalized: list[dict] = []
    unknown: list[dict] = []
    for a in ragged:
        rem = a["extent"] % a["edge"]
        rec = dict(a, remainder=rem)
        if a["axis"] == "contraction length":
            if not n_compute:
                # This trace names no instruction class this module recognizes as a matrix-compute
                # step, so the count bound has nothing to count. That is a limit of THIS check's
                # vocabulary on THIS target, not evidence about the kernel — report UNKNOWN. (A
                # genuinely compute-free contraction is already an ERROR from the movement/compute
                # balance check, so nothing is lost by staying quiet here.)
                unknown.append(dict(rec, reason="no recognized matrix-compute class in this trace"))
            elif contractions and n_compute < min_compute:
                rec.update(min_compute_steps=min_compute, observed_compute=n_compute)
                unlegalized.append(rec)
            continue
        c = cov.get(a["output"])
        if c is None or c["status"] == "unknown":
            unknown.append(rec)
        elif c["status"] != "covered":                 # 'absent' carries no cell counts; see below
            rec.update(covered_cells=c.get("covered_cells"), declared_cells=c.get("declared_cells"),
                       uncovered_rows=c.get("uncovered_rows"), uncovered_cols=c.get("uncovered_cols"),
                       overrunning_stores=c.get("overrunning_stores"))
            unlegalized.append(rec)
    if unlegalized:
        parts = []
        for u in unlegalized:
            head = (f"{u['axis']}" + (f" of {u['output']}" if u["output"] else "") +
                    f" is {u['extent']}, which the RTL-derived tile edge {u['edge']} ({u['edge_from']}) "
                    f"does not divide — the final band is {u['remainder']} wide")
            if u["axis"] == "contraction length":
                parts.append(head + f", and the stream carries {u['observed_compute']} compute step(s) "
                                    f"where the tile geometry needs at least {u['min_compute_steps']}: "
                                    f"a partial band of the sum is never accumulated")
            elif u.get("declared_cells") is None:
                parts.append(head + ", and no store in the emitted stream addresses that output at all")
            else:
                parts.append(head + f", and the emitted stores do not cover it exactly "
                                    f"({u.get('covered_cells')} of {u.get('declared_cells')} cells, "
                                    f"{len(u.get('overrunning_stores') or [])} store(s) past the extent)")
        return Check("T0.extent_tile_legalization", "T0", "warn", "fail",
                     "ragged extent(s) not legalized: " + "; ".join(parts),
                     expected="every ragged axis covered exactly (or walked by the hardware loop)",
                     got=f"{len(unlegalized)} unlegalized axis/axes",
                     evidence={"axes": axes, "unlegalized": unlegalized,
                               "loop_classes_available": sorted(loops), "basis": loop_basis},
                     fix_hint="emit the final band at its true remainder extent (pad the operand if you "
                              "must, but clamp the store); a whole tile over a ragged extent either "
                              "drops the tail or writes past it, and both are silent")
    if unknown:
        return Check("T0.extent_tile_legalization", "T0", "warn", "skipped",
                     f"ragged axis/axes {[u['axis'] for u in unknown]} could not be judged: the stores "
                     f"carry no decodable extent or row pitch — UNKNOWN, asserted neither way",
                     evidence={"axes": axes, "unknown": unknown})
    return Check("T0.extent_tile_legalization", "T0", "warn", "pass",
                 f"every ragged extent {[(a['axis'], a['extent']) for a in ragged]} is legalized against "
                 f"the RTL-derived {mr}x{mc} tile edge", evidence={"axes": axes})


# ----------------------------------------------- kernel-argument ABI, resolved from the ABI CONTRACT
class _Unresolvable(Exception):
    """A contract arg_order token this resolver cannot expand against a buffer. Raised, never guessed:
    an argument order resolved wrongly would bind every field comparison to the wrong tensor."""


def _kernel_abi_rows() -> tuple[list[dict], dict, str]:
    """``(arg_order_by_command_shape rows, arg_order_tokens, reason)`` from the OOT backend ABI contract.

    The contract is the SINGLE source of truth for the pointer list: there is not one argument order but
    one per command shape, and the contract carries both the prose and a machine-checkable ``order``
    token list per shape. Nothing about the order is written down here. ``([], {}, reason)`` when the
    contract cannot be read — the consuming check then reports *skipped with that reason*."""
    from merlin.common.paths import data_path
    path = data_path("contract", "mlir_oot_backend_contract.yaml")
    if not path.is_file():
        return [], {}, f"the kernel-ABI contract is not readable at {path.name}, so the argument order " \
                       f"the harness calls with is not derivable"
    try:
        import yaml
        doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as e:  # noqa: BLE001 — unreadable contract -> UNKNOWN, never an assumed order
        return [], {}, f"the kernel-ABI contract could not be parsed ({type(e).__name__}), so the " \
                       f"argument order is not derivable"
    abi = doc.get("kernel_abi") or {}
    rows = abi.get("arg_order_by_command_shape")
    tokens = abi.get("arg_order_tokens") or {}
    if not isinstance(rows, list) or not rows:
        return [], {}, "the kernel-ABI contract declares no arg_order_by_command_shape rows, so the " \
                       "argument order is not derivable"
    return rows, tokens, ""


def _resident_groups(cb: dict) -> list[tuple[str, list[tuple[str, str]]]]:
    """``[(weight, [(lhs, committed_out), ...]), ...]`` in resident-pack order, group-major per group —
    the grouping the contract's resident_matmul tokens are written against."""
    res_to_weight: dict[str, str] = {}
    order: list[str] = []
    for cmd in cb.get("commands") or []:
        if cmd.get("opcode") != "RES_PACK":
            continue
        ops = cmd.get("operands") or {}
        dst, src = ops.get("dst"), ops.get("src")
        if not isinstance(dst, str) or not isinstance(src, str):
            raise _Unresolvable("a RES_PACK command names no src/dst pair")
        if dst not in res_to_weight:
            res_to_weight[dst] = src
            order.append(dst)
    if not order:
        raise _Unresolvable("the buffer carries no RES_PACK command")
    commit_of = {(c.get("operands") or {}).get("src"): (c.get("operands") or {}).get("dst")
                 for c in (cb.get("commands") or []) if c.get("opcode") == "COMMIT"}
    jobs: dict[str, list[tuple[str, str]]] = {res: [] for res in order}
    for cmd in cb.get("commands") or []:
        if cmd.get("opcode") not in ("MATMUL", "MATMUL_RESIDENT"):
            continue
        ops = cmd.get("operands") or {}
        res, lhs, out = ops.get("rhs"), ops.get("lhs"), commit_of.get((ops.get("dst")))
        if res not in jobs:
            raise _Unresolvable(f"a matmul's rhs {res!r} resolves to no resident weight")
        if not isinstance(lhs, str) or not isinstance(out, str):
            raise _Unresolvable("a matmul has no lhs and/or no committed output")
        jobs[res].append((lhs, out))
    if not any(jobs.values()):
        raise _Unresolvable("the buffer carries no matmul over a resident weight")
    return [(res_to_weight[res], jobs[res]) for res in order]


def _movement_command(cb: dict) -> dict:
    for cmd in cb.get("commands") or []:
        if cmd.get("opcode") == "MOVEMENT":
            return cmd
        if cmd.get("opcode") == "VECTOR_MAP" and (cmd.get("attributes") or {}).get("combine") == "identity":
            return cmd
    raise _Unresolvable("the buffer carries no MOVEMENT / identity VECTOR_MAP command")


#: The external tensor roles the contract's ``interface_external_tensors_in_declaration_order`` token
#: names. Roles are the target-INDEPENDENT command-buffer ABI's own vocabulary (one validator for every
#: target), not a per-target fact.
_EXTERNAL_ROLES = ("input", "weight", "bias", "output")


def _resolve_arg_order_token(token: str, cb: dict) -> list[str]:
    """The pointer argument name(s) one contract ``order`` token expands to for ``cb``.

    One token, one resolution rule. A token this function does not know is a REFUSAL — so when the
    contract grows a shape, the consuming check reports UNKNOWN with that reason rather than screening
    a stream against an order it invented."""
    if token == "movement_src":
        ops = _movement_command(cb).get("operands") or {}
        name = ops.get("src") or ops.get("lhs")
        if not isinstance(name, str):
            raise _Unresolvable("the movement command names no src/lhs")
        return [name]
    if token == "movement_dst":
        name = (_movement_command(cb).get("operands") or {}).get("dst")
        if not isinstance(name, str):
            raise _Unresolvable("the movement command names no dst")
        return [name]
    if token == "interface_external_tensors_in_declaration_order":
        names = [n for n, spec in (cb.get("tensors") or {}).items()
                 if (spec or {}).get("role") in _EXTERNAL_ROLES]
        if not names:
            raise _Unresolvable("the buffer declares no external tensor")
        return names
    if token == "resident_weights_in_resident_pack_order":
        return [w for w, _jobs in _resident_groups(cb)]
    if token == "matmul_lhs_group_major":
        return [lhs for _w, jobs in _resident_groups(cb) for lhs, _o in jobs]
    if token == "commit_outputs_group_major":
        return [out for _w, jobs in _resident_groups(cb) for _l, out in jobs]
    raise _Unresolvable(f"the ABI contract names an arg_order token {token!r} this resolver does not "
                        f"know how to expand")


def resolve_kernel_arg_order(command_buffer: dict | None) -> tuple[list[str], str, str]:
    """``(argument names in harness call order, contract shape name, reason)`` for a command buffer.

    Which argument order applies DEPENDS ON THE COMMAND SHAPE — the contract carries one row per shape,
    and the harness dispatches between them. This resolves the buffer against every row's ``order``
    tokens and demands a UNIQUE answer:

    * a row whose declared ``opcodes`` list is present applies only when the buffer carries exactly one
      command with one of those opcodes (the contract's own discriminator, read as data);
    * any other row applies when every one of its ``order`` tokens expands;
    * if NO row applies, or if MORE THAN ONE does, the result is ``([], "", reason)`` — UNKNOWN. Two
      harness paths exist and a buffer that could be called either way must not be screened against a
      guess. UNKNOWN is also returned when the same tensor would occupy two argument slots, since the
      binding would then be ambiguous.

    Nothing about any order is written here: the rows, their tokens and the whole-op opcode list are all
    read from the contract."""
    if not command_buffer:
        return [], "", ("no command buffer was passed alongside the trace, so which declared tensor each "
                        "kernel argument carries is not derivable")
    if not isinstance(command_buffer.get("commands"), list) or not command_buffer["commands"]:
        return [], "", "the command buffer declares no commands, so no argument order is derivable"
    if not isinstance(command_buffer.get("tensors"), dict) or not command_buffer["tensors"]:
        return [], "", "the command buffer declares no tensors, so no argument order is derivable"
    rows, _tokens, why = _kernel_abi_rows()
    if not rows:
        return [], "", why
    opcodes_present = [c.get("opcode") for c in command_buffer["commands"]]
    applicable: list[tuple[str, list[str]]] = []
    refusals: list[str] = []
    for row in rows:
        shape = str(row.get("shape") or "<unnamed>")
        order = row.get("order")
        if not isinstance(order, list) or not order:
            refusals.append(f"{shape}: the contract row carries no machine-checkable order token list")
            continue
        gate = row.get("opcodes")
        if isinstance(gate, list) and gate:
            hits = [o for o in opcodes_present if o in gate]
            if len(hits) != 1 or len(opcodes_present) != 1:
                refusals.append(f"{shape}: the buffer does not carry exactly one of the shape's "
                                f"declared opcodes {sorted(gate)}")
                continue
        try:
            names: list[str] = []
            for token in order:
                names.extend(_resolve_arg_order_token(str(token), command_buffer))
        except _Unresolvable as e:
            refusals.append(f"{shape}: {e}")
            continue
        applicable.append((shape, names))
    if not applicable:
        return [], "", ("no kernel-ABI contract shape resolves against this command buffer, so the "
                        "argument order the harness calls with is UNKNOWN — " + "; ".join(refusals))
    if len(applicable) > 1:
        return [], "", ("this command buffer resolves against " + str(len(applicable)) +
                        " kernel-ABI contract shapes (" + ", ".join(s for s, _ in applicable) +
                        "), so which argument order the harness calls with is UNKNOWN; it is not "
                        "screened against a guess")
    shape, names = applicable[0]
    if len(set(names)) != len(names):
        return [], "", (f"the {shape} argument order resolves to {names}, in which a tensor occupies two "
                        f"argument slots, so the per-argument binding is ambiguous")
    return names, shape, ""


def kernel_argument_binding(command_buffer: dict | None,
                            capsule: dict | None) -> tuple[list[dict], str, str]:
    """``(binding, contract shape, reason)`` — every kernel argument's DECLARED tensor, in the order the
    ABI contract says the harness calls with.

    ``[{index, name, role, rows, cols, elem_bytes, packed_bytes}]``. The order comes from
    :func:`resolve_kernel_arg_order` (the contract, dispatched on the buffer's command shape);
    ``rows`` x ``cols`` is the declared shape flattened to the 2-D form a DRAM move addresses (``cols``
    the trailing dim, ``rows`` the product of the leading dims), and ``elem_bytes`` its declared dtype's
    width — all three read off the command buffer's own declaration.

    ``capsule`` is used ONLY as a cross-check of the pointer count where the capsule's declaration makes
    one available; it never supplies the order (the capsule's interface declaration order is precisely
    what a package must NOT assume the harness uses).

    FAIL CLOSED: ``([], shape, reason)`` whenever the order is UNKNOWN, a named tensor is missing from
    the declaration, carries no shape, or has a dtype whose byte width is unknown."""
    names, shape, why = resolve_kernel_arg_order(command_buffer)
    if not names:
        return [], shape, why
    tensors = (command_buffer or {}).get("tensors") or {}
    binding: list[dict] = []
    for index, name in enumerate(names):
        spec = tensors.get(name)
        if not isinstance(spec, dict):
            return [], shape, (f"the {shape} argument order names {name!r} at argument {index}, which the "
                               f"command buffer does not declare as a tensor")
        srec = spec.get("shape")
        if not isinstance(srec, list) or not srec:
            return [], shape, f"declared tensor {name!r} carries no shape, so its DRAM extent is not " \
                              f"derivable"
        elem_bytes = _DTYPE_BYTES.get(str(spec.get("dtype")).lower())
        if elem_bytes is None:
            return [], shape, (f"declared tensor {name!r} has dtype {spec.get('dtype')!r}, whose byte "
                               f"width this check does not know, so its DRAM extent is not derivable")
        rows = 1
        for dim in srec[:-1]:
            rows *= int(dim)
        binding.append({"index": index, "name": str(name), "role": str(spec.get("role")),
                        "rows": int(rows), "cols": int(srec[-1]), "elem_bytes": int(elem_bytes),
                        "packed_bytes": int(rows) * int(srec[-1]) * int(elem_bytes)})
    return binding, shape, ""


def _declared_commit(command_buffer: dict | None) -> tuple[dict | None, str]:
    """The single command whose declared attributes configure the store path (its ``epilogue``), or
    ``(None, reason)``. One such command means each store-path configuration field has ONE intended
    value; two or more share one configuration in the stream, and which commit a given configuration was
    meant for is not derivable — so those fields are compared for neither."""
    if not command_buffer:
        return None, "no command buffer was passed alongside the trace"
    cmds = [c for c in (command_buffer.get("commands") or [])
            if isinstance((c.get("attributes") or {}).get("epilogue"), list)]
    if not cmds:
        return None, "no declared command carries an epilogue, so no store-path intent is declared"
    if len(cmds) > 1:
        return None, (f"{len(cmds)} declared commands carry an epilogue and share one store-path "
                      f"configuration in the stream, so which commit a configuration was meant for is "
                      f"not derivable")
    return cmds[0].get("attributes") or {}, ""


def _binding_permutation(binding: list[dict], volume: dict[int, int]) -> list[dict]:
    """The argument slots whose EMITTED DRAM byte volume is another slot's DECLARED extent.

    A kernel compiled against a permuted pointer list moves each tensor's own byte volume, just through
    the wrong pointer — so the volumes are a PERMUTATION of the declared extents. Computed purely as
    data: a slot is reported only when its volume differs from its own declared extent and matches
    exactly ONE other slot's, and only when the resulting map is a non-identity permutation (every
    reported slot is also somebody's target). Anything less exact reports nothing — an argument order is
    not guessed from a coincidence."""
    by_index = {b["index"]: b for b in binding}
    declared = {b["index"]: b["packed_bytes"] for b in binding}
    mapping: dict[int, int] = {}
    for arg, moved in sorted(volume.items()):
        b = by_index.get(arg)
        if b is None or moved <= 0:
            continue                      # a slot no move reached carries no evidence either way
        if moved == declared.get(arg):
            continue                      # this slot receives exactly its own tensor's volume
        matches = [i for i, v in declared.items() if v == moved and i != arg]
        if len(matches) != 1:
            return []                     # ambiguous or unexplained -> assert nothing
        mapping[arg] = matches[0]
    if not mapping or set(mapping) != set(mapping.values()):
        return []                         # not a closed permutation -> assert nothing
    return [{"arg_index": a, "declared_tensor": by_index[a]["name"],
             "declared_bytes": declared[a], "moved_bytes": volume[a],
             "is_the_declared_extent_of": by_index[t]["name"], "at_arg_index": t}
            for a, t in sorted(mapping.items())]


def _check_encoded_field_intent(trace: dict, capsule: dict | None, rtl_facts: dict,
                                command_buffer: dict | None) -> Check:
    """Does the pointer each emitted instruction dereferences carry the tensor the ABI says it does?

    The hole this fills: a correctly-NAMED instruction carrying a wrong FIELD is not an illegal
    instruction, so the ISA linter passes it. And the numeric and trace planes both read the command
    buffer — the DECLARATION — so a package whose command buffer is right and whose emitted hardware
    encoding is wrong passes both and then diverges on the oracle, which can report it as nothing but a
    value error.

    THE INVARIANT: the DRAM base pointer field of every move must be the argument slot the ABI
    contract's ``arg_order`` (for THIS buffer's command shape) puts that move's tensor in. Which order
    applies is resolved from the contract, never assumed — there is one order per command shape and more
    than one harness path, so a buffer that resolves against no row, or against two, yields UNKNOWN with
    that reason rather than a screening against a guess.

    Two derivable, answer-free signals say the pointer list is permuted, and NEITHER is a claim about
    addressing arithmetic:

    * a slot's accesses leave the declared extent of the tensor the ABI binds to it — reading memory the
      declaration does not give it, under the most permissive layout the declared shape admits (rows and
      columns each rounded up to the RTL-derived array edge);
    * the DRAM byte volume the slots receive is an exact PERMUTATION of the declared extents — each
      tensor's own volume moved through the wrong pointer.

    When those agree the finding is reported as ``argument_binding`` and says so explicitly: the
    addressing arithmetic is SELF-CONSISTENT and would be correct against the permuted order, so the
    defect is the argument order, not the address or pitch computation. The per-move address and pitch
    divergences are then carried as that finding's evidence, never as findings of their own — flagging
    them separately would blame arithmetic that is right.

    Also compared, each against the value the declaration derives for it, and each SILENT unless it
    actually diverges: a store's readout dtype (the declared dtype of the output tensor bound to the slot
    it writes), the store configuration's activation and accumulator scale (the single declared commit's
    epilogue; identity scale when no scaling stage is declared), and a move's column extent (the bound
    tensor's declared trailing dim, tile-padded).

    NOTHING here re-derives a field layout. Every field value is the one the target's own RoCC decoder
    (:mod:`merlin.targetgen.rocc.decode`) already extracted, whose bit positions come from that target's
    RTL facts + capability manifest; every field is compared AS DATA (a parsed int/str against a computed
    int/str), never string-matched against a literal.

    Fields with NO derivable intent produce NO finding and are recorded under
    ``evidence["fields_not_derivable"]`` with the reason. A confidently wrong intended value is worse
    than silence.

    Severity ``warn``: advisory only, and it can never move this report's verdict to ``reject``."""
    cid = "T0.encoded_field_intent"
    mesh = _mesh(rtl_facts)
    if mesh is None:
        return Check(cid, "T0", "warn", "skipped",
                     "mesh dims UNKNOWN (no RTL facts derived) — the tile-padded extent a declared "
                     "tensor may legally occupy is not derivable, so no field is compared")
    mr, mc = mesh
    binding, shape, why = kernel_argument_binding(command_buffer, capsule)
    if not binding:
        return Check(cid, "T0", "warn", "skipped", why)
    by_index = {b["index"]: b for b in binding}
    commit, commit_why = _declared_commit(command_buffer)

    findings: list[dict] = []           # divergences that stand on their own
    addressing: list[dict] = []         # per-move address/pitch divergences (binding evidence)
    undecidable: list[dict] = []
    n_compared = 0
    volume: dict[int, int] = {}
    ld_pitch: int | None = None
    st_pitch: int | None = None
    st_config: dict = {}

    def _padded_cols(b: dict) -> int:
        return math.ceil(b["cols"] / mc) * mc

    def _pitches(b: dict) -> tuple[int, int]:
        return b["cols"] * b["elem_bytes"], _padded_cols(b) * b["elem_bytes"]

    def _extent(b: dict) -> int:
        """The tile-padded byte extent — the most permissive layout the declared shape admits."""
        return math.ceil(b["rows"] / mr) * mr * _pitches(b)[1]

    for inst in trace.get("instructions", []):
        cls = inst.get("class")
        dec = inst.get("decoded") or {}
        idx = inst.get("index")
        if cls == "CONFIG_LD":
            v = dec.get("stride")
            ld_pitch = int(v) if isinstance(v, int) and v > 0 else None
            continue
        if cls == "CONFIG_ST":
            v = dec.get("out_stride_bytes")
            st_pitch = int(v) if isinstance(v, int) and v > 0 else None
            st_config = dict(dec)
            if commit is None:
                undecidable.append({"index": idx, "field": "store_activation", "reason": commit_why})
                undecidable.append({"index": idx, "field": "config_scale", "reason": commit_why})
                continue
            epilogue = commit.get("epilogue") or []
            got_relu = dec.get("relu")
            if got_relu is None:
                undecidable.append({"index": idx, "field": "store_activation",
                                    "reason": "the decoder extracted no activation field here"})
            else:
                n_compared += 1
                want_relu = "relu" in epilogue
                if bool(got_relu) != want_relu:
                    findings.append({
                        "index": idx, "instruction": cls, "field": "store_activation",
                        "emitted": bool(got_relu), "intended": want_relu,
                        "intent_from": f"declared epilogue {list(epilogue)}"})
            got_scale = dec.get("acc_scale")
            want_scale = commit.get("acc_scale")
            if want_scale is None and "acc_scale" not in epilogue:
                want_scale = 1.0            # no declared scaling stage => the identity scale
            if got_scale is None:
                undecidable.append({"index": idx, "field": "config_scale",
                                    "reason": "the decoder extracted no accumulator-scale field here"})
            elif want_scale is None:
                undecidable.append({"index": idx, "field": "config_scale",
                                    "reason": "the declared epilogue contains a scaling stage but "
                                              "declares no acc_scale value, so the intended scale is "
                                              "not derivable"})
            else:
                n_compared += 1
                if float(got_scale) != float(want_scale):
                    findings.append({
                        "index": idx, "instruction": cls, "field": "config_scale",
                        "emitted": float(got_scale), "intended": float(want_scale),
                        "intent_from": (f"declared acc_scale {commit['acc_scale']!r}"
                                        if "acc_scale" in commit else
                                        f"declared epilogue {list(epilogue)} names no scaling stage, so "
                                        f"the intended scale is the identity")})
            continue
        if cls not in _MVIN_CLASSES | {"MVOUT"}:
            continue
        dram = dec.get("dram") if isinstance(dec.get("dram"), dict) else {}
        if dram.get("kind") != "argbase":
            undecidable.append({"index": idx, "field": "dram_base_pointer",
                                "reason": f"this move's DRAM operand is {dram.get('kind') or 'absent'!r},"
                                          f" not a kernel argument, so no declared tensor bounds it"})
            continue
        arg = dram.get("arg_index")
        b = by_index.get(arg)
        if b is None:
            undecidable.append({"index": idx, "field": "dram_base_pointer",
                                "reason": f"this move addresses kernel argument {arg}, which the "
                                          f"{shape} argument order's {len(binding)} slot(s) do not cover"})
            continue
        offset, rows, cols = dram.get("offset"), dec.get("rows"), dec.get("cols")
        if cls == "MVOUT" and "maxpool" in (_attrs(capsule).get("epilogue") or []):
            rows, pool_why = _effective_store_rows(
                {"rows": rows, "config": st_config}, capsule, rtl_facts)
            if rows is None:
                undecidable.append({"index": idx, "field": "store_extent",
                                    "reason": f"pooled MVOUT footprint is UNKNOWN: {pool_why}"})
        pitch = ld_pitch if cls in _MVIN_CLASSES else st_pitch
        packed_pitch, padded_pitch = _pitches(b)
        base = {"index": idx, "instruction": cls, "arg_index": arg, "tensor": b["name"],
                "declared_shape": [b["rows"], b["cols"]], "declared_elem_bytes": b["elem_bytes"]}
        if cols is not None:
            volume[arg] = volume.get(arg, 0) + max(0, int(rows or 0)) * int(cols) * b["elem_bytes"]
            n_compared += 1
            if int(cols) > _padded_cols(b):
                findings.append({**base, "field": "column_extent", "emitted": int(cols),
                                 "intended": f"<={_padded_cols(b)}",
                                 "intent_from": f"declared trailing dim {b['cols']} of {b['name']!r}, "
                                                f"padded to the RTL-derived {mc}-column array edge"})
        else:
            undecidable.append({"index": idx, "field": "column_extent",
                                "reason": "the decoder extracted no column extent here"})
        if cls == "MVOUT":
            got_dt = dec.get("readout")
            want_dt = ((command_buffer or {}).get("tensors") or {}).get(b["name"], {}).get("dtype")
            if got_dt is None or want_dt is None:
                undecidable.append({"index": idx, "field": "readout_dtype",
                                    "reason": "the store carries no decoded readout dtype and/or the "
                                              "bound tensor declares no dtype"})
            else:
                n_compared += 1
                if str(got_dt) != str(want_dt):
                    findings.append({**base, "field": "readout_dtype", "emitted": str(got_dt),
                                     "intended": str(want_dt),
                                     "intent_from": f"declared dtype of output tensor {b['name']!r}"})
        if offset is None or rows is None or cols is None:
            undecidable.append({"index": idx, "field": "dram_base_pointer",
                                "reason": "the decoder extracted no byte offset and/or extent here, so "
                                          "the span this move touches is not derivable"})
            continue
        if int(rows) > 1 and pitch is None:
            undecidable.append({"index": idx, "field": "dram_base_pointer",
                                "reason": f"this {rows}-row move is reached with no DRAM row pitch "
                                          f"configured, so the span it touches is not derivable"})
            continue
        n_compared += 1
        if int(rows) > 1 and int(pitch) not in (packed_pitch, padded_pitch):
            addressing.append({**base, "aspect": "row_pitch", "emitted": int(pitch),
                               "this_tensors_own_row": f"{packed_pitch} (packed) or "
                                                       f"{padded_pitch} (tile-padded)"})
        span_end = int(offset) + max(0, int(rows) - 1) * int(pitch or 0) + int(cols) * b["elem_bytes"]
        extent = _extent(b)
        if span_end > extent:
            addressing.append({**base, "aspect": "past_declared_extent", "emitted": span_end,
                               "this_tensors_extent": extent, "byte_offset": int(offset),
                               "rows": int(rows), "cols": int(cols), "row_pitch": int(pitch or 0)})

    permutation = _binding_permutation(binding, volume)
    overruns = [a for a in addressing if a["aspect"] == "past_declared_extent"]
    ev = {"contract_command_shape": shape,
          "argument_binding": binding,
          "argument_order_from": "kernel_abi.arg_order_by_command_shape in the OOT backend ABI contract",
          "mesh": [mr, mc],
          "declared_packed_bytes": {str(b["index"]): b["packed_bytes"] for b in binding},
          "emitted_moved_bytes": {str(k): v for k, v in sorted(volume.items())},
          "fields_compared": n_compared,
          "fields_not_derivable": undecidable[:40],
          "n_fields_not_derivable": len(undecidable),
          "addressing_evidence": addressing[:40], "n_addressing_evidence": len(addressing),
          "findings": findings[:40], "n_findings": len(findings)}
    if permutation:
        ev["argument_volume_permutation"] = permutation

    if permutation:
        # The pointer list is permuted. Report THAT, and say plainly that the addressing arithmetic is
        # not the defect: it is self-consistent and would be correct against the order the stream assumes.
        swaps = ", ".join(
            f"slot {p['arg_index']} carries {p['declared_tensor']!r} ({p['declared_bytes']} B declared) "
            f"but receives {p['moved_bytes']} B, exactly the declared extent of "
            f"{p['is_the_declared_extent_of']!r} at slot {p['at_arg_index']}"
            for p in permutation)
        msg = (f"the DRAM base pointer of the emitted moves does not carry the tensor this buffer's ABI "
               f"puts in that argument slot: the kernel-ABI contract resolves this buffer to the "
               f"{shape!r} command shape, whose argument order is "
               f"{[b['name'] for b in binding]}, and the byte volume the slots receive is an exact "
               f"PERMUTATION of the declared extents ({swaps})")
        if overruns:
            w = max(overruns, key=lambda a: a["emitted"])
            msg += (f"; consistently, {len(overruns)} move(s) reach past the declared extent of the "
                    f"tensor bound to the slot they address — the furthest at instruction {w['index']} "
                    f"touching byte {w['emitted']} of {w['tensor']!r}, whose declared "
                    f"[{by_index[w['arg_index']]['rows']}, {by_index[w['arg_index']]['cols']}] x "
                    f"{w['declared_elem_bytes']} byte(s) occupies at most {w['this_tensors_extent']} "
                    f"byte(s) even tile-padded to the RTL-derived {mr}x{mc} array edge")
        msg += (". The addressing arithmetic itself is NOT the finding: it is self-consistent and would "
                "be correct against the permuted order, so what diverges is which argument slot each "
                "tensor is read from")
        # expected/got are the two pointer lists as DATA: the order the contract gives, and the order
        # the emitted volumes say each slot actually received (the slots no move reached stay as declared,
        # since a slot with no traffic carries no evidence either way).
        received = {p["arg_index"]: p["is_the_declared_extent_of"] for p in permutation}
        return Check(cid, "T0", "warn", "fail", msg,
                     expected=[b["name"] for b in binding],
                     got=[received.get(b["index"], b["name"]) for b in binding],
                     evidence=ev,
                     fix_hint=f"the harness calls the kernel with the {shape!r} shape's argument order "
                              f"from kernel_abi.arg_order_by_command_shape — NOT the capsule's or the "
                              f"interface's declaration order, which coincides with it only for buffers "
                              f"that happen to declare the tensors in that same order. Resolve the "
                              f"pointer list from the command buffer's own command shape and re-emit; "
                              f"the address and pitch computation does not need changing")

    if overruns:
        # No permutation established, so the overrun stands on its own: a move addressing memory the
        # declaration does not give that slot, with no evidence that a permuted pointer list explains it.
        w = max(overruns, key=lambda a: a["emitted"])
        findings.append({**{k: w[k] for k in ("index", "instruction", "arg_index", "tensor",
                                              "declared_shape", "declared_elem_bytes")},
                         "field": "dram_base_pointer", "emitted": w["emitted"],
                         "intended": f"<={w['this_tensors_extent']}",
                         "n_moves": len(overruns),
                         "intent_from": f"declared {w['declared_shape']} x {w['declared_elem_bytes']} "
                                        f"byte(s) of {w['tensor']!r}, rows and columns padded to the "
                                        f"RTL-derived {mr}x{mc} array edge"})
    pitch_only = [a for a in addressing if a["aspect"] == "row_pitch"]
    if pitch_only and not overruns:
        a0 = pitch_only[0]
        findings.append({**{k: a0[k] for k in ("index", "instruction", "arg_index", "tensor",
                                              "declared_shape", "declared_elem_bytes")},
                         "field": "row_pitch", "emitted": a0["emitted"],
                         "intended": a0["this_tensors_own_row"], "n_moves": len(pitch_only),
                         "intent_from": f"declared trailing dim of {a0['tensor']!r}, packed or padded to "
                                        f"the RTL-derived {mc}-column array edge"})
    ev["findings"] = findings[:40]          # refreshed: the two blocks above may have appended
    ev["n_findings"] = len(findings)

    if not n_compared:
        return Check(cid, "T0", "warn", "skipped",
                     "no emitted instruction carried a field this check could compare against a derived "
                     "intent — UNKNOWN, asserted neither way", evidence=ev)
    if not findings:
        return Check(cid, "T0", "warn", "pass",
                     f"the emitted moves all address the argument slot the {shape!r} command shape's "
                     f"contract order gives their tensor, and every one of the {n_compared} decoded "
                     f"field(s) this check could derive an intent for agrees with the declaration "
                     f"({len(undecidable)} field instance(s) had no derivable intent and were compared "
                     f"for neither)", evidence=ev)

    def _extreme(group: list[dict]) -> dict:
        numeric = [g for g in group if isinstance(g.get("emitted"), (int, float))
                   and not isinstance(g.get("emitted"), bool)]
        return max(numeric, key=lambda g: abs(g["emitted"])) if numeric else group[0]

    by_field: dict[str, list[dict]] = {}
    for f in findings:
        by_field.setdefault(f["field"], []).append(f)
    parts = []
    for field, group in by_field.items():
        g = _extreme(group)
        where = (f"kernel argument slot {g['arg_index']} (which the {shape!r} contract order binds to "
                 f"declared tensor {g['tensor']!r}, shape {g['declared_shape']})") \
            if "arg_index" in g else "the stream"
        parts.append(f"{field}: {g.get('n_moves', len(group))} {g['instruction']} instruction(s) on "
                     f"{where} diverge, the furthest at instruction {g['index']} emitting "
                     f"{g['emitted']!r} where the declaration derives {g['intended']!r} "
                     f"[intent from {g['intent_from']}]")
    worst = _extreme(findings)
    return Check(cid, "T0", "warn", "fail",
                 "emitted-encoding field(s) disagree with the declaration this buffer's command shape "
                 "and the capsule make, while the instruction NAMES are all legal — " + "; ".join(parts),
                 expected=worst.get("intended"), got=worst.get("emitted"), evidence=ev,
                 fix_hint=f"each field above is compared against the value YOUR OWN command buffer + "
                          f"capsule declare for the tensor the ABI contract's {shape!r} argument order "
                          f"binds to that slot; check both the encoded field and which argument slot you "
                          f"read it from")


def _check_conv_lowering(trace: dict, capsule: dict | None, rtl_facts: dict, target: str) -> Check:
    """A DECLARED convolution must actually materialize its kernel window.

    Two lowerings exist on a mesh target, and both are derivable: the target's OWN fused convolution loop
    (the window gather in hardware), or an im2col'd contraction whose depth is ``Kh*Kw*Ci``. A
    convolution lowered as if the contraction were only ``Ci`` deep emits a legal, well-formed, fully
    covering stream that computes a DIFFERENT operation — nothing before the oracle notices, and the
    oracle reports it as a value error.

    The bound is the tile geometry the declared shapes and the RTL-derived array force; both the
    fused-loop vocabulary and the array edge are derived, neither is written down here."""
    op = _declared_op(capsule)
    if op not in ("conv2d", "conv"):
        return Check("T0.conv_lowering", "T0", "warn", "skipped", "capsule declares no convolution")
    a = _attrs(capsule)
    mesh = _mesh(rtl_facts)
    if mesh is None:
        return Check("T0.conv_lowering", "T0", "warn", "skipped",
                     "mesh dims UNKNOWN (no RTL facts derived) — no tile bound derivable")
    conv_loops, basis = _vocabulary_classes(target, _CONV_STEMS)
    in_trace = conv_loops & set(_classes(trace))
    if in_trace:
        return Check("T0.conv_lowering", "T0", "warn", "pass",
                     f"the convolution is lowered through the target's own fused convolution loop "
                     f"{sorted(in_trace)}; the window gather is in hardware",
                     evidence={"conv_loop_classes": sorted(in_trace), "basis": basis})
    try:
        kh, kw, ci = int(a["kh"]), int(a["kw"]), int(a["ci"])
    except (KeyError, TypeError, ValueError):
        return Check("T0.conv_lowering", "T0", "warn", "skipped",
                     "capsule declares no kh/kw/ci, so the im2col contraction depth is not derivable")
    contractions = declared_contractions(capsule)
    if not contractions:
        return Check("T0.conv_lowering", "T0", "warn", "skipped",
                     "the declared conv's im2col contraction shape is not derivable")
    if kh * kw <= 1:
        return Check("T0.conv_lowering", "T0", "warn", "pass",
                     "declared 1x1 kernel: the im2col matrix IS the activation, so there is no window "
                     "to materialize", evidence={"basis": basis})
    mr, mc = mesh
    P, Co, K = contractions[0]
    lo = math.ceil(P / mr) * math.ceil(K / mr) * math.ceil(Co / mc)
    shallow = math.ceil(P / mr) * math.ceil(ci / mr) * math.ceil(Co / mc)
    n_compute = sum(1 for c in _classes(trace) if c in _COMPUTE_CLASSES)
    ev = {"im2col_depth_kh_kw_ci": K, "raw_channel_depth_ci": ci, "contraction": [P, Co, K],
          "mesh": [mr, mc], "min_compute_steps": lo, "observed_compute": n_compute,
          "compute_steps_a_ci_deep_contraction_would_take": shallow,
          "conv_loop_classes_available": sorted(conv_loops), "basis": basis}
    if not n_compute:
        return Check("T0.conv_lowering", "T0", "warn", "skipped",
                     "this trace names neither a fused convolution loop nor any instruction class this "
                     "check recognizes as a matrix-compute step, so the im2col depth is UNKNOWN here — "
                     "asserted neither way (a compute-free contraction is reported by "
                     "T0.movement_compute_balance)", evidence=ev)
    if n_compute < lo:
        note = (" — which is what a contraction over the RAW channel depth alone would take, not over "
                "the im2col depth Kh*Kw*Ci") if n_compute and n_compute <= shallow else ""
        return Check("T0.conv_lowering", "T0", "warn", "fail",
                     f"the declared {kh}x{kw} convolution emits neither this target's fused convolution "
                     f"loop {sorted(conv_loops) or '(this target names none)'} nor enough compute for an "
                     f"im2col'd contraction: covering [{P}, {K}] x [{K}, {Co}] over the RTL-derived "
                     f"{mr}x{mc} array takes at least {lo} compute step(s) and the stream carries "
                     f"{n_compute}{note}. The kernel window is not being folded into the contraction.",
                     expected=f">={lo}", got=n_compute,
                     ratio=round(n_compute / lo, 3) if lo else None, evidence=ev,
                     fix_hint="gather the Kh*Kw*Ci window into the contraction (im2col) or emit the fused "
                              "convolution loop; contracting over the channel depth alone is a "
                              "different operation, however well-formed the stream looks")
    return Check("T0.conv_lowering", "T0", "warn", "pass",
                 f"the declared {kh}x{kw} convolution is lowered as an im2col'd contraction of depth "
                 f"{K}: {n_compute} compute step(s) >= the {lo} the tile geometry requires", evidence=ev)


# ----------------------------------------------------------------------------------- public API
def screen(trace: dict, capsule: dict | None = None,
           rtl_facts: dict | None = None, *, target: str,
           command_buffer: dict | None = None) -> CheckReport:
    """Run the Phase-0 T0 checks over a decoded trace; return an advisory :class:`CheckReport`.

    Pure function: no simulation, no frozen-runner imports, no verdict mutation. ``trace`` is a
    ``rocc.decode`` output dict; ``capsule`` is a loaded capsule.yaml dict (for declared
    shapes/modes); ``target`` is REQUIRED and selects the base RTL facts (the run's resolved target,
    via :func:`load_default_facts`); ``rtl_facts`` optionally OVERRIDES specific fact keys on top.

    ``command_buffer`` is the package's OWN emitted command buffer (the declaration the numeric and
    trace planes both read). It is what binds each kernel argument to a declared tensor, so
    :func:`_check_encoded_field_intent` needs it; omitted, that check reports *skipped with that
    reason* — never a pass.
    """
    facts = load_default_facts(target)
    if rtl_facts:
        facts.update(rtl_facts)
    rep = CheckReport(
        capsule=(capsule or {}).get("name"),
        source_trace=trace.get("source"),
        rtl_facts=facts,
    )
    rep.checks.append(_check_decode_clean(trace))
    rep.checks.append(_check_decode_funct_legal(trace, facts))
    rep.checks.append(_check_movement_compute_balance(trace, capsule))
    rep.checks.append(_check_tile_coverage(trace, capsule, facts))
    rep.checks.append(_check_data_movement_reuse(trace, capsule, facts))
    rep.checks.append(_check_spad_capacity(trace, facts))
    rep.checks.append(_check_preload_before_compute(trace))
    rep.checks.append(_check_config_before_use(trace))
    rep.checks.append(_check_fence_bracket(trace))
    # Structural checks over the DECLARED geometry + the author's own emitted stream. Severity 'warn' on
    # purpose: they are feedback, and a warn can never move this report's verdict to 'reject' (which an
    # opt-in caller may use to skip an oracle run). A structurally-odd but conformant kernel must never
    # lose its oracle over an advisory finding.
    rep.checks.append(_check_pool_config(trace, capsule, facts))
    rep.checks.append(_check_output_store_coverage(trace, capsule, facts))
    rep.checks.append(_check_extent_tile_legalization(trace, capsule, facts, target))
    rep.checks.append(_check_conv_lowering(trace, capsule, facts, target))
    rep.checks.append(_check_encoded_field_intent(trace, capsule, facts, command_buffer))
    return rep


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="RTL-derived cheap checks over a decoded RoCC trace "
                                             "(advisory / un-wired)")
    ap.add_argument("trace", help="path to instruction_trace.json (rocc_decode output)")
    ap.add_argument("--target", required=True,
                    help="target whose RTL facts the checks are screened against")
    ap.add_argument("--capsule", default=None, help="path to capsule.yaml (declared shapes/modes)")
    ap.add_argument("--rtl-facts", default=None, help="optional JSON file overriding RTL facts")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    trace = json.loads(Path(a.trace).read_text(encoding="utf-8"))
    capsule = None
    if a.capsule:
        import yaml
        capsule = yaml.safe_load(Path(a.capsule).read_text(encoding="utf-8"))
    facts = None
    if a.rtl_facts:
        facts = json.loads(Path(a.rtl_facts).read_text(encoding="utf-8"))

    rep = screen(trace, capsule, facts, target=a.target)
    out = json.dumps(rep.to_dict(), indent=2)
    if a.out:
        Path(a.out).write_text(out, encoding="utf-8")
        print(f"wrote {a.out}: verdict={rep.verdict} errors={rep.n_error} warns={rep.n_warn}")
    else:
        print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
