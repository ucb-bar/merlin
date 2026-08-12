"""RTL-derived cheap checks layer (Phase 0, ADVISORY / UN-WIRED).

A pure-static pre-screen + richer-feedback layer that consumes the **already-decoded** RoCC
instruction trace (:mod:`merlin.targetgen.rocc_decode` output) plus the target's RTL facts, and
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

This module is **advisory and un-wired**: it is imported by NOTHING in the frozen runner/grader/schema
path, modifies no existing file, and changes no pass/fail verdict. It is scaffolding for the phased
plan in ``artifacts/perf-bench/gemmini/rtl_checks_layer_design.md``.

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
    facts: dict[str, Any] = {"mesh": None, "scratchpad_bytes": None, "legal_funct": None,
                             "custom_opcode": None, "funct3": None,
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
        ft = next((i for i in f.get("interfaces", []) if i.get("name") == "funct_decode_table"), None)
        if ft:
            facts["legal_funct"] = ft.get("legal_funct")
            facts["custom_opcode"] = ft.get("custom_opcode")
            facts["funct3"] = ft.get("funct3")
        facts["from"] = f"circt_introspect facts.json ({rec.get('generator', {}).get('version', '?')})"
    except Exception:  # noqa: BLE001 — facts not derivable -> everything stays UNKNOWN (checks skip)
        pass
    return facts

# capsule.operation.op values whose semantics REQUIRE matrix compute (used by movement/compute
# balance). Movement-only ops legitimately emit no COMPUTE; these do not.
_COMPUTE_OPS = {"matmul", "resident_reuse", "conv2d", "conv", "matmul_resident"}
_COMPUTE_CLASSES = {"COMPUTE_PRELOADED", "COMPUTE_ACCUMULATE"}


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
    n_mvin = classes.count("MVIN")
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
    # single declared matmul: exact tile geometry from RTL mesh + declared shape.
    M, N = shape
    exp = math.ceil(M / mr) * math.ceil(N / mc)
    if n_mvout != exp:
        return Check("T0.tile_coverage", "T0", "warn", "fail",
                     f"MVOUT={n_mvout} but {M}x{N} over {mr}x{mc} mesh needs Mt*Nt={exp} tiles",
                     expected=exp, got=n_mvout,
                     ratio=round(n_mvout / exp, 3) if exp else None,
                     fix_hint=("over-committing outputs" if n_mvout > exp else "under-committing "
                               "outputs") + f" ~{round(n_mvout/exp,1) if exp else '?'}x; "
                               "check the M/N tiling loop bounds")
    return Check("T0.tile_coverage", "T0", "warn", "pass",
                 f"MVOUT={n_mvout} == Mt*Nt={exp} for {M}x{N} over {mr}x{mc} mesh",
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
    n_mvin = _classes(trace).count("MVIN")
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
    """Any MVIN scratchpad row address must lie within the introspected scratchpad capacity.

    Addresses are row indices into the spad banks; the capacity in *rows* is
    scratchpad_bytes / (DIM * elem_bytes) with i8 elements (elem_bytes=1, row width = DIM cols).
    We compare the largest emitted spad_addr against that row count. This is the
    ``must_respect_scratchpad_capacity`` compiler obligation — a real silicon-correctness bound,
    identical for every shape.
    """
    mesh = _mesh(rtl_facts)
    spad_bytes = rtl_facts.get("scratchpad_bytes")
    if mesh is None or spad_bytes is None:
        return Check("T0.spad_capacity", "T0", "error", "skipped",
                     "mesh/scratchpad UNKNOWN (no RTL facts) — cannot bound scratchpad capacity")
    mr, _mc = mesh
    spad_bytes = int(spad_bytes)
    # rows = bytes / (cols-per-row * elem_bytes); i8 => 1 byte, row holds DIM columns.
    rows_capacity = spad_bytes // (mr * 1)
    addrs = [i.get("decoded", {}).get("spad_addr") for i in trace.get("instructions", [])
             if i.get("class") == "MVIN"]
    addrs = [a for a in addrs if isinstance(a, int)]
    if not addrs:
        return Check("T0.spad_capacity", "T0", "error", "skipped",
                     "no MVIN scratchpad addresses to check")
    max_addr = max(addrs)
    if max_addr >= rows_capacity:
        bad = [i["index"] for i in trace.get("instructions", [])
               if i.get("class") == "MVIN"
               and isinstance(i.get("decoded", {}).get("spad_addr"), int)
               and i["decoded"]["spad_addr"] >= rows_capacity][:8]
        return Check("T0.spad_capacity", "T0", "error", "fail",
                     f"max spad_addr {max_addr} >= scratchpad row capacity {rows_capacity} "
                     f"({spad_bytes} bytes / {mr} cols) — would alias/wrap on silicon",
                     expected=f"<{rows_capacity}", got=max_addr,
                     evidence={"instruction_indices": bad},
                     fix_hint="resident footprint exceeds scratchpad; tile K/N smaller or evict")
    return Check("T0.spad_capacity", "T0", "error", "pass",
                 f"max spad_addr {max_addr} < scratchpad row capacity {rows_capacity}",
                 expected=f"<{rows_capacity}", got=max_addr)


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
                     fix_hint="emitted a custom-3 funct the Gemmini decoder rejects; RTL would not accept")
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


# ----------------------------------------------------------------------------------- public API
def screen(trace: dict, capsule: dict | None = None,
           rtl_facts: dict | None = None, *, target: str) -> CheckReport:
    """Run the Phase-0 T0 checks over a decoded trace; return an advisory :class:`CheckReport`.

    Pure function: no simulation, no frozen-runner imports, no verdict mutation. ``trace`` is a
    ``rocc_decode`` output dict; ``capsule`` is a loaded capsule.yaml dict (for declared
    shapes/modes); ``target`` is REQUIRED and selects the base RTL facts (the run's resolved target,
    via :func:`load_default_facts`); ``rtl_facts`` optionally OVERRIDES specific fact keys on top.
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
