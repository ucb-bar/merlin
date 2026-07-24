"""Bridge to the sibling ``mlc`` model-ladder compiler — the CIRCT+xDSL RTL frontend we reuse for
RTL *needle* extraction (finding specific ISA facts in the HW-dialect op graph).

``mlc`` is an EXTERNAL dependency resolved via ``.env MERLIN_MLC_DIR`` (it will be upstreamed to its own
open-source repo). Rather than re-implement the op-graph parser + decoder analysis, we reuse mlc's
``discover.irgraph`` (``circt-opt --mlir-print-op-generic`` -> xDSL ``HwGraph``) and ``discover.decode``
(the legal opcode set from the decoder's ``comb.icmp eq`` fan-out), plus mlc's own prebuilt CIRCT
binaries. ``mlc`` is pip-installed editable into merlin's venv from ``MERLIN_MLC_DIR`` (see the dev
setup docs), so its Python is a normal import. Imports are still function-local behind an availability
guard (``chia_bridge`` style) so importing this module never hard-requires mlc, and a machine without
the mlc package installed degrades honestly rather than crashing. Its NON-python assets (the prebuilt
``circt-opt`` binary, the cached ``runs/circt-arc/<target>`` arc outputs, schemas) still resolve by
directory path via ``MERLIN_MLC_DIR``.

Why this matters: an ISA *header* parse (hand table / ``val NAME = N.U``) is provably wrong vs the
silicon — it lists command codes the decoder never matches and omits ones it does. The decoder-derived
set here is the actual ISA the hardware implements, which is exactly what a functionally-correct compiler
must target.

TARGET-AGNOSTIC: every entry point here takes a ``target`` argument and holds no target name. mlc is
target-parameterized (``artifact_paths(target)`` / ``discovered_memory_map(target)`` /
``discover_opcode_set(graph)`` / per-target ``runs/circt-arc/<target>``), so the same code plugs any HW
RTL repo mlc knows (gemmini, atlas, otbn, muon, nvdla, rocket, ...).
"""
from __future__ import annotations

import os
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


def mlc_available() -> tuple[bool, str]:
    """(ok, reason). ok iff MERLIN_MLC_DIR resolves (assets), circt-opt is built, and mlc imports."""
    d = mlc_dir()
    if d is None:
        return False, "MERLIN_MLC_DIR unset/invalid (no mlc/ package) — set it in .env"
    if circt_opt_bin() is None:
        return False, f"circt-opt not built under {d}/third_party/circt/build/bin"
    try:
        import mlc.discover.irgraph  # noqa: F401
        import mlc.discover.decode  # noqa: F401
    except Exception as e:  # noqa: BLE001 — surface the import failure honestly
        return False, f"mlc import failed: {type(e).__name__}: {e} — pip install -e $MERLIN_MLC_DIR"
    return True, "ok"


def require_mlc() -> None:
    ok, why = mlc_available()
    if not ok:
        raise RuntimeError(f"mlc unavailable: {why}")


def matmul_reuse_prediction(M: int, N: int, K: int, *, dim: int, capacity_bytes: int,
                            elem_bytes: int = 1) -> dict | None:
    """M3 STATIC data-movement prediction for an M×K·K×N matmul — no arc, no sim, no cycles.

    Reuses mlc's pure-arithmetic Model-3 functions (``predict_dma_volume.matmul_refetch_factor`` +
    ``spills.matmul_footprint_rows``) so the reuse/spill math is the mlc source of truth, not a
    re-derivation here. Returns the resident operand footprint (rows), the on-chip capacity (rows), a
    fits flag, and the loop-nest refetch factor (=1 when the operands are fully resident; >1 when they
    spill and the outer loop re-streams). Returns None (fail-closed) when mlc is not importable — these
    functions need only the mlc package installed, NOT circt, so we gate on the import alone."""
    try:
        from mlc.passes.predict_dma_volume import matmul_refetch_factor, ReuseFacts
        from mlc.passes.spills import matmul_footprint_rows
    except Exception:  # noqa: BLE001 — mlc absent/unimportable ⇒ honest unavailable, never a guess
        return None
    # beat_bytes is irrelevant to the refetch/footprint math (it only scales byte→beat counts), so a
    # benign 1 keeps us from fabricating a DMA beat width we have not discovered for this target.
    facts = ReuseFacts(beat_bytes=1, dim=int(dim), capacity_bytes=int(capacity_bytes))
    refetch = int(matmul_refetch_factor(M, K, N, facts, in_bytes=elem_bytes))
    footprint_rows = int(matmul_footprint_rows(M, N, K, int(dim)))
    capacity_rows = int(capacity_bytes) // (int(dim) * max(elem_bytes, 1))
    return {"footprint_rows": footprint_rows, "capacity_rows": capacity_rows,
            "fits": footprint_rows <= capacity_rows, "refetch": refetch,
            "footprint_tiles": footprint_rows // int(dim)}


# ------------------------------------------------------------------- target-agnostic RTL extraction
# Everything below is parameterized by ``target`` and holds NO target name — mlc is target-parameterized
# (artifact_paths/discovered_memory_map/discover_opcode_set/per-target runs/circt-arc/<target>), so the
# same code plugs any HW RTL repo mlc knows (gemmini, atlas, otbn, muon, nvdla, rocket, ...).

def core_hw_mlir(target: str) -> Path | None:
    """The version-matched CORE HW dialect (the module carrying the command decoder) for ANY target,
    from mlc's per-target arc outputs (``runs/circt-arc/<target>/outputs``). Prefers ``*_core_hw.mlir``
    (the core parses cleanly; the SoC dialect carries unparseable sv.verbatim blobs)."""
    d = mlc_dir()
    if d is None:
        return None
    outs = d / "runs" / "circt-arc" / target / "outputs"
    cands = sorted(outs.glob("*_core_hw.mlir")) or sorted(outs.glob("*_hw.mlir"))
    return next((p for p in cands if p.exists() and ".generic." not in p.name), None)


def discover_legal_opcodes(target: str, *, opcode_width: int | None = None) -> dict:
    """Derive the legal command-opcode set the RTL DECODER matches, for ANY target — via mlc's
    ``comb.icmp eq`` fan-out over the target's core HW dialect. The ISA the silicon implements, not a
    hand table/header. Returns ``{legal_opcodes, width, fanout, module, hw_source, method, evidence}``
    (``legal_opcodes=None`` if no decode signal is found / mlc or the HW dialect is unavailable)."""
    require_mlc()
    hw = core_hw_mlir(target)
    if hw is None:
        return {"legal_opcodes": None, "width": opcode_width, "fanout": 0, "module": None,
                "hw_source": None, "method": "decoder_icmp_fanout(mlc)",
                "evidence": f"no core HW dialect for target {target!r} under mlc runs/circt-arc"}
    from mlc.discover import decode, irgraph
    graph = irgraph.load_hw_graph(hw, circt_opt=circt_opt_bin())
    sig = decode.discover_opcode_set(graph, expected_width=opcode_width)
    if sig is None:
        return {"legal_opcodes": None, "width": opcode_width, "fanout": 0, "module": None,
                "hw_source": str(hw), "method": "decoder_icmp_fanout(mlc)",
                "evidence": f"no decode signal in {hw.name}"}
    legal = sorted(int(v) for v in sig.values)
    return {"legal_opcodes": legal, "width": sig.width, "fanout": sig.fanout, "module": sig.module,
            "hw_source": str(hw), "method": "decoder_icmp_fanout(mlc)",
            "evidence": f"union of comb.icmp-eq {sig.width}-bit decode signals in module {sig.module} "
                        f"({sig.fanout} comparisons) -> {len(legal)} legal opcodes"}


# The target-agnostic semantic roles the check compiler reasons over (never gemmini opcode names).
SEMANTIC_ROLES = ("load", "compute", "store", "config", "barrier")


def semantic_roles(target: str) -> dict:
    """DERIVED opcode -> semantic-role map (load/compute/store/config/barrier), or EMPTY when it cannot
    be grounded — the caller then DROPS role-based checks rather than guessing.

    This is the principled replacement for any hand funct->class table. The only rigorous source is mlc's
    Stage-2 behavioural probe (``mlc.discover.probe.effect_signature`` — the value->effect map), whose
    per-opcode effect signature is classified against the discovered mesh (compute) and memory banks
    (load/store). Its fingerprint-stamped cache is ``discovered_roles.json``, the single artifact this
    reads.

    REGEN-ON-DEMAND (mirrors :func:`rtl.facts.ensure_facts`): when the cache is cold AND the target has a
    live arc model (:func:`arc_available`), we run :func:`derive_and_cache_roles` once to populate it, then
    read — so callers get derived roles without a separate manual step. LAZY (only when roles are actually
    requested) + CACHED (a warm cache is never re-probed), so it never slows imports/collection. A target
    with NO arc (SIMT/prototype like radiance, or mlc absent) keeps the honest empty ``{derived: False,
    reason}`` — we never auto-regen where the probe is impossible, and a failed probe degrades honestly
    rather than crashing the bundle. Returns ``{roles: {opcode:int -> role}, source, derived: bool,
    reason}``."""
    d = mlc_dir()
    if d is None:
        return {"roles": {}, "source": None, "derived": False, "reason": "MERLIN_MLC_DIR unset"}
    cache = d / "runs" / "circt-arc" / target / "outputs" / "discovered_roles.json"
    if not cache.is_file():
        # Regenerate the roles cache on demand ONLY where the probe is possible (a live arc model). A
        # SIMT/prototype target with no arc keeps the honest-empty path below — never auto-regen where
        # impossible, never crash the bundle if the probe fails.
        if arc_available(target):
            try:
                derive_and_cache_roles(target)
            except Exception as e:  # noqa: BLE001 — a failed probe is honest-unavailable, never a crash/guess
                return {"roles": {}, "source": None, "derived": False,
                        "reason": f"role derivation failed for {target!r}: {type(e).__name__}: {e}"}
    if not cache.is_file():
        return {"roles": {}, "source": None, "derived": False,
                "reason": f"no discovered_roles cache for {target!r} — run derive_and_cache_roles in the mlc venv"}
    try:
        import json as _json
        data = _json.loads(cache.read_text())
        roles = {int(k): v for k, v in (data.get("roles") or {}).items() if v in SEMANTIC_ROLES}
        return {"roles": roles, "source": data.get("method", "probe:effect_signature(mlc)"),
                "derived": True, "reason": f"{len(roles)} opcode roles from the mlc effect probe"}
    except Exception as e:  # noqa: BLE001 — an unreadable cache is honest-unavailable, never a guess
        return {"roles": {}, "source": None, "derived": False, "reason": f"unreadable roles cache: {e}"}


# --- hand semantic_class <-> derived-role cross-check --------------------------------------------
# The behavioural probe derives a COARSE role (load/compute/store/config/barrier) per funct. The hand ABI
# ``semantic_class`` in target_contract.yaml declares a FINER label (COMPUTE_PRELOADED vs _ACCUMULATE,
# MVIN2/3, LOOP_* variants) that is ABI, not behaviourally nameable. To VERIFY the hand labels against RTL
# behaviour we project each hand label onto the coarse role vocabulary and compare — a declared label whose
# coarse projection disagrees with the behaviourally-derived role is a real alarm (mislabel or RTL drift).

def _coarse_of_hand_class(label: str) -> str | None:
    """Project a hand ABI ``semantic_class`` label onto the coarse behavioural-role vocabulary
    (load/compute/store/config/barrier). Returns None for a label that has no coarse projection (skip —
    nothing to cross-check). Prefix match, no regex:
    ``MVIN*``->load, ``MVOUT*``->store, ``COMPUTE*``/``PRELOAD``/``LOOP_CONV``->compute,
    ``CONFIG*``/``LOOP_WS``/``LOOP_*_CONFIG``->config, ``FLUSH``->barrier."""
    if label.startswith("MVIN"):
        return "load"
    if label.startswith("MVOUT"):
        return "store"
    if label == "FLUSH":
        return "barrier"
    if label.startswith("COMPUTE") or label == "PRELOAD" or label == "LOOP_CONV":
        return "compute"
    if label.startswith("CONFIG") or label == "LOOP_WS" or label.endswith("_CONFIG"):
        return "config"
    return None


def crosscheck_semantic_class(target: str) -> list[str]:
    """ALWAYS-ON cross-check gate: VERIFY the hand ABI ``semantic_class`` labels against the behaviourally
    DERIVED coarse roles. For every funct that has BOTH a derived role (from :func:`semantic_roles`, which
    regenerates the probe cache on demand) AND a hand ``semantic_class`` label
    (:func:`~merlin.targetgen.target_experiment.load_capability_manifest`), assert the derived coarse role
    equals the coarse projection of the hand label (:func:`_coarse_of_hand_class`). Returns the list of
    DISAGREEMENTS (empty = all agree). A disagreement is a real alarm — either the hand label mislabels the
    funct or the RTL behaviour drifted from the declared ABI. Functs with no derived role (not swept / no
    arc) or an unmappable label are skipped, not counted as agreements — so a no-arc/SIMT target yields an
    empty list (nothing derived to cross-check) rather than a crash."""
    roles = semantic_roles(target).get("roles") or {}
    if not roles:
        return []  # nothing behaviourally derived (no arc / SIMT / mlc absent) -> nothing to cross-check
    from ..target_experiment import load_capability_manifest
    hand = (load_capability_manifest(target).encoding or {}).get("semantic_class") or {}
    disagreements: list[str] = []
    for code, label in hand.items():
        derived = roles.get(int(code))
        expect = _coarse_of_hand_class(str(label))
        if derived is None or expect is None:
            continue
        if derived != expect:
            disagreements.append(
                f"funct {int(code)} ({label}): derived role {derived!r} != coarse({label!r})={expect!r}")
    return disagreements


# --- region -> coarse role classification -------------------------------------------------------
# The behavioural probe below drives each legal RoCC funct into the arc model, drives it to (bounded)
# retire across many cycles, diffs the flat state, and buckets the *changed* named states into MODULE
# REGIONS. The multi-cycle drive is what defeats "control-queue domination": a single-cycle diff after a
# RoCC fire is dominated by the reservation-station / command-queue write (the command just landing in
# the ROB), NOT the datapath. Driving to retire lets the command propagate out of the control queue into
# the mesh / DMA / scratchpad, whose region signature is what classifies the role. Idle free-running
# state (cycle counters, im2col housekeeping) is measured once with no command and subtracted as noise.
#
# The region names below are STRUCTURAL heuristics for a RoCC systolic accelerator (gemmini-shaped): the
# systolic-array datapath ("mesh"), the DMA store engine ("store_controller"), the scratchpad control +
# TLB that a stalled DMA read lights up, and the loop-unroll / config CSR modules. The operand/accumulator
# banks come from the DISCOVERED memory_map (target-parameterized). A fully target-agnostic classifier
# would derive the mesh/DMA region names from the interaction/channel discovery too; that generalization
# is the residual work — this is the prototype that proves the signal is there.

def _region_of(name: str, *, operand_pfx: str | None, accum_pfx: str | None) -> str:
    """Bucket a changed state NAME into a coarse module region (structural prefix match, no regex)."""
    n = name
    if operand_pfx and n.startswith(operand_pfx):
        return "operand_bank"
    if accum_pfx and n.startswith(accum_pfx):
        return "accum_bank"
    if n.startswith("reservation_station") or n.startswith("raw_cmd_q") or n.startswith("unrolled_cmd_q"):
        return "controlq"
    if "/cmd_q/" in n or "/tagq/" in n:
        return "controlq"
    if n.startswith("ex_controller/mesh"):
        return "mesh"
    if n.startswith("ex_controller"):
        return "ex_ctrl"
    if n.startswith("store_controller"):
        return "store_ctrl"
    if n.startswith("load_controller"):
        return "load_ctrl"
    if n.startswith("spad"):
        return "spad_ctrl"
    if n.startswith("tlb"):
        return "tlb"
    if n.startswith("mod_1"):
        return "dma_mod1"
    if n.startswith("mod"):
        return "dma_mod0"
    if n.startswith("counters") or n.startswith("pipeline_stall") or "_pipeline_stall_counter" in n:
        return "counters"
    if n.startswith("io_"):
        return "io"
    return "other"


def _classify_role(regions: dict, busy: int, drive_cycles: int) -> str:
    """Coarse role from a funct's region signature. Precedence matters: heavy mesh (systolic datapath)
    is compute even when a macro (LOOP_CONV) also spawns a stalled DMA; store beats load (both stall on
    the absent memory system) because only a store lights the store engine; a light mesh touch is the
    weight-load PRELOAD (compute-adjacent)."""
    g = lambda k: regions.get(k, 0)
    stalled = busy >= int(drive_cycles * 0.9)
    if g("mesh") > 50:
        return "compute"                                   # heavy systolic-array datapath
    if g("store_ctrl") > 0:
        return "store"                                     # DMA store engine active
    if stalled and (g("spad_ctrl") > 0 or g("tlb") > 0 or g("load_ctrl") > 0):
        return "load"                                      # stalled scratchpad DMA read (fill)
    if g("mesh") > 0:
        return "compute"                                   # light mesh touch = weight-preload
    if g("ex_ctrl") > 0 or g("dma_mod0") > 0 or g("dma_mod1") > 0 or g("spad_ctrl") > 0:
        return "config"                                    # CSR / loop-config pokes, no datapath
    return "barrier"                                        # negligible observable effect


def derive_and_cache_roles(target: str, *, base=None, drive_cycles: int = 800) -> dict:
    """Run a behavioural effect probe for ``target`` and classify each legal opcode into a coarse semantic
    role by the RTL region its effect touches — mesh -> compute, DMA store engine -> store, a stalled
    scratchpad DMA read -> load, CSR/loop-config pokes -> config, nothing -> barrier. Writes
    ``discovered_roles.json`` beside the other discovered artifacts. Requires a live arc model (+ the mlc
    package, editable-installed in this venv); raises RuntimeError otherwise (never fabricates roles).
    This is the ONLY writer of the cache :func:`semantic_roles` reads, keeping the derivation in one
    auditable place. Returns the written record.

    Method (defeats control-queue domination): for each legal funct, zero the flat state, issue the RoCC
    command, tick ``drive_cycles`` cycles so the command RETIRES out of the reservation station into the
    datapath, snapshot+diff, subtract an idle-noise baseline, bucket the changed states into module
    regions, and classify. Currently supports RoCC-funct targets (the funct rides ``io_cmd_bits_inst_*``);
    a program/MMIO or TileLink target raises rather than guessing."""
    require_mlc()
    import json as _json
    with _mlc_cwd():
        from mlc.backends.cosim_core import CosimCore
        from mlc.backends.protocols import RoCCAdapter, detect_protocols
        from mlc.discover.probe import build_statemap, snapshot, zero_state, diff
        from mlc.discover.fingerprint import artifact_paths, rtl_fingerprint
        d = mlc_dir()
        p = artifact_paths(target)
        man = str((d / p["man"]).resolve())
        so = str((d / p["so"]).resolve())
        if not Path(man).exists():
            raise RuntimeError(f"mlc arc manifest absent for target {target!r} ({man})")
        # CosimCore._ensure_lib compiles the .so from the sibling *_model.o when the .so is missing, so a
        # transient rebuild race (concurrent session) does not block us — we only need the manifest here.
        core = CosimCore(so, man, clock_name="clock", reset_name="reset", reset_active_low=False)
        if not any(b.kind == "rocc" for b in detect_protocols(core)):
            raise RuntimeError(
                f"{target!r}: behavioural role derivation currently supports RoCC-funct targets only "
                f"(no io_cmd_bits_inst_funct on this model)")
        statemap = build_statemap(man)

        legal = discover_legal_opcodes(target).get("legal_opcodes")
        if not legal:
            raise RuntimeError(f"{target!r}: no legal opcode set discovered — cannot sweep the funct field")

        mm = discovered_memory_map(target) or {}
        operand_pfx = _first_segments(mm.get("operand_mem"))
        accum_pfx = _first_segments(mm.get("accum_mem"))

        def region_counts(changed):
            out: dict[str, int] = {}
            for nm in changed:
                r = _region_of(nm, operand_pfx=operand_pfx, accum_pfx=accum_pfx)
                out[r] = out.get(r, 0) + 1
            return out

        # idle-noise baseline: what free-runs with NO command issued.
        zero_state(core)
        before = snapshot(core)
        for _ in range(drive_cycles):
            core.tick()
        idle = diff(before, snapshot(core), statemap)

        adapter = RoCCAdapter(core)
        roles: dict[str, str] = {}
        evidence: dict[str, dict] = {}
        for funct in legal:
            zero_state(core)
            before = snapshot(core)
            try:
                adapter.issue(funct, rs1=0x1000, rs2=0x40, xd=1, xs1=1, xs2=1, max_wait=3000)
            except RuntimeError:
                # command never accepted (not a legal issue in isolation) — record and move on
                roles[str(funct)] = "barrier"
                evidence[str(funct)] = {"role": "barrier", "accepted": False, "regions": {}, "busy": 0}
                continue
            busy = 0
            for i in range(drive_cycles):
                core.tick()
                if core.has("io_busy") and core.peek("io_busy") == 1:
                    busy = i + 1
            changed = diff(before, snapshot(core), statemap) - idle
            rc = region_counts(changed)
            role = _classify_role(rc, busy, drive_cycles)
            roles[str(funct)] = role
            evidence[str(funct)] = {"role": role, "accepted": True, "busy": busy,
                                    "n_changed": len(changed), "regions": rc}

        record = {
            "target": target,
            "method": "probe:multicycle-effect-region(mlc-arc)",
            "fingerprint": rtl_fingerprint(target).as_dict(),
            "drive_cycles": drive_cycles,
            "legal_opcodes": list(legal),
            "roles": roles,
            "evidence": evidence,
            "note": ("Coarse roles behaviourally derived by driving each RoCC funct to retire on the arc "
                     "model and bucketing the retired-state region signature. Multi-cycle drive + "
                     "idle-noise subtraction defeats reservation-station/control-queue domination. Fine "
                     "ISA names (COMPUTE_PRELOADED vs _ACCUMULATE, MVIN2/3, LOOP_* variants, readout_bits) "
                     "are ABI, not behaviourally nameable — they stay hand-owned in target_contract.yaml."),
        }
        out = Path(man).parent / "discovered_roles.json"
        out.write_text(_json.dumps(record, indent=2) + "\n")
        record["_path"] = str(out)
        return record


def _first_segments(rep_name: str | None) -> str | None:
    """The bank-family name prefix (drops the trailing '<int>/...' bank-index tail) so all sibling banks
    of a discovered operand/accumulator memory share one region key — structural split, no regex
    (e.g. 'spad/spad_mems_0/mem_ext' -> 'spad/spad_mems_')."""
    if not rep_name:
        return None
    segs = rep_name.split("/")
    for i, s in enumerate(segs):
        base, sep, num = s.rpartition("_")
        if sep and num.isdigit():
            return "/".join(segs[:i] + [base + "_"])
    return None


def discovered_memory_map(target: str) -> dict | None:
    """The target's operand-scratchpad / accumulator bank map, DISCOVERED from the RTL by mlc (row
    widths, not hand paths). None if unavailable. Target-agnostic."""
    if mlc_dir() is None:
        return None
    try:
        with _mlc_cwd():
            _ensure_interface_cache(target)
            from mlc.discover.cache import discovered_memory_map as _mm
            return dict(_mm(target))
    except Exception:  # noqa: BLE001
        return None


def discovered_dim(target: str) -> int | None:
    """The target's systolic mesh DIM, DISCOVERED from the RTL by mlc (not a hand literal). None if the
    target has no mesh / is unavailable. Target-agnostic."""
    if mlc_dir() is None:
        return None
    try:
        with _mlc_cwd():
            _ensure_interface_cache(target)
            from mlc.discover.cache import discovered_dim as _dim
            return int(_dim(target))
    except Exception:  # noqa: BLE001
        return None


def discovered_memories(target: str) -> list[dict] | None:
    """The target's SRAM banks (name/depth/row_bytes), DISCOVERED from the RTL by mlc. Target-agnostic."""
    if mlc_dir() is None:
        return None
    try:
        with _mlc_cwd():
            _ensure_interface_cache(target)
            from mlc.discover.cache import load_interface
            return list(load_interface(target).get("memories", []))
    except Exception:  # noqa: BLE001
        return None


def discovered_capacities(target: str) -> dict | None:
    """Total operand-scratchpad + accumulator capacity (bytes) DERIVED from mlc discovery — sum the
    sibling banks of the representative operand/accumulator memory named in the discovered ``memory_map``
    (banks share the name with the bank index stripped). Target-agnostic: no hardcoded bank names.
    Returns ``{operand_bytes, accumulator_bytes, operand_depth, accumulator_depth}`` (values None when
    not derivable)."""
    mm = discovered_memory_map(target)
    mems = discovered_memories(target)
    if not mm or not mems:
        return None

    def _bank_prefix(rep_name: str | None) -> str | None:
        # sibling banks share the name with the bank-index segment ('<base>_<int>') truncated to
        # '<base>_' — structural split, no regex (e.g. 'spad/spad_mems_0/mem_ext' -> 'spad/spad_mems_').
        if not rep_name:
            return None
        segs = rep_name.split("/")
        for i, s in enumerate(segs):
            base, sep, num = s.rpartition("_")
            if sep and num.isdigit():
                return "/".join(segs[:i] + [base + "_"])
        return None

    def _group(rep_name: str | None):
        prefix = _bank_prefix(rep_name)
        if not prefix:
            return None, None
        banks = [x for x in mems if x.get("name", "").startswith(prefix)]
        if not banks:
            return None, None
        total = sum(b["depth"] * b["row_bytes"] for b in banks)
        depth = banks[0]["depth"] if len({b["depth"] for b in banks}) == 1 else None
        return total, depth

    op_bytes, op_depth = _group(mm.get("operand_mem"))
    acc_bytes, acc_depth = _group(mm.get("accum_mem"))
    return {"operand_bytes": op_bytes, "accumulator_bytes": acc_bytes,
            "operand_depth": op_depth, "accumulator_depth": acc_depth}


def target_fact_bundle(target: str) -> dict:
    """The STATIC, provenance-tagged fact bundle DERIVED from a target's CIRCT HW dialect — no model run.

    This is the groundable, non-labeling half of extraction, and it serves BOTH consumers: it is handed
    to the agent as information (the legal ISA it must target, the mesh geometry, the on-chip capacities
    it must respect) AND it is the source of the cheap FileCheck literals. Each field carries provenance
    {value, source, derived}; a field mlc cannot ground is reported derived=False (value None), never
    guessed — so a target with no HW dialect (SIMT/prototype) yields an honest all-unavailable bundle
    rather than a fabricated one. The opcode->role LABELING is deliberately NOT here (see
    semantic_roles / the cross-module analysis it needs); this bundle is exactly what is derivable today.
    """
    def _try(fn, *a):
        try:
            return fn(*a)
        except Exception:  # noqa: BLE001 — a field that can't be grounded is honest-unavailable
            return None

    legal = _try(discover_legal_opcodes, target) or {}
    dim = _try(discovered_dim, target)
    mm = _try(discovered_memory_map, target)
    caps = _try(discovered_capacities, target)
    opcodes = legal.get("legal_opcodes")
    fields = {
        "legal_opcodes": {"value": opcodes, "derived": opcodes is not None,
                          "source": legal.get("method"), "evidence": legal.get("evidence")},
        "mesh_dim": {"value": dim, "derived": dim is not None,
                     "source": "mlc.discover.geometry.discover_mesh_dim"},
        "memory_map": {"value": mm, "derived": mm is not None,
                       "source": "mlc discovered memory_map"},
        "capacities": {"value": caps, "derived": bool(caps and caps.get("operand_bytes")),
                       "source": "sibling-bank sum over discovered memory_map"},
    }
    return {"target": target, "method": "static CIRCT HW-dialect discovery (no model run)",
            "fields": fields,
            "n_derived": sum(1 for f in fields.values() if f["derived"])}


def render_fact_bundle(target: str, bundle: dict | None = None) -> str:
    """Render :func:`target_fact_bundle` as an agent-facing ISA brief (Markdown). This is the information
    handed to the agent so it can target the real ISA/geometry/capacities — every line is DERIVED and
    stamped with its provenance, and an ungrounded field is stated as unavailable, never invented."""
    b = bundle or target_fact_bundle(target)
    f = b["fields"]
    lo = f["legal_opcodes"]["value"]
    lines = [f"# Target ISA facts: {b['target']}",
             f"_Derived by {b['method']}. {b['n_derived']}/4 fields grounded; ungrounded = unavailable, "
             f"not guessed._", ""]
    if f["legal_opcodes"]["derived"]:
        lines += [f"- **Legal opcodes** ({len(lo)}): `{lo}`",
                  f"  - source: {f['legal_opcodes']['source']}"]
    else:
        lines.append("- **Legal opcodes**: unavailable (no decoder signal / no HW dialect for this target)")
    lines.append(f"- **Mesh DIM**: {f['mesh_dim']['value'] if f['mesh_dim']['derived'] else 'unavailable'}")
    caps = f["capacities"]["value"] if f["capacities"]["derived"] else None
    if caps:
        lines.append(f"- **On-chip capacity (bytes)**: operand={caps.get('operand_bytes')}, "
                     f"accumulator={caps.get('accumulator_bytes')}")
    else:
        lines.append("- **On-chip capacity**: unavailable")
    return "\n".join(lines) + "\n"


def _main(argv: list[str] | None = None) -> int:
    import argparse
    import json as _json
    ap = argparse.ArgumentParser(description="Emit a target's DERIVED ISA fact bundle (agent info + "
                                             "FileCheck source). Prints Markdown; --json prints the raw bundle.")
    ap.add_argument("target")
    ap.add_argument("--json", action="store_true", help="print the raw provenance-tagged bundle as JSON")
    a = ap.parse_args(argv)
    b = target_fact_bundle(a.target)
    print(_json.dumps(b, indent=2) if a.json else render_fact_bundle(a.target, b), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())


@contextmanager
def _mlc_cwd():
    """mlc resolves its ``runs/...`` arc artifacts by paths RELATIVE to its own root, so its cosim +
    discovery entry points run with CWD = the mlc dir."""
    d = mlc_dir()
    prev = os.getcwd()
    if d is not None:
        os.chdir(d)
    try:
        yield
    finally:
        os.chdir(prev)


def arc_available(target: str) -> bool:
    """True iff mlc has a prebuilt arc model (.so/.o) + state manifest for ``target`` (any target)."""
    if mlc_dir() is None:
        return False
    try:
        with _mlc_cwd():
            from mlc.runtime.backend import available
            return bool(available(target))
    except Exception:  # noqa: BLE001
        return False


def _ensure_interface_cache(target: str) -> None:
    from mlc.discover.cache import discovered_memory_map, dump_cache
    try:
        discovered_memory_map(target)
    except Exception:  # noqa: BLE001 — build the fingerprint-gated discovery cache on demand
        dump_cache(target)


def arc_core(target: str):
    """The target-AGNOSTIC ctypes arc model (mlc ``CosimCore``) for ``target``: poke/peek/step any signal
    by NAME. Works for every target mlc compiled from RTL — the compile-from-RTL oracle primitive. The
    high-level per-op driver (matmul-WS, SIMT, ...) is target-specific and provided by mlc; use
    :func:`arc_run_command_buffer` for the agnostic high-level path."""
    require_mlc()
    if not arc_available(target):
        raise RuntimeError(f"mlc arc model absent for target {target!r} (runs/circt-arc/{target})")
    with _mlc_cwd():
        _ensure_interface_cache(target)
        from mlc.backends.cosim_core import CosimCore
        from mlc.discover.fingerprint import artifact_paths
        p = artifact_paths(target)
        d = mlc_dir()
        return CosimCore(str((d / p["so"]).resolve()), str((d / p["man"]).resolve()))


def arc_run_command_buffer(cb: dict) -> dict:
    """Answer a merlin command buffer on the RTL-derived arc model — target-AGNOSTIC (mlc infers the
    target/config from the command buffer). Returns mlc's backend contract ``{outputs, metrics, correct,
    oracle, ...}`` where ``metrics`` carries the RTL's internal counts (cycles, bytes_moved,
    resident_hits, accumulator_commits, ...) — verilator-level state without verilator, for ANY target."""
    require_mlc()
    with _mlc_cwd():
        from mlc.runtime.backend import run_command_buffer
        return run_command_buffer(cb)
