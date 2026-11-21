"""Bridge to the sibling ``mlc`` model-ladder compiler — the CIRCT+xDSL RTL frontend we reuse for
RTL *needle* extraction (finding specific ISA facts in the HW-dialect op graph).

``mlc`` is an EXTERNAL dependency resolved via ``.env MERLIN_MLC_DIR`` (it will be upstreamed to its own
open-source repo). Rather than re-implement the op-graph parser + decoder analysis, we reuse mlc's
``discover.irgraph`` (``circt-opt --mlir-print-op-generic`` -> xDSL ``HwGraph``) and ``discover.decode``
(observed field values from the decoder's ``comb.icmp eq`` fan-out), plus mlc's own prebuilt CIRCT
binaries. ``mlc`` is pip-installed editable into merlin's venv from ``MERLIN_MLC_DIR`` (see the dev
setup docs), so its Python is a normal import. Imports are still function-local behind an availability
guard (``chia_bridge`` style) so importing this module never hard-requires mlc, and a machine without
the mlc package installed degrades honestly rather than crashing. Its NON-python assets (the prebuilt
``circt-opt`` binary, the cached ``runs/circt-arc/<target>`` arc outputs, schemas) still resolve by
directory path via ``MERLIN_MLC_DIR``.

Why this matters: a header parse and an RTL comparison set can disagree. Recording
the selected decoder field exposes that discrepancy; neither alone proves an
exhaustive executable ISA or authorizes a code-generation endpoint.

TARGET-AGNOSTIC: every entry point here takes a ``target`` argument and holds no target name. mlc is
target-parameterized (``artifact_paths(target)`` / ``discovered_memory_map(target)`` /
``discover_opcode_set(graph)`` / per-target ``runs/circt-arc/<target>``), so the same code plugs any HW
RTL repo mlc knows (systolic arrays, NPUs, SIMT cores, CPUs, ...).
"""

from __future__ import annotations

import copy
import shutil
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from merlin.common import external_tools as _modelir

# ---- mlc_bridge --------------------------------------------------------------------------------


def discover_config_subtype_field(target: str, *, sub_width: int = 2, low_bit: int = 0) -> dict:
    """Structurally DISCOVER the config-subtype sub-field decode: the values the RTL decoder fans out on
    the low ``sub_width`` bits of the command's rs1 (the ``rs1 & 0x3`` convention that refines the CONFIG
    funct into EX/LD/ST/BERT). Same machinery as :func:`discover_legal_opcodes` (``comb.icmp eq`` over a
    ``hw.constant``), but for the SUB-field: we keep only comparisons whose non-constant operand is a
    ``comb.extract`` of width ``sub_width`` at ``low_bit`` (the low rs1 bits), and take the union of the
    masked-unsigned constants across modules — the set of subtype selector values the silicon decodes.

    Also records, per decoded value, the command controller MODULE names that decode it (corroboration:
    the execute controller decodes the EX selector, the store controller the ST selector). Returns
    ``{values, width, by_value, modules, hw_source, method, evidence}`` (``values=None`` when mlc / the HW
    dialect is unavailable — honest fallback, never a guess).

    RESIDUAL (stated, not hidden): this reads every width-``sub_width`` low-bit extract, so the discovered
    set is the UNION over all such sub-fields, not a single proven-to-be-``cmd.rs1`` signal — isolating the
    exact rs1 net needs cross-module operand tracing. The compiler-emitted subtypes are cross-checked as a
    SUBSET of this union (containment), which is what grounds them as real decoded values."""
    from merlin.target.rtl.extraction.mlc_toolchain import circt_opt_bin, core_hw_mlir, require_mlc

    require_mlc()
    hw = core_hw_mlir(target)
    if hw is None:
        return {
            "values": None,
            "width": sub_width,
            "by_value": {},
            "modules": [],
            "hw_source": None,
            "method": "config_subtype_icmp_extract(mlc)",
            "evidence": f"no core HW dialect for target {target!r}",
        }

    from merlin.target.rtl.hw.graph import load_hw_graph

    graph = load_hw_graph(hw, circt_opt=circt_opt_bin())
    _EQ = 0  # comb.ICmpPredicate eq
    mask = (1 << sub_width) - 1
    by_value: dict[int, set[str]] = {}
    for module in graph.modules.values():
        for icmp in graph.ops("comb.icmp", within=module):
            if graph.icmp_predicate(icmp) != _EQ or len(icmp.operands) != 2:
                continue
            a, b = icmp.operands
            da, db = graph.defining_op(a), graph.defining_op(b)
            a_is_const = da is not None and da.op_name.data == "hw.constant"
            b_is_const = db is not None and db.op_name.data == "hw.constant"
            if a_is_const == b_is_const:
                continue
            const_op, sig = (da, b) if a_is_const else (db, a)
            sig_def = graph.defining_op(sig)
            if sig_def is None or sig_def.op_name.data != "comb.extract":
                continue
            er = graph.extract_range(sig_def)  # (lowBit, width)
            if not er or er[0] != low_bit or er[1] != sub_width:
                continue
            cv = graph.const_value(const_op)
            if cv is None:
                continue
            by_value.setdefault(cv & mask, set()).add(module.name)
    values = sorted(by_value)
    return {
        "values": values,
        "width": sub_width,
        "by_value": {v: sorted(by_value[v]) for v in values},
        "modules": sorted({m for ms in by_value.values() for m in ms}),
        "hw_source": str(hw),
        "method": "config_subtype_icmp_extract(mlc)",
        "evidence": f"union of comb.icmp-eq over width-{sub_width} rs1[{low_bit}:{low_bit + sub_width}] "
        f"extracts across {len({m for ms in by_value.values() for m in ms})} modules "
        f"-> decoded selector values {values}",
    }


def crosscheck_config_subtype(target: str) -> list[str]:
    """Cross-check the hand ABI ``config_subtype`` block against DERIVED sources: the CODES against the
    structurally-discovered rs1[low:low+2] selector value set (:func:`discover_config_subtype_field`,
    containment) and the NAMES against the ISA-header ``#define CONFIG_* <N>`` defines
    (``circt_introspect.crosscheck_config_subtype_names``). Returns the list of DISAGREEMENTS (empty = the
    hand block is fully reproduced by discovery + headers). A code the silicon never decodes on the sub-field,
    or a name the header does not define at that code, is a real alarm. Skips the structural half honestly
    when mlc / the HW dialect is unavailable (names still cross-check via the pure header parse)."""
    from merlin.target.registry.experiment_descriptor import load_capability_manifest
    from merlin.target.rtl.extraction import circt_decode as CI

    hand = (load_capability_manifest(target).encoding or {}).get("config_subtype") or {}
    dis: list[str] = []
    # NAMES: header-derived (pure parse; no mlc).
    nm = CI.crosscheck_config_subtype_names(target, hand)
    for code, name in nm["missing"]:
        dis.append(f"config_subtype {code} ({name}): header defines no `#define {name} <N>`")
    for code, name, hcode in nm["mismatch"]:
        dis.append(f"config_subtype {code} ({name}): header defines {name}={hcode}, not {code}")
    # CODES: structural containment in the discovered rs1[1:0] selector set (best-effort; needs mlc).
    try:
        field = discover_config_subtype_field(target)
    except Exception as e:  # noqa: BLE001 — no HW dialect => skip the structural half, keep the name check
        field = {"values": None, "evidence": f"{type(e).__name__}: {e}"}
    vals = field.get("values")
    if vals is not None:
        for code in hand:
            if int(code) not in set(vals):
                dis.append(f"config_subtype code {int(code)} not in RTL-decoded rs1 selector set {vals}")
    return dis


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
    with NO arc (a SIMT/prototype target, or mlc absent) keeps the honest empty ``{derived: False,
    reason}`` — we never auto-regen where the probe is impossible, and a failed probe degrades honestly
    rather than crashing the bundle. Returns ``{roles: {opcode:int -> role}, source, derived: bool,
    reason}``."""
    from merlin.target.rtl.extraction.mlc_discovery import _arc_model_present
    from merlin.target.rtl.extraction.mlc_toolchain import mlc_dir

    d = mlc_dir()
    if d is None:
        return {"roles": {}, "source": None, "derived": False, "reason": "MERLIN_MLC_DIR unset"}
    cache = d / "runs" / "circt-arc" / target / "outputs" / "discovered_roles.json"
    if not cache.is_file():
        # Regenerate the roles cache on demand ONLY where the probe is possible (a live arc model). A
        # SIMT/prototype target with no arc keeps the honest-empty path below — never auto-regen where
        # impossible, never crash the bundle if the probe fails.
        if _arc_model_present(target):
            try:
                derive_and_cache_roles(target)
            except Exception as e:  # noqa: BLE001 — a failed probe is honest-unavailable, never a crash/guess
                return {
                    "roles": {},
                    "source": None,
                    "derived": False,
                    "reason": f"role derivation failed for {target!r}: {type(e).__name__}: {e}",
                }
    if not cache.is_file():
        return {
            "roles": {},
            "source": None,
            "derived": False,
            "reason": f"no discovered_roles cache for {target!r} — run derive_and_cache_roles in the mlc venv",
        }
    try:
        import json as _json

        data = _json.loads(cache.read_text())
        roles = {int(k): v for k, v in (data.get("roles") or {}).items() if v in SEMANTIC_ROLES}
        return {
            "roles": roles,
            "source": data.get("method", "probe:effect_signature(mlc)"),
            "derived": True,
            "reason": f"{len(roles)} opcode roles from the mlc effect probe",
        }
    except Exception as e:  # noqa: BLE001 — an unreadable cache is honest-unavailable, never a guess
        return {"roles": {}, "source": None, "derived": False, "reason": f"unreadable roles cache: {e}"}


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
    (:func:`~merlin.target.registry.experiment_descriptor.load_capability_manifest`), assert the derived coarse role
    equals the coarse projection of the hand label (:func:`_coarse_of_hand_class`). Returns the list of
    DISAGREEMENTS (empty = all agree). A disagreement is a real alarm — either the hand label mislabels the
    funct or the RTL behaviour drifted from the declared ABI. Functs with no derived role (not swept / no
    arc) or an unmappable label are skipped, not counted as agreements — so a no-arc/SIMT target yields an
    empty list (nothing derived to cross-check) rather than a crash."""
    roles = semantic_roles(target).get("roles") or {}
    if not roles:
        return []  # nothing behaviourally derived (no arc / SIMT / mlc absent) -> nothing to cross-check
    from merlin.target.registry.experiment_descriptor import load_capability_manifest

    hand = (load_capability_manifest(target).encoding or {}).get("semantic_class") or {}
    disagreements: list[str] = []
    for code, label in hand.items():
        derived = roles.get(int(code))
        expect = _coarse_of_hand_class(str(label))
        if derived is None or expect is None:
            continue
        if derived != expect:
            disagreements.append(
                f"funct {int(code)} ({label}): derived role {derived!r} != coarse({label!r})={expect!r}"
            )
    return disagreements


FINE_ROLES = (
    "MVIN",
    "MVIN2",
    "MVIN3",
    "MVOUT",
    "PRELOAD",
    "COMPUTE_PRELOADED",
    "COMPUTE_ACCUMULATE",
    "LOOP_WS",
    "LOOP_CONV",
    "CONFIG",
    "FLUSH",
)


def _classify_fine_role(coarse: str | None, feats: dict) -> tuple[str | None, bool]:
    """Map a funct's coarse role + MEASURED fine behaviour to a fine ISA name. Returns
    ``(name, fine_derived)`` where ``fine_derived`` is True when the name comes from a behavioural feature
    the coarse probe discards (config-id, accumulator-write, weight-flip, loop-FSM), and False when it is a
    coarse-grounded fallback (store->MVOUT, barrier->FLUSH, base config->CONFIG) that is unique among the
    declared functs but not itself finer than the coarse role. The name is a pure function of the MEASURED
    behaviour, never of the funct number — so an RTL change that altered the behaviour would change the
    derived name and trip the cross-check."""
    # loop macros first: a compute/config funct that runs a loop FSM is a loop, whatever its coarse bucket.
    if feats.get("drives_matmul_fsm"):
        return "LOOP_WS", True  # WS matmul loop-unroller
    if feats.get("drives_conv_fsm"):
        return "LOOP_CONV", True  # CONV loop-unroller
    if coarse == "load":
        cid = feats.get("config_id")
        name = {0: "MVIN", 1: "MVIN2", 2: "MVIN3"}.get(cid)
        return (name, True) if name is not None else (None, False)
    if coarse == "store":
        return "MVOUT", False  # coarse-grounded (the sole declared store)
    if coarse == "barrier":
        return "FLUSH", False  # coarse-grounded fence (not separable from other barriers)
    if coarse == "compute":
        if feats.get("writes_accumulator"):
            # weight-flip (propagate toggled on ~all PEs) => uses the freshly PRELOADED weights;
            # no flip => REUSES the stationary weights (ACCUMULATE).
            return ("COMPUTE_PRELOADED", True) if feats.get("weight_flip") else ("COMPUTE_ACCUMULATE", True)
        if feats.get("loads_weights"):
            return "PRELOAD", True  # loads the stationary weights, commits no MAC
        return None, False
    if coarse == "config":
        return "CONFIG", False  # coarse-grounded base config command
    return None, False


def derive_fine_roles(target: str, *, drive_cycles: int = 800) -> dict:
    """Behaviourally DERIVE the FINE ISA role of each legal funct on ``target``'s arc model and write
    ``discovered_fine_roles.json`` beside the coarse ``discovered_roles.json``. Runs the mlc fine probe
    (:func:`mlc.discover.fine_roles.probe_fine_features`), classifies each funct's measured feature vector
    with :func:`_classify_fine_role`, and records the fine name + whether it is behaviourally fine-derived
    (vs a coarse-grounded fallback) + the raw features (auditable). Requires a live arc model (+ the mlc
    package); raises otherwise (never fabricates). This is the ONLY writer of the fine cache that
    :func:`fine_roles` reads. Returns the written record."""
    from merlin.target.rtl.extraction.mlc_discovery import _ensure_interface_cache, _mlc_cwd
    from merlin.target.rtl.extraction.mlc_toolchain import discover_legal_opcodes, mlc_dir, require_mlc

    require_mlc()
    import json as _json

    coarse = semantic_roles(target)
    roles = coarse.get("roles") or {}
    if not roles:
        raise RuntimeError(f"{target!r}: no coarse roles derived — cannot refine (reason: {coarse.get('reason')})")
    legal = discover_legal_opcodes(target).get("legal_opcodes")
    if not legal:
        raise RuntimeError(f"{target!r}: no legal opcode set discovered — cannot probe fine roles")
    with _mlc_cwd():
        _ensure_interface_cache(target)  # (re)build the fingerprint-gated discovery cache if stale/cold
        from mlc.discover.fine_roles import probe_fine_features
        from mlc.discover.fingerprint import artifact_paths, rtl_fingerprint

        probed = probe_fine_features(
            target, {str(k): v for k, v in roles.items()}, list(legal), drive_cycles=drive_cycles
        )
        if not probed.get("available"):
            raise RuntimeError(f"{target!r}: fine probe unavailable — {probed.get('reason')}")
        features = probed.get("features") or {}
        fine: dict[str, str] = {}
        fine_derived: dict[str, bool] = {}
        for code in legal:
            feats = features.get(str(code), {})
            name, derived = _classify_fine_role(roles.get(int(code)), feats)
            if name is not None:
                fine[str(code)] = name
                fine_derived[str(code)] = derived
        record = {
            "target": target,
            "method": "probe:fine-behavioural-role(mlc-arc)",
            "fingerprint": rtl_fingerprint(target).as_dict(),
            "drive_cycles": drive_cycles,
            "legal_opcodes": list(legal),
            "coarse_roles": {str(k): v for k, v in roles.items()},
            "fine_roles": fine,
            "fine_derived": fine_derived,
            "features": features,
            "note": (
                "Fine ISA roles behaviourally derived on the arc model: the compute cluster split by "
                "accumulator-write value-magnitude + the PE propagate-bit (last_s) weight-flip "
                "(PRELOAD/COMPUTE_PRELOADED/COMPUTE_ACCUMULATE); the load cluster split by a per-"
                "config-id DMA row-stride sweep (MVIN/MVIN2/MVIN3); the loop macros split by which "
                "loop FSM (from_matmul_fsm vs from_conv_fsm) tags their spawned reservation-station "
                "commands (LOOP_WS/LOOP_CONV). CONFIG/MVOUT/FLUSH are coarse-grounded fallbacks "
                "(fine_derived=false): unique among the declared functs but not finer than the coarse "
                "role. Other barrier/config functs (e.g. flush-family 22/126, config sub-commands "
                "9-13/16-21/24) are left unnamed — the probe cannot cleanly name them."
            ),
        }
        d = mlc_dir()
        p = artifact_paths(target)
        out = (d / p["man"]).resolve().parent / "discovered_fine_roles.json"
        out.write_text(_json.dumps(record, indent=2) + "\n")
        record["_path"] = str(out)
        return record


def fine_roles(target: str) -> dict:
    """DERIVED opcode -> FINE ISA role map (COMPUTE_PRELOADED/COMPUTE_ACCUMULATE/PRELOAD/MVIN/MVIN2/MVIN3/
    LOOP_WS/LOOP_CONV/MVOUT/CONFIG/FLUSH), or EMPTY when it cannot be grounded. Mirrors :func:`semantic_roles`:
    reads the ``discovered_fine_roles.json`` cache, regenerating on demand ONLY where the probe is possible
    (a live arc model) and degrading honestly (empty ``{derived: False, reason}``) for a no-arc/SIMT target
    or a failed probe. Returns ``{roles: {opcode:int -> fine name}, fine_derived: {opcode:int -> bool},
    source, derived: bool, reason}``."""
    from merlin.target.rtl.extraction.mlc_discovery import _arc_model_present
    from merlin.target.rtl.extraction.mlc_toolchain import mlc_dir

    d = mlc_dir()
    if d is None:
        return {"roles": {}, "fine_derived": {}, "source": None, "derived": False, "reason": "MERLIN_MLC_DIR unset"}
    cache = d / "runs" / "circt-arc" / target / "outputs" / "discovered_fine_roles.json"
    if not cache.is_file():
        if _arc_model_present(target):
            try:
                derive_fine_roles(target)
            except Exception as e:  # noqa: BLE001 — a failed probe is honest-unavailable, never a crash/guess
                return {
                    "roles": {},
                    "fine_derived": {},
                    "source": None,
                    "derived": False,
                    "reason": f"fine role derivation failed for {target!r}: {type(e).__name__}: {e}",
                }
    if not cache.is_file():
        return {
            "roles": {},
            "fine_derived": {},
            "source": None,
            "derived": False,
            "reason": f"no discovered_fine_roles cache for {target!r} — run derive_fine_roles in the mlc venv",
        }
    try:
        import json as _json

        data = _json.loads(cache.read_text())
        roles = {int(k): v for k, v in (data.get("fine_roles") or {}).items() if v in FINE_ROLES}
        fd = {int(k): bool(v) for k, v in (data.get("fine_derived") or {}).items()}
        return {
            "roles": roles,
            "fine_derived": fd,
            "source": data.get("method", "probe:fine-behavioural-role(mlc-arc)"),
            "derived": True,
            "reason": f"{len(roles)} fine opcode roles from the mlc fine probe",
        }
    except Exception as e:  # noqa: BLE001 — an unreadable cache is honest-unavailable, never a guess
        return {
            "roles": {},
            "fine_derived": {},
            "source": None,
            "derived": False,
            "reason": f"unreadable fine roles cache: {e}",
        }


def crosscheck_semantic_class_fine(target: str) -> list[str]:
    """ALWAYS-ON cross-check gate at the FINE level: VERIFY the hand ABI ``semantic_class`` NAMES against
    the behaviourally DERIVED fine roles. For every funct that has BOTH a derived fine role (from
    :func:`fine_roles`) AND a hand ``semantic_class`` label, assert the derived fine name EQUALS the hand
    label. Returns the list of DISAGREEMENTS (empty = every declared name reproduced by RTL behaviour). A
    disagreement is a real alarm: the hand name claims a behaviour the funct does not exhibit (mislabel or
    RTL drift). Functs with no derived fine role (not probed / no arc / unnamed by the probe) are skipped,
    so a no-arc/SIMT target yields an empty list rather than a crash. This is the fine-grained companion to
    :func:`crosscheck_semantic_class` (which verifies only the coarse projection)."""
    fr = fine_roles(target)
    roles = fr.get("roles") or {}
    if not roles:
        return []  # nothing behaviourally derived (no arc / SIMT / mlc absent) -> nothing to cross-check
    from merlin.target.registry.experiment_descriptor import load_capability_manifest

    hand = (load_capability_manifest(target).encoding or {}).get("semantic_class") or {}
    disagreements: list[str] = []
    for code, label in hand.items():
        derived = roles.get(int(code))
        if derived is None:
            continue  # this funct was not fine-named by the probe (report via fine_roles, not an alarm here)
        if derived != str(label):
            disagreements.append(
                f"funct {int(code)} ({label}): derived fine role {derived!r} != hand semantic_class {str(label)!r}"
            )
    return disagreements


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

    def g(key):
        return regions.get(key, 0)

    stalled = busy >= int(drive_cycles * 0.9)
    if g("mesh") > 50:
        return "compute"  # heavy systolic-array datapath
    if g("store_ctrl") > 0:
        return "store"  # DMA store engine active
    if stalled and (g("spad_ctrl") > 0 or g("tlb") > 0 or g("load_ctrl") > 0):
        return "load"  # stalled scratchpad DMA read (fill)
    if g("mesh") > 0:
        return "compute"  # light mesh touch = weight-preload
    if g("ex_ctrl") > 0 or g("dma_mod0") > 0 or g("dma_mod1") > 0 or g("spad_ctrl") > 0:
        return "config"  # CSR / loop-config pokes, no datapath
    return "barrier"  # negligible observable effect


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
    from merlin.target.rtl.extraction.mlc_discovery import _mlc_cwd
    from merlin.target.rtl.extraction.mlc_toolchain import discover_legal_opcodes, mlc_dir, require_mlc

    require_mlc()
    import json as _json

    with _mlc_cwd():
        from mlc.backends.cosim_core import CosimCore
        from mlc.backends.protocols import RoCCAdapter, detect_protocols
        from mlc.discover.fingerprint import artifact_paths, rtl_fingerprint
        from mlc.discover.probe import build_statemap, diff, snapshot, zero_state

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
                f"(no io_cmd_bits_inst_funct on this model)"
            )
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
            evidence[str(funct)] = {
                "role": role,
                "accepted": True,
                "busy": busy,
                "n_changed": len(changed),
                "regions": rc,
            }

        record = {
            "target": target,
            "method": "probe:multicycle-effect-region(mlc-arc)",
            "fingerprint": rtl_fingerprint(target).as_dict(),
            "drive_cycles": drive_cycles,
            "legal_opcodes": list(legal),
            "roles": roles,
            "evidence": evidence,
            "note": (
                "Coarse roles behaviourally derived by driving each RoCC funct to retire on the arc "
                "model and bucketing the retired-state region signature. Multi-cycle drive + "
                "idle-noise subtraction defeats reservation-station/control-queue domination. Fine "
                "ISA names (COMPUTE_PRELOADED vs _ACCUMULATE, MVIN2/3, LOOP_* variants, readout_bits) "
                "are ABI, not behaviourally nameable — they stay hand-owned in target_contract.yaml."
            ),
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
    from merlin.target.rtl.extraction.mlc_discovery import _ensure_interface_cache, _mlc_cwd
    from merlin.target.rtl.extraction.mlc_toolchain import mlc_dir
    from merlin.target.rtl.extraction.source_selection import active_selection

    if active_selection(target) is not None:
        return None  # No manifest tied to this explicit hardware; FIRRTL census supplies stores.
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
    from merlin.target.rtl.extraction.mlc_discovery import _ensure_interface_cache, _mlc_cwd
    from merlin.target.rtl.extraction.mlc_toolchain import mlc_dir
    from merlin.target.rtl.extraction.source_selection import active_selection

    if active_selection(target) is not None:
        mesh = discovered_mesh(target)
        return mesh.get("dim") if mesh else None
    if mlc_dir() is None:
        return None
    try:
        with _mlc_cwd():
            _ensure_interface_cache(target)
            from mlc.discover.cache import discovered_dim as _dim

            return int(_dim(target))
    except Exception:  # noqa: BLE001
        return None


def discovered_mesh(target: str) -> dict | None:
    """The target's compute-mesh record as DISCOVERED from the RTL — geometry plus the corroboration that
    makes it a fact rather than a ranking: which module holds the grid, which element is replicated and how
    many times (summed across CIRCT's structural variants), and the multiply/add/register counts of the
    multiply-accumulate idiom the recognizer confirmed.

    Callers must prefer this over :func:`discovered_dim`. A bare DIM cannot be refuted by whoever reads it
    downstream, and an unrefutable number is exactly how a SIMT cluster came to publish a 17x17 mesh that
    was 289 flip-flops in a divide/sqrt unit. None when the target has no mesh / mlc is unavailable.
    Target-agnostic."""
    from merlin.target.rtl.extraction.mlc_discovery import _ensure_interface_cache, _mlc_cwd
    from merlin.target.rtl.extraction.mlc_toolchain import circt_opt_bin, core_hw_mlir, mlc_dir
    from merlin.target.rtl.extraction.source_selection import active_selection

    selected = active_selection(target)
    if selected is not None:
        if "_mesh_discovery" not in selected:
            with _modelir.discovery_imports(mlc_dir()):
                from mlc.discover.compute_idiom import recognize_mac_mesh

                from merlin.target.rtl.hw.graph import load_hw_graph

                graph = load_hw_graph(core_hw_mlir(target), circt_opt=circt_opt_bin())
                mesh = recognize_mac_mesh(graph)
                selected["_mesh_discovery"] = (
                    None
                    if mesh is None
                    else {
                        "dim": mesh.dim,
                        "parent": mesh.parent,
                        "child": mesh.child,
                        "count": mesh.count,
                        "muls": mesh.muls,
                        "adds": mesh.adds,
                        "regs": mesh.regs,
                        "elements": [list(element) for element in mesh.elements],
                        "corroborated": True,
                    }
                )
        return copy.deepcopy(selected["_mesh_discovery"])
    if mlc_dir() is None:
        return None
    try:
        with _mlc_cwd():
            _ensure_interface_cache(target)
            from mlc.discover.cache import load_interface

            mesh = load_interface(target).get("mesh")
            return dict(mesh) if isinstance(mesh, dict) else None
    except Exception:  # noqa: BLE001
        return None
