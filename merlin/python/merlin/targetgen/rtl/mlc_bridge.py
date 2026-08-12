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

import copy
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

_ARC_TARGET_CACHE: dict[str, str] = {}


def _arc_target(target: str) -> str:
    """The mlc arc/fingerprint key for a merlin ``target``. A composite/aliased target — e.g. a SIMT SoC
    whose bit-exact arc model mlc registers under the embedding cluster's name, not the merlin-facing
    name — declares ``arc_target`` in its ``contracts/residual.yaml``; every other target maps to itself.
    DERIVED from the residual (no target-name literal here), fail-closed to the target itself when there
    is no residual / no alias. Applied ONLY on the ORACLE path — ``arc_available`` / ``arc_core`` (the
    bit-exact model) and the program-oracle cosim lookups, plus the SIMT fact delegation in
    :func:`_simt_fact_bundle`. It is deliberately NOT applied to the structural-fact discovery
    (``core_hw_mlir`` / ``discovered_dim`` / ``discovered_memory_map`` / ``discovered_memories``): a SIMT
    SoC must not inherit the embedding cluster's systolic geometry as if it were its own RoCC mesh (that
    would fabricate spatial levers). The merlin-facing identity (corpus, contract, suite, results) stays
    the original ``target`` everywhere. The residual is a plain YAML side-input, so this never triggers
    manifest derivation (which would re-enter the mlc boundary through this resolver)."""
    if target in _ARC_TARGET_CACHE:
        return _ARC_TARGET_CACHE[target]
    alias = target
    try:
        from ..capability_manifests import _load_residual
        declared = (_load_residual(target) or {}).get("arc_target")
        if isinstance(declared, str) and declared:
            alias = declared
    except Exception:  # noqa: BLE001 — no residual / no alias ⇒ identity
        alias = target
    _ARC_TARGET_CACHE[target] = alias
    return alias


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


def isa_encoding_for(target: str) -> dict | None:
    """The mlc-derived instruction-ENCODING fact (``{inst_width, fields, opcodes, provenance}``) for a
    target, from ``runs/circt-arc/<arc_target>/outputs/muon_isa.json`` — the field bit-ranges + opcode table
    the mlc ``isa_encoding`` pass recovers from the target's RTL decoder. Uses the ORACLE-path arc alias
    (:func:`_arc_target`) because the encoding is the aliased CORE's own ISA (a SIMT SoC's decoder is the
    embedded core's), unlike structural geometry. Returns the parsed fact, or None when mlc / the cache is
    absent — an honest fallback the caller degrades on, never a guessed encoding."""
    d = mlc_dir()
    if d is None:
        return None
    p = d / "runs" / "circt-arc" / _arc_target(target) / "outputs" / "muon_isa.json"
    if not p.is_file():
        return None
    try:
        import json
        fact = json.loads(p.read_text())
        return fact if isinstance(fact, dict) and fact.get("fields") else None
    except Exception:  # noqa: BLE001 — unreadable/garbled cache ⇒ honest None
        return None


def mx_mmio_for(target: str) -> dict | None:
    """The target's MX-Gemmini accelerator MMIO command ABI (``{ctrl_base, reg_offsets, inst_word, funct,
    sf_mem, gpu_dram_offset, dim, group, config_ex, bounds, loop_ws, mxquant_config_mvout, read_smem}``) —
    the memory-mapped RoCC command surface a Muon SIMT kernel pushes to drive the block-scaled matmul PE.
    This is an ABI the RTL decoder cannot ground (like gemmini's ``encoding`` residual), so it is declared,
    header-derived + human-reviewed, in the target's ``contracts/residual.yaml`` under ``mx_mmio``. Returns
    the parsed fact, or None when the target ships no MX MMIO contract — an honest fallback the emitter/harness
    degrade on (the mxfp8 op is unsupported), never a guessed base/opcode."""
    try:
        from ..capability_manifests import _load_residual
        fact = (_load_residual(target) or {}).get("mx_mmio")
        return fact if isinstance(fact, dict) and fact.get("ctrl_base") is not None else None
    except Exception:  # noqa: BLE001 — no residual / unreadable ⇒ honest None
        return None


def opu_artifact_paths(target: str) -> dict | None:
    """Resolved ``{hw, man, so}`` artifact paths for a SPATIAL/OPU target via mlc's per-target
    fingerprint map — which knows the two-level ``runs/circt-arc/<family>/<config>/outputs`` layout the
    OuterProductUnit uses (e.g. ``saturn_opu/mxv256d128``), UNLIKE the flat one-level layout
    :func:`core_hw_mlir` assumes. Absolute paths under the mlc dir. None when mlc is not resolvable or
    the target is not a known mlc arc target — an honest fallback, never a guessed path.

    Only mlc's lightweight ``discover.fingerprint`` is imported (pure pathlib; no circt binary), so OPU
    fact extraction works even when the full ``mlc_available`` gate (circt-opt built) is not met — and
    mlc is imported by-path for this call only (context-managed) so it grounds even when mlc is a sibling
    checkout rather than a pip-installed package."""
    d = mlc_dir()
    if d is None:
        return None
    try:
        with _mlc_importable(d):
            from mlc.discover.fingerprint import artifact_paths
            return {k: Path(v) for k, v in artifact_paths(target, base=d).items()}
    except Exception:  # noqa: BLE001 — unknown target / mlc layout skew ⇒ honest unavailable
        return None


def compute_unit_dtypes(target: str) -> dict | None:
    """Per-compute-unit INPUT dtype map ``{unit_name: [dtype, ...]}`` for ANY target, DERIVED structurally
    from the RTL facts by mlc's GENERAL datapath-dtype extractor
    (:func:`mlc.discover.datapath_dtypes.compute_unit_dtypes_detail` — typed MAC-mesh/FPU + spatial OPU
    families, no per-target ``if``). This is the general lift of the OPU-only dtype code that grounded
    only saturn_opu; it reproduces that same value (``saturn_opu -> [int8, fp8_e4m3, fp8_e5m2]``) and also
    grounds gemmini (``[int8]``) / atlas (``[fp8_e4m3]``) / an FPU core (``[fp16, f32, f64]``).

    Returns ``None`` — an honest fallback the caller degrades to the existing behavior on — when mlc is
    unresolvable OR the extractor reports the target's facts carry no derivable compute datapath
    (``supported=False``); never a fabricated dtype. mlc is imported by-path for this call only
    (context-managed; never a global sys.path insert that would flip ``mlc_available`` process-wide)."""
    d = mlc_dir()
    if d is None:
        return None
    try:
        with _mlc_importable(d):
            from mlc.discover.datapath_dtypes import compute_unit_dtypes_detail
            detail = compute_unit_dtypes_detail(target, base=d)
    except Exception:  # noqa: BLE001 — mlc absent / layout skew ⇒ honest unavailable, never a guess
        return None
    if not detail.get("supported"):
        return None
    return {u["unit"]: u["dtypes"] for u in detail.get("units", [])}


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
    require_mlc()
    hw = core_hw_mlir(target)
    if hw is None:
        return {"values": None, "width": sub_width, "by_value": {}, "modules": [], "hw_source": None,
                "method": "config_subtype_icmp_extract(mlc)",
                "evidence": f"no core HW dialect for target {target!r}"}
    from mlc.discover import irgraph
    graph = irgraph.load_hw_graph(hw, circt_opt=circt_opt_bin())
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
    return {"values": values, "width": sub_width,
            "by_value": {v: sorted(by_value[v]) for v in values},
            "modules": sorted({m for ms in by_value.values() for m in ms}),
            "hw_source": str(hw), "method": "config_subtype_icmp_extract(mlc)",
            "evidence": f"union of comb.icmp-eq over width-{sub_width} rs1[{low_bit}:{low_bit+sub_width}] "
                        f"extracts across {len({m for ms in by_value.values() for m in ms})} modules "
                        f"-> decoded selector values {values}"}


def crosscheck_config_subtype(target: str) -> list[str]:
    """Cross-check the hand ABI ``config_subtype`` block against DERIVED sources: the CODES against the
    structurally-discovered rs1[low:low+2] selector value set (:func:`discover_config_subtype_field`,
    containment) and the NAMES against the ISA-header ``#define CONFIG_* <N>`` defines
    (``circt_introspect.crosscheck_config_subtype_names``). Returns the list of DISAGREEMENTS (empty = the
    hand block is fully reproduced by discovery + headers). A code the silicon never decodes on the sub-field,
    or a name the header does not define at that code, is a real alarm. Skips the structural half honestly
    when mlc / the HW dialect is unavailable (names still cross-check via the pure header parse)."""
    from ..target_experiment import load_capability_manifest
    from . import circt_introspect as CI
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
        if _arc_model_present(target):
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


# --- FINE behavioural role derivation + fine cross-check ----------------------------------------
# The coarse probe collapses several ISA commands that share a coarse role but differ architecturally:
# COMPUTE_PRELOADED vs COMPUTE_ACCUMULATE vs PRELOAD (all "compute"), MVIN vs MVIN2 vs MVIN3 (all "load"),
# LOOP_WS vs LOOP_CONV (the two loop macros). The mlc FINE probe (mlc.discover.fine_roles) measures the
# behaviour that separates them on the arc model: the accumulator-write value-magnitude + the PE
# propagate-bit toggle (compute cluster), a per-config-id DMA stride sweep (load cluster), and which loop
# FSM tags the spawned reservation-station commands (loop cluster). This makes the hand semantic_class
# NAMES behaviourally derived + cross-checked, not merely header-named.

# The fine ISA vocabulary the derivation can behaviourally NAME (a superset of the coarse SEMANTIC_ROLES).
FINE_ROLES = ("MVIN", "MVIN2", "MVIN3", "MVOUT", "PRELOAD", "COMPUTE_PRELOADED", "COMPUTE_ACCUMULATE",
              "LOOP_WS", "LOOP_CONV", "CONFIG", "FLUSH")


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
        return "LOOP_WS", True                     # WS matmul loop-unroller
    if feats.get("drives_conv_fsm"):
        return "LOOP_CONV", True                   # CONV loop-unroller
    if coarse == "load":
        cid = feats.get("config_id")
        name = {0: "MVIN", 1: "MVIN2", 2: "MVIN3"}.get(cid)
        return (name, True) if name is not None else (None, False)
    if coarse == "store":
        return "MVOUT", False                      # coarse-grounded (the sole declared store)
    if coarse == "barrier":
        return "FLUSH", False                      # coarse-grounded fence (not separable from other barriers)
    if coarse == "compute":
        if feats.get("writes_accumulator"):
            # weight-flip (propagate toggled on ~all PEs) => uses the freshly PRELOADED weights;
            # no flip => REUSES the stationary weights (ACCUMULATE).
            return ("COMPUTE_PRELOADED", True) if feats.get("weight_flip") else ("COMPUTE_ACCUMULATE", True)
        if feats.get("loads_weights"):
            return "PRELOAD", True                 # loads the stationary weights, commits no MAC
        return None, False
    if coarse == "config":
        return "CONFIG", False                     # coarse-grounded base config command
    return None, False


def derive_fine_roles(target: str, *, drive_cycles: int = 800) -> dict:
    """Behaviourally DERIVE the FINE ISA role of each legal funct on ``target``'s arc model and write
    ``discovered_fine_roles.json`` beside the coarse ``discovered_roles.json``. Runs the mlc fine probe
    (:func:`mlc.discover.fine_roles.probe_fine_features`), classifies each funct's measured feature vector
    with :func:`_classify_fine_role`, and records the fine name + whether it is behaviourally fine-derived
    (vs a coarse-grounded fallback) + the raw features (auditable). Requires a live arc model (+ the mlc
    package); raises otherwise (never fabricates). This is the ONLY writer of the fine cache that
    :func:`fine_roles` reads. Returns the written record."""
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
        probed = probe_fine_features(target, {str(k): v for k, v in roles.items()}, list(legal),
                                     drive_cycles=drive_cycles)
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
            "note": ("Fine ISA roles behaviourally derived on the arc model: the compute cluster split by "
                     "accumulator-write value-magnitude + the PE propagate-bit (last_s) weight-flip "
                     "(PRELOAD/COMPUTE_PRELOADED/COMPUTE_ACCUMULATE); the load cluster split by a per-"
                     "config-id DMA row-stride sweep (MVIN/MVIN2/MVIN3); the loop macros split by which "
                     "loop FSM (from_matmul_fsm vs from_conv_fsm) tags their spawned reservation-station "
                     "commands (LOOP_WS/LOOP_CONV). CONFIG/MVOUT/FLUSH are coarse-grounded fallbacks "
                     "(fine_derived=false): unique among the declared functs but not finer than the coarse "
                     "role. Other barrier/config functs (e.g. flush-family 22/126, config sub-commands "
                     "9-13/16-21/24) are left unnamed — the probe cannot cleanly name them."),
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
    d = mlc_dir()
    if d is None:
        return {"roles": {}, "fine_derived": {}, "source": None, "derived": False, "reason": "MERLIN_MLC_DIR unset"}
    cache = d / "runs" / "circt-arc" / target / "outputs" / "discovered_fine_roles.json"
    if not cache.is_file():
        if _arc_model_present(target):
            try:
                derive_fine_roles(target)
            except Exception as e:  # noqa: BLE001 — a failed probe is honest-unavailable, never a crash/guess
                return {"roles": {}, "fine_derived": {}, "source": None, "derived": False,
                        "reason": f"fine role derivation failed for {target!r}: {type(e).__name__}: {e}"}
    if not cache.is_file():
        return {"roles": {}, "fine_derived": {}, "source": None, "derived": False,
                "reason": f"no discovered_fine_roles cache for {target!r} — run derive_fine_roles in the mlc venv"}
    try:
        import json as _json
        data = _json.loads(cache.read_text())
        roles = {int(k): v for k, v in (data.get("fine_roles") or {}).items() if v in FINE_ROLES}
        fd = {int(k): bool(v) for k, v in (data.get("fine_derived") or {}).items()}
        return {"roles": roles, "fine_derived": fd, "source": data.get("method", "probe:fine-behavioural-role(mlc-arc)"),
                "derived": True, "reason": f"{len(roles)} fine opcode roles from the mlc fine probe"}
    except Exception as e:  # noqa: BLE001 — an unreadable cache is honest-unavailable, never a guess
        return {"roles": {}, "fine_derived": {}, "source": None, "derived": False, "reason": f"unreadable fine roles cache: {e}"}


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
    from ..target_experiment import load_capability_manifest
    hand = (load_capability_manifest(target).encoding or {}).get("semantic_class") or {}
    disagreements: list[str] = []
    for code, label in hand.items():
        derived = roles.get(int(code))
        if derived is None:
            continue  # this funct was not fine-named by the probe (report via fine_roles, not an alarm here)
        if derived != str(label):
            disagreements.append(
                f"funct {int(code)} ({label}): derived fine role {derived!r} != hand semantic_class {str(label)!r}")
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
    # Robust to non-systolic bundle shapes: a simt bundle (mlc_bridge._simt_fact_bundle) has no
    # legal_opcodes/mesh_dim/capacities keys, so a missing OR underived field renders as "unavailable"
    # rather than raising. Gemmini (systolic) has all three present -> byte-identical brief.
    # TODO(simt-render): a dedicated SIMT/vector renderer so a simt target's brief surfaces its
    # registers/shared-memory/fp-datapath facts instead of this honest-degraded all-"unavailable" line.
    lo_f = f.get("legal_opcodes") or {}
    md_f = f.get("mesh_dim") or {}
    cap_f = f.get("capacities") or {}
    lo = lo_f.get("value")
    lines = [f"# Target ISA facts: {b['target']}",
             f"_Derived by {b['method']}. {b['n_derived']}/4 fields grounded; ungrounded = unavailable, "
             f"not guessed._", ""]
    if lo_f.get("derived") and lo is not None:
        lines += [f"- **Legal opcodes** ({len(lo)}): `{lo}`",
                  f"  - source: {lo_f.get('source')}"]
    else:
        lines.append("- **Legal opcodes**: unavailable (no decoder signal / no HW dialect for this target)")
    lines.append(f"- **Mesh DIM**: {md_f['value'] if md_f.get('derived') else 'unavailable'}")
    caps = cap_f.get("value") if cap_f.get("derived") else None
    if caps:
        lines.append(f"- **On-chip capacity (bytes)**: operand={caps.get('operand_bytes')}, "
                     f"accumulator={caps.get('accumulator_bytes')}")
    else:
        lines.append("- **On-chip capacity**: unavailable")
    return "\n".join(lines) + "\n"


# -------------------------------------------------------------- SPATIAL tensor-tile (OPU) fact bundle
# The spatial siblings of target_fact_bundle / render_fact_bundle. The systolic path (target_fact_bundle
# above) reads the RoCC decoder + mesh DIM; a spatial tensor tile (Saturn OuterProductUnit) has NO RoCC
# funct decode and its geometry is a cluster x cell tile that discover_mesh_dim mis-derives — so it gets
# its own structural extractor (merlin.targetgen.rtl.spatial_introspect), delegated to here.

# generate_prompt.py routes through fact_bundle_for + render_fact_bundle_for (below) by KIND, so a
# spatial/simt target's brief renders from its own extractor; gemmini (systolic) stays byte-identical.

def spatial_fact_bundle(target: str) -> dict:
    """The STATIC, provenance-tagged SPATIAL tensor-tile fact bundle for ``target`` (an OPU) — no model
    run, no arc probe. Delegates to :func:`spatial_introspect.build_fact_bundle`. Same ``{target, method,
    fields, n_derived}`` shape as :func:`target_fact_bundle`, with fields tile_dim/mrf_depth/
    element_widths/dtypes/fma_latency/op_categories/accum_kind."""
    from . import spatial_introspect
    return spatial_introspect.build_fact_bundle(target)


def render_spatial_fact_bundle(target: str, bundle: dict | None = None) -> str:
    """Render :func:`spatial_fact_bundle` as an agent-facing spatial-tile brief (Markdown). Sibling of
    :func:`render_fact_bundle`."""
    from . import spatial_introspect
    return spatial_introspect.render_fact_bundle(target, bundle)


# --- SPATIAL (OPU) op-category EFFECT confirmation ----------------------------------------------------
# The OuterProductUnit has NO RoCC funct decode — its op categories are separate one-hot io_op_* ports
# (macc / mvin / shift), so there is no op ENCODING to discover: spatial_introspect._op_categories reads
# the category NAMES structurally off the port names. This pair CONFIRMS those structural names against the
# measured datapath effect on the arc model (mlc.discover.opu_roles): macc ACCUMULATES (the MRF accumulator
# cells grow monotonically across repeated identical issues), mvin OVERWRITES (idempotent load — the bank
# changes once, the magnitude does not grow), shift RELOCATES readout through the clusters_*/pipe chain (the
# accumulator aggregate magnitude is conserved, no new products). A "shift" that grows magnitude, or a
# "macc" that only overwrites, is a real alarm (mislabel or RTL drift). HONEST RESIDUE: this CONFIRMS the
# port names by effect, it does not DISCOVER categories (there is no encoding); the mvin bcast/col sub-
# variants are only partially separable (by which cells the overwrite lands in) — their naming leans on the
# port name.

# category -> the measured-effect predicate that CONFIRMS its structural port name.
_OPU_CATEGORY_CONFIRM = {
    "macc": lambda f: bool(f.get("writes_accumulator")) and bool(f.get("accumulates")),
    "mvin": lambda f: bool(f.get("writes_accumulator")) and not bool(f.get("accumulates")),
    "shift": lambda f: bool(f.get("conserves_magnitude")) and not bool(f.get("writes_accumulator")),
}


def spatial_effect_roles(target: str) -> dict:
    """DERIVED op-category -> measured-effect signature for a SPATIAL tile (OPU), or EMPTY when it cannot be
    grounded. The spatial analogue of :func:`fine_roles`: reads the ``discovered_opu_roles.json`` cache
    beside the OPU state manifest, regenerating on demand ONLY where the probe is possible (a live arc
    model, :func:`arc_available`) and degrading honestly (empty ``{derived: False, reason}``) for a no-arc
    target or a failed probe. Returns ``{features: {category: {writes_accumulator, accumulates,
    magnitude_grows, changed_cells, conserves_magnitude, ...}}, source, derived, reason}``. Wrapped under
    :func:`_mlc_cwd` (the probe resolves its arc artifacts by paths relative to the mlc root)."""
    if mlc_dir() is None:
        return {"features": {}, "source": None, "derived": False, "reason": "MERLIN_MLC_DIR unset"}
    paths = opu_artifact_paths(target)
    if not paths or paths.get("man") is None:
        return {"features": {}, "source": None, "derived": False,
                "reason": f"no OPU arc artifacts for {target!r} (not in the mlc OPU fingerprint map)"}
    cache = Path(paths["man"]).parent / "discovered_opu_roles.json"
    if not cache.is_file():
        # Regenerate on demand ONLY where the probe is possible (a live arc model). A no-arc target keeps
        # the honest-empty path below — never auto-regen where impossible, never crash the bundle.
        if _arc_model_present(target):
            try:
                with _mlc_cwd():
                    from mlc.discover.opu_roles import derive_and_cache_opu_roles
                    derive_and_cache_opu_roles(target)
            except Exception as e:  # noqa: BLE001 — a failed probe is honest-unavailable, never a crash/guess
                return {"features": {}, "source": None, "derived": False,
                        "reason": f"OPU effect probe failed for {target!r}: {type(e).__name__}: {e}"}
    if not cache.is_file():
        return {"features": {}, "source": None, "derived": False,
                "reason": f"no discovered_opu_roles cache for {target!r} — run derive_and_cache_opu_roles "
                          "in the mlc venv"}
    try:
        import json as _json
        data = _json.loads(cache.read_text())
        if not data.get("available"):
            return {"features": {}, "source": data.get("method"), "derived": False,
                    "reason": data.get("reason") or f"OPU effect probe unavailable for {target!r}"}
        feats = data.get("features") or {}
        return {"features": feats, "source": data.get("method", "probe:opu-effect-confirm(mlc-arc)"),
                "derived": True, "reason": f"{len(feats)} op-category effect signatures from the mlc OPU probe"}
    except Exception as e:  # noqa: BLE001 — an unreadable cache is honest-unavailable, never a guess
        return {"features": {}, "source": None, "derived": False,
                "reason": f"unreadable opu roles cache: {e}"}


def crosscheck_op_categories(target: str) -> list[str]:
    """ALWAYS-ON cross-check gate for a SPATIAL tile: VERIFY each STRUCTURALLY-declared op category
    (:func:`spatial_introspect._op_categories`, read off the ``io_op_<cat>_`` port names) against the
    behaviourally MEASURED effect (:func:`spatial_effect_roles`). For every category the effect probe
    measured, assert its measured signature CONFIRMS its structural name via :data:`_OPU_CATEGORY_CONFIRM`
    (macc accumulates, mvin overwrites idempotently, shift conserves accumulator magnitude). Returns the
    list of DISAGREEMENTS (empty = every structural category name reproduced by RTL behaviour). A
    structurally-declared category the probe never measured, or one whose measured effect contradicts its
    name, is a real alarm. When nothing was behaviourally derived (no arc / mlc absent / not an OPU) the
    list is empty — nothing to cross-check — mirroring :func:`crosscheck_semantic_class_fine`."""
    er = spatial_effect_roles(target)
    feats = er.get("features") or {}
    if not feats:
        return []  # nothing behaviourally derived (no arc / mlc absent / not an OPU) -> nothing to check
    paths = opu_artifact_paths(target)
    if not paths or paths.get("man") is None:
        return []
    from . import spatial_introspect
    _module, _states, names = spatial_introspect._load_state_names(Path(paths["man"]))
    structural = spatial_introspect._op_categories(names)
    disagreements: list[str] = []
    for cat in structural:
        f = feats.get(cat)
        if f is None:
            disagreements.append(f"op category {cat!r} declared structurally (io_op_{cat}_ ports) but the "
                                 "effect probe measured no signature for it")
            continue
        confirm = _OPU_CATEGORY_CONFIRM.get(cat)
        if confirm is None:
            continue  # no confirmation predicate for this category name -> nothing to assert
        if not confirm(f):
            disagreements.append(
                f"op category {cat!r}: measured effect does not confirm its name "
                f"(writes_accumulator={f.get('writes_accumulator')}, accumulates={f.get('accumulates')}, "
                f"conserves_magnitude={f.get('conserves_magnitude')})")
    return disagreements


# mlc arc targets merlin has no committed capability manifest for yet (their facts come from the arc
# artifacts, not an in-tree contract): map target id -> compute-unit kind so fact_bundle_for can route
# them by KIND without a per-target branch in the dispatch itself. Additive; never consulted for a
# target that has a manifest (that wins).
_ARC_TARGET_KINDS: dict[str, str] = {
    "saturn_opu_mxv256d128": "spatial",
    "saturn_opu_v128d64": "spatial",
}


def _resolve_kind(target: str) -> str | None:
    """The compute-unit ``kind`` for ``target``: its capability manifest's kind when one is committed/
    regenerated, else the known arc-target map, else None (caller defaults to the systolic static path).
    Never fabricates a kind."""
    try:
        from ..target_experiment import load_capability_manifest
        return load_capability_manifest(target).kind
    except Exception:  # noqa: BLE001 — no manifest yet (fresh/arc-only target) ⇒ fall through
        return _ARC_TARGET_KINDS.get(target)


def fact_bundle_for(target: str) -> dict:
    """KIND-routed RTL fact bundle dispatch — the seam that keeps fact extraction target-name-free.

    Resolves ``target``'s compute-unit kind and routes to the fact-extraction family named by that kind's
    :class:`~merlin.targetgen.families.FamilyProfile.fact_extractor`:

      * ``circt_static`` (systolic/vector/scalar, and the default when no kind resolves) -> the existing
        CIRCT HW-dialect static bundle :func:`target_fact_bundle` (BYTE-IDENTICAL to the pre-dispatch
        path for gemmini and every current caller);
      * ``simt_config`` (simt) -> the Muon config+FIRRTL introspect, adapted to the bundle shape;
      * ``opu`` (spatial) -> the OuterProductUnit state-manifest introspect
        :func:`spatial_fact_bundle`.

    All branches return the uniform ``{target, method, fields, n_derived}`` shape, so consumers
    (onboarding, prompt rendering) read ``n_derived`` uniformly. A kind that does not resolve degrades to
    the systolic static path — exactly the pre-existing behavior.

    MEMOIZED in-memory per target: the extraction (circt-opt on the target's ``hw.mlir`` / mlc discovery)
    is deterministic for a given target within a process but costs ~seconds, so it ran redundantly on
    every call (dominating test time — re-discovery per test). It now runs ONCE per target and later
    calls return a deep COPY (mutation-safe: no caller can poison the cache). The cache lives only in
    process memory — nothing is written to disk, so nothing is committed. Call
    :func:`clear_fact_bundle_cache` if a process regenerates a target's RTL facts and must re-read."""
    if target not in _FACT_BUNDLE_CACHE:
        _FACT_BUNDLE_CACHE[target] = _fact_bundle_uncached(target)
    return copy.deepcopy(_FACT_BUNDLE_CACHE[target])


#: In-memory (process-lifetime) cache for :func:`fact_bundle_for`. Never persisted — purely a
#: within-run memo so repeated extraction (tests, prompt rendering) doesn't re-run circt/mlc.
_FACT_BUNDLE_CACHE: dict[str, dict] = {}


def clear_fact_bundle_cache() -> None:
    """Drop the in-memory fact-bundle memo (use after regenerating a target's RTL facts in-process)."""
    _FACT_BUNDLE_CACHE.clear()


def _fact_bundle_uncached(target: str) -> dict:
    """The un-memoized extraction (see :func:`fact_bundle_for`)."""
    from ..families import family_profile, known_kinds
    kind = _resolve_kind(target)
    extractor = family_profile(kind).fact_extractor if kind in known_kinds() else "circt_static"
    if extractor == "opu":
        return spatial_fact_bundle(target)
    if extractor == "simt_config":
        return _simt_fact_bundle(target)
    return target_fact_bundle(target)


def render_fact_bundle_for(target: str, bundle: dict | None = None) -> str:
    """KIND-routed sibling of :func:`fact_bundle_for` for RENDERING — the render half of the same seam.

    Routes ``target``'s kind to the matching bundle renderer: ``opu`` (spatial) ->
    :func:`render_spatial_fact_bundle`; everything else (systolic/vector/scalar ``circt_static``, and the
    default when no kind resolves) -> :func:`render_fact_bundle`. SIMT has no dedicated renderer today, so
    it degrades honestly through the systolic renderer (TODO: a muon brief renderer). BYTE-IDENTICAL to
    :func:`render_fact_bundle` for gemmini and every current ``circt_static`` caller — the same reasoning
    as ``fact_bundle_for``: gemmini resolves ``kind='systolic'`` -> ``fact_extractor='circt_static'`` ->
    the ``return render_fact_bundle(...)`` fall-through, on the same ``bundle`` object."""
    from ..families import family_profile, known_kinds
    kind = _resolve_kind(target)
    extractor = family_profile(kind).fact_extractor if kind in known_kinds() else "circt_static"
    if extractor == "opu":
        return render_spatial_fact_bundle(target, bundle)
    return render_fact_bundle(target, bundle)


#: Registry of SIMT RTL introspects, keyed by the introspect's DECLARED identity (its ``TARGET``) — the
#: fact-extraction analog of the runtime backend registry and the capsule_runner sim-oracle registry. A
#: SIMT introspect must expose ``TARGET`` + ``build_facts()``. The reference Muon introspect registers
#: itself (below); a SECOND SIMT core registers its own introspect via :func:`register_simt_introspect`
#: (in-tree, or from its out-of-tree package at import) so :func:`_simt_fact_bundle` resolves it WITHOUT
#: editing this dispatch — the seam that keeps SIMT fact extraction from bottoming out at one core.
_SIMT_INTROSPECTS: dict = {}


def register_simt_introspect(module) -> None:
    """Register a SIMT RTL introspect (must expose ``TARGET`` + ``build_facts()``). Idempotent."""
    _SIMT_INTROSPECTS[module.TARGET] = module


def _resolve_simt_introspect(target: str):
    """The registered SIMT introspect whose declared identity matches ``target``'s arc alias, or None.
    Serves by the ARC-target alias (not the merlin name): a composite SIMT target (e.g. radiance) whose
    RTL IS the introspect's config (RadianceCluster) resolves to that introspect via ``_arc_target``."""
    if not _SIMT_INTROSPECTS:
        # The reference Muon introspect was evicted to its own package (merlin/targets/muon/backend/);
        # resolve it via the registry so mlc_bridge no longer imports it directly, while preserving the
        # on-demand registration guarantee (this fires whenever the registry is empty, independent of
        # discovery order — so SIMT introspection radiance relies on cannot silently go missing).
        try:
            from ...runtime.backends import base as _bk
            register_simt_introspect(_bk.get_backend("muon").muon_introspect)
        except Exception:  # noqa: BLE001 — no SIMT introspect available -> _simt_fact_bundle reports honestly
            pass
    return _SIMT_INTROSPECTS.get(_arc_target(target))


def _simt_fact_bundle(target: str) -> dict:
    """Adapt the SIMT RTL introspect to the uniform fact-bundle shape. The introspect is resolved from
    the registry by the target's declared identity, so a SIMT target no registered introspect serves is
    reported honestly rather than mis-attributed to another core's geometry. Keys on that declared
    identity, never a literal target name."""
    intro = _resolve_simt_introspect(target)
    if intro is None:
        served = sorted(_SIMT_INTROSPECTS)
        fields = {"simt": {"value": None, "derived": False, "source": None,
                           "evidence": f"no SIMT RTL introspect serves {target!r} "
                                       f"(registered: {served})"}}
        return {"target": target,
                "method": f"SIMT RTL introspect (registered: {served})",
                "kind": "simt", "fields": fields, "n_derived": 0}
    facts = intro.build_facts()
    f = facts.get("facts", {})
    present = facts.get("inputs", {}).get("rtl_present", False)
    fields = {name: {"value": f.get(name), "derived": name in f,
                     "source": facts.get("generator", {}).get("name"),
                     "evidence": (f.get(name) or {}).get("evidence")}
              for name in ("simt", "registers", "shared_memory", "fp_datapath", "isa")}
    return {"target": target,
            "method": facts.get("generator", {}).get("method", "muon RTL introspect"),
            "kind": "simt", "rtl_present": present, "fields": fields,
            "n_derived": sum(1 for v in fields.values() if v["derived"])}


def simt_facts(target: str) -> dict:
    """SIMT self-hosted-ISA facts adapted to the ``facts.json`` body shape, so the generic manifest
    deriver (:func:`merlin.targetgen.capability_manifests.derive_manifest`) grounds ``endpoint_kind``
    from them like any other RTL facts. Delegates to the SIMT RTL introspect (``muon_introspect``, via
    the arc-target alias); the core's OWN instruction encoding is surfaced as a ``self_hosted_isa``
    interface (its ``encoding_bits`` + ``instruction_classes``) — the SIMT analog of a decode table.
    Returns ``{}`` when no SIMT introspect serves the target (honest — the deriver then falls back to the
    family default, never a fabricated endpoint)."""
    bundle = _simt_fact_bundle(target)
    fields = bundle.get("fields") or {}
    isa = (fields.get("isa") or {}).get("value") or {}
    if not isa.get("encoding_bits"):
        return {}
    itf = {"name": "self_hosted_isa",
           "encoding_bits": isa.get("encoding_bits"),
           "instruction_classes": list(isa.get("instruction_classes") or []),
           "source": bundle.get("method")}
    body: dict = {"interfaces": [itf], "source": bundle.get("method"), "target": target}
    # SIMT execution geometry (lanes/warp, warps, cores) + shared-memory CAPACITY, DERIVED by the introspect
    # (config + RTL) — surfaced so the manifest deriver grounds capabilities.simt / the SMEM capacity instead
    # of a residual literal. No memory-map BASE here (not recoverable from the op graph — stays residue).
    simt = (fields.get("simt") or {}).get("value") or {}
    if isinstance(simt.get("lanes_per_warp"), int):
        body["simt"] = {k: simt[k] for k in ("lanes_per_warp", "warps_per_core", "cores")
                        if isinstance(simt.get(k), int)}
    smem = (fields.get("shared_memory") or {}).get("value") or {}
    if isinstance(smem.get("bytes_per_cluster"), int):
        body["memories"] = [{"name": "shared_memory", "bytes": smem["bytes_per_cluster"]}]
    return {"schema_version": "simt-facts/v0", "facts": body}


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
def _mlc_importable(d=None):
    """Make ``mlc`` importable for the duration of THIS call ONLY, by inserting its dir on ``sys.path``
    iff ``mlc`` is not already importable (restored on exit). Deliberately NOT a global/module-level
    insert: a permanent insert flips :func:`mlc_available` True process-wide and un-skips the heavy
    mlc-gated tests (a known regression). This lets a machine with mlc as an external sibling checkout
    (resolved via ``MERLIN_MLC_DIR``, not pip-installed) still import mlc's pure-Python discovery — the
    same context-managed pattern as ``program_oracle._mlc_importable``.

    Restoring ``sys.path`` alone is NOT enough: the modules imported inside the block stay cached in
    ``sys.modules``, so ``import mlc`` (and thus :func:`mlc_available`) would keep succeeding process-wide
    afterward, re-introducing the exact un-skip regression this context manager exists to avoid. So when
    WE made mlc importable, we also drop the ``mlc`` modules loaded during the block on exit, leaving the
    interpreter in its pre-call state (a truly scoped import)."""
    import importlib.util
    import sys
    d = d if d is not None else mlc_dir()
    added = None
    before: frozenset[str] = frozenset()
    if d is not None and importlib.util.find_spec("mlc") is None:
        added = str(d)
        sys.path.insert(0, added)
        before = frozenset(sys.modules)
    try:
        yield
    finally:
        if added is not None:
            try:
                sys.path.remove(added)
            except ValueError:
                pass
            # Drop mlc modules loaded during the block so ``import mlc`` does not keep resolving from the
            # module cache after the path entry is gone (which would flip mlc_available() True process-wide).
            for name in [n for n in sys.modules if n == "mlc" or n.startswith("mlc.")]:
                if name not in before:
                    del sys.modules[name]


@contextmanager
def _mlc_cwd():
    """mlc resolves its ``runs/...`` arc artifacts by paths RELATIVE to its own root, so its cosim +
    discovery entry points run with CWD = the mlc dir. Also CONTEXT-INSERT the mlc dir on ``sys.path`` so
    ``import mlc`` resolves even when the process's ``sys.path`` no longer carries the cwd (``''``) entry —
    mlc is not pip-installed, so without this the import relied on cwd-relative resolution and broke inside
    a process that rewrote ``sys.path`` (e.g. the capsule-bench driver's xdsl setup), silently failing the
    arc oracle preflight. The insert is context-managed (removed on exit), NOT global: a permanent insert
    flips ``mlc_available()`` process-wide and un-skips heavy tests."""
    import sys
    d = mlc_dir()
    prev = os.getcwd()
    ds = str(d) if d is not None else None
    inserted = False
    if d is not None:
        os.chdir(d)
        if ds not in sys.path:
            sys.path.insert(0, ds)
            inserted = True
    try:
        yield
    finally:
        os.chdir(prev)
        if inserted:
            try:
                sys.path.remove(ds)
            except ValueError:
                pass


def arc_available(target: str) -> bool:
    """True iff mlc has a prebuilt arc model (.so/.o) + state manifest for ``target`` (any target)."""
    if mlc_dir() is None:
        return False
    try:
        with _mlc_cwd():
            from mlc.runtime.backend import available
            return bool(available(_arc_target(target)))
    except Exception:  # noqa: BLE001
        return False


def _arc_model_present(target: str) -> bool:
    """Un-aliased structural check: does mlc register an arc model under THIS EXACT target name? Structural
    consumers (the RoCC role / mesh probes) gate on this, NOT the oracle-aliased :func:`arc_available` — a
    composite SIMT target (radiance) must not borrow the embedding cluster's arc for structural role
    derivation (that would fabricate systolic roles it does not have); its oracle still reaches muon via
    the alias, but its structural profile stays honestly empty."""
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
        raise RuntimeError(f"mlc arc model absent for target {target!r} (runs/circt-arc/{_arc_target(target)})")
    at = _arc_target(target)
    with _mlc_cwd():
        _ensure_interface_cache(at)
        from mlc.backends.cosim_core import CosimCore
        from mlc.discover.fingerprint import artifact_paths
        p = artifact_paths(at)
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
