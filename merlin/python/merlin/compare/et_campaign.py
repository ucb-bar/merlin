"""Plan, record and summarise a multi-model ours-vs-ExecuTorch int8 campaign.

The headline evidence table for this project is one row per model: our int8 wall against
ExecuTorch's qd8 warm wall, with every guard in :mod:`merlin.compare.executorch_column` live. The
per-cell instrument is ``build_tools/scripts/k1_int8_fair_compare.py``; this module is everything
AROUND it that must not be re-invented per session:

* **Preflight** (:func:`plan_cell`) — resolve the bundle through :func:`merlin.baselines.bundle.resolve`
  (never a hardcoded path: a stale hardcoded map in another driver had all eight entries pointing at
  directories that do not exist and returned ``not_run`` for every cell instead of failing loudly),
  check the goldens the instrument will demand, and price the bundle against the board's usable RAM
  so a cell that cannot fit is refused BEFORE it costs board time rather than after.
* **The row** (:func:`campaign_row`) — a cell's outcome is either a ratio or a refusal string, never
  a bare number, and it carries what makes the number checkable: both bundles, both quantization
  recipes, both measurement protocols, both accuracy references, the board conditions, the source
  digest plus which of those sources were uncommitted, ExecuTorch's ``load_ns``, and the RVV
  coverage of the symbol that actually computed.
* **The summary** (:func:`summarize`) — the project's success criterion is beating ExecuTorch's qd8
  arm on a MAJORITY of a diverse set, so a summary that hides how many cells produced a verdict at
  all would make the criterion unfalsifiable. ``measured`` and ``refused`` are counted separately
  and the majority is computed over the cells ATTEMPTED, with the refusals named.

**A refusal is an outcome, not a failure of the campaign.** Most cells are expected to refuse today
(ExecuTorch export blockers upstream, a runner that cannot load an exported program, a model that
does not fit the board). Recording each one precisely, with its reason, is the campaign's product.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from merlin.baselines import bundle as _bundle
from merlin.common.artifacts import recaptures_dir
from merlin.compare import executorch_column as _etc

#: What OUR arm computes. A CONSTANT, never derived from what the reference happened to run — a
#: recipe derived from the comparand compares the reference against itself and the guard can never
#: fire. Mirrors ``k1_int8_fair_compare.OURS_QUANT_RECIPE``; the two are asserted equal by the tests.
OURS_QUANT_RECIPE = "merlin_int8_w8a8"

#: The tier of ours comparable IN KIND with a reference scored against fp32.
OURS_ACCURACY_REFERENCE = "capture_golden_fp32"

#: Board RAM a whole-model run may use, in bytes. DECLARED, not derived: the campaign prices cells
#: offline (the point of ``--dry-run`` is to spend no board time), and the board's own MemAvailable
#: is not readable without an SSH round trip. Every row records ``footprint.budget_source`` so a
#: reader can see this was a declaration and override it with ``--board-usable-bytes``.
#: 3.4 GB usable of the 3.8 GB board, measured while streaming weights (see the board-access notes).
DEFAULT_BOARD_USABLE_BYTES = 3_400_000_000

#: Files of a capture bundle that become RESIDENT on the board: the weight blob is embedded in the
#: binary and the lifted constants travel with it. Deliberately a LOWER BOUND — it excludes
#: activations and the runtime arena — so the fit test only ever refuses a cell that cannot fit
#: under any accounting, and never refuses a feasible one on a guessed margin.
RESIDENT_BUNDLE_FILES = ("weights.safetensors", "extra.npz")

#: Largest golden compared byte-for-byte when inheriting a provenance record across a layout-only
#: rewrite. Above it the inheritance is declined and the reference is reported UNKNOWN — a check
#: that cannot run must not report success.
_GOLDEN_DIGEST_CAP_BYTES = 64 * 1024 * 1024


# --- what a W8A8-tier pass actually decides ---------------------------------------------------
#
# Our gate's T1 tier scores int8 output against ``golden_w8a8.npy``. That is only evidence about our
# arithmetic if the reference was produced INDEPENDENTLY of our runtime. Where it was produced by
# freezing our own output, a T1 pass says the runtime agrees with itself and decides nothing. The
# distinction is invisible in the file (both are an ndarray of the right shape), so it is DECLARED —
# preferentially by the bundle itself, in a ``golden_w8a8.provenance.json`` sidecar, and otherwise by
# this registry. An unrecorded golden is UNKNOWN and reported as deciding nothing; it is never
# assumed independent.
W8A8_GOLDEN_PROVENANCE: dict[str, dict] = {
    "small_llama_int8_consistent": {
        "independent": True,
        "source": "torchao int8_dyn_act_int8_weight over the bundle's own instance",
        "evidence": ("generated 2026-08-22; the generator refuses to write unless every quantized "
                     "weight tensor matches the bundle's bit-for-bit (measured 15/15), so the "
                     "reference belongs to the weights the bundle ships and was computed by torch, "
                     "not by our runtime"),
    },
}

#: Sidecar a bundle may ship to declare its own W8A8 reference provenance (preferred over the
#: registry above, so a newly generated golden documents itself instead of needing a code edit).
W8A8_PROVENANCE_SIDECAR = "golden_w8a8.provenance.json"

_UNKNOWN_W8A8_NOTE = (
    "W8A8-tier provenance UNRECORDED for this bundle: the reference may be our own runtime's output "
    "frozen, in which case a T1 pass says the runtime agrees with itself and decides nothing about "
    "our int8 arithmetic. Read the fp32 tier (T2) as this row's correctness evidence.")


# --- expectations, held separately from gates -------------------------------------------------
#
# Known upstream blockers are recorded as ADVISORY expectations, never as skips. A blocker that has
# been fixed upstream must show up as a cell that suddenly measures — which it cannot do if the
# campaign refuses to attempt it. ``expectation_status`` compares the expectation against the
# observed outcome and flags a stale entry so this registry cannot quietly outlive its facts.
KNOWN_REFERENCE_BLOCKERS: dict[str, dict] = {
    "lstmnetvit": {
        "stage": "executorch_export",
        "reason": ("ExecuTorch qd8 export blocked upstream: ChannelsLastTaggedReshapePass rebinds "
                   "input_node after its rank-4 validation, which is qd8-specific"),
        "not_a_fallback": ("fp32 and qs8 export fine, but qs8 is static per-tensor activation quant "
                           "— different arithmetic, so it is NOT a substitute comparand"),
        "observed": "2026-09",
    },
    "gemma2_2b": {
        "stage": "pt2e_calibration",
        "reason": ("PT2E calibration corrupts dtypes through cumsum -> aten.index.Tensor; "
                   "reproduced with an EMPTY quantizer, so it is upstream, not our configuration"),
        "not_a_fallback": "",
        "observed": "2026-09",
    },
    "spectformer": {
        "stage": "executorch_runner_load",
        "reason": ("AOT export works; the ExecuTorch runner cannot LOAD the exported program — 12 "
                   "unregistered operators"),
        "not_a_fallback": "",
        "observed": "2026-09",
    },
}


def known_blocker(model: str) -> dict | None:
    """The declared upstream blocker expected for ``model``'s reference arm, or None.

    ADVISORY. It annotates a row so a refusal can be recognised as the expected one; it never
    prevents the cell from being attempted.
    """
    rec = KNOWN_REFERENCE_BLOCKERS.get(model)
    return dict(rec) if rec else None


def expectation_status(model: str, verdict_status: str) -> str:
    """How the observed outcome compares with the declared expectation for ``model``.

    ``stale_expectation`` is the one that matters: a cell we declared blocked upstream just produced
    a verdict, so the entry in :data:`KNOWN_REFERENCE_BLOCKERS` is out of date and must be removed
    rather than left to suppress attention on a cell that now works.
    """
    declared = model in KNOWN_REFERENCE_BLOCKERS
    if declared and verdict_status == "measured":
        return "stale_expectation"
    if declared:
        return "expected_refusal"
    if verdict_status == "measured":
        return "measured"
    return "unexpected_refusal"


# --- bundle footprint ---------------------------------------------------------------------------


def bundle_footprint(root: Path, *, budget_bytes: int = DEFAULT_BOARD_USABLE_BYTES,
                     budget_source: str = "declared default") -> dict:
    """Price a capture bundle against the board's usable RAM.

    The returned ``resident_lower_bound_bytes`` counts only what is certainly resident (the embedded
    weight blob and the lifted constants). It excludes activations and the runtime arena, so
    ``fits=False`` means the cell cannot fit under ANY accounting; ``fits=True`` is not a promise
    that it will, which is why the headroom is reported rather than a verdict.
    """
    parts: dict[str, int | None] = {}
    total = 0
    for name in RESIDENT_BUNDLE_FILES:
        p = root / name
        if p.is_file():
            size = p.stat().st_size
            parts[name] = size
            total += size
        else:
            parts[name] = None
    return {
        "parts_bytes": parts,
        "resident_lower_bound_bytes": total,
        "budget_bytes": int(budget_bytes),
        "budget_source": budget_source,
        "headroom_bytes": int(budget_bytes) - total,
        "fits": total <= int(budget_bytes),
        "note": ("lower bound: embedded weights + lifted constants only, excluding activations and "
                 "the runtime arena. fits=False is decisive; fits=True is necessary, not sufficient."),
    }


# --- W8A8 reference provenance ------------------------------------------------------------------


def _digest(path: Path) -> str | None:
    try:
        if path.stat().st_size > _GOLDEN_DIGEST_CAP_BYTES:
            return None
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def w8a8_reference(bundle_root: Path, *, source_bundle_id: str = "",
                   recaptures_root: Path | None = None) -> dict:
    """What a W8A8-tier (T1) pass on this bundle decides — derived where possible, declared where not.

    Order: the bundle's own sidecar, then the registry, then inheritance across a LAYOUT-ONLY
    rewrite (whose golden must be byte-identical to the source's, checked, not assumed), then
    UNKNOWN. UNKNOWN is reported as deciding nothing rather than defaulting to independent.
    """
    golden = bundle_root / "golden_w8a8.npy"
    if not golden.is_file():
        return {"status": "absent", "independent": None,
                "note": f"no golden_w8a8.npy in {bundle_root.name}; the W8A8 tier cannot be scored"}
    side = bundle_root / W8A8_PROVENANCE_SIDECAR
    if side.is_file():
        try:
            rec = json.loads(side.read_text(encoding="utf-8")) or {}
        except (OSError, ValueError):
            rec = {}
        if isinstance(rec, dict) and "independent" in rec:
            return {"status": "declared_by_bundle", "independent": bool(rec.get("independent")),
                    "source": rec.get("source", ""), "evidence": rec.get("evidence", ""),
                    "note": "" if rec.get("independent") else _UNKNOWN_W8A8_NOTE}
    rec = W8A8_GOLDEN_PROVENANCE.get(bundle_root.name)
    if rec:
        return {"status": "declared_by_registry", "independent": bool(rec.get("independent")),
                "source": rec.get("source", ""), "evidence": rec.get("evidence", ""),
                "note": "" if rec.get("independent") else _UNKNOWN_W8A8_NOTE}
    src_rec = W8A8_GOLDEN_PROVENANCE.get(source_bundle_id) if source_bundle_id else None
    if src_rec:
        root = recaptures_root or recaptures_dir()
        src_golden = root / source_bundle_id / "golden_w8a8.npy"
        d_ours, d_src = _digest(golden), _digest(src_golden)
        if d_ours is not None and d_ours == d_src:
            return {"status": "inherited_across_layout_rewrite", "independent": bool(src_rec.get("independent")),
                    "source": src_rec.get("source", ""), "evidence": src_rec.get("evidence", ""),
                    "inherited_from": source_bundle_id, "golden_sha256": d_ours,
                    "note": "" if src_rec.get("independent") else _UNKNOWN_W8A8_NOTE}
    return {"status": "unknown", "independent": None, "source": "", "evidence": "",
            "note": _UNKNOWN_W8A8_NOTE}


# --- layout-only rewrite discovery ---------------------------------------------------------------


def rewritten_siblings(source_bundle_id: str, *, recaptures_root: Path | None = None) -> list[dict]:
    """Bundles that are declared LAYOUT-ONLY derivatives of ``source_bundle_id``.

    Discovered from the artifact — every candidate's own ``bundle.rewrites.json`` is read by
    :func:`merlin.compare.executorch_column.layout_equivalence`, which is what decides equivalence —
    never from a name rule. A directory whose record names a different source, or names a rewrite
    that changes what is computed, is simply not returned.
    """
    root = recaptures_root or recaptures_dir()
    out: list[dict] = []
    if not root.is_dir():
        return out
    for cand in sorted(root.iterdir()):
        if not cand.is_dir() or cand.name == source_bundle_id:
            continue
        if not (cand / "bundle.rewrites.json").is_file():
            continue
        eq = _etc.layout_equivalence(cand.name, source_bundle_id)
        if eq is not None:
            out.append({"bundle_id": cand.name, "root": cand, "equivalence": eq})
    return out


# --- the plan -------------------------------------------------------------------------------------


@dataclass
class CellPlan:
    """Everything decided about one cell BEFORE any board time is spent."""

    model: str
    variant: str
    reference_bundle_id: str
    reference_bundle_root: Path
    ours_bundle_id: str
    ours_bundle_root: Path
    layout_equivalence: dict | None
    goldens: dict
    w8a8_reference: dict
    footprint: dict
    refusals: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def runnable(self) -> bool:
        """True when nothing known offline blocks the cell. Refusals are recorded, not raised."""
        return not self.refusals

    def as_dict(self) -> dict:
        return {
            "model": self.model, "variant": self.variant,
            "reference_bundle_id": self.reference_bundle_id,
            "reference_bundle_root": str(self.reference_bundle_root),
            "ours_bundle_id": self.ours_bundle_id,
            "ours_bundle_root": str(self.ours_bundle_root),
            "bundle_layout_equivalence": self.layout_equivalence,
            "goldens": self.goldens, "w8a8_reference": self.w8a8_reference,
            "footprint": self.footprint, "refusals": list(self.refusals),
            "notes": list(self.notes), "runnable": self.runnable,
        }


def plan_cell(model: str, *, variant: str = "int8", int8: bool = True,
              budget_bytes: int = DEFAULT_BOARD_USABLE_BYTES,
              budget_source: str = "declared default",
              prefer_rewritten: bool = False,
              recaptures_root: Path | None = None) -> CellPlan:
    """Resolve and price one cell, collecting every reason it cannot be measured.

    ``int8`` says whether OUR package is an int8 package, which is what makes ``golden_w8a8.npy``
    mandatory (the instrument raises without it — better to refuse here, for free, than after the
    build). ``prefer_rewritten`` opts into measuring ours on a declared layout-only derivative of the
    resolved bundle; the equivalence record travels onto the row so a reader can see the assumption
    the ratio rests on rather than infer it.
    """
    b = _bundle.resolve(model, variant)
    ref_root = b.root
    ref_id = ref_root.name
    ours_root, ours_id, eq = ref_root, ref_id, None
    notes: list[str] = []
    refusals: list[str] = []

    if prefer_rewritten:
        sibs = rewritten_siblings(ref_id, recaptures_root=recaptures_root)
        if len(sibs) == 1:
            ours_root, ours_id = sibs[0]["root"], sibs[0]["bundle_id"]
            eq = sibs[0]["equivalence"]
            notes.append(f"ours measured on the layout-only rewrite {ours_id!r} of {ref_id!r}; "
                         "the ratio is legitimate only while BOTH sides do their weight layout once "
                         "outside the timed window (theirs at delegate init, ours at build time)")
        elif len(sibs) > 1:
            notes.append(f"--prefer-rewritten declined: {len(sibs)} layout-only derivatives of "
                         f"{ref_id!r} ({', '.join(s['bundle_id'] for s in sibs)}); which one is "
                         "meant is not derivable, so the unrewritten bundle is used")
        else:
            notes.append(f"--prefer-rewritten had no effect: no bundle declares {ref_id!r} as the "
                         "source of a layout-only rewrite")

    if not ours_root.is_dir():
        refusals.append(f"capture bundle absent: bundle.resolve({model!r}, {variant!r}) -> "
                        f"{ours_root} does not exist. Recapture it; do not substitute another cell.")
    have_mlir = (ours_root / "model.mlir").is_file()
    have_fp32 = (ours_root / "golden.npy").is_file()
    have_w8a8 = (ours_root / "golden_w8a8.npy").is_file()
    if ours_root.is_dir():
        if not have_mlir:
            refusals.append(f"{ours_id} ships no model.mlir: nothing to lower")
        if not have_fp32:
            refusals.append(f"{ours_id} ships no golden.npy: our arm has no fp32 tier to gate "
                            "against, and an ungated wall is not a measurement")
        if int8 and not have_w8a8:
            refusals.append(
                f"{ours_id} ships no golden_w8a8.npy while our package is int8: grading W8A8 output "
                "against the weight-only golden fails cos for a CORRECT build and reads as a codegen "
                "defect. Recapture the W8A8 reference instead of loosening the gate.")

    fp = bundle_footprint(ours_root, budget_bytes=budget_bytes, budget_source=budget_source)
    if ours_root.is_dir() and not fp["fits"]:
        refusals.append(
            f"does not fit the board: {ours_id} is at least "
            f"{fp['resident_lower_bound_bytes'] / 1e9:.2f} GB resident (embedded weights + lifted "
            f"constants) against a {fp['budget_bytes'] / 1e9:.2f} GB budget ({fp['budget_source']}). "
            "Refused BEFORE the board is touched; attempting it would spend a build and a transfer "
            "to learn what the file sizes already say.")

    w8 = w8a8_reference(ours_root, source_bundle_id=ref_id if ours_id != ref_id else "",
                        recaptures_root=recaptures_root)
    if w8.get("note"):
        notes.append(w8["note"])
    notes.append(_etc.gate_basis(model))
    notes.append(_etc.dtype_comparability(variant))
    blocker = known_blocker(model)
    if blocker:
        # ADVISORY only, and worded for what will actually happen: a cell with no OFFLINE refusal is
        # still attempted (so a fix upstream shows up as a cell that suddenly measures), while a cell
        # already refused offline is not, and saying otherwise would misdescribe the run.
        tail = (" [ADVISORY: this cell is still attempted, so a fix upstream shows up as a cell "
                "that suddenly measures]" if not refusals else
                " [ADVISORY: not exercised this run — the cell is refused offline for the reasons "
                "above, so this blocker is neither confirmed nor refuted here]")
        notes.append(f"declared upstream blocker ({blocker['stage']}): {blocker['reason']}"
                     + (f" — {blocker['not_a_fallback']}" if blocker.get("not_a_fallback") else "")
                     + tail)

    return CellPlan(model=model, variant=variant, reference_bundle_id=ref_id,
                    reference_bundle_root=ref_root, ours_bundle_id=ours_id,
                    ours_bundle_root=ours_root, layout_equivalence=eq,
                    goldens={"fp32": have_fp32, "w8a8": have_w8a8, "model_mlir": have_mlir},
                    w8a8_reference=w8, footprint=fp, refusals=refusals, notes=notes)


def plan_campaign(models, **kwargs) -> list[CellPlan]:
    """:func:`plan_cell` over an ordered list of models, cheapest-first as given by the caller."""
    return [plan_cell(m, **kwargs) for m in models]


# --- the row ---------------------------------------------------------------------------------------


def _first(runs, key):
    for r in runs or ():
        v = r.get(key)
        if v:
            return v
    return None


def campaign_row(plan: CellPlan, record: dict | None = None, *, refusal: str = "",
                 command: list | None = None, elapsed_s: float | None = None,
                 arm: str = "verdict_qd8") -> dict:
    """One ledger row: a ratio or a refusal string, plus everything that makes it checkable.

    ``record`` is the JSON ``k1_int8_fair_compare.py`` wrote. ``refusal`` is set instead when the
    cell never reached the instrument (a preflight refusal, or the instrument itself failing to
    produce a record) — in which case the row still carries the plan, because a refusal that does
    not say WHICH bundle it refused is not evidence of anything.
    """
    verdict = dict((record or {}).get(arm) or {})
    if refusal:
        verdict = {"status": "refused", "reason": refusal}
    elif not verdict:
        verdict = {"status": "refused",
                   "reason": f"the instrument wrote no {arm!r} block; nothing to read a ratio from"}
    if verdict.get("status") == "not_measured":
        verdict = {"status": "refused", "reason": verdict.get("reason", "")
                   or "not_measured with no reason recorded"}
    elif verdict.get("status") == "not_comparable":
        verdict = {"status": "refused", "reason": verdict.get("reason", "")
                   or "not_comparable with no reason recorded"}

    ours = (record or {}).get("ours") or {}
    et = (record or {}).get(arm.replace("verdict_", "executorch_")) or {}
    runs = et.get("runs") or []
    gate = ours.get("gate") or {}
    rvv = ours.get("rvv") or {}

    row = {
        "schema": "et_campaign_row/v1",
        "model": plan.model,
        "variant": plan.variant,
        "ran": bool(record),
        "elapsed_s": elapsed_s,
        "command": [str(c) for c in (command or [])],
        # --- identity: which two things were measured -------------------------------------------
        "ours_bundle_id": plan.ours_bundle_id,
        "reference_bundle_id": _first(runs, "bundle_id") or "",
        "resolved_reference_bundle_id": plan.reference_bundle_id,
        "bundle_layout_equivalence": verdict.get("bundle_layout_equivalence") or plan.layout_equivalence,
        # --- what each side computed --------------------------------------------------------------
        "quant_recipe": {"ours": OURS_QUANT_RECIPE,
                         "reference": _first(runs, "quant_recipe") or "",
                         "reference_requested": et.get("recipe_requested", ""),
                         "labels": {k: v for k, v in _etc.QUANT_RECIPE_LABELS.items()}},
        # --- how each side was timed -------------------------------------------------------------
        "protocol": {
            "ours": ours.get("protocol"),
            "reference": {"method": "two-N slope: total(N) = cold + (N-1)*warm",
                          "n_lo": et.get("n_lo"), "n_hi": et.get("n_hi"),
                          "warm_ns": et.get("warm_ns"), "cold_ns": et.get("cold_ns"),
                          "cold_over_warm": et.get("cold_over_warm")},
        },
        # --- conditions the walls were taken under -----------------------------------------------
        "board_conditions": {"ours": ours.get("board_conditions"),
                             "reference": [r.get("board_conditions") for r in runs]},
        "session_drift": (record or {}).get("session_drift"),
        # --- which source bytes produced ours ------------------------------------------------------
        "source_digest": (record or {}).get("source_digest"),
        "source_dirty": (record or {}).get("source_dirty"),
        # --- what ExecuTorch did not pay for inside the timed window --------------------------------
        "executorch_load_ns": _first(runs, "load_ns"),
        # --- accuracy, with WHAT each side was scored against ----------------------------------------
        "accuracy": {
            "ours_reference": OURS_ACCURACY_REFERENCE,
            "reference_reference": _first(runs, "accuracy_reference") or "",
            "ours_fp32_cos": gate.get("fp32_cos"), "ours_fp32_rel": gate.get("fp32_rel"),
            "ours_w8a8_cos": gate.get("w8a8_cos"), "ours_w8a8_rel": gate.get("w8a8_rel"),
            "ours_tiers": gate.get("tiers"), "ours_tier_ok": gate.get("tier_ok"),
            "reference_cos": _first(runs, "cos"), "reference_rel": _first(runs, "rel"),
            "comparability": (verdict.get("accuracy") or {}).get("status") or "not_evaluated",
            "comparability_reason": (verdict.get("accuracy") or {}).get("reason", ""),
        },
        # --- did the code that produced our wall actually vectorize? ---------------------------------
        "rvv": {"compute_symbol": rvv.get("compute_symbol"),
                "compute_symbol_coverage": rvv.get("compute_symbol_coverage"),
                "coverage_overall": rvv.get("coverage_overall"),
                "error": rvv.get("error")},
        # --- what a W8A8 pass on this row decides ------------------------------------------------------
        "w8a8_reference": plan.w8a8_reference,
        "gate_basis": _etc.gate_basis(plan.model),
        "footprint": plan.footprint,
        "preflight_refusals": list(plan.refusals),
        "notes": list(plan.notes),
        # --- the outcome: a ratio or a refusal, never a bare number -------------------------------------
        "verdict": verdict,
    }
    row["expectation"] = expectation_status(plan.model, verdict.get("status", "refused"))
    row["known_blocker"] = known_blocker(plan.model)
    return row


# --- ledger ---------------------------------------------------------------------------------------


def read_ledger(path: Path) -> list[dict]:
    """Rows recorded so far. A malformed line is skipped, never guessed at."""
    if not Path(path).is_file():
        return []
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows


def append_row(path: Path, row: dict) -> None:
    """Append one row and flush, so a session killed mid-campaign keeps every completed cell."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
        f.flush()


def recorded_models(rows) -> set:
    """Models with a recorded outcome — measured OR refused. Both are outcomes; re-running a cell to
    re-derive a refusal it already recorded spends board time to learn nothing."""
    return {r.get("model") for r in rows if r.get("model")}


def completed_models(rows, *, retry_refused: bool = False) -> set:
    """Cells a resumed campaign should SKIP.

    ``retry_refused`` exists because not every refusal is about the cell. A board that went away
    mid-campaign refuses every remaining cell, and a ledger that treats those as settled outcomes
    would skip them forever on every resume — recording a board outage as a property of four models.
    Re-running only the refused cells re-tests exactly the ones whose outcome may have been about
    the session rather than the cell.
    """
    latest: dict[str, dict] = {}
    for r in rows:
        m = r.get("model")
        if m:
            latest[m] = r
    if not retry_refused:
        return set(latest)
    return {m for m, r in latest.items() if (r.get("verdict") or {}).get("status") == "measured"}


# --- the summary ------------------------------------------------------------------------------------


def summarize(rows) -> dict:
    """Per-model verdicts plus the counts the success criterion is stated in.

    The criterion is beating ExecuTorch's qd8 arm on a MAJORITY of a diverse set, so ``measured``
    (cells that produced a verdict at all) is reported beside ``wins`` and the majority is computed
    over the cells ATTEMPTED — a 1-of-1 win on a four-cell campaign is not a majority of anything and
    must not read as one.
    """
    latest: dict[str, dict] = {}
    for r in rows:
        m = r.get("model")
        if m:
            latest[m] = r
    cells = list(latest.values())
    measured = [r for r in cells if (r.get("verdict") or {}).get("status") == "measured"]
    refused = [r for r in cells if (r.get("verdict") or {}).get("status") != "measured"]
    wins = [r for r in measured if (r.get("verdict") or {}).get("beats_executorch")]
    stale = [r["model"] for r in cells if r.get("expectation") == "stale_expectation"]
    return {
        "schema": "et_campaign_summary/v1",
        "cells_attempted": len(cells),
        "verdicts_produced": len(measured),
        "refused": len(refused),
        "wins": len(wins),
        "win_models": sorted(r["model"] for r in wins),
        "refused_models": sorted(r["model"] for r in refused),
        "refusal_reasons": {r["model"]: (r.get("verdict") or {}).get("reason", "") for r in refused},
        "stale_expectations": sorted(stale),
        "majority_of_attempted": len(wins) * 2 > len(cells) if cells else False,
        "majority_of_measured": len(wins) * 2 > len(measured) if measured else False,
        "criterion": ("the project claim is a WIN on a MAJORITY of a diverse set. "
                      "majority_of_attempted is the honest reading: a majority computed over only "
                      "the cells that produced a verdict silently shrinks the set to the cells that "
                      "happened to work. Both are reported so neither can be quoted alone."),
        "per_model": {
            r["model"]: {
                "status": (r.get("verdict") or {}).get("status"),
                "ours_ns": (r.get("verdict") or {}).get("ours_ns"),
                "executorch_warm_ns": (r.get("verdict") or {}).get("executorch_warm_ns"),
                "speedup_vs_executorch": (r.get("verdict") or {}).get("speedup_vs_executorch"),
                "beats_executorch": (r.get("verdict") or {}).get("beats_executorch"),
                "reason": (r.get("verdict") or {}).get("reason", ""),
                "expectation": r.get("expectation"),
                "w8a8_reference_independent": (r.get("w8a8_reference") or {}).get("independent"),
                "ours_bundle_id": r.get("ours_bundle_id"),
                "executorch_load_ns": r.get("executorch_load_ns"),
                "source_dirty": r.get("source_dirty"),
            }
            for r in cells
        },
    }


def format_summary(summary: dict) -> str:
    """The table a human reads. Refusals are printed in full — they are the campaign's product."""
    lines = []
    lines.append("=== ours vs ExecuTorch (qd8), int8, per model ===")
    lines.append(f"{'model':<16} {'status':<10} {'ours_ms':>10} {'et_warm_ms':>11} "
                 f"{'speedup':>8} {'w8a8_ref':>10}  note")
    for model in sorted(summary.get("per_model", {})):
        c = summary["per_model"][model]
        ours = c.get("ours_ns")
        etw = c.get("executorch_warm_ns")
        sp = c.get("speedup_vs_executorch")
        ind = c.get("w8a8_reference_independent")
        ind_s = {True: "indep", False: "ours", None: "UNKNOWN"}[ind if ind in (True, False) else None]
        note = "" if c.get("status") == "measured" else (c.get("reason") or "")[:100]
        lines.append(f"{model:<16} {str(c.get('status')):<10} "
                     f"{(ours / 1e6 if ours else float('nan')):>10.3f} "
                     f"{(etw / 1e6 if etw else float('nan')):>11.3f} "
                     f"{(sp if sp else float('nan')):>8.3f} {ind_s:>10}  {note}")
    lines.append("")
    lines.append(f"cells attempted : {summary.get('cells_attempted')}")
    lines.append(f"VERDICTS produced: {summary.get('verdicts_produced')}   "
                 f"REFUSED: {summary.get('refused')}")
    lines.append(f"wins over ExecuTorch qd8: {summary.get('wins')} "
                 f"{summary.get('win_models')}")
    lines.append(f"majority of attempted: {summary.get('majority_of_attempted')}   "
                 f"majority of measured: {summary.get('majority_of_measured')}")
    lines.append(summary.get("criterion", ""))
    if summary.get("refused_models"):
        lines.append("")
        lines.append("--- refusals (each is an outcome, recorded, not a gap) ---")
        for m in summary["refused_models"]:
            lines.append(f"  {m}: {summary.get('refusal_reasons', {}).get(m, '')}")
    if summary.get("stale_expectations"):
        lines.append("")
        lines.append("--- STALE EXPECTATIONS: these were declared blocked upstream and MEASURED. "
                     "Remove their KNOWN_REFERENCE_BLOCKERS entries. ---")
        for m in summary["stale_expectations"]:
            lines.append(f"  {m}")
    return "\n".join(lines)
