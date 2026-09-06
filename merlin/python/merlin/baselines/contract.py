"""Result contract for external-baseline K1-RVV runs (the shared honesty schema).

Every per-framework runner (TVM / ExecuTorch / Buddy / EXO / ggml) produces one
:class:`BaselineResult` per (framework, model, variant). The contract enforces two invariants
that make the cross-framework comparison honest:

  * ``not_run_is_not_pass`` — :pyattr:`BaselineResult.passed` is True ONLY if the model both BUILT
    and RAN and met its correctness threshold. A model that failed to compile/run is a ``not_built``
    / ``not_run`` gap with an explicit ``gap_reason`` — never silently dropped or counted as pass.
  * scalar fallback is *labeled*, not hidden — :pyattr:`rvv_coverage_overall` +
    :pyattr:`scalar_fallbacks` carry the per-region RVV story from :mod:`.rvv_audit`, so "we pushed
    RVV but region X fell back to scalar" is recorded in the artifact, not averaged away.

The result is pure data (dataclasses) so it serializes to ``baseline_result.json`` next to the
measurement, and :mod:`.aggregate` renders the n-way matrix from a directory of them.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

# The five external baselines (IREE deferred). Matches third_party/baselines/ submodules.
FRAMEWORKS: tuple[str, ...] = ("tvm", "executorch", "buddy", "exo", "ggml")

# The canonical region taxonomy for the "kernel-style" per-region profile. Runners bracket their
# major op regions with these names so region-vs-region diffs across frameworks are trivial.
REGIONS: tuple[str, ...] = ("gemm", "attention", "norm", "elementwise", "other")


@dataclass
class RegionProfile:
    """One "kernel-style" region measurement within a whole-model run.

    The ``region_id`` / ``fqn`` / ``role`` fields carry the SHARED model-layer provenance key so a
    Merlin region and an ExecuTorch region descending from the SAME model layer align region-by-region
    (apples-to-apples), instead of collapsing into one whole-model number. ``cos`` / ``rel`` are the
    optional PER-REGION numerical-equivalence scores (vs the region's boundary golden); left None when
    a per-region golden was unavailable — reported honestly, never a silent pass (see ``region_passed``)."""
    name: str                          # one of REGIONS (free-form allowed, but prefer the taxonomy)
    rdtime_ticks: int | None = None    # raw K1 rdtime ticks bracketing the region
    cycles: int | None = None          # est core cycles (ticks * CPU_HZ/TIMEBASE_HZ); NOT cycle-accurate
    wall_ns: int | None = None
    rvv_coverage: float | None = None  # 0..1 fraction of the region's compute insns that are vector
    calls: int | None = None           # how many times the region ran (loop trip count)
    note: str = ""
    # --- shared model-layer provenance (the cross-compiler alignment / join key) ---
    region_id: str = ""                # prov.region_id (e.g. "matmul_3")
    fqn: str = ""                      # prov.fqn (deepest nn.Module path)
    role: str = ""                     # role_from_fqn(fqn): backbone_once / repeated_head / ...
    # --- per-region numerical equivalence (vs region_goldens.npz), None = not scored ---
    cos: float | None = None
    rel: float | None = None
    golden_ref: str = ""               # which region_goldens key this region was scored against

    def region_passed(self, cos_threshold: float | None, rel_threshold: float | None) -> bool | None:
        """Per-region equivalence verdict, mirroring the whole-model ``not_run_is_not_pass`` at region
        scope. None when the region was not scored (no golden) — an HONEST 'no_gold', never a pass."""
        if self.cos is None:
            return None
        if cos_threshold is not None and self.cos < cos_threshold:
            return False
        if rel_threshold is not None and self.rel is not None and self.rel > rel_threshold:
            return False
        return True


@dataclass
class ScalarFallback:
    """A region/symbol that could NOT be made RVV and fell back to scalar — recorded, not hidden."""
    symbol: str                        # emitted function / kernel symbol
    reason: str                        # why: 'no rvv microkernel' | 'unlegalizable shape' | ...
    region: str = ""                   # which REGIONS bucket, if known


@dataclass
class BaselineResult:
    """One (framework, model, variant) measurement on a substrate (default the K1 board)."""
    framework: str
    model: str
    variant: str = "fp32"              # fp32 | int8 | fp8
    substrate: str = "k1_spacemit"

    # --- lifecycle (drives not_run_is_not_pass) ---
    built: bool = False
    ran: bool = False

    # --- correctness vs golden.npy ---
    cos: float | None = None
    rel: float | None = None
    cos_threshold: float | None = None
    rel_threshold: float | None = None

    # --- whole-model (E2E) profile ---
    e2e_rdtime_ticks: int | None = None
    e2e_cycles: int | None = None
    e2e_wall_ns: int | None = None

    #: Model-LOAD time, kept separately from execute and never folded into it.
    #:
    #: This is where a framework's ahead-of-time work shows up, and dropping it makes a comparison
    #: read better than it is. ExecuTorch's XNNPACK delegate PREPACKS weights into its microkernel's
    #: blocked layout at delegate init -- i.e. during load -- while every ratio in this repo is taken
    #: against its EXECUTE line. So their prepacking is outside the number we compare, and any
    #: equivalent work we do per inference is inside ours. The runner prints both ("Model loaded in
    #: X ms" / "Model executed successfully N time(s) in Y ms") and we parsed both, but only execute
    #: was ever propagated; this field stops the load number being thrown away.
    load_ns: int | None = None

    #: Local path of a pulled ExecuTorch etdump (per-op event trace), when the run asked for one.
    etdump: "Path | None" = None

    # --- per-region "kernel-style" profile ---
    regions: list[RegionProfile] = field(default_factory=list)

    # --- RVV honesty ---
    rvv_coverage_overall: float | None = None      # 0..1 across the whole binary's compute insns
    scalar_fallbacks: list[ScalarFallback] = field(default_factory=list)

    # --- gaps & provenance ---
    gap_reason: str = ""               # MUST be non-empty when not built/ran; explains why
    framework_commit: str = ""         # submodule SHA (part of the measurement)
    toolchain: str = ""                # e.g. 'spacemit-clang-19' / 'llvm-23'
    march: str = ""                    # e.g. 'rv64gcv'
    cycle_accurate: bool = False       # K1 rdtime -> estimate (spike/FireSim remain authorities)
    board_vlenb: int | None = None
    timestamp: str = ""
    notes: str = ""

    #: The capture bundle this measurement was actually taken on -- the DIRECTORY NAME under
    #: ``recaptures/``, e.g. ``rdt2_int8_full`` vs ``rdt2_int8_consistent``. Part of the measurement,
    #: not metadata: ``bundle.resolve()`` prefers ``<model>_<variant>_full`` (the real/native
    #: architecture) over the older TRUNCATED ``_consistent`` bundle when both exist, so two runs of
    #: the "same" (model, variant) can be two different models. A comparison keyed on (model, dtype)
    #: alone cannot see that, and one shipped: a beam wall recorded on ``rdt2_int8_consistent`` was
    #: divided by an ExecuTorch reference exported at native depth, and the resulting "N x behind"
    #: figure was quoted. Empty means the producer did not record it, which is not the same as a match
    #: -- a consumer must treat empty as UNKNOWN and refuse to compare, never as "presumably the same".
    bundle_id: str = ""

    #: WHICH int8 recipe produced this cell. Three exist and they are not comparable to each other:
    #: ``weight_only`` (eager module swap -- its dequant const-folds to an fp32 const weight that
    #: XNNPACK partitions as a NORMAL FP32 GEMM, so it measures fp32 compute with int8 storage and
    #: never reaches an int8 ukernel), ``pt2e_qs8`` (static per-tensor activation quant), and
    #: ``pt2e_qd8`` (per-channel weights + DYNAMIC per-row activation quant -- XNNPACK's qd8 int8
    #: ukernels, and the mirror of merlin's own ``passes_quant_int`` datapath).
    #: Empty means the producer did not record it, which a consumer must treat as UNKNOWN and refuse
    #: to compare -- the same rule as ``bundle_id``. Every cached row predates this field, so every
    #: historical int8 ratio in this repo was taken against an unlabelled recipe.
    quant_recipe: str = ""

    #: WHICH REFERENCE this cell's ``cos``/``rel`` were scored against. Not derivable from the
    #: variant: on the int8 paths ``export_pte`` may RECOMPUTE an fp32 reference from the loaded
    #: model (``compute_golden``) instead of using the capture bundle's ``golden.npy``, and the two
    #: are different numbers with different meanings -- one is a semantic match against the captured
    #: reference, the other only says the lowering was faithful to THIS instantiation. The comparison
    #: harness previously hardcoded ``"recomputed_fp32"`` for every int8 row on the strength of a
    #: comment about the ``int8_subgraph``/``int8_whole_model`` paths; that literal is FALSE on the
    #: ``qd8`` path, which does not force a recompute, so rows scored against the captured golden
    #: were labelled as recomputed and compared with ours under a rule that did not apply.
    #: Derived from the golden path actually handed to the scorer. Empty means the producer did not
    #: record it -- UNKNOWN, and a consumer must refuse to compare rather than assume.
    accuracy_reference: str = ""

    #: The measurement PROTOCOL: how many timed inferences, after how many untimed. ExecuTorch's
    #: runner has no warmup and averages its cold first execution into ``--num_executions``, while
    #: merlin's certify path is min-of-5 after 2 warmup; on small_llama int8 ET's cold inference is
    #: 1.62x its warm one. Dividing one by the other measures the protocols, not the compilers.
    #: ``None`` means unrecorded -- again UNKNOWN, not "presumably matched".
    num_executions: int | None = None

    #: Board conditions observed around the run (governor / current_khz / max_khz / thermal), as
    #: ``{"before": {...}, "after": {...}}``. Two beam runs of a BYTE-IDENTICAL frozen seed measured
    #: 1.9915x apart with nothing in either artifact able to show why; a wall without its conditions
    #: cannot be compared against a wall measured at another time.
    board_conditions: dict | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.framework, str) or self.framework not in FRAMEWORKS:
            raise ValueError(f"unknown framework {self.framework!r}; expected one of {FRAMEWORKS}")

    @property
    def passed(self) -> bool:
        """not_run_is_not_pass: build AND run AND meet correctness thresholds, or it is not a pass."""
        if not (self.built and self.ran):
            return False
        if self.cos is None:
            return False
        if self.cos_threshold is not None and self.cos < self.cos_threshold:
            return False
        if self.rel_threshold is not None and self.rel is not None and self.rel > self.rel_threshold:
            return False
        return True

    def status(self) -> str:
        """Coarse lifecycle status for the matrix cell."""
        if not self.built:
            return "not_built"
        if not self.ran:
            return "not_run"
        if self.cos is None:
            return "no_gold"
        return "pass" if self.passed else "fail"

    def validate(self) -> "BaselineResult":
        """Enforce the honesty invariant: a gap MUST carry a reason. Returns self for chaining."""
        if not (self.built and self.ran) and not self.gap_reason:
            raise ValueError(
                f"{self.framework}/{self.model}/{self.variant}: not built/ran but gap_reason is empty "
                f"(not_run_is_not_pass requires an explicit reason)")
        return self

    def to_dict(self) -> dict:
        d = asdict(self)
        d["passed"] = self.passed
        d["status"] = self.status()
        return d

    def provenance_block(self) -> dict:
        """The hardware-provenance block this result must carry to be attributable.

        A result that claims a verdict has to say WHICH hardware produced it — a number attributed to
        the wrong device is worse than no number, because it gets cited. This board already records
        everything needed (its vector length, its ISA string, its toolchain, the framework commit);
        what was missing was emitting them in the block the provenance gate reads, so 40 passing
        results were unattributable despite the facts sitting in the same file.

        No hardware PIN is cited: the substrate here is a physical board, not an RTL revision, so
        there is no commit sha to verify against. That is recorded explicitly rather than left to look
        like an omission — ``hardware_pins`` comes back empty and ``all_pins_ok`` null, which is the
        honest state for silicon.
        """
        from merlin.common import provenance

        return provenance.record(extra={
            "substrate": self.substrate,
            "board_identity": {
                "march": self.march,
                "vlenb": self.board_vlenb,
                "conditions": self.board_conditions,
            },
            "toolchain": self.toolchain,
            "framework": self.framework,
            "framework_commit": self.framework_commit,
            "note": "physical board: identified by its measured ISA/vector-length facts and the "
                    "toolchain that built the binary, not by an RTL revision sha. No hardware pin "
                    "applies.",
        })

    def write(self, out_dir: str | Path, *, filename: str = "baseline_result.json") -> Path:
        self.validate()
        out = Path(out_dir) / filename
        payload = self.to_dict()
        payload["provenance"] = self.provenance_block()
        out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return out

    @staticmethod
    def load(path: str | Path) -> "BaselineResult":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        raw.pop("passed", None)
        raw.pop("status", None)
        # Written by `write`, not a dataclass field — dropped here so a result round-trips.
        raw.pop("provenance", None)
        regions = [RegionProfile(**r) for r in raw.pop("regions", []) or []]
        fallbacks = [ScalarFallback(**f) for f in raw.pop("scalar_fallbacks", []) or []]
        return BaselineResult(regions=regions, scalar_fallbacks=fallbacks, **raw)
