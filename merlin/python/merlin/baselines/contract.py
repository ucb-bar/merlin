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

    def write(self, out_dir: str | Path, *, filename: str = "baseline_result.json") -> Path:
        self.validate()
        out = Path(out_dir) / filename
        out.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return out

    @staticmethod
    def load(path: str | Path) -> "BaselineResult":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        raw.pop("passed", None)
        raw.pop("status", None)
        regions = [RegionProfile(**r) for r in raw.pop("regions", []) or []]
        fallbacks = [ScalarFallback(**f) for f in raw.pop("scalar_fallbacks", []) or []]
        return BaselineResult(regions=regions, scalar_fallbacks=fallbacks, **raw)
