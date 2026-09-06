"""Resolve a ``(model, variant)`` to its capture bundle — the shared input every baseline ingests.

Each baseline framework starts from the SAME captured bundle under
``artifacts/recaptures/<model>_<variant>_consistent/`` so the comparison is apples-to-apples:

  * ``model.mlir``            — linalg-on-tensors export (Buddy/TVM-via-relax can ingest directly)
  * ``weights.safetensors``   (+ ``.manifest.json``)  — HF weights + arg-index map
  * ``inputs.npz`` / ``input_order.json`` — the seeded inputs
  * ``golden.npy``            — the torch reference output for correctness gating
  * ``extra.npz``             — registered buffers + lifted constants

The original PyTorch loader (for frameworks that ingest torch, e.g. ExecuTorch export / TVM
from_pytorch) lives OUTSIDE this repo at
``$MERLIN_MODEL2MLIR/workloads/<model>/loader.py`` (default ``/path/to/model2MLIR``).
"""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

from merlin.common.artifacts import recaptures_dir
from merlin.dse_guidance import models as _models

# Per-model correctness tolerances, mirroring merlin/tests/rvv/test_vla_models_rvv.py so a baseline
# is gated exactly as our own runtime is. (min_cos, max_rel). Unlisted models use _DEFAULT_TOL.
_DEFAULT_TOL: tuple[float, float] = (0.9999, 1e-3)
TOLERANCES: dict[str, tuple[float, float]] = {
    "rdt2": (0.9999, 1e-3),
    "rdt": (0.9999, 1e-3),
    "groot_n1d7": (0.9999, 1e-2),
    "molmoact": (0.9999, 1e-3),
    "openvla": (0.9999, 1e-3),
    "openvla_oft": (0.9999, 1e-3),
    "bitvla": (0.999, 5e-3),        # BitNet ternary dequant round-off
    "xr0": (0.999, 5e-2),           # sensitive diffusion head
    "smolvla": (0.978, 5e-2),       # bf16 precision amplification in the denoise head
    "pi05": (0.9999, 1e-2),
    "small_llama": (0.9999, 1e-3),
    "tiny_llama": (0.9999, 1e-3),
}

# Precision variants a bundle can carry. The regular floats + int8 are RVV-runnable; fp6/fp4 are
# sub-byte microscaling formats that ingest + lower but have no RVV datapath (routed to gemmini-mx);
# `mixed` is a per-module mixed-precision capture (e.g. attention fp16 + MLP fp4).
_VARIANTS = ("fp32", "fp16", "bf16", "int8", "fp8", "fp6", "fp4", "mixed")

# Optional per-(model, variant) tolerance override. Low-bit variants (fp6/fp4) legitimately miss the
# tight fp32 gate; put a MEASURED floor here per cell — never a guessed loose value (an unjustified
# entry silently weakens the gate, see RANDOM_INIT_GOLDEN_UNREPRODUCIBLE). Empty until measured.
_VARIANT_TOL: dict[tuple[str, str], tuple[float, float]] = {}


def tolerance(model: str, variant: str | None = None) -> tuple[float, float]:
    """(min_cos, max_rel) gate for a model — same regime as our own RVV tests.

    A measured per-(model, variant) override wins when present (low-bit variants need a looser,
    justified floor); otherwise the per-model tolerance (then the default) applies.
    """
    if variant is not None and (model, variant) in _VARIANT_TOL:
        return _VARIANT_TOL[(model, variant)]
    return TOLERANCES.get(model, _DEFAULT_TOL)


def model2mlir_root() -> Path:
    """Root of the external model2MLIR repo (PyTorch loaders live here).

    Accepts EITHER ``MERLIN_MODEL2MLIR`` (this module's historical name) or ``MERLIN_M2M_DIR`` (the
    name the repo ``.env`` / the spike/zephyr/m2m guards actually use, via ``paths.env``). They were
    out of sync, so ExecuTorch export failed with a ``/path/to/model2MLIR`` placeholder even though
    ``.env`` pointed at the real repo. Reading both — and honoring ``.env`` through ``paths.env`` —
    closes that; direct ``os.environ`` still wins for an explicit override."""
    from ..common.paths import env as _env
    root = (os.environ.get("MERLIN_MODEL2MLIR") or os.environ.get("MERLIN_M2M_DIR")
            or _env("MERLIN_MODEL2MLIR") or _env("MERLIN_M2M_DIR") or "/path/to/model2MLIR")
    return Path(root)


@dataclass
class CaptureBundle:
    model: str
    variant: str
    root: Path

    @property
    def mlir(self) -> Path:
        return self.root / "model.mlir"

    @property
    def weights(self) -> Path:
        return self.root / "weights.safetensors"

    @property
    def weights_manifest(self) -> Path:
        return self.root / "weights.safetensors.manifest.json"

    @property
    def inputs(self) -> Path:
        return self.root / "inputs.npz"

    @property
    def input_order(self) -> Path:
        return self.root / "input_order.json"

    @property
    def golden(self) -> Path:
        return self.root / "golden.npy"

    @property
    def extra(self) -> Path:
        return self.root / "extra.npz"

    @property
    def region_goldens(self) -> Path:
        """Per-region boundary tensors (``region_goldens.npz``): for each captured nn.Module region,
        its boundary INPUT (= the upstream region's output) and OUTPUT golden. Keyed by the module
        FQN (the same ``prov.fqn`` the compare + slicer join on). The SHARED substrate for per-region
        equivalence (C7) and standalone-section IO (C8) — optional (older bundles lack it)."""
        return self.root / "region_goldens.npz"

    @property
    def rewrites(self) -> Path:
        """Record of offline rewrites applied to this bundle (``bundle.rewrites.json``).

        A rewritten bundle that does not say so produces measurements attributed to a model that no
        longer exists -- see `baselines.bundle_rewrite`, which writes this and explains why it is a
        sidecar rather than a manifest key. Absent for an unrewritten bundle."""
        return self.root / "bundle.rewrites.json"

    def has_region_goldens(self) -> bool:
        return self.region_goldens.is_file()

    def load_region_goldens(self) -> dict[str, dict[str, "object"]]:
        """Load ``region_goldens.npz`` grouped by region fqn → {slot: ndarray}. Slots are ``in<k>``
        (the region's boundary inputs) and ``out`` (its output golden). ``{}`` if the file is absent.

        Mirrors the writer's flat ``<fqn>::<slot>`` key convention (model2MLIR
        ``m2m.capture.bundle._REGION_KEY``) — kept as ONE source of truth by this docstring so the two
        repos do not drift. Consumers map fqn → region_id/role downstream (``dse_guidance``)."""
        if not self.has_region_goldens():
            return {}
        import numpy as np

        grouped: dict[str, dict[str, object]] = {}
        with np.load(self.region_goldens) as npz:
            for key in npz.files:
                fqn, _, slot = key.partition("::")
                if not slot:
                    continue
                grouped.setdefault(fqn, {})[slot] = npz[key]
        return grouped

    @property
    def torch_loader(self) -> Path:
        """PyTorch loader in the external model2MLIR repo (may not exist for every model)."""
        return model2mlir_root() / "workloads" / self.model / "loader.py"

    @property
    def tolerance(self) -> tuple[float, float]:
        return tolerance(self.model, self.variant)

    def require(self) -> "CaptureBundle":
        """Raise if the essential inputs (mlir + golden) are missing — fail-closed."""
        session_contract = self.root / "session_contract.yaml"
        if session_contract.is_file():
            from merlin.common.yaml import load_yaml

            session = load_yaml(session_contract)
            if isinstance(session, dict) and int(session.get("version", 0)) == 2:
                programs = session.get("programs", ()) or ()
                missing_programs = []
                if not isinstance(programs, list):
                    missing_programs.append("<invalid-program-list>")
                for program in programs if isinstance(programs, list) else ():
                    if not isinstance(program, dict):
                        missing_programs.append("<invalid-program-record>")
                        continue
                    child = self.root / str(program.get("bundle", ""))
                    if not (child / "model.mlir").is_file() or not (child / "golden.npy").is_file():
                        missing_programs.append(str(program.get("name", "<unnamed>")))
                if not programs or missing_programs:
                    raise FileNotFoundError(
                        f"multi-program capture bundle {self.model}/{self.variant} at {self.root} "
                        f"has missing program artifacts {missing_programs or ['<no-programs>']}")
                return self
        missing = [p.name for p in (self.mlir, self.golden) if not p.is_file()]
        if missing:
            raise FileNotFoundError(
                f"capture bundle {self.model}/{self.variant} at {self.root} missing {missing}")
        return self


def _dirname(model: str, variant: str, suffix: str = "consistent") -> str:
    return f"{model}_{variant}_{suffix}"


# K1-runnable set: models whose int8 footprint fits the 3.8 GB board (~3.4 GB usable). The three
# 7B-class VLAs (openvla, molmoact, pi05) are RAM-infeasible for whole-model on-board runs even at int8
# (fp32 embeddings dominate) — attempt+RAM-gap, never a false fit. From the full-fidelity recapture.
#: Capture-directory suffixes, in the order `resolve` prefers them. `_full` is the real/native
#: architecture and wins; `_consistent` is the older truncated capture; `_w8a8_consistent` is a
#: separate activation-quantized capture family that no baseline arm could previously see at all.
_BUNDLE_SUFFIXES: tuple[str, ...] = ("full", "consistent", "w8a8_consistent")

K1_RUNNABLE: frozenset[str] = frozenset(
    {"tiny_llama", "smolvla", "bitvla", "groot_n1d7", "rdt", "rdt2", "xr0", "small_llama"})
# resnet50_v1_5 is not here YET, and the reason has changed. The max-pool blocker is FIXED: its
# `aten.max_pool2d.default` was captured as a linalg.generic whose map (d0,d1,d2*2+d4,d3*2+d5) leaves
# d4/d5 unbound, so linalg's verifier rejected it as non-invertible IN THE READER, before any pass
# ran. The window extent is not recoverable downstream (a 114-wide padded input at stride 2 giving 56
# outputs fits both a 3- and a 4-tall window, which compute different maxima), so the repair is a
# shape-only window operand emitted at capture, as upstream linalg.pooling_* does. The FP32 bundle
# now lowers and gates clean: fp32_cos 1.0, rel 5.05e-07, argmax True.
#
# The int8 bundle's remaining blockers are also FIXED. Its W8A8 capture left
# `torchao.choose_qparams_affine` / `torchao.quantize_affine` as opaque external calls (m2m has a
# decomposition for `dequantize_affine` only), which nothing in merlin defined -- an undefined
# reference at link and an OutlineError in the interpreter; `llvmlower.torchao_affine` now decomposes
# both into linalg (bit-exact against torchao's own implementation) on both paths. And the FC weight
# (`prov.quant_inner`-tagged empties the interpreter bound from `extra.npz` while the compiled path
# left them uninitialized) is now lifted to `@forward` arguments by `llvmlower.qinner` and bound from
# the same npz by the generated argument table. MEASURED on an x86 build of the same prepared IR:
# the compiled output is BIT-IDENTICAL to the interpreter's (cos 1.0 / rel 1.38e-07 at
# int8_compute=False, cos 0.99823 / rel 0.0471 at int8_compute=True). What has not been measured is
# the board itself.
K1_RAM_INFEASIBLE: frozenset[str] = frozenset({"openvla", "molmoact", "pi05"})

# Full-fidelity capture env (the exact loader settings the recapture used to build the REAL/native
# architecture, dropping the truncation defaults). A Phase-2 arm that live-loads the torch model via
# the m2m loader MUST set these first so it ingests the IDENTICAL model the golden was computed on.
FULL_FIDELITY_ENV: dict[str, dict[str, str]] = {
    "tiny_llama": {},                                   # real TinyLlama-1.1B, full 22 layers (no truncation)
    "smolvla": {},                                      # real smolvla_base bf16, native 16 layers
    "bitvla": {"BITVLA_LLM_LAYERS": "30"},              # real BitNet depth (hidden loader-fixed at 256)
    "groot_n1d7": {},                                   # native 16 DiT layers (no M2M_GROOT_LAYERS truncation)
    "rdt": {"M2M_RDT_DEPTH": "28"},                     # real RDT-1B depth
    "rdt2": {"M2M_RDT2_DEPTH": "14"},                   # native depth
    "xr0": {"XR0_DIT_LAYERS": "16"},                    # real DiT-head depth
    "small_llama": {},                                  # synthetic toy reference (no real counterpart)
    # Declared by the bundle's own session_contract.yaml: input_source synthetic_seed_20260830,
    # synthetic_inputs true, checkpoint torchvision/resnet50/IMAGENET1K_V2, full_checkpoint true.
    # M2M_RESNET_RANDOM selects that synthetic seeded stream; weights stay PRETRAINED unless
    # M2M_RESNET_PRETRAINED=0. Without it the loader raises and the cell cannot be built at all.
    "resnet50_v1_5": {"M2M_RESNET_RANDOM": "1"},
    "openvla": {},                                      # real Llama-2-7B+ViT config (RAM-infeasible on K1)
    "molmoact": {"M2M_MOLMOACT_LAYERS": "48", "M2M_MOLMOACT_VOCAB": "152064"},  # real 7B (RAM-infeasible)
    "pi05": {},                                         # full PaliGemma+expert 3.6B (RAM-infeasible)
}


def full_env(model: str) -> dict[str, str]:
    """Loader env that reproduces the full-fidelity model the golden was captured on (may be empty)."""
    return dict(FULL_FIDELITY_ENV.get(model, {}))


def capture_config(model: str) -> dict:
    """The workload's OWN ``capture.toml`` (``{}`` when it declares none, or it cannot be read).

    A model2MLIR workload declares how it must be captured next to its loader: which interpreter
    (``venv``) and which environment (``[env]``). That declaration is the model's, not merlin's, so it
    is READ rather than mirrored -- a second copy in this repo would go stale exactly when the loader
    changed, which is the moment it matters.
    """
    path = model2mlir_root() / "workloads" / model / "capture.toml"
    if not path.is_file():
        return {}
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return {}


def capture_locations(model: str) -> dict[str, str]:
    """Per-host LOCATION env the workload's own ``capture.toml`` declares for its loader.

    Two kinds of entry live in ``[env]`` and only one of them may be replayed:

    * **Locations** -- where a weight cache or an upstream checkout sits ON THIS HOST
      (``HF_HOME = "/.../hf_cache"``). Load-bearing: without them a loader that reads a gated or
      multi-GB checkpoint cannot see the copy already on disk and falls back to a network fetch that
      fails, so the model reads as unexportable when its weights are sitting right there. They are also
      exactly the values merlin must NOT carry itself -- machine-specific absolute paths in a public
      repo -- so they are replayed from the external workload's own config.
    * **Fidelity knobs** -- a layer count, a session mode, a vocab size. In ``capture.toml`` these hold
      the *smoke* setting (2 decoder layers where the shipped bundle holds 26), so replaying one would
      build a smaller model than the golden. Dropped; the knobs that DO matter come from
      :func:`full_env`, which is curated and wins over anything here.

    The split is structural, not a per-model list: a value that resolves to an existing directory is a
    location, anything else is a knob.
    """
    env = capture_config(model).get("env")
    if not isinstance(env, dict):
        return {}
    return {str(k): v for k, v in env.items()
            if isinstance(v, str) and v and Path(v).is_dir()}


def capture_python(model: str) -> "Path | None":
    """The interpreter the workload PINS for its own capture, when it declares one that exists.

    A workload's upstream stack is pinned per model (``capture.toml``'s ``venv``), and for some models
    it is not the shared m2m venv at all -- a loader whose dependency lives only in its own environment
    raises ``ModuleNotFoundError`` under any other interpreter, which reads as "this model cannot be
    captured" when the truth is "it was captured with the wrong python". ``None`` when the workload
    pins nothing, or pins an interpreter that is not on this host, so the caller keeps its own default
    rather than running a python that does not exist.
    """
    declared = str(capture_config(model).get("venv", "") or "")
    if not declared:
        return None
    value = Path(declared)
    venv = value if value.is_absolute() else (model2mlir_root() / "workloads" / model / value)
    py = venv / "bin" / "python"
    return py if py.is_file() else None


def loader_env(model: str) -> dict[str, str]:
    """Full loader environment for a full-fidelity capture: capture locations, then curated knobs."""
    return {**capture_locations(model), **full_env(model)}


# Models whose torch loader instantiates RANDOM weights (no pretrained checkpoint) AND whose capture
# bundle ships no ``weights.safetensors``. Their captured golden came from a *different* seeded
# instantiation, so a re-exported fp32 run can never reproduce it — even when the lowering is exact.
#
# MEASURED evidence (do not add a model here without it — an unjustified entry silently WEAKENS that
# model's correctness gate):
#   bitvla/fp32  cos=0.013323 vs captured golden, but 0.9999999999987 (rel 1.6e-06) vs compute_golden
#   openvla/fp32 cos=0.009043 vs captured golden; TVM's host-VM lowering cos=1.0 on the same model
#
# Deliberately EXCLUDED despite being random-init-ish: ``rdt2`` — its recomputed golden reproduces the
# capture exactly (norm 11.4439 both), so its captured golden IS reachable and must stay the gate.
# ``rdt`` is excluded too (recompute 53.24 vs capture 52.88 — close but unresolved; measure first).
# The llama family (tiny_llama/small_llama) loads real checkpoints and passes against captured goldens.
#
# For these models the correctness gate becomes LOWERING-EXACTNESS — "the framework reproduces eager
# torch for THIS instantiation" — NOT a semantic match against a trained model. Every such cell must
# say so; see ``lowering_exactness_note``. This mirrors the TVM arm (openvla cos=1.0) and the int8
# path, which already recomputes its reference for the same reason.
RANDOM_INIT_GOLDEN_UNREPRODUCIBLE: frozenset[str] = frozenset(
    {"bitvla", "openvla", "smolvla"})


#: Models whose captured weights are RANDOM-INIT rather than a trained checkpoint, with the evidence
#: for that classification. This is a DIFFERENT question from whether the golden is reproducible, and
#: conflating the two mislabels a whole class of cell: smolvla is random-init yet perfectly
#: reproducible (a seeded re-instantiation matches its weights.safetensors bit-for-bit, 500/500
#: tensors), so a passing cosine on it measures framework-vs-eager agreement, NOT a semantic match
#: against trained weights -- while the reproducibility-keyed label called it "semantic".
#: Every entry is declared with what was actually checked; an unlisted model is treated as trained,
#: which is the status quo and is why the label still says what evidence backs it.
RANDOM_INIT_WEIGHTS: dict[str, str] = {
    "smolvla": ("the loader defaults M2M_SMOLVLA_PRETRAINED=0 and the capture was taken with it "
                "unset; the pretrained checkpoint on disk was deliberately NOT used, because "
                "capturing with it would have made the int8 cell a different model from the "
                "already-accepted fp32 cell. The PARAMETER TENSORS are reproducible (two independent "
                "processes agree on the params digest, which equals the digest of the bundle's own "
                "weights.safetensors over all 500 tensors) -- but the GOLDEN is not: measured "
                "2026-09-05, an eager fp32 re-instantiation through the loader scores cos 0.027 / "
                "rel 1.56 against golden.npy, BEFORE any quantization or export, and cos -0.031 "
                "with the zero-parameter perturbation the capture harness applies. A matching "
                "weights digest therefore does not imply a reachable golden here, which is why this "
                "model is also in RANDOM_INIT_GOLDEN_UNREPRODUCIBLE."),
    "bitvla": "random-init capture; the golden's weights are additionally unrecoverable",
    "openvla": "random-init capture; the golden's weights are additionally unrecoverable",
}


def weights_are_random_init(model: str) -> bool:
    """True if ``model``'s captured weights are random-init rather than a trained checkpoint."""
    return model in RANDOM_INIT_WEIGHTS


def random_init_evidence(model: str) -> str:
    """What was actually checked to classify ``model``'s weights. Empty when it is not classified."""
    return RANDOM_INIT_WEIGHTS.get(model, "")


#: The fp32-tier cosine floor our own gate enforces (``zephyr_model._gate`` T2).
FP32_TIER_MIN_COS = 0.99

#: How many times the model's OWN quantization noise an int8 arm may deviate before it is called
#: wrong. Same constant our gate already applies to our arm as ``quant_excess`` -- named once so the
#: reference and we are judged by the same rule rather than by two numbers that happen to differ.
QUANT_EXCESS_K = 4.0

#: Absolute int8-vs-fp32 relative bar, used where a bundle ships no W8A8 reference to derive a floor
#: from. A bar is never LOOSENED below this; the derived one only ever raises it.
ABSOLUTE_INT8_REL = 5e-2


def quantization_floor(root) -> dict:
    """How far a bundle's own W8A8 reference sits from its fp32 golden.

    This is the yardstick an int8 arm is fairly judged against, DERIVED per bundle from the two
    goldens the bundle already ships -- never a constant. It matters because a constant does the
    wrong thing at both ends:

    * ``tiny_llama_int8_consistent``'s W8A8 reference is cos 0.9487 / rel 0.958 from its fp32 golden
      and FLIPS THE ARGMAX. An implementation reproducing that reference bit-for-bit would still
      fail a tier asking cos > 0.99 with a matching argmax, so the tier decides nothing there.
    * ExecuTorch's qd8 arm on that bundle measured rel 0.106 -- nearly ten times CLOSER to fp32 than
      quantizing the model at all is -- and was refused against an absolute 0.05. Mis-specified, not
      strict.

    Missing or unreadable goldens give UNKNOWN (None), never a comfortable True.
    """
    import numpy as np

    from pathlib import Path as _P

    root = _P(root)
    out: dict = {"floor_cos": None, "floor_rel": None, "floor_argmax": None,
                 "fp32_tier_reachable": None, "note": ""}
    fp32_p, w8a8_p = root / "golden.npy", root / "golden_w8a8.npy"
    if not (fp32_p.is_file() and w8a8_p.is_file()):
        out["note"] = ("bundle ships fewer than both goldens, so its quantization floor is UNKNOWN "
                       "and an int8 arm here can only be judged against the absolute bar")
        return out
    try:
        f = np.load(fp32_p, allow_pickle=False).astype("float64").ravel()
        q = np.load(w8a8_p, allow_pickle=False).astype("float64").ravel()
    except Exception as exc:  # noqa: BLE001
        out["note"] = f"goldens unreadable ({type(exc).__name__}: {exc}); floor UNKNOWN"
        return out
    if f.size != q.size:
        # Two references of different extent are not two views of the same output, so the distance
        # between them is not this model's quantization floor. Truncating to the shorter one would
        # produce a confident number from a comparison that was never valid.
        out["note"] = (f"the fp32 golden has {f.size} elements and the W8A8 reference {q.size}: "
                       "they do not describe the same output, so the quantization floor is UNKNOWN "
                       "and an int8 arm here can only be judged against the absolute bar")
        return out
    if np.array_equal(f, q):
        # The two references are the SAME ARRAY. This is not a bundle whose quantization happens to
        # be lossless -- it is a bundle whose W8A8 golden is a copy of its fp32 one, so the pair
        # measures nothing. resnet50_v1_5_int8_w8a8_consistent ships exactly this (both files
        # sha256 3f640279a604...), and the unguarded path returned floor_rel 0.0 / floor_cos 1.0 /
        # fp32_tier_reachable True, which then derived the TIGHTEST POSSIBLE bar from the LEAST
        # informative comparison -- the same trap the all-zeros branch below exists to close. The
        # real floor for that bundle, taken against the fp32 bundle's own golden, is cos 0.999314 /
        # rel 0.029439, and the fp32 tier IS reachable; the identical-file path said so for the
        # wrong reason and would have failed a correct implementation on a fabricated bar.
        out["note"] = ("the fp32 and W8A8 goldens are byte-identical, so this pair measures no "
                       "quantization distance at all: the floor is UNKNOWN, not zero, and an int8 "
                       "arm here can only be judged against the absolute bar")
        return out
    denom = float(np.linalg.norm(q) * np.linalg.norm(f))
    if not denom:
        # One of the references is all zeros, so there is no direction to take a cosine of and no
        # scale to take a relative error against. UNKNOWN, not a floor of zero -- a floor of zero
        # would derive the TIGHTEST possible bar from the least informative reference.
        out["note"] = ("a golden is all zeros, so this bundle has no measurable quantization floor "
                       "and an int8 arm here can only be judged against the absolute bar")
        return out
    out["floor_cos"] = float(q @ f / denom)
    out["floor_rel"] = float(np.abs(q - f).max()) / max(1e-9, float(np.abs(f).max()))
    out["floor_argmax"] = bool(int(np.argmax(q)) == int(np.argmax(f)))
    out["fp32_tier_reachable"] = bool(out["floor_cos"] is not None
                                      and out["floor_cos"] > FP32_TIER_MIN_COS
                                      and out["floor_argmax"])
    if not out["fp32_tier_reachable"]:
        out["note"] = (
            f"the fp32 tier is UNREACHABLE on this bundle: its own W8A8 reference scores cos "
            f"{out['floor_cos']:.6f} / rel {out['floor_rel']:.4f} / argmax {out['floor_argmax']} "
            f"against the fp32 golden, so an implementation matching that reference bit-for-bit "
            f"would still fail a tier asking cos > {FP32_TIER_MIN_COS} and a matching argmax. Read "
            "a failure there as a property of quantizing THIS model, not as evidence about the "
            "implementation.")
    return out


def int8_accuracy_bar(root) -> dict:
    """The (min_cos, max_rel) an int8 arm on this bundle is judged by, and how it was derived.

    Derived from :func:`quantization_floor`, so the reference arm and our arm are held to the same
    RULE (deviate by at most ``QUANT_EXCESS_K`` times the model's own quantization noise) rather
    than to two absolute numbers that happen to differ. The derived bar never falls below the
    absolute one; it only rises where the model's own quantization is itself far from fp32.
    """
    floor = quantization_floor(root)
    cos_thr, rel_thr = FP32_TIER_MIN_COS, ABSOLUTE_INT8_REL
    basis = "absolute (no W8A8 reference to derive a floor from)"
    if floor.get("floor_rel") is not None:
        rel_thr = max(ABSOLUTE_INT8_REL, QUANT_EXCESS_K * floor["floor_rel"])
        # An arm cannot be required to sit CLOSER to fp32 than quantizing the model at all does.
        cos_thr = min(FP32_TIER_MIN_COS, float(floor["floor_cos"]))
        basis = (f"derived: rel = max({ABSOLUTE_INT8_REL:g}, {QUANT_EXCESS_K:g} x floor_rel "
                 f"{floor['floor_rel']:.4f}) = {rel_thr:.4f}; cos = min({FP32_TIER_MIN_COS:g}, "
                 f"floor_cos {floor['floor_cos']:.6f}) = {cos_thr:.6f}")
    return {"cos_threshold": cos_thr, "rel_threshold": rel_thr, "basis": basis, "floor": floor}


def golden_unreproducible(model: str) -> bool:
    """True if ``model``'s captured golden cannot be reproduced by re-instantiating its loader."""
    return model in RANDOM_INIT_GOLDEN_UNREPRODUCIBLE


def lowering_exactness_note(model: str) -> str:
    """The honesty label a cell MUST carry when it is gated on lowering-exactness, not semantics."""
    return (f" random-init({model}): gate=LOWERING-EXACTNESS (framework vs eager-torch on THIS seeded "
            "instantiation), NOT a semantic match — the captured golden's weights are unrecoverable "
            "(bundle ships no weights.safetensors), so cos measures lowering fidelity only.")


def resolve(model: str, variant: str = "fp32") -> CaptureBundle:
    """Locate the capture bundle for a model+variant under artifacts/recaptures/.

    Prefers the full-fidelity recapture (``<model>_<variant>_full``, real/native architecture) over the
    older truncated ``<model>_<variant>_consistent`` bundle when present. Does not require the bundle to
    exist yet (call ``.require()`` to gate) so a runner can report a clean ``gap_reason`` if it's absent.
    """
    if variant not in _VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; expected one of {_VARIANTS}")
    rd = recaptures_dir()
    # Suffix preference. `w8a8_consistent` is LAST deliberately: promoting it would silently
    # re-point lstmnetvit and tiny_llama at a DIFFERENT bundle mid-campaign (their two goldens
    # disagree at cos 0.949), so it is reached only when a model has no other bundle for the
    # variant. Verified strictly additive: every model that resolved before resolves to the same
    # directory, and `resnet50_v1_5` / `smolvla` newly resolve instead of being invisible to every
    # baseline arm.
    root = next((c for c in (rd / _dirname(model, variant, s) for s in _BUNDLE_SUFFIXES)
                 if c.is_dir()), rd / _dirname(model, variant))
    return CaptureBundle(model=model, variant=variant, root=root)


def available_models() -> list[str]:
    """Base models that actually have a captured bundle on disk (any variant)."""
    return sorted(_models.discover_model_captures().keys())


def known_models() -> list[str]:
    """All base models in the registry (whether or not captured yet)."""
    return sorted(_models.MODEL_ARCH.keys())
