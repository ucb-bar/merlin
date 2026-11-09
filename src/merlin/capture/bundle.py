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

import tomllib
from dataclasses import dataclass
from pathlib import Path

from merlin.capture import models as _models
from merlin.common.artifacts import recaptures_dir

# Per-model correctness tolerances, mirroring tests/rvv/test_vla_models_rvv.py so a baseline
# is gated exactly as our own runtime is. (min_cos, max_rel). Unlisted models use _DEFAULT_TOL.
_DEFAULT_TOL: tuple[float, float] = (0.9999, 1e-3)
TOLERANCES: dict[str, tuple[float, float]] = {
    "rdt2": (0.9999, 1e-3),
    "rdt": (0.9999, 1e-3),
    "groot_n1d7": (0.9999, 1e-2),
    "molmoact": (0.9999, 1e-3),
    "openvla": (0.9999, 1e-3),
    "openvla_oft": (0.9999, 1e-3),
    "bitvla": (0.999, 5e-3),  # BitNet ternary dequant round-off
    "xr0": (0.999, 5e-2),  # sensitive diffusion head
    "smolvla": (0.978, 5e-2),  # bf16 precision amplification in the denoise head
    "pi05": (0.9999, 1e-2),
    "small_llama": (0.9999, 1e-3),
    "tiny_llama": (0.9999, 1e-3),
}

# Precision variants a bundle can carry. The regular floats + int8 are RVV-runnable; fp6/fp4 are
# sub-byte microscaling formats that ingest + lower but have no RVV datapath (routed to an MX-capable accelerator);
# `mixed` is a per-module mixed-precision capture (e.g. attention fp16 + MLP fp4).
_VARIANTS = ("fp32", "fp16", "bf16", "int8", "fp8", "fp6", "fp4", "mixed")

# Optional per-(model, variant) tolerance override. Low-bit variants (fp6/fp4) legitimately miss the
# tight fp32 gate; put a MEASURED floor here per cell — never a guessed loose value (an unjustified
# entry silently weakens the gate). Empty until measured.
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
    from merlin.integrations.model2mlir import root

    return root()


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

    def load_region_goldens(self) -> dict[str, dict[str, object]]:
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

    def require(self) -> CaptureBundle:
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
                        f"has missing program artifacts {missing_programs or ['<no-programs>']}"
                    )
                return self
        missing = [p.name for p in (self.mlir, self.golden) if not p.is_file()]
        if missing:
            raise FileNotFoundError(f"capture bundle {self.model}/{self.variant} at {self.root} missing {missing}")
        return self


def _dirname(model: str, variant: str, suffix: str = "consistent") -> str:
    return f"{model}_{variant}_{suffix}"


#: Capture-directory suffixes, in the order `resolve` prefers them. `_full` is the real/native
#: architecture and wins; `_consistent` is the older truncated capture; `_w8a8_consistent` is a
#: separate activation-quantized capture family that no baseline arm could previously see at all.
_BUNDLE_SUFFIXES: tuple[str, ...] = ("full", "consistent", "w8a8_consistent")

# Full-fidelity capture env (the exact loader settings the recapture used to build the REAL/native
# architecture, dropping the truncation defaults). A Phase-2 arm that live-loads the torch model via
# the m2m loader MUST set these first so it ingests the IDENTICAL model the golden was computed on.
FULL_FIDELITY_ENV: dict[str, dict[str, str]] = {
    "tiny_llama": {},  # real TinyLlama-1.1B, full 22 layers (no truncation)
    "smolvla": {},  # real smolvla_base bf16, native 16 layers
    "bitvla": {"BITVLA_LLM_LAYERS": "30"},  # real BitNet depth (hidden loader-fixed at 256)
    "groot_n1d7": {},  # native 16 DiT layers (no M2M_GROOT_LAYERS truncation)
    "rdt": {"M2M_RDT_DEPTH": "28"},  # real RDT-1B depth
    "rdt2": {"M2M_RDT2_DEPTH": "14"},  # native depth
    "xr0": {"XR0_DIT_LAYERS": "16"},  # real DiT-head depth
    "small_llama": {},  # synthetic toy reference (no real counterpart)
    # Declared by the bundle's own session_contract.yaml: input_source synthetic_seed_20260830,
    # synthetic_inputs true, checkpoint torchvision/resnet50/IMAGENET1K_V2, full_checkpoint true.
    # M2M_RESNET_RANDOM selects that synthetic seeded stream; weights stay PRETRAINED unless
    # M2M_RESNET_PRETRAINED=0. Without it the loader raises and the cell cannot be built at all.
    "resnet50_v1_5": {"M2M_RESNET_RANDOM": "1"},
    "openvla": {},  # real Llama-2-7B+ViT config
    "molmoact": {"M2M_MOLMOACT_LAYERS": "48", "M2M_MOLMOACT_VOCAB": "152064"},  # real 7B (RAM-infeasible)
    "pi05": {},  # full PaliGemma+expert 3.6B (RAM-infeasible)
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
    return {str(k): v for k, v in env.items() if isinstance(v, str) and v and Path(v).is_dir()}


def capture_python(model: str) -> Path | None:
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
    root = next(
        (c for c in (rd / _dirname(model, variant, s) for s in _BUNDLE_SUFFIXES) if c.is_dir()),
        rd / _dirname(model, variant),
    )
    return CaptureBundle(model=model, variant=variant, root=root)


def available_models() -> list[str]:
    """Base models that actually have a captured bundle on disk (any variant)."""
    return sorted(_models.discover_model_captures().keys())


def known_models() -> list[str]:
    """All source-owned workload names, even with no captures on disk."""
    return sorted(_models.workload_names())
