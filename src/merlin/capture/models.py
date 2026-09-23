"""Shared workload identity and capture discovery; no analysis dependencies."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelArch:
    """Architecture facts for one base model (reference values; loop count tagged assumed)."""

    name: str
    family: str  # "flow_matching" | "diffusion" | "autoregressive_vla" | "llm"
    #   | "feed_forward" (one pass per input, no host-side loop)
    loop_kind: str  # "denoise_steps" | "action_token_decode" | "token_decode"
    #   | "single_pass"
    loop_count: int  # K: host-side repetitions the single-pass capture hides
    control_rate_hz: float | None  # real-time control budget (VLA action heads), else None
    action_horizon: int | None  # H: actions per chunk, else None
    loop_count_source: str = "assumed"  # reference value; override with a real measurement
    measured_cycles: float | None = None  # FireSim cycle count, if recorded (evidence: measured)
    note: str = ""


# Reference architecture table. Loop counts are reference values (tagged assumed) drawn from the
# published model descriptions; override per-model with a measured temporal YAML when available.
MODEL_ARCH: dict[str, ModelArch] = {
    "smolvla": ModelArch(
        "smolvla",
        "flow_matching",
        "denoise_steps",
        10,
        30.0,
        50,
        note="SmolVLA flow-matching action head; K integration steps.",
    ),
    "pi05": ModelArch(
        "pi05", "flow_matching", "denoise_steps", 10, 50.0, 50, note="pi0.5 flow-matching action expert."
    ),
    "rdt": ModelArch(
        "rdt", "diffusion", "denoise_steps", 5, 30.0, 64, note="RDT-1B diffusion policy; DPM-solver few-step denoise."
    ),
    "rdt2": ModelArch("rdt2", "diffusion", "denoise_steps", 5, 30.0, 64, note="RDT-style diffusion policy."),
    "groot_n1d7": ModelArch(
        "groot_n1d7", "diffusion", "denoise_steps", 4, 30.0, 16, note="GR00T N1.5 diffusion action head."
    ),
    "xr0": ModelArch(
        "xr0",
        "diffusion",
        "denoise_steps",
        5,
        None,
        None,
        measured_cycles=146.2e9,
        loop_count_source="assumed",
        note="DiT timestep model; num_steps=5 in source (was 10 — P19 config-drift fix); "
        "FireSim fp32 measured 146.2 G cycles.",
    ),
    "openvla": ModelArch(
        "openvla",
        "autoregressive_vla",
        "action_token_decode",
        7,
        5.0,
        7,
        note="OpenVLA decodes a 7-DoF action as 7 autoregressive tokens.",
    ),
    "molmoact": ModelArch(
        "molmoact",
        "autoregressive_vla",
        "action_token_decode",
        8,
        5.0,
        8,
        note="MolmoAct action reasoning; autoregressive action tokens.",
    ),
    "bitvla": ModelArch("bitvla", "autoregressive_vla", "action_token_decode", 7, 5.0, 7, note="BitNet ternary VLA."),
    "openvla_oft": ModelArch("openvla_oft", "autoregressive_vla", "action_token_decode", 7, 5.0, 7),
    "small_llama": ModelArch(
        "small_llama",
        "llm",
        "token_decode",
        7,
        None,
        None,
        loop_count_source="recovered_from_ir",
        note="LLaMA-style decoder; K=7 captured decode length (IR-recovered).",
    ),
    "tiny_llama": ModelArch("tiny_llama", "llm", "token_decode", 7, None, None, loop_count_source="recovered_from_ir"),
    # Captured on disk long before it was registered here, which made every one of its capture
    # directories invisible to discover_model_captures() -- a model can be fully captured and still
    # read as absent if its base name is not a key in this table.
    "gemma2_2b": ModelArch(
        "gemma2_2b",
        "llm",
        "token_decode",
        7,
        None,
        None,
        loop_count_source="assumed",
        note="Gemma-2 2B decoder. Differs from the Llama-family entries in ways "
        "the op inventory sees: GeGLU rather than SwiGLU, RMSNorm applied "
        "both pre- and post-block in a (1+w) form, a tanh logit soft-cap, "
        "and sliding-window attention on alternate layers.",
    ),
    "small": ModelArch("small", "llm", "token_decode", 32, None, None),
    # Feed-forward vision / audio / control workloads. loop_count is 1 BY CONSTRUCTION, not as a
    # reference value: one input produces one output, so the single-pass capture hides no
    # host-side repetition (unlike the diffusion and decode families above, whose flat capture
    # hides a loop and therefore makes weight residency illegal until the loop is re-exposed).
    "resnet50": ModelArch(
        "resnet50",
        "feed_forward",
        "single_pass",
        1,
        None,
        None,
        loop_count_source="by_construction",
        note="ResNet-50 v1.5 image classifier: convolution, batch norm, residual "
        "add and a global average pool, one pass per input with no host-side "
        "loop. The m2m workload directory is `resnet50_v1_5`, so a capture "
        "of it resolves to this base by longest-prefix match; without an "
        "entry here a fully captured ResNet reads as ABSENT, which is how a "
        "declared roster model can go missing without anything saying so.",
    ),
    "spectformer": ModelArch(
        "spectformer",
        "feed_forward",
        "single_pass",
        1,
        None,
        None,
        loop_count_source="by_construction",
        note="SpectFormer-Ti classifier; blocks 0-3 spectral gating "
        "(rfft2/irfft2 on the 14x14 token grid), 4-11 attention.",
    ),
    "lstmnetvit": ModelArch(
        "lstmnetvit",
        "feed_forward",
        "single_pass",
        1,
        None,
        None,
        loop_count_source="by_construction",
        note="vitfly ViT+LSTM depth-image controller. The real controller "
        "threads the LSTM state across steps; the capture is one step "
        "from a zero state, so K=1 describes the CAPTURE, not the loop.",
    ),
    "deepjscc": ModelArch(
        "deepjscc",
        "feed_forward",
        "single_pass",
        1,
        None,
        None,
        loop_count_source="by_construction",
        note="DiffJSCC's JSCC encoder+decoder codec only — NOT the "
        "Stable-Diffusion refinement stage that produces its published "
        "reconstruction quality.",
    ),
    "whisper_tiny": ModelArch(
        "whisper_tiny",
        "llm",
        "token_decode",
        1,
        None,
        None,
        loop_count_source="by_construction",
        note="Whisper-tiny encoder + ONE cross-attending decoder step. "
        "Transcription length is data-dependent, so no reference K "
        "is claimed; the capture is the per-step graph.",
    ),
}


def _base_model(dirname: str) -> str | None:
    """Map an output capture dirname to a base model name in MODEL_ARCH (longest match)."""
    stem = dirname
    for suffix in (
        "_fp32_consistent",
        "_int8_consistent",
        "_fp8_consistent",
        "_consistent",
        "_fp32_biasfix",
        "_int8_biasfix",
        "_int8_recap",
        "_lower",
        "_phase2",
        "_rvv",
        "_host",
        "_spike",
        "_fixed",
    ):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    # match against known arch keys (longest first so 'small_llama' beats 'small')
    for key in sorted(MODEL_ARCH, key=len, reverse=True):
        if stem == key or stem.startswith(key + "_"):
            return key
    return None


def discover_model_captures() -> dict[str, list[str]]:
    """Map base model -> list of capture dirs (absolute) that have a model.mlir."""
    from merlin.common.artifacts import recaptures_dir

    out_root = recaptures_dir()  # artifacts/recaptures/ (symlinked to legacy output/ in transition)
    found: dict[str, list[str]] = {}
    if not out_root.is_dir():
        return found
    for d in sorted(out_root.iterdir()):
        if not d.is_dir() or not (d / "model.mlir").is_file():
            continue
        base = _base_model(d.name)
        if base is None:
            continue
        found.setdefault(base, []).append(str(d))
    return found
