"""Registry of the real supported workloads (the model2MLIR VLA / LM zoo).

These are the workloads merlin actually brings up (see ``docs/results.md``): the ten
captured models under ``output/<model>_<dtype>_consistent/model.mlir``. The exhaustive study
runs DSE guidance over every one of them.

The central, *measured* fact this exploits (``docs/results.md``): **whole-model transformer
captures use each weight once — they emit 0 contract facts**. The single-pass capture is flat:
it hides the host-side repetition loop (flow/diffusion denoise steps, autoregressive action-
token decode, or the action-chunk horizon) over which the weights are actually reused. The
multi-rate view re-exposes that loop, and residency becomes legal — for the diffusion/flow VLAs
*and* the autoregressive ones, because both repeat the weights across their decode loop.

Grounding / honesty:
  * Aggregate structural facts (matmul count, total MACs, total weight bytes, epilogue) are read
    from the real ``model.mlir`` when it parses (``analytical`` baseline). Some quantized / newer
    captures do not parse with stock xDSL; those degrade gracefully to arch-table facts only.
  * The host-side loop kind and trip count are architecture facts, tagged ``assumed`` (reference
    values) and individually overridable; the *structural flip* (residency illegal when flat,
    legal under the loop) does not depend on the exact count.
  * The one measured latency anchor on real hardware is xr0 fp32 = 146.2 G cycles (FireSim),
    tagged ``measured``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

from merlin.common import paths
from merlin.design_pressure.ingest import mlir_m2m


@dataclass
class ModelArch:
    """Architecture facts for one base model (reference values; loop count tagged assumed)."""
    name: str
    family: str                      # "flow_matching" | "diffusion" | "autoregressive_vla" | "llm"
                                     #   | "feed_forward" (one pass per input, no host-side loop)
    loop_kind: str                   # "denoise_steps" | "action_token_decode" | "token_decode"
                                     #   | "single_pass"
    loop_count: int                  # K: host-side repetitions the single-pass capture hides
    control_rate_hz: float | None    # real-time control budget (VLA action heads), else None
    action_horizon: int | None       # H: actions per chunk, else None
    loop_count_source: str = "assumed"   # reference value; override with a real measurement
    measured_cycles: float | None = None  # FireSim cycle count, if recorded (evidence: measured)
    note: str = ""


# Reference architecture table. Loop counts are reference values (tagged assumed) drawn from the
# published model descriptions; override per-model with a measured temporal YAML when available.
MODEL_ARCH: dict[str, ModelArch] = {
    "smolvla": ModelArch("smolvla", "flow_matching", "denoise_steps", 10, 30.0, 50,
                         note="SmolVLA flow-matching action head; K integration steps."),
    "pi05": ModelArch("pi05", "flow_matching", "denoise_steps", 10, 50.0, 50,
                      note="pi0.5 flow-matching action expert."),
    "rdt": ModelArch("rdt", "diffusion", "denoise_steps", 5, 30.0, 64,
                     note="RDT-1B diffusion policy; DPM-solver few-step denoise."),
    "rdt2": ModelArch("rdt2", "diffusion", "denoise_steps", 5, 30.0, 64,
                      note="RDT-style diffusion policy."),
    "groot_n1d7": ModelArch("groot_n1d7", "diffusion", "denoise_steps", 4, 30.0, 16,
                            note="GR00T N1.5 diffusion action head."),
    "xr0": ModelArch("xr0", "diffusion", "denoise_steps", 5, None, None,
                     measured_cycles=146.2e9, loop_count_source="assumed",
                     note="DiT timestep model; num_steps=5 in source (was 10 — P19 config-drift fix); "
                          "FireSim fp32 measured 146.2 G cycles."),
    "openvla": ModelArch("openvla", "autoregressive_vla", "action_token_decode", 7, 5.0, 7,
                         note="OpenVLA decodes a 7-DoF action as 7 autoregressive tokens."),
    "molmoact": ModelArch("molmoact", "autoregressive_vla", "action_token_decode", 8, 5.0, 8,
                          note="MolmoAct action reasoning; autoregressive action tokens."),
    "bitvla": ModelArch("bitvla", "autoregressive_vla", "action_token_decode", 7, 5.0, 7,
                        note="BitNet ternary VLA."),
    "openvla_oft": ModelArch("openvla_oft", "autoregressive_vla", "action_token_decode", 7, 5.0, 7),
    "small_llama": ModelArch("small_llama", "llm", "token_decode", 7, None, None,
                             loop_count_source="recovered_from_ir",
                             note="LLaMA-style decoder; K=7 captured decode length (IR-recovered)."),
    "tiny_llama": ModelArch("tiny_llama", "llm", "token_decode", 7, None, None,
                            loop_count_source="recovered_from_ir"),
    # Captured on disk long before it was registered here, which made every one of its capture
    # directories invisible to discover_model_captures() -- a model can be fully captured and still
    # read as absent if its base name is not a key in this table.
    "gemma2_2b": ModelArch("gemma2_2b", "llm", "token_decode", 7, None, None,
                           loop_count_source="assumed",
                           note="Gemma-2 2B decoder. Differs from the Llama-family entries in ways "
                                "the op inventory sees: GeGLU rather than SwiGLU, RMSNorm applied "
                                "both pre- and post-block in a (1+w) form, a tanh logit soft-cap, "
                                "and sliding-window attention on alternate layers."),
    "small": ModelArch("small", "llm", "token_decode", 32, None, None),
    # Feed-forward vision / audio / control workloads. loop_count is 1 BY CONSTRUCTION, not as a
    # reference value: one input produces one output, so the single-pass capture hides no
    # host-side repetition (unlike the diffusion and decode families above, whose flat capture
    # hides a loop and therefore makes weight residency illegal until the loop is re-exposed).
    "spectformer": ModelArch("spectformer", "feed_forward", "single_pass", 1, None, None,
                             loop_count_source="by_construction",
                             note="SpectFormer-Ti classifier; blocks 0-3 spectral gating "
                                  "(rfft2/irfft2 on the 14x14 token grid), 4-11 attention."),
    "lstmnetvit": ModelArch("lstmnetvit", "feed_forward", "single_pass", 1, None, None,
                            loop_count_source="by_construction",
                            note="vitfly ViT+LSTM depth-image controller. The real controller "
                                 "threads the LSTM state across steps; the capture is one step "
                                 "from a zero state, so K=1 describes the CAPTURE, not the loop."),
    "deepjscc": ModelArch("deepjscc", "feed_forward", "single_pass", 1, None, None,
                          loop_count_source="by_construction",
                          note="DiffJSCC's JSCC encoder+decoder codec only — NOT the "
                               "Stable-Diffusion refinement stage that produces its published "
                               "reconstruction quality."),
    "whisper_tiny": ModelArch("whisper_tiny", "llm", "token_decode", 1, None, None,
                              loop_count_source="by_construction",
                              note="Whisper-tiny encoder + ONE cross-attending decoder step. "
                                   "Transcription length is data-dependent, so no reference K "
                                   "is claimed; the capture is the per-step graph."),
}


@dataclass
class CaptureFacts:
    """Aggregate structural facts read from a model.mlir capture (analytical)."""
    n_matmuls: int
    total_macs: int
    total_weight_bytes: int
    total_activation_bytes: int
    has_epilogue: bool
    dtype: str | None
    parsed: bool
    capture_dir: str
    note: str = ""


def _base_model(dirname: str) -> str | None:
    """Map an output capture dirname to a base model name in MODEL_ARCH (longest match)."""
    stem = dirname
    for suffix in ("_fp32_consistent", "_int8_consistent", "_fp8_consistent", "_consistent",
                   "_fp32_biasfix", "_int8_biasfix", "_int8_recap", "_lower", "_phase2",
                   "_rvv", "_host", "_spike", "_fixed"):
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


def _prefer_capture(dirs: list[str]) -> str:
    """Pick the capture most likely to parse with stock xDSL (fp32 first)."""
    for key in ("_fp32_consistent", "_fp32_biasfix", "_consistent"):
        for d in dirs:
            if d.endswith(key):
                return d
    return dirs[0]


@lru_cache(maxsize=None)
def capture_facts(capture_dir: str) -> CaptureFacts:
    """Best-effort aggregate structural facts from a capture's model.mlir (parse cached)."""
    path = f"{capture_dir}/model.mlir"
    try:
        module = mlir_m2m._parse_module(open(path, encoding="utf-8").read())
    except Exception as e:  # newer/quantized captures may not parse with stock xDSL
        return CaptureFacts(0, 0, 0, 0, False, None, parsed=False, capture_dir=capture_dir,
                            note=f"parse failed: {str(e)[:60]}")
    matmuls = [op for op in module.walk() if op.name == "linalg.matmul"]
    total_macs = 0
    weight_bytes = 0
    activation_bytes = 0
    epilogue = False
    dtype = None
    for op in matmuls:
        ls, ld = mlir_m2m._shape_dtype(op.operands[0].type)
        rs, rd = mlir_m2m._shape_dtype(op.operands[1].type)
        if ls and rs and len(ls) == 2 and len(rs) == 2:
            M, K = ls
            N = rs[-1]
            total_macs += M * K * N
            bw = 4 if (rd or "f32").endswith("32") else (2 if "16" in (rd or "") else 1)
            abw = 4 if (ld or "f32").endswith("32") else (2 if "16" in (ld or "") else 1)
            weight_bytes += K * N * bw           # weight matrix: reused across the decode loop
            activation_bytes += (M * K + M * N) * abw   # input + output activations: not reusable
            dtype = dtype or rd
        # Captures use the prov.* provenance namespace (not m2m.*); addmm == matmul+bias.
        op_kind = mlir_m2m._attr(op, "prov.op") or mlir_m2m._attr(op, "m2m.op")
        if op_kind == "addmm":
            epilogue = True
    return CaptureFacts(len(matmuls), total_macs, weight_bytes, activation_bytes, epilogue,
                        dtype, parsed=True, capture_dir=capture_dir)


def temporal_doc(arch: ModelArch) -> dict:
    """A temporal_workload_metadata doc for a model from its architecture facts."""
    K = max(int(arch.loop_count), 1)
    H = arch.action_horizon if arch.action_horizon is not None else K
    control = arch.control_rate_hz if arch.control_rate_hz is not None else 1.0
    # Two regions: a once-per-replan backbone and a K-times repeated head. The flat capture
    # collapses both into a single pass; the multi-rate view separates them so residency is
    # attributed to the head, not the backbone.
    regions = [
        {"name": "backbone", "cadence": "once_per_replan", "role": "backbone_once",
         "invocation_count": 1},
        {"name": f"{arch.loop_kind}", "cadence": "K_times_per_replan", "role": "repeated_head",
         "invocation_count": K, "loop_trip_count": K,
         "loop_invariant_state": (["weights"] if K > 1 else []),
         "loop_carried_state": (["denoise_latent"] if arch.loop_kind == "denoise_steps"
                                else ["kv_cache"])},
    ]
    return {
        "workload": arch.name,
        "class": f"{arch.family}/{arch.loop_kind}",
        "timing": {"K": K, "H": H, "control_rate_hz": control},
        "regions": regions,
    }


def baseline_doc(arch: ModelArch, facts: CaptureFacts) -> dict:
    """An *analytical* baseline_cost doc for a model (unit: cycles), from the capture aggregates.

    Deliberately NOT scaled to any measured total — scaling the analytical shape to the measured
    cycles would be circular and would hide how far off the analytical model is. The measured
    FireSim cycles are surfaced only by the calibration anchor (see :func:`calibration_rows`),
    which honestly reports the prediction-vs-measurement gap.
    """
    from merlin.dse.hardware_space import default_cost_model
    cm = default_cost_model()

    compute = math.ceil(facts.total_macs / cm.get("mac_per_cycle", 256)) if facts.total_macs else 0
    # DMA splits into weight traffic (reused across the decode loop -> reducible by residency)
    # and activation traffic (in/out per pass -> NOT reducible by weight residency).
    weight_dma = facts.total_weight_bytes / cm["dram_bytes_per_cycle"]
    act_dma = facts.total_activation_bytes / cm["dram_bytes_per_cycle"]
    dma = weight_dma + act_dma
    packing = facts.n_matmuls * cm["pack_startup_cycles"] + (
        facts.total_weight_bytes / cm["pack_bytes_per_cycle"])
    cpu_dispatch = facts.n_matmuls * cm["dispatch_fixed_cycles"]

    components = {
        "compute": float(compute),
        "dma_memory": float(dma),
        "packing": float(packing),
        "cpu_dispatch": float(cpu_dispatch),
    }
    total = sum(components.values())
    sources = {f"{k}_ms": "analytical" for k in components}

    doc: dict = {
        "workload": arch.name,
        "baseline": {
            "unit": "cycles",
            "total_ms": total,
            "components": {f"{k}_ms": v for k, v in components.items()},
        },
        "metadata_source": sources,
        # Synthetic target: the real-time control budget is not expressible in cycles without a
        # clock, so we rank against a 0.5x analytical target (clearly tagged analytical).
        "target": {"total_ms": total * 0.5} if total > 0 else {},
    }
    return doc


def predicted_total_cycles(facts: CaptureFacts) -> float:
    """The raw (un-anchored) analytical cycle prediction for a capture."""
    bd = baseline_doc(ModelArch("_", "", "", 1, None, None), facts)["baseline"]
    return float(bd.get("total_ms", 0.0))


def calibration_rows(arch: ModelArch, facts: CaptureFacts) -> list[dict]:
    """Prediction-vs-measurement rows for a model with a recorded measured cycle total.

    Returns [] when no measurement exists (no fabricated anchor). The comparison is honest:
    the analytical model is single-pass and matmul-only, so for a whole-model FireSim total it
    is expected to be wildly off — and we report that, because a known-bad anchor is more useful
    than a hidden one.
    """
    if arch.measured_cycles is None or not facts.parsed:
        return []
    predicted = predicted_total_cycles(facts)
    measured = float(arch.measured_cycles)
    err = ((predicted - measured) / measured * 100.0) if measured else None
    return [{
        "workload": arch.name,
        "quantity": "total_cycles",
        "predicted": predicted,
        "measured": measured,
        "error_pct": None if err is None else round(err, 2),
        "evidence_type": "measured",
        "interpretation": (
            "analytical single-pass matmul-only model vs whole-model FireSim total; "
            f"off by {measured / predicted:.0f}x — the analytical cost model is NOT calibrated "
            "to real cycles, so cross-workload gap_closure magnitudes are not trustworthy"
            if predicted else "analytical prediction is zero (capture parsed no matmuls)"),
    }, {
        "workload": arch.name,
        "quantity": "matmul_count",
        "predicted": facts.n_matmuls,
        "measured": "n/a",
        "error_pct": None,
        "evidence_type": "structural_bound",
        "interpretation": "extraction sanity: linalg.matmul ops read from the real capture IR",
    }]
