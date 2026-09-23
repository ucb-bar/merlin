"""Capture module-path role classification, shared by analysis and baseline adapters."""

from __future__ import annotations

# Roles a region can be attributed to (mirrors topology / temporal roles).
ROLE_BACKBONE = "backbone_once"
ROLE_REPEATED_HEAD = "repeated_head"
ROLE_PREFIX_BUILDER = "prefix_builder"
ROLE_UNKNOWN = "unknown"


# FQN substring -> role inference (Level-1.5: roles recovered from the capture's module path).
# model2MLIR now emits prov.fqn (the deepest nn.Module path); these keywords let a downstream
# tool recover backbone vs action head WITHOUT an operator mapping. ORDER MATTERS: the
# once-per-replan backbone (vision/text encoder) is checked first, so a vision backbone's own
# transformer blocks (vision_backbone.blocks.3.attn) are not mislabeled by the generic
# block-body keywords below. Verified against the real RDT denoise-step capture (module paths
# model.blocks.N.{attn,cross_attn,ffn}, model.{t,freq}_embedder).
_FQN_ROLE_KEYWORDS: list[tuple[tuple[str, ...], str]] = [
    # 1) once-per-replan backbone (vision / multimodal encoder + the vision->LM projector that
    #    runs once per replan to build the decode prefix)
    (
        (
            "vision",
            "backbone",
            "encoder",
            "vlm",
            "patch_embed",
            "siglip",
            "vit",
            "dino",
            "image_encoder",
            "img_encoder",
            "projector",
        ),
        "backbone_once",
    ),
    # 2) prefix / KV state produced once, reused across the head
    (("kv_cache", "prefix_kv", "kv_proj"), "prefix_builder"),
    # 3) the repeated action / denoise / decode head: explicit head names, diffusion-timestep
    #    conditioning embedders, LLaMA-style decoder attention/MLP projections, and (last,
    #    generic) transformer-block bodies. Reached only after backbone/encoder is ruled out.
    (
        (
            "action_expert",
            "action_head",
            "denoise",
            "flow",
            "diffusion",
            "dit",
            "noise_pred",
            "t_embedder",
            "freq_embedder",
            "timestep",
            "time_embed",
            "decoder",
            "language_model",
            "llm",
            "lm_head",
            "self_attn",
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            # DiT / action-head epilogue projections: final-layer adaLN modulation (rdt2/DiT) and the
            # action-head output projections (groot proj_out_*) run every denoise step -> repeated head.
            "final_layer",
            "adaln",
            "modulation",
            "proj_out",
            "blocks",
            "transformer_block",
            "cross_attn",
            "ffn",
        ),
        "repeated_head",
    ),
]


def role_from_fqn(fqn: str | None) -> str | None:
    """Infer a topology role from a module FQN (``prov.fqn``); None if no keyword matches.

    Priority order (see ``_FQN_ROLE_KEYWORDS``): backbone/encoder, then prefix/KV, then the
    repeated head. The ordering prevents a backbone's own blocks from being read as head.
    """
    if not fqn:
        return None
    low = fqn.lower()
    # the bare "lm" leaf is the vocab/output projection (lm-head); it runs once per decode step.
    if low.rsplit(".", 1)[-1] == "lm":
        return ROLE_REPEATED_HEAD
    # The flow-matching VLAs (smolVLA, pi0.5) wrap the VLM backbone AND the action expert in one
    # container ("vlm_with_expert" / "paligemma_with_expert"); the container name contains the
    # backbone token "vlm", so the expert submodule would be misread as backbone. Resolve the
    # action expert + its per-step action/time/state projections FIRST (specific tokens only, never
    # the bare "expert" of the container) — they run every denoise step -> repeated head.
    if any(
        k in low
        for k in (
            "lm_expert",
            "gemma_expert",
            "action_expert",
            "action_in_proj",
            "action_out_proj",
            "action_time_mlp",
            "time_mlp",
            "state_proj",
            "action_proj",
            "action_output_layer",
            "action_output",
        )
    ):
        return ROLE_REPEATED_HEAD
    for keywords, role in _FQN_ROLE_KEYWORDS:
        if any(k in low for k in keywords):
            return role
    return None
