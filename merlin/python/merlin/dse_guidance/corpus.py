"""Leaf corpus accessors + shared writers for the dse_guidance study.

Hoisted out of ``case_study`` so the analysis modules can use them (``_csv``, ``available_models``,
``_recap_dir``) WITHOUT importing the ``case_study`` hub back. ``case_study`` is the top-level study
driver; this is a leaf (imports nothing from the analysis modules except ``attribution`` lazily inside
``available_models``, and ``attribution`` imports neither), so it breaks the driver↔analysis coupling.
"""
from __future__ import annotations

import os

from merlin.common import paths

# P23: the PRIMARY corpus is the loop-preserving captures (recaptures_loop/) — K, KV, the repeated
# region, and residency are all IR-exact, and role attribution is structural (the scf.for boundary).
# The flat single-forward captures remain on disk at recaptures/. Set MERLIN_DSE_CORPUS=flat to use them.
_CORPUS_SUBDIR = "recaptures" if os.environ.get("MERLIN_DSE_CORPUS") == "flat" else "recaptures_loop"

# Recaptured real workloads (class + reference loop count K). K is assumed/reference, not measured.
RECAP_MODELS: dict[str, dict] = {
    "rdt": {"class": "diffusion/denoise_steps", "K": 5,
            "note": "RDT denoise step (depth 2, random init)"},
    "openvla": {"class": "autoregressive_vla/action_token_decode", "K": 7,
                "note": "OpenVLA: fused ViT vision backbone + Llama decode head (small config)"},
    "small_llama": {"class": "llm/token_decode", "K": 7,
                    "note": "small Llama decoder (2 layers); flat-corpus only (loop capture is generic-only)"},
    "tiny_llama": {"class": "llm/token_decode", "K": 7,
                   "note": "tiny Llama decoder; K=7 captured decode length (IR-recovered from scf.for)"},
    # full-corpus recaptures (prov.fqn via model2MLIR; small/random configs, structure real).
    # Studyable = parses with the ingest xDSL (shared `} -> (T1,T2)` normalizer) AND has linear-layer
    # GEMMs with prov.fqn roles. xr0's linears are batched (3D/4D activation x 2D weight) and bitvla's
    # are plain 2D -- both handled by extract_matmuls' leading-dim fold (attention bmms stay uncounted,
    # uniformly with the rest of the corpus, which counts linear-layer GEMMs).
    "rdt2": {"class": "diffusion/denoise_steps", "K": 5,
             "note": "RDT2 diffusion denoise step (depth 2, random init)"},
    "groot_n1d7": {"class": "diffusion/denoise_steps", "K": 4,
                   "note": "GR00T N1.5 flow-matching action head (2 layers, random init)"},
    "molmoact": {"class": "autoregressive_vla/action_token_decode", "K": 8,
                 "note": "MolmoAct causal LM forward (4 layers, random init)"},
    "smolvla": {"class": "flow_matching/denoise_steps", "K": 10,
                "note": "SmolVLA: SmolVLM2 backbone + action expert, denoise step (2 vlm layers)"},
    "pi05": {"class": "flow_matching/denoise_steps", "K": 10,
             "note": "pi0.5: PaliGemma backbone + gemma action expert, flow-matching step"},
    "xr0": {"class": "diffusion/denoise_steps", "K": 5,
            "note": "XR-0 batched-attention DiT denoise step (2 dit layers, random init); "
                    "K=5 from source num_steps (P19 config-drift fix; was 10)"},
    "bitvla": {"class": "autoregressive_vla/action_token_decode", "K": 7,
               "note": "BitVLA: BitNet ternary LM decode (2 layers, fp32 fake-quant capture)"},
}


def _recap_dir_in(workload: str, subdir: str):
    # Small reduced-config captures are committed under merlin/benchmarks/; oversized ones
    # (pi05/smolvla/groot) live out-of-git under out/artifacts/recaptures/ (regenerable via m2m) to
    # keep git lean. Prefer the committed copy, fall back to the artifacts overflow, else the committed
    # path (absent -> available_models()/callers skip it via the model.mlir is_file() check).
    committed = paths.merlin_dir() / "benchmarks" / "dse_guidance" / subdir / workload
    if (committed / "model.mlir").is_file():
        return committed
    from merlin.common.artifacts import recaptures_dir
    overflow = recaptures_dir() / "dse_guidance" / subdir / workload
    if (overflow / "model.mlir").is_file():
        return overflow
    return committed


def _recap_dir(workload: str):
    """Resolve a capture in the ACTIVE corpus (loop-preserving unless MERLIN_DSE_CORPUS=flat)."""
    return _recap_dir_in(workload, _CORPUS_SUBDIR)


def available_models() -> list[str]:
    """Workloads with a studyable capture. On the loop corpus, a model must have >=1 linalg.matmul
    (excludes the synthetic small_llama toy, whose functional-weight loop wrapper lowered its GEMMs
    to linalg.generic -> 0 matmuls; its flat capture is unaffected)."""
    out = []
    for w in RECAP_MODELS:
        d = _recap_dir(w)
        mlir = d / "model.mlir"
        if not mlir.is_file():
            continue
        # Presence check only — a cheap regex-free substring scan, not a full xDSL parse per model
        # (this runs at test-collection time over every candidate; deep attribution is per-workload).
        if _CORPUS_SUBDIR == "recaptures_loop" and "linalg.matmul" not in mlir.read_text(errors="ignore"):
            continue
        out.append(w)
    return out


def _csv(rows: list[dict], cols: list[str]) -> str:
    import csv
    import io
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow({k: ("" if r.get(k) is None else r[k]) for k in cols})
    return buf.getvalue()
