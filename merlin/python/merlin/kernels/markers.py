"""Deterministic marker table: the heart of cheap, reproducible motif extraction.

A *marker* is a regex whose presence in a kernel's text is evidence that an optimization
*decision* was made. Markers are keyed by ``(isa_family, motif)`` rather than by producer:
because every source ultimately becomes C (XNNPACK ships C; Autocomp emits C; Exo is
compiled to C), the intrinsics that appear depend on the *target ISA*, not on who wrote it.
This means Autocomp-Gemmini C and Exo-Gemmini C share one marker set, and XNNPACK-RVV and a
hypothetical Exo-RVV share another.

We deliberately match *decisions* (a packed-weight pointer advance, an accumulator address
flag, a vector-length-agnostic loop) and never *constants* (tile sizes, LMUL values).

``markers_for(target)`` resolves a kernel's ``target`` to its family and returns the compiled
regex table for that family, merged over the ``generic`` baseline.
"""
from __future__ import annotations

import re
from functools import lru_cache

# Canonical motif vocabulary (also the keys produced by classify.py).
MOTIFS = (
    "packed_rhs",
    "accumulator_lifetime",
    "epilogue_before_commit",
    "vector_length_polymorphic",
    "tiling_blocking",
    "double_buffering",
    "weight_stationary_dataflow",
    "intrinsic_lowering",
)

# Map a kernel.target string to an ISA family used for marker lookup.
_TARGET_FAMILY = {
    "rvv": "rvv",
    "riscv": "rvv",
    "gemmini": "gemmini",
    "avx2": "avx",
    "avx512": "avx",
    "x86": "avx",
    "neon": "neon",
    "aarch64": "neon",
    "arm": "neon",
    # Exo *schedules* (the .py decision record), distinct from Exo compiled to C.
    "exo_schedule": "exo_schedule",
    # Triton GPU kernels (@triton.jit).
    "triton": "triton",
    "gpu": "triton",
    "cuda": "triton",
}


def target_family(target: str) -> str:
    return _TARGET_FAMILY.get((target or "").lower(), "generic")


# Raw (uncompiled) marker patterns per (family, motif). Each value is a list of regex
# strings; a motif "fires" if ANY pattern matches. Patterns are intentionally tolerant.
_RAW: dict[str, dict[str, list[str]]] = {
    "generic": {
        # Structural loop nesting works across all C sources.
        "tiling_blocking": [r"\bfor\s*\(", r"\bdo\s*\{"],
    },
    "rvv": {
        # Packed weights: XNNPACK's pointer-advance idiom `w = (const T*) w + nr/vl/vlmax`,
        # and OpenBLAS's packed-B index advance `bi += NR` / `B[bi + n]` scalar broadcasts.
        "packed_rhs": [r"\bw\s*=\s*\(const[^)]*\)\s*w\s*\+\s*(?:nr|vl|vlmax|k\b)",
                       r"const\s+void\s*\*\s*restrict\s+w",
                       r"\bbi\s*\+=\s*\d+", r"\bB\[bi\s*\+"],
        # Accumulators: widening / fma multiply-accumulate and live vacc registers.
        "accumulator_lifetime": [r"__riscv_v[wf]?macc", r"\bvacc\d*\b"],
        # Epilogue before store: fp clamp (vfmax/vfmin) and requant narrowing converts.
        "epilogue_before_commit": [r"__riscv_vfm(?:ax|in)", r"__riscv_vf?n?cvt"],
        # Vector-length-agnostic loop: vsetvl/vsetvlmax driving the trip count.
        "vector_length_polymorphic": [r"__riscv_vsetvl(?:max)?_e\d+m\d+"],
        "intrinsic_lowering": [r"__riscv_v"],
    },
    "gemmini": {
        # Packed RHS / weight staged into scratchpad and reused across compute.
        "packed_rhs": [r"\bmvin[23]\b", r"gemmini_extended\d*_mvin[23]?", r"sp_tiled_matmul"],
        # Accumulator addressing (bit 31 / 0x40000000) + preloaded systolic compute.
        "accumulator_lifetime": [r"1u?\s*<<\s*31", r"0x40000000",
                                  r"compute_preloaded", r"gemmini_extended_compute_preloaded"],
        # Epilogue: bias loaded into the accumulator, or a non-NO_ACTIVATION config.
        "epilogue_before_commit": [r"mvin\s*\(\s*&?\s*bias",
                                    r"config_ex\([^)]*,\s*(?:RELU|GELU|IGELU)\b"],
        # Manual double buffering / DMA overlap.
        "double_buffering": [r"\balt_wbuf\b", r"\bcur_wbuf\b", r"\bdb_sel\b"],
        # Dataflow mode is explicit in config_ex.
        "weight_stationary_dataflow": [r"WEIGHT_STATIONARY", r"OUTPUT_STATIONARY"],
        "intrinsic_lowering": [r"config_(?:ex|ld|st)\b", r"\bpreload\b",
                               r"compute_preloaded", r"\bmvout\b"],
    },
    "avx": {
        "accumulator_lifetime": [r"_mm\d*_fmadd_ps", r"_mm\d*_fmadd_pd"],
        "epilogue_before_commit": [r"_mm\d*_max_ps", r"_mm\d*_min_ps"],
        "packed_rhs": [r"_mm\d*_broadcast_ss", r"B_reg"],
        "intrinsic_lowering": [r"_mm\d+_"],
    },
    "neon": {
        "accumulator_lifetime": [r"vfmaq?_f32", r"vmlaq?_f32"],
        "epilogue_before_commit": [r"vmaxq_f32", r"vminq_f32"],
        "intrinsic_lowering": [r"vld1q?_f32", r"vst1q?_f32", r"vfmaq?_f32"],
    },
    # Exo schedule .py: the optimization is named directly as a scheduling directive, so the
    # decision is *explicit* (the strongest evidence form). These markers recover the
    # schedule-level signal that compiling Exo to C discards.
    "exo_schedule": {
        "packed_rhs": [r"stage_mem\s*\([^)]*[\"'][AB]?_?[Ww]?[^\"']*[\"']",
                       r"set_memory\s*\([^)]*(GEMM_SCRATCH|SCRATCH)",
                       r"\bpack\w*\b", r"_reg\b"],
        "accumulator_lifetime": [r"set_memory\s*\([^)]*(GEMM_ACCUM|ACCUM)",
                                  r"lift_alloc\s*\([^)]*res", r"stage_mem\s*\([^)]*[\"']C"],
        "epilogue_before_commit": [r"\bacc_scale\b", r"\bclamp\b", r"\brelu\b"],
        "tiling_blocking": [r"\bdivide_loop\b", r"\btile_outer_loops\b", r"\btile\b",
                            r"\breorder_loops\b", r"\bsplit\b"],
        "vector_length_polymorphic": [r"\bvectorize\b.*sve", r"\brvm\b"],
        "weight_stationary_dataflow": [r"replace_gemmini_calls", r"GEMM_ACCUM"],
        "intrinsic_lowering": [r"\breplace_all\b", r"replace_gemmini_calls",
                               r"mm256_\w+", r"mm512_\w+", r"\bNeon\b"],
    },
    # Triton @triton.jit kernels: block-pointer staging, tl.dot accumulation, masked tails,
    # software pipelining via num_stages. Decisions are explicit in the Python intrinsics.
    "triton": {
        "packed_rhs": [r"tl\.make_block_ptr", r"tl\.advance", r"b_ptrs?\s*\+=",
                       r"\bBLOCK_SIZE_[KN]\b"],
        "accumulator_lifetime": [r"tl\.dot\b", r"\bacc(?:umulator)?\s*\+=", r"tl\.zeros\b"],
        "epilogue_before_commit": [r"accumulator\.to\(", r"\*\s*scale", r"tl\.sigmoid",
                                   r"leaky_relu", r"\bmaximum\(", r"tl\.where"],
        "tiling_blocking": [r"tl\.program_id", r"tl\.arange", r"\bBLOCK_SIZE_M\b",
                            r"for\s+\w+\s+in\s+range\([^)]*BLOCK"],
        "double_buffering": [r"num_stages\s*=\s*[2-9]", r"tl\.async", r"num_stages"],
        "intrinsic_lowering": [r"\btl\.\w+", r"@triton\.jit"],
    },
}


@lru_cache(maxsize=None)
def _compiled_for_family(family: str) -> tuple[tuple[str, tuple[re.Pattern, ...]], ...]:
    base = dict(_RAW.get("generic", {}))
    fam = _RAW.get(family, {})
    merged: dict[str, list[str]] = {}
    for motif in MOTIFS:
        pats = list(base.get(motif, [])) + list(fam.get(motif, []))
        if pats:
            merged[motif] = pats
    return tuple(
        (motif, tuple(re.compile(p) for p in pats)) for motif, pats in merged.items()
    )


def markers_for(target: str) -> dict[str, tuple[re.Pattern, ...]]:
    """Return ``{motif: (compiled_regex, ...)}`` for the ISA family of ``target``."""
    return dict(_compiled_for_family(target_family(target)))


def fired_markers(text: str, target: str) -> dict[str, list[str]]:
    """Return, per motif, the list of matched marker substrings found in ``text``.

    The returned snippets become the ``evidence.code_markers`` of a kernel record. Only
    motifs that fired appear in the result.
    """
    out: dict[str, list[str]] = {}
    for motif, patterns in markers_for(target).items():
        hits: list[str] = []
        for pat in patterns:
            m = pat.search(text)
            if m:
                hits.append(m.group(0).strip())
        if hits:
            # de-dup while preserving order
            out[motif] = list(dict.fromkeys(hits))
    return out
