"""Select the host ISA, ABI and vectorization policy from compiler flags."""

from __future__ import annotations

from typing import Any


def _harness_cflags(model_flags: list[str]) -> list[str]:
    """Give GCC the model's ISA/ABI while retaining GCC-compatible harness options."""

    from .spike_model import RVV_CFLAGS

    flags = list(RVV_CFLAGS)
    for prefix in ("-march=", "-mabi=", "-mcmodel="):
        selected = [flag for flag in model_flags if flag.startswith(prefix)]
        if selected:
            flags = [flag for flag in flags if not flag.startswith(prefix)]
            flags.append(selected[-1])
    return flags


def _select_host_vectorize(
    model_flags: list[str],
    schedule: str | None,
    requested: bool | None,
) -> bool:
    """Select host scheduling from the declared ISA or the caller's explicit policy.

    Fixed-width vectors can be scalarized by LLVM, so an explicit schedule or
    override may intentionally use them on a scalar ISA. The default RVV
    schedule should only be selected for an ISA declaring floating vectors.
    A ``zvl`` minimum-length extension alone does not provide vector execution.
    """
    if requested is not None and not isinstance(requested, bool):
        raise ValueError("host_vectorize must be a bool or None")
    if requested is False and schedule is not None:
        raise ValueError("host_vectorize=False conflicts with an explicit rvv_schedule")
    if requested is not None:
        return requested
    if schedule is not None:
        return True
    marches = [flag[len("-march=") :].lower() for flag in model_flags if flag.startswith("-march=")]
    if not marches:
        return False
    base, *extensions = marches[-1].split("_")
    if base.startswith(("rv32", "rv64")) and "v" in base[4:]:
        return True
    return any(extension.startswith(("zve32f", "zve64f", "zve64d")) for extension in extensions)


def selected_model_compiler_plan(
    *,
    backend: str,
    cflags_override: list[str] | None,
    vlen: int | None,
    features: frozenset[str] | None,
) -> dict[str, Any]:
    """Observe exactly the compiler/index selection used by a whole-model build.

    The private source preflight and the actual post-preparation build call the
    same producer owner. If preparation changes features or flags, their full
    observations differ and the downstream source proof must refuse.
    """
    from ...llvmlower.impr_features import apply_cflags
    from .spike_model import (
        _C_DESCRIPTOR_INDEX_BITS,
        CLANG_TARGET,
        RVV_CFLAGS,
        SpikeModelError,
        _harness_cflags,
        target_data_layout,
        toolchain,
    )
    from .zephyr_model import march_with_vlen

    if backend not in {"rvv", "scalar"}:
        raise SpikeModelError(f"unknown whole-model backend {backend!r}")
    clang_cflags = list(cflags_override or (RVV_CFLAGS if backend == "rvv" else ["-march=rv64gc", *RVV_CFLAGS[1:]]))
    marches = [flag for flag in clang_cflags if flag.startswith("-march=")]
    if len(marches) != 1:
        raise SpikeModelError("whole-model build requires exactly one -march flag")
    gcc_cflags = _harness_cflags(clang_cflags)
    if vlen is not None and backend == "rvv":
        clang_cflags = march_with_vlen(clang_cflags, vlen)
        gcc_cflags = march_with_vlen(gcc_cflags, vlen)
    model_cflags = apply_cflags(clang_cflags, frozenset(features or frozenset()))
    clang = toolchain.clang()
    observation = target_data_layout.observe_index_width(clang, [CLANG_TARGET, *model_cflags])
    from merlin.common.digest import is_sha256

    if (
        not isinstance(observation, dict)
        or observation.get("schema") != "merlin.selected-index-lowering.v1"
        or observation.get("compiler_requested") != str(clang)
        or not isinstance(observation.get("compiler_resolved"), str)
        or not is_sha256(observation.get("compiler_sha256"))
        or observation.get("cross_flags") != [CLANG_TARGET, *model_cflags]
        or type(observation.get("index_bits")) is not int
        or observation["index_bits"] <= 0
        or not isinstance(observation.get("data_layout"), str)
        or target_data_layout.default_index_bits(observation["data_layout"]) != observation["index_bits"]
    ):
        raise SpikeModelError("selected cross compiler returned an incomplete index-width observation")
    if observation["index_bits"] != _C_DESCRIPTOR_INDEX_BITS:
        raise SpikeModelError("selected compiler index width is incompatible with the int64_t C memref ABI")
    return {
        "observation": observation,
        "clang_cflags": clang_cflags,
        "gcc_cflags": gcc_cflags,
        "model_cflags": model_cflags,
    }
