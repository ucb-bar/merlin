"""Select the host ISA, ABI and vectorization policy from compiler flags."""

from __future__ import annotations


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
