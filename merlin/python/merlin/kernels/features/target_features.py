"""Target-specific configuration features.

Decision recorded: the *dataflow mode* exposed by the target (weight- vs output-stationary),
the ISA family, and whether the kernel commits to target-specific intrinsics/config. These
are the "target-specific configuration" facts the acceptance criteria call for.
"""
from __future__ import annotations

from merlin.kernels.markers import target_family
from merlin.kernels.types import NormalizedKernel


def extract_target_features(nk: NormalizedKernel, fired: dict[str, list[str]]) -> dict:
    text = nk.raw_text
    dataflow = "na"
    if "weight_stationary_dataflow" in fired:
        dataflow = "output_stationary" if "OUTPUT_STATIONARY" in text else "weight_stationary"
    return {
        "dataflow": dataflow,
        "isa_family": target_family(nk.target),
        "target_specific_config": "intrinsic_lowering" in fired,
    }
