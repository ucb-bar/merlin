from workloads.base import Workload
from workloads.primitives import ALL_PRIMITIVES
from workloads.composites import ALL_COMPOSITES
from workloads.models import ALL_MODELS
from workloads.quantized import ALL_QUANTIZED

ALL_WORKLOADS = ALL_PRIMITIVES + ALL_COMPOSITES + ALL_MODELS + ALL_QUANTIZED

WORKLOADS_BY_LEVEL = {}
for w in ALL_WORKLOADS:
    WORKLOADS_BY_LEVEL.setdefault(w.level, []).append(w)

__all__ = [
    "Workload",
    "ALL_WORKLOADS",
    "WORKLOADS_BY_LEVEL",
    "ALL_PRIMITIVES",
    "ALL_COMPOSITES",
    "ALL_MODELS",
]
