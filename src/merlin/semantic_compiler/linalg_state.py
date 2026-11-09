"""State shared by ordered semantic source-translation families."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .model import KernelRequest
from .reference_values import TensorValue

@dataclass(frozen=True)
class LinalgTranslation:
    request: KernelRequest
    constants: dict[str, TensorValue]
    source_operations: tuple[str, ...]

@dataclass
class TranslationState:
    nodes: Any
    env: Any
    partial_tensors: Any
    scalars: Any
    i64_scalars: Any
    f32_scalars: Any
    f64_scalars: Any
    boolean_scalars: Any
    bf16_zeros: Any
    zero_fills: Any
    constants: Any
    constant_bindings: Any
    boundaries: Any
    source_operations: Any
    operations: Any
    function: Any
    boundary_storage: Any
    target_identity: Any
    lowering_policy: Any
    source: Any
    append_splat: Any
    append_scalar_region: Any
    scan_prelude: Any
    ordered_scans: Any
    indexed_prelude: Any
    indexed_updates: Any
