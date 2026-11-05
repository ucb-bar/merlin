"""Target-independent xDSL transformations with explicit verification handoff."""

from .certificates import (
    BufferInterval,
    BufferReuseWitness,
    ChangeCertificate,
    DependencyWitness,
    IntegerWitness,
    LoopWitness,
    ShapeWitness,
    TransformationChange,
)
from .loops import iter_segments, ordered_loop_obligation
from .pipeline import PipelineOptions, PipelineResult, apply_common_pipeline
from .planning import DependencyPlan, StoragePlan, plan_dependencies, plan_storage, validate_order
from .validation import validate_change

__all__ = [
    "BufferInterval",
    "BufferReuseWitness",
    "ChangeCertificate",
    "DependencyPlan",
    "DependencyWitness",
    "IntegerWitness",
    "LoopWitness",
    "PipelineOptions",
    "PipelineResult",
    "ShapeWitness",
    "StoragePlan",
    "TransformationChange",
    "apply_common_pipeline",
    "iter_segments",
    "ordered_loop_obligation",
    "plan_dependencies",
    "plan_storage",
    "validate_change",
    "validate_order",
]
