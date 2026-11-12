"""Synthesize the five plan artifacts from collected evidence.

For ``toy_npu`` the synthesizers emit concrete, schema-valid plans consistent with the
authored ``examples/targets/toy_npu/target/contracts/``. For real targets they emit conservative
plans derived from keyword-detected concepts, with every result flagged
``confidence: low|medium`` and ``requires_human_review: true``.

Each function returns a plain ``dict`` (schema-shaped); the pipeline validates and writes it.
"""

from __future__ import annotations

from merlin.development.targetgen.synthesize.dialect_plan import synthesize_dialect_plan
from merlin.development.targetgen.synthesize.llvm_extension_plan import synthesize_llvm_extension_plan
from merlin.development.targetgen.synthesize.runtime_adapter_plan import synthesize_runtime_adapter_plan
from merlin.development.targetgen.synthesize.target_contract import synthesize_target_contract
from merlin.development.targetgen.synthesize.zephyr_plan import synthesize_zephyr_plan

__all__ = [
    "synthesize_target_contract",
    "synthesize_dialect_plan",
    "synthesize_runtime_adapter_plan",
    "synthesize_zephyr_plan",
    "synthesize_llvm_extension_plan",
]
