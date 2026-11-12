"""Per-layer RTL evidence: build one small layer program, run it on a cycle-accurate engine, keep the receipt.

The unit of measurement is one fused group at its real shape. Target facts (build recipe, engine
command) come from the target's registered backend; nothing here names a target.
"""

from merlin.analysis.perf.layer_bench.build import BuildError, BuiltProgram, build_program, loaded_bytes
from merlin.analysis.perf.layer_bench.cache import ReceiptCache, ReceiptError, seal_receipt, verify_receipt
from merlin.analysis.perf.layer_bench.console import (
    ConsoleError,
    EngineFinish,
    LayerRecord,
    parse_engine_finish,
    parse_layer_records,
)
from merlin.analysis.perf.layer_bench.key import LayerKey
from merlin.analysis.perf.layer_bench.run import EngineRun, EngineUnavailable, run_on_gsim

__all__ = [
    "BuildError",
    "BuiltProgram",
    "ConsoleError",
    "EngineFinish",
    "EngineRun",
    "EngineUnavailable",
    "LayerKey",
    "LayerRecord",
    "ReceiptCache",
    "ReceiptError",
    "build_program",
    "loaded_bytes",
    "parse_engine_finish",
    "parse_layer_records",
    "run_on_gsim",
    "seal_receipt",
    "verify_receipt",
]
