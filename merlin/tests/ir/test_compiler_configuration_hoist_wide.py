"""Installed canonical original C64 owner qualification, never huge native loops.

This file intentionally has no scratch overlay. Promote only with the reviewed
source bundle and run sequentially on the coordinated CPU29 lane.
"""

import json
from pathlib import Path

from configuration_elision_qualification_cases import QUALIFICATION
from test_compiler_configuration_hoist import actual

from merlin.compile.scheduling.configuration_c_consumer import ConfigurationCConsumer
from merlin.sched.isa import InstrDef, InstructionSet, Operand
from merlin.verify import compiler_configuration_hoist as gate
from merlin.verify import pass_validation
from merlin.verify.speculative_gate import SpeculativeGate

MAX64 = (1 << 63) - 1


def native_iset():
    operands = tuple(Operand(key, "int") for key in ("slot", "a", "b"))
    return InstructionSet(
        "configuration_fixture",
        {
            "pair": InstrDef("pair", operands, lambda args: "pair(" + ",".join(args) + ");"),
            "pulse": InstrDef("pulse", operands, lambda args: "pulse(" + ",".join(args) + ");"),
            "work": InstrDef("work", operands, lambda args: "if (work(" + ",".join(args) + ")) return;"),
        },
        {},
        "/* synchronous */",
    )


def checked_public(work, tools, *, extent):
    chosen, original, _ = actual(work, extent=extent)
    consumer = ConfigurationCConsumer(index_bits=64)
    worker = SpeculativeGate(work / "worker", job_timeout=300)
    validation = pass_validation.PassValidation(worker, tools, frozenset({gate.SELECTOR}))
    result = consumer.hoist(
        original,
        chosen.selected_contracts.configuration_adapter(QUALIFICATION),
        selected_contracts=chosen.selected_contracts,
        pass_validation=validation,
    )
    assert len(worker.verdicts) == 1 and worker.verdicts[0]["verdict"] == "accepted"
    detail = worker.verdicts[0]["detail"]
    assert detail["schema"] == "merlin.compiler-configuration-hoist.gate.v2"
    assert detail["c_consumer"] == consumer.record()
    report = json.loads(Path(detail["receipt"]).read_text())
    assert report["schema"] == "merlin.compiler-configuration-hoist.checked.v2"
    assert report["source_request"]["c_consumer"] == consumer.record()
    assert report["candidate"]["all_index_model_premise_retained"] is True
    assert report["hardware_model_approved"] is False
    return consumer, chosen, original, result, detail, report

    # No compilation, loading, interpretation or execution of these huge loops.
