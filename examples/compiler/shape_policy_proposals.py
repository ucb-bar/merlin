"""Runnable training -> independent validation -> current-IR shared-pass example.

The cycle and numerical labels below are simulated API fixtures, not observed
performance, numerical qualification or target authority. Real tools must supply
their own exact CompletedObservation records and selected PipelineContext.
"""

import json
from dataclasses import replace

from merlin.transforms.certificates import ir_text, parse_ir
from merlin.transforms.evaluated_exploration import (
    CompletedObservation,
    ExplorationIdentity,
    exploration_context_identity,
)
from merlin.transforms.pass_exploration import ExplorationGeometry, ValidationPartition
from merlin.transforms.pipeline import PipelineOptions, apply_common_pipeline
from merlin.transforms.pipeline_policy import PipelineVariant, PolicyFacts, RegisteredPass
from merlin.transforms.pipeline_state import PassSpec, PipelineContext, PipelineSession
from merlin.transforms.shape_policy import PerformanceDomain, ShapeCaseFeedback, propose_shape_policies

CONTEXT = PipelineContext(library="example_simulated_domain")
PATHS = (PipelineVariant("original", ("emit",)), PipelineVariant("normalize", ("normalize", "emit")))


def session_and_facts(size):
    module = parse_ir(f"""builtin.module {{func.func @forward(%x: tensor<{size}xi32>) -> i32 {{
        %a = arith.constant 3 : i32
        %b = arith.constant 4 : i32
        %sum = arith.addi %a, %b : i32
        func.return %sum : i32
    }}}}""")
    session = PipelineSession(module, context=CONTEXT)
    operand = module.body.block.first_op.body.block.args[0]
    return session, PolicyFacts(session.stamp(), shapes=(operand.type.get_shape(),))


def fixture(size, simulated_cycles):
    session, facts = session_and_facts(size)
    identity = ExplorationIdentity(*("1"*64 for _ in range(9)), exploration_context_identity(CONTEXT),
                                   "xdsl", "modeled_roi")
    identity = replace(identity, source_sha256=session.stamp().source_sha256)
    control = CompletedObservation(identity, "2"*64, "3"*64, "pass", "complete", True, 100)
    return ShapeCaseFeedback(ExplorationGeometry("public_integer_example", facts.shapes), facts,
        (("original", control), ("normalize", replace(control, program_sha256="4"*64, cycles=simulated_cycles))),
        "5"*64)


def main():
    training = (fixture(3, 110), fixture(4, 80), fixture(6, 80))
    heldout_geometry = ExplorationGeometry("public_integer_example", ((5,),))
    design = ValidationPartition(tuple(c.geometry for c in training), (heldout_geometry,))
    proposal = propose_shape_policies(training, PATHS, design=design, incumbent="original", axes=((0, 0),))[0]
    requests = proposal.requests(training)  # Only the independent size-5 pair remains.
    heldout = fixture(5, 75)
    assessed = proposal.assess((*training, heldout))  # Independent results may veto the frozen path.

    def normalize(session):
        changed = apply_common_pipeline(session.module, PipelineOptions(64, reuse_buffers=False), plan_only=True)
        session.module = changed.module
        return {}

    registry = {
        "normalize": RegisteredPass(PassSpec("normalize", 0), normalize),
        "emit": RegisteredPass(PassSpec("emit", 1, produces=("emitted_ir",), mutates_ir=False),
                               lambda session: {"emitted_ir": ir_text(session.module)}),
    }
    current, facts = session_and_facts(8)  # Never emitted/evaluated in the training or heldout fixtures.
    domain = PerformanceDomain.from_observation(training[0].observations[0][1].identity,
                                               implementation_sha256="5"*64)
    result = assessed.execute(current, registry, facts, output="emitted_ir", domain=domain)
    assert result.receipt["selected_variant"] == "normalize" and "arith.addi" not in result.artifact
    return dict(status="API_demonstration_only", observations="simulated; no performance claim",
                proposed_guards=proposal.report["guards"], pending_validation=requests,
                selected_passes=[p["name"] for p in result.receipt["passes"]],
                emitted_source_sha256=result.receipt["after"]["source_sha256"])


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
