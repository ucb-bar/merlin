"""Synthetic observations check owner custody; they are not timing evidence."""
import hashlib
import json
from types import SimpleNamespace

import pytest
from author_skill_fixtures import load_script

from merlin.transforms.evaluated_exploration import CompletedObservation, ExplorationIdentity, exploration_context_identity
from merlin.transforms.pass_exploration import EmittedProgram, ExplorationGeometry, ValidationPartition
from merlin.transforms.pipeline_policy import PipelineVariant, PolicyFacts, RegisteredPass
from merlin.transforms.pipeline_state import PassSpec, PipelineContext, PipelineSession
from merlin.transforms.shape_policy import ShapeCaseFeedback


def sha(value):
    return hashlib.sha256(value).hexdigest()


def owner(*, unknown=False, changed=False):
    context = PipelineContext(hardware="selected", resources="capacity", library="library")
    level = "test_source"
    names = ("original", "fold")
    paths = tuple(PipelineVariant(n, (n,)) for n in names)
    calls = []

    def program(m, name):
        return EmittedProgram(f"{name}:{m}".encode(), "test-effects", sha(str(m).encode()))

    def binding(geometry):
        m = geometry.operand_shapes[0][0]
        def verify(source):
            if type(source) is not dict or set(source) != {"m"} or type(source["m"]) is not int:
                raise ValueError("source requires an integer extent")
        session = PipelineSession(dict(m=m), context=context, ir_level=level,
            fingerprint=lambda s: sha(json.dumps(s, sort_keys=True).encode()), verify_ir=verify)
        facts = PolicyFacts(session.stamp(), shapes=((m,),))
        registry = {n: RegisteredPass(PassSpec(n, 0, produces=("program",), mutates_ir=False,
            ir_level=level), lambda s, n=n: {"program": program(m, n+"changed" if changed else n)})
            for n in names}
        return dict(session=session, registry=registry, facts=facts, output="program")

    def case(m):
        geometry = ExplorationGeometry("integer", ((m,),))
        facts = binding(geometry)["facts"]
        identity = ExplorationIdentity(facts.stamp.source_sha256, *(sha(n.encode()) for n in
            ("abi", f"input:{m}", f"reference:{m}", "facts", "library", "provider", "scope", "roi")),
            exploration_context_identity(context), "test-effects", "modeled_roi")
        observations = tuple((n, CompletedObservation(identity, sha(program(m,n).data), sha(f"feedback:{m}:{n}".encode()),
            "pass", "unknown" if unknown and m == 5 and n == "fold" else "complete", True,
            None if unknown and m == 5 and n == "fold" else 100 if n == "original" else 110 if m == 3 else 80)) for n in names)
        return ShapeCaseFeedback(geometry, facts, observations, sha(b"implementation"))

    training = tuple(case(m) for m in (3,4,6))
    design = ValidationPartition(tuple(c.geometry for c in training), (case(5).geometry,))

    def evaluate(requests):
        calls.append(requests)
        assert requests["missing_boundary_sources"] == []
        assert {r["geometry_sha256"] for r in requests["pending_evaluations"]} == {case(5).geometry.identity}
        return (case(5),)

    return SimpleNamespace(prepare=lambda: dict(training=training, alternatives=paths, design=design,
        incumbent="original", axes=((0,0),)), evaluate=evaluate, binding=binding, calls=calls)


def test_owner_trains_then_requests_heldout_then_reexecutes_exact_observed_emissions():
    adapter = owner()
    report = load_script("heavy-pass-recipes", "library_pipeline_policy").target_example(adapter)
    assert len(adapter.calls) == 1
    assert report["proposal"]["training_only"]
    assert report["requests"]["pending_evaluations"] == []
    assert any(r["changed_emission"] and r["selected"]["variant"] == "fold" for r in report["executions"])
    assert not report["performance_qualification"]


def test_unknown_owner_timing_keeps_incumbent_and_current_emission_must_match_feedback():
    recipe = load_script("heavy-pass-recipes", "library_pipeline_policy")
    report = recipe.target_example(owner(unknown=True))
    assert all(r["selected"]["variant"] == "original" for r in report["executions"])
    with pytest.raises(ValueError, match="differs from the original observed program"):
        recipe.target_example(owner(changed=True))
    with pytest.raises(ValueError, match="requires prepare"):
        recipe.target_example(SimpleNamespace())
