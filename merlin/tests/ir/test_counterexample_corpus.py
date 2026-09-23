"""A refuted shape must actually become a capsule — through the real generator, not beside it.

The earlier version of this path wrote a capsule directory itself and got none of what the generator
provides: no `golden.yaml` for the grader, and `update_provenance_manifest` would have classified a
solver-produced capsule as `hand_authored`, which is exactly backwards. So the test that matters is
not "does emit_witness write files" but "would this entry materialise as a capsule the bench grades".
"""

from __future__ import annotations

import pytest


def _binding(declaration):
    from merlin_experiments.phase0 import generation, profiles

    from merlin.targetgen import corpus_spec as CS
    from merlin.targetgen.target_experiment import load_target_experiment

    generation._ensure_contract_on_path(declaration.descriptor)
    te = load_target_experiment(declaration.descriptor)
    profile = profiles.load_profile(declaration.profile, include_holdouts=False, **declaration.profile_inputs())
    return CS.derive_binding(te, profile.get("datapath", {}))


def _a_declaration():
    from merlin_experiments.phase0.declarations import all_declarations

    for declaration in all_declarations():
        try:
            _binding(declaration)
            return declaration
        except Exception:
            continue
    pytest.skip("no target in this checkout can build a corpus binding")


def test_a_refuted_shape_builds_a_schema_valid_capsule():
    """The whole claim of this path, asserted against the real builder and the real schema."""
    import json

    import jsonschema

    from merlin.common.paths import merlin_dir
    from merlin.targetgen import corpus_spec as CS
    from merlin.verify.counterexamples import counterexample_entry

    declaration = _a_declaration()
    target = declaration.target
    entry = counterexample_entry(target=target, m=15, k=17, n=15, bound_ms=300_000)
    capsule, mlir = CS.build(entry, _binding(declaration))

    schema = json.loads((merlin_dir() / "contract" / "schemas" / "capsule.schema.json").read_text())
    jsonschema.validate(capsule, schema)

    assert capsule["source_role"] == "smt_counterexample", (
        "provenance must survive into the capsule, or a solver-produced case is indistinguishable "
        "from one an author chose"
    )
    # the refuting SHAPE is what the entry exists to carry
    shapes = {tuple(i["shape"]) for i in capsule["inputs"]}
    assert (15, 17) in shapes and (17, 15) in shapes, f"the refuting shape was lost: {shapes}"
    assert mlir.strip(), "no interface program was emitted"


def test_the_entry_matches_what_the_generator_already_consumes():
    """A key the generator reads and the entry omits would fail only at materialisation time."""
    import yaml

    from merlin.verify.counterexamples import counterexample_entry

    declaration = _a_declaration()
    target = declaration.target
    synth = declaration.synth_profile
    if synth is None or not synth.is_file():
        pytest.skip(f"{target} has no synthesized profile to compare against")
    reference = (yaml.safe_load(synth.read_text()) or {}).get("capsules") or []
    if not reference:
        pytest.skip("synthesized profile is empty")

    entry = counterexample_entry(target=target, m=8, k=8, n=8)
    missing = sorted(set(reference[0]) - set(entry))
    assert not missing, f"the generator reads keys this entry does not provide: {missing}"


def test_names_cannot_collide_with_the_other_generators():
    """A solver capsule and a synthesized one must be distinguishable and never overwrite."""
    from merlin.targetgen.corpus_synth import SYNTH_PREFIX
    from merlin.verify.counterexamples import CX_PREFIX, counterexample_entry, merge_entries

    assert CX_PREFIX != SYNTH_PREFIX
    e = counterexample_entry(target="t", m=4, k=4, n=4)
    assert e["name"].startswith(CX_PREFIX + "_")
    assert all(c.isalnum() or c == "_" for c in e["name"]), "schema requires ^[A-Za-z0-9_]+$"

    # the same shape refuting twice must not duplicate the entry
    merged, added = merge_entries([e], [counterexample_entry(target="t", m=4, k=4, n=4)])
    assert len(merged) == 1 and added == 0


def test_the_entry_does_not_claim_to_carry_input_values():
    """The correction, asserted so it cannot quietly regress.

    A capsule has nowhere to put a stimulus (`inputs[]` has no values field,
    `additionalProperties: false`) and the grader fills leaves with `Tensor.deterministic`. The entry
    must say so rather than implying the solver's values are what gets graded.
    """
    from merlin.verify.counterexamples import counterexample_entry

    entry = counterexample_entry(target="t", m=4, k=4, n=4)
    reference = entry["source_reference"]
    assert "SHAPE is the solver's" in reference
    assert "deterministic fill" in reference
    assert not any(k in entry for k in ("values", "inputs", "counterexample")), (
        "the entry must not carry input values; a capsule cannot express them"
    )
