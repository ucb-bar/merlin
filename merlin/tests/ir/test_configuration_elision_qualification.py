"""SDK-free independent configuration model/fixture controls, not proofs."""

import hashlib
import json

import pytest
from configuration_elision_qualification_cases import cases, declarations
from configuration_elision_qualification_model import execute


def operation(name, *, slot=0, a=3, b=5):
    return {"instruction": name, "operation": "fixture." + name, "bindings": dict(slot=slot, a=a, b=b)}


def initial(value=91):
    return {"configuration:" + str(index): value + 13 * index for index in range(4)}


@pytest.mark.parametrize("bits", [32, 64])
def test_schema_is_pinned_and_marks_pulses_nonpure(bits):
    facts, software = declarations(bits)
    facts_doc, spec = json.loads(facts), json.loads(software)
    assert spec["isa"]["facts_sha256"] == hashlib.sha256(facts).hexdigest()
    assert spec["scheduling"]["configuration"]["pulse"]["pure_write"] is False
    pulse = next(row for row in spec["isa"]["instructions"] if row["name"] == "pulse")
    assert "events" in pulse["effects"]["writes"]
    assert facts_doc["facts"]["widths"][2:] == [bits, bits]


def test_core_corpus_denominator_is_fixed_and_every_case_expects_real_change():
    rows = cases()
    assert len(rows) == 10 and len({row.name for row in rows}) == 10
    assert all(row.removed > 0 for row in rows)


@pytest.mark.parametrize("seed", [-37, 0, 91, 1007])
@pytest.mark.parametrize("failure", [None, 0])
def test_redundant_setters_preserve_observations_and_terminal_failure_world(seed, failure):
    pair, work = operation("pair"), operation("work")
    assert execute([pair, pair, work, pair], initial(seed), fail_observation=failure) == execute(
        [pair, work], initial(seed), fail_observation=failure
    )


@pytest.mark.parametrize("barrier", [operation("reset"), operation("unknown"), operation("unordered", a=7)])
def test_forgetting_clobbers_and_unknown_order_cannot_drop_next_required_write(barrier):
    pair, work = operation("pair"), operation("work")
    before = [pair, barrier, pair, work]
    faulty = [pair, barrier, work]
    assert execute(before, initial()) != execute(faulty, initial())


def test_first_write_cannot_be_justified_by_cross_invocation_state():
    pair, work = operation("pair"), operation("work")
    assert execute([pair, work], initial(91)) != execute([work], initial(91))
    assert execute([pair, work], initial(0)) != execute([work], initial(0))


def test_one_changed_field_requires_the_whole_multifield_write():
    pair, changed, work = operation("pair"), operation("pair", b=7), operation("work")
    assert execute([pair, changed, work], initial()) != execute([pair, work], initial())


def test_pulse_or_observer_deletion_changes_output_trace():
    pair, pulse, work = operation("pair"), operation("pulse"), operation("work")
    assert execute([pair, pulse, work], initial()) != execute([pair, work], initial())
    assert execute([pair, work], initial()) != execute([pair], initial())


def test_false_pure_declaration_has_a_concrete_hidden_effect_counterexample():
    pair = operation("pair")
    assert execute([pair, pair], initial(), hidden_set_pulse=True) != execute([pair], initial(), hidden_set_pulse=True)
