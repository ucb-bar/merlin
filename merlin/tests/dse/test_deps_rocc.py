"""The command-trace -> dependence-graph adapter, and the two ways it can silently lose an edge.

Every assertion here is against a module constant or a derived value; none spells a target's opcode,
field name or mask as a literal, because a test that hardcodes what the code derives passes when the
derivation breaks.
"""
from __future__ import annotations

import pytest

from merlin.perf import depgraph
from merlin.perf.deps import rocc

TILE = 16


def _issue(cycles: float = 1.0) -> depgraph.IssueModel:
    return depgraph.IssueModel(issue_cycles=cycles, stall_unit=cycles, tier="test",
                               provenance="constructed by the test, not measured")


def _row(index: int, klass: str, decoded: dict, **operands) -> dict:
    row = {"index": index, "class": klass, "funct": None, "decoded": decoded}
    for name in ("rs1", "rs2"):
        row[name] = {"raw": None, "kind": operands.get(name, "const"),
                     "arg_index": None, "offset": None}
    return row


DEFINES_SPAD = next(field for field, file in rocc.DEFINES.items() if file == "spad")
CONSUMES_SPAD = sorted(field for field, file in rocc.CONSUMES.items() if file == "spad")[0]
DEFINES_ACC = next(field for field, file in rocc.DEFINES.items() if file == "acc")
CONSUMES_ACC = next(field for field, file in rocc.CONSUMES.items() if file == "acc")
WRITER, STAGER = next(iter(rocc.INHERITS_DESTINATION.items()))
FLAG_FIELD, FLAG_FILE = next(iter(rocc.FLAG_EVIDENCE.items()))


def _stream(*, acc_flag_bit: int, span: int = TILE, chains: int = 2) -> list[dict]:
    """``chains`` INDEPENDENT load/stage/write/readout chains, emitted one after another.

    Independence is what makes this a schedule at all. A single chain has exactly one legal order, so
    every candidate ties and the test would pass on a graph that discriminates nothing. Two chains
    that touch disjoint addresses may be interleaved, which is the reordering the ranking is about.
    """
    rows: list[dict] = []
    for chain in range(chains):
        spad, acc = chain * span, chain * span
        rows += [
            _row(len(rows), "LOAD", {DEFINES_SPAD: spad, rocc.WIDTH_FIELD: span}),
            _row(len(rows) + 1, STAGER,
                 {CONSUMES_SPAD: spad, DEFINES_ACC: acc | acc_flag_bit, FLAG_FIELD: True}),
            _row(len(rows) + 2, WRITER, {CONSUMES_SPAD: spad}),
            _row(len(rows) + 3, "READOUT", {CONSUMES_ACC: acc, FLAG_FIELD: True}),
        ]
    return [dict(row, index=i) for i, row in enumerate(rows)]


def test_a_wide_definition_covers_every_slot_it_wrote():
    """A transfer writing a run of rows must order a later writer of any row in that run after it."""
    effects = rocc.effects_of_row(_row(0, "LOAD", {DEFINES_SPAD: 0, rocc.WIDTH_FIELD: TILE}))
    assert len(effects.defs) == TILE, "the declared row count is the span of the definition"
    assert {a.slot for a in effects.defs} == set(range(TILE))
    # A consumer names only the base; expanding the use would invent edges rather than derive them.
    consumer = rocc.effects_of_row(_row(1, "USE", {CONSUMES_SPAD: 0}))
    assert len(consumer.uses) == 1


def test_mode_bits_are_stripped_only_where_a_mask_was_derived():
    """The same accumulator tile addressed with and without a mode bit is ONE tile, or no edge exists."""
    bit = 1 << 30
    masks = {FLAG_FILE: bit}
    staged = rocc.effects_of_row(
        _row(1, STAGER, {DEFINES_ACC: bit, FLAG_FIELD: True}), flag_masks=masks)
    read = rocc.effects_of_row(
        _row(3, "READOUT", {CONSUMES_ACC: 0, FLAG_FIELD: True}), flag_masks=masks)
    assert staged.defs[0].slot == read.uses[0].slot, (
        "with the mask applied the stager and the readout name the same slot")

    # Without a derived mask the address cannot be identified, and that is UNRESOLVED -- never the
    # raw address, which would put the two on different slots and delete the dependence silently.
    unmasked = rocc.effects_of_row(_row(1, STAGER, {DEFINES_ACC: bit, FLAG_FIELD: True}))
    assert not unmasked.defs
    assert unmasked.unresolved and FLAG_FILE in unmasked.unresolved[0]


def test_a_writer_inherits_the_destination_staged_for_it():
    """Without this the readout depends on the STAGER, and may legally hoist above the writer."""
    rows = _stream(acc_flag_bit=0, chains=1)
    _instructions, effects = rocc.instructions_and_effects(rows, flag_masks={FLAG_FILE: 0})
    writer = effects[2]
    assert any(a.file == FLAG_FILE for a in writer.defs), "the writer defines what was staged for it"
    assert any(a.file == FLAG_FILE for a in writer.uses), "and reads it, since it may accumulate"
    assert not writer.unresolved


def test_a_writer_with_no_stager_is_unresolved_rather_than_effect_free():
    orphan = [_row(0, WRITER, {CONSUMES_SPAD: 0})]
    _instructions, effects = rocc.instructions_and_effects(orphan)
    assert effects[0].unresolved, "a writer whose destination nobody staged has an unknown footprint"
    assert STAGER in effects[0].unresolved[0]


def test_an_undecodable_command_is_unresolved_not_empty():
    effects = rocc.effects_of_row({"index": 0, "class": "UNKNOWN", "decoded": {}})
    assert effects.unresolved and not effects.observed
    assert not effects.defs and not effects.uses


def test_pricing_the_one_separation_class_makes_the_graph_discriminate():
    """The whole point of the adapter: two orderings of one program must score differently.

    Unpriced, every separation weighs zero and the graph degenerates to the instruction count, so
    every legal ordering ties -- which is what it did before this adapter existed. Priced, the
    orderings separate and `differential.compare` returns an EXACT delta rather than a refusal.
    """
    from merlin.perf import differential

    rows = _stream(acc_flag_bit=0)
    program = rocc.program_from_trace(rows, flag_masks={FLAG_FILE: 0},
                                      roles={r["class"]: "accelerator" for r in rows})
    order = list(range(len(program.instructions)))

    unpriced = depgraph.build_dag(program.instructions, program.effects, issue=_issue(),
                                 stall_mnemonic="", roles=program.roles)
    assert set(unpriced.exposed_classes()) == {f"{depgraph.SEPARATION}.accelerator"}, (
        "one structural role must yield exactly one unknown, or two candidates cannot cancel it")

    priced = depgraph.build_dag(program.instructions, program.effects, issue=_issue(),
                                stall_mnemonic="", roles=program.roles,
                                resolved_separations={f"{depgraph.SEPARATION}.accelerator": 20.0})
    assert priced.exposed_classes() == {}, "pricing the only class leaves nothing exposed"

    schedules = depgraph.candidates_for(priced, order, stall_mnemonic="",
                                        hoist_role="accelerator", roles=program.roles)
    costs = {name: depgraph.makespan(priced, seq) for name, seq in schedules.items()}
    flat = {name: depgraph.makespan(unpriced, seq) for name, seq in schedules.items()}
    assert len(set(flat.values())) == 1, "unpriced, every ordering ties -- the old behaviour"
    assert max(costs.values()) > min(costs.values()), "priced, the orderings must separate"

    composed = {name: depgraph.to_composed(c, priced) for name, c in costs.items()}
    demands = {name: depgraph.demands_of(priced) for name in composed}
    names = sorted(composed)
    verdict = differential.compare(composed[names[0]], composed[names[-1]],
                                   demands_a=demands[names[0]], demands_b=demands[names[-1]],
                                   label_a=names[0], label_b=names[-1])
    assert verdict.basis == "exact"


def test_an_illegal_ordering_is_refused_rather_than_priced():
    """A hoist above a real dependence must raise; pricing one produces a number that beats every
    legal ordering, which is how a missing edge turns into a recommendation."""
    rows = _stream(acc_flag_bit=0, chains=1)
    program = rocc.program_from_trace(rows, flag_masks={FLAG_FILE: 0},
                                      roles={r["class"]: "accelerator" for r in rows})
    dag = depgraph.build_dag(program.instructions, program.effects, issue=_issue(),
                             stall_mnemonic="", roles=program.roles)
    readout_first = [3, 0, 1, 2]
    with pytest.raises(ValueError):
        depgraph.makespan(dag, readout_first)


def test_main_memory_is_declared_untracked_rather_than_assumed_absent():
    assert "dram" in rocc.untracked_files()
