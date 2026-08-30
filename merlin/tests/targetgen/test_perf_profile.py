"""Archetypes are priors; TRAITS decide what applies -- and an unestablished trait stays UNKNOWN.

The property under test is the anti-overfit one: **the same code must derive a profile for two
machines of different archetypes and produce different, correct answers.** A tool that only works
where it was written is manual overfitting with extra steps, and the way that failure hides is a
default -- a trait that reads True because somebody's machine had it, not because this machine's
sources say so. So most of these tests delete a fact and assert the trait goes ``None``.

Two kinds of fixture, deliberately:

* **Synthetic machines** (``_self_hosted_facts`` / ``_host_queued_facts``) -- two fabricated targets
  of different archetypes, so the anti-overfit property is checked even on a host with no RTL facts
  cached, and so a fact can be deleted without touching anything on disk.
* **The two real targets**, which are the regression fixtures: the tool is wrong if it does not
  reproduce their measured numbers. They SKIP (never pass) where the facts artifact is not on this
  host -- a check that could not run is ``not_run``, never a pass.
"""
from __future__ import annotations

import pytest

from merlin.perf.profile import (TIER_FACTS, TIER_NONE, TIER_RESIDUAL, TRAITS, TargetProfile,
                                 derive_profile, profile_table, timing_walk, load_sources)

# The two real machines this program is measured against. They are different archetypes on purpose:
# a self-hosted-ISA tensor core and a host-dispatched decoupled-queue systolic co-processor, so a
# tool that silently assumed one shape fails visibly rather than quietly agreeing.
SELF_HOSTED_TARGET = "atlas"
HOST_QUEUED_TARGET = "gemmini"


# ---------------------------------------------------------------------------------------------
# Synthetic fixtures: two machines, described only by their own sources
# ---------------------------------------------------------------------------------------------


def _facts(body: dict, *, digest: str = "d0b4135a") -> dict:
    return {"schema_version": 2.0,
            "generator": {"name": "a.test.extractor", "version": "test-v1"},
            "inputs": {"core_hw_mlir": "machine_hw.mlir", "core_hw_sha": digest},
            "facts": body}


def _timing(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        rec = {"module": r["module"], "registers": r.get("registers", 0),
               "source": "mlc_hw_graph_walk", "n_outputs": r.get("n_outputs", 4),
               "n_cyclic": r.get("n_cyclic", 0), "pipeline_depth": r.get("pipeline_depth"),
               "partial_depth": r.get("partial_depth"),
               "evidence": r.get("evidence", "a walk over this module")}
        out.append(rec)
    return out


def _self_hosted_facts() -> dict:
    """A machine that fetches and decodes its own instruction stream: opcodes far too wide for a
    host co-processor funct field, no movement interface described, no memory discovered."""
    return _facts({
        "arrays": [{"name": "mesh", "rows": 32, "cols": 32, "container": "BigMesh",
                    "element": "Cell", "source": "mlc_discovery"}],
        "interfaces": [{"name": "funct_decode_table", "legal_funct": [87, 4311, 9943],
                        "funct3": 3, "custom_opcode": None,
                        "hw_source": "/somewhere/machine_hw.mlir"}],
        "timing": _timing([
            {"module": "BigMesh", "pipeline_depth": 31, "registers": 992, "n_outputs": 32},
            {"module": "Cell", "pipeline_depth": 1, "registers": 2, "n_outputs": 3},
            {"module": "Sequencer", "pipeline_depth": None, "n_outputs": 8, "n_cyclic": 8,
             "evidence": "8 of 8 hw.output operands are reached through feedback"},
        ]),
    })


def _self_hosted_residual() -> dict:
    return {"name": "a_self_hosted_machine", "version": "0.1",
            "compute_units": [{"name": "mxu", "kind": "systolic", "ops": ["matmul"]}],
            "memory_model": {"resident": True, "accumulators": True}}


def _host_queued_facts() -> dict:
    """A host-dispatched co-processor: a narrow funct decode, a command queue, a DMA engine with its
    own translation, two discovered memories, and a mesh whose accumulation feeds back."""
    return _facts({
        "arrays": [{"name": "mesh", "rows": 16, "cols": 16, "container": "Mesh",
                    "element": "Tile", "source": "mlc_discovery"}],
        "memories": [{"name": "scratchpad", "bytes": 262144, "depth": 4096,
                      "source": "mlc_discovery"},
                     {"name": "accumulator", "bytes": 65536, "depth": 512,
                      "source": "mlc_discovery"}],
        "interfaces": [
            {"name": "rocc_cmd", "evidence": "module ReservationStation (decode/dispatch)"},
            {"name": "dma_tlb", "evidence": "module FrontendTLB"},
            {"name": "funct_decode_table", "legal_funct": [0, 1, 2, 126], "custom_opcode": 123,
             "funct3": 3, "hw_source": "/somewhere/queued_core_hw.mlir"},
        ],
        "timing": _timing([
            {"module": "Mesh", "pipeline_depth": None, "n_outputs": 36, "n_cyclic": 36,
             "registers": 2340,
             "evidence": "36 of 36 hw.output operands of module Mesh are reached through feedback"},
            {"module": "Tile", "pipeline_depth": 0, "registers": 0, "n_outputs": 10},
        ]),
    })


def _host_queued_residual() -> dict:
    return {"name": "a_host_queued_machine", "version": "0.1",
            "endpoint_kind": "inline_asm_insn",
            "compute_units": [{"name": "systolic_mesh", "kind": "systolic", "ops": ["matmul"]}],
            "memory_model": {"resident": True, "accumulators": True},
            "encoding": {"addr_len": 32,
                         "config_subtype": {0: "CONFIG_EX", 1: "CONFIG_LD", 2: "CONFIG_ST"}}}


def _synthetic(kind: str, **over) -> TargetProfile:
    if kind == "self_hosted":
        facts, residual = _self_hosted_facts(), _self_hosted_residual()
    else:
        facts, residual = _host_queued_facts(), _host_queued_residual()
    facts = over.get("facts", facts)
    residual = over.get("residual", residual)
    return derive_profile(f"a_{kind}_machine", facts=facts, residual=residual)


def _real_or_skip(target: str) -> TargetProfile:
    prof = derive_profile(target)
    if not prof.sources.body:
        pytest.skip(f"no RTL facts artifact for {target!r} on this host: the fixture could not "
                    "run, which is not_run and never a pass")
    return prof


# ---------------------------------------------------------------------------------------------
# THE ANTI-OVERFIT GATE: one code path, two machines, different and correct answers
# ---------------------------------------------------------------------------------------------


def test_two_archetypes_derive_different_trait_sets():
    """Same deriver, two machines: the trait sets must differ the way the machines do."""
    a = _synthetic("self_hosted")
    b = _synthetic("host_queued")

    assert a.archetype.dispatch == "device_native"
    assert b.archetype.dispatch == "host_instruction"
    assert a.archetype.label != b.archetype.label

    # The self-hosted machine decodes its own stream; the co-processor does not.
    assert a.has("self_hosted_program") is True
    assert b.has("self_hosted_program") is False

    # Only the co-processor's facts describe a command queue and a movement engine.
    assert b.has("host_dispatched_queue") is True
    assert b.has("explicit_dma") is True
    assert a.has("host_dispatched_queue") is None      # UNKNOWN, and specifically NOT False
    assert a.has("explicit_dma") is None

    # Both have a managed store -- but one is grounded in the RTL and the other is only declared,
    # and the profile must not let those read the same.
    assert a.has("managed_scratchpad") is True and a.trait_tier["managed_scratchpad"] == TIER_RESIDUAL
    assert b.has("managed_scratchpad") is True and b.trait_tier["managed_scratchpad"] == TIER_FACTS

    assert set(a.satisfied()) != set(b.satisfied())
    assert set(a.unestablished()) != set(b.unestablished())


def test_real_targets_derive_different_profiles():
    """The regression fixture: the two real machines, through the identical code path."""
    a = _real_or_skip(SELF_HOSTED_TARGET)
    b = _real_or_skip(HOST_QUEUED_TARGET)

    assert a.archetype.dispatch == "device_native", a.archetype.evidence
    assert b.archetype.dispatch == "host_instruction", b.archetype.evidence
    assert a.has("self_hosted_program") is True
    assert b.has("self_hosted_program") is False
    assert b.has("explicit_dma") is True
    assert a.has("explicit_dma") is None
    assert set(a.satisfied()) != set(b.satisfied())

    table = profile_table([a, b])
    assert SELF_HOSTED_TARGET in table and HOST_QUEUED_TARGET in table
    for name in TRAITS:
        assert name in table


def test_archetype_is_a_prior_not_a_gate():
    """The archetype only chooses the questions; the traits answer them, and can refuse."""
    a = _synthetic("self_hosted")
    assert "explicit_dma" in a.archetype.questions          # the prior asks
    assert a.has("explicit_dma") is None                    # the evidence declines to answer
    worklist = dict(a.worklist())
    assert "explicit_dma" in worklist                       # so it becomes a work item
    assert worklist["explicit_dma"], "an UNKNOWN trait must name what would settle it"
    # ... and a question the evidence DID settle is not on the worklist.
    assert "self_hosted_program" not in worklist


# ---------------------------------------------------------------------------------------------
# Deleting a fact yields UNKNOWN, never a default
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("drop,trait", [
    ("interfaces", "explicit_dma"),
    ("interfaces", "host_dispatched_queue"),
    ("timing", "structural_pipeline_depth"),
    ("timing", "feedback_sequenced_units"),
])
def test_deleting_a_fact_makes_the_trait_unknown(drop, trait):
    facts = _host_queued_facts()
    before = _synthetic("host_queued", facts=facts)
    assert before.has(trait) is not None, "fixture must establish the trait before it is deleted"

    del facts["facts"][drop]
    after = _synthetic("host_queued", facts=facts)
    assert after.has(trait) is None, (
        f"deleting facts.{drop} must leave {trait!r} UNKNOWN, not fall back to a default")
    assert after.traits[trait].missing, "an UNKNOWN trait must name what is missing"
    assert after.trait_tier[trait] == TIER_NONE


def test_deleting_the_memories_fact_drops_the_tier_but_not_to_a_fabricated_capacity():
    """The residual can still DECLARE a managed store; it can never ground one."""
    facts = _host_queued_facts()
    del facts["facts"]["memories"]
    prof = _synthetic("host_queued", facts=facts)
    assert prof.has("managed_scratchpad") is True
    assert prof.trait_tier["managed_scratchpad"] == TIER_RESIDUAL
    assert "DECLARES" in prof.trait("managed_scratchpad").evidence


def test_an_absent_timing_block_is_uncached_not_absent():
    """A missing fact class means nobody could look, which is not "this design has no timing"."""
    facts = _host_queued_facts()
    del facts["facts"]["timing"]
    prof = _synthetic("host_queued", facts=facts)
    assert prof.timing.status == "uncached"
    assert prof.has("structural_pipeline_depth") is None
    assert "uncached" in prof.trait("structural_pipeline_depth").evidence

    # An EMPTY walk is a different state: it ran, and found nothing.
    facts["facts"]["timing"] = []
    empty = _synthetic("host_queued", facts=facts)
    assert empty.timing.status == "empty"
    assert empty.has("structural_pipeline_depth") is False

    # And no facts at all is a third.
    none = derive_profile("a_machine_nobody_extracted", facts={}, residual={})
    assert none.timing.status == "no_facts"
    assert none.has("structural_pipeline_depth") is None


def test_no_facts_at_all_never_produces_a_confident_profile():
    prof = derive_profile("a_machine_nobody_extracted", facts={}, residual={})
    assert prof.satisfied() == ()
    assert prof.refuted() == ()
    assert set(prof.unestablished()) == set(TRAITS)
    assert prof.archetype.dispatch is None


# ---------------------------------------------------------------------------------------------
# depth == 0 is a REAL answer
# ---------------------------------------------------------------------------------------------


def test_a_resolved_depth_of_zero_is_preserved_as_zero():
    """``0`` (combinational) and ``None`` (sequenced) are different answers all the way through."""
    prof = _synthetic("host_queued")
    depth, evidence = prof.timing.depth("Tile")
    assert depth == 0 and depth is not None
    assert evidence
    # The hazard, spelled out: the falsy check collapses the two states, `is None` keeps them apart.
    assert not depth                      # `if not depth:` would treat this real 0 as missing
    assert depth is not None              # ... which `is None` does not

    refused, why = prof.timing.depth("Mesh")
    assert refused is None
    assert "feedback" in why

    missing, why_missing = prof.timing.depth("NoSuchModule")
    assert missing is None
    assert "not among" in why_missing


def test_partial_depth_is_never_read_as_the_modules_depth():
    """A module with a partial depth still has NO pipeline depth. Two numbers, two names."""
    facts = _host_queued_facts()
    facts["facts"]["timing"] = _timing([
        {"module": "Mesh", "pipeline_depth": None, "partial_depth": 12, "n_outputs": 36,
         "n_cyclic": 4, "evidence": "4 of 36 hw.output operands are reached through feedback"},
    ])
    prof = _synthetic("host_queued", facts=facts)
    depth, why = prof.timing.depth("Mesh")
    assert depth is None, "partial_depth answers a different question and is never the latency"
    assert "feedback" in why


def test_real_target_timing_reproduces_the_measured_depths():
    """The measured regression fixture: one machine's array resolves, the other's refuses."""
    a = _real_or_skip(SELF_HOSTED_TARGET)
    b = _real_or_skip(HOST_QUEUED_TARGET)
    for prof in (a, b):
        if prof.timing.status != "present":
            pytest.skip(f"{prof.target}: the timing fact class is {prof.timing.status} on this "
                        "host (uncached, not absent) -- the fixture could not run")

    a_array = a.sources.arrays()[0]
    b_array = b.sources.arrays()[0]

    # A feed-forward array resolves; a weight-stationary one accumulates back through itself and
    # correctly refuses, so no finite wiring depth is its latency.
    a_depth, _ = a.timing.depth(a_array["container"])
    b_depth, b_why = b.timing.depth(b_array["container"])
    assert a_depth == a_array["rows"] - 1, "the container's depth is rows-1 on this elaboration"
    assert b_depth is None and "feedback" in b_why

    # And its element is combinational: a real 0.
    b_element, _ = b.timing.depth(b_array["element"])
    assert b_element == 0


# ---------------------------------------------------------------------------------------------
# Provenance of the profile itself
# ---------------------------------------------------------------------------------------------


def test_the_elaboration_prefers_the_dialect_actually_read():
    prof = _synthetic("host_queued")
    assert prof.elaboration.dialect == "machine_hw.mlir"
    assert prof.elaboration.evidenced is True
    assert "machine_hw.mlir" in prof.elaboration.describe()


def test_an_unrecorded_digest_makes_the_elaboration_asserted_not_evidenced():
    facts = _host_queued_facts()
    facts["inputs"] = {"hw_mlir": "some_soc.hw.mlir", "hw_sha": "missing"}
    prof = _synthetic("host_queued", facts=facts)
    assert prof.elaboration.evidenced is False
    assert "ASSERTED" in prof.elaboration.note
    assert "digest NOT recorded" in prof.elaboration.describe()


def test_every_trait_carries_evidence_and_every_unknown_names_what_is_missing():
    for prof in (_synthetic("self_hosted"), _synthetic("host_queued")):
        for name in TRAITS:
            trait = prof.traits[name]
            assert trait.evidence, f"{name} states no evidence"
            if trait.satisfied is None:
                assert trait.missing, f"{name} is UNKNOWN and does not say what would settle it"


def test_sources_report_what_was_missing():
    src = load_sources("a_machine_nobody_extracted", facts={}, residual={})
    assert "rtl_facts" in src.missing and "residual" in src.missing
    src2 = load_sources("a_machine", facts=_host_queued_facts(), residual=_host_queued_residual())
    assert set(src2.present) == {"rtl_facts", "residual"}


def test_timing_walk_counts_are_labelled_as_module_counts_not_coverage():
    walk = timing_walk(load_sources("m", facts=_self_hosted_facts(), residual={}))
    d = walk.to_dict()
    assert d["resolved_modules"] == 2 and d["refused_modules"] == 1
    assert "MODULE COUNTS, not coverage" in d["note"]


def test_profile_serializes_with_its_tiers_and_worklist():
    d = _synthetic("host_queued").to_dict()
    assert set(d["traits"]) == set(TRAITS)
    assert d["traits"]["explicit_dma"]["tier"] == TIER_FACTS
    assert d["archetype"]["dispatch"] == "host_instruction"
    assert "MODULE COUNTS" in d["timing"]["note"]
