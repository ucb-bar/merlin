"""Scheduling/memory capabilities are supported only through a named RTL or ordering fact."""

import copy
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from merlin.targetgen.rtl.command_ordering_selection import SCHEMA as SELECTION_SCHEMA
from merlin_experiments.phase1 import capability_facts as C
from merlin_experiments.phase1 import compiler_library as CL
from merlin_experiments.phase1 import joint_control
from merlin_experiments.phase1 import optimization_suggestions as S
from merlin_experiments.phase1.feedback import next_steps as N

REPO = Path(__file__).resolve().parents[3]

FACTS = {
    "facts": {
        "memory_address_spaces": [{"name": "scratchpad", "view_kind": "address_space", "bytes": 4096}],
        "timing_structure": {
            "command_classes": {
                "status": "derived",
                "dispatch_module": "Dispatch",
                "classes": {"qa": {"entries": {"status": "derived", "value": 4}}},
            },
            "memory_clients": {
                "status": "derived",
                "clients": {"reader": {"id_space": 4, "in_flight": {"status": "derived", "value": 4}}},
            },
        },
    }
}
CONFIG_TERM = {"code": 1, "operand": "rs1", "offset": 16, "width": 16, "reset": 0}
ORDERING = {
    "schema": "merlin.rtl_command_ordering.v1",
    "status": "derived",
    "funct_queue": [{"funct": 0, "by_code": {"0": "qa", "1": "qb"}}, {"funct": 1, "queue": "qb"}],
    "accesses": {
        "IN.dst": {"class": "IN", "direction": "write", "queue": "qb"},
        "USE.src": {"class": "USE", "direction": "read", "queue": "qa"},
    },
    "same_queue": {"qa": {"release": "issue"}, "qb": {"release": "issue"}},
    "cross_queue": [
        {"older": "IN.dst", "younger": "USE.src", "space": "scratchpad", "overlap": "completion",
         "disjoint": "issue", "tracked": True},
    ],
    "extents": {
        "IN.dst": {"status": "derived", "verified": [[1, None, 1]], "rows": {"operand": "rs2", "offset": 48},
                   "scale": None,
                   "blocks": {"cols": {"operand": "rs2", "offset": 32}, "block": 8, "stride": dict(CONFIG_TERM)}},
        "USE.src": {"status": "derived", "verified": [[1, None, 1]], "rows": {"operand": "rs1", "offset": 48},
                    "scale": dict(CONFIG_TERM, code=0), "blocks": None},
    },
    "completion_order": {"qb": {"order": "out_of_order", "clients": ["reader: 4 response IDs, 4 transaction slots"]}},
    "classes": {"config": {"funct": 0, "by_code": {"0": "SET_A", "1": "SET_B"}}},
    "unknown": [],
}


def derived(facts=FACTS, ordering=ORDERING):
    return C.derive(copy.deepcopy(facts), copy.deepcopy(ordering))


@pytest.mark.parametrize("trait,fact", [
    ("scheduling.queues", "timing_structure.command_classes.classes.qa.entries"),
    ("scheduling.finite_queues", "timing_structure.command_classes.classes.qa.entries"),
    ("scheduling.async_issue", "command_ordering.same_queue.qa.release"),
    ("scheduling.multiple_outstanding", "command_ordering.completion_order.qb"),
    ("scheduling.dependency_tokens", "command_ordering.cross_queue[IN.dst -> USE.src, scratchpad].tracked"),
    ("scheduling.persistent_configuration", "command_ordering.extents.IN.dst.blocks.stride"),
    ("memory.resident_operands", "command_ordering.cross_queue[IN.dst -> USE.src, scratchpad]"),
    ("memory.strided_local_rows", "command_ordering.extents.IN.dst.blocks"),
])
def test_each_capability_is_supported_by_its_named_fact(trait, fact):
    values, evidence, provenance = derived()
    assert values[trait] == "supported" and provenance[trait]["fact"] == fact
    assert evidence[trait].startswith(fact) and provenance[trait]["basis"]


def _without(path):
    """The fixture with one fact removed (path: (document, key, key, ...))."""
    facts, ordering = copy.deepcopy(FACTS), copy.deepcopy(ORDERING)
    root = {"facts": facts["facts"], "ordering": ordering}[path[0]]
    for key in path[1:-1]:
        root = root[key]
    root.pop(path[-1])
    return facts, ordering


@pytest.mark.parametrize("trait,removed", [
    ("scheduling.finite_queues", ("facts", "timing_structure", "command_classes")),
    ("scheduling.async_issue", ("ordering", "same_queue")),
    ("scheduling.multiple_outstanding", ("ordering", "completion_order")),
    ("scheduling.dependency_tokens", ("ordering", "cross_queue")),
    ("scheduling.persistent_configuration", ("ordering", "classes")),
    ("memory.resident_operands", ("facts", "memory_address_spaces")),
    ("memory.resident_operands", ("ordering", "cross_queue")),
    ("memory.strided_local_rows", ("ordering", "extents", "IN.dst")),
])
def test_an_absent_fact_leaves_the_capability_unknown(trait, removed):
    facts, ordering = _without(removed)
    if removed[0] == "ordering" and removed[1] == "same_queue":
        ordering["cross_queue"] = []  # the cross-queue releases would establish it too
    values, _, provenance = C.derive(facts, ordering)
    assert trait not in values
    assert provenance[trait]["status"] == "unknown" and provenance[trait]["reason"]


def test_queues_fall_back_to_ordering_queue_names_only():
    facts, ordering = _without(("facts", "timing_structure", "command_classes"))
    values, _, provenance = C.derive(facts, ordering)
    assert values["scheduling.queues"] == "supported"
    assert provenance["scheduling.queues"]["fact"] == "command_ordering.funct_queue"


def test_unverified_or_partial_extents_establish_nothing():
    ordering = copy.deepcopy(ORDERING)
    for extent in ordering["extents"].values():
        extent["verified"] = []
    values, _, _ = C.derive(copy.deepcopy(FACTS), ordering)
    assert "scheduling.persistent_configuration" not in values and "memory.strided_local_rows" not in values


def test_no_capability_is_supported_without_a_fact():
    values, evidence, provenance = C.derive({}, None)
    assert values == {} and evidence == {}
    assert set(provenance) == set(C.DERIVED) and all(row["status"] == "unknown" for row in provenance.values())
    values, _, provenance = C.derive(copy.deepcopy(FACTS), None)
    # Without the ordering record only the dispatch-table facts decide anything.
    assert set(values) == {"scheduling.queues", "scheduling.finite_queues"}
    for row in derived()[2].values():
        assert row["status"] == "unknown" or (row["fact"] and row["source"] in ("rtl_facts", "command_ordering"))


def test_unsupported_only_when_a_fact_excludes_it():
    ordering = copy.deepcopy(ORDERING)
    ordering["same_queue"] = {"qa": {"release": "completion"}, "qb": {"release": "completion"}}
    ordering["cross_queue"] = [dict(ORDERING["cross_queue"][0], disjoint="completion", tracked=None)]
    facts = copy.deepcopy(FACTS)
    facts["facts"]["timing_structure"]["memory_clients"]["clients"]["reader"]["in_flight"]["value"] = 1
    values, _, _ = C.derive(facts, ordering)
    assert values["scheduling.async_issue"] == "unsupported"
    assert values["scheduling.multiple_outstanding"] == "unsupported"
    ordering["unknown"] = ["queue qb: release could not be established"]
    assert "scheduling.async_issue" not in C.derive(facts, ordering)[0]  # an incomplete record excludes nothing


def _selection(tmp_path, facts_path, contract_path, ordering=ORDERING, **changes):
    doc = {"schema": SELECTION_SCHEMA,
           "base_facts": {"path": str(facts_path), "sha256": C._sha(facts_path)},
           "base_contract": {"path": str(contract_path), "sha256": C._sha(contract_path)},
           "ordering_sha256": C._digest(ordering), "ordering": ordering}
    doc.update(changes)
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(doc))
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


@pytest.fixture
def inputs(tmp_path):
    facts, contract = tmp_path / "facts.json", tmp_path / "contract.yaml"
    facts.write_text(json.dumps(FACTS))
    contract.write_text(yaml.safe_dump({"endpoint_kind": "inline_asm_insn"}))
    return facts, contract


def test_an_ordering_selection_is_used_only_when_bound_to_these_facts_and_contract(tmp_path, inputs):
    facts, contract = inputs
    pin = _selection(tmp_path, facts, contract)
    record, provenance = C.ordering_input(pin, facts_path=facts, contract_path=contract)
    assert record == ORDERING and provenance["selection"] == pin and provenance["status"] == "derived"
    other = tmp_path / "other.yaml"
    other.write_text("endpoint_kind: other\n")
    record, provenance = C.ordering_input(pin, facts_path=facts, contract_path=other)
    assert record is None and "another target contract" in provenance["reason"]
    tampered = _selection(tmp_path, facts, contract, ordering_sha256="0" * 64)
    assert C.ordering_input(tampered, facts_path=facts, contract_path=contract)[0] is None
    stale = dict(pin, sha256="1" * 64)
    assert C.ordering_input(stale, facts_path=facts, contract_path=contract)[0] is None
    own = {"facts": {"command_ordering": ORDERING}}
    assert C.ordering_input(None, facts_document=own)[0] == ORDERING
    assert C.ordering_input(None, facts_document=FACTS) == (None, C.ordering_input(None)[1])


def _workspace(tmp_path, facts):
    ws, run = tmp_path / "ws", tmp_path / "run"
    (ws / "public_declarations").mkdir(parents=True)
    (ws / "rtl_facts").mkdir()
    run.mkdir()
    spec = {"numerical_semantics": {"operand_dtype": "int8", "accumulator_dtype": "int32"}}
    (ws / "public_declarations/software-spec.yaml").write_text(yaml.safe_dump(spec))
    (ws / "rtl_facts/facts.json").write_bytes(facts.read_bytes())
    return ws, run


def _status(report, name):
    return next(row for row in report["all_mechanisms"] if row["id"] == name)["status"]


def test_prepare_takes_the_ordering_selection_and_records_each_fact(tmp_path, inputs):
    facts, contract = inputs
    ws, run = _workspace(tmp_path, facts)
    inventory = {"contract_sha256": "a" * 64,
                 "public_modules": [{"module": "merlin.compile.scheduling.effect_cleanup"}]}
    S.prepare(ws, run, repo=REPO, inventory=inventory, contract_path=contract)
    without = json.loads((ws / "OPTIMIZATION_SUGGESTIONS.json").read_text())
    assert _status(without, "barrier_cleanup") == "unknown"
    assert without["capability_provenance"]["ordering"]["status"] == "unavailable"
    pin = _selection(tmp_path, ws / "rtl_facts/facts.json", contract)
    S.prepare(ws, run, repo=REPO, inventory=inventory, contract_path=contract, ordering=pin)
    report = json.loads((ws / "OPTIMIZATION_SUGGESTIONS.json").read_text())
    assert _status(report, "barrier_cleanup") == "supported"
    derived_rows = report["capability_provenance"]["derived"]
    assert derived_rows["scheduling.dependency_tokens"]["input"]["selection"] == pin
    assert report["input_pins"]["command_ordering_selection_sha256"] == pin["sha256"]
    capabilities = json.loads((ws / "OPTIMIZATION_CAPABILITIES.json").read_text())
    assert capabilities["traits"]["scheduling.dependency_tokens"] == "supported"
    assert capabilities["provenance"] == report["capability_provenance"]
    for root in (ws, run):
        staged = root / "COMMAND_ORDERING_SELECTION.json"
        assert staged.read_bytes() == Path(pin["path"]).read_bytes()
        assert C._sha(staged) == pin["sha256"]
        assert staged.stat().st_mode & 0o222 == 0


def test_the_guidance_passes_the_runs_ordering_pin(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(joint_control, "selected", lambda: object())
    monkeypatch.setattr(S, "prepare", lambda *a, **k: seen.update(k) or "")
    pin = {"path": "/abs/selection.json", "sha256": "2" * 64}
    monkeypatch.setenv("MERLIN_COMPLETION_ORDER_SELECTION", json.dumps(pin))
    CL._performance_guidance(tmp_path, tmp_path, {"public_modules": []}, repo=REPO)
    assert seen["ordering"] == pin
    monkeypatch.delenv("MERLIN_COMPLETION_ORDER_SELECTION")
    CL._performance_guidance(tmp_path, tmp_path, {"public_modules": []}, repo=REPO)
    assert seen["ordering"] is None


def test_next_steps_rank_fact_supported_mechanisms_ahead_of_unconfirmed_ones():
    catalog = json.loads((REPO / "docs/design/joint_optimization_catalog.json").read_text())
    modules = [entry["compiler"]["module"] for entry in catalog["mechanisms"]]
    base = {"compute.systolic": "supported", "memory.scratchpad": "supported", "memory.accumulator": "supported",
            "memory.explicit_local_storage": "supported", "memory.local_memory": "supported",
            "memory.explicit_dma": "supported", "submission.cpu_issued": "supported"}

    def ranked(capabilities):
        selection = S.select(catalog, capabilities=capabilities, public_modules=modules, limit=64)
        return N.mechanism_candidates(selection, N.FENCE_FACETS, N.FENCE_TERMS, "fence", {})

    before = ranked(base)
    values, _, _ = derived()
    after = ranked({**base, **values})
    order = [c["row"]["id"] for c in after]
    cleanup = next(c for c in after if c["row"]["id"] == "barrier_cleanup")
    assert cleanup["unconfirmed"] == []
    assert next(c for c in before if c["row"]["id"] == "barrier_cleanup")["unconfirmed"]
    assert order.index("barrier_cleanup") < [c["row"]["id"] for c in before].index("barrier_cleanup")
    # Within the mechanisms that name the bottleneck, every confirmed one precedes every unconfirmed one.
    named = [bool(c["unconfirmed"]) for c in after if not c["rank"][0]]
    assert named == sorted(named)


def test_invalid_ordering_selection_is_not_exposed_to_the_author(tmp_path, inputs):
    facts, contract = inputs
    ws, run = _workspace(tmp_path, facts)
    pin = _selection(tmp_path, ws / "rtl_facts/facts.json", contract, ordering_sha256="0" * 64)
    S.prepare(ws, run, repo=REPO, inventory={"contract_sha256": "a" * 64, "public_modules": []},
              contract_path=contract, ordering=pin)
    assert not (ws / "COMMAND_ORDERING_SELECTION.json").exists()
    assert not (run / "COMMAND_ORDERING_SELECTION.json").exists()
