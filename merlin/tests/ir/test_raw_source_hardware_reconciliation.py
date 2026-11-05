"""Raw unknown counts require retained actual-source negative hardware joins."""

from copy import deepcopy

import pytest
from test_source_payload_roles import select_member
from xdsl.dialects.builtin import StringAttr

from merlin.common import mlir_query as mq
from merlin.targetgen import eligibility as E
from merlin.targetgen import model_coverage as MC
from merlin.targetgen.compute_units import SemanticCapability
from merlin.xdsl_dialects.lowering import compute_groups as CG

D = pytest.importorskip("merlin_experiments.phase1.feedback.private_source_demand")


def original(monkeypatch, dtype="f32"):
    caps = {"elementwise_map": SemanticCapability("elementwise_map", dtypes=("int8",))}
    monkeypatch.setattr(E, "capability_map_from_contract", lambda contract: caps)
    member = select_member(dtype)
    member.attributes["prov.family"] = StringAttr("elementwise")
    module = member
    while module.parent_op() is not None:
        module = module.parent_op()
    operations = MC.region_ops(module)
    verdicts = [E.is_eligible(region, caps) for region in MC.regions_from_module(module)]
    assert len(operations) == 1 and verdicts[0].undetermined
    group = CG.Group(0, CG.HOST, None, members=[member], stages=[CG.ELEMENTWISE], in_dtype="int8")
    return [group], operations, verdicts, tuple(mq.walk(module))


def derive(groups, operations, verdicts, parsed):
    return D.derive(groups, operations, verdicts, source_sha256="a" * 64, normalized_sha256="b" * 64,
                    capability={"original": "public integer declaration"}, software={},
                    facts_sha256="c" * 64, numeric_policy_sha256="d" * 64, source_operations=parsed)


def test_actual_source_join_resolves_effective_count_without_rewriting_raw_unknown(monkeypatch):
    record = derive(*original(monkeypatch))
    assert record["n_raw_undetermined_regions"] == 1 and record["n_unresolved_raw_regions"] == 0
    row, = record["raw_hardware_reconciliation"]["rows"]
    assert row["actual_typed_hardware"]["input_formats"] == ["fp32", "fp32"]
    assert row["actual_typed_hardware"]["refusal"] == "input_dtype"
    assert len(row["operands"]) == 4 and row["source_ordinal"] >= 0
    assert all("block_owner" in value for value in row["operands"])
    D.require_raw_reconciliation(record)


def test_missing_or_foreign_source_join_never_resolves_unknown(monkeypatch):
    groups, operations, verdicts, parsed = original(monkeypatch)
    with pytest.raises(D.SourceDemandIncomplete) as missing:
        derive(groups, operations, verdicts, ())
    assert missing.value.receipt["n_unresolved_raw_regions"] == 1
    foreign = original(monkeypatch)[3]
    with pytest.raises(ValueError, match="absent from the original source"):
        derive(groups, operations, verdicts, foreign)
    with pytest.raises(ValueError, match="complete original parsed"):
        derive(groups, operations, verdicts, parsed[1:])


def test_supported_integer_payload_never_becomes_a_negative_exclusion(monkeypatch):
    with pytest.raises(D.SourceDemandIncomplete) as error:
        derive(*original(monkeypatch, "i8"))
    record = error.value.receipt
    assert record["n_raw_undetermined_regions"] == record["n_unresolved_raw_regions"] == 1
    assert record["raw_hardware_reconciliation"]["rows"] == []
    assert record["n_unknown_host_groups"] == 1


@pytest.mark.parametrize("change", ["missing", "source", "facts", "numeric", "negative", "ordinal"])
def test_link_refuses_missing_stale_or_mutated_negative_evidence(monkeypatch, change):
    record = deepcopy(derive(*original(monkeypatch)))
    if change == "missing":
        record.pop("raw_hardware_reconciliation")
    elif change == "source":
        record["normalized_source_sha256"] = "e" * 64
    elif change in {"facts", "numeric"}:
        key = "rtl_facts" if change == "facts" else "numeric_policy"
        record["declarations"][key] = "e" * 64
    elif change == "negative":
        record["raw_hardware_reconciliation"]["rows"][0]["actual_typed_hardware"]["status"] = "admitted"
    else:
        record["raw_hardware_reconciliation"]["rows"][0]["source_ordinal"] = -1
    with pytest.raises(ValueError, match="not completely reconciled"):
        D.link({"accelerator_source_demand": record}, {}, 0)


def test_linked_consumer_requires_the_same_retained_raw_evidence(monkeypatch):
    record = derive(*original(monkeypatch))
    source = {"accelerator_source_demand": record, "source_sha256": "a" * 64,
              "normalized_source_sha256": "b" * 64, "eligible_group_provenance": []}
    linked = {"capture_tree_sha256": "e" * 64, "elf_sha256": "f" * 64, "candidate_tree_sha256": "1" * 64}
    D.link(source, linked, 0)
    entry = {**linked, "linked_device_groups": 0}
    assert D.linked_complete(source, entry, "1" * 64)
    record["raw_hardware_reconciliation"]["rows"] = []
    assert not D.linked_complete(source, entry, "1" * 64)
