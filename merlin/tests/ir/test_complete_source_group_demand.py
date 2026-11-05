"""Every original admitted source group is demanded, including dequantized bodies."""

from copy import deepcopy

import pytest
from fake_quant_layer import Oracle
from fake_quant_layer import module as fake_quant_module
from xdsl.dialects.builtin import ArrayAttr, StringAttr

from merlin.common import mlir_query as mq
from merlin.compile import complete_group_routing, route_before_build
from merlin.llvmlower import device_offload
from merlin.llvmlower.device_build import DeviceRouting
from merlin.llvmlower.source_compute_groups import form_source_groups
from merlin.targetgen import model_coverage as MC
from merlin.targetgen.compute_units import SemanticCapability
from merlin.targetgen.eligibility import is_eligible
from merlin.xdsl_dialects.lowering import compute_groups as CG

demand = pytest.importorskip("merlin_experiments.phase1.feedback.private_source_demand")
provenance = pytest.importorskip("merlin_experiments.phase1.feedback.private_group_provenance")


def original_source():
    module = mq.parse(fake_quant_module(weight_dequantize="per_tensor"))
    groups = form_source_groups(module, "synthetic", oracle=Oracle())
    for group in groups:
        if group.root is not None:
            group.root.attributes["prov.region_id"] = StringAttr(f"public-region-{group.index}")
            group.root.attributes["prov.source_node_ids"] = ArrayAttr([StringAttr(f"public-node-{group.index}")])
    cap_map = {"contraction": SemanticCapability("contraction", dtypes=("int8",))}
    operations = MC.region_ops(module)
    verdicts = [is_eligible(region, cap_map) for region in MC.regions_from_module(module)]
    return groups, operations, verdicts


def derive(groups, operations, verdicts):
    return demand.derive(groups, operations, verdicts, source_sha256="a" * 64, normalized_sha256="b" * 64,
                         capability={"declared": "integer"}, software={"selected_numeric": "original"},
                         facts_sha256="c" * 64, numeric_policy_sha256="d" * 64)


def test_real_dequant_source_composition_remains_in_complete_demand():
    groups, operations, verdicts = original_source()
    assert len(groups) == 1 and groups[0].in_dtype == "int8"
    assert not any(verdict.eligible for verdict in verdicts)  # raw operands are floating
    receipt = derive(groups, operations, verdicts)
    assert receipt["n_demanded_groups"] == 1 and receipt["n_raw_eligible_regions"] == 0
    assert receipt["demanded_group_provenance"][0]["source_region_id"] == "public-region-0"


@pytest.mark.parametrize("refusal", ["no_device_form", "readout_epilogue_undeclared", "missing_scale",
                                    "readout_requires_scale", None])
def test_compiler_form_or_unknown_readout_is_never_a_hardware_host_exclusion(refusal, monkeypatch):
    from merlin.targetgen import eligibility

    caps = {family: SemanticCapability(family, dtypes=("int8",))
            for family in ("contraction", "elementwise_map")}
    monkeypatch.setattr(eligibility, "capability_map_from_contract", lambda contract: caps)
    groups, operations, verdicts = original_source()
    groups[0].placement, groups[0].refusal = CG.HOST, refusal
    with pytest.raises(demand.SourceDemandIncomplete) as error:
        derive(groups, operations, verdicts)
    assert error.value.receipt["n_unknown_host_groups"] == 1


def test_host_reason_requires_actual_selected_hardware_exclusion(monkeypatch):
    from merlin.targetgen import eligibility

    caps = {family: SemanticCapability(family, dtypes=("int8",))
            for family in ("contraction", "elementwise_map")}
    monkeypatch.setattr(eligibility, "capability_map_from_contract", lambda contract: caps)
    groups, operations, verdicts = original_source()
    groups[0].placement, groups[0].refusal = CG.HOST, "input_dtype"
    with pytest.raises(demand.SourceDemandIncomplete):
        derive(groups, operations, verdicts)  # a reason cannot exclude the admitted integer body
    parsed = mq.parse(fake_quant_module(weight_dequantize="per_channel", weight_axis=0))
    original_groups = form_source_groups(parsed, "synthetic", oracle=Oracle())
    floating = next(group for group in original_groups if CG.CONTRACTION in group.stages)
    assert floating.in_dtype == "fp32" and floating.refusal == "input_dtype"
    # This isolated group actually contains only floating contraction work.
    # Separate dequantization groups still need their own output-format proof.
    assert demand._hardware_exclusion(floating, {"independent": "selected integer capability"})


def test_complete_group_authority_is_an_original_boolean():
    with pytest.raises(TypeError, match="original Boolean"):
        DeviceRouting("synthetic", "original-package", "int8", "i32", require_complete_groups=1)
    with pytest.raises(ValueError, match="granularity"):
        DeviceRouting("synthetic", "original-package", "int8", "i32", require_complete_groups=True)


def test_complete_demand_cannot_be_replaced_by_the_raw_region_subset_or_skipped_reason():
    groups, operations, verdicts = original_source()
    receipt = derive(groups, operations, verdicts)
    receipt["n_unresolved_raw_regions"] = 0
    source = {"eligible_groups": [0], "eligible_group_provenance": receipt["demanded_group_provenance"],
              "accelerator_source_demand": receipt}
    emitted = {"routed": [], "skipped": [["original", "the placement did not select this group's root"]]}
    with pytest.raises(ValueError):
        provenance.verify_routed_source_coverage(emitted, source)
    emitted["routed"] = [{"group": 7, "source_region_id": "public-region-0", "source_node_ids": ["public-node-0"]}]
    assert provenance.verify_routed_source_coverage(emitted, source) == {7: 0}
    source["eligible_groups"], source["eligible_group_provenance"] = [], []
    with pytest.raises(ValueError, match="independent accelerator demand"):
        provenance.verify_routed_source_coverage(emitted, source)


def test_all_defined_functions_and_ungrouped_control_flow_are_accounted():
    source = fake_quant_module(weight_dequantize="per_tensor")
    second = source[source.index("  func.func"):source.rindex("}")].replace("@forward", "@second")
    both = source[:source.rindex("}")] + second + "}"
    assert len(form_source_groups(mq.parse(both), "synthetic", oracle=Oracle())) == 2
    class IncompleteOracle(Oracle):
        pass
    parsed = mq.parse(source)
    # A group producer that accidentally inventories no computations must refuse.
    from unittest.mock import patch
    with patch.object(CG, "form_groups", return_value=[]), pytest.raises(ValueError, match="membership"):
        form_source_groups(parsed, "synthetic", oracle=IncompleteOracle())


def test_original_full_route_uses_source_group_types_instead_of_raw_placement(monkeypatch):
    monkeypatch.setattr(CG, "TargetOracle", lambda target: Oracle())
    observed = []
    monkeypatch.setattr(complete_group_routing, "_accum_from_facts",
                        lambda target, operand, weight: observed.append((operand, weight)) or "i32")
    from merlin.targetgen import capsule_source
    monkeypatch.setattr(capsule_source, "model_op_demands_checked",
                        lambda *args: pytest.fail("raw-format placement must not narrow complete source demand"))
    selected = route_before_build.plan_before_build(
        "synthetic", fake_quant_module(weight_dequantize="per_tensor"), datapath="int8",
        device_package="original-package", granularity="group", require_all_accelerator_groups=True,
    )["device_routing"]
    assert observed == [("int8", "int8")] and selected.require_complete_groups is True
    assert selected.select(None) is True and selected.granularity == "group"


def test_original_negative_result_scope_ends_before_complete_group_routing(monkeypatch):
    """Typed host exclusions are not re-admitted by the later composition planner."""
    import sys
    from hashlib import sha256
    from types import ModuleType

    from merlin.targetgen import capsule_source
    from merlin.targetgen import eligibility as E
    from merlin.targetgen import operation_accounting as OA

    hardware = pytest.importorskip("merlin_experiments.phase1.feedback.private_source_member_hardware")
    caps = {"contraction": SemanticCapability("contraction", dtypes=("int8",)),
            "elementwise_map": SemanticCapability("elementwise_map", dtypes=("int8",),
                                                   composed_with=("contraction",))}
    calls = []
    contract, facts = b"public original contract", b"public original facts"

    class Proof:
        def record(self):
            return {"target": "synthetic", "contract_sha256": sha256(contract).hexdigest(),
                    "facts_sha256": sha256(facts).hexdigest()}

        def refusal(self, family, output, selected):
            assert selected == caps
            calls.append((family, output))
            return {"negative_source_proof": True} if output == "fp32" else None

        def revalidate_sources(self):
            pass

    evidence = ModuleType("merlin.targetgen.rtl.result_format_ceiling")
    evidence.NativeResultFormatCeiling = Proof
    monkeypatch.setitem(sys.modules, evidence.__name__, evidence)
    monkeypatch.setattr(E, "capability_map_from_contract", lambda contract: caps)
    original_ask = CG.TargetOracle.ask

    class TypedOracle(Oracle):
        def __init__(self):
            super().__init__()
            self._E, self.cap_map, self.undetermined, self.providers = E, caps, (), {}

        def ask(self, **question):
            # Exercise the production composition descriptor with the fixture's
            # independently declared readout granularity rather than a registry.
            typed = original_ask(self, **{**question, "granularity": None})
            declared = super().ask(**question)
            return typed if not typed.admitted else declared

    monkeypatch.setattr(CG, "TargetOracle", lambda target: TypedOracle())
    source = fake_quant_module(weight_dequantize="per_tensor")
    parsed = mq.parse(source)
    member = next(op for op in parsed.walk() if mq.op_name(op) == "quant_ext.dequantize_per_tensor")
    with E.original_result_format_scope(Proof(), target="synthetic", original_contract_bytes=contract,
                                        original_facts_bytes=facts):
        assert hardware.observe(member, CG.DEQUANTIZE, {})["refusal"] == "result_dtype"
        assert len(form_source_groups(parsed, "synthetic")) == 1
    assert ("elementwise_map", "fp32") in calls
    assert any(output is None for _, output in calls)
    monkeypatch.setattr(OA, "admit_operation_row", lambda *args, **kwargs: pytest.fail("typed re-admission"))
    monkeypatch.setattr(hardware, "observe", lambda *args, **kwargs: pytest.fail("typed re-admission"))
    monkeypatch.setattr(capsule_source, "model_op_demands_checked",
                        lambda *args: pytest.fail("raw placement must not narrow complete source demand"))
    monkeypatch.setattr(complete_group_routing, "_accum_from_facts", lambda *args: "i32")
    selected = route_before_build.plan_before_build(
        "synthetic", source, datapath="int8", device_package="original-package",
        granularity="group", require_all_accelerator_groups=True,
    )["device_routing"]
    assert selected.require_complete_groups is True and selected.select(None) is True


def test_required_rewrite_refuses_before_compilation_when_one_requested_group_has_no_form(monkeypatch):
    from merlin.llvmlower import device_shim
    from merlin.xdsl_dialects.lowering import group_command

    monkeypatch.setattr(device_shim, "kernel_abi_for", lambda target: None)
    def refuse(*args, **kwargs):
        raise CG.NoCapsuleForm("original group cannot be stated")
    monkeypatch.setattr(group_command, "program", refuse)
    source = fake_quant_module(weight_dequantize="per_tensor")
    legacy = device_offload.rewrite_groups_to_device(mq.parse(source), "synthetic", select=lambda shape: True,
                                                   oracle=Oracle(), weight_args={1})
    assert not legacy.routed and legacy.skipped
    with pytest.raises(ValueError, match="omitted or refused"):
        device_offload.rewrite_groups_to_device(mq.parse(source), "synthetic", select=lambda shape: True,
                                               oracle=Oracle(), weight_args={1}, require_complete_groups=True)


def test_linked_receipt_requires_same_full_source_candidate_and_denominator():
    groups, operations, verdicts = original_source()
    receipt = derive(groups, operations, verdicts)
    receipt["n_unresolved_raw_regions"] = 0
    source = {"source_sha256": "a" * 64, "normalized_source_sha256": "b" * 64,
              "eligible_group_provenance": deepcopy(receipt["demanded_group_provenance"]),
              "accelerator_source_demand": receipt}
    linked = {"capture_tree_sha256": "e" * 64, "elf_sha256": "f" * 64, "candidate_tree_sha256": "1" * 64}
    selected = DeviceRouting("synthetic", "original-package", "int8", "i32", granularity="group",
                             require_complete_groups=True, select=complete_group_routing.select_complete_group)
    demand.request(source, selected)
    assert receipt["n_requested_groups"] == 1
    demand.link(source, linked, 1)
    entry = {**linked, "linked_device_groups": 1}
    assert demand.linked_complete(source, entry, "1" * 64)
    assert not demand.linked_complete(source, entry, "2" * 64)
    assert not demand.linked_complete(source, {**entry, "linked_device_groups": 0}, "1" * 64)
    assert not demand.linked_complete({key: value for key, value in source.items()
                                      if key != "accelerator_source_demand"}, entry, "1" * 64)
