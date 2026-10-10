"""What the ARR coverage certificate is evidence OF.

Every number in the certificate is derived from the routing PLAN, which lists what the router assigned
and not what ran. That is the same conflation already fixed on the lane side, where one submission
assigned 15 matmuls to the mesh and fell back on all 15 at run time -- and the certificate, which is the
surface people actually quote "how much of what it could accelerate did it accelerate" from, had never
been told about it.

So it must state its own evidence, and carry the run beside the plan when a run exists.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace

from merlin.targetgen import coverage_certificate as CC
from merlin.targetgen.compute_units import SemanticCapability
from merlin.targetgen.routing import OpDemand, RouteResult

# A plan with no regions: these tests are about the evidence FIELDS, which must be present and honest
# whatever the regions say. The recall arithmetic itself is exercised elsewhere.
_EMPTY_PLAN = {"mesh": [], "fallback": [], "scalar_rvv": [], "results": []}


def test_the_certificate_states_that_its_numbers_are_plan_derived():
    assert CC.build(_EMPTY_PLAN, {}, target="t")["arr_evidence"] == "routing_plan"


def test_without_an_execution_record_the_crosscheck_is_absent_not_agreeing():
    assert CC.build(_EMPTY_PLAN, {}, target="t")["execution_crosscheck"] is None


def test_a_plan_the_run_did_not_carry_out_is_reported_as_disagreeing():
    """15 assigned, 0 executed: the recalls describe an intent that did not happen."""
    x = CC.build(
        _EMPTY_PLAN,
        {},
        target="t",
        execution={"matmul_layers_routed": 15, "matmul_layers_on_mesh": 0, "matmul_layers_host_fallback": 15},
    )["execution_crosscheck"]
    assert x["agrees"] is False
    assert x["matmul_layers_on_mesh"] == 0 and x["matmul_layers_host_fallback"] == 15


def test_a_plan_the_run_carried_out_agrees():
    x = CC.build(
        _EMPTY_PLAN,
        {},
        target="t",
        execution={"matmul_layers_routed": 15, "matmul_layers_on_mesh": 15, "matmul_layers_host_fallback": 0},
    )["execution_crosscheck"]
    assert x["agrees"] is True


def test_an_unknown_count_leaves_agreement_undecided_rather_than_true():
    """`UNKNOWN` is a sentinel string, not a number, and "nobody could tell" is not "they agree"."""
    x = CC.build(
        _EMPTY_PLAN, {}, target="t", execution={"matmul_layers_routed": "UNKNOWN", "matmul_layers_on_mesh": 3}
    )["execution_crosscheck"]
    assert x["agrees"] is None


# --- the execution-evidenced half ---------------------------------------------------------------------
# The plan-derived recalls are per REGION; there is no join from a region to a completed call. But both
# `mesh_route_symbols` and the dispatch ledger key on the KERNEL SYMBOL, so "which assigned kernel did
# not run on the accelerator" is answerable without inventing the join that does not exist.


def _exec(routed, ran):
    return {
        "mesh_route_symbols": list(routed),
        "dispatch_ledger": [
            {"ordinal": i, "symbol": s, "lane": "on_mesh", "status": "pass"} for i, s in enumerate(ran)
        ],
    }


def test_an_assigned_kernel_that_never_ran_on_the_accelerator_is_named():
    got = CC.executed_false_fallbacks(_exec(["k$kernel_0", "k$kernel_1"], ["k$kernel_0"]))
    assert got["status"] == "measured"
    assert got["n_routed"] == 2 and got["n_executed_on_accelerator"] == 1
    assert got["false_fallback_symbols"] == ["k$kernel_1"], "a count alone is not actionable"


def test_every_assigned_kernel_running_there_is_no_false_fallback():
    got = CC.executed_false_fallbacks(_exec(["a", "b"], ["a", "b"]))
    assert got["n_false_fallback"] == 0 and got["false_fallback_symbols"] == []


def test_a_missing_ledger_is_not_measured_rather_than_no_fallbacks():
    """ "No symbols fell back" and "nobody recorded which symbols ran" are opposite conclusions."""
    assert CC.executed_false_fallbacks({"mesh_route_symbols": ["a"]})["status"] == "not_measured"
    assert CC.executed_false_fallbacks({"dispatch_ledger": []})["status"] == "not_measured"
    assert CC.executed_false_fallbacks(None)["status"] == "not_measured"


def test_a_call_that_did_not_pass_does_not_count_as_having_run_there():
    ex = {
        "mesh_route_symbols": ["a"],
        "dispatch_ledger": [{"ordinal": 0, "symbol": "a", "lane": "on_mesh", "status": "fail"}],
    }
    assert CC.executed_false_fallbacks(ex)["n_false_fallback"] == 1


def test_the_certificate_carries_it_beside_the_plan_derived_recalls():
    cert = CC.build(_EMPTY_PLAN, {}, target="t", execution=_exec(["a", "b"], ["a"]))
    assert cert["executed_false_fallbacks"]["n_false_fallback"] == 1
    assert cert["arr_evidence"] == "routing_plan", "the recalls themselves are still plan-derived"


def test_source_region_execution_exposes_a_planned_acceleration_that_ran_on_host():
    demand = OpDemand(op="add", in_fmt="int8", family="elementwise_map", region_id="add_0", carrier_op="linalg.generic")
    route = RouteResult(demand, unit="tensor_unit", acc=None, gap=None)
    plan = {"results": [route], "mesh": [route], "fallback": [], "scalar_rvv": []}
    execution = {
        "outlined_dispatches": [
            {"symbol": "forward$kernel_4__radd_0", "region_id": "add_0", "root_op": "linalg.generic", "prov_op": "add"}
        ],
        "dispatch_ledger": [
            {"ordinal": 0, "symbol": "forward$kernel_4__radd_0", "lane": "native_cpu", "status": "pass"}
        ],
    }
    cert = CC.build(
        plan,
        {"elementwise_map": SemanticCapability(family="elementwise_map", dtypes=("int8",))},
        execution=execution,
    )
    observed = cert["source_region_execution"]
    assert observed["status"] == "measured"
    assert observed["outline_inventory_status"] == "matched"
    assert len(observed["outlined_dispatches_sha256"]) == 64
    assert len(observed["dispatch_ledger_sha256"]) == 64
    assert observed["eligible_host_region_ids"] == ["add_0"]
    assert observed["n_eligible_executed_on_accelerator"] == 0
    execution["dispatch_ledger"][0]["lane"] = "scalar_rvv_lane"
    scalar = CC.build(
        plan,
        {"elementwise_map": SemanticCapability(family="elementwise_map", dtypes=("int8",))},
        execution=execution,
    )["source_region_execution"]
    assert scalar["eligible_host_region_ids"] == ["add_0"]


def test_requested_precision_is_not_a_verified_capture_conversion():
    demand = OpDemand(op="matmul", in_fmt="int8", elem_fmt="fp32", family="contraction")
    route = RouteResult(demand, unit="tensor_unit", acc=None, gap=None)
    plan = {"results": [route], "mesh": [route], "fallback": [], "scalar_rvv": []}
    cert = CC.build(plan, {"contraction": SemanticCapability(family="contraction", dtypes=("int8",))}, linalg_mlir="")
    assert cert["source_mlir_sha256"] == hashlib.sha256(b"").hexdigest()
    assert cert["n_requested_format_mismatches"] == 1
    assert cert["n_precision_transform_obligations"] == 0
    assert cert["accelerated_ineligible_count"] == 1
    unknown = OpDemand(op="add", in_fmt="int8", family="elementwise_map", carrier_op="linalg.generic")
    unknown_route = RouteResult(unknown, unit="tensor_unit", acc=None, gap=None)
    unknown_plan = {"results": [unknown_route], "mesh": [unknown_route], "fallback": [], "scalar_rvv": []}
    unknown_cert = CC.build(
        unknown_plan, {"elementwise_map": SemanticCapability(family="elementwise_map", dtypes=("int8",))}
    )
    assert unknown_cert["n_unknown_capture_formats"] == 1
    assert unknown_cert["precision_transform_verification"]["status"] == "unknown_capture_format"


def test_typed_host_source_operations_have_a_complete_precision_census():
    """Frozen typed scalar/control/result values are evidence, not a target capability."""
    from merlin.targetgen.capsule_source import model_op_demands_checked
    from merlin.targetgen.routing import route_plan_on

    source = '''"builtin.module"() ({
      "func.func"() <{sym_name = "forward", function_type = (tensor<2xf32>) -> tensor<2xi8>}> ({
      ^bb0(%x: tensor<2xf32>):
        %scale = "arith.constant"() <{value = 1.000000e+00 : f32}>
          {prov.op = "quantize", prov.family = "quantize"} : () -> f32
        %s = "tensor.splat"(%scale) {prov.op = "quantize", prov.family = "quantize"} : (f32) -> tensor<f32>
        %zero = "arith.constant"() <{value = 0 : i64}> {prov.op = "quantize", prov.family = "quantize"} : () -> i64
        %z = "tensor.splat"(%zero) {prov.op = "quantize", prov.family = "quantize"} : (i64) -> tensor<i64>
        %q = "quant_ext.quantize_per_tensor"(%x, %s, %z)
          <{quant_min = -128 : i64, quant_max = 127 : i64}>
          {prov.op = "quantize", prov.family = "quantize", prov.region_id = "q0"}
          : (tensor<2xf32>, tensor<f32>, tensor<i64>) -> tensor<2xi8>
        "func.return"(%q) : (tensor<2xi8>) -> ()
      }) : () -> ()
    }) : () -> ()'''
    demands = model_op_demands_checked(source, "int8")
    assert len(demands) == 5
    plan = route_plan_on(demands, [])
    certificate = CC.build(plan, {}, linalg_mlir=source)
    assert certificate["n_eligible"] == 0
    assert certificate["n_unknown_capture_formats"] == 0
    assert certificate["n_precision_transform_obligations"] == 0
    assert certificate["precision_transform_verification"]["status"] == "not_required"
    assert all(row["target_eligible"] is False for row in certificate["regions"])
    assert certificate["regions"][4]["source_precision_witness"]["operand_types"] == [
        "tensor<2xf32>", "tensor<f32>", "tensor<i64>"
    ]

    # A typed operation may not be lent to another demand by a reordered plan.
    reordered = route_plan_on([demands[1], demands[0], *demands[2:]], [])
    assert CC.build(reordered, {}, linalg_mlir=source)["n_unknown_capture_formats"] == 5

    # A dynamic tensor shape is not a complete source type witness.
    dynamic = source.replace("tensor<2xf32>", "tensor<?xf32>")
    dynamic_demands = model_op_demands_checked(dynamic, "int8")
    dynamic_cert = CC.build(route_plan_on(dynamic_demands, []), {}, linalg_mlir=dynamic)
    assert dynamic_cert["n_unknown_capture_formats"] > 0

    # A known bf16 source remains a known source type, not host admission.
    bf16 = source.replace("f32", "bf16")
    bf16_demands = model_op_demands_checked(bf16, "int8")
    assert len(bf16_demands) == len(demands)
    bf16_cert = CC.build(route_plan_on(bf16_demands, []), {}, linalg_mlir=bf16)
    assert bf16_cert["n_unknown_capture_formats"] == 0
    assert "bf16" in bf16_cert["regions"][4]["source_precision_witness"]["operand_types"][0]


def test_typed_source_cannot_clear_unverified_accelerator_contraction():
    from merlin.targetgen.capsule_source import model_op_demands_checked
    from merlin.targetgen.routing import route_plan_on

    source = '''builtin.module {
      func.func @forward(%a: tensor<2x2xi8>, %b: tensor<2x2xi8>, %c: tensor<2x2xi32>) -> tensor<2x2xi32> {
        %r = "linalg.matmul"(%a, %b, %c)
          {prov.op = "matmul", prov.family = "contraction", prov.region_id = "m0"}
          : (tensor<2x2xi8>, tensor<2x2xi8>, tensor<2x2xi32>) -> tensor<2x2xi32>
        func.return %r : tensor<2x2xi32>
      }
    }'''
    parsed = model_op_demands_checked(source, "int8")
    assert len(parsed) == 1 and parsed[0].captured_input_formats == ("int8", "int8")
    incomplete = replace(parsed[0], captured_input_formats=(None, "int8"))
    plan = route_plan_on([incomplete], [])
    supported = {"contraction": SemanticCapability(family="contraction", dtypes=("int8",))}
    guarded = CC.build(plan, supported, linalg_mlir=source)
    assert guarded["n_eligible"] == 0
    assert guarded["n_unknown_capture_formats"] == 1
    assert "source_precision_witness" not in guarded["regions"][0]

    # Without a declared contraction capability, the same source is a
    # definitive hardware noncandidate; this does not admit host semantics.
    unsupported = CC.build(plan, {}, linalg_mlir=source)
    assert unsupported["n_unknown_capture_formats"] == 0
    assert unsupported["regions"][0]["source_precision_witness"]["status"] == "typed_source"


def test_mixed_source_closes_only_when_every_possible_tensor_input_is_hardware_refused():
    from merlin.targetgen.capsule_source import model_op_demands_checked
    from merlin.targetgen.routing import route_plan_on

    # The index operand's element type must name no registered storage format (``int64`` is one), so
    # the census sees an operand whose capture format it cannot read.
    source = '''builtin.module {
      func.func @forward(%idx: tensor<2xi48>, %weight: tensor<2xf32>) -> tensor<2xf32> {
        %r = "linalg.generic"(%idx, %weight)
          {prov.op = "embedding", prov.family = "gather_scatter", prov.region_id = "g0"}
          : (tensor<2xi48>, tensor<2xf32>) -> tensor<2xf32>
        func.return %r : tensor<2xf32>
      }
    }'''
    demands = model_op_demands_checked(source, "int8")
    assert len(demands) == 1 and demands[0].captured_input_formats == (None,)
    plan = route_plan_on(demands, [])
    int8_copy = {"movement": SemanticCapability(family="movement", dtypes=("int8",), forms=("copy",))}
    refused = CC.build(plan, int8_copy, linalg_mlir=source)
    assert refused["n_unknown_capture_formats"] == 0
    assert refused["regions"][0]["source_precision_witness"]["operand_element_types"] == ["i48", "f32"]

    # One supported tensor format is enough to keep eligibility unresolved;
    # the census may not decide which of mixed index/data operands is hardware.
    f32_copy = {"movement": SemanticCapability(family="movement", dtypes=("fp32",), forms=("copy",))}
    possible = CC.build(plan, f32_copy, linalg_mlir=source)
    assert possible["n_unknown_capture_formats"] == 1
    assert "source_precision_witness" not in possible["regions"][0]


def test_unconverted_elementwise_is_not_eligible_at_requested_integer_format():
    demand = OpDemand(op="mul", in_fmt="int8", elem_fmt="fp32", family="elementwise_map")
    route = RouteResult(demand, unit=None, acc=None, gap="no int8 execution of captured fp32 op")
    plan = {"results": [route], "mesh": [], "fallback": [], "scalar_rvv": [route]}
    cert = CC.build(plan, {"elementwise_map": SemanticCapability(family="elementwise_map", dtypes=("int8",))})
    assert cert["n_eligible"] == 0
    assert cert["false_fallback_count"] == 0
    assert cert["n_precision_transform_obligations"] == 0


def test_copy_capability_does_not_make_permutation_eligible():
    from merlin.targetgen.eligibility import RegionDescriptor, is_eligible

    cap = {"movement": SemanticCapability(family="movement", dtypes=("int8",), forms=("copy",))}
    verdict = is_eligible(RegionDescriptor(op="permute", family="movement", in_dtype="int8", form="permutation"), cap)
    assert verdict.eligible is False
    assert verdict.refusal == "form"


def test_source_region_execution_refuses_unattributed_or_mixed_execution():
    demand = OpDemand(op="add", in_fmt="int8", family="elementwise_map", region_id="add_0", carrier_op="linalg.generic")
    route = RouteResult(demand, unit="tensor_unit", acc=None, gap=None)
    plan = {"results": [route], "mesh": [route], "fallback": [], "scalar_rvv": []}
    caps = {"elementwise_map": SemanticCapability(family="elementwise_map", dtypes=("int8",))}
    spoofed = CC.build(
        plan,
        caps,
        execution={
            "outlined_dispatches": [
                {
                    "symbol": "forward$kernel_0__radd_0",
                    "region_id": "add_0",
                    "root_op": "linalg.generic",
                    "prov_op": "add",
                }
            ],
            "dispatch_ledger": [{"symbol": "xnn__radd_0", "lane": "on_mesh", "status": "pass"}],
        },
    )["source_region_execution"]
    assert spoofed["status"] == "incomplete"
    assert spoofed["unobserved_region_ids"] == ["add_0"]
    mixed = CC.build(
        plan,
        caps,
        execution={
            "outlined_dispatches": [
                {
                    "symbol": "forward$kernel_0__radd_0",
                    "region_id": "add_0",
                    "root_op": "linalg.generic",
                    "prov_op": "add",
                },
                {
                    "symbol": "forward$kernel_1__radd_0",
                    "region_id": "add_0",
                    "root_op": "linalg.generic",
                    "prov_op": "add",
                },
            ],
            "dispatch_ledger": [
                {"symbol": "forward$kernel_0__radd_0", "lane": "on_mesh", "status": "pass"},
                {"symbol": "forward$kernel_1__radd_0", "lane": "native_cpu", "status": "pass"},
            ],
        },
    )["source_region_execution"]
    assert mixed["status"] == "incomplete"  # extra outlined op also lacks a source correspondence
    assert mixed["eligible_mixed_region_ids"] == ["add_0"]


def test_source_region_requires_every_same_provenance_operation_and_completed_symbol():
    demands = [
        OpDemand(op="add", in_fmt="int8", family="elementwise_map", region_id="add_0", carrier_op="linalg.generic"),
        OpDemand(op="mul", in_fmt="int8", family="elementwise_map", region_id="add_0", carrier_op="linalg.generic"),
    ]
    routes = [RouteResult(d, unit="tensor_unit", acc=None, gap=None) for d in demands]
    plan = {"results": routes, "mesh": routes, "fallback": [], "scalar_rvv": []}
    caps = {"elementwise_map": SemanticCapability(family="elementwise_map", dtypes=("int8",))}
    one_symbol = "forward$kernel_0__radd_0"
    two_symbols = [one_symbol, "forward$kernel_1__radd_0"]

    def execution(symbols, completed):
        return {
            "outlined_dispatches": [
                {"symbol": symbol, "region_id": "add_0", "root_op": "linalg.generic", "prov_op": op}
                for symbol, op in symbols
            ],
            "dispatch_ledger": [
                {"ordinal": index, "symbol": symbol, "lane": "on_mesh", "status": "pass"}
                for index, symbol in enumerate(completed)
            ],
        }

    missing_outline = CC.build(plan, caps, execution=execution([(one_symbol, "add")], [one_symbol]))[
        "source_region_execution"
    ]
    assert missing_outline["status"] == "incomplete"
    assert missing_outline["unmatched_source_operation_counts"] == {"add_0|linalg.generic|mul": 1}
    outlined = [(two_symbols[0], "add"), (two_symbols[1], "mul")]
    missing_call = CC.build(plan, caps, execution=execution(outlined, [one_symbol]))["source_region_execution"]
    assert missing_call["status"] == "incomplete"
    assert missing_call["unexecuted_outlined_symbols"] == [two_symbols[1]]
    complete = CC.build(plan, caps, execution=execution(outlined, two_symbols))["source_region_execution"]
    assert complete["status"] == "measured"
    assert complete["n_eligible_executed_on_accelerator"] == 1


def test_contraction_split_requires_both_children_but_only_contraction_child_on_device():
    demand = OpDemand(
        op="batch_matmul",
        in_fmt="int8",
        weight_fmt="int8",
        family="contraction",
        region_id="matmul_0",
        carrier_op="linalg.generic",
    )
    route = RouteResult(demand, unit="tensor_unit", acc="i32", gap=None)
    plan = {"results": [route], "mesh": [route], "fallback": [], "scalar_rvv": []}
    caps = {"contraction": SemanticCapability(family="contraction", dtypes=("int8",))}
    contraction = "forward$kernel_0__rmatmul_0"
    requant = "forward$kernel_1__rmatmul_0"
    outline = [
        {
            "symbol": contraction,
            "root_op": "linalg.generic",
            "prov_op": "batch_matmul",
            "region_id": "matmul_0",
            "prov_role": "contraction",
        },
        {
            "symbol": requant,
            "root_op": "linalg.generic",
            "prov_op": "batch_matmul",
            "region_id": "matmul_0",
            "prov_role": "requant",
        },
    ]
    calls = [
        {"ordinal": 0, "symbol": contraction, "lane": "on_mesh", "status": "pass"},
        {"ordinal": 1, "symbol": requant, "lane": "native_cpu", "status": "pass"},
    ]
    measured = CC.build(plan, caps, execution={"outlined_dispatches": outline, "dispatch_ledger": calls})[
        "source_region_execution"
    ]
    assert measured["status"] == "measured"
    assert measured["eligible_accelerator_region_ids"] == ["matmul_0"]
    assert measured["eligible_mixed_region_ids"] == []
    assert measured["auxiliary_host_symbols"] == [requant]
    missing = CC.build(plan, caps, execution={"outlined_dispatches": outline, "dispatch_ledger": calls[:1]})[
        "source_region_execution"
    ]
    assert missing["status"] == "incomplete"
    assert missing["unexecuted_outlined_symbols"] == [requant]
    missing_child = CC.build(
        plan,
        caps,
        execution={"outlined_dispatches": outline[:1], "dispatch_ledger": calls[:1]},
    )["source_region_execution"]
    assert missing_child["status"] == "incomplete"
    assert missing_child["invalid_split_source_keys"] == ["matmul_0|linalg.generic|batch_matmul"]
    bad_role = [dict(outline[0], prov_role="unknown"), outline[1]]
    unknown = CC.build(
        plan,
        caps,
        execution={"outlined_dispatches": bad_role, "dispatch_ledger": calls},
    )["source_region_execution"]
    assert unknown["status"] == "incomplete"
    assert unknown["invalid_outlined_rows"] == 1


def test_source_region_without_static_outline_inventory_never_qualifies_execution():
    demand = OpDemand(op="add", in_fmt="int8", family="elementwise_map", region_id="add_0", carrier_op="linalg.generic")
    route = RouteResult(demand, unit="tensor_unit", acc=None, gap=None)
    plan = {"results": [route], "mesh": [route], "fallback": [], "scalar_rvv": []}
    cert = CC.build(
        plan,
        {"elementwise_map": SemanticCapability(family="elementwise_map", dtypes=("int8",))},
        execution={"dispatch_ledger": [{"symbol": "forward$kernel_0__radd_0", "lane": "on_mesh", "status": "pass"}]},
    )
    assert cert["source_region_execution"]["status"] == "incomplete"


def test_certificate_uses_the_verified_capture_family_not_only_an_op_spelling():
    demand = OpDemand(
        op="target_neutral_integer_product",
        in_fmt="int8",
        weight_fmt="int8",
        family="contraction",
        region_id="product_0",
        m=2,
        k=8,
        n=8,
    )
    route = RouteResult(demand, unit="tensor_unit", acc="i32", gap=None)
    plan = {"results": [route], "mesh": [route], "fallback": [], "scalar_rvv": []}
    cert = CC.build(plan, {"contraction": SemanticCapability(family="contraction", dtypes=("int8",))})
    assert cert["n_eligible"] == 1
    assert cert["n_eligible_accelerated"] == 1
