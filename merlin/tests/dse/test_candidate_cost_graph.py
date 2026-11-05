"""Conditional cost composition does not qualify synthetic observations."""

from copy import deepcopy

from merlin.perf.activity_schedule import ActivityEvent, ActivityTimeline, ScheduledEvent
from merlin.perf.candidate_cost_graph import compose_candidate_cost_graph, observe_candidate_cost_graph
from merlin.xdsl_dialects.lowering.global_plan import CycleInterval


def observed_graph(*, dependency=True):
    # Pure producer boundary fixture; these dictionaries issue no source/hardware authority.
    diagnostics = {
        "source_emission_join": {"status": "verified", "host_activity": {"status": "derived",
            "task_activity_coverage": "complete", "tasks": [
                {"task": str(i), "static_operations": {"integer_arithmetic": 1},
                 "dynamic_operations": {"integer_arithmetic": 3},
                 "load_payload_bytes": 4, "store_payload_bytes": 4} for i in range(2)]}},
        "dependency_stages": {"status": "verified_source_emission", "stages": [
            {"task_index": i, "source_op_indices": [i], "reads": ["a"], "writes": ["b"],
             "instruction_start": i, "instruction_end": i + 1} for i in range(2)]},
        "source": {"operations": [
            {"source_op_index": i, "ordinal": i, "operation": "arith.addi",
             "dependencies": [0] if i == 1 and dependency else [],
             "inputs": [{"shape": [2], "dtype": "i32"}]} for i in range(2)]},
    }
    return observe_candidate_cost_graph(diagnostics, command_buffer={"commands": [], "tensors": {}})


def test_owned_host_dependencies_and_interval_serial_premise_add():
    graph = observed_graph()
    assert graph["stages"][1]["depends_on"] == ["task:0"]
    assert graph["stages"][0]["host_effects"]["load_payload_bytes"] == 4
    interval = compose_candidate_cost_graph(graph, stage_prices={
        "task:0": CycleInterval(3, 4, provenance=("external first price",)),
        "task:1": CycleInterval(10, 12, provenance=("external second price",))})
    assert (interval.lo, interval.hi) == (13, 16)
    assert "conditional serial stage addition" in interval.provenance
    assert graph["authority"] == "none" and graph["excluded_costs"]


def test_absent_price_opaque_effect_and_unowned_instruction_preserve_unknown():
    graph = observed_graph()
    assert not compose_candidate_cost_graph(graph, stage_prices={"task:0": CycleInterval.point(1)}).resolved
    graph["completeness"] = {"status": "UNKNOWN", "missing": ["opaque assembly", "unowned instructions"]}
    prices = {"task:0": CycleInterval.point(1), "task:1": CycleInterval.point(2)}
    result = compose_candidate_cost_graph(graph, stage_prices=prices)
    assert result.missing == ("opaque assembly", "unowned instructions")
    graph = observed_graph()
    graph["stages"][0]["depends_on"] = ["task:1"]
    assert not compose_candidate_cost_graph(graph, stage_prices=prices).resolved


def test_overlap_needs_complete_exact_independent_observed_intervals():
    graph = observed_graph(dependency=False)
    prices = {"task:0": CycleInterval.point(5), "task:1": CycleInterval.point(8)}
    timeline = ActivityTimeline((
        ScheduledEvent(ActivityEvent("task:0", "first", "priced_stage", 5,
                                     provenance="external observed interval"), 0, 5),
        ScheduledEvent(ActivityEvent("task:1", "second", "priced_stage", 8,
                                     provenance="external observed interval"), 0, 8),
    ), 8, 8)
    assert compose_candidate_cost_graph(graph, stage_prices=prices).lo == 13
    assert compose_candidate_cost_graph(graph, stage_prices=prices, observed_timeline=timeline).lo == 8
    dependent = observed_graph()
    assert not compose_candidate_cost_graph(dependent, stage_prices=prices, observed_timeline=timeline).resolved
    incomplete = ActivityTimeline(timeline.events[:1], 5, 5)
    assert not compose_candidate_cost_graph(graph, stage_prices=prices, observed_timeline=incomplete).resolved
    uncertain = {**prices, "task:0": CycleInterval(4, 6)}
    assert not compose_candidate_cost_graph(graph, stage_prices=uncertain, observed_timeline=timeline).resolved


def test_declared_device_requests_do_not_establish_executed_volume_or_price():
    graph = observed_graph()
    diagnostics = {"source_emission_join": {"status": "UNKNOWN"}, "dependency_stages": {"status": "UNKNOWN"}}
    cb = {"commands": [{"opcode": "candidate_operation", "operands": {"src": "a"},
                        "attributes": {"shape": [2, 3]}}],
          "tensors": {"a": {"shape": [2, 3], "dtype": "i8", "data": "WITHHELD_LITERAL_PAYLOAD"}}}
    requested = observe_candidate_cost_graph(diagnostics, command_buffer=cb)
    row = requested["requested_commands"][0]
    assert row["operands"]["src"] == {"tensor": "a", "shape": [2, 3], "dtype": "i8"}
    assert row["physical_bytes"] is row["executed_multiplicity"] is row["service_cycles"] is None
    assert "WITHHELD_LITERAL_PAYLOAD" not in str(requested)
    assert requested["completeness"]["status"] == "UNKNOWN"
    changed = deepcopy(graph)
    changed["stages"][1]["id"] = "task:0"
    assert not compose_candidate_cost_graph(changed, stage_prices={"task:0": CycleInterval.point(1)}).resolved
