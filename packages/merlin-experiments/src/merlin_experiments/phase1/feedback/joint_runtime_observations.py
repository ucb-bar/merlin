"""Scoped native runtime observations for the compact joint diagnostics adapter."""
from __future__ import annotations

from merlin.perf.candidate_cost_graph import compose_candidate_cost_graph, observe_candidate_cost_graph
from merlin.perf.schedule_pressure import pressure_of

from .joint_diagnostics import _artifact


def produce(native, *, context, artifact, decoded_trace=None, candidate_diagnostics=None,
            command_buffer=None, stage_prices=None, observed_timeline=None):
    """Read existing observer/decoder/compiler products; never run a target.

    The observer's selected counter-window metadata owns the feature-count
    scope. Whole-trace resource totals never become call-local queue or wait
    measurements. Static decoded pressure remains an unqualified heuristic.
    Stage prices must be supplied independently; instruction counts are not prices.
    """
    artifact = _artifact(artifact)
    pins = {key: context.get(key) for key in
            ("compiler_sha256", "capsule_sha256", "scope_sha256", "model_sha256")}
    observation = native.get("feature_observation") or {}
    windows = native.get("counter_windows") or []
    index = observation.get("window_index")
    scoped = (observation.get("scope_kind") == "counter_pc_pair"
              and type(index) is int and 0 <= index < len(windows)
              and bool(context.get("elf_sha256")) and native.get("elf_sha256") == context["elf_sha256"]
              and bool(pins["scope_sha256"]) and native.get("scope_sha256") == pins["scope_sha256"]
              and bool(pins["model_sha256"]) and native.get("model_sha256") == pins["model_sha256"])
    cpu, schedule = {}, {}
    if scoped and native.get("execution_completed") is True and native.get("aborted_instructions") == 0:
        instructions = windows[index].get("instructions")
        if (type(instructions) is int and instructions >= 0
                and instructions == observation.get("scope_instructions") == observation.get("counted_cpu_instructions")):
            cpu["host_instructions"] = instructions
        if observation.get("complete") is True:
            counts = native.get("feature_counts") or {}
            for output, feature in (("fences", "cpu.fence"),):
                if type(counts.get(feature)) is int and counts[feature] >= 0:
                    cpu[output] = counts[feature]
            for output, feature in (("commands_total", "target.commands"), ("load_commands", "target.class.load"),
                                    ("store_commands", "target.class.store"), ("config_commands", "target.class.config"),
                                    ("preload_commands", "target.class.preload"),
                                    ("device_read_bytes", "device.requested_read_bytes"),
                                    ("device_write_bytes", "device.requested_write_bytes"),
                                    ("useful_macs", "target.useful_shape_macs"),
                                    ("padded_macs", "target.nominal_padded_macs")):
                if type(counts.get(feature)) is int and counts[feature] >= 0:
                    schedule[output] = counts[feature]
            # Class counters are disjoint in the selected observer vocabulary.
            # Only aggregate when they cover the complete command population.
            classes = {k.removeprefix("target.class."): v for k, v in counts.items()
                       if k.startswith("target.class.")}
            if (classes and all(type(v) is int and v >= 0 for v in classes.values())
                    and sum(classes.values()) == schedule.get("commands_total")):
                schedule["compute_commands"] = sum(v for k, v in classes.items()
                                                   if k == "compute" or k.startswith("compute_"))
                schedule["store_commands"] = sum(v for k, v in classes.items()
                                                 if k in {"store", "pool_store"}
                                                 or k.startswith(("store_", "pool_store_")))
    records = {name: {"pins": pins, "artifact": artifact, "planned": {}, "emitted": metrics,
                      "origin": "native_observer", "maturity": "scoped_runtime_feature_counts",
                      "limits": ["Counts and requested bytes are not latency or saved cycles.",
                                 "No reservation-station occupancy or component wait attribution."]}
               for name, metrics in (("cpu_sync", cpu), ("schedule_queues", schedule))}
    pressure = {"status": "UNKNOWN", "reason": "no original decoded emitted trace supplied"}
    if decoded_trace is not None:
        trace = decoded_trace.get("trace", {})
        if (not all(pins.values()) or decoded_trace.get("pins") != pins
                or len(trace.get("instructions", ())) > 4096):
            pressure = {"status": "UNKNOWN", "reason": "trace pins differ or bounded pressure analysis exceeds 4096 commands"}
        else:
            trace_ref = _artifact(decoded_trace["artifact"])
            pressure = {**pressure_of(trace), "artifact": trace_ref, "ranking_qualified": False,
                        "scope": "static emitted operand-tile liveness; not reservation-station occupancy"}
    diagnostics = candidate_diagnostics or {}
    if (not pins["compiler_sha256"] or diagnostics.get("source_emission_join", {}).get("candidate_sha256")
            != pins["compiler_sha256"]):
        diagnostics = {}
        command_buffer = None
    graph = observe_candidate_cost_graph(diagnostics, command_buffer=command_buffer)
    graph_prices = stage_prices or {}
    composition = compose_candidate_cost_graph(graph, stage_prices=graph_prices, observed_timeline=observed_timeline)
    whole = {"status": "UNKNOWN", "scope": "emitted candidate kernel work only",
             "stage_count": len(graph["stages"]), "completeness": graph["completeness"],
             "outer_estimate_cycles": native.get("estimated_cycles") if scoped and native.get("timing_complete") else None,
             "stage_price_count": len(graph_prices), "stage_composition_resolved": composition.resolved,
             "stage_composition_qualified": False,
             "excluded_costs": graph["excluded_costs"], "resource_cycles_additive": False,
             "reason": "No full-model cost-scope binding; absent stage prices remain unknown"}
    return {"schema": "merlin.joint_runtime_observations.v1", "observations": records,
            "selected_window": index if scoped else None, "feature_counts_complete": observation.get("complete") is True,
            "pressure": pressure, "wholechain": whole,
            "unavailable": ["CPU submission latency", "completion wait cycles", "accelerator RS occupancy",
                            "dependency and ready stall cycles", "priced full-model host/device composition"]}


def profile_chain(profiles, *, artifact, elf_sha256, hardware_sha256, scope_sha256,
                  sequential_host_intervals=False):
    """Keep physical outer counters beside reported nested host-call counters.

    Caller pins an independently inspected counter scope. A sum is displayed
    only with an explicit sequential-host-interval premise; its remainder is
    unallocated outer time, never measured overlap, device stalls or savings.
    """
    reference = _artifact(artifact)
    pins = {"elf_sha256": elf_sha256, "hardware_sha256": hardware_sha256, "scope_sha256": scope_sha256}
    if any(not isinstance(value, str) or len(value) != 64
           or any(character not in "0123456789abcdef" for character in value) for value in pins.values()):
        raise ValueError("physical profile needs explicit ELF/hardware/counter-scope pins")
    windows = []
    for profile in profiles[:8]:
        result = profile.get("result", {})
        cycles = result.get("cycles")
        if type(cycles) is not int or cycles <= 0:
            continue
        stages = profile.get("stage_cycles", [])
        valid = (len(stages) <= 1024 and len({r.get("index") for r in stages}) == len(stages)
                 and all(type(r.get("cycles")) is int and r["cycles"] >= 0 for r in stages))
        total = sum(r["cycles"] for r in stages) if valid and stages and sequential_host_intervals else None
        input_cycles = profile.get("input_cycles")
        input_cycles = input_cycles if type(input_cycles) is int and input_cycles >= 0 else None
        remainder = cycles - total - input_cycles if total is not None and input_cycles is not None else None
        windows.append({"variant": str(result.get("variant_id", "unknown"))[:120], "outer_cycles": cycles,
                        "reported_stage_count": len(stages), "reported_sequential_host_stage_sum": total,
                        "input_cycles": input_cycles, "unallocated_outer_cycles": remainder,
                        "interval_partition_consistent": remainder is not None and remainder >= 0,
                        "largest_reported_host_stages": sorted(stages, key=lambda r: r.get("cycles", -1), reverse=True)[:8] if valid else [],
                        "checked_outputs": result.get("checked_elements"), "mismatches": result.get("mismatches")})
    return {"schema": "merlin.joint_physical_chain_observation.v1", "status": "OBSERVED" if windows else "UNKNOWN",
            "pins": pins, "artifact": reference, "windows": windows, "promotion_authority": False,
            "scope": "original selected outer model-call and reported nested host-call intervals",
            "scope_qualification": "caller-inspected counter boundaries, not original model/reference equivalence",
            "resource_cycles_additive": False, "overlap_or_savings_inferred": False,
            "independent_reference_provenance": "UNKNOWN", "simulator_transfer_qualified": False}
