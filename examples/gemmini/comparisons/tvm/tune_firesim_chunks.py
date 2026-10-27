"""Stock TVM DSE over small semantic programs, with a bounded FireSim job pool.

The installed manager adapter is supplied explicitly; see REMOTE_TUNING_README.
This driver reuses the existing TensorIR extraction/build/verification path.
TVM owns the gradient scheduler, EvolutionarySearch, XGB, traces and database.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from firesim_job_pool import FireSimPool, identity


def write_json(path, value):
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        stream.write(json.dumps(value, indent=2) + "\n")
        temporary = stream.name
    os.replace(temporary, path)


def load_toolchain(path):
    import measure

    class FireSimToolchain(measure.Toolchain):
        def memory_configuration(self):
            if self.hardware_receipt is None:
                return dict(
                    kind="firesim",
                    expected_memory_model="DDR3",
                    scope="Memory model is already set by the provided bitstream; no software memory configuration",
                )
            receipt = identity(self.hardware_receipt)
            return dict(
                kind="firesim",
                fingerprint=receipt["sha256"],
                hardware_receipt=receipt,
                scope="Actual remote image receipt; no Verilator timing labels",
            )

    config = json.loads(Path(path).read_text())
    config.setdefault("hardware_receipt", None)
    toolchain = FireSimToolchain.from_json(config)
    if toolchain.timing_policy != "fixed-warm" or toolchain.cycle_watchdog or toolchain.memory_variant_receipt:
        raise ValueError("Use a fixed-warm FireSim build configuration, without the Verilator observer")
    toolchain.adapter  # Verify object bytes before any build.
    receipt = json.loads(toolchain.spike_receipt.read_text())
    for field in ("spike", "plugin"):
        if identity(receipt[field + "_path"])["sha256"] != receipt[field + "_sha256"]:
            raise ValueError("Changed functional engine")
    return toolchain


def build_request(path):
    import measure
    import tvm

    request = json.loads(Path(path).read_text())
    toolchain = load_toolchain(request["toolchain"])
    measure.build_candidate(
        tvm.ir.load_json(Path(request["module"]).read_text()), request["workload"], request["folder"], toolchain
    )


def tune(args):
    import measure
    import tvm
    from firesim_runner_seconds import SecondsRunner, cycles_to_seconds
    from graph_plans import GraphPlan
    from program_io import import_program
    from tune_models import bind_operator_workloads, extract_inventory
    from tvm import meta_schedule as ms
    from tvm.relax.backend.contrib.gemmini_search import (
        WORKLOAD_ATTR,
        GemminiBuilder,
        GemminiRunner,
        candidate_workload,
        open_gemmini_database,
        tune_gemmini_tasks,
    )
    from tvm.relax.backend.contrib.gemmini_sketch import default_schedule
    from tvm.relax.backend.contrib.gemmini_target import GemminiCapabilities

    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    toolchain = load_toolchain(args.toolchain)
    pool = FireSimPool(args.pool, deadline_epoch=args.deadline_epoch,
                       finish_active_runs=args.finish_active_runs,
                       active_run_timeout_seconds=args.active_run_timeout_seconds)
    cycles_to_seconds(1, args.target_clock_hz)
    write_json(
        root / "measurement-units.json",
        dict(
            database_cost_unit="seconds",
            raw_measurement_unit="target_cycles",
            target_cycle_counter_hz=args.target_clock_hz,
            frequency_scope="Documented target cycle-counter frequency, not FPGA host/emulation throughput",
            pool=pool.config_identity,
        ),
    )
    if (
        toolchain.hardware_receipt is not None
        and pool.hardware.get("sha256") != identity(toolchain.hardware_receipt)["sha256"]
    ):
        raise ValueError("Build and runner hardware identities differ")
    caps = GemminiCapabilities()
    program = import_program(args.program)
    graph_plan = GraphPlan.from_args(args)
    prepared, eligible, inventory = extract_inventory(program, caps, **graph_plan.preparation_options)
    write_json(root / "graph-plan.json", graph_plan.identity(program, prepared))
    selected_names = {row["task"] for row in inventory if row["kind"] == "contraction"}
    tasks = [task for task in eligible if task.task_name in selected_names]
    if args.workload_key:
        requested = set(args.workload_key)
        available = {str(task.dispatched[0]["main"].attrs[WORKLOAD_ATTR]) for task in tasks}
        if requested - available:
            raise ValueError("Requested workload keys are absent from this exact prepared chunk")
        tasks = [task for task in tasks if str(task.dispatched[0]["main"].attrs[WORKLOAD_ATTR]) in requested]
        selected_names = {task.task_name for task in tasks}
    if not tasks:
        raise ValueError("Chunk has no supported contractions")
    write_json(
        root / "tasks.json",
        dict(
            tasks=inventory,
            selected=sorted(selected_names),
            selected_workload_keys=[str(task.dispatched[0]["main"].attrs[WORKLOAD_ATTR]) for task in tasks],
            scope="Contraction DSE; surrounding CPU work retained in whole-chunk validation",
        ),
    )
    bound, binding_identity = bind_operator_workloads(prepared, tasks, root, input_fixture=args.fixture)
    for task in tasks:
        func = task.dispatched[0]["main"]
        key = str(func.attrs[WORKLOAD_ATTR])
        binding = bound[key]
        selected = binding["provenance"]["selected_call_index"]
        if not next(row for row in binding["provenance"]["occurrences"] if row["call_index"] == selected)[
            "complete_inputs"
        ]:
            raise ValueError("Every task must use actual chunk boundary/constant inputs")
        measure.prepare_workload(
            func,
            root / "workloads" / key,
            input_overrides=binding["input_overrides"],
            input_provenance=binding["provenance"],
        )
    lock = threading.Lock()
    counter, incumbents, duplicates = 0, {}, {}

    def build(mod, key):
        nonlocal counter
        if time.time() >= args.deadline_epoch:
            raise TimeoutError("Study deadline reached")
        with lock:
            index = counter
            counter += 1
        folder = root / "candidates" / f"{index:05d}-{key}"
        folder.parent.mkdir(parents=True, exist_ok=True)
        module = folder.with_suffix(".module.json")
        module.write_text(tvm.ir.save_json(mod))
        request = folder.with_suffix(".request.json")
        write_json(
            request,
            dict(
                module=str(module),
                workload=str(root / "workloads" / key),
                folder=str(folder),
                toolchain=str(args.toolchain.resolve()),
            ),
        )
        with folder.with_suffix(".build.log").open("w") as log:
            process = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--build-request", str(request)],
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=min(300, max(0.1, args.deadline_epoch - time.time())),
            )
        if process.returncode:
            raise RuntimeError("Candidate build failed: " + str(folder))
        return folder / "artifact.json"

    def run(path):
        artifact = json.loads(Path(path).read_text())
        if not artifact.get("native"):
            raise ValueError("Eligible contraction candidates must retain Gemmini placement")
        folder = Path(artifact["folder"])
        elf = artifact["elfs"]["valid"]
        key = Path(artifact["workload"]).name
        with lock:
            duplicate_lock = duplicates.setdefault(elf["sha256"], threading.Lock())
        with duplicate_lock:
            with lock:
                cached = duplicates.get("cost:" + elf["sha256"])
                incumbent = incumbents.get(key)
            if cached is not None:
                if incumbent is not None and cached >= incumbent:
                    raise TimeoutError("Cached duplicate is dominated by the current incumbent")
                write_json(
                    folder / "measurement.json",
                    dict(status="passed", cycles=cached, duplicate_of_elf_sha256=elf["sha256"], hardware=pool.hardware),
                )
                return cached
            for label, expected, marker in (("valid", 0, "DSE_PASS"), ("bad-oracle", 22, "TENSORIR_NUMERICAL_FAILURE")):
                item = artifact["elfs"][label]
                if identity(item["path"])["sha256"] != item["sha256"]:
                    raise ValueError("Changed candidate ELF")
                remaining = args.deadline_epoch - time.time()
                if remaining <= 0:
                    raise TimeoutError("Study deadline reached during functional admission")
                code, text = measure._spike(item["path"], toolchain, timeout=min(120, remaining))
                (folder / (label + ".spike.log")).write_text(text)
                if code != expected or marker not in text:
                    raise ValueError("Functional control failed before paid execution")

            def current_limit():
                with lock:
                    return incumbents.get(key)

            cycles = pool.run(
                elf["path"],
                folder / "firesim",
                measured_limit=current_limit,
                live_limit=dict(path=str(root / "incumbents.json"), key=key),
            )
            with lock:
                # A peer may have improved the limit while this job was running.
                current = incumbents.get(key)
                if current is not None and cycles >= current:
                    raise TimeoutError("Completed after a faster incumbent; censored")
                incumbents[key] = cycles
                duplicates["cost:" + elf["sha256"]] = cycles
                write_json(root / "incumbents.json", incumbents)
            write_json(
                folder / "measurement.json",
                dict(
                    status="passed",
                    cycles=cycles,
                    elf=elf,
                    hardware=pool.hardware,
                    result=identity(folder / "firesim/result.json"),
                ),
            )
            return cycles

    baseline_records = []
    seed_database = (
        open_gemmini_database(args.seed_database, prepared, capabilities=caps) if args.seed_database else None
    )
    # Establish every baseline before search; subsequent requests carry a limit.
    for task in tasks:
        schedule = default_schedule(task.dispatched[0]["main"])
        key = str(task.dispatched[0]["main"].attrs[WORKLOAD_ATTR])
        artifact = build(schedule.mod, key)
        cycles = run(artifact)
        baseline_records.append((task, schedule.trace, cycles, str(artifact)))
        if seed_database is not None:
            record = seed_database.query_tuning_record(task.dispatched[0], caps.target(), "main")
            if record is not None:
                schedule = tvm.tir.Schedule(task.dispatched[0])
                record.trace.apply_to_schedule(schedule, remove_postproc=False)
                try:
                    artifact = build(schedule.mod, key)
                    cycles = run(artifact)
                    baseline_records.append((task, record.trace, cycles, str(artifact)))
                except (RuntimeError, ValueError, TimeoutError) as error:
                    write_json(
                        root / (key + "-seed-failure.json"), dict(error=str(error), historical_costs_imported=False)
                    )
    write_json(
        root / "baselines.json", [dict(task=t.task_name, cycles=c, artifact=a) for t, _, c, a in baseline_records]
    )

    def describe(path):
        proof_path = Path(path).parent / "measurement.json" if path else None
        proof = json.loads(proof_path.read_text()) if proof_path and proof_path.exists() else {}
        cycles = proof.get("cycles") if proof.get("status") == "passed" else None
        return dict(
            cycles=cycles,
            run_secs=cycles_to_seconds(cycles, args.target_clock_hz) if cycles else None,
            database_cost_unit="seconds",
            raw_measurement_unit="target_cycles",
        )

    database = tune_gemmini_tasks(
        tasks,
        work_dir=root / "database",
        capabilities=caps,
        builder=GemminiBuilder(build, max_workers=args.builders),
        runner=SecondsRunner(GemminiRunner(run, max_workers=len(pool.slots)), args.target_clock_hz),
        max_trials_global=args.trials,
        num_trials_per_iter=args.batch,
        include_cpu=False,
        compact=True,
        log=root / "trials.jsonl",
        describe=describe,
    )
    for task, trace, cycles, _ in baseline_records:
        database.commit_tuning_record(
            ms.database.TuningRecord(
                trace,
                database.commit_workload(task.dispatched[0]),
                [cycles_to_seconds(cycles, args.target_clock_hz)],
                caps.target(),
            )
        )

    # Retain the complete stock search database as evidence. Export only labels
    # witnessed by a successful runner admission, excluding stock failure costs.
    def trace_key(trace):
        return hashlib.sha256(tvm.ir.save_json(trace.as_json()).encode()).hexdigest()

    admitted = {
        (
            str(task.dispatched[0]["main"].attrs[WORKLOAD_ATTR]),
            trace_key(trace),
            cycles_to_seconds(cycles, args.target_clock_hz),
        )
        for task, trace, cycles, _ in baseline_records
    }
    for line in (root / "trials.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["cycles"] is None or row["build_error"] or row["run_error"]:
            continue
        proof = json.loads((Path(row["artifact"]).parent / "measurement.json").read_text())
        if proof["status"] != "passed" or proof["cycles"] != row["cycles"]:
            raise ValueError("Search label differs from runner admission")
        native_trace = json.dumps(row["trace"])
        trace_json = tvm.ir.save_json(tvm.ir.load_json(native_trace))
        admitted.add((row["workload"], hashlib.sha256(trace_json.encode()).hexdigest(), float(row["run_secs"])))
    (root / "validated-database").mkdir()
    validated = ms.database.JSONDatabase(work_dir=str(root / "validated-database"))
    for record in database.get_all_tuning_records():
        if (candidate_workload(record.workload.mod), trace_key(record.trace), float(record.run_secs[0])) in admitted:
            validated.commit_tuning_record(
                ms.database.TuningRecord(
                    record.trace,
                    validated.commit_workload(record.workload.mod),
                    record.run_secs,
                    record.target,
                    record.args_info,
                )
            )
    if not validated.get_all_tuning_records():
        raise ValueError("No admitted records to export")
    write_json(
        root / "firesim-campaign.json",
        dict(
            schema="gemmini_firesim_chunk_dse_v1",
            status="complete",
            hardware=pool.hardware,
            pool=pool.config_identity,
            program=identity(args.program),
            fixture=identity(args.fixture),
            bindings=binding_identity,
            proposal_budget=args.trials,
            batch=args.batch,
            builders=args.builders,
            slots=len(pool.slots),
            validated_records=len(validated.get_all_tuning_records()),
            historical_costs_imported=False,
            deadline_epoch=args.deadline_epoch,
            finish_active_runs=args.finish_active_runs,
            active_run_timeout_seconds=args.active_run_timeout_seconds,
            graph_cse=args.graph_cse,
            prepack_weights=args.prepack_weights,
            bias_preload=args.bias_preload,
            graph_plan=identity(root / "graph-plan.json"),
            measurement_units=identity(root / "measurement-units.json"),
            validated_database_sources=[
                identity(path) for path in sorted((root / "validated-database").glob("*.json"))
            ],
            optimizer="stock TVM gradient scheduler / EvolutionarySearch / XGB / JSONDatabase",
            qualification="Operator experiments; independently confirm whole chunks and full shapes before deployment",
        ),
    )


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--build-request":
        build_request(sys.argv[2])
        return
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("program", "fixture", "toolchain", "pool", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument(
        "--deadline-epoch",
        type=float,
        required=True,
        help="Shared absolute deadline across all chunk studies; never renewed per process",
    )
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument(
        "--workload-key", action="append", help="Tune only these exact structural keys; repeat as needed"
    )
    parser.add_argument(
        "--target-clock-hz",
        type=float,
        required=True,
        help="Documented frequency of the target cycle counter; never substitute FPGA host or emulation frequency",
    )
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--builders", type=int, default=2)
    parser.add_argument("--finish-active-runs", action="store_true",
                        help="Stop new slot admissions at the deadline; drain active measurements")
    parser.add_argument("--active-run-timeout-seconds", type=float,
                        help="Separate remote per-run hang watchdog when draining; not a tuning budget")
    parser.add_argument("--graph-cse", action="store_true")
    parser.add_argument("--no-prepack-weights", dest="prepack_weights", action="store_false", default=True)
    parser.add_argument("--no-bias-preload", dest="bias_preload", action="store_false", default=True)
    parser.add_argument(
        "--graph-fusion-level",
        type=int,
        choices=(0, 1, 2, 3),
        default=2,
        help="Existing TVM FuseOps level, saved as a replayable Relax graph choice; fixed within each inner DSE",
    )
    parser.add_argument(
        "--seed-database", type=Path, help="Replay matching traces, remeasure here; never import their historical costs"
    )
    args = parser.parse_args()
    if not (args.builders >= 1 and args.batch >= 1 and args.trials >= 4):
        parser.error("Use positive builder/batch counts and at least four proposals to bootstrap ML ranking")
    if not time.time() < args.deadline_epoch <= time.time() + 4 * 3600:
        parser.error("Supply a future shared deadline at most four hours away")
    tune(args)


if __name__ == "__main__":
    main()
