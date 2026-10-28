"""Stock TVM DSE over small semantic programs, with a bounded FireSim job pool.

The installed manager adapter is supplied explicitly; see REMOTE_TUNING_README.
This driver reuses TensorIR extraction/build and remote guest correctness checks.
It does not run a sequential Spike preflight for each candidate.
TVM owns the gradient scheduler, EvolutionarySearch, XGB, traces and database.
"""

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from firesim_job_pool import FireSimPool, identity
from firesim_timeout import MedianWatchdog, positive_seconds


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
    config.setdefault("spike_receipt", None)  # Unused by the FireSim build/run path.
    toolchain = FireSimToolchain.from_json(config)
    if toolchain.timing_policy != "fixed-warm" or toolchain.cycle_watchdog or toolchain.memory_variant_receipt:
        raise ValueError("Use a fixed-warm FireSim build configuration, without the Verilator observer")
    toolchain.adapter  # Verify object bytes before any build.
    return toolchain


def measurement_sources(tvm, measure):
    """Bind Python lowering/build/runner code as well as the native compiler."""
    backend = Path(tvm.__file__).parent / "relax/backend/contrib"
    paths = set(backend.glob("gemmini*.py"))
    paths.update(Path(measure.__file__).parent.glob("*.py"))
    paths.update(Path(__file__).with_name(name) for name in (
        "tune_firesim_chunks.py", "firesim_timeout.py", "firesim_job_pool.py",
        "firesim_runner_seconds.py", "run_firesim_study.py",
    ))
    if not paths or not (backend / "gemmini_search.py").is_file():
        raise ValueError("Cannot bind the active Gemmini Python compiler sources")
    return [identity(path) for path in sorted(paths)]


def load_candidate(path):
    """Bind both retained control ELFs before remote execution or cache reuse."""
    artifact = json.loads(Path(path).read_text())
    if not artifact.get("native"):
        raise ValueError("Eligible contraction candidates must retain Gemmini placement")
    for label in ("valid", "bad-oracle"):
        item = artifact["elfs"][label]
        if identity(item["path"])["sha256"] != item["sha256"]:
            raise ValueError("Changed candidate ELF: " + label)
    return artifact


class CandidateMeasurements:
    """One path for calibrated baselines, completed timings and exact ELF reuse."""

    def __init__(self, pool, watchdog):
        self.pool, self.watchdog = pool, watchdog
        self.lock = threading.Lock()
        self.duplicates = {}

    def __call__(self, path, *, baseline_index=None):
        artifact = load_candidate(path)
        folder = Path(artifact["folder"])
        elf = artifact["elfs"]["valid"]
        key = Path(artifact["workload"]).name
        baseline = baseline_index is not None
        with self.lock:
            duplicate_lock = self.duplicates.setdefault(elf["sha256"], threading.Lock())
        with duplicate_lock:
            with self.lock:
                cached = self.duplicates.get("cost:" + elf["sha256"])
            if cached is not None and not baseline:
                write_json(folder / "measurement.json",
                           dict(cached, duplicate_of_elf_sha256=elf["sha256"]))
                return cached["cycles"]
            execution = folder / (f"firesim-baseline-{baseline_index}" if baseline else "firesim")
            cycles = self.pool.run(
                elf["path"], execution,
                wall_timeout=lambda: self.watchdog.timeout(key, baseline=baseline),
            )
            occupancy = json.loads((execution / "occupancy.json").read_text())
            proof = dict(status="passed", cycles=cycles, elf=elf, hardware=self.pool.hardware,
                         result=identity(execution / "result.json"),
                         occupancy=identity(execution / "occupancy.json"))
            write_json(folder / (f"baseline-{baseline_index}.json" if baseline else "measurement.json"), proof)
            self.watchdog.observe(key, occupancy["occupied_seconds"], baseline=baseline)
            if not baseline:
                with self.lock:
                    self.duplicates["cost:" + elf["sha256"]] = proof
            return cycles

    def calibrate(self, path):
        built = load_candidate(path)
        key = Path(built["workload"]).name
        samples = [self(path, baseline_index=index) for index in range(3)]
        cycles = statistics.median(samples)
        self.watchdog.timeout(key)  # Refuse an inconsistent pilot ceiling.
        proof = dict(status="passed", cycles=cycles, cycle_samples=samples,
                     aggregation="median of three correct baseline jobs",
                     elf=built["elfs"]["valid"], hardware=self.pool.hardware,
                     results=[identity(Path(path).parent / f"baseline-{i}.json") for i in range(3)])
        write_json(Path(path).parent / "measurement.json", proof)
        with self.lock:
            self.duplicates["cost:" + built["elfs"]["valid"]["sha256"]] = proof
        return cycles


def build_request(path):
    import measure
    import tvm

    request = json.loads(Path(path).read_text())
    toolchain = load_toolchain(request["toolchain"])
    measure.build_candidate(
        tvm.ir.load_json(Path(request["module"]).read_text()), request["workload"], request["folder"], toolchain
    )


def tune(args):
    import xgboost  # Check before baseline jobs consume any FireSim time.
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
                       active_run_timeout_seconds=args.candidate_timeout_ceiling_seconds)
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
    watchdog = MedianWatchdog(
        root / "timeout-history.json",
        bootstrap_seconds=args.candidate_bootstrap_seconds,
        ceiling_seconds=args.candidate_timeout_ceiling_seconds,
        binding=dict(program=identity(args.program), fixture=identity(args.fixture),
                     graph_plan=identity(root / "graph-plan.json"),
                     toolchain=identity(args.toolchain), pool=pool.config_identity,
                     compiler=identity(tvm._ffi.base._LIB._name),
                     python_sources=measurement_sources(tvm, measure)),
    )
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
    counter = 0

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

    run = CandidateMeasurements(pool, watchdog)

    baseline_records = []
    seed_database = (
        open_gemmini_database(args.seed_database, prepared, capabilities=caps) if args.seed_database else None
    )
    # Three actual baseline jobs calibrate each workload; cached results do not count.
    for task in tasks:
        schedule = default_schedule(task.dispatched[0]["main"])
        key = str(task.dispatched[0]["main"].attrs[WORKLOAD_ATTR])
        artifact = build(schedule.mod, key)
        cycles = run.calibrate(artifact)
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
        warmup=args.cost_model_warmup,
        eps_greedy=0.0,
        max_trials_per_task=55,
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
            cost_model_warmup=args.cost_model_warmup,
            xgboost_version=xgboost.__version__,
            max_trials_per_task=55,
            eps_greedy=0.0,
            builders=args.builders,
            slots=len(pool.slots),
            validated_records=len(validated.get_all_tuning_records()),
            historical_costs_imported=False,
            deadline_epoch=args.deadline_epoch,
            finish_active_runs=args.finish_active_runs,
            candidate_bootstrap_seconds=args.candidate_bootstrap_seconds,
            candidate_timeout_ceiling_seconds=args.candidate_timeout_ceiling_seconds,
            timeout_policy=identity(root / "timeout-history.json"),
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
    parser.add_argument("--trials", type=int, default=55)
    parser.add_argument(
        "--workload-key", action="append", help="Tune only these exact structural keys; repeat as needed"
    )
    parser.add_argument(
        "--target-clock-hz",
        type=float,
        required=True,
        help="Documented frequency of the target cycle counter; never substitute FPGA host or emulation frequency",
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--cost-model-warmup", type=int, default=50)
    parser.add_argument("--builders", type=int, default=2)
    parser.add_argument("--finish-active-runs", action="store_true", default=True,
                        help="Stop new slot admissions at the deadline; drain active measurements")
    parser.add_argument("--candidate-bootstrap-seconds", type=float, required=True,
                        help="Explicit pilot watchdog for the three baseline jobs per workload")
    parser.add_argument("--candidate-timeout-ceiling-seconds", type=float, required=True,
                        help="Hard ceiling for candidate median watchdogs; not an E2E timeout")
    parser.add_argument("--graph-cse", action="store_true")
    parser.add_argument("--no-prepack-weights", dest="prepack_weights", action="store_false", default=False)
    parser.add_argument("--no-bias-preload", dest="bias_preload", action="store_false", default=False)
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
    try:
        bootstrap = positive_seconds(args.candidate_bootstrap_seconds)
        ceiling = positive_seconds(args.candidate_timeout_ceiling_seconds)
        if bootstrap > ceiling:
            raise ValueError("Bootstrap watchdog exceeds the hard ceiling")
    except ValueError as error:
        parser.error(str(error))
    if not (args.builders >= 1 and args.batch >= 1 and args.trials >= 1 and args.cost_model_warmup >= 1):
        parser.error("Use positive builder/batch counts and positive proposal and XGB warmup counts")
    if not time.time() < args.deadline_epoch <= time.time() + 4 * 3600:
        parser.error("Supply a future shared deadline at most four hours away")
    tune(args)


if __name__ == "__main__":
    main()
