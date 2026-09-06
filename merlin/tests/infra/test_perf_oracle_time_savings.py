"""Oracle time a performance campaign no longer has to buy -- and the record that says so.

Every saving here is a MEASUREMENT AVOIDED, never an estimate substituted. That distinction is the
whole point: cheap ranking is refuted on this path (the correctness simulator orders two candidates
for one workload correctly 46.1% of the time, and a cost model accurate to 8.1% on absolute
magnitude manages 39.3% -- worse than a coin), so nothing may stand in for a cycle count. What may
happen is that a number the pinned engine ALREADY returned for these very bytes is returned again,
or that two executions that were going to run anyway run at the same time.

Three mechanisms, and what each of these tests is here to catch:

* **The frozen baseline is measured once.** The baseline arm is the byte-identical functional
  submission in every trial and every cell, so its cycles are a campaign constant. The measurement
  store already returns it -- but only to a cell that starts after the first one finished, so a
  fanned-out matrix forfeits the entire saving. ``baseline_lead_prefix`` restores it, and must do so
  as a PREFIX, because the checkpoint chain is a linear hash chain committed in declared order.
* **A carried number is reported as carried.** The per-row provenance said so; nothing said it at
  campaign level, so the saving was invisible and therefore unauditable. Silence is now a refusal.
* **Fan-out.** Cycle counts do not move with concurrency; wall time does. The record produced by a
  wide run must be byte-identical to the serial one, and the conditions that make concurrency unsafe
  (process-global counter environment, a relative path a chdir window can re-root) must drop the
  width back to one rather than proceed.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import threading
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from merlin.common.paths import merlin_dir

_SCRIPTS = merlin_dir() / "experiments/gemmini_perf_bench/scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import run_paired_perf_bench as PAIR  # noqa: E402

_ORCH_SOURCE = _SCRIPTS / "run_agentic_perf_experiment.py"
_ORCH_SPEC = importlib.util.spec_from_file_location("run_agentic_perf_experiment_savings",
                                                    _ORCH_SOURCE)
assert _ORCH_SPEC is not None and _ORCH_SPEC.loader is not None
ORCH = importlib.util.module_from_spec(_ORCH_SPEC)
sys.modules[_ORCH_SPEC.name] = ORCH
_ORCH_SPEC.loader.exec_module(ORCH)

CB = {"abi_version": "0.1", "commands": [{"opcode": "MATMUL", "operands": {"dst": "Y"}}]}
SCOPE = "r000/unprofiled"
FIRRTL, MODEL = "c" * 64, "d" * 64


# =====================================================================================
# 1. the frozen baseline is measured once per campaign, at any width
# =====================================================================================
def _stage(name: str, *, launches: bool) -> ORCH.ChildStage:
    return ORCH.ChildStage(name, (lambda: None) if launches else None, lambda: name)


def test_the_lead_wave_covers_every_corpus_before_anything_fans_out():
    """One cell per corpus lands first, so the frozen baseline for that corpus is bought once."""
    stages = [_stage(f"m:{trial}:{phase}", launches=True)
              for trial in ("t0", "t1", "t2") for phase in ("tuning", "held_out")]
    phases = ["tuning", "held_out"] * 3
    lead = ORCH.baseline_lead_prefix(stages, phases)
    assert set(phases[:lead]) == {"tuning", "held_out"}
    # and it is the SHORTEST such prefix -- a longer one serialises measurements for nothing
    assert set(phases[:lead - 1]) != {"tuning", "held_out"}


def test_the_lead_wave_is_a_prefix_so_the_commit_order_never_changes():
    """The checkpoint chain is a linear hash chain committed in declared order; a filtered subset
    would reorder it and the resumed campaign would not re-derive the same chain."""
    stages = [_stage(f"m:{trial}:{phase}", launches=True)
              for trial in ("t0", "t1", "t2") for phase in ("tuning", "held_out")]
    phases = ["tuning", "held_out"] * 3
    lead = ORCH.baseline_lead_prefix(stages, phases)
    assert [stage.name for stage in stages[:lead] + stages[lead:]] == [s.name for s in stages]


def test_an_adopted_cell_does_not_stand_in_for_a_measurement_nobody_ran():
    """A cell with no launch is being adopted from an artifact; it warms nothing in this campaign,
    so the lead must reach the first cell of that corpus that actually runs."""
    stages = [_stage("m:t0:tuning", launches=False), _stage("m:t0:held_out", launches=True),
              _stage("m:t1:tuning", launches=True), _stage("m:t1:held_out", launches=True)]
    phases = ["tuning", "held_out", "tuning", "held_out"]
    lead = ORCH.baseline_lead_prefix(stages, phases)
    assert lead == 3 and all(stage.launch is not None for stage in stages[1:lead])


def test_nothing_left_to_launch_needs_no_lead_wave():
    stages = [_stage("m:t0:tuning", launches=False), _stage("m:t0:held_out", launches=False)]
    assert ORCH.baseline_lead_prefix(stages, ["tuning", "held_out"]) == 0


class _Checkpoints:
    """The linear chain, reduced to what `_measure_cells` reads and appends."""

    def __init__(self):
        self.order: list[str] = []
        self._rows: dict[str, dict] = {}

    def evidence(self, name):
        return self._rows.get(name)

    def append(self, name, row):
        self.order.append(name)
        self._rows[name] = row


def _measure_waves(monkeypatch, tmp_path: Path, *, workers: int, sim_workers: int = 1
                   ) -> tuple[list[list[str]], list[str], list[list[str]]]:
    """Run the full 3-trial x 2-phase matrix with every child stubbed; report the wave split."""
    waves: list[list[str]] = []
    commands: list[list[str]] = []

    def run_child_stages(stages, *, workers):
        waves.append([stage.name for stage in stages if stage.launch is not None])
        for stage in stages:
            if stage.launch is not None:
                stage.launch()
            stage.commit()
        return []

    monkeypatch.setattr(ORCH, "run_child_stages", run_child_stages)
    monkeypatch.setattr(ORCH, "_uncheckpointed_state", lambda root, artifact, *, label: "absent")
    monkeypatch.setattr(ORCH, "child_environment", lambda config, certificate: {})
    monkeypatch.setattr(ORCH, "_run_checked",
                        lambda runner, command, **k: commands.append(list(command)))
    monkeypatch.setattr(ORCH, "_verify_measurement_manifest",
                        lambda path, **kwargs: {"path": str(path)})
    monkeypatch.setattr(ORCH, "_verify_saved_file", lambda saved, output, *, label: Path(output))

    handoff = SimpleNamespace(record_path=tmp_path / "candidate.json",
                              corpus_root=tmp_path / "corpus")
    certificate = SimpleNamespace(path=tmp_path / "certificate.json", sha256="b" * 64)
    cells = [ORCH._MeasurementCell(
        trial=trial, phase=phase, handoff=handoff, corpus_root=tmp_path / "corpus",
        corpus_manifest=tmp_path / "corpus/manifest.json", corpus_manifest_sha256="c" * 64,
        corpus_capsules_sha256="d" * 64, certificate=certificate,
        stage=f"measurement:{trial}:{phase}", run_id=f"{trial}__{phase}",
        output=tmp_path / trial / phase / "campaign_manifest.json")
        for trial in ("t0", "t1", "t2") for phase in ("tuning", "held_out")]
    config = SimpleNamespace(functional_run_id="functional", functional_submission_sha256="a" * 64,
                             rtl_facts=tmp_path / "rtl.json", measurement_timeout=90,
                             hardware_counters=False, sim_workers=sim_workers)
    state = _Checkpoints()
    ORCH._measure_cells(cells, config, state, command_runner=lambda *a, **k: None, workers=workers)
    return waves, state.order, commands


def test_a_serial_matrix_is_launched_exactly_as_before(monkeypatch, tmp_path):
    """workers == 1 takes the single unsplit call: the serial campaign is the reference."""
    waves, order, _ = _measure_waves(monkeypatch, tmp_path, workers=1)
    assert len(waves) == 1 and len(waves[0]) == 6
    assert order == [f"measurement:{t}:{p}" for t in ("t0", "t1", "t2")
                     for p in ("tuning", "held_out")]


def test_a_wide_matrix_lands_one_cell_per_corpus_before_the_rest(monkeypatch, tmp_path):
    """The frozen baseline of each corpus is bought by the lead wave and READ by the other four."""
    waves, order, _ = _measure_waves(monkeypatch, tmp_path, workers=6)
    assert waves[0] == ["measurement:t0:tuning", "measurement:t0:held_out"]
    assert len(waves) == 2 and len(waves[1]) == 4
    # the checkpoint chain is unchanged: splitting into waves must not reorder a single commit
    assert order == [f"measurement:{t}:{p}" for t in ("t0", "t1", "t2")
                     for p in ("tuning", "held_out")]


# =====================================================================================
# 2. a carried measurement is reported as carried -- and silence refuses the campaign
# =====================================================================================
class _Certificate:
    def __init__(self, binary_sha256: str):
        self.pins = {"gsim_binary": {"sha256": binary_sha256},
                     "gsim_firrtl": {"sha256": FIRRTL}, "gsim_model": {"sha256": MODEL}}


class _Engine:
    """One stubbed pinned GSIM: a binary whose bytes hash to its own pin, and a counted oracle."""

    def __init__(self, tmp_path: Path, monkeypatch, name: str = "engineA"):
        self.binary = tmp_path / f"{name}.bin"
        self.binary.write_bytes(name.encode())
        self.elf = tmp_path / f"{name}.elf"
        self.elf.write_bytes(b"elf")
        self.pin = hashlib.sha256(self.binary.read_bytes()).hexdigest()
        self.certificate = _Certificate(self.pin)
        self.calls = 0
        self._monkeypatch = monkeypatch
        self.activate()
        monkeypatch.setattr(PAIR.CERTPROD, "encode_declared_outputs", lambda outputs, cb: ("s" * 64, {}))

    def activate(self):
        """Make THIS engine the one the run resolves; the backend resolver is process-global."""
        import merlin.runtime.backends.base as backends
        self._monkeypatch.setattr(
            backends, "get_backend",
            lambda target: types.SimpleNamespace(gsim_path=lambda: str(self.binary)))

        def run_on_oracle(cb, llvm_text, *, simulator, target, workdir, timeout):
            self.calls += 1
            return {"elf": str(self.elf), "cycles": 4242, "outputs": {"Y": [1]}, "console": "",
                    "oracle": {"kind": "rtl_gsim", "derived_from_rtl": True},
                    "timing": {"build_s": 0.5, "sim_active_s": 110.0, "oracle_wait_s": 0.0}}

        self._monkeypatch.setattr(PAIR.OOT, "run_on_oracle", run_on_oracle)

    def measure(self, store, cb=CB, llvm="module {}"):
        evidence: dict = {}
        result = PAIR._gsim_l3_adapter("t", evidence, self.certificate,
                                       reuse_scope=SCOPE, store=store)(cb, llvm, self.elf.parent, 60)
        return evidence, result


@pytest.fixture(autouse=True)
def _empty_memo():
    PAIR._L3_MEMO.clear()
    yield
    PAIR._L3_MEMO.clear()


@pytest.fixture()
def store(tmp_path):
    return PAIR.L3MeasurementStore(tmp_path / "l3_cache")


def _row(reused, *, simulator="gsim", citable=True, arm="baseline"):
    provenance = {"tier": "L3", "simulator": simulator, "reused_measurement": reused}
    return {"phase": "tuning", "arm": arm, "family": "f", "capsule": "c", "replicate": "r000",
            "simulator": simulator, "citable": citable, "provenance": provenance}


def test_a_fresh_measurement_says_it_was_measured_here(tmp_path, monkeypatch, store):
    """SILENCE IS NOT ONE OF THE ANSWERS. An absent key used to mean both 'measured here' and
    'nobody recorded which', and a reader could audit neither."""
    evidence, _ = _Engine(tmp_path, monkeypatch).measure(store)
    assert evidence["gsim"]["reused_measurement"] is False


def test_an_identical_program_is_carried_and_every_record_says_so(tmp_path, monkeypatch, store):
    """MUTATION 1 of 3: the same bytes on the same pin -> a hit, reported at every level."""
    engine = _Engine(tmp_path, monkeypatch)
    engine.measure(store)
    evidence, result = engine.measure(store)
    assert engine.calls == 1, "the engine ran twice for one program"
    assert result["cycles"] == 4242 and result["reused_measurement"] is True
    stamp = evidence["gsim"]["reused_measurement"]
    assert PAIR._is_sha256(stamp["measured_program_sha256"])
    # ... and it reaches the campaign-level accounting, which is what a reader reads
    report = PAIR.reuse_report([_row(False), _row(stamp)])
    assert (report["measured_here"], report["carried"], report["auditable"]) == (1, 1, True)
    assert report["carried_cells"][0]["measured_program_sha256"] == stamp["measured_program_sha256"]


def test_one_edited_byte_is_measured_again(tmp_path, monkeypatch, store):
    """MUTATION 2 of 3: change one byte of the emitted program and the cache must not answer."""
    engine = _Engine(tmp_path, monkeypatch)
    engine.measure(store)
    edited = json.loads(json.dumps(CB))
    edited["commands"][0]["operands"]["dst"] = "Z"
    evidence, _ = engine.measure(store, cb=edited)
    assert engine.calls == 2 and evidence["gsim"]["reused_measurement"] is False
    engine.measure(store, llvm="module { }")
    assert engine.calls == 3, "a one-byte lowered-module edit was answered from the cache"


def test_a_different_engine_or_pin_is_measured_again(tmp_path, monkeypatch, store):
    """MUTATION 3 of 3: a cycle count is a fact about a program AND the build that ran it."""
    engine = _Engine(tmp_path, monkeypatch)
    engine.measure(store)
    other = _Engine(tmp_path, monkeypatch, name="engineB")
    assert other.pin != engine.pin
    other.measure(store)
    assert other.calls == 1, "a measurement was reused across simulator builds"
    # and a stored row whose RTL pin was edited is refused even at the right address
    key = PAIR._l3_memo_key(CB, "module {}", engine.pin, SCOPE)
    record = json.loads((store.root / f"{key}.json").read_text(encoding="utf-8"))
    record["engine_pins"]["gsim_firrtl"] = "e" * 64
    (store.root / f"{key}.json").write_text(json.dumps(record), encoding="utf-8")
    PAIR._L3_MEMO.clear()
    engine.activate()
    engine.measure(store)
    assert engine.calls == 2, "a measurement was reused across a changed RTL pin"


def test_a_cell_that_states_neither_refuses_the_campaign():
    """FAIL CLOSED. An unstamped cited row is a hole in the record, not a freshly measured row."""
    report = PAIR.reuse_report([_row(False), _row(None), _row({"measured_program_sha256": "nope"})])
    assert report["auditable"] is False
    assert report["unstated"] == 2 and report["measured_here"] == 1 and report["carried"] == 0


def test_the_correctness_screen_is_not_counted_as_a_cited_cell():
    """Spike carries no timing; counting its rows would inflate both halves of the accounting."""
    report = PAIR.reuse_report([_row(None, simulator="spike", citable=False), _row(False)])
    assert report["cited_cells"] == 1 and report["auditable"] is True


def test_the_headline_pairs_say_which_half_of_the_ratio_was_carried():
    stamp = {"measured_program_sha256": "a" * 64}
    rows = [_row(stamp, arm="baseline"), _row(False, arm="candidate")]
    pair = PAIR.paired_cycle_rows([{**row, "cycles": 10, "correct": True} for row in rows])[0]
    assert pair["baseline_carried"] is True and pair["candidate_carried"] is False


# =====================================================================================
# 3. fan-out: same campaign, less wall clock -- or a refusal to fan out at all
# =====================================================================================
def _plan(tmp_path: Path, members: int = 3):
    descriptor = {"operation": {"op": "movement", "attributes": {"src": "X", "out": "Y"}},
                  "inputs": [{"name": "X", "shape": [1], "dtype": "i8"}],
                  "numeric_policy": {"compare": "exact_int"}}
    loaded = []
    for index in range(members):
        source = (tmp_path / f"m{index}").resolve()
        source.mkdir(parents=True, exist_ok=True)
        (source / "capsule.yaml").write_text(yaml.safe_dump(descriptor), encoding="utf-8")
        loaded.append(SimpleNamespace(family="f", capsule=f"m{index}", source_dir=source,
                                      source_sha256=str(index) * 64, descriptor=descriptor))
    workloads = [PAIR._gsim_workload(member) for member in loaded]
    certificate = SimpleNamespace(
        sha256="d" * 64, unresolved={}, pins={},
        members={PAIR.GATE.workload_sha256(workload): {} for workload in workloads})
    baseline, candidate = (tmp_path / "baseline").resolve(), (tmp_path / "candidate").resolve()
    inputs = PAIR.PairedInputs(
        SimpleNamespace(run_id="functional", digest="a" * 64),
        SimpleNamespace(record_sha256="b" * 64), SimpleNamespace(capsules=tuple(loaded)),
        "held_out", baseline, "a" * 64, candidate, "c" * 64, certificate)
    return PAIR.build_measurement_plan(inputs)


def _mock_measurement(spec, cycles: int) -> dict:
    digest = hashlib.sha256(f"{spec.pair_id}{spec.arm}".encode()).hexdigest()
    return {"status": "pass", "numeric": "pass", "work_volume": {},
            "command_buffer_artifact": None, "gsim_qualification": {"admitted": True},
            "per_sim": {
                "spike": {"correct": True, "cycles": None, "correctness_cycles": 5,
                          "provenance": {"tier": "L2", "simulator": "spike"}},
                "gsim": {"correct": True, "cycles": cycles,
                         "provenance": {"tier": "L3", "simulator": "gsim",
                                        "oracle_kind": "rtl_gsim", "derived_from_rtl": True,
                                        "cycle_accurate": True, "elf_sha256": digest,
                                        "reused_measurement": False}}}}


def _run(plan, out_dir: Path, *, width: int, hardware_counters: bool = False, hook=None):
    out_dir.mkdir(parents=True)
    seen: list[int] = []

    def executor(spec, workspace, *_args, **kwargs):
        workspace.mkdir(parents=True)
        seen.append(int(kwargs.get("workers") or 0))
        if hook is not None:
            hook()
        return {"schema": "mock", "execution": spec.as_dict(),
                "measurement": _mock_measurement(spec, 100 + spec.execution_index)}

    fanout = PAIR.schedule_fanout(width, plan, hardware_counters=hardware_counters)
    rows, roofline = PAIR.execute_schedule(
        plan, out_dir, timeout=1, target_experiment=object(), rtl_identity={},
        hardware_counters=hardware_counters, executor=executor, progress=lambda _line: None,
        fanout=fanout)
    return rows, roofline, fanout, seen


def test_a_wide_run_records_exactly_what_the_serial_run_records(tmp_path):
    """Cycles are invariant under fan-out; so, here, is every byte of the recorded campaign."""
    plan = _plan(tmp_path)
    serial_rows, _, serial_fanout, _ = _run(plan, tmp_path / "serial", width=1)
    wide_rows, _, wide_fanout, _ = _run(plan, tmp_path / "wide", width=8)
    assert serial_fanout["effective"] == 1 and wide_fanout["effective"] == 8
    # the raw-result PATH names the run directory and nothing else; every other byte must match
    def _comparable(text: str, run: str) -> str:
        return text.replace(str(tmp_path / run), "<run>")

    assert (_comparable(json.dumps(wide_rows, sort_keys=True), "wide")
            == _comparable(json.dumps(serial_rows, sort_keys=True), "serial"))
    for name in ("raw_results.index.json", "paired_completion_cells.json"):
        assert (_comparable((tmp_path / "wide" / name).read_text(encoding="utf-8"), "wide")
                == _comparable((tmp_path / "serial" / name).read_text(encoding="utf-8"), "serial")), (
            f"{name} depends on the fan-out")
    assert PAIR.completion_report(wide_rows, plan.expected)["complete"] is True


def test_a_wide_run_actually_overlaps_its_executions(tmp_path):
    """Otherwise the flag is inert: measured, not assumed, by counting concurrent entries."""
    plan = _plan(tmp_path)
    live, peak, lock = 0, 0, threading.Lock()
    barrier = threading.Barrier(4, timeout=30)

    def hook():
        nonlocal live, peak
        with lock:
            live += 1
            peak = max(peak, live)
        barrier.wait()
        with lock:
            live -= 1

    _run(plan, tmp_path / "wide", width=4, hook=hook)
    assert peak >= 4, f"executions never overlapped (peak {peak})"


def test_every_row_is_stamped_with_the_width_it_actually_ran_at(tmp_path):
    """A timing block is only comparable to another run's if both say what they contended with."""
    plan = _plan(tmp_path)
    _, _, fanout, widths = _run(plan, tmp_path / "wide", width=3)
    assert set(widths) == {fanout["effective"]} == {3}


def test_hardware_counters_refuse_to_fan_out(tmp_path):
    """The pass is selected by PROCESS environment variables; two of them cannot both be set."""
    plan = _plan(tmp_path)
    fanout = PAIR.schedule_fanout(16, plan, hardware_counters=True)
    assert fanout["effective"] == 1 and "environment" in fanout["reason"]


def test_a_relative_input_path_refuses_to_fan_out(tmp_path, monkeypatch):
    """Some capsule paths chdir the process; a relative path resolved in that window names another
    file. Measured on the functional grader: 18 of 26 capsules wrote into a sibling checkout."""
    plan = _plan(tmp_path)
    relative = [PAIR.copy.copy(spec) for spec in plan.schedule]
    object.__setattr__(relative[0], "package", Path("baseline"))
    narrowed = PAIR.MeasurementPlan(plan.phase, tuple(relative), plan.expected,
                                    plan.declaration, plan.declaration_sha256)
    fanout = PAIR.schedule_fanout(16, narrowed, hardware_counters=False)
    assert fanout["effective"] == 1 and "relative" in fanout["reason"]


def test_a_width_below_one_is_refused_rather_than_rounded(tmp_path):
    with pytest.raises(PAIR.PC.CampaignGateError):
        PAIR.schedule_fanout(0, _plan(tmp_path), hardware_counters=False)


def test_the_declared_width_reaches_the_measurement_child(monkeypatch, tmp_path):
    """A width nobody forwards is inert; this is the flag the child actually receives."""
    _, _, commands = _measure_waves(monkeypatch, tmp_path, workers=1, sim_workers=12)
    assert commands and all("--sim-workers" in command and
                            command[command.index("--sim-workers") + 1] == "12"
                            for command in commands)


def test_an_unset_width_leaves_the_campaign_serial():
    source = (_SCRIPTS / "run_paired_perf_bench.py").read_text(encoding="utf-8")
    assert '"--sim-workers", type=int, default=1' in source, (
        "--sim-workers no longer defaults to serial; a formal campaign's width must be DECLARED")
