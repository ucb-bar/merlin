"""The private full-model execution gate: declaration, judgement, and revalidation (no simulator runs)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from merlin_experiments.phase1.feedback import private_full_model_execution as PFX
from merlin_experiments.phase1.feedback import private_full_models as PFM

from merlin.common.paths import repo_root

ROSTER = {"net": ("model",), "policy": ("encode", "step")}
EXECUTED = {
    "engine": "gsim",
    "max_cycles": 1000,
    "timeout_s": 60,
    "integer_reference": ["integer-reference.json"],
    "host_execution": {"atol": 0.01, "rtol": 0.0},
    "fp32_sanity": {"reference": "float_reference.json", "cosine_margin": 4, "top1": True},
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _descriptor(tmp_path: Path, programs: dict) -> Path:
    path = tmp_path / "descriptor.yaml"
    document = {"target": "t", "phase1_gates": {"whole_model": {"private_full_models": {"programs": programs}}}}
    path.write_text(yaml.safe_dump(document, sort_keys=False))
    return path


def _programs(**overrides) -> dict:
    programs = {
        "net": {"model": dict(EXECUTED)},
        "policy": {"encode": {"deferred": "two outputs"}, "step": dict(EXECUTED, engine="spike")},
    }
    for key, value in overrides.items():
        model, program = key.split("__")
        programs[model][program] = value
    return programs


# ------------------------------------------------------------------------------------------ declaration


def test_the_target_descriptor_declares_every_program_of_the_static_roster():
    descriptor = repo_root() / "examples" / "gemmini" / "target" / "descriptor.yaml"
    roster = PFM.program_requirements_for(descriptor)
    gate = PFX.gate_for(descriptor, required_programs=roster)
    assert gate is not None and gate["required"] is True
    declared = [(model, program) for model, program, _ in PFX.roster_of(gate)]
    assert declared == [(model, program) for model, names in roster.items() for program in names]
    engines = {(m, p): e.get("engine") for m, p, e in PFX.roster_of(gate)}
    # Every program has a numerical check: one on the RTL-equivalent engine, the rest functional.
    assert engines == {
        ("resnet50", "model"): "gsim",
        **{(m, p): "spike" for m, names in roster.items() if m != "resnet50" for p in names},
    }
    for model, program, entry in PFX.roster_of(gate):
        if entry["engine"] == "spike":
            assert entry["board"] == "gemmini_rocket_spike_functional"
        else:
            assert entry["board"] == "gemmini_rocket_verilator" and entry["max_cycles"] > 0
        assert entry["group_profile"] is True and entry["timeout_s"] > 0
        # Correctness is the quantized program's semantics: its integer reference, value-exact, then a
        # host execution of the same quantized program; the float model is only a calibrated sanity bound.
        assert entry["integer_reference"] == ["integer-reference.json"]
        assert entry["host_execution"] is not None
        assert entry["fp32_sanity"]["reference"] == "float_reference.json"
        assert entry["fp32_sanity"]["top1"] is (model == "resnet50")
        assert entry["require_integer_reference"] is ((model, program) != ("smolvla", "action_decode"))


def test_a_descriptor_without_the_declaration_has_no_execution_gate(tmp_path):
    path = tmp_path / "descriptor.yaml"
    path.write_text("target: t\nphase1_gates:\n  private_full_models: {required: false}\n")
    assert PFX.gate_for(path, required_programs=ROSTER) is None
    assert PFX.gate_for(tmp_path / "absent.yaml", required_programs=ROSTER) is None


def test_a_valid_declaration_is_normalized(tmp_path):
    gate = PFX.gate_for(_descriptor(tmp_path, _programs()), required_programs=ROSTER)
    entries = {(m, p): e for m, p, e in PFX.roster_of(gate)}
    assert entries[("policy", "encode")] == {"deferred": "two outputs", "estimate": None}
    assert entries[("net", "model")]["require_integer_reference"] is False
    assert entries[("policy", "step")]["engine"] == "spike" and entries[("policy", "step")]["max_cycles"] == 1000


@pytest.mark.parametrize(
    "programs,match",
    [
        ({"net": {"model": dict(EXECUTED)}}, "different model roster"),
        (_programs(policy__step=None) | {"policy": {"encode": {"deferred": "x"}}}, "every program"),
        (_programs(policy__encode={"deferred": "  "}), "state its reason"),
        (_programs(policy__encode={"deferred": "x", "engine": "gsim"}), "not both"),
        (_programs(net__model=dict(EXECUTED, engine="board")), "engine must be one of"),
        (_programs(net__model=dict(EXECUTED, timeout_s=0)), "timeout_s"),
        (_programs(net__model=dict(EXECUTED, integer_reference=["../x.npy"])), "plain capture file name"),
        (_programs(net__model=dict(EXECUTED, require_integer_reference=True, integer_reference=[])), "needs a named"),
        (_programs(net__model=dict(EXECUTED, host_execution={"atol": 0.1})), "exactly atol and rtol"),
        (_programs(net__model={k: v for k, v in EXECUTED.items() if k != "host_execution"}), "host_execution"),
        (
            _programs(net__model=dict(EXECUTED, fp32_sanity=dict(EXECUTED["fp32_sanity"], cosine_margin=0.5))),
            "at least 1",
        ),
        (_programs(net__model=dict(EXECUTED, end_to_end={})), "unknown key"),
        (_programs(net__model=dict(EXECUTED, surprise=1)), "unknown key"),
        (
            _programs(net__model={"deferred": "x"}, policy__step={"deferred": "y"}),
            "would check nothing",
        ),
    ],
)
def test_a_malformed_declaration_is_refused_by_name(tmp_path, programs, match):
    with pytest.raises(PFX.ExecutionGateError, match=match):
        PFX.gate_for(_descriptor(tmp_path, programs), required_programs=ROSTER)


# ------------------------------------------------------------------------------------------ the run


def _frame(name: str, values: np.ndarray) -> bytes:
    raw = np.ascontiguousarray(values, dtype="<f4").tobytes()
    digest = 0xCBF29CE484222325
    for byte in raw:
        digest = ((digest ^ byte) * 0x100000001B3) & ((1 << 64) - 1)
    return (
        f"OUT_BIN_BEGIN v1 {name} 1 {values.size} 4 u {len(raw)}\n".encode()
        + raw
        + f"OUT_BIN_END v1 {digest:016x}\n".encode()
    )


def _console(*results: np.ndarray, build_hash: str = "bh", cycles: int = 1200) -> bytes:
    """A full-readback console: one OUT_BIN frame per result, then the metrics and DONE."""
    frames = b"".join(_frame(f"out{i}", np.asarray(v, np.float32).reshape(-1)) for i, v in enumerate(results))
    tail = f"METRIC build_hash {build_hash}\nMETRIC memref_rank_mismatch 0\nMETRIC cycles {cycles}\n"
    return frames + tail.encode() + b"GROUP_ID 0 g1_dev_0\nGAP 0 10\nGROUP 0 90\nGAP 1 5\nDONE\n"


@pytest.fixture(autouse=True)
def _result_roster(monkeypatch):
    """The forward's results, as the capture's ``results.json`` states them (a test stand-in for the
    signature the gate reads from ``model.mlir``), and the host execution of the saved program, as the
    capture's ``host.npz`` states it (a stand-in for executing it)."""

    def specs(path):
        return [(shape, "f32") for shape in json.loads((Path(path).parent / "results.json").read_text())]

    def hosted(capture):
        archive = np.load(Path(capture) / "host.npz")
        return [archive[f"out{i}"] for i in range(len(archive.files))]

    monkeypatch.setattr("merlin.llvmlower.c_runtime._out_specs", specs)
    monkeypatch.setattr(PFX, "_host_outputs_in_scratch", hosted)


def _as_list(value) -> list:
    return [np.asarray(v, np.float32) for v in value] if isinstance(value, list) else [np.asarray(value, np.float32)]


def _capture(root: Path, *, golden, integer=None, fp32=None, host=None, results=None) -> Path:
    """A capture: golden.npy and (when given) integer-reference.json bound by its receipt; the float
    model's float_reference.json beside it, unbound, as the capture worker writes it."""
    capture = root / "capture"
    capture.mkdir(parents=True)
    golden = np.asarray(golden, np.float32)
    shapes = results or [list(golden.shape)]
    (capture / "results.json").write_text(json.dumps(shapes))
    np.save(capture / "golden.npy", golden)
    artifacts = {"golden.npy": {"sha256": _sha(capture / "golden.npy")}}
    if integer is not None:
        values = _as_list(integer)
        document = {
            "schema": PFX.INTEGER_REFERENCE_SCHEMA,
            "output_abi": [{"dtype": "f32", "shape": list(v.shape)} for v in values],
            "outputs": [v.tolist() for v in values],
        }
        (capture / "integer-reference.json").write_text(json.dumps(document))
        artifacts["integer-reference.json"] = {"sha256": _sha(capture / "integer-reference.json")}
    float_values = _as_list(golden if fp32 is None else fp32)
    document = float_values[0].tolist() if len(shapes) == 1 else [v.tolist() for v in float_values]
    (capture / "float_reference.json").write_text(json.dumps(document))
    hosted = _as_list(golden if host is None else host)
    np.savez(capture / "host.npz", **{f"out{i}": v for i, v in enumerate(hosted)})
    (capture / "capture_receipt.json").write_text(json.dumps({"artifacts": artifacts}))
    return capture


class Fixture:
    def __init__(
        self,
        tmp_path: Path,
        *,
        golden: np.ndarray,
        integer=None,
        fp32=None,
        host=None,
        results: list[list[int]] | None = None,
        readback: str = "full",
    ):
        self.static_out = tmp_path / "static"
        self.static_rows = []
        programs = {"net": ["model"], "policy": ["encode", "step"]}
        for model, names in programs.items():
            entries = []
            for program in names:
                where = self.static_out / model / program
                (where / "build").mkdir(parents=True)
                elf = where / "build" / "model.elf"
                elf.write_bytes(f"{model}:{program}".encode())
                capture = _capture(where, golden=golden, integer=integer, fp32=fp32, host=host, results=results)
                receipt = {
                    "inputs": {"capture": str(capture), "board": "fixture_board", "group_profile": True},
                    "output": {"elf": str(elf), "elf_sha256": _sha(elf), "readback": readback},
                }
                receipt["output"]["build_hash"] = "bh"
                (where / PFX.RECEIPT).write_text(json.dumps(receipt))
                entries.append({"program": program, "elf_sha256": _sha(elf)})
            build = {"programs": entries, "accelerator_rtl_config": "Cfg", "accelerator_rtl_facts_sha256": "f" * 64}
            self.static_rows.append({"model": model, "status": "pass", "checks": {"build": build}})
        self.static = {"candidate_tree_sha256": "c" * 64, "models": self.static_rows}


def _gate(tmp_path: Path, **overrides) -> dict:
    return PFX.gate_for(_descriptor(tmp_path, _programs(**overrides)), required_programs=ROSTER)


def test_executed_programs_pass_and_deferrals_are_recorded_not_passed(tmp_path):
    golden = np.array([1.0, -2.0, 0.5], np.float32)
    fixture = Fixture(tmp_path, golden=golden, integer=golden.copy())
    calls = []

    def execute(elf, **kwargs):
        calls.append((Path(elf).read_text(), kwargs["engine"], kwargs["max_cycles"], kwargs["rtl_config"]))
        return {"console": _console(golden), "wall_s": 2.0, "engine": kwargs["engine"]}

    gate = _gate(tmp_path)
    result = PFX.run(fixture.static, gate, target="t", static_out=fixture.static_out, execute=execute)
    assert calls == [("net:model", "gsim", 1000, "Cfg"), ("policy:step", "spike", 1000, "Cfg")]
    assert result["passed"] is True and result["deferred"] == ["policy:encode"]
    rows = {(r["model"], r["program"]): r for r in result["programs"]}
    assert rows[("policy", "encode")]["status"] == PFX.DEFERRED
    net = rows[("net", "model")]
    assert net["status"] == "pass" and net["cycles"] == 1200 and net["simulated_cycles_per_s"] == 600.0
    integer = net["checks"]["integer_reference"]
    assert integer["status"] == "compared" and integer["reference"] == "integer-reference.json"
    assert integer["passed"] is True
    assert integer["results"] == [{"index": 0, "passed": True, "mismatched_elements": 0, "of": 3}]
    assert net["checks"]["host_execution"]["results"] == []  # every result had an integer reference
    sanity = net["checks"]["fp32_sanity"]
    assert sanity["passed"] is True and sanity["receipt_bound"] is False and sanity["top1"]["passed"] is True
    assert net["checks"]["per_group_exactness"]["status"] == "not_observable"
    assert "deferrals are not numerical evidence" in result["numerical_scope"]
    assert PFX.complete(result, gate, static=fixture.static, candidate_sha256="c" * 64)
    # No tensor value is carried by the record.
    assert "1.0" not in json.dumps(result["programs"][0]["checks"]["integer_reference"])


def test_an_integer_reference_is_value_exact_and_absent_falls_back_to_host_execution(tmp_path):
    golden = np.array([1.0, 2.0], np.float32)
    off_by_one_ulp = np.nextafter(golden, np.float32(3.0))
    fixture = Fixture(tmp_path, golden=golden, integer=golden)

    def execute(elf, **kwargs):
        return {"console": _console(off_by_one_ulp), "wall_s": 1.0}

    result = PFX.run(fixture.static, _gate(tmp_path), target="t", static_out=fixture.static_out, execute=execute)
    row = result["programs"][0]
    integer = row["checks"]["integer_reference"]
    assert row["status"] == "fail" and integer["results"][0]["mismatched_elements"] == 2
    assert row["checks"]["fp32_sanity"]["passed"] is True  # close to the float model, still not exact
    assert result["passed"] is False
    signed_zero = Fixture(tmp_path / "zero", golden=np.array([0.0], np.float32), integer=np.array([-0.0], np.float32))
    result = PFX.run(
        signed_zero.static,
        _gate(tmp_path),
        target="t",
        static_out=signed_zero.static_out,
        execute=lambda elf, **kw: {"console": _console(np.array([0.0], np.float32)), "wall_s": 1.0},
    )
    assert result["programs"][0]["checks"]["integer_reference"]["passed"] is True

    fixture = Fixture(tmp_path / "no-int", golden=golden)
    result = PFX.run(
        fixture.static,
        _gate(tmp_path),
        target="t",
        static_out=fixture.static_out,
        execute=lambda elf, **kw: {"console": _console(golden), "wall_s": 1.0},
    )
    assert result["passed"] is True
    checks = result["programs"][0]["checks"]
    assert checks["integer_reference"]["status"] == "not_available"
    assert checks["host_execution"]["results"][0]["passed"] is True
    required = dict(EXECUTED, require_integer_reference=True)
    result = PFX.run(
        fixture.static,
        _gate(tmp_path, net__model=required),
        target="t",
        static_out=fixture.static_out,
        execute=lambda elf, **kw: {"console": _console(golden), "wall_s": 1.0},
    )
    assert result["programs"][0]["status"] == "fail"


def test_an_integer_reference_the_host_execution_does_not_reproduce_falls_back_to_it(tmp_path):
    reference = np.array([1.0, 2.0, 3.0], np.float32)
    program = np.array([1.0, 2.03125, 3.0], np.float32)  # the saved program's own float nonlinear arithmetic
    fixture = Fixture(tmp_path, golden=reference, integer=reference, host=program)

    def run(printed):
        execute = lambda elf, **kw: {"console": _console(printed), "wall_s": 1.0}  # noqa: E731
        return PFX.run(fixture.static, _gate(tmp_path), target="t", static_out=fixture.static_out, execute=execute)

    passing = run(program)["programs"][0]
    row = passing["checks"]["integer_reference"]["results"][0]
    assert passing["status"] == "pass" and row["status"] == "unconfirmed_reference"
    assert row["reference_confirmed_by_host_execution"] is False and row["host_mismatched_elements"] == 1
    assert passing["checks"]["host_execution"]["results"][0]["index"] == 0
    failing = run(program + np.float32(0.25))["programs"][0]
    assert failing["status"] == "fail" and failing["checks"]["host_execution"]["passed"] is False


def test_results_without_an_integer_reference_are_held_to_host_execution_and_completeness(tmp_path):
    golden = np.array([1.0, 2.0, 3.0], np.float32)
    fixture = Fixture(tmp_path, golden=golden)
    far = golden + np.float32(0.5)
    result = PFX.run(
        fixture.static,
        _gate(tmp_path),
        target="t",
        static_out=fixture.static_out,
        execute=lambda elf, **kw: {"console": _console(far), "wall_s": 1.0},
    )
    hosted = result["programs"][0]["checks"]["host_execution"]["results"][0]
    assert hosted["passed"] is False and hosted["within"] == 0 and hosted["of"] == 3
    partial = PFX.run(
        fixture.static,
        _gate(tmp_path),
        target="t",
        static_out=fixture.static_out,
        execute=lambda elf, **kw: {"console": _console(golden[:2]), "wall_s": 1.0},
    )
    assert "printed 2 of 3" in partial["programs"][0]["checks"]["completion"]["note"]
    stale = PFX.run(
        fixture.static,
        _gate(tmp_path),
        target="t",
        static_out=fixture.static_out,
        execute=lambda elf, **kw: {"console": _console(golden, build_hash="other"), "wall_s": 1.0},
    )
    assert stale["programs"][0]["checks"]["completion"]["passed"] is False


def test_the_float_model_is_a_sanity_bound_calibrated_from_the_quantized_program():
    fp32 = [np.array([[1.0, 2.0, 3.0, 4.0]])]
    quantized = [np.array([[1.1, 2.0, 2.9, 4.0]])]  # the quantized program's own result
    sanity = {"cosine_margin": 4.0, "top1": True}
    own = PFX._cosine(quantized[0], fp32[0])
    near = PFX._fp32_sanity(quantized, quantized, fp32, sanity)
    assert near["passed"] is True
    assert near["results"][0]["floor"] == pytest.approx(1 - 4 * (1 - own))
    worse = [np.array([[1.4, 2.0, 2.5, 4.0]])]
    assert PFX._cosine(worse[0], fp32[0]) < 1 - 4 * (1 - own)
    assert PFX._fp32_sanity(worse, quantized, fp32, sanity)["passed"] is False
    # Top-1 counts only rows where the quantized program itself keeps the float model's top-1.
    flipped = [np.array([[1.0, 2.0, 3.0, 4.0], [4.0, 3.9, 0.0, 0.0]])]
    own_rows = [np.array([[1.0, 2.0, 3.0, 4.1], [3.9, 4.0, 0.0, 0.0]])]
    device = [np.array([[1.0, 2.0, 3.0, 4.1], [3.9, 4.0, 0.0, 0.0]])]
    top1 = PFX._fp32_sanity(device, own_rows, flipped, sanity)["top1"]
    assert top1 == {
        "rows": 2,
        "applicable_rows": 1,
        "agreeing_rows": 1,
        "passed": True,
        "note": "rows where the quantized program's own top-1 differs from the float model are excluded",
    }
    wrong = [np.array([[1.0, 2.0, 4.2, 4.1], [3.9, 4.0, 0.0, 0.0]])]
    assert PFX._fp32_sanity(wrong, own_rows, flipped, sanity)["top1"]["passed"] is False


def test_the_executed_elf_must_be_the_one_the_static_gate_verified(tmp_path):
    golden = np.array([1.0], np.float32)
    fixture = Fixture(tmp_path, golden=golden)
    (fixture.static_out / "net" / "model" / "build" / "model.elf").write_bytes(b"rebuilt")
    ran = []
    result = PFX.run(
        fixture.static,
        _gate(tmp_path),
        target="t",
        static_out=fixture.static_out,
        execute=lambda elf, **kw: ran.append(elf) or {"console": _console(golden), "wall_s": 1.0},
    )
    row = result["programs"][0]
    assert row["status"] == "fail" and "differs from the bytes the static gate verified" in row["reason"]
    assert all("net" not in str(path) for path in ran)


def test_a_model_that_failed_the_static_gate_is_not_run_and_fails_the_gate(tmp_path):
    golden = np.array([1.0], np.float32)
    fixture = Fixture(tmp_path, golden=golden)
    fixture.static_rows[0]["status"] = "fail"
    result = PFX.run(
        fixture.static,
        _gate(tmp_path),
        target="t",
        static_out=fixture.static_out,
        execute=lambda elf, **kw: {"console": _console(golden), "wall_s": 1.0},
    )
    assert result["programs"][0]["status"] == PFX.NOT_RUN and result["passed"] is False


def test_an_engine_failure_is_a_failed_program_not_a_crash(tmp_path):
    golden = np.array([1.0], np.float32)
    fixture = Fixture(tmp_path, golden=golden)

    def execute(elf, **kwargs):
        raise TimeoutError("hang bound reached")

    result = PFX.run(fixture.static, _gate(tmp_path), target="t", static_out=fixture.static_out, execute=execute)
    assert [r["status"] for r in result["programs"]] == ["fail", PFX.DEFERRED, "fail"]
    assert "hang bound reached" in result["programs"][0]["reason"]


def test_the_static_gate_links_every_executed_program_with_full_readback(tmp_path):
    gate = _gate(tmp_path, net__model=dict(EXECUTED, group_profile=False))
    assert PFX.build_options(gate) == {
        "net": {"model": {"readback": "full", "group_profile": False}},
        "policy": {"step": {"readback": "full", "group_profile": True}},
    }


def test_results_the_integer_reference_omits_are_held_to_the_host_execution(tmp_path, monkeypatch):
    golden, second = np.array([1.0, -2.0], np.float32), np.array([0.25, 4.0, 8.0], np.float32)
    fixture = Fixture(tmp_path, golden=golden, integer=golden, results=[[2], [3]], fp32=[golden, second])
    hosted = []

    def host(capture):
        hosted.append(capture)
        return [golden, second]

    monkeypatch.setattr(PFX, "_host_outputs_in_scratch", host)
    gate = _gate(
        tmp_path,
        net__model=dict(EXECUTED, require_integer_reference=True, host_execution=None),
        policy__step=dict(EXECUTED, engine="spike"),
    )

    def execute(elf, **kwargs):
        assert kwargs["capture_bytes"] is True
        return {"console": _console(golden, second), "wall_s": 1.0, "engine": kwargs["engine"]}

    result = PFX.run(fixture.static, gate, target="t", static_out=fixture.static_out, execute=execute)
    net, _, step = result["programs"]
    assert step["status"] == "pass" and len(hosted) == 1
    assert step["checks"]["integer_reference"]["results"][0]["passed"] is True
    assert step["checks"]["host_execution"]["results"][0]["index"] == 1
    assert [row["index"] for row in step["checks"]["fp32_sanity"]["results"]] == [0, 1]
    assert step["scope"].startswith("spike-functional") and step["certification"] is False
    assert step["group_profile"]["groups"][0] == {"index": 0, "name": "g1_dev_0", "cycles": 90, "gap_before": 10}
    # A result with no integer reference and no declared host tolerance is not reported checked.
    assert net["status"] == "fail" and net["certification"] is True
    assert "no host_execution tolerance" in net["checks"]["host_execution"]["results"][0]["note"]


def test_a_result_that_disagrees_with_the_host_execution_fails(tmp_path):
    golden, second = np.array([1.0], np.float32), np.array([2.0], np.float32)
    fixture = Fixture(tmp_path, golden=golden, results=[[1], [1]], fp32=[golden, second], host=[golden, second])

    def execute(elf, **kwargs):
        return {"console": _console(golden, second + 0.5), "wall_s": 1.0, "engine": kwargs["engine"]}

    step = PFX.run(fixture.static, _gate(tmp_path), target="t", static_out=fixture.static_out, execute=execute)
    row = step["programs"][2]
    assert row["status"] == "fail" and row["checks"]["host_execution"]["passed"] is False


@pytest.mark.parametrize(
    ("readback", "entry", "reason"),
    [
        ("prefix", {}, "not linked with full readback"),
        ("full", {"board": "another_board"}, "linked for board 'fixture_board'"),
    ],
)
def test_a_program_not_linked_as_declared_is_not_run(tmp_path, readback, entry, reason):
    golden = np.array([1.0], np.float32)
    fixture = Fixture(tmp_path, golden=golden, readback=readback)
    ran = []

    def execute(elf, **kwargs):
        ran.append(elf)
        return {"console": _console(golden), "wall_s": 1.0, "engine": kwargs["engine"]}

    gate = _gate(tmp_path, net__model=dict(EXECUTED, **entry))
    net = PFX.run(fixture.static, gate, target="t", static_out=fixture.static_out, execute=execute)["programs"][0]
    assert net["status"] == "fail" and reason in net["reason"]
    assert all("net" not in Path(elf).read_text() for elf in ran)


def test_a_console_of_another_build_fails_completion(tmp_path):
    golden = np.array([1.0], np.float32)
    fixture = Fixture(tmp_path, golden=golden)

    def execute(elf, **kwargs):
        return {"console": _console(golden, build_hash="other"), "wall_s": 1.0, "engine": kwargs["engine"]}

    net = PFX.run(fixture.static, _gate(tmp_path), target="t", static_out=fixture.static_out, execute=execute)
    assert net["programs"][0]["status"] == "fail"
    assert net["programs"][0]["checks"]["completion"]["passed"] is False


# ------------------------------------------------------------------------------------------ revalidation


def test_complete_refuses_a_record_that_does_not_match_its_declaration(tmp_path):
    golden = np.array([1.0], np.float32)
    fixture = Fixture(tmp_path, golden=golden)
    gate = _gate(tmp_path)
    good = PFX.run(
        fixture.static,
        gate,
        target="t",
        static_out=fixture.static_out,
        execute=lambda elf, **kw: {"console": _console(golden), "wall_s": 1.0},
    )
    assert PFX.complete(good, gate, static=fixture.static, candidate_sha256="c" * 64)
    assert not PFX.complete(good, gate, static=fixture.static, candidate_sha256="d" * 64)
    reworded = json.loads(json.dumps(good))
    reworded["programs"][1]["reason"] = "something else"
    assert not PFX.complete(reworded, gate, static=fixture.static, candidate_sha256="c" * 64)
    swapped = json.loads(json.dumps(good))
    swapped["programs"][0]["elf_sha256"] = "0" * 64
    assert not PFX.complete(swapped, gate, static=fixture.static, candidate_sha256="c" * 64)
    other_engine = json.loads(json.dumps(good))
    other_engine["programs"][0]["engine"] = "spike"
    assert not PFX.complete(other_engine, gate, static=fixture.static, candidate_sha256="c" * 64)
    assert not PFX.complete(PFX.not_run(gate, "x"), gate, static=fixture.static, candidate_sha256="c" * 64)


def test_the_official_grade_requires_a_declared_execution_gate_to_be_complete(tmp_path, monkeypatch):
    from merlin_experiments.phase1.feedback import certification

    golden = np.array([1.0], np.float32)
    fixture = Fixture(tmp_path / "f", golden=golden)
    gate = _gate(tmp_path)
    record = PFX.run(
        fixture.static,
        gate,
        target="t",
        static_out=fixture.static_out,
        execute=lambda elf, **kw: {"console": _console(golden), "wall_s": 1.0},
    )
    run_dir = tmp_path / "run"
    (run_dir / "submission").mkdir(parents=True)
    candidate = "c" * 64
    monkeypatch.setattr("merlin.compile.model_execution_inputs.strict_tree_sha256", lambda _root: {"sha256": candidate})
    monkeypatch.setattr(PFM, "complete", lambda *_args, **_kwargs: True)

    def grade(execution_record):
        manifest = {
            "completion": {
                "formal_grade_complete": True,
                "required_tier": "L3",
                "required_full_models": list(ROSTER),
                "required_full_programs": {k: list(v) for k, v in ROSTER.items()},
            },
            "private_full_models": fixture.static,
            **({} if execution_record is None else {"private_full_model_execution": execution_record}),
        }
        (run_dir / "run_manifest.yaml").write_text(yaml.safe_dump(manifest))
        return certification._official_grade_result(
            0, run_dir, required_models=tuple(ROSTER), required_programs=ROSTER, execution_gate=gate
        )["failures"]

    assert "private_full_model_execution_incomplete" not in grade(record)
    assert "private_full_model_execution_incomplete" in grade(None)
    assert "private_full_model_execution_incomplete" in grade(dict(record, passed=False))
    optional = dict(gate, required=False)
    (run_dir / "run_manifest.yaml").write_text(yaml.safe_dump({"private_full_models": fixture.static}))
    failures = certification._official_grade_result(
        0, run_dir, required_models=tuple(ROSTER), required_programs=ROSTER, execution_gate=optional
    )["failures"]
    assert "private_full_model_execution_incomplete" not in failures


def test_a_hang_bound_reaches_the_engine_through_the_variable_the_backend_names(monkeypatch):
    class Backend:
        GSIM_MAXCYCLES_ENV = "FIXTURE_GSIM_MAXCYCLES"

    monkeypatch.delenv("FIXTURE_GSIM_MAXCYCLES", raising=False)
    with PFX._engine_hang_bound(Backend, "gsim", 77):
        import os

        assert os.environ["FIXTURE_GSIM_MAXCYCLES"] == "77"
    assert "FIXTURE_GSIM_MAXCYCLES" not in __import__("os").environ
    with pytest.raises(PFX.ExecutionGateError, match="no hang-bound variable"):
        with PFX._engine_hang_bound(Backend, "spike", 5):
            pass
