"""Synthetic historical provenance, not actual engine qualification."""

import base64
import contextlib
import json

import pytest
from merlin_experiments.phase2 import retained_public_measurement as M


@pytest.fixture
def selected(tmp_path):
    def write(name, value):
        path = tmp_path / name
        path.parent.mkdir(exist_ok=True, parents=True)
        path.write_text(json.dumps(value) if not isinstance(value, str) else value)
        return path

    source = write("source.mlir", "public synthetic source")
    source_commands = write("source_commands.json", {"source": "synthetic"})
    manifest = write(
        "manifest.json",
        {
            "members": [
                {
                    "identity": "public_control",
                    "source_sha256": M.pin(source)["sha256"],
                    "command_buffer_path": str(source_commands),
                    "command_buffer_sha256": M.digest({"source": "synthetic"}),
                }
            ]
        },
    )
    generator = write("generator.py", "# independent public generator")
    declaration = write(
        "declaration.json",
        {
            "schema": "merlin.public_development_source_selection.v1",
            "phase": "phase2",
            "visibility": "public",
            "manifest": M.pin(manifest),
            "generators": [M.pin(generator)],
            "source_groups": {M.pin(source)["sha256"]: "test"},
        },
    )
    elf = write("elf", "synthetic linked bytes")
    executable = write("sim", "synthetic independent simulator")
    clock = write("clock.scala", "synthetic counter source")
    selection = {
        "schema": "merlin.kernel_call_timing.v1",
        "selection_sha256": "1" * 64,
        "counter_asm": "counter",
        "boundary": "immediately before kernel entry through explicitly selected completion",
        "max_delta_cycles": 10000,
        "counter_semantics": {"unit": "cycles", "width_bits": 64, "wrap": "modulo", "source_pins": [M.pin(clock)]},
    }
    encoded = base64.b64encode(json.dumps(selection).encode()).decode()
    write(
        "harness.c",
        f"/* merlin.kernel_call_timing.v1 selection_sha256={'1' * 64} "
        f"boundary=entry-through-completion record_b64={encoded} */",
    )
    facts, observed = write("facts.json", {}), write("observed.json", {})
    write("commands.json", {"source": "synthetic"})
    write("inputs.npz", "synthetic signed input bytes")
    write("lowering_recipe.json", {})
    review = {
        "schema": "merlin.counter_window_review.v1",
        "elf_sha256": M.pin(elf)["sha256"],
        "begin_pc": 10,
        "entry_call_pc": 20,
        "completion_call_pc": 30,
        "end_pc": 40,
        "counter_asm": "counter",
        "counter_reads": [{"pc": 10}, {"pc": 40}],
        "program_counter_reads": 2,
        "completion_contains_fence": True,
    }
    write("counter-protocol-audit.json", review)
    dump = write("dump", "complete synthetic dump")
    stdout, stderr = write("stdout", "METRIC cycles 123\nDONE\n"), write("stderr", "")
    region = write("regions", "addresses")
    invocation = write(
        "invocation.json",
        {
            "schema": "merlin.invocation_record.v1",
            "kind": "subprocess",
            "status": "completed",
            "returncode": 0,
            "inputs_unchanged": True,
            "dependencies_unchanged": True,
            "executable_unchanged": True,
            "executable": M.pin(executable),
            "inputs": [M.pin(elf), M.pin(region)],
            "outputs": [],
            "dependencies": [M.pin(generator)],
            "stdout": M.pin(stdout),
            "stderr": M.pin(stderr),
            "argv": [str(executable), str(elf), "+dump-out=" + str(dump)],
        },
    )
    kernel_object = write("kernel.o", "original synthetic object")
    build = {
        "object_sha256": M.pin(kernel_object)["sha256"],
        "elf": str(elf),
        "elf_sha256": M.pin(elf)["sha256"],
        "inputs_sha256": M.pin(tmp_path / "inputs.npz")["sha256"],
        "command_buffer_sha256": M.digest({"source": "synthetic"}),
        "counter_audit_sha256": M.pin(tmp_path / "counter-protocol-audit.json")["sha256"],
        "lowering_recipe_sha256": M.pin(tmp_path / "lowering_recipe.json")["sha256"],
        "simulators": {"gsim": M.pin(executable)["sha256"]},
        "context": {
            "source_sha256": M.pin(source)["sha256"],
            "facts_sha256": M.pin(facts)["sha256"],
            "contract_sha256": M.pin(observed)["sha256"],
        },
    }
    write("build-receipt.json", build)
    write(
        "readback_build.json",
        {
            "schema": "merlin_readback_build_v2",
            "elf_sha256": build["elf_sha256"],
            "harness_sha256": M.pin(tmp_path / "harness.c")["sha256"],
            "kernel_object_name": "kernel.o",
            "kernel_object_sha256": build["object_sha256"],
        },
    )
    path = write(
        "measurements/label.json",
        {
            "schema": "phase2.exploratory.measurement.v1",
            "engine": "gsim",
            "cycles": 123,
            "exact": True,
            "problems": [],
            "member": "public_control",
            "source_sha256": M.pin(source)["sha256"],
            "build": build,
            "console_sha256": M.pin(stdout)["sha256"],
            "cold_warm": "UNKNOWN",
            "memory_evidence": {
                "status": "complete",
                "elf_sha256": M.pin(elf)["sha256"],
                "console_sha256": M.pin(stdout)["sha256"],
                "output_path": str(dump),
                "artifact_sha256": M.pin(dump)["sha256"],
            },
        },
    )
    domain = {
        "label_engine_sha256": M.pin(executable)["sha256"],
        "scope_sha256": M.digest(
            {
                "schema": "merlin.kernel_call_scope.v1",
                "cycle_scope": "entry-through-completion",
                "timing_selection_sha256": "1" * 64,
                "selection_record_sha256": M.digest(selection),
            }
        ),
        "memory_regime_sha256": M.digest({"cache_warmth": "unknown", "protocol": None}),
    }
    args = dict(
        manifest_path=manifest,
        source_path=source,
        declaration_path=declaration,
        invocation_path=invocation,
        facts_path=facts,
        observed_contract_path=observed,
        domain=domain,
    )
    return path, args, write


def test_intake_preserves_historical_origin_and_separate_fresh_obligations(selected):
    path, args, _ = selected
    report = M.intake(path, **args)
    assert report["measured_now"] is False and report["timing_qualification"] is False
    assert report["fresh_numerical_and_count_observation"] == "REQUIRED"
    assert report["source_phase"] == "phase2" and report["split"] == "test"


@pytest.mark.parametrize(
    "fault",
    [
        "source",
        "manifest",
        "private",
        "invocation",
        "elf",
        "console",
        "counter",
        "arm",
        "scope",
        "dump",
        "object",
        "caller",
    ],
)
def test_changed_or_relabelled_record_refuses(selected, fault):
    path, args, write = selected
    if fault in {"scope"}:
        args["domain"]["scope_sha256"] = "0" * 64
    elif fault == "arm":
        data = M._read(path)
        data["build"]["context"]["compiler_sha256"] = "foreign"
        write("measurements/label.json", data)
    else:
        names = {
            "source": "source.mlir",
            "manifest": "manifest.json",
            "private": "declaration.json",
            "invocation": "invocation.json",
            "elf": "elf",
            "console": "stdout",
            "counter": "counter-protocol-audit.json",
            "dump": "dump",
            "object": "kernel.o",
            "caller": "harness.c",
        }
        name = names[fault]
        if fault in {"private", "invocation", "counter"}:
            data = M._read(path.parent.parent / name)
            if fault == "private":
                data["visibility"] = "private"
            elif fault == "invocation":
                data["inputs_unchanged"] = False
            else:
                data["completion_call_pc"] = 50
            write(name, data)
        else:
            write(name, "changed")
    with pytest.raises((ValueError, KeyError, TypeError)):
        M.intake(path, **args)


def test_historical_source_successor_requires_identical_bytes_and_no_input_override(selected):
    path, args, write = selected
    generator = path.parent.parent / "generator.py"
    saved = write("original-generator.py", generator.read_text())
    generator.write_text("# changed current implementation")
    args["source_successors"] = {str(generator): str(saved)}
    invocation, custody = M.historical_invocation(args["invocation_path"], source_successors=args["source_successors"])
    assert custody[-1]["current_bytes"] == M.pin(saved)
    assert invocation["status"] == "completed"
    saved.write_text("wrong historical source")
    with pytest.raises(ValueError, match="changed"):
        M.historical_invocation(args["invocation_path"], source_successors=args["source_successors"])


@pytest.mark.parametrize("closed", [True, False])
def test_original_console_values_require_complete_framing_and_roster(selected, closed):
    path, args, write = selected
    cb = {"tensors": {"Y0": {"role": "output", "shape": [1, 2], "dtype": "i32"}}}
    write("commands.json", cb)
    write("source_commands.json", cb)
    manifest = M._read(args["manifest_path"])
    manifest["members"][0]["command_buffer_sha256"] = M.digest(cb)
    write("manifest.json", manifest)
    declaration = M._read(args["declaration_path"])
    declaration["manifest"] = M.pin(args["manifest_path"])
    write("declaration.json", declaration)
    payload = base64.b64encode((-7).to_bytes(4, "little", signed=True) + (5).to_bytes(4, "little")).decode()
    console = f"METRIC cycles 123\nOUT_B64_BEGIN v1 Y0 1 2 4 s\nOUT_B64_CHUNK 00000000 0008 {payload}\n"
    console += "OUT_B64_END\nDONE\n" if closed else ""
    stdout = write("stdout", console)
    invocation = M._read(args["invocation_path"])
    invocation["stdout"] = M.pin(stdout)
    write("invocation.json", invocation)
    build = M._read(path.parent.parent / "build-receipt.json")
    build["command_buffer_sha256"] = M.digest(cb)
    write("build-receipt.json", build)
    label = M._read(path)
    label.update(build=build, console_sha256=M.pin(stdout)["sha256"])
    label.pop("memory_evidence")
    write("measurements/label.json", label)
    if closed:
        report = M.intake(path, **args)
        assert report["historical_output_kind"] == "complete_console_values"
        assert report["measured_now"] is False
    else:
        with pytest.raises(ValueError, match="without END"):
            M.intake(path, **args)


def test_batch_keeps_each_original_arm_and_refusal(selected, monkeypatch):
    from merlin.common import cpu_pool

    path, args, write = selected
    args = {key: str(value) if key.endswith("_path") else value for key, value in args.items()}
    request = write(
        "batch.json",
        {
            "schema": "merlin.phase2.retained_public_measurement_batch.v1",
            "records": [{"path": str(path), **args}, {"path": str(path), **args, "foreign_arm": True}],
        },
    )
    monkeypatch.setenv("TASK_DIR", str(path.parent.parent))
    monkeypatch.setattr(cpu_pool, "shared_compute_pool", lambda **kwargs: contextlib.nullcontext())
    output = path.parent.parent / "batch-output"
    assert M.main(["--request", str(request), "--output", str(output)]) == 2
    report = M._read(output / "report.json")
    assert [r["status"] for r in report["records"]] == ["historical_provenance_checked", "UNKNOWN"]
    assert report["new_executions"] == 0 and report["qualification"] is False
    assert report["request"] == M.pin(request)


def test_batch_refuses_request_mutation_before_publishing(selected, monkeypatch):
    from merlin.common import cpu_pool

    path, args, write = selected
    args = {key: str(value) if key.endswith("_path") else value for key, value in args.items()}
    request = write(
        "batch.json",
        {
            "schema": "merlin.phase2.retained_public_measurement_batch.v1",
            "records": [{"path": str(path), **args}],
        },
    )
    monkeypatch.setenv("TASK_DIR", str(path.parent.parent))
    monkeypatch.setattr(cpu_pool, "shared_compute_pool", lambda **kwargs: contextlib.nullcontext())
    original = M.intake

    def mutate(*positional, **kwargs):
        result = original(*positional, **kwargs)
        request.write_text("{}")
        return result

    monkeypatch.setattr(M, "intake", mutate)
    output = path.parent.parent / "batch-output"
    with pytest.raises(ValueError, match="request changed"):
        M.main(["--request", str(request), "--output", str(output)])
    assert not output.exists()
